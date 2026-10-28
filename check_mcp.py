"""真实 stdio 协议验证：python check_mcp.py；加 --rag 验证向量检索。"""

# MCP 客户端：负责启动并连接本地服务端。
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from mcp import Client, StdioServerParameters


ROOT = Path(__file__).resolve().parent


def server_parameters(knowledge_base_path=None):
    """使用当前虚拟环境解释器，启动同目录的 MCP 服务端。"""
    child_env = {
        "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }
    for key in ("FAULT_AGENT_KB_PATH", "FAULT_AGENT_MODEL_CACHE"):
        if os.environ.get(key):
            child_env[key] = os.environ[key]
    if knowledge_base_path is not None:
        child_env["FAULT_AGENT_KB_PATH"] = str(Path(knowledge_base_path).resolve())
    return StdioServerParameters(
        command=sys.executable,
        args=["-X", "utf8", "-u", "-B", str(ROOT / "ticket_mcp_server.py")],
        cwd=str(ROOT),
        env=child_env,
    )


def business_result(result):
    """区分 MCP 执行错误与业务上的 not_found。"""
    if result.is_error:
        raise RuntimeError("MCP 工具执行失败，请查看服务端错误信息。")
    data = result.structured_content
    if not isinstance(data, dict) or "status" not in data:
        raise ValueError("MCP 工具没有返回约定的业务结构。")
    return data


async def check(include_rag=False):
    report = {
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "sdk_version": version("mcp"), "transport": "stdio", "calls": [],
    }
    async with Client(server_parameters(), read_timeout_seconds=120.0) as client:
        listing = await client.list_tools()
        names = [tool.name for tool in listing.tools]
        if not {"get_ticket", "search_knowledge"}.issubset(names):
            raise ValueError("服务端未提供约定的两个工具。")
        report["tool_names"] = names
        report["protocol_version"] = client.protocol_version
        print("MCP连接成功，协议版本：", client.protocol_version)
        print("发现工具：", ", ".join(names))

        async def call(name, arguments):
            started = perf_counter()
            result = await client.call_tool(name, arguments)
            elapsed_ms = (perf_counter() - started) * 1000
            data = business_result(result)
            report["calls"].append({
                "tool": name, "arguments": arguments,
                "elapsed_ms": round(elapsed_ms, 3), "result": data,
            })
            return data

        normal = await call("get_ticket", {"ticket_id": "tk-001"})
        if normal.get("status") != "ok" or normal.get("ticket", {}).get("ticket_id") != "TK-001":
            raise ValueError("正常工单查询结果不符合约定。")
        print("\n通过MCP查询工单：")
        print(json.dumps(normal, ensure_ascii=False, indent=2))

        missing = await call("get_ticket", {"ticket_id": "TK-999"})
        if missing.get("status") != "not_found":
            raise ValueError("不存在的工单应返回 not_found。")
        print("\n不存在的工单：", missing["status"])

        if include_rag:
            # 两次检索使用同一个客户端和服务进程。
            for query in (
                "网络正常，但报表软件启动后闪退。",
                "多台电脑无法连接无线网络，有线可以使用。",
            ):
                result = await call("search_knowledge", {"query": query, "top_k": 3})
                if result.get("status") != "ok" or not result.get("matches"):
                    raise ValueError("模拟知识库应返回检索候选。")
                print("\n通过MCP检索：", query)
                for item in result["matches"]:
                    print(item["chunk_id"], f"相似度：{item['score']:.4f}")

    report["status"] = "passed"
    folder = ROOT / ".state" / "mcp_checks"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid4().hex}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n检查通过，客户端已关闭服务进程。")
    print("协议调用记录：", path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag", action="store_true", help="同时验证本地向量检索")
    args = parser.parse_args()
    asyncio.run(check(args.rag))
