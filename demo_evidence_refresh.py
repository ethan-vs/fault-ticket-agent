"""隔离演示：真实MCP与BGE、模拟模型；不修改正式知识库，不调用DeepSeek。"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import Client
from openai import APITimeoutError
from openai.types.chat import ChatCompletionMessage

import mcp_agent
from check_mcp import server_parameters
from evidence_policy import rag_records
from graph_agent import build_graph


def response(content=None, tool=None, arguments=None):
    calls = None if tool is None else [{
        "id": "demo_search", "type": "function",
        "function": {"name": tool, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }]
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls" if calls else "stop",
        message=ChatCompletionMessage(role="assistant", content=content, tool_calls=calls),
    )])


def final_model_response(**kwargs):
    data = json.loads(next(m["content"] for m in reversed(kwargs["messages"]) if m["role"] == "tool"))
    chunk = data["matches"][0]
    return response(content=json.dumps({
        "status": "answered",
        "selected_sources": [{"chunk_id": chunk["chunk_id"], "evidence": [chunk["text"]], "reason": "采用刷新后的有效规范。"}],
        "excluded_sources": [],
        "answer": f"根据新版规范：{chunk['text']} [{chunk['chunk_id']}]",
    }, ensure_ascii=False))


async def main():
    run_id = uuid4().hex
    folder = Path(__file__).resolve().parent / ".state" / "evidence_demos" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    docs_path = folder / "documents.json"
    db_path = folder / "checkpoints.sqlite"
    config = mcp_agent.make_config("evidence-demo-" + run_id[:12])
    doc = {
        "id": "RULE-DEMO-001", "title": "软件闪退检查", "version": "1.0",
        "source": "隔离演示模拟规范", "updated_at": "2026-10-29",
        "is_current": True, "type": "规范",
        "content": "软件启动后闪退时，先记录软件版本，再收集错误日志。",
    }
    docs_path.write_text(json.dumps([doc], ensure_ascii=False), encoding="utf-8")
    print("本演示使用真实MCP、BGE和SQLite；模型响应用Mock，不调用DeepSeek。")
    print("第1阶段：检索1.0版后，模拟模型请求超时。")
    first_model = MagicMock()
    first_model.chat.completions.create.side_effect = [
        response(tool="search_knowledge", arguments={"query": "软件启动后闪退"}),
        APITimeoutError(request=MagicMock()),
    ]
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver, Client(server_parameters(knowledge_base_path=docs_path), read_timeout_seconds=120.0) as client:
        graph = build_graph(first_model, checkpointer=saver,
                            async_tool_executor=mcp_agent.create_mcp_executor(client),
                            model_tools=mcp_agent.discover_model_tools(await client.list_tools()))
        try:
            await mcp_agent.start_turn(graph, config, "软件启动后闪退，请查询规范。", mcp_client=client)
        except APITimeoutError:
            pass
        snapshot = await graph.aget_state(config)
        old_result = rag_records(snapshot.values["messages"], current_turn_only=True)[-1]["result"]
        old_revision = old_result["knowledge_revision"]
        print("检查点已保存，下一节点：", snapshot.next)

    doc["version"] = "2.0"
    doc["content"] = "软件启动后闪退时，先记录具体错误代码，再补充软件版本和崩溃日志。"
    docs_path.write_text(json.dumps([doc], ensure_ascii=False), encoding="utf-8")
    print("\n第2阶段：独立演示规范已更新为2.0版，重新打开连接和数据库。")
    second_model = MagicMock()
    second_model.chat.completions.create.side_effect = final_model_response
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver, Client(server_parameters(knowledge_base_path=docs_path), read_timeout_seconds=120.0) as client:
        graph = build_graph(second_model, checkpointer=saver,
                            async_tool_executor=mcp_agent.create_mcp_executor(client),
                            model_tools=mcp_agent.discover_model_tools(await client.list_tools()))
        result = await mcp_agent.resume_turn(graph, config, mcp_client=client)
        new_result = rag_records(result["messages"], current_turn_only=True)[-1]["result"]
    selected = result["rag_review"]["selected_sources"][0]["chunk_id"]
    if "@2.0#" not in selected or new_result["knowledge_revision"] == old_revision:
        raise ValueError("演示未成功刷新为2.0版。")
    report = {
        "mocked_model": True, "live_mcp": True, "live_local_embedding": True,
        "old_revision": old_revision, "new_revision": new_result["knowledge_revision"],
        "selected_chunk": selected, "model_requests_after_restart": second_model.chat.completions.create.call_count,
        "status": result["status"], "answer": result["messages"][-1]["content"],
    }
    report_path = folder / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n恢复后的回答：", report["answer"])
    print("恢复后模拟模型请求次数：", report["model_requests_after_restart"])
    print("验证通过：采用2.0版证据，未继续引用1.0版。")
    print("演示记录：", report_path)


if __name__ == "__main__":
    asyncio.run(main())

