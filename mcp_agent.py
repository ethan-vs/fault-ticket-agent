"""Agent + LangGraph + SQLite + MCP 入口：python mcp_agent.py。"""

import argparse
import asyncio
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import Client
from mcp.shared.exceptions import MCPError
from openai import APIError, OpenAI

from check_mcp import business_result, server_parameters
from graph_agent import SYSTEM_PROMPT, build_graph, record
from persistent_agent import error_message, make_config
from trace_logger import new_session_id, write_event
from plan_tasks import create_test_plan
from planning_flow import build_planning_graph, plan_config, start_plan, resume_plan, cancel_plan, show_plan_status
from evidence_policy import EvidenceUnavailable, prepare_history, refresh_before_resume


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / ".state" / "checkpoints.sqlite"
ALLOWED_MCP_TOOLS = {"get_ticket", "search_knowledge"}


class MCPToolUnavailable(RuntimeError):
    """工具连接或执行失败，保留检查点，等待人工恢复。"""


def discover_model_tools(listing):
    """读取真实 MCP 工具说明，仅向模型开放项目允许的两个工具。"""
    tools = [tool for tool in listing.tools if tool.name in ALLOWED_MCP_TOOLS]
    if {tool.name for tool in tools} != ALLOWED_MCP_TOOLS:
        raise ValueError("MCP 服务端缺少工单查询或知识库检索工具。")
    return [{
        "type": "function",
        "function": {
            "name": tool.name, "description": tool.description or tool.name,
            "parameters": tool.input_schema,
        },
    } for tool in tools]


def create_mcp_executor(mcp_client):
    async def execute(call, state):
        name = call["function"]["name"]
        if name not in ALLOWED_MCP_TOOLS:
            return {"status": "invalid_tool", "message": "工具未获开放，未执行。"}
        try:
            arguments = json.loads(call["function"]["arguments"])
        except json.JSONDecodeError:
            return {"status": "invalid_arguments", "message": "参数不是合法 JSON。"}
        if not isinstance(arguments, dict):
            return {"status": "invalid_arguments", "message": "工具参数必须是对象。"}

        started = perf_counter()
        record(state, "mcp_request", {
            "tool_call_id": call["id"], "name": name,
            "arguments": arguments, "transport": "stdio",
        })
        try:
            result = await mcp_client.call_tool(name, arguments)
        except MCPError as error:
            record(state, "mcp_error", {
                "tool_call_id": call["id"], "name": name,
                "error_type": type(error).__name__, "code": error.code,
            })
            if error.code == -32602:
                return {"status": "invalid_arguments", "message": "MCP 服务端拒绝了不合法参数。"}
            raise MCPToolUnavailable("MCP 连接或请求失败，没有改用本地工具。") from error
        except TimeoutError as error:
            record(state, "mcp_error", {
                "tool_call_id": call["id"], "name": name, "error_type": "TimeoutError",
            })
            raise MCPToolUnavailable("MCP 工具请求超时。") from error

        if result.is_error:
            record(state, "mcp_error", {
                "tool_call_id": call["id"], "name": name, "error_type": "ToolExecutionError",
            })
            raise MCPToolUnavailable("MCP 工具执行失败，请检查参数或服务端日志。")
        try:
            data = business_result(result)
        except (ValueError, RuntimeError) as error:
            raise MCPToolUnavailable("MCP 返回的业务结构不符合约定。") from error

        record(state, "mcp_response", {
            "tool_call_id": call["id"], "name": name, "status": data["status"],
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        })
        return data
    return execute


async def start_turn(graph, config, user_text, mcp_client=None):
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("输入不能为空。")
    snapshot = await graph.aget_state(config)
    if snapshot.next:
        raise ValueError("当前有未完成任务，请先输入 /resume。")
    session_id = config["configurable"]["thread_id"]
    history = snapshot.values.get("messages", [{"role": "system", "content": SYSTEM_PROMPT}])
    turn_id = new_session_id()
    history = await prepare_history(history, mcp_client, session_id, turn_id)
    write_event(session_id, "turn_start", {"turn_id": turn_id, "input": user_text.strip(), "entry": "mcp_agent"})
    return await graph.ainvoke({
        "messages": deepcopy(history) + [{"role": "user", "content": user_text.strip()}],
        "session_id": session_id, "turn_id": turn_id,
        "model_calls": 0, "tool_calls": 0, "status": "running", "rag_review": None,
    }, config=config, durability="sync")


async def resume_turn(graph, config, mcp_client=None):
    snapshot = await graph.aget_state(config)
    if not snapshot.next:
        raise ValueError("没有待恢复任务，可以输入新需求。")
    await refresh_before_resume(graph, config, mcp_client)
    snapshot = await graph.aget_state(config)
    write_event(config["configurable"]["thread_id"], "turn_resume", {
        "turn_id": snapshot.values.get("turn_id"), "next_nodes": list(snapshot.next),
    })
    return await graph.ainvoke(None, config=config, durability="sync")


async def show_status(graph, config):
    snapshot = await graph.aget_state(config)
    print("会话编号：", config["configurable"]["thread_id"])
    if not snapshot.values:
        print("暂无保存的任务。")
        return
    print("轮次编号：", snapshot.values.get("turn_id"))
    print("已成功返回的模型请求数：", snapshot.values.get("model_calls", 0))
    print("模型发起的工具执行尝试数：", snapshot.values.get("tool_calls", 0))
    if snapshot.next:
        print("任务未完成，下一节点：", ", ".join(snapshot.next))
        print("输入 /resume 继续。")
    else:
        print("图执行已结束，状态：", snapshot.values.get("status"))


def parse_plan_command(text):
    parts = text.strip().split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "/plan":
        raise ValueError("用法：/plan TK-002")
    ticket_id = parts[1].strip().upper()
    if not re.fullmatch(r"TK-[0-9]{3}", ticket_id):
        raise ValueError("当前模拟工单编号格式为 TK-001，请只提供一个编号。")
    return ticket_id


async def run_plan_command(graph, config, text, mcp_client, model_client, output_dir, planning_graph=None):
    ticket_id = parse_plan_command(text)
    if (await graph.aget_state(config)).next:
        raise ValueError("当前分析任务未完成，请先 /resume，再生成草稿。")
    session_id = config["configurable"]["thread_id"]
    if planning_graph is not None:
        state = await start_plan(planning_graph, plan_config(session_id), session_id, ticket_id, output_dir)
        return exported_plan(state)
    plan_run_id = new_session_id()
    write_event(session_id, "plan_start", {
        "plan_run_id": plan_run_id, "ticket_id": ticket_id,
    })
    try:
        result = await create_test_plan(ticket_id, mcp_client, model_client, output_dir)
    except (APIError, MCPError, ValueError, RuntimeError) as error:
        write_event(session_id, "plan_failed", {
            "plan_run_id": plan_run_id, "ticket_id": ticket_id,
            "error_type": type(error).__name__, "status_code": getattr(error, "status_code", None),
            "validation_error": str(error) if isinstance(error, ValueError) else None,
        })
        raise
    write_event(session_id, "plan_exported", {
        "plan_run_id": plan_run_id, "ticket_id": ticket_id,
        "task_count": len(result["plan"]["tasks"]), "model_requests": result["plan"]["model_requests"],
        "plan_status": result["plan"]["plan_status"], "execution_status": "not_run",
        "json_path": str(result["json_path"]), "markdown_path": str(result["markdown_path"]),
    })
    return result


def exported_plan(state):
    if state.get("status") != "completed" or not state.get("plan"):
        raise ValueError(f"规划已结束，未生成可导出草稿：{state.get('status')}。可核对工单或资料后重新 /plan。")
    return {
        "plan": state["plan"], "json_path": Path(state["json_path"]),
        "markdown_path": Path(state["markdown_path"]),
    }


def print_plan_result(exported):
    print("草稿已生成，任务数：", len(exported["plan"]["tasks"]))
    print("JSON：", exported["json_path"])
    print("Markdown：", exported["markdown_path"])
    print("全部任务尚未执行，需要人工审核。")


async def main():
    parser = argparse.ArgumentParser(description="通过真实 MCP 调用工具的持久化 Agent")
    parser.add_argument("--session", default="study-001")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    config = make_config(args.session)
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError("请先在 .env 配置 DEEPSEEK_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=30.0, max_retries=0)
    args.db.parent.mkdir(parents=True, exist_ok=True)

    # 同一次运行复用 MCP 连接和服务进程；退出时统一关闭。
    async with (
        AsyncSqliteSaver.from_conn_string(str(args.db)) as saver,
        Client(server_parameters(), read_timeout_seconds=120.0) as mcp_client,
    ):
        listing = await mcp_client.list_tools()
        if "get_knowledge_revision" not in {tool.name for tool in listing.tools}:
            raise ValueError("MCP 服务端缺少资料版本检查工具，请同步更新 ticket_mcp_server.py。")
        model_tools = discover_model_tools(listing)
        graph = build_graph(
            client, checkpointer=saver,
            async_tool_executor=create_mcp_executor(mcp_client), model_tools=model_tools,
        )
        planning_graph = build_planning_graph(mcp_client, client, checkpointer=saver)
        plan_cfg = plan_config(args.session)
        write_event(args.session, "session_open", {
            "entry": "mcp_agent", "transport": "stdio", "protocol_version": mcp_client.protocol_version,
            "tools": [tool["function"]["name"] for tool in model_tools],
        })
        print("MCP Agent 已启动，工具调用方式：MCP stdio。")
        print("发现工具：", ", ".join(tool["function"]["name"] for tool in model_tools))
        print("命令：/plan TK-002；/resume-plan 恢复规划；/cancel-plan 取消规划；/status；/resume 恢复分析；exit 退出。")
        await show_status(graph, config)
        await show_plan_status(planning_graph, plan_cfg)

        while True:
            try:
                user_text = (await asyncio.to_thread(input, "\n你：")).strip()
            except EOFError:
                break
            if user_text.lower() == "exit":
                break
            if user_text == "/status":
                await show_status(graph, config)
                await show_plan_status(planning_graph, plan_cfg)
                continue
            if not user_text:
                print("输入不能为空。")
                continue
            if user_text in ("/resume-plan", "/cancel-plan"):
                if not (await planning_graph.aget_state(plan_cfg)).next:
                    print("当前没有待恢复或待取消的规划任务。草稿已生成时，可直接查看文件，或用 /plan 新建任务。")
                    continue
                try:
                    if user_text == "/cancel-plan":
                        await cancel_plan(planning_graph, plan_cfg)
                        print("规划已取消；本地检查点和审计记录保留。")
                    else:
                        state = await resume_plan(planning_graph, plan_cfg, mcp_client)
                        print_plan_result(exported_plan(state))
                except (APIError, MCPError, ValueError, RuntimeError, OSError) as error:
                    snapshot = await planning_graph.aget_state(plan_cfg)
                    write_event(args.session, "plan_attempt_failed", {
                        "plan_run_id": snapshot.values.get("plan_id"),
                        "error_type": type(error).__name__, "status_code": getattr(error, "status_code", None),
                        "validation_error": str(error) if isinstance(error, ValueError) else None,
                    })
                    print("规划未完成：", str(error) if isinstance(error, ValueError) else type(error).__name__)
                    await show_plan_status(planning_graph, plan_cfg)
                continue
            if user_text.lower() == "/plan" or user_text.lower().startswith("/plan "):
                try:
                    exported = await run_plan_command(
                        graph, config, user_text, mcp_client, client, ROOT / ".state" / "test_plans", planning_graph=planning_graph,
                    )
                except (APIError, MCPError, ValueError, RuntimeError, OSError) as error:
                    print("草稿未导出：", str(error) if isinstance(error, ValueError) else type(error).__name__)
                    print("规划进度已记录；若有待执行步骤，输入 /resume-plan，或 /cancel-plan 后重建。")
                    await show_plan_status(planning_graph, plan_cfg)
                    continue
                print_plan_result(exported)
                continue
            if user_text.startswith("/") and user_text != "/resume":
                print("未知命令，请使用 /plan、/resume-plan、/cancel-plan、/status 或 /resume。")
                continue
            if user_text != "/resume" and (await planning_graph.aget_state(plan_cfg)).next:
                print("当前规划未完成，请先 /resume-plan 或 /cancel-plan。")
                continue
            snapshot = await graph.aget_state(config)
            if user_text != "/resume" and snapshot.next:
                print("当前有未完成任务，请先 /resume，或另用一个会话编号。")
                continue
            if user_text == "/resume" and not snapshot.next:
                print("没有待恢复任务，可以直接输入新需求。")
                continue
            try:
                result = await resume_turn(graph, config, mcp_client=mcp_client) if user_text == "/resume" else await start_turn(graph, config, user_text, mcp_client=mcp_client)
            except (APIError, ValueError, MCPToolUnavailable) as error:
                snapshot = await graph.aget_state(config)
                text = str(error) if isinstance(error, (MCPToolUnavailable, EvidenceUnavailable)) else error_message(error)
                write_event(args.session, "turn_attempt_failed", {
                    "turn_id": snapshot.values.get("turn_id"), "error_type": type(error).__name__,
                    "status_code": getattr(error, "status_code", None), "message": text,
                    "next_nodes": list(snapshot.next),
                    "validation_error": str(error) if isinstance(error, ValueError) else None,
                })
                print("\n助手：", text)
                await show_status(graph, config)
                continue
            text = result["messages"][-1].get("content", "")
            write_event(args.session, "turn_success" if result["status"] == "completed" else "turn_failed", {
                "turn_id": result["turn_id"], "status": result["status"],
                "model_calls": result["model_calls"], "tool_calls": result["tool_calls"], "answer": text,
            })
            print("\n助手：", text)
        write_event(args.session, "session_close", {"message": "关闭MCP客户端，保留检查点"})
    print("MCP连接已关闭，服务进程已退出。")


if __name__ == "__main__":
    asyncio.run(main())
