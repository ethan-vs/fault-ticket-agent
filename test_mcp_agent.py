"""模拟模型 + 真实 MCP 子进程 + 真实 SQLite 的集成测试。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import Client
from mcp.shared.exceptions import MCPError
from openai import APITimeoutError
from openai.types.chat import ChatCompletionMessage

import graph_agent
import mcp_agent
from check_mcp import server_parameters


def response(content=None, tool=None, args=None, call_id="call_001"):
    calls = None if tool is None else [{"id": call_id, "type": "function", "function": {
        "name": tool, "arguments": json.dumps(args, ensure_ascii=False),
    }}]
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls" if calls else "stop",
        message=ChatCompletionMessage(role="assistant", content=content, tool_calls=calls),
    )])


class TestMCPAgent(unittest.IsolatedAsyncioTestCase):
    async def test_real_mcp_rag_flow(self):
        """模型响应模拟，工单查询与BGE检索实际通过MCP子进程执行。"""
        count = 0

        def model_response(**kwargs):
            nonlocal count
            count += 1
            if count == 1:
                return response(tool="get_ticket", args={"ticket_id": "TK-002"})
            if count == 2:
                return response(tool="search_knowledge", args={"query": "网络正常但报表软件启动后闪退", "top_k": 3}, call_id="call_search")
            raw = next(m["content"] for m in reversed(kwargs["messages"]) if m["role"] == "tool")
            matches = json.loads(raw)["matches"]
            selected = next(m for m in matches if m["document_id"] == "RULE-SOFT-001")
            quote = selected["text"].split("。")[0] + "。"
            return response(content=json.dumps({
                "status": "answered",
                "selected_sources": [{"chunk_id": selected["chunk_id"], "evidence": [quote], "reason": "对应软件启动异常。"}],
                "excluded_sources": [{"chunk_id": m["chunk_id"], "reason": "本测试只采用该规范，其他候选不用于回答。"} for m in matches if m["chunk_id"] != selected["chunk_id"]],
                "answer": f"请收集软件版本、错误日志和复现步骤。[{selected['chunk_id']}]",
            }, ensure_ascii=False))

        model = MagicMock()
        model.chat.completions.create.side_effect = model_response
        with tempfile.TemporaryDirectory() as folder, patch("graph_agent.write_event") as log, patch("mcp_agent.write_event"), patch("builtins.print"):
            async with AsyncSqliteSaver.from_conn_string(str(Path(folder) / "state.sqlite")) as saver, Client(server_parameters(), read_timeout_seconds=120.0) as client:
                graph = graph_agent.build_graph(model, checkpointer=saver, async_tool_executor=mcp_agent.create_mcp_executor(client), model_tools=mcp_agent.discover_model_tools(await client.list_tools()))
                config = mcp_agent.make_config("integration-test")
                result = await mcp_agent.start_turn(graph, config, "分析 TK-002 并检索相关资料")
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["model_calls"], 3)
                self.assertEqual(result["tool_calls"], 2)
                self.assertEqual(result["rag_review"]["status"], "answered")
                self.assertFalse((await graph.aget_state(config)).next)
        names = [c.args[2]["name"] for c in log.call_args_list if c.args[1] == "mcp_request"]
        self.assertEqual(names, ["get_ticket", "search_knowledge"])

    async def test_resume_after_closing_mcp_and_database(self):
        with tempfile.TemporaryDirectory() as folder, patch("graph_agent.write_event"), patch("mcp_agent.write_event"), patch("builtins.print"):
            path = str(Path(folder) / "state.sqlite")
            config = mcp_agent.make_config("resume-test")
            first_model = MagicMock()
            first_model.chat.completions.create.side_effect = [
                response(tool="get_ticket", args={"ticket_id": "TK-001"}),
                APITimeoutError(request=MagicMock()),
            ]
            async with AsyncSqliteSaver.from_conn_string(path) as saver, Client(server_parameters(), read_timeout_seconds=30.0) as client:
                graph = graph_agent.build_graph(first_model, checkpointer=saver, async_tool_executor=mcp_agent.create_mcp_executor(client), model_tools=mcp_agent.discover_model_tools(await client.list_tools()))
                with self.assertRaises(APITimeoutError):
                    await mcp_agent.start_turn(graph, config, "查询 TK-001")
                self.assertEqual((await graph.aget_state(config)).next, ("model",))
            second_model = MagicMock()
            second_model.chat.completions.create.side_effect = [response(content="根据已保存的工单结果继续回答。")]
            async with AsyncSqliteSaver.from_conn_string(path) as saver, Client(server_parameters(), read_timeout_seconds=30.0) as client:
                executor = AsyncMock(wraps=mcp_agent.create_mcp_executor(client))
                graph = graph_agent.build_graph(second_model, checkpointer=saver, async_tool_executor=executor, model_tools=mcp_agent.discover_model_tools(await client.list_tools()))
                result = await mcp_agent.resume_turn(graph, config)
                executor.assert_not_called()
                self.assertEqual(result["status"], "completed")
                self.assertEqual(second_model.chat.completions.create.call_count, 1)

    async def test_mcp_connection_failure_is_not_silently_local(self):
        client = MagicMock()
        client.call_tool = AsyncMock(side_effect=MCPError(code=-32000, message="Connection closed"))
        call = {"id": "call_test", "function": {"name": "get_ticket", "arguments": '{"ticket_id":"TK-001"}'}}
        state = {"session_id": "unit", "turn_id": "turn"}
        with patch("graph_agent.write_event"), patch("graph_agent.dispatch_tool") as local:
            with self.assertRaises(mcp_agent.MCPToolUnavailable):
                await mcp_agent.create_mcp_executor(client)(call, state)
        local.assert_not_called()

    async def test_invalid_params_are_reported(self):
        client = MagicMock()
        client.call_tool = AsyncMock(side_effect=MCPError(code=-32602, message="Invalid params"))
        call = {"id": "call_test", "function": {"name": "get_ticket", "arguments": '{"ticket_id":123}'}}
        with patch("graph_agent.write_event"):
            result = await mcp_agent.create_mcp_executor(client)(call, {"session_id": "unit", "turn_id": "turn"})
        self.assertEqual(result["status"], "invalid_arguments")

    async def test_unlisted_tool_is_rejected_without_rpc(self):
        client = MagicMock()
        client.call_tool = AsyncMock()
        result = await mcp_agent.create_mcp_executor(client)(
            {"id": "test", "function": {"name": "delete_ticket", "arguments": "{}"}},
            {"session_id": "unit", "turn_id": "turn"},
        )
        self.assertEqual(result["status"], "invalid_tool")
        client.call_tool.assert_not_called()

    async def test_discovered_schema_only_exposes_allowed_tools(self):
        listing = SimpleNamespace(tools=[
            SimpleNamespace(name=name, description="工具", input_schema={"type": "object"})
            for name in ("get_ticket", "search_knowledge", "delete_ticket")
        ])
        model_tools = mcp_agent.discover_model_tools(listing)
        self.assertEqual({t["function"]["name"] for t in model_tools}, {"get_ticket", "search_knowledge"})

    async def test_cli_status_and_exit_do_not_call_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cli.sqlite"
            model = MagicMock()
            with (
                patch("sys.argv", ["mcp_agent.py", "--session", "cli-test", "--db", str(path)]),
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}),
                patch("mcp_agent.load_dotenv"),
                patch("mcp_agent.OpenAI", return_value=model),
                patch("mcp_agent.write_event"),
                patch("builtins.input", side_effect=["/status", "exit"]),
                patch("builtins.print"),
            ):
                await mcp_agent.main()
            model.chat.completions.create.assert_not_called()
            self.assertTrue(path.exists())

    async def test_mcp_failure_preserves_pending_tool_node(self):
        with tempfile.TemporaryDirectory() as folder, patch("graph_agent.write_event"), patch("mcp_agent.write_event"), patch("builtins.print"):
            path = str(Path(folder) / "failure.sqlite")
            config = mcp_agent.make_config("tool-failure")
            model = MagicMock()
            model.chat.completions.create.side_effect = [response(tool="get_ticket", args={"ticket_id": "TK-001"})]
            broken = MagicMock()
            broken.call_tool = AsyncMock(side_effect=MCPError(code=-32000, message="Connection closed"))
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                graph = graph_agent.build_graph(model, checkpointer=saver, async_tool_executor=mcp_agent.create_mcp_executor(broken))
                with self.assertRaises(mcp_agent.MCPToolUnavailable):
                    await mcp_agent.start_turn(graph, config, "查询 TK-001")
                self.assertEqual((await graph.aget_state(config)).next, ("tools",))

            recovered = MagicMock()
            recovered.call_tool = AsyncMock(return_value=SimpleNamespace(
                is_error=False, structured_content={"status": "ok", "ticket": {"ticket_id": "TK-001"}},
            ))
            second_model = MagicMock()
            second_model.chat.completions.create.side_effect = [response(content="查询已完成。")]
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                graph = graph_agent.build_graph(second_model, checkpointer=saver, async_tool_executor=mcp_agent.create_mcp_executor(recovered))
                result = await mcp_agent.resume_turn(graph, config)
            self.assertEqual(result["status"], "completed")
            recovered.call_tool.assert_awaited_once_with("get_ticket", {"ticket_id": "TK-001"})
            self.assertEqual(second_model.chat.completions.create.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
