import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import Client
from mcp.shared.exceptions import MCPError
from openai.types.chat import ChatCompletionMessage

import evidence_policy
import graph_agent
import mcp_agent
from check_mcp import business_result, server_parameters
from rag_search import get_knowledge_revision


OLD = "a" * 64
NEW = "b" * 64


def retrieval(version="1.0", revision=OLD):
    return {
        "status": "ok", "knowledge_revision": revision,
        "matches": [{"chunk_id": f"RULE-SOFT-001@{version}#001", "text": "软件闪退时，先收集软件版本和错误日志。"}],
    }


def history(result=None):
    result = result or retrieval()
    return [
        {"role": "user", "content": "软件闪退，查询资料并给出建议"},
        {"role": "assistant", "tool_calls": [{
            "id": "search_1", "type": "function", "function": {"name": "search_knowledge", "arguments": '{"query":"软件闪退"}'},
        }]},
        {"role": "tool", "tool_call_id": "search_1", "content": json.dumps(result, ensure_ascii=False)},
    ]


def final_response(**kwargs):
    raw = next(m["content"] for m in reversed(kwargs["messages"]) if m["role"] == "tool")
    chunk = json.loads(raw)["matches"][0]
    review = {
        "status": "answered",
        "selected_sources": [{"chunk_id": chunk["chunk_id"], "evidence": [chunk["text"]], "reason": "使用当前版本资料。"}],
        "excluded_sources": [], "answer": f"请收集必要信息。[{chunk['chunk_id']}]",
    }
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=ChatCompletionMessage(role="assistant", content=json.dumps(review, ensure_ascii=False)),
    )])


def tool_result(data):
    return SimpleNamespace(is_error=False, structured_content=data)


class TestEvidencePolicy(unittest.IsolatedAsyncioTestCase):
    async def seed(self, graph, config, result=None):
        await graph.aupdate_state(config, {
            "messages": history(result), "session_id": "evidence-test", "turn_id": "turn-test",
            "model_calls": 2, "tool_calls": 2, "status": "running", "rag_review": None,
        }, as_node="tools")

    async def test_content_change_changes_revision_even_without_version_bump(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "docs.json"
            doc = {"id": "D1", "title": "规范", "version": "1.0", "source": "模拟资料", "updated_at": "2026-10-29", "is_current": True, "type": "规范", "content": "旧的检查要求"}
            path.write_text(json.dumps([doc], ensure_ascii=False), encoding="utf-8")
            before = get_knowledge_revision(path)["knowledge_revision"]
            doc["content"] = "新的检查要求"
            path.write_text(json.dumps([doc], ensure_ascii=False), encoding="utf-8")
            self.assertNotEqual(before, get_knowledge_revision(path)["knowledge_revision"])

    async def test_history_invalidation_preserves_user_input(self):
        original = history() + [{"role": "assistant", "content": "旧建议。[RULE-SOFT-001@1.0#001]"}]
        cleaned, count = evidence_policy.invalidate_history(original, NEW)
        self.assertEqual(count, 1)
        self.assertEqual(cleaned[0], original[0])
        self.assertEqual(json.loads(cleaned[2]["content"])["status"], "stale_evidence")
        self.assertNotIn("旧建议", cleaned[-1]["content"])
        self.assertEqual(json.loads(original[2]["content"])["status"], "ok")

    async def test_new_turn_discards_stale_derived_answer(self):
        client = MagicMock()
        client.call_tool = AsyncMock(return_value=tool_result({"status": "ok", "knowledge_revision": NEW}))
        original = history() + [{"role": "assistant", "content": "旧建议。[RULE-SOFT-001@1.0#001]"}]
        with patch("graph_agent.write_event"), patch("builtins.print"):
            cleaned = await evidence_policy.prepare_history(original, client, "session", "turn")
        self.assertNotIn("旧建议", cleaned[-1]["content"])
        client.call_tool.assert_awaited_once_with("get_knowledge_revision", {})

    async def test_unchanged_revision_reuses_saved_results(self):
        with tempfile.TemporaryDirectory() as folder, patch("graph_agent.write_event"), patch("mcp_agent.write_event"), patch("builtins.print"):
            model = MagicMock()
            model.chat.completions.create.side_effect = final_response
            client = MagicMock()
            client.call_tool = AsyncMock(return_value=tool_result({"status": "ok", "knowledge_revision": OLD}))
            config = mcp_agent.make_config("evidence-test")
            async with AsyncSqliteSaver.from_conn_string(str(Path(folder) / "state.sqlite")) as saver:
                graph = graph_agent.build_graph(model, checkpointer=saver)
                await self.seed(graph, config)
                result = await mcp_agent.resume_turn(graph, config, mcp_client=client)
            self.assertEqual(result["status"], "completed")
            client.call_tool.assert_awaited_once_with("get_knowledge_revision", {})
            self.assertEqual(model.chat.completions.create.call_count, 1)

    async def test_changed_revision_refreshes_checkpoint_before_answer(self):
        with tempfile.TemporaryDirectory() as folder, patch("graph_agent.write_event"), patch("mcp_agent.write_event"), patch("builtins.print"):
            model = MagicMock()
            model.chat.completions.create.side_effect = final_response
            client = MagicMock()
            client.call_tool = AsyncMock(side_effect=[
                tool_result({"status": "ok", "knowledge_revision": NEW}),
                tool_result(retrieval("2.0", NEW)),
                tool_result({"status": "ok", "knowledge_revision": NEW}),
            ])
            config = mcp_agent.make_config("evidence-test")
            async with AsyncSqliteSaver.from_conn_string(str(Path(folder) / "state.sqlite")) as saver:
                graph = graph_agent.build_graph(model, checkpointer=saver)
                await self.seed(graph, config)
                result = await mcp_agent.resume_turn(graph, config, mcp_client=client)
                self.assertFalse((await graph.aget_state(config)).next)
            self.assertIn("@2.0#001", result["messages"][-1]["content"])
            self.assertNotIn("@1.0#001", result["messages"][-1]["content"])
            self.assertEqual(model.chat.completions.create.call_count, 1)
            self.assertEqual([c.args[0] for c in client.call_tool.call_args_list], ["get_knowledge_revision", "search_knowledge", "get_knowledge_revision"])

    async def test_failed_version_check_prevents_model_call(self):
        with tempfile.TemporaryDirectory() as folder, patch("graph_agent.write_event"), patch("mcp_agent.write_event"):
            model = MagicMock()
            client = MagicMock()
            client.call_tool = AsyncMock(side_effect=MCPError(code=-32000, message="Connection closed"))
            config = mcp_agent.make_config("evidence-test")
            async with AsyncSqliteSaver.from_conn_string(str(Path(folder) / "state.sqlite")) as saver:
                graph = graph_agent.build_graph(model, checkpointer=saver)
                await self.seed(graph, config)
                with self.assertRaises(evidence_policy.EvidenceUnavailable):
                    await mcp_agent.resume_turn(graph, config, mcp_client=client)
                self.assertEqual((await graph.aget_state(config)).next, ("model",))
            model.chat.completions.create.assert_not_called()

    async def test_retired_all_documents_does_not_generate_old_advice(self):
        with tempfile.TemporaryDirectory() as folder, patch("graph_agent.write_event"), patch("mcp_agent.write_event"), patch("builtins.print"):
            model = MagicMock()
            client = MagicMock()
            client.call_tool = AsyncMock(side_effect=[
                tool_result({"status": "ok", "knowledge_revision": NEW}),
                tool_result({"status": "no_documents", "knowledge_revision": NEW, "matches": []}),
                tool_result({"status": "ok", "knowledge_revision": NEW}),
            ])
            config = mcp_agent.make_config("evidence-test")
            async with AsyncSqliteSaver.from_conn_string(str(Path(folder) / "state.sqlite")) as saver:
                graph = graph_agent.build_graph(model, checkpointer=saver)
                await self.seed(graph, config)
                result = await mcp_agent.resume_turn(graph, config, mcp_client=client)
            self.assertEqual(result["rag_review"]["status"], "insufficient_evidence")
            model.chat.completions.create.assert_not_called()

    async def test_real_mcp_reports_updated_revision_and_rebuilds_index(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "docs.json"
            doc = {"id": "RULE-DEMO-001", "title": "软件闪退检查", "version": "1.0", "source": "测试模拟资料", "updated_at": "2026-10-29", "is_current": True, "type": "规范", "content": "软件闪退时，先记录软件版本，再收集错误日志。"}
            path.write_text(json.dumps([doc], ensure_ascii=False), encoding="utf-8")
            async with Client(server_parameters(knowledge_base_path=path), read_timeout_seconds=120.0) as client:
                await client.list_tools()
                old_meta = business_result(await client.call_tool("get_knowledge_revision", {}))
                old = business_result(await client.call_tool("search_knowledge", {"query": "软件启动闪退"}))
                self.assertEqual(old["knowledge_revision"], old_meta["knowledge_revision"])
                doc["version"] = "2.0"
                doc["content"] = "软件闪退时，先记录错误代码，再补充软件版本和崩溃日志。"
                path.write_text(json.dumps([doc], ensure_ascii=False), encoding="utf-8")
                new_meta = business_result(await client.call_tool("get_knowledge_revision", {}))
                new = business_result(await client.call_tool("search_knowledge", {"query": "软件启动闪退"}))
                self.assertNotEqual(old_meta["knowledge_revision"], new_meta["knowledge_revision"])
                self.assertEqual(new["matches"][0]["version"], "2.0")
                self.assertIn("先记录错误代码", new["matches"][0]["text"])
                self.assertEqual(new["knowledge_revision"], new_meta["knowledge_revision"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
