import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import APITimeoutError

import planning_flow
from test_plan_tasks import MATCHES, TICKET, valid_plan


OLD = "a" * 64
NEW = "b" * 64


def data(value):
    return SimpleNamespace(is_error=False, structured_content=value)


def ticket():
    return data({"status": "ok", "ticket": TICKET})


def retrieval(revision=OLD, matches=None):
    return data({"status": "ok", "matches": matches or MATCHES, "knowledge_revision": revision})


def revision(value=OLD):
    return data({"status": "ok", "knowledge_revision": value})


def response(plan=None):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(plan or valid_plan(), ensure_ascii=False)))])


class TestPlanningFlow(unittest.IsolatedAsyncioTestCase):
    async def test_four_steps_export_draft(self):
        with tempfile.TemporaryDirectory() as folder, patch("planning_flow.write_event"), patch("builtins.print"):
            root = Path(folder)
            client, model = MagicMock(), MagicMock()
            client.call_tool = AsyncMock(side_effect=[ticket(), retrieval(), revision()])
            model.chat.completions.create.return_value = response()
            config = planning_flow.plan_config("study")
            async with AsyncSqliteSaver.from_conn_string(str(root / "state.sqlite")) as saver:
                graph = planning_flow.build_planning_graph(client, model, saver)
                result = await planning_flow.start_plan(graph, config, "study", "TK-002", root / "plans")
                self.assertFalse((await graph.aget_state(config)).next)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(Path(result["markdown_path"]).exists())
            self.assertEqual(Path(result["json_path"]).parent.name, result["plan_id"])
            self.assertEqual(result["plan"]["execution_status"], "not_run")

    async def test_timeout_resumes_generation_without_retrieval(self):
        with tempfile.TemporaryDirectory() as folder, patch("planning_flow.write_event"), patch("builtins.print"):
            root = Path(folder)
            path = str(root / "state.sqlite")
            config = planning_flow.plan_config("study")
            client, model = MagicMock(), MagicMock()
            client.call_tool = AsyncMock(side_effect=[ticket(), retrieval()])
            model.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                graph = planning_flow.build_planning_graph(client, model, saver)
                with self.assertRaises(APITimeoutError):
                    await planning_flow.start_plan(graph, config, "study", "TK-002", root / "plans")
                self.assertEqual((await graph.aget_state(config)).next, ("generate",))
            client2, model2 = MagicMock(), MagicMock()
            client2.call_tool = AsyncMock(side_effect=[ticket(), revision(), revision()])
            model2.chat.completions.create.return_value = response()
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                graph = planning_flow.build_planning_graph(client2, model2, saver)
                result = await planning_flow.resume_plan(graph, config, client2)
            self.assertEqual(result["status"], "completed")
            self.assertNotIn("search_knowledge", [c.args[0] for c in client2.call_tool.call_args_list])
            self.assertEqual(model2.chat.completions.create.call_count, 1)

    async def test_export_retry_reuses_saved_draft_and_directory(self):
        with tempfile.TemporaryDirectory() as folder, patch("planning_flow.write_event"), patch("builtins.print"):
            root = Path(folder)
            path = str(root / "state.sqlite")
            config = planning_flow.plan_config("study")
            client, model = MagicMock(), MagicMock()
            client.call_tool = AsyncMock(side_effect=[ticket(), retrieval(), revision()])
            model.chat.completions.create.return_value = response()
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                graph = planning_flow.build_planning_graph(client, model, saver)
                with patch("planning_flow.export_plan_files", side_effect=OSError("模拟导出失败")):
                    with self.assertRaises(OSError):
                        await planning_flow.start_plan(graph, config, "study", "TK-002", root / "plans")
                saved = await graph.aget_state(config)
                self.assertEqual(saved.next, ("export",))
                plan_id = saved.values["plan_id"]
            client2, model2 = MagicMock(), MagicMock()
            client2.call_tool = AsyncMock(side_effect=[ticket(), revision(), revision()])
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                graph = planning_flow.build_planning_graph(client2, model2, saver)
                result = await planning_flow.resume_plan(graph, config, client2)
            model2.chat.completions.create.assert_not_called()
            self.assertEqual(Path(result["json_path"]).parent.name, plan_id)
            self.assertEqual(len(list((root / "plans").iterdir())), 1)

    async def test_updated_evidence_discards_old_draft_and_regenerates(self):
        with tempfile.TemporaryDirectory() as folder, patch("planning_flow.write_event"), patch("builtins.print"):
            root = Path(folder)
            path = str(root / "state.sqlite")
            config = planning_flow.plan_config("study")
            client, model = MagicMock(), MagicMock()
            client.call_tool = AsyncMock(side_effect=[ticket(), retrieval(), revision(NEW)])
            model.chat.completions.create.return_value = response()
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                graph = planning_flow.build_planning_graph(client, model, saver)
                with self.assertRaisesRegex(ValueError, "资料发生变化"):
                    await planning_flow.start_plan(graph, config, "study", "TK-002", root / "plans")
                self.assertEqual((await graph.aget_state(config)).next, ("export",))
            new_matches = [{**MATCHES[0], "chunk_id": "RULE-SOFT-001@2.0#001"}]
            new_plan = valid_plan()
            new_plan["evidence"][0]["chunk_id"] = new_matches[0]["chunk_id"]
            new_plan["tasks"][0]["source_ids"] = [new_matches[0]["chunk_id"]]
            client2, model2 = MagicMock(), MagicMock()
            client2.call_tool = AsyncMock(side_effect=[ticket(), revision(NEW), retrieval(NEW, new_matches), revision(NEW)])
            model2.chat.completions.create.return_value = response(new_plan)
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                graph = planning_flow.build_planning_graph(client2, model2, saver)
                result = await planning_flow.resume_plan(graph, config, client2)
            self.assertEqual(result["generation_count"], 2)
            self.assertEqual(result["plan"]["knowledge_revision"], NEW)
            self.assertEqual(result["plan"]["tasks"][0]["source_ids"], [new_matches[0]["chunk_id"]])

    async def test_missing_ticket_finishes_without_model_call(self):
        with tempfile.TemporaryDirectory() as folder, patch("planning_flow.write_event"), patch("builtins.print"):
            client, model = MagicMock(), MagicMock()
            client.call_tool = AsyncMock(return_value=data({"status": "not_found"}))
            async with AsyncSqliteSaver.from_conn_string(str(Path(folder) / "state.sqlite")) as saver:
                graph = planning_flow.build_planning_graph(client, model, saver)
                result = await planning_flow.start_plan(graph, planning_flow.plan_config("study"), "study", "TK-999", Path(folder) / "plans")
            self.assertEqual(result["status"], "not_found")
            model.chat.completions.create.assert_not_called()

    async def test_cancel_clears_pending_plan(self):
        with tempfile.TemporaryDirectory() as folder, patch("planning_flow.write_event"), patch("builtins.print"):
            client, model = MagicMock(), MagicMock()
            client.call_tool = AsyncMock(side_effect=[ticket(), retrieval()])
            model.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
            config = planning_flow.plan_config("study")
            async with AsyncSqliteSaver.from_conn_string(str(Path(folder) / "state.sqlite")) as saver:
                graph = planning_flow.build_planning_graph(client, model, saver)
                with self.assertRaises(APITimeoutError):
                    await planning_flow.start_plan(graph, config, "study", "TK-002", Path(folder) / "plans")
                with self.assertRaisesRegex(ValueError, "/resume-plan"):
                    await planning_flow.start_plan(graph, config, "study", "TK-001", Path(folder) / "plans")
                await planning_flow.cancel_plan(graph, config)
                saved = await graph.aget_state(config)
                self.assertFalse(saved.next)
                self.assertEqual(saved.values["status"], "cancelled")


if __name__ == "__main__":
    unittest.main(verbosity=2)
