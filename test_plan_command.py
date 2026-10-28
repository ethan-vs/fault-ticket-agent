import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import mcp_agent


class TestPlanCommand(unittest.IsolatedAsyncioTestCase):
    async def test_command_normalization_and_invalid_format(self):
        self.assertEqual(mcp_agent.parse_plan_command("/plan tk-002"), "TK-002")
        for text in ("/plan", "/plan TK-001 TK-002", "/plan invalid", "TK-002"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    mcp_agent.parse_plan_command(text)

    async def test_pending_analysis_blocks_new_plan(self):
        graph = MagicMock()
        graph.aget_state = AsyncMock(return_value=SimpleNamespace(next=("model",)))
        with patch("mcp_agent.create_test_plan", new_callable=AsyncMock) as create:
            with self.assertRaisesRegex(ValueError, "/resume"):
                await mcp_agent.run_plan_command(graph, {"configurable": {"thread_id": "study-001"}}, "/plan TK-002", MagicMock(), MagicMock(), Path("unused"))
        create.assert_not_called()

    async def test_success_reuses_existing_clients_and_records_result(self):
        graph = MagicMock()
        graph.aget_state = AsyncMock(return_value=SimpleNamespace(next=()))
        mcp_client, model_client = MagicMock(), MagicMock()
        with tempfile.TemporaryDirectory() as folder:
            output_dir = Path(folder)
            expected = {
                "plan": {"tasks": [{"task_id": "TASK-001"}], "model_requests": 1, "plan_status": "draft_pending_review"},
                "json_path": output_dir / "plan.json", "markdown_path": output_dir / "plan.md",
            }
            with patch("mcp_agent.create_test_plan", new_callable=AsyncMock, return_value=expected) as create, patch("mcp_agent.write_event") as log:
                result = await mcp_agent.run_plan_command(graph, {"configurable": {"thread_id": "study-001"}}, "/plan tk-002", mcp_client, model_client, output_dir)
            create.assert_awaited_once_with("TK-002", mcp_client, model_client, output_dir)
            self.assertEqual(result, expected)
            self.assertEqual([c.args[1] for c in log.call_args_list], ["plan_start", "plan_exported"])
            self.assertEqual(log.call_args_list[-1].args[2]["execution_status"], "not_run")

    async def test_failure_is_recorded_without_success_event(self):
        graph = MagicMock()
        graph.aget_state = AsyncMock(return_value=SimpleNamespace(next=()))
        with patch("mcp_agent.create_test_plan", new_callable=AsyncMock, side_effect=ValueError("证据失效")), patch("mcp_agent.write_event") as log:
            with self.assertRaisesRegex(ValueError, "证据失效"):
                await mcp_agent.run_plan_command(graph, {"configurable": {"thread_id": "study-001"}}, "/plan TK-002", MagicMock(), MagicMock(), Path("unused"))
        self.assertEqual([c.args[1] for c in log.call_args_list], ["plan_start", "plan_failed"])

    async def test_cli_routes_plan_command_without_starting_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = {
                "plan": {"tasks": [{"task_id": "TASK-001"}], "model_requests": 1, "plan_status": "draft_pending_review"},
                "json_path": root / "plan.json", "markdown_path": root / "plan.md",
            }
            model = MagicMock()
            with (
                patch("sys.argv", ["mcp_agent.py", "--session", "command-test", "--db", str(root / "state.sqlite")]),
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}),
                patch("mcp_agent.load_dotenv"),
                patch("mcp_agent.OpenAI", return_value=model),
                patch("mcp_agent.start_plan", new_callable=AsyncMock, return_value={**expected, "status": "completed"}) as create,
                patch("mcp_agent.write_event"),
                patch("builtins.input", side_effect=["/plan TK-002", "exit"]),
                patch("builtins.print"),
            ):
                await mcp_agent.main()
            create.assert_awaited_once()
            self.assertEqual(create.call_args.args[3], "TK-002")
            model.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
