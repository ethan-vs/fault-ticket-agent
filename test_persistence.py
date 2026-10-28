import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langgraph.checkpoint.sqlite import SqliteSaver
from openai import APITimeoutError
from openai.types.chat import ChatCompletionMessage

import graph_agent
import persistent_agent


def mock_response(content=None, use_tool=False):
    calls = None
    if use_tool:
        calls = [{"id": "call_test_001", "type": "function", "function": {
            "name": "get_ticket", "arguments": '{"ticket_id": "TK-001"}',
        }}]
    message = ChatCompletionMessage(role="assistant", content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(
        message=message, finish_reason="tool_calls" if calls else "stop"
    )])


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "checkpoints.sqlite"
        self.config = persistent_agent.make_config("test-session")

    @contextmanager
    def open_graph(self, responses):
        client = MagicMock()
        client.chat.completions.create.side_effect = responses
        with (
            SqliteSaver.from_conn_string(str(self.db)) as saver,
            patch("graph_agent.write_event"),
            patch("persistent_agent.write_event"),
            patch("builtins.print"),
        ):
            yield graph_agent.build_graph(client, checkpointer=saver), client

    def test_completed_history_survives_reopen(self):
        """关闭数据库再打开，原会话历史仍然存在。"""
        with self.open_graph([mock_response(content="第一轮模拟回复。")]) as (graph, client):
            persistent_agent.start_turn(graph, self.config, "第一轮输入")

        with self.open_graph([mock_response(content="第二轮模拟回复。")]) as (graph, client):
            saved = graph.get_state(self.config)
            self.assertEqual(saved.values["messages"][-1]["content"], "第一轮模拟回复。")
            self.assertFalse(saved.next)
            persistent_agent.start_turn(graph, self.config, "第二轮输入")
            sent = client.chat.completions.create.call_args.kwargs["messages"]
            self.assertEqual([m["content"] for m in sent if m["role"] == "user"], ["第一轮输入", "第二轮输入"])

    def test_resume_after_second_request_timeout(self):
        """工具完成后模型超时，重新打开只执行待完成模型节点。"""
        with self.open_graph([
            mock_response(use_tool=True), APITimeoutError(request=MagicMock()),
        ]) as (graph, client):
            with self.assertRaises(APITimeoutError):
                persistent_agent.start_turn(graph, self.config, "查询 TK-001")
            saved = graph.get_state(self.config)
            self.assertEqual(saved.next, ("model",))
            self.assertEqual(saved.values["tool_calls"], 1)

        with self.open_graph([mock_response(content="根据已保存结果完成回答。")]) as (graph, client):
            with patch("graph_agent.dispatch_tool") as query:
                result = persistent_agent.resume_turn(graph, self.config)
            query.assert_not_called()
            self.assertEqual(client.chat.completions.create.call_count, 1)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["tool_calls"], 1)
            self.assertFalse(graph.get_state(self.config).next)

    def test_pending_task_cannot_be_overwritten(self):
        with self.open_graph([APITimeoutError(request=MagicMock())]) as (graph, client):
            with self.assertRaises(APITimeoutError):
                persistent_agent.start_turn(graph, self.config, "查询 TK-001")
            with self.assertRaisesRegex(ValueError, "/resume"):
                persistent_agent.start_turn(graph, self.config, "改查 TK-002")
            self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_sessions_are_isolated(self):
        other = persistent_agent.make_config("other-session")
        with self.open_graph([
            mock_response(content="A回复"), mock_response(content="B回复"),
        ]) as (graph, client):
            persistent_agent.start_turn(graph, self.config, "A输入")
            persistent_agent.start_turn(graph, other, "B输入")
            self.assertEqual(graph.get_state(self.config).values["messages"][-1]["content"], "A回复")
            sent = client.chat.completions.create.call_args.kwargs["messages"]
            self.assertEqual([m["content"] for m in sent if m["role"] == "user"], ["B输入"])

    def test_no_pending_task_does_not_call_model(self):
        with self.open_graph([]) as (graph, client):
            with self.assertRaisesRegex(ValueError, "没有待恢复任务"):
                persistent_agent.resume_turn(graph, self.config)
            client.chat.completions.create.assert_not_called()

    def test_recovery_across_two_python_processes(self):
        """两个独立 Python 进程共享数据库，第二个进程恢复任务。"""
        worker = '''
import json, sys
from unittest.mock import MagicMock, patch
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import APITimeoutError
import graph_agent, persistent_agent
from test_persistence import mock_response
client = MagicMock()
config = persistent_agent.make_config("process-test")
with SqliteSaver.from_conn_string(sys.argv[1]) as saver, patch("graph_agent.write_event"), patch("persistent_agent.write_event"), patch("builtins.print"):
    graph = graph_agent.build_graph(client, checkpointer=saver)
    if sys.argv[2] == "fail":
        client.chat.completions.create.side_effect = [mock_response(use_tool=True), APITimeoutError(request=MagicMock())]
        try:
            persistent_agent.start_turn(graph, config, "查询 TK-001")
        except APITimeoutError:
            pass
        saved = graph.get_state(config)
        output = {"next": list(saved.next), "tool_calls": saved.values["tool_calls"]}
    else:
        client.chat.completions.create.side_effect = [mock_response(content="跨进程恢复成功。")]
        with patch("graph_agent.dispatch_tool") as query:
            result = persistent_agent.resume_turn(graph, config)
        output = {"status": result["status"], "model_requests": client.chat.completions.create.call_count, "tool_executed_on_resume": query.called}
print(json.dumps(output))
'''
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([
            str(Path(__file__).resolve().parent),
            str(Path(graph_agent.__file__).resolve().parent),
            str(Path(graph_agent.execute_ticket_tool.__code__.co_filename).resolve().parent),
            env.get("PYTHONPATH", ""),
        ])

        def run_worker(phase):
            result = subprocess.run(
                [sys.executable, "-X", "utf8", "-B", "-c", worker, str(self.db), phase],
                capture_output=True, text=True, encoding="utf-8", timeout=30, env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

        self.assertEqual(run_worker("fail"), {"next": ["model"], "tool_calls": 1})
        self.assertEqual(run_worker("resume"), {
            "status": "completed", "model_requests": 1, "tool_executed_on_resume": False,
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
