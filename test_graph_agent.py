import json
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from openai import APITimeoutError
from openai.types.chat import ChatCompletionMessage

import graph_agent


def response(content=None, tool_name=None, arguments=None, call_id="call_001"):
    calls = None
    if tool_name:
        calls = [{"id": call_id, "type": "function", "function": {
            "name": tool_name, "arguments": json.dumps(arguments, ensure_ascii=False),
        }}]
    message = ChatCompletionMessage(role="assistant", content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(
        message=message, finish_reason="tool_calls" if calls else "stop"
    )])


def initial_state():
    return {
        "messages": [{"role": "user", "content": "分析 TK-001，查询规范并给出建议"}],
        "session_id": "test-session", "turn_id": "test-turn",
        "model_calls": 0, "tool_calls": 0, "status": "running",
    }


class TestGraphAgent(unittest.TestCase):

    def invoke(self, responses):
        client = MagicMock()
        client.chat.completions.create.side_effect = responses
        with patch("graph_agent.write_event") as log, patch("builtins.print"):
            result = graph_agent.build_graph(client).invoke(
                initial_state(), config={"recursion_limit": 20}
            )
        return result, client, log

    def test_two_tools_in_sequence(self):
        """真实图执行两次工具节点和三次模拟模型请求。"""
        result, client, log = self.invoke([
            response(tool_name="get_ticket", arguments={"ticket_id": "TK-001"}),
            response(tool_name="get_handling_rule", arguments={"category": "网络"}, call_id="call_002"),
            response(content="参考 RULE-NET-001 v1.0，先确认影响范围。"),
        ])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["tool_calls"], 2)
        outputs = [json.loads(m["content"]) for m in result["messages"] if m["role"] == "tool"]
        self.assertEqual(outputs[0]["ticket"]["ticket_id"], "TK-001")
        self.assertEqual(outputs[1]["rule"]["rule_id"], "RULE-NET-001")
        self.assertEqual(client.chat.completions.create.call_count, 3)
        self.assertTrue(all(c.args[2]["turn_id"] == "test-turn" for c in log.call_args_list))

    def test_no_tool_needed(self):
        result, client, log = self.invoke([response(content="请提供工单编号。")])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_unknown_tool_is_rejected(self):
        with patch("graph_agent.execute_ticket_tool") as query:
            result, client, log = self.invoke([
                response(tool_name="delete_ticket", arguments={"ticket_id": "TK-001"}),
                response(content="未知工具，未执行。"),
            ])
        query.assert_not_called()
        output = next(m for m in result["messages"] if m["role"] == "tool")
        self.assertEqual(json.loads(output["content"])["status"], "invalid_tool")

    def test_invalid_rule_arguments(self):
        with patch("graph_agent.get_handling_rule") as query:
            result, client, log = self.invoke([
                response(tool_name="get_handling_rule", arguments={"category": 123}),
                response(content="规范查询参数不合法。"),
            ])
        query.assert_not_called()
        output = next(m for m in result["messages"] if m["role"] == "tool")
        self.assertEqual(json.loads(output["content"])["status"], "invalid_arguments")

    def test_loop_stops_at_budget(self):
        result, client, log = self.invoke([
            response(tool_name="get_ticket", arguments={"ticket_id": "TK-001"}, call_id=f"call_{i}")
            for i in range(graph_agent.MAX_MODEL_CALLS)
        ])
        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(client.chat.completions.create.call_count, 4)
        self.assertEqual(result["tool_calls"], 4)
        self.assertIn("任务未完整完成", result["messages"][-1]["content"])

    def test_timeout_does_not_mutate_input(self):
        state = initial_state()
        original = json.dumps(state, ensure_ascii=False)
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            response(tool_name="get_ticket", arguments={"ticket_id": "TK-001"}),
            APITimeoutError(request=MagicMock()),
        ]
        with patch("graph_agent.write_event"), patch("builtins.print"):
            with self.assertRaises(APITimeoutError):
                graph_agent.build_graph(client).invoke(state)
        self.assertEqual(json.dumps(state, ensure_ascii=False), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
