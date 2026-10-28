import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai import APITimeoutError, APIStatusError
from openai.types.chat import ChatCompletionMessage

import agent


def make_response(content=None, use_tool=False):
    """构造模拟模型响应，不发送网络请求。"""
    tool_calls = None
    if use_tool:
        tool_calls = [{
            "id": "call_test_001",
            "type": "function",
            "function": {
                "name": "get_ticket",
                "arguments": '{"ticket_id": "TK-001"}',
            },
        }]

    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )


class TestAgentErrors(unittest.TestCase):

    def run_app(self, api_results, user_inputs):
        """替换模型、键盘输入和日志写入，运行真实 main()。"""
        client = MagicMock()
        results = iter(api_results)
        histories = []

        def fake_create(**kwargs):
            # 必须复制，否则后续追加消息会改变测试看到的历史。
            histories.append(deepcopy(kwargs["messages"]))
            result = next(results)
            if isinstance(result, Exception):
                raise result
            return result

        client.chat.completions.create.side_effect = fake_create

        with (
            patch("agent.OpenAI", return_value=client),
            patch("agent.load_dotenv"),
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}),
            patch("builtins.input", side_effect=user_inputs),
            patch("agent.write_event") as mock_log,
            patch("builtins.print") as mock_print,
        ):
            agent.main()

        events = [call.args for call in mock_log.call_args_list]
        return histories, events, mock_print

    def test_first_request_timeout(self):
        """第一次请求超时：记录错误，并允许用户退出。"""
        error = APITimeoutError(request=MagicMock())

        histories, events, output = self.run_app(
            [error],
            ["查询 TK-001", "exit"],
        )

        self.assertEqual(len(histories), 1)
        errors = [e[2] for e in events if e[1] == "model_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error_type"], "APITimeoutError")
        self.assertFalse(any(e[1] == "tool_result" for e in events))
        self.assertEqual(events[-1][1], "session_end")
        output.assert_any_call("\n助手：", "模型请求超时，请稍后重试。")

    def test_second_request_timeout(self):
        """查询已经成功，第二次模型请求超时：保留过程日志。"""
        error = APITimeoutError(request=MagicMock())

        histories, events, output = self.run_app(
            [make_response(use_tool=True), error],
            ["查询 TK-001", "exit"],
        )

        self.assertEqual(len(histories), 2)
        event_types = [e[1] for e in events]
        self.assertIn("tool_result", event_types)
        self.assertIn("model_error", event_types)
        self.assertLess(
            event_types.index("tool_result"),
            event_types.index("model_error"),
        )
        self.assertNotIn("assistant_message", event_types)
        self.assertEqual(events[-1][1], "session_end")
        output.assert_any_call("\n助手：", "模型请求超时，请稍后重试。")

    def test_second_request_balance_error_and_history(self):
        """第二次请求余额不足：提示错误，下一轮不带半截历史。"""
        response = MagicMock()
        response.status_code = 402
        error = APIStatusError(
            "Insufficient Balance",
            response=response,
            body={"error": {"message": "Insufficient Balance"}},
        )

        histories, events, output = self.run_app(
            [
                make_response(use_tool=True),
                error,
                make_response(content="模拟的下一轮回复。"),
            ],
            ["查询 TK-001", "你好", "exit"],
        )

        self.assertEqual(len(histories), 3)
        errors = [e[2] for e in events if e[1] == "model_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["status_code"], 402)
        output.assert_any_call(
            "\n助手：", "模型账户余额不足，请检查账户余额。"
        )

        # 第三次请求是新一轮：只保留系统消息和新输入。
        self.assertEqual(
            [m["role"] for m in histories[2]],
            ["system", "user"],
        )
        self.assertEqual(histories[2][-1]["content"], "你好")
        self.assertTrue(any(e[1] == "tool_result" for e in events))
        self.assertTrue(any(e[1] == "assistant_message" for e in events))
        self.assertEqual(events[-1][1], "session_end")


if __name__ == "__main__":
    unittest.main(verbosity=2)
