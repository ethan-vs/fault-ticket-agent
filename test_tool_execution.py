import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent import execute_tool


def make_tool_call(name="get_ticket", arguments='{"ticket_id": "TK-001"}'):
    """模拟模型提出的工具调用，不请求模型。"""
    return SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        )
    )


class TestToolExecution(unittest.TestCase):

    def test_valid_arguments(self):
        """合法调用应执行查询，并原样返回查询结果。"""
        expected = {
            "status": "ok",
            "ticket": {"ticket_id": "TK-001"},
        }

        with patch("agent.get_ticket", return_value=expected) as query:
            result = execute_tool(make_tool_call())

        query.assert_called_once_with("TK-001")
        self.assertEqual(result, expected)

    def test_unknown_tool(self):
        """未知工具应被拒绝，不能执行查询。"""
        with patch("agent.get_ticket") as query:
            result = execute_tool(
                make_tool_call(name="delete_ticket")
            )

        self.assertEqual(result["status"], "invalid_tool")
        query.assert_not_called()

    def test_invalid_json(self):
        """不合法的 JSON 参数应被拒绝。"""
        with patch("agent.get_ticket") as query:
            result = execute_tool(
                make_tool_call(arguments="这不是 JSON")
            )

        self.assertEqual(result["status"], "invalid_arguments")
        query.assert_not_called()

    def test_invalid_argument_structure(self):
        """缺少字段、多余字段、错误类型等参数应被拒绝。"""
        cases = {
            "missing_field": {},
            "extra_field": {"ticket_id": "TK-001", "force": True},
            "number_id": {"ticket_id": 123},
            "null_id": {"ticket_id": None},
            "array": ["TK-001"],
            "null": None,
        }

        for name, arguments in cases.items():
            with self.subTest(case=name):
                with patch("agent.get_ticket") as query:
                    result = execute_tool(
                        make_tool_call(arguments=json.dumps(arguments))
                    )

                self.assertEqual(result["status"], "invalid_arguments")
                query.assert_not_called()

    def test_blank_id(self):
        """空白编号由真实查询函数校验，应返回无效输入。"""
        result = execute_tool(
            make_tool_call(arguments='{"ticket_id": "   "}')
        )

        self.assertEqual(result["status"], "invalid_input")


if __name__ == "__main__":
    unittest.main(verbosity=2)
