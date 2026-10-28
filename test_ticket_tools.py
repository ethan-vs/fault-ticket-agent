import unittest

from ticket_tools import get_ticket


class TestGetTicket(unittest.TestCase):

    def test_existing_ticket(self):
        """存在的工单应查询成功。"""
        result = get_ticket("TK-001")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["ticket"]["ticket_id"], "TK-001")
        self.assertEqual(
            result["ticket"]["processing_status"],
            "待处理",
        )

    def test_normalize_ticket_id(self):
        """应支持小写编号，并忽略首尾空格。"""
        result = get_ticket("  tk-001  ")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["ticket"]["ticket_id"], "TK-001")

    def test_unknown_ticket(self):
        """不存在的工单应返回未找到。"""
        result = get_ticket("TK-999")

        self.assertEqual(result["status"], "not_found")
        self.assertNotIn("ticket", result)

    def test_empty_ticket_id(self):
        """空字符串应被拒绝。"""
        result = get_ticket("")

        self.assertEqual(result["status"], "invalid_input")

    def test_spaces_ticket_id(self):
        """只有空格也属于无效输入。"""
        result = get_ticket("   ")

        self.assertEqual(result["status"], "invalid_input")

    def test_non_string_ticket_id(self):
        """数字、None、列表等类型应被拒绝。"""
        for value in [123, None, ["TK-001"]]:
            with self.subTest(value=value):
                result = get_ticket(value)

                self.assertEqual(result["status"], "invalid_input")


if __name__ == "__main__":
    unittest.main(verbosity=2)