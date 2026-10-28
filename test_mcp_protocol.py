"""启动真实服务端子进程，验证 MCP 协议；不调用 DeepSeek。"""

import unittest

from mcp import Client
from mcp.shared.exceptions import MCPError

from check_mcp import business_result, server_parameters
from ticket_tools import get_ticket


class TestMCPProtocol(unittest.IsolatedAsyncioTestCase):
    async def test_tool_discovery(self):
        async with Client(server_parameters(), read_timeout_seconds=30.0) as client:
            result = await client.list_tools()
            tools = {tool.name: tool for tool in result.tools}
            self.assertIn("get_ticket", tools)
            self.assertIn("search_knowledge", tools)
            self.assertIn("ticket_id", tools["get_ticket"].input_schema["properties"])
            self.assertTrue(tools["get_ticket"].annotations.read_only_hint)

    async def test_protocol_result_matches_local_function(self):
        async with Client(server_parameters(), read_timeout_seconds=30.0) as client:
            await client.list_tools()
            result = await client.call_tool("get_ticket", {"ticket_id": "  tk-001  "})
            self.assertFalse(result.is_error)
            self.assertEqual(business_result(result), get_ticket("  tk-001  "))

    async def test_missing_ticket_is_business_not_found(self):
        async with Client(server_parameters(), read_timeout_seconds=30.0) as client:
            await client.list_tools()
            result = await client.call_tool("get_ticket", {"ticket_id": "TK-999"})
            self.assertFalse(result.is_error)
            self.assertEqual(business_result(result)["status"], "not_found")

    async def test_invalid_argument_is_rejected_and_connection_survives(self):
        async with Client(server_parameters(), read_timeout_seconds=30.0) as client:
            await client.list_tools()
            try:
                result = await client.call_tool("get_ticket", {"ticket_id": 123})
            except MCPError as error:
                self.assertEqual(error.code, -32602)
            else:
                self.assertTrue(result.is_error)
            # 参数错误不能被误当查询成功，也不应使后续正常调用失效。
            normal = await client.call_tool("get_ticket", {"ticket_id": "TK-001"})
            self.assertEqual(business_result(normal)["status"], "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
