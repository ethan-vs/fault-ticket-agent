"""MCP 2.3 服务端：只提供模拟工单读取和本地知识库检索。"""

import sys
from contextlib import redirect_stdout
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from ticket_tools import get_ticket as read_ticket
from rag_search import search_knowledge as retrieve_knowledge
from rag_search import get_knowledge_revision as read_knowledge_revision


server = MCPServer("FaultTicketTools", version="0.1.0", log_level="WARNING")
READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    open_world_hint=False,
)


@server.tool(annotations=READ_ONLY, structured_output=True)
def get_ticket(
    ticket_id: Annotated[str, Field(strict=True, min_length=1, max_length=64)],
) -> dict[str, Any]:
    """根据用户提供的工单编号读取模拟工单，不修改工单。"""
    return read_ticket(ticket_id)


@server.tool(annotations=READ_ONLY, structured_output=True)
def search_knowledge(
    query: Annotated[str, Field(strict=True, min_length=1, max_length=1000)],
    top_k: Annotated[int, Field(strict=True, ge=1, le=5)] = 3,
) -> dict[str, Any]:
    """按故障描述检索有效模拟资料，返回片段、来源、版本和相似度。"""
    # stdio 的 stdout 专供协议消息；第三方库的普通打印转到 stderr。
    with redirect_stdout(sys.stderr):
        return retrieve_knowledge(query, top_k=top_k)


@server.tool(annotations=READ_ONLY, structured_output=True)
def get_knowledge_revision() -> dict[str, Any]:
    """读取有效知识库的内容摘要，供应用检查证据是否需要刷新，不加载向量模型。"""
    return read_knowledge_revision()


if __name__ == "__main__":
    server.run(transport="stdio")
