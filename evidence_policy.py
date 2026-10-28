"""资料版本策略：新轮次清理旧证据，恢复前检查并刷新检索结果。"""

import json
from copy import deepcopy

from mcp.shared.exceptions import MCPError

from graph_agent import record
from rag_answer import INSUFFICIENT_ANSWER


class EvidenceUnavailable(ValueError):
    """无法确认资料有效性时停止，不让模型继续使用旧证据。"""


def rag_records(messages, current_turn_only=False):
    start = 0
    if current_turn_only:
        start = next((i for i in range(len(messages) - 1, -1, -1)
                      if messages[i].get("role") == "user"), len(messages))
    calls = {}
    records = []
    for index in range(start, len(messages)):
        message = messages[index]
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if call["function"]["name"] == "search_knowledge":
                    calls[call["id"]] = call
        elif message.get("role") == "tool" and message.get("tool_call_id") in calls:
            result = json.loads(message["content"])
            if result.get("status") in ("ok", "no_documents"):
                records.append({
                    "index": index, "call": calls[message["tool_call_id"]], "result": result,
                })
    return records


async def policy_call(client, name, arguments, state):
    if client is None:
        raise EvidenceUnavailable("缺少MCP连接，无法确认资料版本，已停止使用旧证据。")
    record(state, "mcp_request", {"name": name, "arguments": arguments, "scope": "evidence_policy"})
    try:
        result = await client.call_tool(name, arguments)
    except (MCPError, TimeoutError) as error:
        record(state, "evidence_check_failed", {"name": name, "error_type": type(error).__name__})
        raise EvidenceUnavailable("资料版本检查或刷新失败，旧证据未获确认，请修复连接后再恢复。") from error
    data = result.structured_content
    if result.is_error or not isinstance(data, dict) or data.get("status") not in ("ok", "no_documents"):
        raise EvidenceUnavailable("资料服务返回错误或结构异常，已停止使用旧证据。")
    record(state, "mcp_response", {"name": name, "status": data["status"], "scope": "evidence_policy"})
    return data


async def current_revision(client, state):
    data = await policy_call(client, "get_knowledge_revision", {}, state)
    value = data.get("knowledge_revision")
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise EvidenceUnavailable("资料服务缺少合法内容摘要，请同步更新MCP服务端。")
    return value


def invalidate_history(history, revision):
    """保留用户输入、工单事实与调用配对，移除过期检索原文和派生回答。"""
    messages = deepcopy(history)
    stale = [r for r in rag_records(history) if r["result"].get("knowledge_revision") != revision]
    stale_ids = {m["chunk_id"] for r in stale for m in r["result"].get("matches", [])}
    notice = "此前的知识库依据已发生变化，旧分析已失效；需要重新检索才能提供有依据的建议。"
    for entry in stale:
        messages[entry["index"]]["content"] = json.dumps({
            "status": "stale_evidence", "matches": [], "message": notice,
            "knowledge_revision": revision,
        }, ensure_ascii=False)
    for message in messages:
        if message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        text = message.get("content") or ""
        if any(f"[{chunk_id}]" in text for chunk_id in stale_ids) or (stale and text == INSUFFICIENT_ANSWER):
            message["content"] = notice
    return messages, len(stale)


async def prepare_history(history, client, session_id, turn_id):
    records = rag_records(history)
    if not records:
        return history
    state = {"session_id": session_id, "turn_id": turn_id}
    revision = await current_revision(client, state)
    messages, count = invalidate_history(history, revision)
    record(state, "evidence_checked", {"knowledge_revision": revision, "stale_records": count, "stage": "new_turn"})
    if count:
        print("[资料检查] 旧依据已变化或缺少版本摘要，旧检索内容和派生分析不再作为当前依据。")
    return messages


async def refresh_before_resume(graph, config, client):
    snapshot = await graph.aget_state(config)
    if "model" not in snapshot.next:
        return
    records = rag_records(snapshot.values.get("messages", []), current_turn_only=True)
    if not records:
        return
    entry = records[-1]
    state = snapshot.values
    revision = await current_revision(client, state)
    old_revision = entry["result"].get("knowledge_revision")
    record(state, "evidence_checked", {
        "stage": "resume", "old_revision": old_revision, "knowledge_revision": revision,
    })
    if old_revision == revision:
        print("[资料检查] 知识库未变，复用已保存的检索结果。")
        return

    record(state, "evidence_invalidated", {"old_revision": old_revision, "observed_revision": revision})
    print("[资料检查] 知识库版本变化或旧记录缺少摘要，正在重新检索……")
    arguments = json.loads(entry["call"]["function"]["arguments"])
    refreshed = await policy_call(client, "search_knowledge", arguments, state)
    fresh_revision = refreshed.get("knowledge_revision")
    if fresh_revision != await current_revision(client, state):
        raise EvidenceUnavailable("资料在刷新过程中再次变化，请稳定资料版本后恢复；未展示旧建议。")

    messages, _ = invalidate_history(state["messages"], fresh_revision)
    messages[entry["index"]]["content"] = json.dumps(refreshed, ensure_ascii=False)
    # next=model 的情况下，更新 tools 节点产物后仍然从 model 继续。
    await graph.aupdate_state(config, {
        "messages": messages, "rag_review": None, "status": "running",
    }, as_node="tools")
    record(state, "evidence_refreshed", {
        "old_revision": old_revision, "knowledge_revision": fresh_revision,
        "chunk_ids": [m["chunk_id"] for m in refreshed.get("matches", [])],
    })
    print("[资料检查] 新检索结果已写入检查点，将使用新证据继续。")
