"""规划支路：查询、检索、生成、导出分别保存检查点。"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from check_mcp import business_result
from plan_tasks import export_plan_files, generate_draft
from trace_logger import write_event


class PlanningState(TypedDict):
    session_id: str
    plan_id: str
    ticket_id: str
    output_dir: str
    status: str
    ticket: dict | None
    retrieval: dict | None
    plan: dict | None
    generation_count: int
    json_path: str | None
    markdown_path: str | None


def plan_config(session_id):
    # 与分析流程使用不同的内部 thread_id，用户不需要记住它。
    return {"configurable": {"thread_id": "planning:" + session_id}, "recursion_limit": 12}


def event(state, kind, data):
    write_event(state["session_id"], kind, {"plan_run_id": state["plan_id"], "ticket_id": state["ticket_id"], **data})


async def read_tool(client, name, arguments, state):
    event(state, "plan_mcp_request", {"name": name, "arguments": arguments})
    data = business_result(await client.call_tool(name, arguments))
    event(state, "plan_mcp_response", {"name": name, "status": data["status"]})
    return data


def build_planning_graph(mcp_client, model_client, checkpointer=None):
    async def query(state):
        data = await read_tool(mcp_client, "get_ticket", {"ticket_id": state["ticket_id"]}, state)
        if data.get("status") != "ok":
            return {"status": "not_found", "ticket": None}
        ticket = data["ticket"]
        if ticket.get("ticket_id") != state["ticket_id"]:
            raise ValueError("工具返回工单编号不一致，规划已停止。")
        print("[规划] 当前工单查询完成")
        return {"ticket": ticket, "status": "running"}

    async def retrieve(state):
        data = await read_tool(mcp_client, "search_knowledge", {"query": state["ticket"]["description"], "top_k": 3}, state)
        if not data.get("matches"):
            return {"retrieval": data, "status": "insufficient_evidence"}
        if not isinstance(data.get("knowledge_revision"), str):
            raise ValueError("检索结果缺少内容摘要，规划已停止。")
        print("[规划] 资料检索完成，已保存证据")
        return {"retrieval": data, "status": "running"}

    async def generate(state):
        attempt_id = uuid4().hex
        event(state, "plan_model_request", {"attempt_id": attempt_id})

        def capture(response):
            choice = response.choices[0] if response.choices else None
            event(state, "plan_model_response", {
                "attempt_id": attempt_id,
                "finish_reason": choice.finish_reason if choice else None,
                "raw_output": choice.message.content if choice else None,
            })

        try:
            draft = await generate_draft(model_client, state["ticket"], state["retrieval"]["matches"], on_response=capture)
        except Exception as error:
            event(state, "plan_generation_failed", {"attempt_id": attempt_id, "error_type": type(error).__name__})
            raise
        count = state.get("generation_count", 0) + 1
        draft.update({
            "known_facts": dict(state["ticket"]),
            "knowledge_revision": state["retrieval"]["knowledge_revision"],
            "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "saved_generation_count": count,
            "model_requests_in_latest_generation": 1,
        })
        print("[规划] 草稿校验通过，已保存；尚未执行任何任务")
        return {"plan": draft, "generation_count": count, "status": "running"}

    async def export(state):
        current = await read_tool(mcp_client, "get_knowledge_revision", {}, state)
        if current.get("knowledge_revision") != state["retrieval"]["knowledge_revision"]:
            raise ValueError("生成后资料发生变化，尚未导出；输入 /resume-plan 重新获取证据并生成。")
        files = export_plan_files(state["plan"], Path(state["output_dir"]), plan_id=state["plan_id"])
        event(state, "plan_exported", {
            "json_path": str(files["json_path"]), "markdown_path": str(files["markdown_path"]),
            "task_count": len(state["plan"]["tasks"]), "execution_status": "not_run",
        })
        return {"json_path": str(files["json_path"]), "markdown_path": str(files["markdown_path"]), "status": "completed"}

    def next_step(state):
        return "continue" if state["status"] == "running" else "stop"

    builder = StateGraph(PlanningState)
    for name, node in (("query", query), ("retrieve", retrieve), ("generate", generate), ("export", export)):
        builder.add_node(name, node)
    builder.add_edge(START, "query")
    builder.add_conditional_edges("query", next_step, {"continue": "retrieve", "stop": END})
    builder.add_conditional_edges("retrieve", next_step, {"continue": "generate", "stop": END})
    builder.add_edge("generate", "export")
    builder.add_edge("export", END)
    return builder.compile(checkpointer=checkpointer)


async def start_plan(graph, config, session_id, ticket_id, output_dir):
    if (await graph.aget_state(config)).next:
        raise ValueError("已有未完成规划，请先 /resume-plan 或 /cancel-plan。")
    state = {
        "session_id": session_id, "plan_id": uuid4().hex, "ticket_id": ticket_id,
        "output_dir": str(output_dir), "status": "running", "ticket": None,
        "retrieval": None, "plan": None, "generation_count": 0,
        "json_path": None, "markdown_path": None,
    }
    event(state, "plan_start", {})
    return await graph.ainvoke(state, config=config, durability="sync")


async def resume_plan(graph, config, mcp_client):
    snapshot = await graph.aget_state(config)
    if not snapshot.next:
        raise ValueError("没有待恢复的规划任务。")
    state = snapshot.values
    if state.get("retrieval") and any(n in snapshot.next for n in ("generate", "export")):
        current_ticket = await read_tool(mcp_client, "get_ticket", {"ticket_id": state["ticket_id"]}, state)
        if current_ticket.get("status") != "ok":
            raise ValueError("原工单已无法查询，未继续规划；可 /cancel-plan。")
        ticket = current_ticket["ticket"]
        if ticket.get("ticket_id") != state["ticket_id"]:
            raise ValueError("恢复时工单编号不一致，未继续规划。")
        revision = await read_tool(mcp_client, "get_knowledge_revision", {}, state)
        if ticket != state["ticket"] or revision.get("knowledge_revision") != state["retrieval"].get("knowledge_revision"):
            retrieved = await read_tool(mcp_client, "search_knowledge", {"query": ticket["description"], "top_k": 3}, state)
            if not isinstance(retrieved.get("knowledge_revision"), str):
                raise ValueError("刷新资料时缺少内容摘要，未继续规划。")
            status = "running" if retrieved.get("matches") else "insufficient_evidence"
            await graph.aupdate_state(config, {
                "ticket": ticket, "retrieval": retrieved, "plan": None, "status": status,
                "json_path": None, "markdown_path": None,
            }, as_node="retrieve")
            event(state, "plan_evidence_refreshed", {"knowledge_revision": retrieved["knowledge_revision"]})
            print("[规划] 工单或资料已变化，旧草稿已失效；将基于新证据生成，需一次模型请求。")
            if status != "running":
                return (await graph.aget_state(config)).values
        elif "export" in snapshot.next:
            print("[规划] 资料未变，复用已保存草稿，仅重试导出，不请求模型。")
        else:
            print("[规划] 资料未变，复用查询与检索结果，继续生成草稿。")
    event(state, "plan_resume", {"next_nodes": list((await graph.aget_state(config)).next)})
    return await graph.ainvoke(None, config=config, durability="sync")


async def cancel_plan(graph, config):
    snapshot = await graph.aget_state(config)
    if not snapshot.values or not snapshot.next:
        raise ValueError("没有待取消的规划任务。")
    await graph.aupdate_state(config, {"status": "cancelled", "plan": None}, as_node="export")
    event(snapshot.values, "plan_cancelled", {})


async def show_plan_status(graph, config):
    snapshot = await graph.aget_state(config)
    print("\n规划流程：")
    if not snapshot.values:
        print("暂无规划记录。")
        return
    state = snapshot.values
    print("工单：", state["ticket_id"], "；状态：", state["status"])
    print("已保存的有效草稿次数：", state.get("generation_count", 0))
    if snapshot.next:
        print("待执行步骤：", ", ".join(snapshot.next), "；输入 /resume-plan 继续。")
    elif state.get("markdown_path"):
        print("Markdown：", state["markdown_path"])
