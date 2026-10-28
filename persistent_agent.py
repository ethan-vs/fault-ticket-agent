"""SQLite 持久化入口：python persistent_agent.py --session study-001。"""

import argparse
import os
import re
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from graph_agent import SYSTEM_PROMPT, build_graph
from trace_logger import new_session_id, write_event


DEFAULT_DB = Path(__file__).resolve().parent / ".state" / "checkpoints.sqlite"


def make_config(session_id):
    """固定 thread_id，才能找到上次保存的检查点。"""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
        raise ValueError("会话编号只能包含英文字母、数字、下划线或连字符，长度1—64。")
    return {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 20,
    }


def start_turn(graph, config, user_text):
    """没有待恢复任务时，开始新一轮；每个步骤同步保存检查点。"""
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("输入不能为空。")

    snapshot = graph.get_state(config)
    if snapshot.next:
        raise ValueError("当前有未完成任务，请先输入 /resume 恢复；不要直接覆盖它。")

    session_id = config["configurable"]["thread_id"]
    history = snapshot.values.get(
        "messages", [{"role": "system", "content": SYSTEM_PROMPT}]
    )
    turn_id = new_session_id()
    initial_state = {
        "messages": deepcopy(history) + [{"role": "user", "content": user_text.strip()}],
        "session_id": session_id,
        "turn_id": turn_id,
        "model_calls": 0,
        "tool_calls": 0,
        "status": "running",
    }
    write_event(session_id, "turn_start", {"turn_id": turn_id, "input": user_text.strip()})
    return graph.invoke(initial_state, config=config, durability="sync")


def resume_turn(graph, config):
    """传入 None，继续保存的执行进度，不重新提交原始任务。"""
    snapshot = graph.get_state(config)
    if not snapshot.next:
        raise ValueError("没有待恢复任务，可以直接输入新需求。")
    write_event(config["configurable"]["thread_id"], "turn_resume", {
        "turn_id": snapshot.values.get("turn_id"),
        "next_nodes": list(snapshot.next),
    })
    return graph.invoke(None, config=config, durability="sync")


def show_status(graph, config):
    snapshot = graph.get_state(config)
    print("会话编号：", config["configurable"]["thread_id"])
    if not snapshot.values:
        print("暂无保存的任务。")
        return
    print("轮次编号：", snapshot.values.get("turn_id"))
    print("已成功返回的模型请求数：", snapshot.values.get("model_calls", 0))
    print("工具执行尝试数：", snapshot.values.get("tool_calls", 0))
    if snapshot.next:
        print("任务未完成，下一节点：", ", ".join(snapshot.next))
        print("输入 /resume 从保存的进度继续。")
    else:
        print("图执行已结束，状态：", snapshot.values.get("status"))
        print("可以输入新需求，继续使用当前会话历史。")


def error_message(error):
    status_code = getattr(error, "status_code", None)
    if isinstance(error, APITimeoutError):
        return "模型请求超时。"
    if isinstance(error, APIConnectionError):
        return "无法连接模型服务，请检查网络。"
    if status_code == 402:
        return "模型账户余额不足。"
    if status_code == 401:
        return "模型身份验证失败，请检查 API Key 配置。"
    if status_code == 429:
        return "模型服务限流，请稍后重试。"
    if isinstance(error, ValueError):
        return "输入、任务状态或模型响应不符合约定，请用 /status 查看进度。"
    return "模型请求失败，请查看运行日志。"


def main():
    parser = argparse.ArgumentParser(description="支持 SQLite 检查点的工单助手")
    parser.add_argument(
        "--session",
        default="study-001",
        help="会话编号，默认使用 study-001",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="检查点数据库文件")
    args = parser.parse_args()
    config = make_config(args.session)

    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError("请先在 .env 配置 DEEPSEEK_API_KEY")

    client = OpenAI(
        api_key=api_key, base_url="https://api.deepseek.com",
        timeout=30.0, max_retries=0,
    )
    args.db.parent.mkdir(parents=True, exist_ok=True)

    # with 结束时关闭数据库连接；数据库文件会保留。
    with SqliteSaver.from_conn_string(str(args.db)) as saver:
        graph = build_graph(client, checkpointer=saver)
        write_event(args.session, "session_open", {"entry": "persistent_agent"})
        print("持久化工单助手已启动。数据库：", args.db)
        print("命令：/status 查看进度；/resume 恢复；exit 退出。")
        show_status(graph, config)

        while True:
            try:
                user_text = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出。重新启动时请使用相同的 --session。")
                break
            if user_text.lower() == "exit":
                break
            if user_text == "/status":
                show_status(graph, config)
                continue
            if not user_text:
                print("输入不能为空。")
                continue
            if user_text.startswith("/") and user_text != "/resume":
                print("未知命令，请使用 /status 或 /resume。")
                continue
            if user_text != "/resume" and graph.get_state(config).next:
                print("当前有未完成任务，请先 /resume；新任务可另用一个会话编号启动。")
                continue
            if user_text == "/resume" and not graph.get_state(config).next:
                print("没有待恢复任务，可以直接输入新需求。")
                continue

            try:
                result = (
                    resume_turn(graph, config)
                    if user_text == "/resume"
                    else start_turn(graph, config, user_text)
                )
            except (APIError, ValueError, KeyboardInterrupt) as error:
                snapshot = graph.get_state(config)
                text = "执行被中断。" if isinstance(error, KeyboardInterrupt) else error_message(error)
                write_event(args.session, "turn_attempt_failed", {
                    "turn_id": snapshot.values.get("turn_id"),
                    "error_type": type(error).__name__,
                    "status_code": getattr(error, "status_code", None),
                    "message": text,
                    "next_nodes": list(snapshot.next),
                })
                print("\n助手：", text)
                show_status(graph, config)
                continue

            text = result["messages"][-1].get("content", "")
            write_event(args.session, "turn_success" if result["status"] == "completed" else "turn_failed", {
                "turn_id": result["turn_id"], "status": result["status"],
                "model_calls": result["model_calls"], "tool_calls": result["tool_calls"],
                "answer": text,
            })
            print("\n助手：", text)

        write_event(args.session, "session_close", {"message": "关闭入口，保留检查点"})


if __name__ == "__main__":
    main()
