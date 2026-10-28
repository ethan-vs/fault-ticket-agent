"""独立评测入口。默认仅预览；--live 才会请求模型，不修改日常会话数据库。"""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import threading
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4


ROOT = Path(__file__).resolve().parent


class RequestBudgetExceeded(RuntimeError):
    pass


class MeteredModel:
    """统计实际SDK调用尝试，包含失败请求；不读取或记录API Key。"""
    def __init__(self, client, max_requests):
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("请求上限必须为正整数。")
        self.client, self.limit = client, max_requests
        self.requests, self.case_id = [], None
        self.lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        with self.lock:
            if len(self.requests) >= self.limit:
                raise RequestBudgetExceeded("达到本次评测请求上限；没有继续调用模型。")
            row = {"case_id": self.case_id, "requested_model": kwargs.get("model"), "status": "started",
                   "request": deepcopy(kwargs)}
            self.requests.append(row)
        started = perf_counter()
        try:
            response = self.client.chat.completions.create(**kwargs)
            row.update({"status": "returned", "returned_model": getattr(response, "model", None)})
            row["responses"] = [{"finish_reason": choice.finish_reason,
                                  "message": choice.message.model_dump(exclude_none=True)}
                                 for choice in response.choices]
            usage = getattr(response, "usage", None)
            row["usage"] = ({key: getattr(usage, key, None) for key in
                             ("prompt_tokens", "completion_tokens", "total_tokens")} if usage else None)
            return response
        except Exception as error:
            row.update({"status": "error", "error_type": type(error).__name__,
                        "status_code": getattr(error, "status_code", None)})
            raise
        finally:
            row["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)


class RecordingMCP:
    def __init__(self, client):
        self.client, self.calls = client, []

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def call_tool(self, name, arguments):
        row = {"name": name, "arguments": deepcopy(arguments)}
        self.calls.append(row)
        try:
            result = await self.client.call_tool(name, arguments)
            row["result"] = deepcopy(result.structured_content)
            row["is_error"] = result.is_error
            return result
        except Exception as error:
            row["error_type"] = type(error).__name__
            raise


def load_cases(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("评测集格式错误。")
    seen = set()
    for case in data["cases"]:
        if not isinstance(case, dict) or not re.fullmatch(r"E[0-9]{2}", case.get("id", "")) or case["id"] in seen:
            raise ValueError("用例编号格式错误或重复。")
        seen.add(case["id"])
        if case.get("kind") not in ("analysis", "plan") or not case.get("name") or not case.get("human_checks"):
            raise ValueError("用例类型、名称或人工复核项缺失。")
        if case["kind"] == "analysis":
            if not case.get("turns"):
                raise ValueError("分析用例缺少轮次。")
            for turn in case["turns"]:
                if not isinstance(turn.get("input"), str) or not turn["input"].strip():
                    raise ValueError("用例输入不能为空。")
            expectations = [turn.get("expect") for turn in case["turns"]]
        else:
            if not re.fullmatch(r"TK-[0-9]{3}", case.get("ticket_id", "")):
                raise ValueError("规划用例的工单编号不符合约定。")
            expectations = [case.get("expect")]
        for expected in expectations:
            if not isinstance(expected, dict) or not isinstance(expected.get("tools"), list):
                raise ValueError("缺少工具调用预期。")
            if any(name not in ("get_ticket", "search_knowledge") for name in expected["tools"]):
                raise ValueError("用例指定了未开放工具。")
            if "get_ticket" in expected["tools"] and not (expected.get("ticket_id") and expected.get("ticket_status") in ("ok", "not_found")):
                raise ValueError("工单查询预期不完整。")
            if case["kind"] == "plan" and expected.get("ticket_id") != case["ticket_id"]:
                raise ValueError("规划预期编号与输入不一致。")
    return data


def ticket_citation_flags(answer, ticket):
    """启发式标记明确的工单事实段误挂资料引用；不判断所有语义支持关系。"""
    findings = []
    for paragraph in re.split(r"\n\s*\n", answer):
        plain = re.sub(r"[#*`>]", "", paragraph).strip()
        fact_heading = re.match(r"^(?:工单\s*(?:TK-\d{3})?\s*(?:记录的事实|记录|事实)|(?:已知|工单)?事实)", plain)
        contains_ticket = any(isinstance(ticket.get(key), str) and ticket[key] and ticket[key] in plain
                              for key in ("ticket_id", "device"))
        if fact_heading and contains_ticket and re.search(r"\[(?:RULE|CASE)-[^\[\]\n]+\]", paragraph):
            findings.append({"code": "BC-CITE-001", "paragraph": paragraph,
                "reason": "工单事实段混挂了规范或案例引用，请分开标明查询事实与资料建议的来源。"})
    return findings


def check_trace(expected, calls, state, plan=False):
    """只做可核验的结构与轨迹检查，不把关键词匹配当作语义正确。"""
    checks = []
    def add(name, condition, actual=None):
        checks.append({"name": name, "passed": bool(condition), "actual": actual})
    relevant = [row for row in calls if row["name"] in ("get_ticket", "search_knowledge")]
    names = [row["name"] for row in relevant]
    add("工具调用顺序与次数", names == expected["tools"], names)
    add("图执行完成", state.get("status") == "completed", state.get("status"))
    if not plan:
        messages = state.get("messages", [])
        turn_start = next((i for i in range(len(messages) - 1, -1, -1)
                           if messages[i].get("role") == "user"), 0)
        proposed = [call["function"]["name"] for message in messages[turn_start:]
                    for call in message.get("tool_calls", [])]
        add("模型提出的工具顺序与次数", proposed == expected["tools"], proposed)
    if expected.get("ticket_id"):
        tickets = [row for row in relevant if row["name"] == "get_ticket"]
        expected_id = expected["ticket_id"]
        correct = bool(tickets) and all(
            str(row["arguments"].get("ticket_id", "")).strip().upper() == expected_id
            and isinstance(row.get("result"), dict)
            and row["result"].get("status") == expected["ticket_status"]
            and (expected["ticket_status"] != "ok" or row["result"].get("ticket", {}).get("ticket_id") == expected_id)
            for row in tickets)
        add("查询目标和业务状态", correct, expected_id)
    retrievals = [row.get("result") for row in relevant if row["name"] == "search_knowledge"]
    for result in retrievals:
        add("检索返回有效片段与摘要", isinstance(result, dict) and result.get("status") == "ok"
            and bool(result.get("matches")) and bool(result.get("knowledge_revision")))
    if plan:
        draft = state.get("plan") or {}
        ids = [item["chunk_id"] for item in draft.get("evidence", [])]
        add("草稿仍待审核、任务未执行", draft.get("plan_status") == "draft_pending_review"
            and draft.get("execution_status") == "not_run" and draft.get("requires_human_review") is True)
        add("草稿编号与任务存在", draft.get("ticket_id") == expected.get("ticket_id") and bool(draft.get("tasks")))
    else:
        answer = state.get("messages", [{}])[-1].get("content", "")
        add("最终回答非空", isinstance(answer, str) and bool(answer.strip()))
        review = state.get("rag_review") or {}
        if review and isinstance(answer, str):
            facts = next((row.get("result", {}).get("ticket") for row in reversed(relevant)
                          if row["name"] == "get_ticket" and isinstance(row.get("result"), dict)
                          and row["result"].get("status") == "ok"), None)
            if facts:
                flags = ticket_citation_flags(answer, facts)
                add("工单事实引用启发式筛查（仍需人工复核）", not flags, flags)
        ids = [item["chunk_id"] for item in review.get("selected_sources", [])]
        if expected.get("required_document"):
            add("本轮生成通过引用校验", review.get("status") == "answered" and bool(ids))
    documents = {chunk.split("@", 1)[0] for chunk in ids}
    if expected.get("required_document"):
        add("采用预期规范", expected["required_document"] in documents, sorted(documents))
    forbidden = set(expected.get("forbidden_documents", []))
    if forbidden:
        add("没有采用预设干扰资料", not documents & forbidden, sorted(documents & forbidden))
    return checks


def summarize(selected_ids, records, total_cases, stopped_reason=None):
    attempted = {row["id"] for row in records}
    passed = sum(row["status"] == "auto_pass_pending_review" for row in records)
    errors = sum(row["status"] == "execution_error" for row in records)
    return {"requested_cases": len(selected_ids), "attempted_cases": len(records),
            "auto_check_passed": passed, "auto_check_failed": len(records) - passed - errors,
            "execution_errors": errors, "not_run": [cid for cid in selected_ids if cid not in attempted],
            "selected_batch_complete": set(selected_ids) == attempted,
            "whole_dataset_executed": set(selected_ids) == attempted and len(selected_ids) == total_cases,
            "manual_review": "pending", "overall_acceptance": "not_determined",
            "stopped_reason": stopped_reason}


def render_html(report):
    esc = lambda x: html.escape(str(x), quote=True)
    sections = []
    for row in report["cases"]:
        turns = "".join(f"<h4>第{i}轮：{esc(t['input'])}</h4><pre>{esc(t.get('answer',''))}</pre>" for i, t in enumerate(row.get("turns", []), 1))
        if row.get("plan"):
            turns += f"<h4>草稿</h4><pre>{esc(json.dumps(row['plan'],ensure_ascii=False,indent=2))}</pre>"
        checks = "".join(f"<li>{'通过' if c['passed'] else '失败'}：{esc(c['name'])} · {esc(c.get('actual',''))}</li>" for c in row["checks"])
        manual = "".join(f"<li>{esc(x)}</li>" for x in row["human_checks"])
        error = f"<p>执行错误：{esc(row.get('error',{}))}</p>" if row.get("error") else ""
        sections.append(f"<section><h2>{esc(row['id'])} · {esc(row['name'])}</h2><p>{esc(row['status'])} · {row['elapsed_seconds']:.2f} 秒</p>{error}{turns}<h3>自动检查</h3><ul>{checks}</ul><h3>人工复核（尚未完成）</h3><ul>{manual}</ul></section>")
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent真实模型评测</title><style>body{{background:#f1f5f9;color:#18304b;font:15px/1.7 'Microsoft YaHei',sans-serif;margin:0}}main{{max-width:1050px;margin:32px auto;padding:0 20px}}header,section{{background:white;border-radius:14px;padding:24px;margin:18px 0}}header{{border-top:6px solid #346fc2}}h1{{font-size:26px}}h2{{font-size:20px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fb;padding:15px;border-radius:8px}}.note{{border-left:4px solid #d49a31;padding:12px;background:#fff5df}}</style><main><header><h1>故障工单 Agent · 真实模型评测</h1><p>{esc(report['timestamp'])}</p><p class="note">模型API实际调用；工单与知识库仍是项目模拟数据。自动检查通过不等于业务结论正确，所有语义质量待人工复核。本集合用于开发回归，不是独立行业基准。</p><pre>{esc(json.dumps(report['summary'],ensure_ascii=False,indent=2))}</pre><p>实际SDK调用尝试：{len(report['model_requests'])}；资料摘要前后一致：{esc(report.get('knowledge_unchanged'))}。</p><p>调用耗时是单次观察，不是负载测试或生产性能结论。完整轨迹、版本清单和Token记录见 report.json。</p></header>{''.join(sections)}</main></html>'''


def save_reports(folder, report):
    folder.mkdir(parents=True, exist_ok=True)
    payloads = {"report.json": report,
                "badcases.json": [r for r in report["cases"] if r["status"] != "auto_pass_pending_review"]}
    for name, payload in payloads.items():
        target = folder / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
    (folder / "report.html").write_text(render_html(report), encoding="utf-8")


async def evaluate(data, chosen, model, folder, project):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from mcp import Client
    from check_mcp import business_result, server_parameters
    from graph_agent import build_graph
    from mcp_agent import create_mcp_executor, discover_model_tools, start_turn
    from persistent_agent import make_config
    from planning_flow import build_planning_graph, plan_config, start_plan
    import trace_logger

    folder.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "evaluation_type": "live_model_synthetic_business_data",
              "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "dataset_sha256": hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
              "source_manifest": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(project.glob('*.py'))},
              "cases": [], "model_requests": model.requests, "knowledge_unchanged": None,
              "summary": summarize([c['id'] for c in chosen], [], len(data['cases']))}
    old_log_dir = trace_logger.LOG_DIR
    trace_logger.LOG_DIR = folder / "logs"
    stopped = None
    try:
        async with Client(server_parameters(), read_timeout_seconds=120.0) as connected:
            mcp_client = RecordingMCP(connected)
            initial_revision = business_result(await connected.call_tool("get_knowledge_revision", {}))
            report["knowledge_before"] = initial_revision
            model_tools = discover_model_tools(await connected.list_tools())
            for case in chosen:
                print(f"\n[评测] {case['id']} {case['name']}")
                row = {"id": case["id"], "name": case["name"], "kind": case["kind"],
                       "turns": [], "checks": [], "human_checks": case["human_checks"],
                       "status": "execution_error", "manual_review": "pending"}
                started = perf_counter()
                model.case_id = case["id"]
                session = f"eval-{uuid4().hex[:12]}-{case['id']}"
                row["session_id"] = session
                call_start = len(mcp_client.calls)
                try:
                    async with AsyncSqliteSaver.from_conn_string(str(folder / "checkpoints.sqlite")) as saver:
                        if case["kind"] == "analysis":
                            graph = build_graph(model, checkpointer=saver,
                                async_tool_executor=create_mcp_executor(mcp_client), model_tools=model_tools)
                            config = make_config(session)
                            for turn in case["turns"]:
                                calls_at_start = len(mcp_client.calls)
                                turn_row = {"input": turn["input"], "status": "started"}
                                row["turns"].append(turn_row)
                                result = await start_turn(graph, config, turn["input"], mcp_client=mcp_client)
                                turn_row.update({"answer": result["messages"][-1].get("content", ""),
                                    "status": result["status"], "rag_review": result.get("rag_review"),
                                    "model_calls": result.get("model_calls"), "tool_calls": result.get("tool_calls")})
                                row["checks"].extend(check_trace(turn["expect"], mcp_client.calls[calls_at_start:], result))
                        else:
                            graph = build_planning_graph(mcp_client, model, checkpointer=saver)
                            config = plan_config(session)
                            result = await start_plan(graph, config, session, case["ticket_id"], folder / "plans")
                            row["plan"] = result.get("plan")
                            row["checks"].extend(check_trace(case["expect"], mcp_client.calls[call_start:], result, plan=True))
                        row["status"] = "auto_pass_pending_review" if row["checks"] and all(c["passed"] for c in row["checks"]) else "auto_check_failed"
                except Exception as error:
                    # 每个任务边界记录失败并继续或停止，不把异常吞成成功，不记录可能含敏感信息的服务端全文。
                    row["error"] = {"type": type(error).__name__, "status_code": getattr(error, "status_code", None)}
                    # SQLite连接已退出，不再读取检查点；文件和事件日志保留用于排查。
                    if isinstance(error, RequestBudgetExceeded):
                        stopped = "request_budget_exhausted"
                    elif getattr(error, "status_code", None) in (401, 402, 429):
                        stopped = "upstream_access_or_quota_error"
                row["mcp_calls"] = deepcopy(mcp_client.calls[call_start:])
                row["elapsed_seconds"] = round(perf_counter() - started, 3)
                report["cases"].append(row)
                report["summary"] = summarize([c['id'] for c in chosen], report["cases"], len(data["cases"]), stopped)
                save_reports(folder, report)
                print("[结果]", row["status"], "；语义质量待人工复核")
                if stopped:
                    break
            final_revision = business_result(await connected.call_tool("get_knowledge_revision", {}))
            report["knowledge_after"] = final_revision
            report["knowledge_unchanged"] = initial_revision.get("knowledge_revision") == final_revision.get("knowledge_revision")
    except Exception as error:
        report["run_error"] = {"type": type(error).__name__, "status_code": getattr(error, "status_code", None)}
        stopped = stopped or "evaluation_infrastructure_error"
    finally:
        trace_logger.LOG_DIR = old_log_dir
        report["summary"] = summarize([c['id'] for c in chosen], report["cases"], len(data["cases"]), stopped)
        report["summary"]["source_snapshot_consistent"] = report["knowledge_unchanged"]
        save_reports(folder, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="实际调用DeepSeek，可能产生费用")
    parser.add_argument("--limit", type=int, default=3, help="按顺序选择前N个用例，默认3")
    parser.add_argument("--case", help="只选择一个用例，如E04")
    parser.add_argument("--max-requests", type=int, default=8, help="SDK调用尝试总上限，失败也计入")
    parser.add_argument("--dataset", type=Path, default=ROOT / "agent_eval_cases.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data = load_cases(args.dataset)
    if args.limit < 1 or args.max_requests < 1:
        parser.error("limit 和 max-requests 必须大于零。")
    chosen = [c for c in data["cases"] if c["id"] == args.case] if args.case else data["cases"][:args.limit]
    if not chosen:
        parser.error("没有匹配用例。")
    for case in chosen:
        print(case["id"], case["name"])
    print(f"选中 {len(chosen)}/{len(data['cases'])} 个用例；请求尝试上限 {args.max_requests} 次。")
    if not args.live:
        print("当前仅预览，未读取密钥、启动MCP或调用模型。确认后显式添加 --live 执行。")
        return 0
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise ValueError("请在本地.env中配置DEEPSEEK_API_KEY，不要发送或上传密钥。")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=45.0, max_retries=0)
    model = MeteredModel(client, args.max_requests)
    folder = (args.output_dir or ROOT / ".state" / "live_evaluations") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    try:
        report = asyncio.run(evaluate(data, chosen, model, folder, ROOT))
    finally:
        client.close()
    print("\n评测统计：", json.dumps(report["summary"], ensure_ascii=False))
    print("报告：", folder / "report.html")
    print("详情：", folder / "report.json")
    print("自动失败记录：", folder / "badcases.json")
    print("检查完成不代表业务验收通过；请按报告中的人工复核项检查输出。")
    return 0 if report["summary"]["selected_batch_complete"] and not report["summary"]["execution_errors"] and not report["summary"]["auto_check_failed"] and not report.get("run_error") and report["knowledge_unchanged"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
