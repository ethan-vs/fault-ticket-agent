"""运行固定离线业务测试并生成HTML/JSON报告，不主动请求DeepSeek。"""

import argparse
import hashlib
import html
import importlib.metadata
import json
import sys
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter


OFFLINE_MODULES = [
    "test_ticket_tools", "test_agent_errors", "test_tool_execution", "test_graph_agent",
    "test_persistence", "test_rag_search", "test_rag_answer", "test_graph_rag",
    "test_rag_normalization", "test_ticket_switch", "test_mcp_protocol", "test_mcp_agent",
    "test_evidence_policy", "test_plan_tasks", "test_plan_command", "test_planning_flow",
]


class TimedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = {}

    def startTest(self, test):
        super().startTest(test)
        self.records[test.id()] = {
            "id": test.id(), "description": test.shortDescription() or "",
            "status": "running", "details": [], "started": perf_counter(),
        }

    def mark(self, test, status, details=None):
        row = self.records[test.id()]
        row["status"] = status
        if details:
            row["details"].append(details)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.mark(test, "passed")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.mark(test, "failed", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self.mark(test, "error", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.mark(test, "skipped", reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.mark(test, "expected_failure", self._exc_info_to_string(err, test))

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.mark(test, "unexpected_success", "预期失败的测试意外成功，请检查测试约定。")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.mark(test, "failed", f"{subtest}\n{self._exc_info_to_string(err, test)}")

    def stopTest(self, test):
        row = self.records[test.id()]
        row["elapsed_ms"] = round((perf_counter() - row.pop("started")) * 1000, 3)
        super().stopTest(test)


def summarize(records, missing_modules):
    counts = Counter(row["status"] for row in records)
    failed = sum(counts[s] for s in ("failed", "error", "unexpected_success", "running"))
    complete = not missing_modules and bool(records) and not counts["running"]
    return {
        "test_methods": len(records), "passed": counts["passed"], "failed": failed,
        "skipped": counts["skipped"], "expected_failures": counts["expected_failure"],
        "complete": complete,
        "status": "passed" if complete and failed == 0 and not counts["skipped"] and not counts["expected_failure"] else "needs_review",
    }


def source_manifest(project):
    paths = list(project.glob("*.py")) + list((project / "knowledge_base").glob("*.json"))
    return {p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def collect_artifacts(project):
    result = {"knowledge": {"status": "unavailable"}, "latest_plan": None}
    try:
        from rag_search import get_knowledge_revision
        result["knowledge"] = get_knowledge_revision(project / "knowledge_base" / "documents.json")
    except (OSError, ValueError) as error:
        result["knowledge"] = {"status": "invalid", "error_type": type(error).__name__}
    paths = list((project / ".state" / "test_plans").glob("*/plan.json"))
    if paths:
        latest = max(paths, key=lambda p: p.stat().st_mtime)
        try:
            plan = json.loads(latest.read_text(encoding="utf-8"))
            result["latest_plan"] = {
                "artifact_id": latest.parent.name, "ticket_id": plan.get("ticket_id"),
                "title": plan.get("title"), "task_count": len(plan.get("tasks", [])),
                "plan_status": plan.get("plan_status"), "execution_status": plan.get("execution_status"),
                "requires_human_review": plan.get("requires_human_review"),
                "source_revision_current": (
                    plan.get("knowledge_revision") == result["knowledge"].get("knowledge_revision")
                    if plan.get("knowledge_revision") and result["knowledge"].get("knowledge_revision") else None
                ),
            }
        except (OSError, ValueError, TypeError) as error:
            result["latest_plan"] = {"status": "invalid", "error_type": type(error).__name__}
    return result


def render_html(report):
    escape = lambda value: html.escape(str(value), quote=True)
    summary = report["summary"]
    rows = []
    for item in report["tests"]:
        details = "\n\n".join(item["details"])
        expanded = f"<details><summary>查看详情</summary><pre>{escape(details)}</pre></details>" if details else ""
        rows.append(f"<tr><td><span class='badge {escape(item['status'])}'>{escape(item['status'])}</span></td><td><code>{escape(item['id'])}</code><div class='muted'>{escape(item['description'])}</div>{expanded}</td><td>{item['elapsed_ms']:.1f}</td></tr>")
    plan = report["artifacts"].get("latest_plan")
    plan_text = json.dumps(plan, ensure_ascii=False, indent=2) if plan else "暂无已导出草稿。可先使用 /plan 生成，再重新运行本报告。"
    knowledge = report["artifacts"]["knowledge"]
    packages = " · ".join(f"{escape(k)} {escape(v)}" for k, v in report["packages"].items())
    cards = "".join(f"<div class='card'><div class='muted'>{label}</div><strong>{value}</strong></div>" for label, value in (
        ("测试方法", summary["test_methods"]), ("通过", summary["passed"]),
        ("失败/错误", summary["failed"]), ("总耗时（秒）", f"{report['elapsed_seconds']:.2f}"),
    ))
    missing = "、".join(report["missing_modules"]) or "无"
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>故障工单Agent · 自动化质量报告</title>
<style>body{{margin:0;background:#f3f6fa;color:#172338;font:15px/1.65 'Segoe UI','Microsoft YaHei',sans-serif}}main{{max-width:1150px;margin:35px auto;padding:0 22px}}header{{padding:28px;border-radius:18px;background:#142844;color:white}}h1{{margin:0 0 8px;font-size:27px}}h2{{font-size:19px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0}}.card,section{{background:white;border:1px solid #e0e7ef;border-radius:13px;padding:20px}}section{{margin-top:18px}}strong{{display:block;font-size:30px}}.muted{{color:#65758a;font-size:13px}}header .muted{{color:#c9d7e8}}.note{{background:#edf4ff;border-left:4px solid #4c83d6;padding:13px}}.badge{{padding:4px 9px;border-radius:20px;background:#fff0d8;font-size:12px}}.passed{{background:#ddf4e8;color:#17643e}}.failed,.error,.unexpected_success{{background:#fde4e5;color:#9b2730}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;vertical-align:top;border-bottom:1px solid #e5ebf2;padding:12px 8px}}code{{word-break:break-all}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fb;padding:12px;border-radius:8px}}input{{box-sizing:border-box;width:100%;padding:11px;border:1px solid #cbd5e1;border-radius:8px;margin-bottom:14px}}@media(max-width:650px){{.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:22px}}}}</style></head>
<body><main><header><h1>故障工单 Agent · 自动化质量报告</h1><div>检查结论：{escape(summary['status'])}</div><div class="muted">{escape(report['timestamp'])} · {escape(report['project_name'])}</div></header>
<div class="cards">{cards}</div><div class="note">本报告运行固定的离线业务检查：模型响应使用模拟数据，部分MCP、BGE和SQLite集成实际运行。通过数不是模型准确率、业务成功率或节省工时。测试耗时包含启动与加载，不是生产性能基准。</div>
<section><h2>检查范围与可复现信息</h2><p>缺少的测试模块：{escape(missing)}</p><p>{packages}</p><p>源码与资料清单摘要：<code>{escape(report['manifest_sha256'])}</code>（完整清单见 report.json；不读取 .env）</p></section>
<section><h2>当前知识库</h2><pre>{escape(json.dumps(knowledge,ensure_ascii=False,indent=2))}</pre><p class="muted">有效标记由资料维护者提供；内容摘要只能检测变化。</p></section>
<section><h2>最新任务草稿</h2><pre>{escape(plan_text)}</pre><p class="muted">草稿尚未执行，内容和执行条件仍需人工审核；资料一致性仅反映本次检查时的本地资料状态。</p></section>
<section><h2>真实模型评测</h2><p>本报告未执行真实 DeepSeek 评测。后续将单独记录任务完成、引用和错误处理结果，不能用上面的通过数替代。</p></section>
<section><h2>逐项结果</h2><input id="filter" placeholder="按模块、测试名称或状态筛选"><table><thead><tr><th>状态</th><th>测试方法</th><th>耗时ms</th></tr></thead><tbody id="rows">{''.join(rows)}</tbody></table></section>
<script>document.getElementById('filter').addEventListener('input',function(){{const q=this.value.toLowerCase();document.querySelectorAll('#rows tr').forEach(row=>{{row.hidden=!row.textContent.toLowerCase().includes(q)}})}});</script>
</main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description="生成本地自动化质量报告；不执行真实模型评测")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    if not (project / "mcp_agent.py").is_file():
        raise ValueError("项目目录中未找到 mcp_agent.py。")
    sys.path.insert(0, str(project))
    missing = [name for name in OFFLINE_MODULES if not (project / f"{name}.py").is_file()]
    present = [name for name in OFFLINE_MODULES if name not in missing]
    started = perf_counter()
    suite = unittest.TestLoader().loadTestsFromNames(present)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, resultclass=TimedResult).run(suite)
    tests = list(result.records.values())
    manifest = source_manifest(project)
    packages = {"Python": sys.version.split()[0]}
    for name in ("mcp", "langgraph", "langgraph-checkpoint-sqlite", "sentence-transformers", "openai"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "未安装"
    now = datetime.now(timezone(timedelta(hours=8)))
    report = {
        "schema_version": 1, "timestamp": now.isoformat(), "project_name": project.name,
        "evaluation_type": "offline_automated_checks", "live_model_evaluation": False,
        "elapsed_seconds": round(perf_counter() - started, 3), "packages": packages,
        "summary": summarize(tests, missing), "missing_modules": missing, "tests": tests,
        "source_manifest": manifest,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifacts": collect_artifacts(project),
    }
    root = args.output_dir or project / ".state" / "quality_reports"
    folder = root / now.strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (folder / "report.html").write_text(render_html(report),encoding="utf-8")
    print("\n报告结论：", report["summary"]["status"])
    print("HTML：", folder / "report.html")
    print("JSON：", folder / "report.json")
    return 0 if report["summary"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
