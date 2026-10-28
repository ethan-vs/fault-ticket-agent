"""评测器自身测试；模型和MCP使用Mock，LangGraph与SQLite实际执行。"""

from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from openai import APITimeoutError
from openai.types.chat import ChatCompletionMessage

from evaluate_agent_live import (
    MeteredModel, RequestBudgetExceeded, check_trace, evaluate, load_cases,
    render_html, summarize, ticket_citation_flags,
)


ROOT = Path(__file__).resolve().parent


def response(content=None, tool=None, args=None):
    calls = None if tool is None else [{"id": "call_eval", "type": "function",
        "function": {"name": tool, "arguments": json.dumps(args)}}]
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls" if calls else "stop",
        message=ChatCompletionMessage(role="assistant", content=content, tool_calls=calls))],
        model="mock-model", usage=None)


class FakeConnection:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def list_tools(self):
        return SimpleNamespace(tools=[SimpleNamespace(name=name, description=name,
            input_schema={"type": "object", "properties": {}})
            for name in ("get_ticket", "search_knowledge")])

    async def call_tool(self, name, arguments):
        if name == "get_knowledge_revision":
            data = {"status": "ok", "knowledge_revision": "test-revision"}
        elif name == "get_ticket":
            data = {"status": "ok", "ticket": {"ticket_id": arguments['ticket_id'].strip().upper(),
                "description": "测试无线故障", "device": "AP-01", "processing_status": "待处理"}}
        elif name == "search_knowledge":
            data = {"status": "ok", "knowledge_revision": "test-revision", "matches": [{
                "chunk_id": "RULE-SOFT-001@1.0#001", "text": "软件闪退时，先记录软件版本，再收集错误日志。"}]}
        else:
            raise AssertionError("本测试不应发起检索")
        return SimpleNamespace(structured_content=data, is_error=False)


class TestLiveEvaluation(unittest.TestCase):
    def test_recorded_fact_citation_badcase_is_flagged(self):
        answer = "工单 TK-001 记录的事实：多台电脑无法连接Wi-Fi，涉及AP-01，状态待处理。[RULE-NET-001@1.0#001]"
        flags = ticket_citation_flags(answer, {"ticket_id": "TK-001", "device": "AP-01"})
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]['code'], 'BC-CITE-001')

    def test_separate_fact_source_and_advice_citation_are_allowed(self):
        answer = "工单事实（来源：get_ticket，TK-001）\n设备AP-01，状态待处理。\n\n资料建议：先收集连接错误提示。[RULE-NET-001@1.0#001]"
        self.assertEqual(ticket_citation_flags(answer, {"ticket_id": "TK-001", "device": "AP-01"}), [])

    def test_failure_consumes_request_budget(self):
        raw = MagicMock()
        raw.chat.completions.create.side_effect = [RuntimeError("mock failure"), response("ok")]
        model = MeteredModel(raw, 2)
        with self.assertRaises(RuntimeError):
            model.chat.completions.create(model="mock")
        model.chat.completions.create(model="mock")
        with self.assertRaises(RequestBudgetExceeded):
            model.chat.completions.create(model="mock")
        self.assertEqual(raw.chat.completions.create.call_count, 2)
        self.assertEqual([row["status"] for row in model.requests], ["error", "returned"])

    def test_duplicate_ids_are_rejected(self):
        data = load_cases(ROOT / "agent_eval_cases.json")
        data["cases"].append(deepcopy(data["cases"][0]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_partial_batch_is_not_reported_as_full_dataset(self):
        rows = [{"id": "E01", "status": "auto_pass_pending_review"}]
        result = summarize(["E01", "E02"], rows, 8, "request_budget_exhausted")
        self.assertFalse(result["selected_batch_complete"])
        self.assertFalse(result["whole_dataset_executed"])
        self.assertEqual(result["not_run"], ["E02"])
        self.assertEqual(result["overall_acceptance"], "not_determined")

    def test_error_is_not_a_failed_semantic_judgment(self):
        rows = [{"id": "E01", "status": "execution_error"}, {"id": "E02", "status": "auto_check_failed"}]
        result = summarize(["E01", "E02"], rows, 8)
        self.assertEqual(result["execution_errors"], 1)
        self.assertEqual(result["auto_check_failed"], 1)
        self.assertEqual(result["auto_check_passed"], 0)

    def test_wrong_query_target_is_rejected(self):
        expected = {"tools": ["get_ticket"], "ticket_id": "TK-002", "ticket_status": "ok"}
        calls = [{"name": "get_ticket", "arguments": {"ticket_id": "TK-001"},
                  "result": {"status": "ok", "ticket": {"ticket_id": "TK-001"}}}]
        checks = check_trace(expected, calls, {"status": "completed", "messages": [{"content": "查询完成"}]})
        self.assertTrue(any(not c["passed"] for c in checks))

    def test_source_direction_rejects_wrong_domain(self):
        state = {"status": "completed", "messages": [{"content": "回答"}],
            "rag_review": {"status": "answered", "selected_sources": [{"chunk_id": "RULE-NET-001@1.0#001"}]}}
        checks = check_trace({"tools": [], "required_document": "RULE-SOFT-001",
            "forbidden_documents": ["RULE-NET-001"]}, [], state)
        self.assertEqual(sum(not c["passed"] for c in checks), 2)

    def test_html_escapes_model_output(self):
        report = {"timestamp": "now", "summary": {}, "model_requests": [], "cases": [{
            "id": "E01", "name": "test", "status": "auto_check_failed", "elapsed_seconds": 1,
            "turns": [{"input": "test", "answer": "<script>alert(1)</script>"}],
            "checks": [], "human_checks": ["<img src=x>"]}]}
        rendered = render_html(report)
        self.assertNotIn("<script>alert(1)</script>", rendered)
        self.assertNotIn("<img src=x>", rendered)
        self.assertIn("&lt;script&gt;", rendered)


class TestEvaluationFlow(unittest.IsolatedAsyncioTestCase):
    async def run_fake(self, responses, chosen_count=1, budget=8, indices=None):
        data = load_cases(ROOT / "agent_eval_cases.json")
        raw = MagicMock()
        raw.chat.completions.create.side_effect = responses
        model = MeteredModel(raw, budget)
        with tempfile.TemporaryDirectory() as folder, patch("mcp.Client", return_value=FakeConnection()), patch("builtins.print"):
            path = Path(folder)
            chosen = [data['cases'][i] for i in indices] if indices is not None else data['cases'][:chosen_count]
            report = await evaluate(data, chosen, model, path, ROOT)
            saved = json.loads((path / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["summary"], report["summary"])
            self.assertTrue((path / "report.html").is_file())
            self.assertTrue((path / "checkpoints.sqlite").is_file())
            logs = list((path / "logs").glob("*.jsonl"))
            self.assertTrue(logs)
            return report

    async def test_real_graph_with_mock_model_completes(self):
        report = await self.run_fake([response(tool="get_ticket", args={"ticket_id": "tk-001"}), response("测试工单记录")])
        self.assertEqual(report["summary"]["auto_check_passed"], 1)
        self.assertEqual(len(report["model_requests"]), 2)
        self.assertEqual(report["summary"]["manual_review"], "pending")

    async def test_failed_second_request_is_recorded_and_next_case_runs(self):
        report = await self.run_fake([response(tool="get_ticket", args={"ticket_id": "TK-001"}),
            APITimeoutError(request=MagicMock()), response("请提供工单编号")], chosen_count=2)
        self.assertEqual(report["summary"]["execution_errors"], 1)
        self.assertEqual(report["summary"]["auto_check_passed"], 1)
        self.assertEqual(len(report["model_requests"]), 3)
        self.assertTrue(report["summary"]["selected_batch_complete"])

    async def test_budget_stop_keeps_unrun_cases_visible(self):
        report = await self.run_fake([response(tool="get_ticket", args={"ticket_id": "TK-001"})], chosen_count=2, budget=1)
        self.assertEqual(report["summary"]["not_run"], ["E02"])
        self.assertEqual(report["summary"]["stopped_reason"], "request_budget_exhausted")
        self.assertEqual(len(report["model_requests"]), 1)

    async def test_planning_graph_exports_reviewable_draft(self):
        draft = {"ticket_id": "TK-002", "title": "测试草稿", "missing_information": ["软件版本"],
            "evidence": [{"chunk_id": "RULE-SOFT-001@1.0#001", "quote": "软件闪退时，先记录软件版本，再收集错误日志。"}],
            "tasks": [{"task_id": "TASK-001", "title": "采集信息", "priority": "P1", "mode": "read_only",
                "preconditions": ["有权限读取"], "steps": ["采集软件版本"],
                "expected_observations": ["记录实际版本或标记缺失"], "source_ids": ["RULE-SOFT-001@1.0#001"]}]}
        report = await self.run_fake([response(content=json.dumps(draft, ensure_ascii=False))], indices=[5])
        self.assertEqual(report['summary']['auto_check_passed'], 1)
        self.assertEqual(report['cases'][0]['plan']['execution_status'], 'not_run')
        self.assertEqual(len(report['model_requests']), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
