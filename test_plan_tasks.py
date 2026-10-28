import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from plan_tasks import create_test_plan, validate_plan


TICKET = {"ticket_id": "TK-002", "description": "软件闪退", "device": "PC-02", "processing_status": "处理中"}
MATCHES = [{"chunk_id": "RULE-SOFT-001@1.0#001", "text": "软件闪退时，先记录软件版本，再收集错误日志。"}]


def valid_plan():
    return {
        "ticket_id": "TK-002", "title": "软件闪退验证任务草稿", "missing_information": ["软件版本"],
        "evidence": [{"chunk_id": MATCHES[0]["chunk_id"], "quote": MATCHES[0]["text"]}],
        "tasks": [{
            "task_id": "TASK-001", "title": "采集软件版本和错误日志", "priority": "P1", "mode": "read_only",
            "preconditions": ["获得读取相关信息的权限"], "steps": ["记录软件版本与错误提示"],
            "expected_observations": ["形成可核对的版本与错误记录，缺失时明确标记"],
            "source_ids": [MATCHES[0]["chunk_id"]],
        }],
    }


class TestPlanTasks(unittest.IsolatedAsyncioTestCase):
    def parse(self, data):
        return validate_plan(json.dumps(data, ensure_ascii=False), TICKET, MATCHES)

    async def test_valid_plan_is_draft_and_not_run(self):
        result = self.parse(valid_plan())
        self.assertEqual(result["execution_status"], "not_run")
        self.assertTrue(result["requires_human_review"])

    async def test_wrong_ticket_is_rejected(self):
        data = valid_plan()
        data["ticket_id"] = "TK-001"
        with self.assertRaisesRegex(ValueError, "工单编号"):
            self.parse(data)

    async def test_fake_evidence_is_rejected(self):
        data = valid_plan()
        data["evidence"][0]["quote"] = "当前故障已经被确认是硬件损坏。"
        with self.assertRaisesRegex(ValueError, "原文"):
            self.parse(data)

    async def test_uncited_task_is_rejected(self):
        data = valid_plan()
        data["tasks"][0]["source_ids"] = ["UNKNOWN"]
        with self.assertRaisesRegex(ValueError, "引用证据"):
            self.parse(data)

    async def test_duplicate_tasks_or_empty_steps_are_rejected(self):
        data = valid_plan()
        data["tasks"].append(deepcopy(data["tasks"][0]))
        with self.assertRaisesRegex(ValueError, "重复"):
            self.parse(data)
        data = valid_plan()
        data["tasks"][0]["steps"] = []
        with self.assertRaises(ValueError):
            self.parse(data)

    async def test_plan_export_and_revision_change(self):
        def result(data):
            return SimpleNamespace(is_error=False, structured_content=data)
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=json.dumps(valid_plan(), ensure_ascii=False)),
        )])
        for revision, should_export in (("a" * 64, True), ("b" * 64, False)):
            with self.subTest(should_export=should_export), tempfile.TemporaryDirectory() as folder:
                client = MagicMock()
                client.call_tool = AsyncMock(side_effect=[
                    result({"status": "ok", "ticket": TICKET}),
                    result({"status": "ok", "matches": MATCHES, "knowledge_revision": "a" * 64}),
                    result({"status": "ok", "knowledge_revision": revision}),
                ])
                model = MagicMock()
                model.chat.completions.create.return_value = response
                with patch("builtins.print"):
                    if should_export:
                        exported = await create_test_plan("TK-002", client, model, folder)
                        self.assertTrue(exported["markdown_path"].exists())
                        self.assertIn("尚未执行", exported["markdown_path"].read_text(encoding="utf-8"))
                        self.assertEqual(json.loads(exported["json_path"].read_text(encoding="utf-8"))["known_facts"], TICKET)
                    else:
                        with self.assertRaisesRegex(ValueError, "资料发生变化"):
                            await create_test_plan("TK-002", client, model, folder)
                        self.assertEqual(list(Path(folder).iterdir()), [])
                self.assertEqual(model.chat.completions.create.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
