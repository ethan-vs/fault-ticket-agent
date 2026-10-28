import json
import unittest
from copy import deepcopy
from unittest.mock import MagicMock

from rag_answer import INSUFFICIENT_ANSWER, parse_review, request_review


MATCHES = [
    {"chunk_id": "RULE-SOFT-001@1.0#001", "text": "软件启动后闪退时，应收集软件版本、崩溃日志和复现步骤。"},
    {"chunk_id": "CASE-NET-001@1.0#001", "text": "这是无线认证失败的虚构案例，不证明当前故障具有相同根因。"},
]


def valid_review():
    return {
        "status": "answered",
        "selected_sources": [{
            "chunk_id": MATCHES[0]["chunk_id"],
            "evidence": "应收集软件版本、崩溃日志和复现步骤。",
            "reason": "对应软件闪退现象。",
        }],
        "excluded_sources": [{
            "chunk_id": MATCHES[1]["chunk_id"],
            "reason": "当前网络正常，不应套用无线认证故障。",
        }],
        "answer": "请收集软件版本、崩溃日志和复现步骤。[RULE-SOFT-001@1.0#001]",
    }


class TestRAGAnswer(unittest.TestCase):
    def parse(self, review):
        return parse_review(json.dumps(review, ensure_ascii=False), MATCHES)

    def test_valid_sources_and_quotes(self):
        review = self.parse(valid_review())
        self.assertEqual(review["status"], "answered")

    def test_fake_quote_is_rejected(self):
        review = valid_review()
        review["selected_sources"][0]["evidence"] = "已经确认故障来自接入点硬件损坏。"
        with self.assertRaisesRegex(ValueError, "原文"):
            self.parse(review)

    def test_unknown_citation_is_rejected(self):
        review = valid_review()
        review["answer"] += "[RULE-UNKNOWN@1.0#001]"
        with self.assertRaisesRegex(ValueError, "引用"):
            self.parse(review)

    def test_omitted_candidate_is_rejected(self):
        review = valid_review()
        review["excluded_sources"] = []
        with self.assertRaisesRegex(ValueError, "全部候选"):
            self.parse(review)

    def test_duplicate_selection_is_rejected(self):
        review = valid_review()
        review["excluded_sources"].append({
            "chunk_id": MATCHES[0]["chunk_id"], "reason": "重复条目",
        })
        with self.assertRaisesRegex(ValueError, "重复"):
            self.parse(review)

    def test_insufficient_evidence_uses_fixed_message(self):
        review = valid_review()
        review["status"] = "insufficient_evidence"
        review["selected_sources"] = []
        review["excluded_sources"].append({
            "chunk_id": MATCHES[0]["chunk_id"], "reason": "本测试假设没有适用依据",
        })
        review["answer"] = "不应展示的无依据结论。"
        self.assertEqual(self.parse(review)["answer"], INSUFFICIENT_ANSWER)

    def test_empty_candidates_do_not_call_api(self):
        client = MagicMock()
        raw_output = request_review(client, "没有资料", [])
        self.assertEqual(parse_review(raw_output, [])["status"], "insufficient_evidence")
        client.chat.completions.create.assert_not_called()

    def test_invalid_json_or_schema_is_rejected(self):
        for raw_output in ("", "not-json", "[]", '{"answer":"缺字段"}'):
            with self.subTest(raw_output=raw_output):
                with self.assertRaises(ValueError):
                    parse_review(raw_output, deepcopy(MATCHES))


if __name__ == "__main__":
    unittest.main(verbosity=2)
