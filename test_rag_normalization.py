import json
import unittest
from copy import deepcopy

from rag_answer import parse_review


CHUNK = {
    "chunk_id": "RULE-SOFT-001@1.0#001",
    "text": "软件闪退时先收集版本和错误日志。对比其他设备是否也出现相同问题。",
}
QUOTE_A = "软件闪退时先收集版本和错误日志。"
QUOTE_B = "对比其他设备是否也出现相同问题。"


def selected(evidence, reason="用于软件故障排查"):
    return {"chunk_id": CHUNK["chunk_id"], "evidence": evidence, "reason": reason}


def review(items):
    return {
        "status": "answered", "selected_sources": items,
        "excluded_sources": [],
        "answer": "先收集信息并对比其他设备。[RULE-SOFT-001@1.0#001]",
    }


class TestRAGNormalization(unittest.TestCase):
    def parse(self, data):
        return parse_review(json.dumps(data, ensure_ascii=False), [CHUNK])

    def test_repeated_selected_id_merges_valid_quotes(self):
        result = self.parse(review([selected(QUOTE_A), selected(QUOTE_B, "用于确认影响范围")]))
        self.assertEqual(len(result["selected_sources"]), 1)
        self.assertEqual(result["selected_sources"][0]["evidence"], [QUOTE_A, QUOTE_B])

    def test_list_of_quotes_is_supported(self):
        result = self.parse(review([selected([QUOTE_A, QUOTE_B])]))
        self.assertEqual(result["selected_sources"][0]["evidence"], [QUOTE_A, QUOTE_B])

    def test_old_string_remains_supported(self):
        result = self.parse(review([selected(QUOTE_A)]))
        self.assertEqual(result["selected_sources"][0]["evidence"], [QUOTE_A])

    def test_conflicting_selection_is_rejected(self):
        data = review([selected(QUOTE_A)])
        data["excluded_sources"] = [{"chunk_id": CHUNK["chunk_id"], "reason": "冲突条目"}]
        with self.assertRaisesRegex(ValueError, "同时采用和排除"):
            self.parse(data)

    def test_fake_second_quote_is_not_hidden_by_merge(self):
        data = review([selected(QUOTE_A), selected("已经确认当前问题是硬件损坏。")])
        with self.assertRaisesRegex(ValueError, "原文"):
            self.parse(data)

    def test_duplicate_identical_quote_is_deduplicated(self):
        result = self.parse(review([selected([QUOTE_A, QUOTE_A]), selected(QUOTE_A)]))
        self.assertEqual(result["selected_sources"][0]["evidence"], [QUOTE_A])


if __name__ == "__main__":
    unittest.main(verbosity=2)
