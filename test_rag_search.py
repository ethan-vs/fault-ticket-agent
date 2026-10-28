import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag_search import KnowledgeSearcher, chunk_documents, load_documents


def document(doc_id="DOC-001", version="1.0", is_current=True, content="示例规范正文"):
    return {
        "id": doc_id, "title": "示例标题", "version": version,
        "source": "项目自编模拟资料", "updated_at": "2026-10-28",
        "is_current": is_current, "type": "规范", "content": content,
    }


class FixedEncoder:
    """固定向量只用于验证排序与缓存逻辑，不验证真实语义效果。"""
    def __init__(self):
        self.document_calls = 0

    def encode(self, texts, **kwargs):
        if isinstance(texts, list):
            self.document_calls += 1
            return np.array([[1.0, 0.0] if i == 0 else [0.0, 1.0] for i in range(len(texts))])
        return np.array([1.0, 0.0])


class TestRAGSearch(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "documents.json"

    def save(self, docs):
        self.path.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")

    def test_outdated_documents_are_excluded(self):
        self.save([document(version="0.9", is_current=False), document()])
        searcher = KnowledgeSearcher(self.path, encoder=FixedEncoder())
        matches = searcher.search("查询资料")["matches"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["version"], "1.0")

    def test_chunk_text_and_citation_match_source(self):
        doc = document(content="甲乙丙丁戊己庚辛壬癸")
        chunks = chunk_documents([doc], chunk_size=6, overlap=2)
        self.assertEqual([c["text"] for c in chunks], ["甲乙丙丁戊己", "戊己庚辛壬癸"])
        for chunk in chunks:
            self.assertEqual(chunk["text"], doc["content"][chunk["start"]:chunk["end"]])
            self.assertEqual(chunk["document_id"], doc["id"])
            self.assertEqual(chunk["source"], doc["source"])

    def test_source_update_rebuilds_vectors(self):
        self.save([document()])
        encoder = FixedEncoder()
        searcher = KnowledgeSearcher(self.path, encoder=encoder)
        searcher.search("首次查询")
        searcher.search("第二次查询")
        self.assertEqual(encoder.document_calls, 1)
        self.save([document(version="2.0", content="更新后的规范正文")])
        matches = searcher.search("更新后查询")["matches"]
        self.assertEqual(encoder.document_calls, 2)
        self.assertEqual(matches[0]["version"], "2.0")
        self.assertEqual(matches[0]["text"], "更新后的规范正文")

    def test_invalid_queries_and_top_k(self):
        searcher = KnowledgeSearcher(self.path, encoder=FixedEncoder())
        for query in ("", "   ", None, 123, "字" * 1001):
            with self.subTest(query_type=type(query).__name__):
                with self.assertRaises(ValueError):
                    searcher.search(query)
        for top_k in (0, -1, 6, True, "3"):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    searcher.search("有效查询", top_k)

    def test_empty_knowledge_base(self):
        self.save([])
        encoder = FixedEncoder()
        result = KnowledgeSearcher(self.path, encoder=encoder).search("查询资料")
        self.assertEqual(result["status"], "no_documents")
        self.assertEqual(result["matches"], [])
        self.assertEqual(encoder.document_calls, 0)

    def test_conflicting_current_versions_are_rejected(self):
        self.save([document(), document(version="2.0")])
        with self.assertRaises(ValueError):
            load_documents(self.path)

    def test_missing_source_is_rejected(self):
        doc = document()
        del doc["source"]
        self.save([doc])
        with self.assertRaises(ValueError):
            load_documents(self.path)

    def test_top_k_limits_results_and_preserves_order(self):
        self.save([document("DOC-A"), document("DOC-B")])
        result = KnowledgeSearcher(self.path, encoder=FixedEncoder()).search("查询", top_k=1)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["document_id"], "DOC-A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
