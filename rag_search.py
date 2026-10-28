"""独立语义检索工具：保留来源、版本与片段编号，尚未接入 Agent。"""

import hashlib
import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DOCUMENTS_PATH = Path(os.environ.get("FAULT_AGENT_KB_PATH", ROOT / "knowledge_base" / "documents.json"))
MODEL_CACHE = Path(os.environ.get("FAULT_AGENT_MODEL_CACHE", ROOT / ".models"))
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def load_documents(path):
    """验证资料结构，过滤过期版本，不自动推断资料是否仍然有效。"""
    documents = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(documents, list):
        raise ValueError("知识库必须是资料列表。")

    seen_versions = set()
    current_ids = set()
    active = []
    fields = ("id", "title", "version", "source", "updated_at", "type", "content")
    for doc in documents:
        if not isinstance(doc, dict):
            raise ValueError("每条资料必须是对象。")
        if any(not isinstance(doc.get(k), str) or not doc[k].strip() for k in fields):
            raise ValueError("资料缺少编号、标题、版本、来源、日期、类型或正文。")
        if type(doc.get("is_current")) is not bool:
            raise ValueError("is_current 必须明确设置为 true 或 false。")
        if doc["type"] not in ("规范", "模拟案例"):
            raise ValueError("资料类型必须为规范或模拟案例。")
        date.fromisoformat(doc["updated_at"])
        key = (doc["id"], doc["version"])
        if key in seen_versions:
            raise ValueError("出现重复的资料编号和版本。")
        seen_versions.add(key)
        if not doc["is_current"]:
            continue
        if doc["id"] in current_ids:
            raise ValueError("同一资料编号不能同时存在两个有效版本。")
        current_ids.add(doc["id"])
        active.append(doc)
    return active


def revision_of(documents):
    """内容摘要用于检测变化，不表示资料内容已经过专业验证。"""
    return hashlib.sha256(json.dumps(
        documents, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()


def get_knowledge_revision(documents_path=DOCUMENTS_PATH):
    documents = load_documents(documents_path)
    return {
        "status": "ok", "knowledge_revision": revision_of(documents),
        "active_document_count": len(documents),
    }


def chunk_documents(documents, chunk_size=220, overlap=40):
    """按字符切片，保留重叠文本以及可定位的来源信息。"""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("分片长度或重叠长度不合法。")
    chunks = []
    for doc in documents:
        for index, start in enumerate(range(0, len(doc["content"]), chunk_size - overlap), 1):
            end = min(start + chunk_size, len(doc["content"]))
            chunks.append({
                "chunk_id": f"{doc['id']}@{doc['version']}#{index:03d}",
                "document_id": doc["id"], "title": doc["title"],
                "version": doc["version"], "source": doc["source"],
                "updated_at": doc["updated_at"], "type": doc["type"],
                "start": start, "end": end,
                "text": doc["content"][start:end],
            })
            if end == len(doc["content"]):
                break
    return chunks


class KnowledgeSearcher:
    def __init__(self, documents_path=DOCUMENTS_PATH, model_cache=MODEL_CACHE, encoder=None):
        self.documents_path = Path(documents_path)
        self.model_cache = Path(model_cache)
        self.encoder = encoder
        self.fingerprint = None
        self.chunks = []
        self.vectors = None

    def get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(
                MODEL_NAME, device="cpu", cache_folder=str(self.model_cache),
                local_files_only=True,
            )
        return self.encoder

    def ensure_index(self):
        documents = load_documents(self.documents_path)
        fingerprint = revision_of(documents)
        if fingerprint == self.fingerprint:
            return

        chunks = chunk_documents(documents)
        vectors = None
        if chunks:
            texts = [f"{c['title']}\n{c['text']}" for c in chunks]
            vectors = np.asarray(self.get_encoder().encode(
                texts, normalize_embeddings=True, show_progress_bar=False,
            ), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or not np.isfinite(vectors).all():
                raise ValueError("资料向量的形状或数值异常。")
            lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(lengths == 0):
                raise ValueError("资料向量不能是零向量。")
            vectors = vectors / lengths

        self.chunks, self.vectors, self.fingerprint = chunks, vectors, fingerprint

    def search(self, query, top_k=3):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("查询必须是非空字符串。")
        query = query.strip()
        if len(query) > 1000:
            raise ValueError("查询不能超过1000个字符。")
        if type(top_k) is not int or not 1 <= top_k <= 5:
            raise ValueError("top_k 必须是1到5的整数。")

        self.ensure_index()
        if not self.chunks:
            return {"status": "no_documents", "query": query, "matches": [], "knowledge_revision": self.fingerprint}

        query_vector = np.asarray(self.get_encoder().encode(
            QUERY_PREFIX + query, normalize_embeddings=True, show_progress_bar=False,
        ), dtype=np.float32)
        if (
            query_vector.ndim != 1
            or query_vector.shape[0] != self.vectors.shape[1]
            or not np.isfinite(query_vector).all()
            or np.linalg.norm(query_vector) == 0
        ):
            raise ValueError("查询向量的形状或数值异常。")
        query_vector = query_vector / np.linalg.norm(query_vector)
        scores = self.vectors @ query_vector
        order = np.argsort(-scores, kind="stable")[:top_k]
        return {
            "status": "ok", "query": query, "knowledge_revision": self.fingerprint,
            "matches": [{**self.chunks[int(i)], "score": float(scores[i])} for i in order],
        }


@lru_cache(maxsize=1)
def get_searcher():
    return KnowledgeSearcher()


def search_knowledge(query, top_k=3):
    return get_searcher().search(query, top_k)


if __name__ == "__main__":
    text = input("请输入要检索的故障描述：")
    print("首次检索需要加载本地模型并生成资料向量，请稍候……")
    result = search_knowledge(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
