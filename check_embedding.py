from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_CACHE = Path(__file__).resolve().parent / ".models"

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def main():
    print("正在加载向量模型，首次运行需要联网下载……")

    model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5",
        device="cpu",
        cache_folder=str(MODEL_CACHE),
    )

    # 三段项目自编的模拟资料
    documents = [
        {
            "id": "DOC-NET-001",
            "text": (
                "无线网络连接异常时，先确认影响范围，"
                "并对比有线连接是否正常，"
                "再收集认证及地址分配提示。"
            ),
        },
        {
            "id": "DOC-SOFT-001",
            "text": (
                "办公软件出现闪退，即使 Wi-Fi 正常，"
                "也应收集软件版本、崩溃日志和复现步骤。"
            ),
        },
        {
            "id": "DOC-ACCOUNT-001",
            "text": (
                "邮箱密码遗忘时，按身份核验和密码重置流程处理。"
            ),
        },
    ]

    query = "多台电脑无线连不上，但插网线可以正常上网。"

    # 将资料分别转换成向量
    document_vectors = model.encode(
        [doc["text"] for doc in documents],
        normalize_embeddings=True,
    )

    # 将查询转换成向量
    query_vector = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    )

    # 已归一化的向量，点积等于余弦相似度
    scores = query_vector @ document_vectors.T

    ranked_results = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    print("\n查询：", query)
    print("查询向量形状：", query_vector.shape)
    print("\n资料相关程度排序：")

    for doc, score in ranked_results:
        print(f"\n{doc['id']}，相似度：{float(score):.4f}")
        print(doc["text"])


if __name__ == "__main__":
    main()