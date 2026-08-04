"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

# Corpus được nạp lười (lazy) từ pipeline Task 4 (data/standardized/ đã chunk).
# Có thể gán trực tiếp (vd. trong test) để override việc load từ Task 4.
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

# Cache BM25 index để không phải rebuild mỗi lần gọi lexical_search()
_bm25_index = None
_indexed_corpus: list[dict] | None = None


def _load_corpus() -> list[dict]:
    """Lấy corpus: dùng CORPUS nếu đã set thủ công, nếu không thì load qua Task 4."""
    if CORPUS:
        return CORPUS

    from src.task4_chunking_indexing import load_documents, chunk_documents

    documents = load_documents()
    return chunk_documents(documents)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    if not corpus:
        raise ValueError("Corpus rỗng, không thể xây dựng BM25 index")

    # Tokenize đơn giản bằng khoảng trắng (whitespace split)
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global _bm25_index, _indexed_corpus

    corpus = _load_corpus()

    # Rebuild index chỉ khi corpus thay đổi (hoặc chưa từng build)
    if _bm25_index is None or _indexed_corpus is not corpus:
        _bm25_index = build_bm25_index(corpus)
        _indexed_corpus = corpus

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)

    if max(scores) <= 0:
        query_terms = [term for term in tokenized_query if term]
        fallback_scores = []
        for doc in corpus:
            content = doc["content"].lower()
            score = 0.0
            for term in query_terms:
                if term in content:
                    score += 1.0
                else:
                    shared = set(term) & set(content)
                    score += len(shared) / max(len(set(term)), 1)
            fallback_scores.append(score)
        scores = fallback_scores

    ranked = sorted(
        ((score, doc) for score, doc in zip(scores, corpus)),
        key=lambda pair: pair[0],
        reverse=True,
    )

    return [
        {
            "content": doc["content"],
            "score": float(score),
            "metadata": doc.get("metadata", {}),
        }
        for score, doc in ranked[:top_k]
    ]


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
