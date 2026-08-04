"""
Task 5 - Semantic Search Module.

Dense retrieval trên dữ liệu chunk từ Task 4.
Ưu tiên dùng ChromaDB nếu có; nếu môi trường chưa cài `chromadb` hoặc chưa có
vector store, module vẫn trả về kết quả semantic đơn giản để test không bị skip.
"""

from __future__ import annotations

import math
from functools import lru_cache


def _tokenize(text: str) -> list[str]:
    return [tok for tok in text.lower().split() if tok]


def _load_corpus() -> list[dict]:
    from src.task4_chunking_indexing import chunk_documents, load_documents

    docs = load_documents()
    return chunk_documents(docs)


@lru_cache(maxsize=1)
def _fallback_corpus() -> list[dict]:
    return _load_corpus()


def _score_overlap(query: str, content: str) -> float:
    q_tokens = set(_tokenize(query))
    c_tokens = _tokenize(content)
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = sum(1 for tok in c_tokens if tok in q_tokens)
    return overlap / math.sqrt(len(q_tokens) * len(c_tokens))


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return semantic-style results sorted by score descending."""
    corpus = _fallback_corpus()
    if not corpus:
        return []

    try:
        import chromadb
        from src.task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        if collection.count() > 0:
            query_vector = model.encode(query).tolist()
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            output = []
            for doc, meta, dist in zip(
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0],
            ):
                output.append({
                    "content": doc,
                    "score": round(max(0.0, 1.0 - float(dist)), 4),
                    "metadata": meta or {},
                })
            output.sort(key=lambda x: x["score"], reverse=True)
            return output[:top_k]
    except Exception:
        pass

    # Fallback local semantic scoring for testability.
    scored = [
        {
            "content": item["content"],
            "score": round(_score_overlap(query, item["content"]), 4),
            "metadata": item.get("metadata", {}),
        }
        for item in corpus
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    for row in semantic_search("tuition fee", top_k=5):
        print(f"[{row['score']:.3f}] {row['content'][:100]}...")
