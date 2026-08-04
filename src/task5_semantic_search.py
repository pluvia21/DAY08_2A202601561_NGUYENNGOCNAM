"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
    - Tích hợp HyDE hypothetical document
"""

import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Sử dụng các hằng số từ Task 4
try:
    from src.task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
except ImportError:
    from pathlib import Path
    CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
    COLLECTION_NAME = "university_services_docs"
    EMBEDDING_MODEL = "BAAI/bge-m3"

_embedding_model = None

def get_embedding_model():
    """Tải và trả về embedding model (Singleton pattern để tối ưu hóa)"""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_collection():
    """Khởi tạo và trả về ChromaDB collection"""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def generate_hyde_document(query: str) -> str:
    """
    Sử dụng LLM để sinh ra một tài liệu giả định (hypothetical document) 
    chứa câu trả lời tiềm năng cho câu hỏi. Điều này giúp cải thiện semantic search.
    """
    try:
        from openai import OpenAI
        
        # Hỗ trợ OpenAI hoặc OpenRouter
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[WARNING] Không tìm thấy API Key (OPENROUTER_API_KEY hoặc OPENAI_API_KEY). Bỏ qua HyDE.")
            return query
            
        base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        system_prompt = (
            "Bạn là trợ lý ảo hỗ trợ thông tin trường Đại học Công nghệ (UET). "
            "Hãy viết một đoạn văn ngắn (khoảng 2-3 câu) cung cấp câu trả lời tiềm năng "
            "và các từ khóa liên quan trực tiếp đến câu hỏi của người dùng. Không cần giải thích thêm."
        )
        
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[WARNING] Lỗi khi tạo HyDE document: {e}. Sử dụng query gốc.")
        return query


def semantic_search(query: str, top_k: int = 10, use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (và HyDE nếu bật).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        use_hyde: Cờ kích hoạt Hypothetical Document Embeddings

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    search_query = query
    
    # Bước 0: Tạo Hypothetical Document nếu dùng HyDE
    if use_hyde:
        print(f"[*] Đang tạo Hypothetical Document (HyDE) cho query: '{query}'...")
        hypothetical_doc = generate_hyde_document(query)
        # Nối query gốc và hypothetical document để có vector biểu diễn tốt nhất
        search_query = f"{query}\n{hypothetical_doc}"

    # Bước 1: Embed query bằng model ở Task 4
    model = get_embedding_model()
    query_vector = model.encode(search_query).tolist()

    # Bước 2: Query vector store (cosine similarity)
    collection = get_collection()
    
    if collection.count() == 0:
        print("[WARNING] Collection hiện tại đang trống. Vui lòng chạy Task 4 để index dữ liệu.")
        return []

    # Giới hạn top_k không vượt quá số document có trong DB
    k = min(top_k, collection.count())
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    output = []
    
    if not results["documents"] or not results["documents"][0]:
        return output

    # Bước 3: Tính toán score và định dạng kết quả
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # Lưu ý: ChromaDB trả về distance. Score = max(0.0, 1.0 - distance).
        score = max(0.0, 1.0 - dist)
        output.append({
            "content": doc,
            "score": round(score, 4),
            "metadata": meta or {}
        })

    # Đảm bảo sắp xếp giảm dần theo điểm số
    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Test (Yêu cầu phải chạy Task 4 để có data trong ChromaDB trước)
    test_queries = [
        "Lịch nghỉ Tết Nguyên Đán là khi nào?",
        "Làm sao để xin học bổng Vallet?"
    ]
    
    print("=== TEST SEMANTIC SEARCH (Tiêu chuẩn) ===")
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = semantic_search(q, top_k=3, use_hyde=False)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['content'][:80]}...")
            
    print("\n=== TEST SEMANTIC SEARCH (HyDE) ===")
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = semantic_search(q, top_k=3, use_hyde=True)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['content'][:80]}...")
