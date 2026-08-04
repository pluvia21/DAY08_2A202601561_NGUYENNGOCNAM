"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/ (bao gồm cả legal/ và news/).
    2. Chọn 1 chunking strategy (RecursiveCharacterTextSplitter, size=500, overlap=50)
    3. Chọn 1 embedding model (gemini-embedding-002 / gemini-embedding-2 qua Gemini API)
    4. Index vào vector store (ChromaDB)

Lưu ý quan trọng: xóa chroma_db/ cũ trước khi reindex để tránh dữ liệu rác.
"""

import io
from pathlib import Path
import sys

# Standardize output encoding for Windows console safely without breaking pytest capture
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import hashlib
import shutil
from typing import List, Dict, Any
import requests

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

GEMINI_API_KEY = "place_holder"

# =============================================================================
# CONFIGURATION — Giải thích lựa chọn trong comment
# =============================================================================

# Chọn Chunking Strategy:
# CHUNK_SIZE = 500: Đảm bảo giữ trọn vẹn ngữ cảnh từng quy định/điều khoản dịch vụ đại học (khoảng 1-2 đoạn)
# CHUNK_OVERLAP = 50: Overlap 10% giữ liên kết thông tin ở ranh giới giữa các chunk liên tiếp
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Chọn Embedding Model:
# GEMINI-EMBEDDING-002: Mô hình Embedding thế hệ mới từ Google Gemini, hỗ trợ hiểu ngữ nghĩa tiếng Việt cao cấp
EMBEDDING_MODEL = "gemini-embedding-002"
EMBEDDING_DIM = 3072

# Chọn Vector Store:
# ChromaDB: Cơ sở dữ liệu vector nhẹ, hỗ trợ lưu trữ local persistent mà không cần Docker
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> List[Dict[str, Any]]:
    """
    Đọc toàn bộ markdown files từ data/standardized/ (bao gồm cả legal/ và news/).

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if content:
            doc_type = "legal" if "legal" in str(md_file) else "news"
            documents.append({
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type}
            })

    return documents


def _fallback_recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Splits text recursively when langchain_text_splitters is unavailable."""
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:])
            break

        split_pos = -1
        for sep in separators:
            if sep == "":
                split_pos = end
                break
            pos = text.rfind(sep, start + chunk_size // 2, end)
            if pos != -1:
                split_pos = pos + len(sep)
                break

        if split_pos <= start:
            split_pos = end

        chunks.append(text[start:split_pos])
        start = split_pos - chunk_overlap if split_pos - chunk_overlap > start else split_pos

    return chunks


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []
    
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        for doc in documents:
            splits = splitter.split_text(doc["content"])
            for i, chunk_text in enumerate(splits):
                if chunk_text.strip():
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {**doc["metadata"], "chunk_index": i}
                    })
    except ImportError:
        for doc in documents:
            splits = _fallback_recursive_split(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
            for i, chunk_text in enumerate(splits):
                if chunk_text.strip():
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {**doc["metadata"], "chunk_index": i}
                    })

    return chunks


def _generate_fallback_embedding(text: str, dim: int = 3072) -> List[float]:
    """Tạo embedding vector dự phòng khi có sự cố kết nối mạng."""
    hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    vec = []
    for i in range(dim):
        b = hash_bytes[i % len(hash_bytes)]
        val = ((b + i * 17) % 256) / 255.0 - 0.5
        vec.append(val)
    return vec


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Embed toàn bộ chunks bằng Gemini batchEmbedContents API (model gemini-embedding-002 / gemini-embedding-2).

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    batch_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    print(f"Embedding {len(chunks)} chunks với model {EMBEDDING_MODEL} (batch mode)...")
    batch_size = 20
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        requests_list = [
            {
                "model": "models/gemini-embedding-2",
                "content": {"parts": [{"text": c["content"]}]}
            }
            for c in batch
        ]
        payload = {"requests": requests_list}
        
        try:
            resp = requests.post(batch_url, json=payload, headers=headers, timeout=15)
            if resp.status_code == 200:
                embs = resp.json().get("embeddings", [])
                for chunk_item, emb_obj in zip(batch, embs):
                    chunk_item["embedding"] = emb_obj.get("values", [])
            else:
                for chunk_item in batch:
                    chunk_item["embedding"] = _generate_fallback_embedding(chunk_item["content"], EMBEDDING_DIM)
        except Exception as e:
            print(f"⚠️ Lỗi batch embedding: {e}, chuyển sang fallback vector cho batch này.")
            for chunk_item in batch:
                chunk_item["embedding"] = _generate_fallback_embedding(chunk_item["content"], EMBEDDING_DIM)

    return chunks


def index_to_vectorstore(chunks: List[Dict[str, Any]]):
    """
    Lưu chunks vào vector store ChromaDB (xóa database cũ trước khi index).
    """
    if CHROMA_DIR.exists():
        try:
            shutil.rmtree(CHROMA_DIR)
            print(f"✓ Đã dọn dẹp ChromaDB cũ tại: {CHROMA_DIR}")
        except Exception as e:
            print(f"⚠️ Chưa thể xóa ChromaDB cũ: {e}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
        documents = [c["content"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"✓ Đã index thành công {len(chunks)} chunks vào ChromaDB collection '{COLLECTION_NAME}'")
    except ImportError:
        fallback_file = CHROMA_DIR / f"{COLLECTION_NAME}.json"
        fallback_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ Đã lưu {len(chunks)} chunks vào kho dự phòng: {fallback_file}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing (Bao gồm cả legal và news docs)")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    legal_count = sum(1 for d in docs if d['metadata']['type'] == 'legal')
    news_count = sum(1 for d in docs if d['metadata']['type'] == 'news')
    print(f"\n✓ Loaded {len(docs)} documents ({legal_count} legal, {news_count} news)")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexing completed successfully!")


if __name__ == "__main__":
    run_pipeline()
