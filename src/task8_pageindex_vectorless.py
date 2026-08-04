"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Cache doc_id để không phải upload lại document mỗi lần chạy
DOC_ID_CACHE_PATH = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 60


def _load_doc_id_cache() -> dict:
    if DOC_ID_CACHE_PATH.exists():
        return json.loads(DOC_ID_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_doc_id_cache(cache: dict) -> None:
    DOC_ID_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_ID_CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def upload_documents() -> dict:
    """
    Upload toàn bộ PDF trong data/landing/legal/ lên PageIndex (PageIndex chỉ
    nhận PDF trực tiếp). doc_id trả về được cache vào JSON file
    (data/pageindex_doc_ids.json) để lần chạy sau không phải upload lại.

    Returns:
        dict: {filename: doc_id}
    """
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache = _load_doc_id_cache()

    pdf_files = sorted(LEGAL_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"  ⚠ Không tìm thấy file PDF nào trong {LEGAL_DIR}")

    for pdf_path in pdf_files:
        if pdf_path.name in cache:
            print(f"  = Đã cache: {pdf_path.name} -> {cache[pdf_path.name]}")
            continue
        resp = client.submit_document(str(pdf_path))
        doc_id = resp.get("doc_id") or resp.get("id")
        cache[pdf_path.name] = doc_id
        print(f"  ✓ Uploaded: {pdf_path.name} -> {doc_id}")

    _save_doc_id_cache(cache)
    return cache


def _wait_for_retrieval(client: PageIndexClient, retrieval_id: str) -> dict:
    """Poll get_retrieval() cho tới khi có kết quả hoặc vượt timeout."""
    start = time.monotonic()
    while True:
        retrieval = client.get_retrieval(retrieval_id)
        status = retrieval.get("status")
        if "retrieved_nodes" in retrieval or status in ("completed", "ready", "done"):
            if retrieval.get("deprecation"):
                print(f"  ⚠ PageIndex API deprecation notice: {retrieval['deprecation']}")
            return retrieval

        if time.monotonic() - start > POLL_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"PageIndex retrieval {retrieval_id} không hoàn thành sau "
                f"{POLL_TIMEOUT_SECONDS}s (status={status})"
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Chưa set PAGEINDEX_API_KEY trong file .env")

    cache = _load_doc_id_cache()
    if not cache:
        raise RuntimeError(
            "Chưa có doc_id nào được cache — chạy upload_documents() trước"
        )

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    raw_items = []  # (content, metadata) theo đúng thứ tự relevance PageIndex trả về
    for filename, doc_id in cache.items():
        submit_resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = submit_resp.get("retrieval_id") or submit_resp.get("id")

        retrieval = _wait_for_retrieval(client, retrieval_id)

        for node in retrieval.get("retrieved_nodes", []):
            for group in node.get("relevant_contents", []):
                for item in group:
                    raw_items.append((
                        item.get("relevant_content", ""),
                        {
                            "source": filename,
                            "doc_id": doc_id,
                            "section": item.get("section_title"),
                        },
                    ))

    # PageIndex không trả score trực tiếp — tự gán theo rank (reciprocal rank)
    results = [
        {
            "content": content,
            "score": round(1.0 / (rank + 1), 4),
            "metadata": metadata,
            "source": "pageindex",
        }
        for rank, (content, metadata) in enumerate(raw_items)
    ]

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
