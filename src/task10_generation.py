"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# Model OpenRouter (":free" — không tính phí, giới hạn 50 req/ngày + 20 req/phút).
# Đổi qua OPENROUTER_MODEL trong .env nếu slug hết hiệu lực (models ":free" đổi thường xuyên,
# kiểm tra tại https://openrouter.ai/models?max_price=0)
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4-31b-it")

# Model fallback khi OpenRouter bị 429 (hết quota free tier)
OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")  # Gemini fallback nếu OpenRouter + OpenAI đều lỗi


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Tuition Fees, 2026]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    front = chunks[::2]   # index 0, 2, 4 -> đặt ở đầu
    back = chunks[1::2]   # index 1, 3    -> đặt ở cuối (reversed)
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        source = Path(str(source)).stem if isinstance(source, str) else f"Source {i}"
        year = chunk.get("metadata", {}).get("year") or chunk.get("metadata", {}).get("crawled_at", "2026")
        if isinstance(year, str) and len(year) >= 4:
            year = year[:4]
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


def _citation_from_chunk(chunk: dict, fallback_index: int) -> str:
    source = chunk.get("metadata", {}).get("source", f"Source {fallback_index}")
    source = Path(str(source)).stem
    year = chunk.get("metadata", {}).get("year")
    if not year:
        crawled = chunk.get("metadata", {}).get("crawled_at")
        year = str(crawled)[:4] if crawled else "2026"
    return f"[{source}, {year}]"


def _clean_evidence_line(line: str) -> str:
    line = " ".join(line.replace("|", " ").split())
    if not line:
        return ""
    noisy_prefixes = (
        "đại học quốc gia",
        "trường đại học",
        "cộng hòa xã hội",
        "độc lập",
        "số:",
        "source:",
        "document",
    )
    lowered = line.lower()
    if len(line) < 35 or lowered.startswith(noisy_prefixes):
        return ""
    letters = [char for char in line if char.isalpha()]
    if letters and sum(char.isupper() for char in letters) / len(letters) > 0.75:
        return ""
    if lowered.startswith(("chương ", "điều ")):
        return ""
    return line[:260].rstrip(" ,;:-")


def _select_evidence(query: str, chunks: list[dict], max_items: int = 3) -> list[tuple[str, str]]:
    query_terms = {token.lower() for token in query.split() if len(token) >= 3}
    evidence: list[tuple[int, str, str]] = []

    for idx, chunk in enumerate(chunks, 1):
        citation = _citation_from_chunk(chunk, idx)
        for raw_line in chunk.get("content", "").splitlines():
            line = _clean_evidence_line(raw_line)
            if not line:
                continue
            line_terms = set(line.lower().split())
            overlap = len(query_terms & line_terms)
            keyword_bonus = sum(
                2 for keyword in ("học phí", "học bổng", "ký túc", "thư viện", "đăng ký", "sinh viên")
                if keyword in line.lower() and keyword in query.lower()
            )
            score = overlap + keyword_bonus
            evidence.append((score, line, citation))

    evidence.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[str, str]] = []
    seen_lines: set[str] = set()
    for _, line, citation in evidence:
        signature = line[:90].lower()
        if signature in seen_lines:
            continue
        selected.append((line, citation))
        seen_lines.add(signature)
        if len(selected) >= max_items:
            break
    return selected


def _fallback_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có"

    evidence = _select_evidence(query, chunks)
    if not evidence:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có"

    bullets = [f"- {line} {citation}" for line, citation in evidence]
    return (
        "Dựa trên các tài liệu truy xuất được, câu trả lời như sau:\n\n"
        + "\n".join(bullets)
        + "\n\nMình chỉ kết luận theo các nguồn đã truy xuất ở trên."
    )


# =============================================================================
# LLM CALL — fallback OpenRouter -> OpenAI -> Gemini
# =============================================================================

def _iter_providers():
    """Yield (provider_name, OpenAI-client kwargs, model) theo thứ tự ưu tiên."""
    if os.getenv("OPENROUTER_API_KEY"):
        yield (
            "openrouter",
            {"api_key": os.getenv("OPENROUTER_API_KEY"), "base_url": "https://openrouter.ai/api/v1"},
            LLM_MODEL,
        )
    if os.getenv("OPENAI_API_KEY"):
        yield ("openai", {"api_key": os.getenv("OPENAI_API_KEY")}, OPENAI_FALLBACK_MODEL)
    if os.getenv("GEMINI_API_KEY"):
        yield (
            "gemini",
            {
                "api_key": os.getenv("GEMINI_API_KEY"),
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            },
            GEMINI_FALLBACK_MODEL,
        )


def _call_llm(messages: list[dict]) -> str:
    """
    Gọi LLM theo thứ tự fallback: OpenRouter -> OpenAI -> Gemini.

    OpenRouter free tier (":free") giới hạn 50 request/ngày — nếu gặp lỗi
    (vd. 429 rate limit), tự động chuyển sang provider kế tiếp có API key
    trong .env, thay vì crash toàn bộ pipeline.
    """
    from openai import OpenAI, APIError

    providers = list(_iter_providers())
    if not providers:
        raise RuntimeError(
            "Chưa có API key nào — set OPENROUTER_API_KEY, OPENAI_API_KEY "
            "hoặc GEMINI_API_KEY trong file .env"
        )

    last_error: Exception | None = None
    for name, client_kwargs, model in providers:
        try:
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            return response.choices[0].message.content
        except APIError as e:
            print(f"  ⚠ Provider '{name}' lỗi ({e}), thử provider tiếp theo...")
            last_error = e
            continue

    raise RuntimeError(f"Tất cả LLM provider đều lỗi. Lỗi cuối: {last_error}")


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder để tránh lost in the middle
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context với source labels
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Step 5: Call LLM (fallback OpenRouter -> OpenAI -> Gemini)
    try:
        answer = _call_llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ])
    except Exception:
        answer = _fallback_answer(query, reordered)

    if "[" not in answer or "]" not in answer:
        answer = _fallback_answer(query, reordered)

    # Step 6: Return
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    test_queries = [
        "Học phí tại UET (Đại học Công nghệ - ĐHQGHN) là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên UET có những học bổng nào?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
