"""
RAG Chatbot - University Services (Starter Template)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.task10_generation import generate_with_citation

load_dotenv()

st.set_page_config(
    page_title="University Services RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: "Segoe UI", "Arial", "Helvetica", sans-serif;
    }
    .stButton > button {
        white-space: normal;
        min-height: 3.25rem;
        line-height: 1.25;
    }
    [data-testid="stSidebar"] .stButton > button {
        min-height: 2.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SUGGESTIONS = [
    "Học phí tại UET là bao nhiêu?",
    "Làm sao để đặt phòng học nhóm ở thư viện?",
    "Điều kiện xét học bổng khuyến khích học tập là gì?",
    "Dịch vụ hỗ trợ chỗ ở cho sinh viên như thế nào?",
    "Cách đăng ký học phần qua cổng thông tin sinh viên UET?",
]


def source_citation(source: dict) -> str:
    meta = source.get("metadata", {})
    source_name = Path(str(meta.get("source", "Unknown"))).stem
    year = str(meta.get("year") or meta.get("crawled_at") or meta.get("date_crawled") or "2026")[:4]
    return f"[{source_name}, {year}]"


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            doc_type = meta.get("type", "unknown")
            score = src.get("score", 0)
            citation = source_citation(src)
            st.markdown(f"**[{i}] {citation}** `{doc_type}` | score: `{score:.4f}`")
            st.text(src.get("content", "")[:300] + "...")
            st.divider()


with st.sidebar:
    st.title("🎓 University Services RAG")
    st.caption(
        "Trợ lý hỏi đáp về dịch vụ và chính sách đại học: học phí, học bổng, "
        "ký túc xá và thư viện."
    )

    st.divider()
    st.subheader("💡 Câu hỏi gợi ý")
    for idx, suggestion in enumerate(SUGGESTIONS):
        if st.button(suggestion, use_container_width=True, key=f"sidebar_suggestion_{idx}"):
            st.session_state["pending_query"] = suggestion

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)

    st.divider()
    st.subheader("Supervisor")
    st.caption("Điều phối truy vấn qua Hybrid Retrieval, Rerank, PageIndex fallback và Task 10 Generation.")
    st.caption("Semantic Search + BM25 → RRF Rerank → PageIndex → LLM Generation có citation")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

st.title("🎓 University Services RAG Chatbot")
st.caption("Hệ thống hỏi đáp thông tin dịch vụ đại học: học phí, học bổng, ký túc xá, thư viện.")

suggestion_cols = st.columns(len(SUGGESTIONS))
for idx, (col, label) in enumerate(zip(suggestion_cols, SUGGESTIONS)):
    with col:
        if st.button(label, use_container_width=True, key=f"main_suggestion_{idx}"):
            st.session_state["pending_query"] = label

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách hoặc dịch vụ đại học...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            try:
                response = generate_with_citation(query, top_k=top_k)
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])
            except NotImplementedError:
                answer = (
                    "⚠️ **Task 10 chưa được implement.** Hãy hoàn thành "
                    "`src/task10_generation.py` để kết nối pipeline vào UI."
                )
                sources = []
            except Exception as exc:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {exc}"
                sources = []

            st.markdown(answer)
            render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )