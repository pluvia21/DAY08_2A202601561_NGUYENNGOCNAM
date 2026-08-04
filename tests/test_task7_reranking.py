import json
import math

import httpx
import pytest

from src.task7_reranking import reciprocal_rank_fusion, rerank_results


SPARSE = [
    {"id": "a", "content": "Thông báo học phí", "score": 9.0,
     "metadata": {"kind": "fee"}},
    {"id": "b", "content": "Học bổng sinh viên", "score": 8.0,
     "metadata": {"from_sparse": True}},
]
SEMANTIC = [
    {"id": "b", "content": "Học bổng sinh viên", "score": 0.95,
     "metadata": {"from_semantic": True}},
    {"id": "c", "content": "Dịch vụ thư viện", "score": 0.85,
     "metadata": {"kind": "library"}},
]


async def _run(handler, monkeypatch, *, sparse=None, semantic=None, top_k=3):
    monkeypatch.setenv("JINA_API_KEY", "test-key-never-sent-to-real-api")
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        return await rerank_results(
            "học bổng và thư viện",
            SPARSE if sparse is None else sparse,
            SEMANTIC if semantic is None else semantic,
            top_k=top_k,
            client=client,
        )


@pytest.mark.asyncio
async def test_jina_success_preserves_mapping_metadata_and_model(monkeypatch):
    monkeypatch.setenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")

    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/rerank"
        assert request.headers["Authorization"] == "Bearer test-key-never-sent-to-real-api"
        assert body["documents"] == [
            "Thông báo học phí", "Học bổng sinh viên", "Dịch vụ thư viện"
        ]
        assert body["return_documents"] is False
        return httpx.Response(200, json={"results": [
            {"index": 2, "relevance_score": 0.97},
            {"index": 0, "relevance_score": 0.61},
        ]})

    results = await _run(handler, monkeypatch, top_k=2)
    assert [item["document_id"] for item in results] == ["c", "a"]
    assert [item["score"] for item in results] == [0.97, 0.61]
    assert all(item["rerank_method"] == "jina" for item in results)
    assert results[0]["rerank_model"] == "jina-reranker-v2-base-multilingual"
    assert results[0]["metadata"] == {"kind": "library"}


@pytest.mark.asyncio
async def test_missing_key_uses_rrf(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    results = await rerank_results("học bổng", SPARSE, SEMANTIC, top_k=3)
    assert results
    assert all(item["rerank_method"] == "rrf" for item in results)


@pytest.mark.asyncio
async def test_jina_timeout_falls_back_to_rrf(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("provider timeout", request=request)

    results = await _run(handler, monkeypatch)
    assert results and all(item["rerank_method"] == "rrf" for item in results)


@pytest.mark.asyncio
async def test_jina_http_error_falls_back_to_rrf(monkeypatch):
    def handler(request):
        return httpx.Response(429, json={"detail": "rate limited"})

    results = await _run(handler, monkeypatch)
    assert results and all(item["rerank_method"] == "rrf" for item in results)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"unexpected": []},
    {"results": []},
    {"results": [{"index": 0}]},
    {"results": [{"index": 0, "relevance_score": "not-a-number"}]},
    {"results": [{"index": 99, "relevance_score": 0.5}]},
])
async def test_invalid_jina_schema_falls_back_to_rrf(monkeypatch, payload):
    def handler(request):
        return httpx.Response(200, json=payload)

    results = await _run(handler, monkeypatch)
    assert results and all(item["rerank_method"] == "rrf" for item in results)


@pytest.mark.asyncio
async def test_deduplicates_and_merges_metadata(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    results = await rerank_results("học bổng", SPARSE, SEMANTIC, top_k=10)
    assert len(results) == 3
    duplicate = next(item for item in results if item["document_id"] == "b")
    assert duplicate["source"] == "hybrid"
    assert duplicate["sources"] == ["sparse", "semantic"]
    assert duplicate["retrieval_scores"] == {"sparse": 8.0, "semantic": 0.95}
    assert duplicate["metadata"] == {
        "from_sparse": True, "from_semantic": True
    }


def test_rrf_formula_is_exact_and_sorted():
    first = [
        {"id": "a", "content": "A", "score": 10, "metadata": {}},
        {"id": "b", "content": "B", "score": 9, "metadata": {}},
    ]
    second = [{"id": "a", "content": "A", "score": 0.9, "metadata": {}}]
    results = reciprocal_rank_fusion([first, second], top_k=2, k=60)
    assert results[0]["document_id"] == "a"
    assert math.isclose(results[0]["score"], 2.0 / 61.0)
    assert math.isclose(results[1]["score"], 1.0 / 62.0)
    assert [item["score"] for item in results] == sorted(
        (item["score"] for item in results), reverse=True
    )


def test_rrf_applies_top_k_and_generates_stable_id():
    results = reciprocal_rank_fusion([SPARSE], top_k=1)
    assert len(results) == 1
    without_id = reciprocal_rank_fusion(
        [[{"content": "Không có ID", "score": 1, "metadata": {}}]], top_k=1
    )
    assert without_id[0]["document_id"].startswith("content-sha256:")


@pytest.mark.asyncio
async def test_empty_query_and_empty_candidates_return_empty(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "unused-test-key")
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await rerank_results("   ", SPARSE, SEMANTIC, client=client) == []
        assert await rerank_results("query", [], [], client=client) == []
    assert calls == 0


@pytest.mark.asyncio
async def test_candidate_limit_controls_jina_request(monkeypatch):
    monkeypatch.setenv("JINA_RERANK_MAX_CANDIDATES", "2")

    def handler(request):
        body = json.loads(request.content)
        assert len(body["documents"]) == 2
        return httpx.Response(200, json={"results": [
            {"index": 0, "relevance_score": 0.8}
        ]})

    results = await _run(handler, monkeypatch)
    assert len(results) == 1 and results[0]["rerank_method"] == "jina"
