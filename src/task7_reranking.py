"""Task 7: async Jina reranking with deterministic RRF fallback."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import time
from typing import Any

import httpx

SearchResult = dict[str, Any]
RerankedResult = dict[str, Any]
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
DEFAULT_JINA_MODEL = "jina-reranker-v2-base-multilingual"
DEFAULT_JINA_TIMEOUT = 10.0
DEFAULT_MAX_CANDIDATES = 50
DEFAULT_RRF_K = 60
logger = logging.getLogger(__name__)


class JinaRerankError(RuntimeError):
    """Jina returned a response that cannot be mapped safely."""


def _env_number(name: str, default: int | float, cast: type) -> int | float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = cast(raw)
    except ValueError:
        value = default
    if not math.isfinite(float(value)) or value <= 0:
        value = default
    if value == default and raw != str(default):
        logger.warning("invalid_config name=%s using_default=%s", name, default)
    return value


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _content(item: SearchResult) -> str | None:
    value = item.get("content")
    document = item.get("document")
    if not isinstance(value, str) and isinstance(document, str):
        value = document
    if not isinstance(value, str) and isinstance(document, dict):
        value = document.get("text")
    return value if isinstance(value, str) and value.strip() else None


def _document_id(item: SearchResult, content: str) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    explicit = (item.get("document_id") or item.get("id") or
                metadata.get("document_id") or metadata.get("id"))
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    if metadata.get("source") is not None and metadata.get("chunk_index") is not None:
        return f"{metadata['source']}::chunk-{metadata['chunk_index']}"
    return "content-sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


class _Candidates:
    def __init__(self) -> None:
        self.items: dict[str, SearchResult] = {}
        self.by_content: dict[str, str] = {}
        self.order: dict[str, int] = {}

    def add(self, item: SearchResult, source: str) -> str | None:
        if not isinstance(item, dict):
            logger.warning("skip_candidate reason=not_mapping source=%s", source)
            return None
        content = _content(item)
        if content is None:
            logger.warning("skip_candidate reason=missing_content source=%s", source)
            return None
        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
        document_id = _document_id(item, content)
        key = self.by_content.get(fingerprint, document_id)
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if key not in self.items:
            normalized = dict(item)
            normalized.update({
                "document_id": document_id, "content": content,
                "metadata": dict(metadata),
                "original_score": item.get("original_score", item.get("score")),
                "retrieval_scores": {}, "sources": [],
            })
            self.items[key] = normalized
            self.by_content[fingerprint] = key
            self.order[key] = len(self.order)
        else:
            normalized = self.items[key]
            for name, value in metadata.items():
                normalized["metadata"].setdefault(name, value)
        if source not in normalized["sources"]:
            normalized["sources"].append(source)
        score = _finite_float(item.get("score"))
        if score is not None:
            normalized["retrieval_scores"][source] = score
        normalized["source"] = "hybrid" if len(normalized["sources"]) > 1 else source
        return key


def _deduplicate(sparse: list[SearchResult], semantic: list[SearchResult],
                 limit: int) -> list[SearchResult]:
    candidates = _Candidates()
    inputs = (("sparse", sparse), ("semantic", semantic))
    for rank in range(max((len(items) for _, items in inputs), default=0)):
        for source, items in inputs:
            if rank < len(items):
                candidates.add(items[rank], source)
                if len(candidates.items) >= limit:
                    return list(candidates.items.values())
    return list(candidates.items.values())


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]], top_k: int = 10,
    k: int | None = None, source_names: list[str] | None = None,
) -> list[RerankedResult]:
    """Apply RRF(d) = sum(1 / (k + rank_r(d))), rank starts at 1."""
    if top_k <= 0 or not ranked_lists:
        return []
    rrf_k = int(_env_number("RRF_K", DEFAULT_RRF_K, int)) if k is None else k
    if rrf_k <= 0:
        raise ValueError("RRF k must be greater than zero")
    if source_names is not None and len(source_names) != len(ranked_lists):
        raise ValueError("source_names must match ranked_lists")
    candidates, scores = _Candidates(), {}
    for index, ranked in enumerate(ranked_lists):
        source = source_names[index] if source_names else f"ranker_{index}"
        seen: set[str] = set()
        for rank, item in enumerate(ranked, 1):
            key = candidates.add(item, source)
            if key is None or key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    keys = sorted(scores, key=lambda key: (-scores[key], candidates.order[key]))
    output = []
    for key in keys[:top_k]:
        result = dict(candidates.items[key])
        result.update(score=scores[key], rrf_score=scores[key], rerank_method="rrf")
        result["metadata"] = dict(result["metadata"])
        result["retrieval_scores"] = dict(result["retrieval_scores"])
        result["sources"] = list(result["sources"])
        output.append(result)
    logger.info("rerank method=rrf candidates=%d output=%d k=%d",
                len(scores), len(output), rrf_k)
    return output


def rerank_rrf(ranked_lists: list[list[SearchResult]], top_k: int = 5,
               k: int = 60) -> list[RerankedResult]:
    """Backward-compatible RRF API."""
    return reciprocal_rank_fusion(ranked_lists, top_k, k)


def _parse_jina(payload: Any, candidates: list[SearchResult], top_k: int,
                model: str) -> list[RerankedResult]:
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise JinaRerankError("results must be a non-empty list")
    output, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise JinaRerankError("result must be an object")
        index = row.get("index")
        score = _finite_float(row.get("relevance_score"))
        if (isinstance(index, bool) or not isinstance(index, int) or index < 0 or
                index >= len(candidates) or index in seen or score is None):
            raise JinaRerankError("invalid index or relevance_score")
        seen.add(index)
        result = dict(candidates[index])
        result.update(score=score, rerank_score=score, rerank_method="jina",
                      rerank_model=model)
        result["metadata"] = dict(result.get("metadata", {}))
        output.append(result)
    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


async def jina_rerank(
    query: str, candidates: list[SearchResult], top_k: int = 10, *,
    api_key: str | None = None, model: str | None = None,
    timeout: float | None = None, client: httpx.AsyncClient | None = None,
) -> list[RerankedResult]:
    """Call POST /v1/rerank and map provider indices to original documents."""
    query = query.strip() if isinstance(query, str) else ""
    if not query or top_k <= 0 or not candidates:
        return []
    key = (api_key or os.getenv("JINA_API_KEY") or "").strip()
    if not key:
        raise JinaRerankError("JINA_API_KEY is not configured")
    model = (model or os.getenv("JINA_RERANK_MODEL") or DEFAULT_JINA_MODEL).strip()
    timeout = timeout or float(_env_number(
        "JINA_RERANK_TIMEOUT", DEFAULT_JINA_TIMEOUT, float))
    body = {"model": model, "query": query,
            "documents": [item["content"] for item in candidates],
            "top_n": min(top_k, len(candidates)), "return_documents": False}
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json",
               "Content-Type": "application/json"}
    if client:
        response = await client.post(JINA_RERANK_URL, headers=headers, json=body,
                                     timeout=timeout)
    else:
        async with httpx.AsyncClient(timeout=timeout) as owned:
            response = await owned.post(JINA_RERANK_URL, headers=headers, json=body)
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as error:
        raise JinaRerankError("response is not valid JSON") from error
    return _parse_jina(payload, candidates, top_k, model)


def _fallback_reason(error: Exception) -> str:
    if isinstance(error, httpx.TimeoutException):
        return "timeout"
    if isinstance(error, httpx.HTTPStatusError):
        return f"http_{error.response.status_code}"
    if isinstance(error, JinaRerankError):
        return "invalid_response"
    if isinstance(error, httpx.HTTPError):
        return "network_error"
    return "provider_error"


async def rerank_results(
    query: str, sparse_results: list[SearchResult],
    semantic_results: list[SearchResult], top_k: int = 10, *,
    client: httpx.AsyncClient | None = None,
) -> list[RerankedResult]:
    """Deduplicate candidates, prefer Jina, and safely fall back to RRF."""
    started = time.perf_counter()
    query = query.strip() if isinstance(query, str) else ""
    if not query or top_k <= 0:
        logger.info("rerank method=none candidates=0 output=0 reason=invalid_input")
        return []
    sparse = sparse_results if isinstance(sparse_results, list) else []
    semantic = semantic_results if isinstance(semantic_results, list) else []
    limit = int(_env_number("JINA_RERANK_MAX_CANDIDATES",
                            DEFAULT_MAX_CANDIDATES, int))
    candidates = _deduplicate(sparse, semantic, limit)
    if not candidates:
        logger.info("rerank method=none candidates=0 output=0 reason=no_candidates")
        return []
    api_key = (os.getenv("JINA_API_KEY") or "").strip()
    model = (os.getenv("JINA_RERANK_MODEL") or DEFAULT_JINA_MODEL).strip()
    if api_key:
        try:
            output = await jina_rerank(query, candidates, top_k, api_key=api_key,
                                       model=model, client=client)
            if not output:
                raise JinaRerankError("empty provider result")
            logger.info(
                "rerank method=jina candidates=%d output=%d model=%s latency_ms=%.2f",
                len(candidates), len(output), model,
                (time.perf_counter() - started) * 1000)
            return output
        except Exception as error:
            logger.warning(
                "rerank method=rrf candidates=%d model=%s latency_ms=%.2f "
                "fallback_reason=%s error_type=%s", len(candidates), model,
                (time.perf_counter() - started) * 1000, _fallback_reason(error),
                type(error).__name__)
    else:
        logger.info(
            "rerank method=rrf candidates=%d fallback_reason=missing_api_key",
            len(candidates))
    return reciprocal_rank_fusion(
        [sparse, semantic], top_k, source_names=["sparse", "semantic"])


def rerank_mmr(query_embedding: list[float], candidates: list[SearchResult],
               top_k: int = 5, lambda_param: float = 0.7) -> list[RerankedResult]:
    """Small compatibility implementation of Maximal Marginal Relevance."""
    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must be between 0 and 1")

    def cosine(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            raise ValueError("invalid embedding dimensions")
        dot = sum(x * y for x, y in zip(left, right))
        left_norm = sum(x * x for x in left)
        right_norm = sum(y * y for y in right)
        return dot / math.sqrt(left_norm * right_norm) if left_norm and right_norm else 0.0

    valid = [item for item in candidates if isinstance(item.get("embedding"), list)]
    selected: list[int] = []
    remaining = list(range(len(valid)))
    while remaining and len(selected) < max(0, top_k):
        def mmr(index: int) -> float:
            relevance = cosine(query_embedding, valid[index]["embedding"])
            diversity = max(
                (cosine(valid[index]["embedding"], valid[old]["embedding"])
                 for old in selected), default=0.0)
            return lambda_param * relevance - (1 - lambda_param) * diversity

        best = max(remaining, key=mmr)
        selected.append(best)
        remaining.remove(best)
    output = [dict(valid[index]) for index in selected]
    for item in output:
        item["rerank_method"] = "mmr"
    return output


def _run_async(coroutine: Any) -> list[RerankedResult]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    raise RuntimeError("Use 'await rerank_results(...)' inside an event loop")


def rerank_cross_encoder(query: str, candidates: list[SearchResult],
                         top_k: int = 5) -> list[RerankedResult]:
    """Backward-compatible synchronous Jina/RRF API."""
    return _run_async(rerank_results(query, candidates, [], top_k))


def rerank(query: str, candidates: list[SearchResult], top_k: int = 5,
           method: str = "rrf") -> list[RerankedResult]:
    """Backward-compatible synchronous reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return reciprocal_rank_fusion([candidates], top_k)
    if method == "mmr":
        embedding = next(
            (item.get("query_embedding") for item in candidates
             if isinstance(item.get("query_embedding"), list)), None)
        if embedding is None:
            raise ValueError("MMR requires query_embedding")
        return rerank_mmr(embedding, candidates, top_k)
    raise ValueError(f"Unknown rerank method: {method}")
