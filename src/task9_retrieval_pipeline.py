"""Task 9: retrieval pipeline with score-gated PageIndex fallback.

The fallback decision always uses the original dense cosine score, never the
RRF score.  This matters because RRF scores describe rank agreement rather
than semantic relevance.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import Any

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf


SCORE_THRESHOLD = 0.48
DEFAULT_TOP_K = 5

logger = logging.getLogger(__name__)


def pageindex_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Load Task 8 lazily so Task 9 remains importable without its optional SDK."""
    from .task8_pageindex_vectorless import pageindex_search as task8_search

    return task8_search(query, top_k=top_k)


def _request_id(query: str) -> str:
    """Return a privacy-safe identifier without logging the full query."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]


def _valid_threshold(value: Any) -> float:
    """Parse a threshold in the cosine-similarity range, or use the default."""
    if isinstance(value, bool):
        logger.warning("invalid_score_threshold using_default=%s", SCORE_THRESHOLD)
        return SCORE_THRESHOLD
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        logger.warning("invalid_score_threshold using_default=%s", SCORE_THRESHOLD)
        return SCORE_THRESHOLD
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        logger.warning("invalid_score_threshold using_default=%s", SCORE_THRESHOLD)
        return SCORE_THRESHOLD
    return threshold


def _dense_top_score(dense_results: Any) -> tuple[float | None, str | None]:
    """Validate ``dense_results[0]['score']`` and explain invalid input."""
    if dense_results is None:
        return None, "dense_results_none"
    if not isinstance(dense_results, list):
        return None, "dense_results_invalid_type"
    if not dense_results:
        return None, "dense_results_empty"
    first = dense_results[0]
    if not isinstance(first, dict):
        return None, "dense_result_invalid_type"
    if "score" not in first:
        return None, "dense_score_missing"
    raw_score = first.get("score")
    if isinstance(raw_score, bool) or raw_score is None:
        return None, "dense_score_invalid"
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None, "dense_score_invalid"
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None, "dense_score_out_of_range"
    return score, None


def _normalize_results(results: Any, source: str, top_k: int) -> list[dict]:
    """Keep the retrieval schema stable even when a provider is permissive."""
    if not isinstance(results, list):
        return []
    normalized: list[dict] = []
    for raw in results:
        if not isinstance(raw, dict):
            continue
        content = raw.get("content")
        score = raw.get("score")
        if not isinstance(content, str) or not content.strip():
            continue
        if isinstance(score, bool):
            continue
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric_score):
            continue
        item = dict(raw)
        item["content"] = content
        item["score"] = numeric_score
        item["metadata"] = (
            dict(raw["metadata"]) if isinstance(raw.get("metadata"), dict) else {}
        )
        item["source"] = source
        normalized.append(item)
        if len(normalized) >= top_k:
            break
    return normalized


def _log_summary(
    *,
    query_id: str,
    dense_count: int,
    dense_top_score: float | None,
    threshold: float,
    fallback_triggered: bool,
    fallback_reason: str,
    pageindex_count: int,
    dense_latency_ms: float,
    sparse_latency_ms: float,
    rerank_latency_ms: float,
    pageindex_latency_ms: float,
) -> None:
    logger.info(
        "retrieval query_id=%s dense_result_count=%d dense_top_score=%s "
        "score_threshold=%.4f fallback_triggered=%s fallback_reason=%s "
        "pageindex_result_count=%d dense_latency_ms=%.2f sparse_latency_ms=%.2f "
        "rerank_latency_ms=%.2f pageindex_latency_ms=%.2f",
        query_id,
        dense_count,
        dense_top_score,
        threshold,
        fallback_triggered,
        fallback_reason,
        pageindex_count,
        dense_latency_ms,
        sparse_latency_ms,
        rerank_latency_ms,
        pageindex_latency_ms,
    )


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Retrieve documents and fall back when the best dense score is below threshold.

    Invalid/missing dense scores are treated as low confidence.  PageIndex is
    called at most once.  If PageIndex itself fails, valid dense results are
    returned as a safe degraded result instead of crashing the pipeline.
    """
    clean_query = query.strip() if isinstance(query, str) else ""
    if not clean_query or isinstance(top_k, bool) or top_k <= 0:
        logger.info(
            "retrieval query_id=invalid dense_result_count=0 dense_top_score=None "
            "score_threshold=%.4f fallback_triggered=False "
            "fallback_reason=invalid_request pageindex_result_count=0",
            _valid_threshold(score_threshold),
        )
        return []

    threshold = _valid_threshold(score_threshold)
    query_id = _request_id(clean_query)
    dense_latency_ms = sparse_latency_ms = 0.0
    rerank_latency_ms = pageindex_latency_ms = 0.0

    dense_started = time.perf_counter()
    try:
        dense_results = semantic_search(clean_query, top_k=top_k * 2)
    except Exception as error:
        dense_results = None
        logger.warning(
            "retrieval query_id=%s dense_error_type=%s",
            query_id,
            type(error).__name__,
        )
    dense_latency_ms = (time.perf_counter() - dense_started) * 1000.0
    dense_count = len(dense_results) if isinstance(dense_results, list) else 0
    top_score, invalid_reason = _dense_top_score(dense_results)

    fallback_reason = invalid_reason
    if fallback_reason is None and top_score is not None and top_score < threshold:
        fallback_reason = "dense_score_below_threshold"

    if fallback_reason is not None:
        pageindex_started = time.perf_counter()
        pageindex_error: Exception | None = None
        try:
            fallback_raw = pageindex_search(clean_query, top_k=top_k)
            fallback_results = _normalize_results(fallback_raw, "pageindex", top_k)
        except Exception as error:
            pageindex_error = error
            fallback_results = []
            logger.warning(
                "retrieval query_id=%s pageindex_error_type=%s fallback_reason=%s",
                query_id,
                type(error).__name__,
                fallback_reason,
            )
        pageindex_latency_ms = (time.perf_counter() - pageindex_started) * 1000.0
        logged_reason = fallback_reason
        if pageindex_error is not None:
            logged_reason += f":pageindex_{type(pageindex_error).__name__}"
        _log_summary(
            query_id=query_id,
            dense_count=dense_count,
            dense_top_score=top_score,
            threshold=threshold,
            fallback_triggered=True,
            fallback_reason=logged_reason,
            pageindex_count=len(fallback_results),
            dense_latency_ms=dense_latency_ms,
            sparse_latency_ms=sparse_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            pageindex_latency_ms=pageindex_latency_ms,
        )
        if fallback_results:
            return fallback_results
        return _normalize_results(dense_results, "hybrid", top_k)

    sparse_started = time.perf_counter()
    try:
        sparse_results = lexical_search(clean_query, top_k=top_k * 2)
        if not isinstance(sparse_results, list):
            sparse_results = []
    except Exception as error:
        sparse_results = []
        logger.warning(
            "retrieval query_id=%s sparse_error_type=%s",
            query_id,
            type(error).__name__,
        )
    sparse_latency_ms = (time.perf_counter() - sparse_started) * 1000.0

    rerank_started = time.perf_counter()
    if use_reranking:
        try:
            primary_raw = rerank_rrf(
                [sparse_results, dense_results], top_k=top_k
            )
        except Exception as error:
            primary_raw = dense_results
            logger.warning(
                "retrieval query_id=%s rerank_error_type=%s",
                query_id,
                type(error).__name__,
            )
    else:
        primary_raw = dense_results
    primary_results = _normalize_results(primary_raw, "hybrid", top_k)
    rerank_latency_ms = (time.perf_counter() - rerank_started) * 1000.0

    _log_summary(
        query_id=query_id,
        dense_count=dense_count,
        dense_top_score=top_score,
        threshold=threshold,
        fallback_triggered=False,
        fallback_reason="none",
        pageindex_count=0,
        dense_latency_ms=dense_latency_ms,
        sparse_latency_ms=sparse_latency_ms,
        rerank_latency_ms=rerank_latency_ms,
        pageindex_latency_ms=pageindex_latency_ms,
    )
    return primary_results


if __name__ == "__main__":
    for test_query in (
        "What is the tuition fee at UET?",
        "How do I book a library study room?",
        "What scholarships are available for UET students?",
    ):
        print(retrieve(test_query, top_k=3))
