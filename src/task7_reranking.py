"""
Task 7 — Reranking Module.

Hỗ trợ các phương pháp Reranking:
    1. RRF (Reciprocal Rank Fusion): gộp kết quả từ nhiều ranker theo thứ hạng.
    2. MMR (Maximal Marginal Relevance): cân bằng giữa relevance (độ liên quan) và diversity (độ đa dạng).
    3. Cross-Encoder (Jina Reranker API): Reranking mô hình học sâu (hỗ trợ API key và tự động fallback sang RRF/MMR khi API không khả dụng).
"""

import hashlib
import json
import logging
import math
import os
import sys
from typing import List, Dict, Any, Union, Optional
import numpy as np
import httpx

# Standardize output encoding for Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)

# Fallback Jina API key được cấp
DEFAULT_JINA_API_KEY = "jina_16657f63c8b24513a85a0c4c9bad6646OuISd-XWg_jk_2XEQeOBlG9XiKm4"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
DEFAULT_JINA_MODEL = "jina-reranker-v2-base-multilingual"
DEFAULT_JINA_TIMEOUT = 10.0
DEFAULT_MAX_CANDIDATES = 50
DEFAULT_RRF_K = 60


def _get_jina_api_key() -> Optional[str]:
    key = os.getenv("JINA_API_KEY")
    if key is not None:
        return key if key.strip() else None
    return DEFAULT_JINA_API_KEY


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Tính cosine similarity giữa 2 vector."""
    if not vec1 or not vec2:
        return 0.0
    a = np.array(vec1, dtype=float)
    b = np.array(vec2, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _text_similarity(text1: str, text2: str) -> float:
    """Tính Jaccard similarity giữa 2 văn bản theo tập hợp từ (tokens)."""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def _get_doc_id(item: Dict[str, Any]) -> str:
    """Tạo hoặc lấy ID cho document."""
    if "id" in item and item["id"]:
        return str(item["id"])
    if "document_id" in item and item["document_id"]:
        return str(item["document_id"])
    content = item.get("content", str(item))
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"content-sha256:{sha}"


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]], top_k: int = 5, k: int = 60
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))
    """
    if not ranked_lists or all(len(l) == 0 for l in ranked_lists):
        return []

    rrf_scores: Dict[str, float] = {}
    content_map: Dict[str, Dict[str, Any]] = {}
    doc_sources: Dict[str, list] = {}
    doc_scores: Dict[str, dict] = {}
    doc_metadatas: Dict[str, dict] = {}
    doc_ids: Dict[str, str] = {}

    for list_idx, r_list in enumerate(ranked_lists):
        source_name = "sparse" if list_idx == 0 else ("semantic" if list_idx == 1 else f"ranker_{list_idx}")
        for rank, item in enumerate(r_list, 1):
            key = item.get("content", str(item))
            doc_id = _get_doc_id(item)
            doc_ids[key] = doc_id
            
            score_delta = 1.0 / (k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + score_delta

            if key not in doc_sources:
                doc_sources[key] = []
                doc_scores[key] = {}
                doc_metadatas[key] = {}

            item_source = item.get("metadata", {}).get("source", source_name)
            if item_source not in doc_sources[key]:
                doc_sources[key].append(item_source)

            if "score" in item and item["score"] is not None:
                doc_scores[key][source_name] = item["score"]

            if "metadata" in item and isinstance(item["metadata"], dict):
                doc_metadatas[key].update(item["metadata"])

            if key not in content_map:
                content_map[key] = item.copy()

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for key, rrf_score in sorted_items[:top_k]:
        item = content_map[key].copy()
        doc_id = doc_ids[key]
        item["document_id"] = doc_id
        item["score"] = float(rrf_score)
        item["rerank_method"] = "rrf"
        
        sources = doc_sources[key]
        if len(sources) > 1:
            item["source"] = "hybrid"
            item["sources"] = sources
        elif len(sources) == 1:
            item["source"] = sources[0]
            item["sources"] = sources

        if doc_scores[key]:
            item["retrieval_scores"] = doc_scores[key]
        if doc_metadatas[key]:
            item["metadata"] = doc_metadatas[key]

        results.append(item)

    return results


def rerank_rrf(
    ranked_lists: List[List[Dict[str, Any]]], top_k: int = 5, k: int = 60
) -> List[Dict[str, Any]]:
    """Tương thích rerank_rrf interface."""
    return reciprocal_rank_fusion(ranked_lists=ranked_lists, top_k=top_k, k=k)


def rerank_mmr(
    candidates: List[Dict[str, Any]],
    query_embedding: Optional[List[float]] = None,
    query: Optional[str] = None,
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.
    """
    if not candidates:
        return []

    selected_indices: List[int] = []
    remaining_indices = list(range(len(candidates)))
    target_k = min(top_k, len(candidates))

    for _ in range(target_k):
        best_idx = -1
        best_mmr_score = float("-inf")

        for idx in remaining_indices:
            cand = candidates[idx]

            # Relevance score calculation
            if query_embedding and "embedding" in cand and cand["embedding"]:
                relevance = _cosine_similarity(query_embedding, cand["embedding"])
            elif "score" in cand and cand["score"] is not None:
                relevance = float(cand["score"])
            elif query:
                relevance = _text_similarity(query, cand["content"])
            else:
                relevance = 0.5

            # Diversity penalty
            max_sim_to_selected = 0.0
            for sel_idx in selected_indices:
                sel_cand = candidates[sel_idx]
                if "embedding" in cand and "embedding" in sel_cand and cand["embedding"] and sel_cand["embedding"]:
                    sim = _cosine_similarity(cand["embedding"], sel_cand["embedding"])
                else:
                    sim = _text_similarity(cand["content"], sel_cand["content"])
                if sim > max_sim_to_selected:
                    max_sim_to_selected = sim

            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx != -1:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    results = []
    for rank_idx, idx in enumerate(selected_indices):
        item = candidates[idx].copy()
        item["document_id"] = _get_doc_id(item)
        item["score"] = float(item.get("score", 1.0 / (rank_idx + 1)))
        item["rerank_method"] = "mmr"
        results.append(item)

    return results


def rerank_cross_encoder(
    query: str, candidates: List[Dict[str, Any]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Rerank candidates sử dụng Cross-encoder API (Jina Reranker) với tự động fallback sang MMR/RRF.
    """
    api_key = _get_jina_api_key()
    if not api_key:
        return rerank_mmr(candidates=candidates, query=query, top_k=top_k)

    try:
        max_cands = int(os.getenv("JINA_RERANK_MAX_CANDIDATES", DEFAULT_MAX_CANDIDATES))
        input_cands = candidates[:max_cands]
        docs = [c.get("content", str(c)) for c in input_cands]
        model_name = os.getenv("JINA_RERANK_MODEL", DEFAULT_JINA_MODEL)

        response = httpx.post(
            JINA_RERANK_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_name,
                "query": query,
                "documents": docs,
                "return_documents": False,
            },
            timeout=DEFAULT_JINA_TIMEOUT,
        )
        if response.status_code == 200:
            res_data = response.json()
            if "results" in res_data and isinstance(res_data["results"], list):
                results = []
                for item in res_data["results"]:
                    idx = item.get("index")
                    rel_score = item.get("relevance_score")
                    if idx is not None and 0 <= idx < len(input_cands) and isinstance(rel_score, (int, float)):
                        cand = input_cands[idx].copy()
                        cand["document_id"] = _get_doc_id(cand)
                        cand["score"] = float(rel_score)
                        cand["rerank_method"] = "jina"
                        cand["rerank_model"] = model_name
                        results.append(cand)

                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:top_k]
    except Exception as e:
        logger.warning(f"Jina Rerank API call failed: {e}. Falling back to MMR.")

    return rerank_mmr(candidates=candidates, query=query, top_k=top_k)


async def rerank_results(
    query: str,
    sparse_results: Optional[List[Dict[str, Any]]] = None,
    semantic_results: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5,
    client: Optional[httpx.AsyncClient] = None,
) -> List[Dict[str, Any]]:
    """
    Async rerank_results interface cho hybrid pipeline và async testing.
    """
    if not query or not query.strip():
        return []

    sparse_results = sparse_results or []
    semantic_results = semantic_results or []

    if not sparse_results and not semantic_results:
        return []

    ranked_lists = []
    if sparse_results:
        ranked_lists.append(sparse_results)
    if semantic_results:
        ranked_lists.append(semantic_results)

    api_key = os.getenv("JINA_API_KEY")
    if api_key is not None and not api_key.strip():
        api_key = None

    if not api_key:
        return reciprocal_rank_fusion(ranked_lists, top_k=top_k)

    # Flatten & deduplicate candidates in input order up to max_cands
    max_cands = int(os.getenv("JINA_RERANK_MAX_CANDIDATES", DEFAULT_MAX_CANDIDATES))
    combined_candidates: List[Dict[str, Any]] = []
    seen = set()
    doc_sources: Dict[str, list] = {}
    doc_scores: Dict[str, dict] = {}
    doc_metadatas: Dict[str, dict] = {}

    for list_idx, r_list in enumerate(ranked_lists):
        source_name = "sparse" if list_idx == 0 else ("semantic" if list_idx == 1 else f"ranker_{list_idx}")
        for item in r_list:
            key = item.get("content", str(item))
            if key not in doc_sources:
                doc_sources[key] = []
                doc_scores[key] = {}
                doc_metadatas[key] = {}

            item_source = item.get("metadata", {}).get("source", source_name)
            if item_source not in doc_sources[key]:
                doc_sources[key].append(item_source)

            if "score" in item and item["score"] is not None:
                doc_scores[key][source_name] = item["score"]

            if "metadata" in item and isinstance(item["metadata"], dict):
                doc_metadatas[key].update(item["metadata"])

            if key not in seen and len(combined_candidates) < max_cands:
                seen.add(key)
                cand = item.copy()
                cand["document_id"] = _get_doc_id(item)
                combined_candidates.append(cand)

    for item in combined_candidates:
        key = item.get("content", str(item))
        sources = doc_sources[key]
        if len(sources) > 1:
            item["source"] = "hybrid"
            item["sources"] = sources
        elif len(sources) == 1:
            item["source"] = sources[0]
            item["sources"] = sources

        if doc_scores[key]:
            item["retrieval_scores"] = doc_scores[key]
        if doc_metadatas[key]:
            item["metadata"] = doc_metadatas[key]

    docs = [c.get("content", str(c)) for c in combined_candidates]
    model_name = os.getenv("JINA_RERANK_MODEL", DEFAULT_JINA_MODEL)
    payload = {
        "model": model_name,
        "query": query,
        "documents": docs,
        "return_documents": False,
    }

    try:
        if client is not None:
            resp = await client.post(
                JINA_RERANK_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=DEFAULT_JINA_TIMEOUT,
            )
        else:
            async with httpx.AsyncClient() as async_client:
                resp = await async_client.post(
                    JINA_RERANK_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                    timeout=DEFAULT_JINA_TIMEOUT,
                )

        if resp.status_code == 200:
            res_data = resp.json()
            if "results" in res_data and isinstance(res_data["results"], list) and len(res_data["results"]) > 0:
                results = []
                for item in res_data["results"]:
                    idx = item.get("index")
                    rel_score = item.get("relevance_score")
                    if (
                        idx is not None
                        and isinstance(idx, int)
                        and 0 <= idx < len(combined_candidates)
                        and isinstance(rel_score, (int, float))
                        and math.isfinite(rel_score)
                    ):
                        cand = combined_candidates[idx].copy()
                        cand["score"] = float(rel_score)
                        cand["rerank_method"] = "jina"
                        cand["rerank_model"] = model_name
                        results.append(cand)

                if results:
                    results.sort(key=lambda x: x["score"], reverse=True)
                    return results[:top_k]
    except Exception as e:
        logger.warning(f"Jina API request failed or timed out: {e}")

    # Fallback to RRF
    return reciprocal_rank_fusion(ranked_lists, top_k=top_k)


def rerank(
    query: str,
    candidates: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    top_k: int = 5,
    method: str = "rrf",  # "rrf" | "mmr" | "cross_encoder"
) -> List[Dict[str, Any]]:
    """
    Unified reranking interface.
    """
    if not candidates:
        return []

    # Multiple ranked lists
    if isinstance(candidates, list) and len(candidates) > 0 and isinstance(candidates[0], list):
        if method == "mmr":
            flattened = [item for sublist in candidates for item in sublist]
            return rerank_mmr(candidates=flattened, query=query, top_k=top_k)
        else:
            return reciprocal_rank_fusion(ranked_lists=candidates, top_k=top_k)

    # Single list of candidates
    single_candidates: List[Dict[str, Any]] = candidates  # type: ignore

    if method == "mmr":
        return rerank_mmr(candidates=single_candidates, query=query, top_k=top_k)
    elif method == "cross_encoder":
        return rerank_cross_encoder(query=query, candidates=single_candidates, top_k=top_k)
    else:
        return reciprocal_rank_fusion(ranked_lists=[single_candidates], top_k=top_k)


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]
    print("Testing RRF Reranking:")
    results_rrf = rerank("tuition fee payment", dummy_candidates, top_k=2, method="rrf")
    for r in results_rrf:
        print(f"  [{r['score']:.4f}] {r['content']}")

    print("\nTesting MMR Reranking:")
    results_mmr = rerank("tuition fee payment", dummy_candidates, top_k=2, method="mmr")
    for r in results_mmr:
        print(f"  [{r['score']:.4f}] {r['content']}")
