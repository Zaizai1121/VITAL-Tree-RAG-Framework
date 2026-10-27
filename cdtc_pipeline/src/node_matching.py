'Node normalization, text similarity, and greedy matching.'

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz

from .schema import ClinicalNode

LOGGER = logging.getLogger("cdtc_pipeline")
_EMBEDDING_MODEL = None
_EMBEDDING_AVAILABLE = None


def normalize_label(text: str) -> str:
    'Normalizes free text labels to lowercase and underlined form.'
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    'Lazy loading of sentence-transformers; upon failure, the upper layer will fall back to fuzzy.'
    global _EMBEDDING_MODEL, _EMBEDDING_AVAILABLE
    if _EMBEDDING_AVAILABLE is False:
        return None
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL = SentenceTransformer(model_name)
        _EMBEDDING_AVAILABLE = True
    except Exception as exc:  # pragma: no cover - triggered when dependencies or network are unavailable
        LOGGER.warning("sentence-transformers unavailable, fallback to fuzzy similarity: %s", exc)
        _EMBEDDING_AVAILABLE = False
        _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


@lru_cache(maxsize=20000)
def _embedding_text_similarity_cached(a: str, b: str, model_name: str) -> Optional[float]:
    'Cache embedding text similarity to reduce repeated encoding overhead.'
    model = _load_embedding_model(model_name)
    if model is None:
        return None
    vectors = model.encode([a, b], normalize_embeddings=True)
    return float(np.dot(vectors[0], vectors[1]))


def compute_text_similarity(
    a: str,
    b: str,
    method: str = "embedding",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    'Calculate the similarity between two pieces of text; automatically fall back to rapidfuzz when embedding is unavailable.'
    a = a or ""
    b = b or ""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if method == "embedding":
        embedding_score = _embedding_text_similarity_cached(a, b, model_name)
        if embedding_score is not None:
            return max(0.0, min(1.0, embedding_score))
    # Note: fuzzy score is 0-100, scaled here to 0-1.
    return float(fuzz.token_set_ratio(a, b) / 100.0)


def _node_time_compatible(
    model_node: ClinicalNode,
    expert_node: ClinicalNode,
    time_window: Optional[float],
) -> bool:
    'Determine whether the relative time of the node is within the allowed window.'
    if time_window is None:
        return True
    if model_node.relative_day is None or expert_node.relative_day is None:
        return True
    return abs(float(model_node.relative_day) - float(expert_node.relative_day)) <= float(time_window)


def _node_text(node: ClinicalNode) -> str:
    'When matching, standardized tags are preferred, supplemented by original descriptions.'
    parts = [node.normalized_label or normalize_label(node.label), node.description or ""]
    return " ".join(part for part in parts if part).strip()


def match_nodes(
    model_nodes: Iterable[ClinicalNode],
    expert_nodes: Iterable[ClinicalNode],
    threshold: float = 0.80,
    time_window: Optional[float] = 1,
    mode: str = "semantic",
    text_similarity_method: str = "embedding",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    'Performs node matching as defined and returns S_node with unmatched nodes.\n\n    exact mode requires node_type to be exactly the same as normalized_label;\n    Semantic mode uses text similarity thresholds on a type and time compatible basis.'
    model_nodes = list(model_nodes)
    expert_nodes = list(expert_nodes)
    if len(model_nodes) == 0 and len(expert_nodes) == 0:
        return {
            "matched_pairs": [],
            "unmatched_model_nodes": [],
            "unmatched_expert_nodes": [],
            "S_node": 1.0,
        }

    candidates: List[Tuple[float, int, int]] = []
    for i, model_node in enumerate(model_nodes):
        for j, expert_node in enumerate(expert_nodes):
            if model_node.node_type != expert_node.node_type:
                continue
            if not _node_time_compatible(model_node, expert_node, time_window):
                continue

            model_label = normalize_label(model_node.normalized_label)
            expert_label = normalize_label(expert_node.normalized_label)
            if mode == "exact":
                score = 1.0 if model_label == expert_label else 0.0
            else:
                if model_label == expert_label:
                    score = 1.0
                else:
                    score = compute_text_similarity(
                        _node_text(model_node),
                        _node_text(expert_node),
                        method=text_similarity_method,
                        model_name=embedding_model,
                    )
            if score >= threshold:
                candidates.append((score, i, j))

    # Note: Greedy matching from high to low based on similarity to prevent a node from being reused.
    candidates.sort(key=lambda item: item[0], reverse=True)
    used_model = set()
    used_expert = set()
    matched_pairs: List[Dict[str, Any]] = []
    for score, model_idx, expert_idx in candidates:
        if model_idx in used_model or expert_idx in used_expert:
            continue
        used_model.add(model_idx)
        used_expert.add(expert_idx)
        matched_pairs.append(
            {
                "model_node_id": model_nodes[model_idx].node_id,
                "expert_node_id": expert_nodes[expert_idx].node_id,
                "similarity": float(score),
                "node_type": model_nodes[model_idx].node_type,
            }
        )

    unmatched_model_nodes = [
        node for idx, node in enumerate(model_nodes) if idx not in used_model
    ]
    unmatched_expert_nodes = [
        node for idx, node in enumerate(expert_nodes) if idx not in used_expert
    ]
    union_size = len(model_nodes) + len(expert_nodes) - len(matched_pairs)
    s_node = float(len(matched_pairs) / union_size) if union_size else 1.0
    return {
        "matched_pairs": matched_pairs,
        "unmatched_model_nodes": unmatched_model_nodes,
        "unmatched_expert_nodes": unmatched_expert_nodes,
        "S_node": max(0.0, min(1.0, s_node)),
    }

