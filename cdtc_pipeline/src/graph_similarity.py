'Graph edit distance and structural topological similarity calculation.'

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import networkx as nx

LOGGER = logging.getLogger("cdtc_pipeline")


def node_substitution_cost(node_a: dict, node_b: dict) -> float:
    'Node replacement cost function.'
    if (
        node_a.get("node_type") == node_b.get("node_type")
        and node_a.get("normalized_label") == node_b.get("normalized_label")
    ):
        return 0.0
    if node_a.get("node_type") == node_b.get("node_type"):
        return 0.5
    return 1.0


def edge_substitution_cost(edge_a: dict, edge_b: dict) -> float:
    'Edge replacement cost function.'
    return 0.0 if edge_a.get("relation") == edge_b.get("relation") else 0.5


def _constant_cost(_: Optional[dict] = None) -> float:
    'Insertion and deletion costs.'
    return 1.0


def _normalizer(model_graph: nx.DiGraph, expert_graph: nx.DiGraph, normalization: str) -> float:
    'Calculate the GED normalized denominator.'
    if normalization == "nodes":
        value = max(model_graph.number_of_nodes(), expert_graph.number_of_nodes())
    elif normalization == "nodes_edges":
        value = max(
            model_graph.number_of_nodes() + model_graph.number_of_edges(),
            expert_graph.number_of_nodes() + expert_graph.number_of_edges(),
        )
    else:
        raise ValueError("normalization must be 'nodes' or 'nodes_edges'")
    return float(max(1, value))


def compute_graph_edit_distance(
    model_graph: nx.DiGraph,
    expert_graph: nx.DiGraph,
    exact: bool = True,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    'Calculate exact GED; return None on failure or timeout for upper-level approximation.'
    if not exact:
        return {"ged_value": None, "ged_method": "approximate"}
    try:
        ged_value = nx.graph_edit_distance(
            model_graph,
            expert_graph,
            node_subst_cost=node_substitution_cost,
            node_del_cost=_constant_cost,
            node_ins_cost=_constant_cost,
            edge_subst_cost=edge_substitution_cost,
            edge_del_cost=_constant_cost,
            edge_ins_cost=_constant_cost,
            timeout=timeout,
        )
    except TypeError:
        # Note: Compatible with the old version of networkx without timeout parameter.
        ged_value = nx.graph_edit_distance(
            model_graph,
            expert_graph,
            node_subst_cost=node_substitution_cost,
            node_del_cost=_constant_cost,
            node_ins_cost=_constant_cost,
            edge_subst_cost=edge_substitution_cost,
            edge_del_cost=_constant_cost,
            edge_ins_cost=_constant_cost,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Exact GED failed, fallback to approximate: %s", exc)
        return {"ged_value": None, "ged_method": "approximate"}

    if ged_value is None:
        return {"ged_value": None, "ged_method": "approximate"}
    return {"ged_value": float(ged_value), "ged_method": "exact"}


def approximate_graph_similarity(
    model_graph: nx.DiGraph,
    expert_graph: nx.DiGraph,
    node_matching: Dict[str, Any],
    normalization: str = "nodes_edges",
) -> Dict[str, Any]:
    'Approximate graph similarity based on node matching and edge overlap between matching nodes.'
    matched_pairs = node_matching.get("matched_pairs", [])
    model_to_expert = {p["model_node_id"]: p["expert_node_id"] for p in matched_pairs}
    unmatched_node_cost = len(node_matching.get("unmatched_model_nodes", [])) + len(
        node_matching.get("unmatched_expert_nodes", [])
    )

    matched_edge_cost = 0.0
    model_matched_edges = set()
    expert_matched_edges = set()
    for model_source, model_target, model_attr in model_graph.edges(data=True):
        if model_source not in model_to_expert or model_target not in model_to_expert:
            continue
        expert_source = model_to_expert[model_source]
        expert_target = model_to_expert[model_target]
        model_matched_edges.add((model_source, model_target))
        if expert_graph.has_edge(expert_source, expert_target):
            expert_matched_edges.add((expert_source, expert_target))
            expert_attr = expert_graph.get_edge_data(expert_source, expert_target) or {}
            matched_edge_cost += edge_substitution_cost(model_attr, expert_attr)
        else:
            matched_edge_cost += 1.0

    # Note: Edges not covered by matching are processed at insertion/deletion costs.
    unmatched_model_edges = model_graph.number_of_edges() - len(model_matched_edges)
    unmatched_expert_edges = expert_graph.number_of_edges() - len(expert_matched_edges)
    approximate_ged = float(unmatched_node_cost + matched_edge_cost + unmatched_model_edges + unmatched_expert_edges)
    normalizer = _normalizer(model_graph, expert_graph, normalization)
    approximate_s_graph = 1.0 - approximate_ged / normalizer
    return {
        "ged_value": approximate_ged,
        "S_graph": max(0.0, min(1.0, float(approximate_s_graph))),
        "ged_method": "approximate",
    }


def compute_S_graph(
    model_graph: nx.DiGraph,
    expert_graph: nx.DiGraph,
    node_matching: Optional[Dict[str, Any]] = None,
    normalization: str = "nodes_edges",
    exact: bool = True,
    max_exact_nodes: int = 12,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    'Calculate the normalized structural topological similarity S_graph.'
    total_nodes = model_graph.number_of_nodes() + expert_graph.number_of_nodes()
    if total_nodes == 0:
        return {"S_graph": 1.0, "ged_value": 0.0, "ged_method": "exact"}

    can_try_exact = exact and total_nodes <= max_exact_nodes
    ged_result = compute_graph_edit_distance(
        model_graph,
        expert_graph,
        exact=can_try_exact,
        timeout=timeout,
    )
    if ged_result["ged_value"] is None:
        if node_matching is None:
            raise ValueError("node_matching is required for approximate graph similarity")
        return approximate_graph_similarity(model_graph, expert_graph, node_matching, normalization)

    normalizer = _normalizer(model_graph, expert_graph, normalization)
    s_graph = 1.0 - float(ged_result["ged_value"]) / normalizer
    return {
        "S_graph": max(0.0, min(1.0, float(s_graph))),
        "ged_value": float(ged_result["ged_value"]),
        "ged_method": ged_result["ged_method"],
    }
