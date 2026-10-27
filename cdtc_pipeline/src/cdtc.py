'CDTC main calculation logic.'

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from .graph_builder import trajectory_to_digraph
from .graph_similarity import compute_S_graph
from .node_matching import match_nodes
from .schema import CDTCResult, ClinicalTrajectory
from .validation import normalize_redline_status, validate_alpha, validate_unit_interval

LOGGER = logging.getLogger("cdtc_pipeline")


def _alpha_score(s_node: float, s_graph: float, alpha: float) -> float:
    'Synthesize CDTC based on alpha.'
    validate_alpha(alpha)
    return float(alpha * s_node + (1.0 - alpha) * s_graph)


def compute_cdtc(
    model_trajectory: ClinicalTrajectory,
    expert_trajectory: ClinicalTrajectory,
    alpha: float = 0.5,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    'Calculate the CDTC of a single model output relative to the expert trajectory.'
    validate_alpha(alpha)
    config = config or {}
    matching_cfg = config.get("matching", {})
    graph_cfg = config.get("graph", {})

    if len(model_trajectory.nodes) == 0 or len(expert_trajectory.nodes) == 0:
        LOGGER.warning(
            "Empty graph skipped or penalized: output_id=%s case_id=%s",
            model_trajectory.output_id,
            model_trajectory.case_id,
        )

    node_match = match_nodes(
        model_trajectory.nodes,
        expert_trajectory.nodes,
        threshold=float(matching_cfg.get("semantic_threshold", 0.80)),
        time_window=matching_cfg.get("time_window", 1),
        mode=matching_cfg.get("mode", "semantic"),
        text_similarity_method=matching_cfg.get("text_similarity_method", "embedding"),
        embedding_model=matching_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    model_graph = trajectory_to_digraph(model_trajectory)
    expert_graph = trajectory_to_digraph(expert_trajectory)
    graph_result = compute_S_graph(
        model_graph,
        expert_graph,
        node_matching=node_match,
        normalization=graph_cfg.get("normalization", "nodes_edges"),
        exact=bool(graph_cfg.get("exact_ged", True)),
        max_exact_nodes=int(graph_cfg.get("max_exact_nodes", 12)),
        timeout=float(graph_cfg.get("ged_timeout_seconds", 2)),
    )

    s_node = float(node_match["S_node"])
    s_graph = float(graph_result["S_graph"])
    cdtc_value = _alpha_score(s_node, s_graph, alpha)
    for metric_name, value in {
        "S_node": s_node,
        "S_graph": s_graph,
        "CDTC": cdtc_value,
    }.items():
        validate_unit_interval(value, metric_name)

    return {
        "S_node": s_node,
        "S_graph": s_graph,
        "CDTC": cdtc_value,
        "matched_nodes": node_match["matched_pairs"],
        "unmatched_model_nodes": node_match["unmatched_model_nodes"],
        "unmatched_expert_nodes": node_match["unmatched_expert_nodes"],
        "ged_value": float(graph_result["ged_value"]),
        "ged_method": graph_result["ged_method"],
    }


def _score_lookup(scores: pd.DataFrame) -> Dict[str, dict]:
    'Index the expert rating table by output_id.'
    return {str(row["output_id"]): row.to_dict() for _, row in scores.iterrows()}


def _expert_lookup(expert_trajectories: Iterable[ClinicalTrajectory]) -> Dict[Tuple[str, str], ClinicalTrajectory]:
    'Index expert tracks by case_id and disease_task.'
    return {
        (trajectory.case_id, trajectory.disease_task or ""): trajectory
        for trajectory in expert_trajectories
    }


def compute_cdtc_for_all(
    model_trajectories: List[ClinicalTrajectory],
    expert_trajectories: List[ClinicalTrajectory],
    scores: pd.DataFrame,
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    'Batch compute CDTC for all model outputs and merge expert scores.'
    config = config or {}
    expert_by_case = _expert_lookup(expert_trajectories)
    scores_by_output = _score_lookup(scores)
    rows: List[dict] = []

    for model_trajectory in model_trajectories:
        output_id = str(model_trajectory.output_id)
        key = (model_trajectory.case_id, model_trajectory.disease_task or "")
        if key not in expert_by_case:
            message = f"No expert trajectory for case_id={key[0]}, disease_task={key[1]}"
            LOGGER.error(message)
            raise KeyError(message)
        if output_id not in scores_by_output:
            message = f"No expert score for output_id={output_id}"
            LOGGER.error(message)
            raise KeyError(message)

        score_row = scores_by_output[output_id]
        cdtc_result = compute_cdtc(
            model_trajectory,
            expert_by_case[key],
            alpha=float(config.get("cdtc", {}).get("primary_alpha", 0.5)),
            config=config,
        )
        s_node = cdtc_result["S_node"]
        s_graph = cdtc_result["S_graph"]
        result = CDTCResult(
            case_id=model_trajectory.case_id,
            output_id=output_id,
            model_name=model_trajectory.model_name or str(score_row.get("model_name", "")),
            configuration=model_trajectory.configuration or str(score_row.get("configuration", "")),
            disease_task=model_trajectory.disease_task or str(score_row.get("disease_task", "")),
            redline_status=normalize_redline_status(score_row.get("redline_status")),
            expert_overall_score=float(score_row.get("expert_overall_score")),
            expert_diagnosis_score=float(score_row.get("expert_diagnosis_score")),
            expert_treatment_score=float(score_row.get("expert_treatment_score")),
            expert_disease_course_score=float(score_row.get("expert_disease_course_score")),
            S_node=s_node,
            S_graph=s_graph,
            CDTC_alpha_0_3=_alpha_score(s_node, s_graph, 0.3),
            CDTC_alpha_0_5=_alpha_score(s_node, s_graph, 0.5),
            CDTC_alpha_0_7=_alpha_score(s_node, s_graph, 0.7),
            matched_node_count=len(cdtc_result["matched_nodes"]),
            unmatched_model_node_count=len(cdtc_result["unmatched_model_nodes"]),
            unmatched_expert_node_count=len(cdtc_result["unmatched_expert_nodes"]),
            ged_value=float(cdtc_result["ged_value"]),
            ged_method=cdtc_result["ged_method"],
        )
        row = result.to_dict()
        for column in ["CDTC_alpha_0.3", "CDTC_alpha_0.5", "CDTC_alpha_0.7"]:
            validate_unit_interval(float(row[column]), column)
        rows.append(row)

    return pd.DataFrame(rows)
