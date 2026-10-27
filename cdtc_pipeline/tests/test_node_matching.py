from src.node_matching import match_nodes
from src.schema import ClinicalNode


def test_exact_node_matching_with_time_window():
    model_nodes = [
        ClinicalNode(
            node_id="M1",
            node_type="diagnosis",
            relative_day=1,
            label="Sepsis",
            normalized_label="sepsis",
        )
    ]
    expert_nodes = [
        ClinicalNode(
            node_id="E1",
            node_type="diagnosis",
            relative_day=0,
            label="Sepsis suspected",
            normalized_label="sepsis",
        )
    ]
    result = match_nodes(model_nodes, expert_nodes, mode="exact", time_window=1)
    assert result["S_node"] == 1.0
    assert len(result["matched_pairs"]) == 1


def test_node_matching_prevents_duplicate_matches():
    model_nodes = [
        ClinicalNode(node_id="M1", node_type="test", label="PCT high", normalized_label="procalcitonin_elevated"),
        ClinicalNode(node_id="M2", node_type="test", label="PCT high again", normalized_label="procalcitonin_elevated"),
    ]
    expert_nodes = [
        ClinicalNode(node_id="E1", node_type="test", label="PCT increased", normalized_label="procalcitonin_elevated")
    ]
    result = match_nodes(model_nodes, expert_nodes, mode="exact")
    assert len(result["matched_pairs"]) == 1
    assert len(result["unmatched_model_nodes"]) == 1

