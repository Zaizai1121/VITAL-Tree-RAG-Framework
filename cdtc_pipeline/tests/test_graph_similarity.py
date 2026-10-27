from src.graph_builder import trajectory_to_digraph
from src.graph_similarity import compute_S_graph
from src.node_matching import match_nodes
from src.schema import ClinicalEdge, ClinicalNode, ClinicalTrajectory


def _trajectory(prefix: str) -> ClinicalTrajectory:
    return ClinicalTrajectory(
        case_id="P001",
        disease_task="Sepsis",
        nodes=[
            ClinicalNode(node_id=f"{prefix}1", node_type="test", label="PCT increased", normalized_label="pct_high"),
            ClinicalNode(node_id=f"{prefix}2", node_type="diagnosis", label="Sepsis", normalized_label="sepsis"),
        ],
        edges=[ClinicalEdge(source=f"{prefix}1", target=f"{prefix}2", relation="supportive")],
    )


def test_identical_graph_similarity_is_one():
    model = _trajectory("M")
    expert = _trajectory("E")
    matching = match_nodes(model.nodes, expert.nodes, mode="exact")
    result = compute_S_graph(
        trajectory_to_digraph(model),
        trajectory_to_digraph(expert),
        node_matching=matching,
        exact=False,
    )
    assert result["S_graph"] == 1.0

