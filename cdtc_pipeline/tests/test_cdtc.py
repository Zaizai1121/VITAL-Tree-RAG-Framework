import pandas as pd

from src.cdtc import compute_cdtc_for_all
from src.schema import ClinicalEdge, ClinicalNode, ClinicalTrajectory


def test_compute_cdtc_for_all_returns_expected_columns():
    expert = ClinicalTrajectory(
        case_id="P001",
        disease_task="Sepsis",
        nodes=[
            ClinicalNode(node_id="E1", node_type="test", label="PCT increased", normalized_label="pct_high"),
            ClinicalNode(node_id="E2", node_type="diagnosis", label="Sepsis", normalized_label="sepsis"),
        ],
        edges=[ClinicalEdge(source="E1", target="E2", relation="supportive")],
    )
    model = ClinicalTrajectory(
        case_id="P001",
        output_id="O001",
        model_name="Test model",
        configuration="Full VITAL-Tree RAG",
        disease_task="Sepsis",
        is_model_trajectory=True,
        nodes=[
            ClinicalNode(node_id="M1", node_type="test", label="PCT increased", normalized_label="pct_high"),
            ClinicalNode(node_id="M2", node_type="diagnosis", label="Sepsis", normalized_label="sepsis"),
        ],
        edges=[ClinicalEdge(source="M1", target="M2", relation="supportive")],
    )
    scores = pd.DataFrame(
        [
            {
                "output_id": "O001",
                "case_id": "P001",
                "model_name": "Test model",
                "configuration": "Full VITAL-Tree RAG",
                "disease_task": "Sepsis",
                "expert_overall_score": 5,
                "expert_diagnosis_score": 5,
                "expert_treatment_score": 5,
                "expert_disease_course_score": 5,
                "redline_status": "no",
            }
        ]
    )
    config = {"matching": {"mode": "exact"}, "graph": {"exact_ged": False}}
    results = compute_cdtc_for_all([model], [expert], scores, config=config)
    assert results.loc[0, "CDTC_alpha_0.5"] == 1.0
    assert results.loc[0, "redline_status"] == 0

