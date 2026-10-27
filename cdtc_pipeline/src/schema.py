'Data structure definition and basic verification.'

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


VALID_NODE_TYPES = {"diagnosis", "test", "treatment", "outcome"}
VALID_RELATIONS = {
    "temporal",
    "causal",
    "supportive",
    "response_to_treatment",
    "uncertainty",
}


class ClinicalNode(BaseModel):
    'Clinical trajectory nodes.'

    node_id: str
    node_type: str
    time: Optional[str] = None
    relative_day: Optional[float] = None
    label: str
    normalized_label: str
    description: Optional[str] = ""

    @field_validator("node_type")
    @classmethod
    def validate_node_type(cls, value: str) -> str:
        'Check node type.'
        if value not in VALID_NODE_TYPES:
            raise ValueError(f"node_type must be one of {sorted(VALID_NODE_TYPES)}")
        return value

    @field_validator("node_id", "label", "normalized_label")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        'Verify that the required string is not empty.'
        if value is None or str(value).strip() == "":
            raise ValueError("required string field cannot be empty")
        return str(value).strip()


class ClinicalEdge(BaseModel):
    'Clinical trajectory edge.'

    source: str
    target: str
    relation: str
    description: Optional[str] = ""

    @field_validator("relation")
    @classmethod
    def validate_relation(cls, value: str) -> str:
        'Verify edge relationship type.'
        if value not in VALID_RELATIONS:
            raise ValueError(f"relation must be one of {sorted(VALID_RELATIONS)}")
        return value


class ClinicalTrajectory(BaseModel):
    'Case-level clinical decision trajectories.'

    case_id: str
    disease_task: Optional[str] = ""
    nodes: List[ClinicalNode] = Field(default_factory=list)
    edges: List[ClinicalEdge] = Field(default_factory=list)
    output_id: Optional[str] = None
    model_name: Optional[str] = None
    configuration: Optional[str] = None
    is_model_trajectory: bool = False

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        'Both expert and model trajectories must contain case_id.'
        if value is None or str(value).strip() == "":
            raise ValueError("case_id cannot be empty")
        return str(value).strip()

    @model_validator(mode="after")
    def validate_trajectory(self) -> "ClinicalTrajectory":
        'Verify edge endpoints, model trajectory output_id and configuration fields.'
        node_ids = {node.node_id for node in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(
                    f"edge source/target must exist in nodes: {edge.source}->{edge.target}"
                )
        if self.is_model_trajectory:
            if not self.output_id:
                raise ValueError("model trajectory must have output_id")
            if not self.configuration or str(self.configuration).strip() == "":
                raise ValueError("model trajectory configuration cannot be empty")
        return self


@dataclass
class CDTCResult:
    'CDTC results for a single model output.'

    case_id: str
    output_id: str
    model_name: str
    configuration: str
    disease_task: str
    redline_status: Any
    expert_overall_score: float
    expert_diagnosis_score: float
    expert_treatment_score: float
    expert_disease_course_score: float
    S_node: float
    S_graph: float
    CDTC_alpha_0_3: float
    CDTC_alpha_0_5: float
    CDTC_alpha_0_7: float
    matched_node_count: int
    unmatched_model_node_count: int
    unmatched_expert_node_count: int
    ged_value: float
    ged_method: str

    def to_dict(self) -> Dict[str, Any]:
        'Convert to a dictionary suitable for writing to CSV.'
        data = self.__dict__.copy()
        data["CDTC_alpha_0.3"] = data.pop("CDTC_alpha_0_3")
        data["CDTC_alpha_0.5"] = data.pop("CDTC_alpha_0_5")
        data["CDTC_alpha_0.7"] = data.pop("CDTC_alpha_0_7")
        return data

