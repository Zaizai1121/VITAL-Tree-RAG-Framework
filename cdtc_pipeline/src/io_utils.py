'Input and output utility functions.'

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yaml

from .schema import ClinicalTrajectory


def ensure_dir(path: str | Path) -> Path:
    'Ensures the directory exists and returns a Path object.'
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def setup_logging(output_dir: str | Path) -> logging.Logger:
    'Configure logging output to both the console and a file.'
    ensure_dir(output_dir)
    log_path = Path(output_dir) / "pipeline.log"
    logger = logging.getLogger("cdtc_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_config(config_path: str | Path) -> dict:
    'Read the YAML configuration file.'
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path: str | Path) -> List[dict]:
    'Read JSONL file.'
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    'Write to JSONL file.'
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_trajectories(path: str | Path, is_model: bool = False) -> List[ClinicalTrajectory]:
    'Read and verify the trajectory JSONL.'
    trajectories: List[ClinicalTrajectory] = []
    for record in read_jsonl(path):
        record = dict(record)
        record["is_model_trajectory"] = is_model
        trajectories.append(ClinicalTrajectory(**record))
    return trajectories


def load_expert_scores(path: str | Path) -> pd.DataFrame:
    'Read expert ratings CSV.'
    scores = pd.read_csv(path)
    required_cols = {
        "output_id",
        "case_id",
        "model_name",
        "configuration",
        "disease_task",
        "expert_overall_score",
        "expert_diagnosis_score",
        "expert_treatment_score",
        "expert_disease_course_score",
        "redline_status",
    }
    missing = required_cols.difference(scores.columns)
    if missing:
        raise ValueError(f"expert_scores.csv missing required columns: {sorted(missing)}")
    return scores


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    'Save DataFrame to CSV.'
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)


def find_input_file(input_dir: str | Path, canonical_name: str) -> Path:
    'The standard file name is read first, and if it does not exist, the simulated_ prefix file is read.'
    input_dir = Path(input_dir)
    direct = input_dir / canonical_name
    if direct.exists():
        return direct
    simulated = input_dir / f"simulated_{canonical_name}"
    if simulated.exists():
        return simulated
    raise FileNotFoundError(f"Cannot find {canonical_name} or simulated_{canonical_name} in {input_dir}")


def index_expert_trajectories(
    expert_trajectories: List[ClinicalTrajectory],
) -> Dict[Tuple[str, str], ClinicalTrajectory]:
    'Build expert trajectory index by case_id and disease_task.'
    indexed: Dict[Tuple[str, str], ClinicalTrajectory] = {}
    for trajectory in expert_trajectories:
        key = (trajectory.case_id, trajectory.disease_task or "")
        indexed[key] = trajectory
    return indexed
