'Quality control and field standardization.'

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger("cdtc_pipeline")


def normalize_redline_status(value: Any) -> int:
    'Normalize redline status to 0/1.'
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in {0, 1}:
        return int(value)
    text = str(value).strip().lower()
    mapping = {
        "0": 0,
        "1": 1,
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "non-redline": 0,
        "redline": 1,
    }
    if text not in mapping:
        raise ValueError(f"Invalid redline_status: {value}")
    return mapping[text]


def validate_alpha(alpha: float) -> None:
    'Check alpha range.'
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0, 1]")


def validate_unit_interval(value: float, name: str) -> None:
    'The check index is located at [0, 1].'
    if value < -1e-9 or value > 1 + 1e-9:
        raise ValueError(f"{name} out of [0, 1]: {value}")

