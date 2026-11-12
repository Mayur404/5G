"""Simple reference policies for the single-decision (v1) action interface."""

from __future__ import annotations

import numpy as np


def no_reservation(state: np.ndarray, mask: np.ndarray, context: dict) -> int:
    return 0


def certainty_equivalent(state: np.ndarray, mask: np.ndarray, context: dict) -> int:
    """Reserve the full forecast demand immediately when the forecast score is high."""
    pred = context.get("prediction")
    if pred is None or pred.probability < 0.42:
        return 0
    return next((a for a in (2, 1) if a < len(mask) and mask[a]), 0)
