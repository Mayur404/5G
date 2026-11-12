"""Calibrated handover-target confidence for forecast-driven reservation.

The neural forecaster returns trajectories, not probabilities. This module
learns P(next cell change within the horizon goes to the forecast target) from
observable forecast descriptors with L2-regularized logistic regression. It is
fit on dedicated calibration trajectories and scored on held-out ones.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import SimConfig
from .simulator import RANHandoverEnv

FEATURE_NAMES = (
    "lead",
    "signal_margin",
    "serving_signal",
    "speed",
    "heading",
    "bisector_gap",
    "closing_speed",
)


def expand(features: np.ndarray) -> np.ndarray:
    """Fixed basis: raw descriptors, two interactions, squared lead and an intercept."""
    f = np.atleast_2d(np.asarray(features, dtype=float))
    lead, margin, gap, closing = f[:, 0], f[:, 1], f[:, 5], f[:, 6]
    return np.column_stack([f, lead * margin, gap * closing, lead**2, np.ones(len(f))])


class LogisticCalibrator:
    def __init__(self, weights: np.ndarray | None = None, l2: float = 1e-3):
        self.weights = weights
        self.l2 = l2

    def fit(self, features: np.ndarray, labels: np.ndarray, iterations: int = 50) -> None:
        x = expand(features)
        y = np.asarray(labels, dtype=float)
        w = np.zeros(x.shape[1])
        penalty = self.l2 * len(x) * np.eye(len(w))
        penalty[-1, -1] = 0.0  # Do not shrink the intercept.
        for _ in range(iterations):  # Newton/IRLS with a ridge penalty.
            p = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
            gradient = x.T @ (p - y) + penalty @ w
            hessian = (x * (p * (1 - p))[:, None]).T @ x + penalty
            step = np.linalg.solve(hessian, gradient)
            w -= step
            if np.max(np.abs(step)) < 1e-9:
                break
        self.weights = w

    def predict(self, features) -> np.ndarray:
        if self.weights is None:
            raise ValueError("calibrator is not fitted")
        z = expand(features) @ self.weights
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def __call__(self, features) -> float:
        return float(self.predict(features)[0])

    def save(self, path: str | Path, metadata: dict) -> None:
        np.savez(path, weights=self.weights, metadata=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls, path: str | Path) -> LogisticCalibrator:
        with np.load(path, allow_pickle=False) as data:
            weights = data["weights"].copy()
        if weights.shape != (len(FEATURE_NAMES) + 4,) or not np.isfinite(weights).all():
            raise ValueError("calibrator weights are invalid")
        return cls(weights)


def collect(cfg: SimConfig, seeds, predictor, steps: int = 320):
    """Forecast descriptors and realized outcomes under a no-reservation policy.

    Label 1 means the UE's next cell change occurs within the forecast horizon
    and reaches the forecast target. Labels use only the run's own future.
    """
    features, labels, leads, raw_scores = [], [], [], []
    for seed in seeds:
        env = RANHandoverEnv(replace(cfg, seed=seed, episode_steps=steps, max_decisions=10**6))
        env.neural_predictor = predictor
        env.confidence_model = None
        env.reset(seed)
        pending = []  # (t, ue, target, lead, features, raw score)
        moves: dict[int, list[tuple[int, int]]] = {}
        for _ in range(steps):
            for p in env.predictions():
                pending.append(
                    (env.t, p.ue_id, p.target, p.time_to_handover, p.features, p.probability)
                )
            observations = env.observe_many()
            _, _, info = env.step_many([0] * len(observations), observations)
            for event in info["handover_details"]:
                moves.setdefault(event["ue"], []).append((env.t - 1, event["target"]))
        horizon = cfg.horizon
        for t, ue, target, lead, feat, score in pending:
            if t + horizon > steps:
                continue
            following = [m for m in moves.get(ue, []) if t <= m[0] < t + horizon]
            labels.append(int(bool(following) and following[0][1] == target))
            features.append(feat)
            leads.append(lead)
            raw_scores.append(score)
    return np.array(features), np.array(labels), np.array(leads), np.array(raw_scores)


def reliability(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        selected = index == b
        if not selected.any():
            continue
        confidence, accuracy = float(p[selected].mean()), float(y[selected].mean())
        ece += selected.mean() * abs(confidence - accuracy)
        rows.append(
            {"bin": b, "count": int(selected.sum()), "confidence": confidence, "accuracy": accuracy}
        )
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    positives = y.sum()
    negatives = len(y) - positives
    auc = (
        float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
        if positives and negatives
        else float("nan")
    )
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(ece),
        "auc": auc,
        "base_rate": float(y.mean()),
        "bins": rows,
    }
