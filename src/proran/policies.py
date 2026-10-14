from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _legal(preferred: list[int], mask: np.ndarray) -> int:
    return next((a for a in preferred if a < len(mask) and mask[a]), 0)


def no_reservation(state: np.ndarray, mask: np.ndarray, context: dict) -> int:
    return 0


def reactive(state: np.ndarray, mask: np.ndarray, context: dict) -> int:
    pred = context.get("prediction")
    return _legal([2, 1], mask) if pred is not None and pred.time_to_handover <= 1 else 0


def certainty_equivalent(state: np.ndarray, mask: np.ndarray, context: dict) -> int:
    pred = context.get("prediction")
    if pred is None or pred.probability < 0.42:
        return 0
    return _legal([2, 1], mask)  # Full predicted demand, immediately.


def uncertainty_aware(state: np.ndarray, mask: np.ndarray, context: dict) -> int:
    pred = context.get("prediction")
    if pred is None:
        return 0
    reliability = (
        pred.probability
        * (1.0 - pred.target_uncertainty)
        * (1.0 - min(1.0, 4 * pred.demand_uncertainty))
    )
    if reliability < 0.46:
        return 0
    if pred.target_uncertainty > 0.18:
        return _legal([3, 5, 1], mask)  # Half amount, delayed activation.
    return _legal([4, 6, 2], mask)  # Full amount, midpoint activation.


def oracle(state: np.ndarray, mask: np.ndarray, context: dict) -> int:
    pred = context.get("prediction")
    if pred is None or pred.target != context.get("actual_target"):
        return 0
    return _legal([2, 1, 4, 3, 6, 5], mask)


BASELINES = {
    "No reservation": no_reservation,
    "Reactive": reactive,
    "Certainty-equivalent": certainty_equivalent,
    "Uncertainty-aware": uncertainty_aware,
    "Projection reference": oracle,
}


@dataclass
class SavedPPOPolicy:
    weights: np.ndarray
    bias: np.ndarray
    deterministic: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.weights.ndim != 2 or self.bias.shape != (self.weights.shape[1],):
            raise ValueError("PPO weights must be a matrix with one bias per action")
        if not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all():
            raise ValueError("PPO parameters must be finite")
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _probabilities(
        state: np.ndarray, mask: np.ndarray, weights: np.ndarray, bias: np.ndarray
    ) -> np.ndarray:
        logits = state @ weights + bias
        if not np.any(mask):
            raise ValueError("At least one action must be legal")
        logits = np.where(mask, logits, -np.inf)
        logits -= np.max(logits)
        exp = np.exp(logits) * mask
        return exp / exp.sum()

    def __call__(self, state: np.ndarray, mask: np.ndarray, context: dict) -> int:
        probs = self._probabilities(state, mask, self.weights, self.bias)
        if self.deterministic:
            return int(np.argmax(probs))
        return int(self.rng.choice(len(probs), p=probs))

    @classmethod
    def load(cls, path: str, deterministic: bool = True) -> "SavedPPOPolicy":
        with np.load(path) as data:
            return cls(data["actor_w"].copy(), data["actor_b"].copy(), deterministic=deterministic)
