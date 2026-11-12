from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class ActionSpec:
    amount_fraction: float
    timing: str


# Action 0 is always the legal no-op. Positive actions jointly choose amount/timing.
ACTION_CATALOG = (
    ActionSpec(0.0, "none"),
    ActionSpec(0.5, "now"),
    ActionSpec(1.0, "now"),
    ActionSpec(0.5, "mid"),
    ActionSpec(1.0, "mid"),
    ActionSpec(0.5, "jit"),
    ActionSpec(1.0, "jit"),
)


def decode_action(action: int, predicted_demand: int, time_to_handover: int) -> tuple[int, int]:
    if (
        isinstance(action, bool)
        or not isinstance(action, Integral)
        or not 0 <= action < len(ACTION_CATALOG)
    ):
        raise ValueError("action must be an integer in the action catalog")
    spec = ACTION_CATALOG[action]
    if action == 0:
        return 0, 0
    if (
        isinstance(predicted_demand, bool)
        or not isinstance(predicted_demand, Integral)
        or predicted_demand <= 0
    ):
        raise ValueError("positive reservation requires positive integer predicted demand")
    if (
        isinstance(time_to_handover, bool)
        or not isinstance(time_to_handover, Integral)
        or time_to_handover < 1
    ):
        raise ValueError("positive reservation requires a future integer arrival interval")
    amount = max(1, min(predicted_demand, round(spec.amount_fraction * predicted_demand)))
    if spec.timing == "now":
        delay = 0
    elif spec.timing == "mid":
        delay = max(0, time_to_handover // 2)
    else:
        delay = max(0, time_to_handover - 1)
    return amount, delay


@dataclass(frozen=True)
class ActionSpecV2:
    amount: str  # "none", "floor", "demand" or "upper"
    timing: str


# The v2 catalog adds the slice admission floor and an upper demand quantile.
# Holding only the floor protects admission at a lower displacement cost.
ACTION_CATALOG_V2 = (
    ActionSpecV2("none", "none"),
    ActionSpecV2("floor", "now"),
    ActionSpecV2("floor", "mid"),
    ActionSpecV2("floor", "jit"),
    ActionSpecV2("demand", "now"),
    ActionSpecV2("demand", "mid"),
    ActionSpecV2("demand", "jit"),
    ActionSpecV2("upper", "now"),
    ActionSpecV2("upper", "mid"),
    ActionSpecV2("upper", "jit"),
)


def decode_action_v2(
    action: int, predicted_demand: int, time_to_handover: int, floor_rbs: int, upper_demand: int
) -> tuple[int, int]:
    if (
        isinstance(action, bool)
        or not isinstance(action, Integral)
        or not 0 <= action < len(ACTION_CATALOG_V2)
    ):
        raise ValueError("action must be an integer in the v2 action catalog")
    if action == 0:
        return 0, 0
    if predicted_demand <= 0 or time_to_handover < 1 or floor_rbs <= 0:
        raise ValueError("positive reservation requires positive demand, floor and lead time")
    spec = ACTION_CATALOG_V2[action]
    amount = {
        "floor": min(floor_rbs, max(predicted_demand, upper_demand)),
        "demand": predicted_demand,
        "upper": max(predicted_demand, upper_demand),
    }[spec.amount]
    delay = {"now": 0, "mid": time_to_handover // 2, "jit": time_to_handover - 1}[spec.timing]
    return max(1, int(amount)), max(0, int(delay))
