from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class UE:
    uid: int
    slice_id: int
    pos: np.ndarray
    vel: np.ndarray
    cell: int
    demand: int = 1


@dataclass
class Prediction:
    ue_id: int
    source: int
    target: int
    probability: float
    time_to_handover: int
    demand_rbs: int
    target_uncertainty: float
    demand_uncertainty: float
    actual_target_hint: int


@dataclass
class Reservation:
    rid: int
    ue_id: int
    target: int
    amount: int
    created_at: int
    activate_at: int
    expires_at: int
    predicted_target: int
    active: bool = False
    activated_amount: int = 0


@dataclass
class StepResult:
    state: np.ndarray
    mask: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool
    info: dict


@dataclass
class RunMetrics:
    steps: int = 0
    handovers: int = 0
    successes: int = 0
    qos_successes: int = 0
    activated_rbs: int = 0
    used_reserved_rbs: int = 0
    wrong_target_contracts: int = 0
    completed_contracts: int = 0
    active_coverage: int = 0
    activation_attempts: int = 0
    activation_rejects: int = 0
    blocked_requests: int = 0
    admission_requests: int = 0
    occupancy_rb_intervals: int = 0
    allocated_rb_intervals: int = 0
    capacity_rb_intervals: int = 0
    reward_sum: float = 0.0
    latency_ms: list[float] | None = None
    latency_slice_ids: list[int] | None = None
    handovers_by_slice: list[int] | None = None
    successes_by_slice: list[int] | None = None
    qos_by_slice: list[int] | None = None
    lead_intervals: list[int] | None = None
    service_ue_intervals: int = 0
    full_service_intervals: int = 0
    sla_service_intervals: int = 0
    offered_rbs: int = 0
    served_rbs: int = 0
    pingpong_handovers: int = 0
    service_ue_intervals_by_slice: list[int] = field(default_factory=lambda: [0, 0, 0])
    full_service_intervals_by_slice: list[int] = field(default_factory=lambda: [0, 0, 0])
    sla_service_intervals_by_slice: list[int] = field(default_factory=lambda: [0, 0, 0])
    offered_rbs_by_slice: list[int] = field(default_factory=lambda: [0, 0, 0])
    served_rbs_by_slice: list[int] = field(default_factory=lambda: [0, 0, 0])
    offered_rbs_by_ue: dict[int, int] = field(default_factory=dict)
    served_rbs_by_ue: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.latency_ms = [] if self.latency_ms is None else self.latency_ms
        self.latency_slice_ids = [] if self.latency_slice_ids is None else self.latency_slice_ids
        self.handovers_by_slice = (
            [0, 0, 0] if self.handovers_by_slice is None else self.handovers_by_slice
        )
        self.successes_by_slice = (
            [0, 0, 0] if self.successes_by_slice is None else self.successes_by_slice
        )
        self.qos_by_slice = [0, 0, 0] if self.qos_by_slice is None else self.qos_by_slice
        self.lead_intervals = [] if self.lead_intervals is None else self.lead_intervals

    @staticmethod
    def _ratio(a: float, b: float) -> float:
        return float(a / b) if b else float("nan")

    def summary(self, dt_s: float) -> dict[str, float]:
        waste = self.activated_rbs - self.used_reserved_rbs
        latency = np.asarray(self.latency_ms, dtype=float)
        result = {
            "handover_success_rate": self._ratio(self.successes, self.handovers),
            "qos_rate": self._ratio(self.qos_successes, self.handovers),
            "unused_reservation_fraction": self._ratio(waste, self.activated_rbs),
            "reservation_occupancy": self._ratio(
                self.occupancy_rb_intervals, self.capacity_rb_intervals
            ),
            "rb_utilization": self._ratio(self.allocated_rb_intervals, self.capacity_rb_intervals),
            "wrong_target_rate": self._ratio(self.wrong_target_contracts, self.completed_contracts),
            "active_reservation_coverage": self._ratio(self.active_coverage, self.handovers),
            "activation_reject_rate": self._ratio(
                self.activation_rejects, self.activation_attempts
            ),
            "block_rate": self._ratio(self.blocked_requests, self.admission_requests),
            "mean_reservation_lead_s": float(np.mean(self.lead_intervals) * dt_s)
            if self.lead_intervals
            else float("nan"),
            "mean_latency_ms": float(np.mean(latency)) if latency.size else float("nan"),
            "p95_latency_ms": float(np.quantile(latency, 0.95)) if latency.size else float("nan"),
            "mean_step_reward": self._ratio(self.reward_sum, self.steps),
        }
        for sid, name in enumerate(("embb", "urllc", "mmtc")):
            result[f"{name}_handover_success_rate"] = self._ratio(
                self.successes_by_slice[sid], self.handovers_by_slice[sid]
            )
            result[f"{name}_qos_rate"] = self._ratio(
                self.qos_by_slice[sid], self.handovers_by_slice[sid]
            )
        if self.service_ue_intervals:
            result.update(
                {
                    "full_service_rate": self.full_service_intervals / self.service_ue_intervals,
                    "sla_service_rate": self.sla_service_intervals / self.service_ue_intervals,
                    "served_demand_fraction": self._ratio(self.served_rbs, self.offered_rbs),
                    "handovers_per_ue_second": self.handovers / (self.service_ue_intervals * dt_s),
                    "pingpong_fraction": self._ratio(self.pingpong_handovers, self.handovers),
                }
            )
            for sid, name in enumerate(("embb", "urllc", "mmtc")):
                result[f"{name}_full_service_rate"] = self._ratio(
                    self.full_service_intervals_by_slice[sid],
                    self.service_ue_intervals_by_slice[sid],
                )
                result[f"{name}_sla_service_rate"] = self._ratio(
                    self.sla_service_intervals_by_slice[sid],
                    self.service_ue_intervals_by_slice[sid],
                )
                result[f"{name}_served_demand_fraction"] = self._ratio(
                    self.served_rbs_by_slice[sid], self.offered_rbs_by_slice[sid]
                )
            fractions = np.array(
                [
                    self.served_rbs_by_ue.get(uid, 0) / offered
                    for uid, offered in self.offered_rbs_by_ue.items()
                    if offered > 0
                ]
            )
            result["jain_served_demand_fairness"] = self._ratio(
                float(fractions.sum() ** 2), float(len(fractions) * np.sum(fractions**2))
            )
        return result
