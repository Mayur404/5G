from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from numbers import Integral


@dataclass(frozen=True)
class SliceProfile:
    name: str
    demand_mean: float
    demand_std: float
    sla_rbs: int
    latency_max_ms: float
    priority: int
    color: str


@dataclass
class SimConfig:
    seed: int = 7
    n_ues: int = 32
    cell_capacity: int = 60
    world_size: float = 100.0
    dt_s: float = 0.1
    horizon: int = 8
    episode_steps: int = 180
    mobility_noise: float = 0.10
    prediction_noise: float = 0.18
    urllc_burst_prob: float = 0.08
    reward_success: float = 2.0
    reward_qos: float = 3.0
    reward_waste: float = 1.5
    reward_occupancy: float = 0.25
    centers: tuple[tuple[float, float], ...] | None = None
    hex_radius: float | None = None
    handover_mode: str = "nearest"
    switch_margin: float = 0.06
    time_to_trigger: int = 2
    signal_radius: float = 48.0
    minimum_signal: float = 0.35
    common_radio_stream: bool = False
    prediction_mode: str = "synthetic"
    research_metrics: bool = False
    slices: tuple[SliceProfile, ...] = field(
        default_factory=lambda: (
            SliceProfile("eMBB", 10.5, 3.0, 7, 20.0, 2, "#2878B5"),
            SliceProfile("uRLLC", 7.0, 2.0, 5, 5.0, 3, "#D9534F"),
            SliceProfile("mMTC", 3.2, 1.2, 2, 100.0, 1, "#45A36A"),
        )
    )

    def __post_init__(self) -> None:
        if self.handover_mode not in {"nearest", "qos_aware"}:
            raise ValueError("handover_mode must be nearest or qos_aware")
        if self.prediction_mode not in {"synthetic", "kinematic"}:
            raise ValueError("prediction_mode must be synthetic or kinematic")
        if not math.isfinite(self.signal_radius) or self.signal_radius <= 0:
            raise ValueError("signal_radius must be finite and positive")
        if not 0 <= self.minimum_signal <= 1 or not 0 <= self.switch_margin <= 1:
            raise ValueError("signal thresholds must be between zero and one")
        if (
            isinstance(self.time_to_trigger, bool)
            or not isinstance(self.time_to_trigger, Integral)
            or self.time_to_trigger < 1
        ):
            raise ValueError("time_to_trigger must be a positive integer")
        # Counts are discrete scheduling resources; accepting fractional values
        # would silently truncate grants in the simulator's integer ledger.
        for name in ("n_ues", "cell_capacity", "horizon", "episode_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("world_size", "dt_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.world_size <= 10:
            raise ValueError("world_size must exceed the 10-unit initialization margin")
        for name in (
            "mobility_noise",
            "reward_success",
            "reward_qos",
            "reward_waste",
            "reward_occupancy",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("prediction_noise", "urllc_burst_prob"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.centers is not None:
            if not self.centers or any(
                len(c) != 2 or any(not math.isfinite(v) for v in c) for c in self.centers
            ):
                raise ValueError("centers must contain finite two-dimensional coordinates")
            if len({tuple(c) for c in self.centers}) != len(self.centers):
                raise ValueError("cell centers must be distinct")
            if any(not 0 <= v <= self.world_size for c in self.centers for v in c):
                raise ValueError("cell centers must lie inside the simulated world")
        elif self.world_size < 75:
            raise ValueError("world_size below 75 requires explicit cell centers")
        if self.hex_radius is not None:
            radius = self.hex_radius
            if not math.isfinite(radius) or radius <= 0 or self.centers is None:
                raise ValueError("hex_radius must be positive and requires explicit centers")
            apothem = math.sqrt(3) * radius / 2
            origin_x, origin_y = self.centers[0]
            for x, y in self.centers:
                if (
                    x - radius < 0
                    or x + radius > self.world_size
                    or y - apothem < 0
                    or y + apothem > self.world_size
                ):
                    raise ValueError("hexagonal cells must fit inside the world")
                q = (x - origin_x) / (1.5 * radius)
                r = (y - origin_y) / (math.sqrt(3) * radius) - q / 2
                if abs(q - round(q)) > 1e-7 or abs(r - round(r)) > 1e-7:
                    raise ValueError("hexagonal centers must lie on a shared flat-top lattice")
        if tuple(s.name for s in self.slices) != ("eMBB", "uRLLC", "mMTC"):
            raise ValueError("slices must be eMBB, uRLLC, mMTC in that order")
        for profile in self.slices:
            if not math.isfinite(profile.demand_mean) or profile.demand_mean <= 0:
                raise ValueError("slice demand mean must be finite and positive")
            if not math.isfinite(profile.demand_std) or profile.demand_std < 0:
                raise ValueError("slice demand deviation must be finite and nonnegative")
            if (
                isinstance(profile.sla_rbs, bool)
                or not isinstance(profile.sla_rbs, Integral)
                or profile.sla_rbs <= 0
            ):
                raise ValueError("slice minimum RBs must be a positive integer")
            if not math.isfinite(profile.latency_max_ms) or profile.latency_max_ms <= 0:
                raise ValueError("slice latency limit must be finite and positive")

    @property
    def total_capacity(self) -> int:
        n_cells = len(self.centers) if self.centers is not None else 4
        return n_cells * self.cell_capacity

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    episodes: int = 60
    gamma: float = 0.985
    gae_lambda: float = 0.95
    clip: float = 0.2
    actor_lr: float = 0.002
    critic_lr: float = 0.025
    update_epochs: int = 6
    entropy_coef: float = 0.0005
    imitation_episodes: int = 12
    imitation_epochs: int = 45
    imitation_lr: float = 0.012
    seed: int = 41

    def __post_init__(self) -> None:
        for name in ("episodes", "update_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("imitation_episodes", "imitation_epochs", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("gamma", "gae_lambda"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if not math.isfinite(self.clip) or not 0 < self.clip < 1:
            raise ValueError("clip must be between zero and one, exclusively")
        for name in ("actor_lr", "critic_lr", "imitation_lr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.entropy_coef) or self.entropy_coef < 0:
            raise ValueError("entropy_coef must be finite and nonnegative")
