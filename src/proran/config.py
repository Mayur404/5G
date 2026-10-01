from __future__ import annotations

from dataclasses import asdict, dataclass, field


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
    slices: tuple[SliceProfile, ...] = field(default_factory=lambda: (
        SliceProfile("eMBB", 10.5, 3.0, 7, 20.0, 2, "#2878B5"),
        SliceProfile("uRLLC", 7.0, 2.0, 5, 5.0, 3, "#D9534F"),
        SliceProfile("mMTC", 3.2, 1.2, 2, 100.0, 1, "#45A36A"),
    ))

    @property
    def total_capacity(self) -> int:
        return 4 * self.cell_capacity

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
