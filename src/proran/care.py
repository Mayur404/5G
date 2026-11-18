"""CARE: Calibrated, Attribution-Rewarded rEservation.

For every forecast handover, CARE decides whether to hold RBs at the forecast
target, how many (admission floor, forecast demand or an upper demand
quantile) and when to activate them. Each committed hold is credited with its
own contract outcome:

    R = w_s * [(1 + outage) * demand * rescued + qos_value * qos_rescued]
        - lambda * displaced,

where ``rescued`` is the simulator's one-interval counterfactual (the hold
turned a failed admission into a successful one) and ``displaced`` counts the
incumbent RB-intervals the hold actually withheld from unmet demand. Holding
idle capacity therefore costs nothing, while holding contested capacity is
charged at the operator's exchange rate ``lambda``. Counterfactual quantities
are training signals only; the policy observes causal state.

The final learner is fitted Q iteration with optimal-stopping wait targets
(``train_care_fqi``), deployed as a greedy policy over the mean of several
independently trained Q-networks (``CareEnsemblePolicy``). Masked categorical
PPO (``CareAgent``) is kept as the ablation reported in the paper.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .config import SimConfig
from .simulator import RANHandoverEnv


@dataclass
class CareConfig:
    episodes: int = 120
    episodes_per_update: int = 4
    steps: int = 320
    hidden: tuple[int, int] = (64, 64)
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    update_epochs: int = 6
    minibatch: int = 512
    clip: float = 0.2
    entropy_coef: float = 0.01
    entropy_final: float = 0.001
    displacement_weight: float = 1.0
    qos_value: float = 4.0
    slice_weights: tuple[float, float, float] = (1.0, 2.0, 0.5)
    reward_scale: float = 32.0
    wait_discount: float = 0.98
    # "stopping": a wait is worth the discounted value of the best continuation,
    # which is never below the always-available "never hold" outcome (0).
    # "on_policy": a wait inherits the sampled next decision's return.
    wait_backup: str = "stopping"
    # "fqi": fitted Q iteration with optimal-stopping targets (CARE);
    # "ppo": masked categorical PPO on per-decision returns (ablation).
    learner: str = "fqi"
    q_lr: float = 1e-3
    fqi_iterations: int = 2
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_fraction: float = 0.75
    seed: int = 101


def decision_returns(
    actions, immediate, following, discount: float, backup: str = "stopping"
) -> np.ndarray:
    """Per-decision returns along each UE's decision chain.

    ``immediate[i]`` is the contract reward of a committed decision (ignored for
    waits); ``following[i]`` is the index of the same UE's next decision for the
    same crossing, or None. A wait with no following decision ends without a
    contract, whose reward is 0 by definition of the contract reward.
    """
    if backup not in {"stopping", "on_policy"}:
        raise ValueError("backup must be stopping or on_policy")
    returns = np.zeros(len(actions))
    for index in range(len(actions) - 1, -1, -1):
        if actions[index]:
            returns[index] = immediate[index]
        elif following[index] is not None:
            nxt = returns[following[index]]
            returns[index] = discount * (max(0.0, nxt) if backup == "stopping" else nxt)
    return returns


class _Adam:
    def __init__(self, params):
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads, lr):
        """Gradient descent on the supplied loss gradients."""
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v, strict=True):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p -= lr * (m / (1 - 0.9**self.t)) / (np.sqrt(v / (1 - 0.999**self.t)) + 1e-8)


class MLP:
    def __init__(self, sizes, rng, out_scale=0.01):
        self.weights, self.biases = [], []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            scale = np.sqrt(1 / a) * (out_scale if i == len(sizes) - 2 else 1.0)
            self.weights.append(rng.normal(0, scale, (a, b)))
            self.biases.append(np.zeros(b))

    @property
    def params(self):
        return [p for pair in zip(self.weights, self.biases, strict=True) for p in pair]

    def forward(self, x, cache=False):
        activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = activations[-1] @ w + b
            activations.append(np.tanh(z) if i < len(self.weights) - 1 else z)
        return (activations[-1], activations) if cache else activations[-1]

    def backward(self, activations, grad_out):
        grads = []
        delta = grad_out
        for i in reversed(range(len(self.weights))):
            grads.append((activations[i].T @ delta, delta.sum(axis=0)))
            if i:
                delta = (delta @ self.weights[i].T) * (1 - activations[i] ** 2)
        return [g for pair in reversed(grads) for g in pair]


def masked_softmax(logits, masks):
    logits = np.where(masks, logits, -np.inf)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits) * masks
    return exp / exp.sum(axis=-1, keepdims=True)


class CareAgent:
    def __init__(self, state_dim, n_actions, cfg: CareConfig | None = None):
        self.cfg = cfg or CareConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        sizes = (state_dim, *self.cfg.hidden)
        self.actor = MLP((*sizes, n_actions), self.rng)
        self.critic = MLP((*sizes, 1), self.rng, out_scale=1.0)
        self.actor_opt = _Adam(self.actor.params)
        self.critic_opt = _Adam(self.critic.params)

    def probs(self, states, masks):
        return masked_softmax(self.actor.forward(states), masks)

    def act(self, state, mask, greedy=False):
        p = self.probs(state[None], mask[None])[0]
        action = int(np.argmax(p)) if greedy else int(self.rng.choice(len(p), p=p))
        return action, float(np.log(p[action] + 1e-12))

    def update(self, states, masks, actions, old_logp, rewards, entropy_coef):
        cfg = self.cfg
        n = len(states)
        stats = []
        for _ in range(cfg.update_epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, cfg.minibatch):
                ids = order[start : start + cfg.minibatch]
                s, m, a = states[ids], masks[ids], actions[ids]
                values, critic_cache = self.critic.forward(s, cache=True)
                values = values[:, 0]
                advantage = rewards[ids] - values
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                logits, actor_cache = self.actor.forward(s, cache=True)
                p = masked_softmax(logits, m)
                k = len(ids)
                ratio = np.exp(np.log(p[np.arange(k), a] + 1e-12) - old_logp[ids])
                unclipped = np.where(
                    advantage >= 0, ratio <= 1 + cfg.clip, ratio >= 1 - cfg.clip
                ).astype(float)
                onehot = np.zeros_like(p)
                onehot[np.arange(k), a] = 1.0
                # d(-surrogate)/dlogits for samples inside the trust region.
                grad = -(advantage * ratio * unclipped)[:, None] * (onehot - p) / k
                logp = np.log(np.where(m, p, 1.0))
                entropy = -np.sum(p * logp, axis=1)
                grad += entropy_coef * p * (logp + entropy[:, None]) / k
                self.actor_opt.step(
                    self.actor.params, self.actor.backward(actor_cache, grad), cfg.actor_lr
                )
                verror = (values - rewards[ids])[:, None] * 2 / k
                self.critic_opt.step(
                    self.critic.params,
                    self.critic.backward(critic_cache, verror),
                    cfg.critic_lr,
                )
                stats.append(
                    (
                        float(np.mean(np.minimum(ratio, np.clip(ratio, 0.8, 1.2)) * advantage)),
                        float(np.mean((values - rewards[ids]) ** 2)),
                        float(entropy.mean()),
                    )
                )
        return np.mean(stats, axis=0)

    def policy(self):
        return CarePolicy(
            [w.copy() for w in self.actor.weights], [b.copy() for b in self.actor.biases]
        )

    def save(self, path, metadata):
        arrays = {f"w{i}": w for i, w in enumerate(self.actor.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.actor.biases)})
        np.savez(path, **arrays, metadata=np.array(json.dumps(metadata)))


class CarePolicy:
    """Greedy policy over one saved CARE network (Q-values or PPO logits)."""

    def __init__(self, weights, biases):
        self.net = MLP((1, 1), np.random.default_rng(0))
        self.net.weights, self.net.biases = weights, biases

    def __call__(self, state, mask, context):
        if mask.sum() <= 1:
            return 0
        logits = self.net.forward(state[None])[0]
        return int(np.argmax(np.where(mask, logits, -np.inf)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            count = sum(1 for key in data.files if key.startswith("w"))
            weights = [data[f"w{i}"].copy() for i in range(count)]
            biases = [data[f"b{i}"].copy() for i in range(count)]
        if not all(np.isfinite(p).all() for p in [*weights, *biases]):
            raise ValueError("CARE parameters must be finite")
        return cls(weights, biases)


class CareEnsemblePolicy:
    """Greedy action over the mean output of independently trained networks.

    For action-value networks, averaging replicas reduces the variance of the
    per-action estimates, and with it the optimistic bias of a greedy argmax.
    """

    def __init__(self, members: list[CarePolicy]):
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = members

    def values(self, state) -> np.ndarray:
        return np.mean([m.net.forward(state[None])[0] for m in self.members], axis=0)

    def __call__(self, state, mask, context):
        if mask.sum() <= 1:
            return 0
        return int(np.argmax(np.where(mask, self.values(state), -np.inf)))

    @classmethod
    def load(cls, paths):
        return cls([CarePolicy.load(path) for path in paths])


def contract_reward(outcome: dict, outage: int, cfg: CareConfig) -> float:
    """Slice-weighted service gained by one contract, net of displaced RBs."""
    weight = cfg.slice_weights[outcome.get("slice_id", 0)]
    rescue = weight * (1 + outage) * outcome["arrival_demand"] * float(outcome["rescued"])
    qos = weight * cfg.qos_value * float(outcome["qos_rescued"])
    return (rescue + qos - cfg.displacement_weight * outcome["displaced"]) / cfg.reward_scale


def collect_episode(sim_cfg: SimConfig, predictor, calibrator, steps: int, seed: int, choose):
    """Run one training episode; ``choose(state, mask) -> (action, extra)``.

    Returns the finished environment and its decisions, each
    ``[state, mask, action, extra, reservation_id, following_index]`` where
    ``following_index`` links a wait to the same UE's next decision for the
    same crossing (only when that decision comes in the very next interval).
    """
    env = RANHandoverEnv(replace(sim_cfg, seed=seed, episode_steps=steps))
    env.neural_predictor = predictor
    env.confidence_model = calibrator
    env.reset(seed)
    decisions: list[list] = []
    open_chain: dict[int, tuple[int, int]] = {}  # UE -> (last wait index, time)
    for _ in range(steps):
        observations = env.observe_many()
        actions, indices = [], []
        for state, mask, context in observations:
            ue = context["prediction"].ue_id
            if mask.sum() > 1:
                action, extra = choose(state, mask)
                index = len(decisions)
                decisions.append([state, mask, action, extra, None, None])
                previous = open_chain.pop(ue, None)
                if previous is not None and previous[1] == env.t - 1:
                    decisions[previous[0]][5] = index
                if action == 0:
                    open_chain[ue] = (index, env.t)
                indices.append(index)
            else:
                action = 0
                if ue in open_chain and open_chain[ue][1] == env.t - 1:
                    open_chain[ue] = (open_chain[ue][0], env.t)  # Pass-through.
                indices.append(None)
            actions.append(action)
        _, _, info = env.step_many(actions, observations)
        for index, rid in zip(indices, info["reservation_ids"], strict=True):
            if index is not None:
                decisions[index][4] = rid
    env.settle_open_contracts()
    return env, decisions


def _immediate_rewards(env, decisions, sim_cfg: SimConfig, cfg: CareConfig) -> list[float]:
    return [
        contract_reward(env.contract_outcomes[d[4]], sim_cfg.failure_outage, cfg) if d[2] else 0.0
        for d in decisions
    ]


def _history_row(episode, env, decisions, rewards, sim_cfg) -> dict:
    summary = env.metrics.summary(sim_cfg.dt_s)
    return {
        "episode": episode + 1,
        "decisions": len(decisions),
        "mean_reward": float(np.mean(rewards)) if len(rewards) else 0.0,
        "reserve_fraction": float(np.mean([d[2] > 0 for d in decisions])) if decisions else 0,
        "handover_success_rate": summary["handover_success_rate"],
        "displaced_rb_fraction": summary.get("displaced_rb_fraction", float("nan")),
        "outage_fraction": summary.get("outage_fraction", float("nan")),
    }


def _write_history(output, history) -> None:
    if output is None:
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "care_training.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_care(
    sim_cfg: SimConfig,
    predictor,
    calibrator,
    cfg: CareConfig,
    output: str | Path | None = None,
    seed_base: int = 20000,
    log=print,
):
    """Train CARE on dedicated training trajectories (never evaluation seeds)."""
    if cfg.learner == "fqi":
        return train_care_fqi(sim_cfg, predictor, calibrator, cfg, output, seed_base, log)
    if cfg.learner != "ppo":
        raise ValueError("learner must be fqi or ppo")
    env = RANHandoverEnv(replace(sim_cfg, episode_steps=cfg.steps))
    agent = CareAgent(env.state_dim, env.n_actions, cfg)
    history = []
    batch = {"s": [], "m": [], "a": [], "lp": [], "r": []}
    for episode in range(cfg.episodes):
        env, decisions = collect_episode(
            sim_cfg, predictor, calibrator, cfg.steps, seed_base + episode, agent.act
        )
        # Returns along each UE's decision chain (see decision_returns).
        returns = decision_returns(
            [d[2] for d in decisions],
            _immediate_rewards(env, decisions, sim_cfg, cfg),
            [d[5] for d in decisions],
            cfg.wait_discount,
            cfg.wait_backup,
        )
        for (state, mask, action, logp, _, _), reward in zip(decisions, returns, strict=True):
            batch["s"].append(state)
            batch["m"].append(mask)
            batch["a"].append(action)
            batch["lp"].append(logp)
            batch["r"].append(float(reward))
        row = _history_row(episode, env, decisions, returns, sim_cfg)
        if (episode + 1) % cfg.episodes_per_update == 0:
            progress = episode / max(1, cfg.episodes - 1)
            entropy_coef = cfg.entropy_coef + (cfg.entropy_final - cfg.entropy_coef) * progress
            objective, value_loss, entropy = agent.update(
                np.array(batch["s"]),
                np.array(batch["m"]),
                np.array(batch["a"]),
                np.array(batch["lp"]),
                np.array(batch["r"]),
                entropy_coef,
            )
            row.update(policy_objective=objective, value_loss=value_loss, entropy=entropy)
            batch = {key: [] for key in batch}
        else:
            row.update(policy_objective=None, value_loss=None, entropy=None)
        history.append(row)
        log(
            f"CARE episode {episode + 1:03d}: reward {row['mean_reward']:+.4f}, "
            f"reserve {row['reserve_fraction']:.2f}, HSR {row['handover_success_rate']:.3f}, "
            f"displaced {row['displaced_rb_fraction']:.4f}"
        )
    _write_history(output, history)
    return agent, history


def stopping_targets(q_next_max, actions, immediate, following, discount) -> np.ndarray:
    """Fitted-Q targets with optimal stopping.

    A hold earns its own contract reward. A wait earns the discounted value of
    acting optimally at the same UE's next decision, which is never below the
    always-available "never hold" outcome (0). A wait with no later decision
    (``following < 0``) ends without a contract and earns 0.
    """
    targets = np.asarray(immediate, dtype=float).copy()
    for i, (action, nxt) in enumerate(zip(actions, following, strict=True)):
        if action == 0:
            targets[i] = 0.0 if nxt < 0 else discount * max(0.0, float(q_next_max[nxt]))
    return targets


class CareQAgent:
    """Action-value network over the masked v2 action catalog."""

    def __init__(self, state_dim, n_actions, cfg: CareConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.q = MLP((state_dim, *cfg.hidden, n_actions), self.rng, out_scale=1.0)
        self.opt = _Adam(self.q.params)
        self.epsilon = 1.0

    def masked_max(self, states, masks) -> np.ndarray:
        return np.where(masks, self.q.forward(states), -np.inf).max(axis=1)

    def act(self, state, mask):
        legal = np.flatnonzero(mask)
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal)), None
        values = self.q.forward(state[None])[0]
        return int(legal[np.argmax(values[legal])]), None

    def fit(self, states, masks, actions, immediate, following) -> float:
        cfg = self.cfg
        n = len(states)
        losses = []
        for _ in range(cfg.fqi_iterations):
            # Targets stay frozen within one iteration (fitted Q iteration).
            targets = stopping_targets(
                self.masked_max(states, masks), actions, immediate, following, cfg.wait_discount
            )
            for _ in range(cfg.update_epochs):
                order = self.rng.permutation(n)
                for start in range(0, n, cfg.minibatch):
                    ids = order[start : start + cfg.minibatch]
                    values, cache = self.q.forward(states[ids], cache=True)
                    k = len(ids)
                    error = values[np.arange(k), actions[ids]] - targets[ids]
                    grad = np.zeros_like(values)
                    grad[np.arange(k), actions[ids]] = 2 * error / k
                    self.opt.step(self.q.params, self.q.backward(cache, grad), cfg.q_lr)
                    losses.append(float(np.mean(error**2)))
        return float(np.mean(losses))

    def save(self, path, metadata):
        arrays = {f"w{i}": w for i, w in enumerate(self.q.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.q.biases)})
        np.savez(path, **arrays, metadata=np.array(json.dumps(metadata)))

    def policy(self):
        return CarePolicy([w.copy() for w in self.q.weights], [b.copy() for b in self.q.biases])


def train_care_fqi(
    sim_cfg: SimConfig,
    predictor,
    calibrator,
    cfg: CareConfig,
    output: str | Path | None = None,
    seed_base: int = 20000,
    log=print,
):
    """Fitted Q iteration with optimal-stopping wait targets.

    The behavior policy is epsilon-greedy over the current Q, annealed from
    uniform over legal actions to near-greedy. All transitions are kept: hold
    targets are realized contract rewards, so they remain valid off-policy.
    """
    env = RANHandoverEnv(replace(sim_cfg, episode_steps=cfg.steps))
    agent = CareQAgent(env.state_dim, env.n_actions, cfg)
    buffer = {"s": [], "m": [], "a": [], "r": [], "f": []}
    history = []
    for episode in range(cfg.episodes):
        progress = episode / max(1, cfg.episodes - 1)
        agent.epsilon = cfg.epsilon_start + (cfg.epsilon_final - cfg.epsilon_start) * min(
            1.0, progress / cfg.epsilon_decay_fraction
        )
        env, decisions = collect_episode(
            sim_cfg, predictor, calibrator, cfg.steps, seed_base + episode, agent.act
        )
        immediate = _immediate_rewards(env, decisions, sim_cfg, cfg)
        offset = len(buffer["a"])
        for (state, mask, action, _, _, following), reward in zip(
            decisions, immediate, strict=True
        ):
            buffer["s"].append(state)
            buffer["m"].append(mask)
            buffer["a"].append(action)
            buffer["r"].append(reward)
            buffer["f"].append(-1 if following is None else offset + following)
        row = _history_row(episode, env, decisions, immediate, sim_cfg)
        row["epsilon"] = agent.epsilon
        if (episode + 1) % cfg.episodes_per_update == 0:
            row["value_loss"] = agent.fit(
                np.array(buffer["s"]),
                np.array(buffer["m"]),
                np.array(buffer["a"]),
                np.array(buffer["r"]),
                np.array(buffer["f"]),
            )
        else:
            row["value_loss"] = None
        history.append(row)
        log(
            f"CARE-FQI episode {episode + 1:03d}: eps {agent.epsilon:.2f}, "
            f"reserve {row['reserve_fraction']:.2f}, HSR {row['handover_success_rate']:.3f}, "
            f"displaced {row['displaced_rb_fraction']:.4f}"
        )
    _write_history(output, history)
    return agent, history
