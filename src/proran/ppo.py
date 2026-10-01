from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np

from .config import SimConfig, TrainConfig
from .policies import SavedPPOPolicy
from .policies import uncertainty_aware
from .simulator import RANHandoverEnv


@dataclass
class _Adam:
    shape: tuple[int, ...]
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.m = np.zeros(self.shape)
        self.v = np.zeros(self.shape)
        self.t = 0

    def ascend(self, parameter: np.ndarray, gradient: np.ndarray, lr: float) -> None:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        mhat = self.m / (1 - self.beta1**self.t)
        vhat = self.v / (1 - self.beta2**self.t)
        parameter += lr * mhat / (np.sqrt(vhat) + self.eps)


class LinearMaskedPPO:
    """Masked categorical PPO with a linear actor and critic.

    The compact function approximator keeps the research artifact fast and
    inspectable while retaining the clipped likelihood-ratio/GAE mechanics.
    """

    def __init__(self, state_dim: int, n_actions: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.actor_w = self.rng.normal(0, 0.04, size=(state_dim, n_actions))
        self.actor_b = np.zeros(n_actions)
        self.critic_w = np.zeros(state_dim)
        self.critic_b = np.zeros(1)
        self.actor_w_opt = _Adam(self.actor_w.shape)
        self.actor_b_opt = _Adam(self.actor_b.shape)
        self.critic_w_opt = _Adam(self.critic_w.shape)
        self.critic_b_opt = _Adam(self.critic_b.shape)

    def probs(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        logits = states @ self.actor_w + self.actor_b
        logits = np.where(masks, logits, -1e9)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits) * masks
        return exp / exp.sum(axis=1, keepdims=True)

    def value(self, states: np.ndarray) -> np.ndarray:
        return states @ self.critic_w + self.critic_b[0]

    def act(self, state: np.ndarray, mask: np.ndarray) -> tuple[int, float, float]:
        p = self.probs(state[None, :], mask[None, :])[0]
        action = int(self.rng.choice(len(p), p=p))
        return action, float(np.log(p[action] + 1e-12)), float(self.value(state[None, :])[0])

    def update(self, states: np.ndarray, masks: np.ndarray, actions: np.ndarray,
               old_logp: np.ndarray, advantages: np.ndarray, returns: np.ndarray,
               cfg: TrainConfig) -> tuple[float, float]:
        last_objective = last_vloss = 0.0
        n = len(states)
        for _ in range(cfg.update_epochs):
            probs = self.probs(states, masks)
            selected = probs[np.arange(n), actions]
            ratio = np.exp(np.log(selected + 1e-12) - old_logp)
            active = np.where(advantages >= 0, ratio <= 1 + cfg.clip, ratio >= 1 - cfg.clip)
            coeff = advantages * ratio * active / n
            onehot = np.zeros_like(probs)
            onehot[np.arange(n), actions] = 1.0
            grad_logits = coeff[:, None] * (onehot - probs)
            entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
            grad_logits += cfg.entropy_coef * (-probs * (np.log(probs + 1e-12) + entropy[:, None])) / n
            self.actor_w_opt.ascend(self.actor_w, states.T @ grad_logits, cfg.actor_lr)
            self.actor_b_opt.ascend(self.actor_b, grad_logits.sum(axis=0), cfg.actor_lr)

            values = self.value(states)
            error = returns - values
            self.critic_w_opt.ascend(self.critic_w, 2 * states.T @ error / n, cfg.critic_lr)
            self.critic_b_opt.ascend(self.critic_b, np.array([2 * error.mean()]), cfg.critic_lr)
            clipped = np.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)
            last_objective = float(np.mean(np.minimum(ratio * advantages, clipped * advantages)))
            last_vloss = float(np.mean(error**2))
        return last_objective, last_vloss

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, actor_w=self.actor_w, actor_b=self.actor_b,
                 critic_w=self.critic_w, critic_b=self.critic_b)

    def policy(self, deterministic: bool = True) -> SavedPPOPolicy:
        return SavedPPOPolicy(self.actor_w.copy(), self.actor_b.copy(), deterministic)

    def behavior_clone(self, states: np.ndarray, masks: np.ndarray, actions: np.ndarray,
                       epochs: int, learning_rate: float) -> list[float]:
        """Warm-start the masked actor from a transparent uncertainty heuristic."""
        losses = []
        n = len(states)
        if n == 0 or epochs <= 0:
            return losses
        for _ in range(epochs):
            probs = self.probs(states, masks)
            onehot = np.zeros_like(probs)
            onehot[np.arange(n), actions] = 1.0
            # Ascend the categorical log-likelihood.
            grad_logits = (onehot - probs) / n
            self.actor_w_opt.ascend(self.actor_w, states.T @ grad_logits, learning_rate)
            self.actor_b_opt.ascend(self.actor_b, grad_logits.sum(axis=0), learning_rate)
            losses.append(float(-np.mean(np.log(probs[np.arange(n), actions] + 1e-12))))
        return losses


def train_ppo(sim_cfg: SimConfig, train_cfg: TrainConfig, output_dir: str | Path) -> tuple[LinearMaskedPPO, list[dict]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = LinearMaskedPPO(RANHandoverEnv.state_dim, RANHandoverEnv.n_actions, train_cfg.seed)
    if train_cfg.imitation_episodes > 0:
        clone_states, clone_masks, clone_actions = [], [], []
        for clone_episode in range(train_cfg.imitation_episodes):
            clone_env = RANHandoverEnv(sim_cfg)
            clone_obs = clone_env.reset(600_000 + train_cfg.seed * 100 + clone_episode)
            for _ in range(sim_cfg.episode_steps):
                clone_state, clone_mask, clone_context = clone_obs
                clone_action = int(uncertainty_aware(clone_state, clone_mask, clone_context))
                clone_states.append(clone_state.copy())
                clone_masks.append(clone_mask.copy())
                clone_actions.append(clone_action)
                clone_result = clone_env.step(clone_action, clone_obs)
                clone_obs = (clone_result.next_state, clone_result.next_mask, clone_result.info["next_context"])
        clone_states_a = np.asarray(clone_states)
        clone_masks_a = np.asarray(clone_masks)
        clone_actions_a = np.asarray(clone_actions)
        # No-op naturally dominates time series. Balance it against positive
        # examples so the actor learns the reservation boundary, not inactivity.
        positive = np.flatnonzero(clone_actions_a > 0)
        no_op = np.flatnonzero(clone_actions_a == 0)
        if positive.size:
            keep_no_op = agent.rng.choice(no_op, size=min(no_op.size, 2 * positive.size), replace=False)
            keep = np.concatenate([positive, keep_no_op])
            agent.rng.shuffle(keep)
            clone_states_a, clone_masks_a, clone_actions_a = (
                clone_states_a[keep], clone_masks_a[keep], clone_actions_a[keep])
        clone_losses = agent.behavior_clone(clone_states_a, clone_masks_a, clone_actions_a,
                                             train_cfg.imitation_epochs,
                                             train_cfg.imitation_lr)
        np.savez(output_dir / "imitation_model.npz", actor_w=agent.actor_w, actor_b=agent.actor_b,
                 critic_w=agent.critic_w, critic_b=agent.critic_b)
        with (output_dir / "imitation_history.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["epoch", "cross_entropy"])
            writer.writeheader()
            writer.writerows({"epoch": i + 1, "cross_entropy": loss} for i, loss in enumerate(clone_losses))
    history: list[dict] = []
    for episode in range(train_cfg.episodes):
        env = RANHandoverEnv(sim_cfg)
        observation = env.reset(train_cfg.seed * 10000 + episode)
        states, masks, actions, old_logps, values, rewards, dones = [], [], [], [], [], [], []
        for _ in range(sim_cfg.episode_steps):
            state, mask, _context = observation
            action, logp, value = agent.act(state, mask)
            result = env.step(action, observation)
            states.append(state.copy()); masks.append(mask.copy()); actions.append(action)
            old_logps.append(logp); values.append(value); rewards.append(result.reward); dones.append(result.done)
            observation = (result.next_state, result.next_mask, result.info["next_context"])
        rewards_a = np.asarray(rewards)
        values_a = np.asarray(values)
        advantages = np.zeros_like(rewards_a)
        gae = 0.0
        next_value = 0.0
        for t in range(len(rewards_a) - 1, -1, -1):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards_a[t] + train_cfg.gamma * next_value * nonterminal - values_a[t]
            gae = delta + train_cfg.gamma * train_cfg.gae_lambda * nonterminal * gae
            advantages[t] = gae
            next_value = values_a[t]
        returns = advantages + values_a
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        objective, vloss = agent.update(np.asarray(states), np.asarray(masks), np.asarray(actions),
                                        np.asarray(old_logps), adv_norm, returns, train_cfg)
        row = {"episode": episode + 1, "reward": float(rewards_a.sum()),
               "policy_objective": objective, "value_loss": vloss,
               "handover_success_rate": env.metrics.summary(sim_cfg.dt_s)["handover_success_rate"]}
        history.append(row)
    agent.save(output_dir / "ppo_model.npz")
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    return agent, history
