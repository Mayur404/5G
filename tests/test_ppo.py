from pathlib import Path

import numpy as np
import pytest

from proran.config import SimConfig, TrainConfig
from proran.experiment import evaluate_policy
from proran.policies import SavedPPOPolicy, no_reservation
from proran.ppo import LinearMaskedPPO, train_ppo
from proran.simulator import RANHandoverEnv


def test_masked_policy_assigns_zero_probability_to_illegal_actions():
    agent = LinearMaskedPPO(RANHandoverEnv.state_dim, RANHandoverEnv.n_actions, seed=1)
    state = np.ones((1, RANHandoverEnv.state_dim))
    mask = np.array([[True, False, True, False, False, False, False]])
    probabilities = agent.probs(state, mask)[0]
    assert np.all(probabilities[~mask[0]] == 0)
    assert np.isclose(probabilities.sum(), 1.0)


def test_short_training_run_is_finite(tmp_path):
    sim = SimConfig(n_ues=12, episode_steps=25)
    train = TrainConfig(episodes=2, update_epochs=2, imitation_episodes=0)
    agent, history = train_ppo(sim, train, tmp_path)
    assert len(history) == 2
    assert np.isfinite(agent.actor_w).all()
    assert (tmp_path / "ppo_model.npz").exists()


def test_extreme_negative_legal_logits_stay_normalized():
    agent = LinearMaskedPPO(1, 3)
    agent.actor_w[:] = 0
    agent.actor_b[:] = [-2e9, -2e9 - 1, 0]
    state = np.ones((1, 1))
    mask = np.array([[True, True, False]])
    actual = agent.probs(state, mask)[0]
    expected = np.array([1, np.exp(-1), 0]) / (1 + np.exp(-1))
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        SavedPPOPolicy._probabilities(state[0], mask[0], agent.actor_w, agent.actor_b), expected
    )
    with pytest.raises(ValueError, match="legal"):
        agent.probs(state, np.zeros_like(mask))


def test_corrupted_saved_actor_parameters_are_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, actor_w=np.full((14, 7), np.nan), actor_b=np.zeros(7))
    with pytest.raises(ValueError, match="finite"):
        SavedPPOPolicy.load(str(path))
    np.savez(path, actor_w=np.ones((14, 7)), actor_b=np.zeros(6))
    with pytest.raises(ValueError, match="bias"):
        SavedPPOPolicy.load(str(path))


def test_ppo_clipped_and_entropy_gradient_matches_finite_differences():
    agent = LinearMaskedPPO(2, 3, seed=18)
    states = np.array([[1.0, 0.2], [1.0, -0.5], [1.0, 0.7]])
    masks = np.array([[True, True, False], [True, False, True], [True, True, True]])
    actions = np.array([0, 2, 1])
    advantages = np.array([1.0, -0.4, 0.7])
    returns = np.array([0.4, -0.3, 1.2])
    agent.critic_w[:] = [0.1, 0.2]
    agent.critic_b[:] = 0.05
    cfg = TrainConfig(update_epochs=1, entropy_coef=0.02)
    probs = agent.probs(states, masks)
    old_logp = np.log(probs[np.arange(3), actions]) - np.log([1.4, 0.6, 1.0])
    captured = {}
    for name in ("actor_w", "actor_b", "critic_w", "critic_b"):
        optimizer = getattr(agent, name + "_opt")
        optimizer.ascend = lambda parameter, gradient, lr, name=name: captured.update(
            {name: gradient.copy()}
        )
    agent.update(states, masks, actions, old_logp, advantages, returns, cfg)

    def objective():
        p = agent.probs(states, masks)
        ratio = np.exp(np.log(p[np.arange(3), actions]) - old_logp)
        clipped = np.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)
        entropy = -np.sum(p * np.log(p + 1e-12), axis=1)
        return np.mean(
            np.minimum(ratio * advantages, clipped * advantages) + cfg.entropy_coef * entropy
        )

    for name in ("actor_w", "actor_b"):
        parameter = getattr(agent, name)
        expected = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + 1e-6
            plus = objective()
            parameter[index] = original - 1e-6
            minus = objective()
            parameter[index] = original
            expected[index] = (plus - minus) / 2e-6
        np.testing.assert_allclose(captured[name], expected, atol=1e-7)

    for name in ("critic_w", "critic_b"):
        parameter = getattr(agent, name)
        expected = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + 1e-6
            plus = -np.mean((returns - agent.value(states)) ** 2)
            parameter[index] = original - 1e-6
            minus = -np.mean((returns - agent.value(states)) ** 2)
            parameter[index] = original
            expected[index] = (plus - minus) / 2e-6
        np.testing.assert_allclose(captured[name], expected, atol=1e-7)


def test_saved_research_policy_is_active_and_beats_no_reservation():
    model_path = Path("results/ppo_model.npz")
    assert model_path.exists(), "The reproducible trained model is part of the artifact"
    policy = SavedPPOPolicy.load(str(model_path))
    cfg = SimConfig()
    seeds = [9000, 9001, 9002]
    baseline, _ = evaluate_policy("baseline", no_reservation, cfg, seeds, 260)
    learned, _ = evaluate_policy("learned", policy, cfg, seeds, 260)
    assert learned["handover_success_rate"] > baseline["handover_success_rate"]
    assert learned["qos_rate"] > baseline["qos_rate"]
    assert learned["reservation_occupancy"] > 0
