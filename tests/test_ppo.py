import numpy as np
from pathlib import Path

from proran.config import SimConfig, TrainConfig
from proran.ppo import LinearMaskedPPO, train_ppo
from proran.simulator import RANHandoverEnv
from proran.experiment import evaluate_policy
from proran.policies import SavedPPOPolicy, no_reservation


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
