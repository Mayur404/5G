"""Regression tests for the opt-in v2 (CARE) semantics, calibration and learner."""

from dataclasses import replace

import numpy as np
import pytest

from proran.actions import ACTION_CATALOG_V2, decode_action_v2
from proran.calibration import LogisticCalibrator, expand, reliability
from proran.care import (
    MLP,
    CareConfig,
    CareEnsemblePolicy,
    CarePolicy,
    contract_reward,
    decision_returns,
    masked_softmax,
    stopping_targets,
    train_care,
)
from proran.config import SimConfig
from proran.models import Reservation
from proran.simulator import RANHandoverEnv

V2 = SimConfig(
    n_ues=24,
    episode_steps=60,
    prediction_mode="kinematic",
    research_metrics=True,
    action_set="v2",
    max_decisions=24,
    failure_outage=3,
    activation_rule="floor_protected",
    sla_capped_by_demand=True,
    preparation_delay=1,
    retime_pending=True,
    common_radio_stream=True,
)


class RadioDraw:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def single_ue_env(capacity=12, demand=8, **overrides):
    env = RANHandoverEnv(replace(V2, n_ues=1, cell_capacity=capacity, **overrides))
    ue = env.ues[0]
    ue.slice_id, ue.cell, ue.demand = 1, 0, demand
    env.reservations.clear()
    env._allocate_incumbents(count_metrics=False)
    env.radio_rng = RadioDraw(0.5)
    env.cfg.common_radio_stream = False
    return env, ue


def active_hold(env, amount, target=1):
    r = Reservation(1, 0, target, amount, 0, 0, 8, target, active=True, activated_amount=amount)
    env.reservations.append(r)
    env.metrics.activated_rbs += amount
    return r


def test_v2_defaults_do_not_change_legacy_configuration():
    cfg = SimConfig()
    assert (cfg.action_set, cfg.max_decisions, cfg.failure_outage) == ("legacy", 1, 0)
    assert (cfg.activation_rule, cfg.sla_capped_by_demand, cfg.preparation_delay) == (
        "free",
        False,
        0,
    )
    env = RANHandoverEnv(cfg)
    assert (env.state_dim, env.n_actions) == (14, 7)
    assert RANHandoverEnv(V2).n_actions == len(ACTION_CATALOG_V2)


def test_config_round_trip_restores_slices():
    cfg = replace(V2, centers=((20.0, 20.0), (60.0, 20.0)))
    assert SimConfig.from_dict(cfg.as_dict()) == cfg


@pytest.mark.parametrize(
    "bad",
    [
        {"action_set": "x"},
        {"activation_rule": "x"},
        {"max_decisions": 0},
        {"failure_outage": -1},
        {"preparation_delay": True},
    ],
)
def test_invalid_v2_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        replace(V2, **bad)


def test_v2_action_decoding_respects_floor_demand_and_timing():
    floor_jit = next(
        i for i, a in enumerate(ACTION_CATALOG_V2) if (a.amount, a.timing) == ("floor", "jit")
    )
    upper_mid = next(
        i for i, a in enumerate(ACTION_CATALOG_V2) if (a.amount, a.timing) == ("upper", "mid")
    )
    assert decode_action_v2(0, 8, 5, 5, 10) == (0, 0)
    assert decode_action_v2(floor_jit, 8, 5, 5, 10) == (5, 4)
    assert decode_action_v2(upper_mid, 8, 5, 5, 10) == (10, 2)
    # The floor never exceeds what the UE could use.
    assert decode_action_v2(floor_jit, 2, 3, 5, 3) == (3, 2)
    with pytest.raises(ValueError):
        decode_action_v2(len(ACTION_CATALOG_V2), 8, 5, 5, 10)


def test_low_demand_ue_can_succeed_with_capped_admission_floor():
    env, ue = single_ue_env(capacity=12, demand=3)  # uRLLC floor is 5 RBs.
    ue.cell = 1
    successes, _, _ = env._resolve_handovers([(ue, 0, 1)])
    assert successes == 1
    legacy, legacy_ue = single_ue_env(capacity=12, demand=3, sla_capped_by_demand=False)
    legacy_ue.cell = 1
    assert legacy._resolve_handovers([(legacy_ue, 0, 1)])[0] == 0


def test_pivotal_hold_is_credited_as_rescue():
    env, ue = single_ue_env(capacity=12, demand=8)
    active_hold(env, 8)
    env.allocations[1] = 4  # 0 free RBs outside the hold.
    env.cell_unmet[1] = 6  # Incumbents would absorb 6 of the 8 held RBs.
    ue.cell = 1
    successes, _, _ = env._resolve_handovers([(ue, 0, 1)])
    outcome = env.contract_outcomes[1]
    assert successes == 1 and outcome["rescued"] and outcome["status"] == "settled"
    assert env.metrics.rescued_handovers == 1


def test_hold_in_idle_cell_is_not_credited_and_displaces_nothing():
    env, ue = single_ue_env(capacity=30, demand=8)
    active_hold(env, 8)
    env.cell_unmet[1] = 0
    ue.cell = 1
    env._resolve_handovers([(ue, 0, 1)])
    outcome = env.contract_outcomes[1]
    assert not outcome["rescued"] and outcome["displaced"] == 0


def test_displacement_is_only_charged_for_unmet_incumbent_demand():
    env = RANHandoverEnv(replace(V2, n_ues=3, cell_capacity=10))
    for ue, demand in zip(env.ues, (6, 6, 1), strict=True):
        ue.cell, ue.slice_id, ue.demand = 0, 2, demand
    r = Reservation(1, 2, 0, 4, 0, 0, 8, 0, active=True, activated_amount=4)
    env.reservations = [r]
    env.ues[2].cell = 1
    env._allocate_incumbents(count_metrics=True)
    # Budget 6 for demand 12: 6 unmet RBs, of which the 4 held RBs displaced 4.
    assert env.cell_unmet[0] == 6
    assert r.displaced == pytest.approx(4)
    assert env.metrics.displaced_rb_intervals == pytest.approx(4)


def test_failed_handover_triggers_outage_without_grants_or_attempts():
    env, ue = single_ue_env(capacity=12, demand=8)
    env.allocations[1] = 12
    ue.cell = 1
    env._resolve_handovers([(ue, 0, 1)])
    assert env.outage_until[ue.uid] == env.t + 3
    for _ in range(3):
        env.t += 1
        assert env.in_outage(ue)
        assert env._allocate_incumbents().get(ue.uid, 0) == 0
    env.t += 1
    assert not env.in_outage(ue)


def test_v2_runs_preserve_capacity_preparation_and_retiming_invariants():
    env = RANHandoverEnv(V2)
    env.reset(17)
    upper_now = next(
        i for i, a in enumerate(ACTION_CATALOG_V2) if (a.amount, a.timing) == ("upper", "now")
    )
    seen_contracts = 0
    for _ in range(V2.episode_steps):
        observations = env.observe_many()
        assert len(observations) <= V2.max_decisions
        actions = [upper_now if mask[upper_now] else 0 for _, mask, _ in observations]
        targets = {}
        for r in env.reservations:
            targets[r.rid] = r.target
            assert r.activate_at >= r.created_at + V2.preparation_delay
        _, _, info = env.step_many(actions, observations)
        for r in env.reservations:
            assert targets.get(r.rid, r.target) == r.target
            assert r.activate_at >= r.created_at + V2.preparation_delay
        seen_contracts += sum(rid is not None for rid in info["reservation_ids"])
        assert np.all(env.allocations + env._reservation_totals() <= V2.cell_capacity)
        assert np.all(env.allocations >= 0)
    env.settle_open_contracts()
    assert seen_contracts > 0
    assert set(env.contract_outcomes) == set(range(1, env.next_rid))
    summary = env.metrics.summary(V2.dt_s)
    assert 0 <= summary["outage_fraction"] <= 1
    assert summary["displaced_rb_fraction"] >= 0


def test_matched_seeds_are_reproducible_for_multi_decision_runs():
    def run():
        env = RANHandoverEnv(V2)
        return env.run_many(lambda s, m, c: 3 if m[3] else 0, seed=5).summary(0.1)

    first, second = run(), run()
    for key, value in first.items():
        assert value == pytest.approx(second[key], nan_ok=True)


def test_step_many_rejects_masked_actions():
    env = RANHandoverEnv(V2)
    env.reset(3)
    observations = env.observe_many()
    while not observations:
        env.step_many([], [])
        observations = env.observe_many()
    _, mask, _ = observations[0]
    illegal = [a for a in range(len(mask)) if not mask[a]]
    if illegal:
        with pytest.raises(ValueError):
            env.step_many([illegal[0]] + [0] * (len(observations) - 1), observations)
    with pytest.raises(ValueError):
        env.step_many([0], [])


def test_calibrator_recovers_a_logistic_relationship_and_metrics_are_sane():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (6000, 7))
    true_w = np.array([1.5, -2.0, 0.5, 0, 0, 1.0, 0, 0, 0, 0, 0.3])
    p = 1 / (1 + np.exp(-(expand(x) @ true_w)))
    y = rng.random(len(p)) < p
    model = LogisticCalibrator(l2=1e-6)
    model.fit(x, y)
    assert np.max(np.abs(model.weights - true_w)) < 0.25
    report = reliability(model.predict(x), y)
    assert report["ece"] < 0.03 and 0.5 < report["auc"] <= 1
    assert reliability(np.array([0.1, 0.9]), np.array([0, 1]))["auc"] == 1.0


def test_contract_reward_prices_rescue_and_displacement():
    cfg = CareConfig(displacement_weight=2.0, reward_scale=1.0, qos_value=4.0)
    base = {
        "arrival_demand": 6,
        "rescued": True,
        "qos_rescued": False,
        "displaced": 3.0,
        "slice_id": 1,
    }
    assert contract_reward(base, 3, cfg) == pytest.approx(2.0 * 4 * 6 - 2.0 * 3.0)
    idle = {**base, "rescued": False, "displaced": 0.0}
    assert contract_reward(idle, 3, cfg) == 0


def test_mlp_backward_matches_finite_differences():
    rng = np.random.default_rng(1)
    net = MLP((5, 7, 4), rng, out_scale=1.0)
    x = rng.normal(size=(3, 5))
    upstream = rng.normal(size=(3, 4))
    _, cache = net.forward(x, cache=True)
    grads = net.backward(cache, upstream)
    for param, grad in zip(net.params, grads, strict=True):
        index = tuple(rng.integers(s) for s in param.shape)
        old = param[index]
        param[index] = old + 1e-6
        plus = np.sum(net.forward(x) * upstream)
        param[index] = old - 1e-6
        minus = np.sum(net.forward(x) * upstream)
        param[index] = old
        assert grad[index] == pytest.approx((plus - minus) / 2e-6, rel=1e-4, abs=1e-7)


def test_masked_softmax_gives_zero_probability_to_illegal_actions():
    p = masked_softmax(np.array([[5.0, 1.0, 9.0]]), np.array([[True, True, False]]))
    assert p[0, 2] == 0 and p.sum() == pytest.approx(1)


def test_wait_returns_use_the_never_hold_floor_under_the_stopping_backup():
    # UE chain: wait (0) -> wait (1) -> hold (2) with a negative contract reward,
    # plus an unrelated wait (3) with no later decision and a positive hold (4).
    actions = [0, 0, 5, 0, 5]
    immediate = [0.0, 0.0, -2.0, 0.0, 3.0]
    following = [1, 2, None, None, None]
    stopping = decision_returns(actions, immediate, following, 0.5)
    # Waiting is worth at least "never hold" (0), never the bad hold's -2.
    assert stopping.tolist() == [0.0, 0.0, -2.0, 0.0, 3.0]
    on_policy = decision_returns(actions, immediate, following, 0.5, "on_policy")
    assert on_policy.tolist() == [-0.5, -1.0, -2.0, 0.0, 3.0]
    good = decision_returns([0, 5], [0.0, 4.0], [1, None], 0.5)
    assert good.tolist() == [2.0, 4.0]
    with pytest.raises(ValueError):
        decision_returns([0], [0.0], [None], 0.5, "unknown")


def test_fitted_q_stopping_targets():
    # Decisions: wait->(index 2), hold, wait with no successor, wait->(index 1).
    q_next_max = np.array([0.0, -3.0, 2.0, 0.0])
    targets = stopping_targets(
        q_next_max,
        actions=[0, 4, 0, 0],
        immediate=[9.0, -1.5, 9.0, 9.0],
        following=[2, -1, -1, 1],
        discount=0.5,
    )
    # Hold: its own contract reward. Wait: discounted best continuation, floored
    # at the never-hold value 0; a wait with no successor ends with no contract.
    assert targets.tolist() == [1.0, -1.5, 0.0, 0.0]


def test_ensemble_policy_acts_greedily_on_mean_values():
    def member(bias):
        return CarePolicy([np.zeros((3, 3))], [np.array(bias, dtype=float)])

    ensemble = CareEnsemblePolicy([member([0.0, 5.0, -1.0]), member([0.0, -3.0, 4.0])])
    assert ensemble.values(np.ones(3)).tolist() == [0.0, 1.0, 1.5]
    mask = np.array([True, True, True])
    assert ensemble(np.ones(3), mask, {}) == 2
    assert ensemble(np.ones(3), np.array([True, True, False]), {}) == 1
    assert ensemble(np.ones(3), np.array([True, False, False]), {}) == 0
    with pytest.raises(ValueError):
        CareEnsemblePolicy([])


@pytest.mark.parametrize("learner", ["fqi", "ppo"])
def test_both_learners_train_save_and_reload(tmp_path, learner):
    cfg = CareConfig(
        episodes=2, episodes_per_update=1, steps=30, minibatch=64, update_epochs=1, learner=learner
    )
    agent, history = train_care(V2, None, None, cfg, seed_base=777, log=lambda _: None)
    assert len(history) == 2 and history[-1]["decisions"] > 0
    agent.save(tmp_path / "m.npz", {})
    policy = CarePolicy.load(tmp_path / "m.npz")
    env = RANHandoverEnv(V2)
    env.reset(5)
    for state, mask, context in env.observe_many():
        assert mask[policy(state, mask, context)]
