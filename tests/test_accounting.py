import numpy as np
import pytest

from proran.actions import decode_action
from proran.config import SimConfig
from proran.policies import certainty_equivalent, no_reservation
from proran.simulator import RANHandoverEnv


def test_noop_is_always_legal_and_masks_have_fixed_shape():
    env = RANHandoverEnv(SimConfig(n_ues=12, episode_steps=20))
    observation = env.reset(11)
    for _ in range(20):
        state, mask, _ = observation
        assert state.shape == (env.state_dim,)
        assert mask.shape == (env.n_actions,)
        assert mask[0]
        result = env.step(0, observation)
        observation = (result.next_state, result.next_mask, result.info["next_context"])


def test_action_bounds_respect_predicted_demand_and_time():
    for action in range(1, 7):
        amount, delay = decode_action(action, predicted_demand=9, time_to_handover=5)
        assert 1 <= amount <= 9
        assert 0 <= delay <= 5


def test_capacity_invariant_under_aggressive_reservation():
    cfg = SimConfig(n_ues=30, cell_capacity=45, episode_steps=160)
    env = RANHandoverEnv(cfg)
    observation = env.reset(4)
    for _ in range(cfg.episode_steps):
        state, mask, _ = observation
        action = next((a for a in (2, 4, 6, 1, 3, 5) if mask[a]), 0)
        result = env.step(action, observation)
        totals = result.info["allocations"] + result.info["reservations"]
        assert np.all(totals <= cfg.cell_capacity)
        assert np.all(totals >= 0)
        observation = (result.next_state, result.next_mask, result.info["next_context"])


def test_no_reservation_has_zero_occupancy_and_undefined_waste():
    cfg = SimConfig(n_ues=20, episode_steps=80)
    metrics = RANHandoverEnv(cfg).run(no_reservation, seed=5)
    summary = metrics.summary(cfg.dt_s)
    assert summary["reservation_occupancy"] == 0
    assert np.isnan(summary["unused_reservation_fraction"])
    assert len(metrics.latency_ms) == len(metrics.latency_slice_ids) == metrics.handovers
    assert set(metrics.latency_slice_ids) <= {0, 1, 2}
    assert sum(metrics.handovers_by_slice) == metrics.handovers
    assert sum(metrics.successes_by_slice) == metrics.successes
    assert sum(metrics.qos_by_slice) == metrics.qos_successes


def test_reproducible_matched_trace():
    cfg = SimConfig(n_ues=18, episode_steps=60)
    first = RANHandoverEnv(cfg).run(no_reservation, seed=99).summary(cfg.dt_s)
    second = RANHandoverEnv(cfg).run(no_reservation, seed=99).summary(cfg.dt_s)
    for key in ("handover_success_rate", "qos_rate", "rb_utilization", "mean_latency_ms"):
        assert first[key] == pytest.approx(second[key])


def test_exogenous_handover_count_is_policy_independent():
    cfg = SimConfig(n_ues=22, episode_steps=100)
    no_res = RANHandoverEnv(cfg).run(no_reservation, seed=123)
    proactive = RANHandoverEnv(cfg).run(certainty_equivalent, seed=123)
    assert no_res.handovers == proactive.handovers


def test_simultaneous_arrivals_do_not_reuse_the_same_free_rbs():
    cfg = SimConfig(n_ues=2, cell_capacity=10)
    env = RANHandoverEnv(cfg)
    env.reset(19)
    first, second = env.ues
    for ue in env.ues:
        ue.slice_id = 0  # eMBB needs at least 7 RB for admission.
        ue.cell = 0
        ue.demand = 8
    env.reservations.clear()
    env._allocate_incumbents(count_metrics=False)
    for ue in env.ues:
        ue.cell = 1  # _move has already selected the common target.
    successes, _, _ = env._resolve_handovers([(first, 0, 1), (second, 0, 1)])
    assert successes == 1
    assert env.allocations[1] == 8
    assert env.ue_allocations.get(first.uid) == 8
    assert second.uid not in env.ue_allocations
    assert np.all(env.allocations + env._reservation_totals() <= cfg.cell_capacity)


def test_urllc_arrival_has_priority_over_embb_arrival():
    cfg = SimConfig(n_ues=2, cell_capacity=10)
    env = RANHandoverEnv(cfg)
    env.reset(29)
    embb, urllc = env.ues
    embb.slice_id, urllc.slice_id = 0, 1
    for ue in env.ues:
        ue.cell = 0
        ue.demand = 8
    env.reservations.clear()
    env._allocate_incumbents(count_metrics=False)
    for ue in env.ues:
        ue.cell = 1
    env._resolve_handovers([(embb, 0, 1), (urllc, 0, 1)])
    assert env.ue_allocations.get(urllc.uid) == 8
    assert embb.uid not in env.ue_allocations


def test_reservation_occupancy_counts_the_held_interval():
    cfg = SimConfig(n_ues=1, cell_capacity=60)
    env = RANHandoverEnv(cfg)
    observation = env.reset(23)
    # Move the UE just before the vertical boundary so activation and handover
    # occur in the same interval. Its hold still consumes one RB interval.
    ue = env.ues[0]
    ue.pos[:] = [49.9, 25.0]
    ue.vel[:] = [2.0, 0.0]
    ue.cell = 0
    env._allocate_incumbents(count_metrics=False)
    observation = env.observe()
    action = next(a for a in (2, 1) if observation[1][a])
    result = env.step(action, observation)
    assert result.info["held_rbs"] > 0
    assert env.metrics.occupancy_rb_intervals == result.info["held_rbs"]
