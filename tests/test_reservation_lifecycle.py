"""Independent edge cases for the reservation accounting contract."""

import numpy as np
import pytest

from proran.config import SimConfig
from proran.models import Reservation
from proran.simulator import RANHandoverEnv


class RadioDraw:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def arrival_env(capacity=12, demand=8):
    env = RANHandoverEnv(SimConfig(n_ues=1, cell_capacity=capacity))
    ue = env.ues[0]
    ue.slice_id = 1
    ue.cell = 0
    ue.demand = demand
    env.reservations.clear()
    env._allocate_incumbents(count_metrics=False)
    env.radio_rng = RadioDraw(0.5)
    return env, ue


def hold(env, amount, target=1, active=True, activate_at=0):
    r = Reservation(
        1,
        0,
        target,
        amount,
        0,
        activate_at,
        8,
        target,
        active=active,
        activated_amount=amount if active else 0,
    )
    env.reservations.append(r)
    env.metrics.activated_rbs = amount if active else 0
    return r


def cross(env, ue, target=1):
    ue.cell = target
    return env._resolve_handovers([(ue, 0, target)])


def test_wrong_target_hold_cannot_supply_arrival_and_is_wasted():
    env, ue = arrival_env()
    hold(env, 8, target=2)
    env.allocations[1] = 9  # Only 3 RB are free at the actual target.
    successes, qos, waste = cross(env, ue)
    assert (successes, qos, waste) == (0, 0, 8)
    assert env.metrics.used_reserved_rbs == 0
    assert not env.reservations
    assert env.last_handovers[0]["reserved_rbs"] == 0
    assert env.last_handovers[0]["allocated_rbs"] == 0


def test_early_arrival_cancels_pending_hold_without_occupancy_or_waste():
    env, ue = arrival_env()
    hold(env, 8, active=False, activate_at=4)
    env.allocations[1] = 9
    assert cross(env, ue) == (0, 0, 0)
    assert env.metrics.activated_rbs == env.metrics.used_reserved_rbs == 0
    assert not env.reservations
    assert env.last_contract_events[0]["kind"] == "cancelled"


def test_radio_failure_uses_zero_reserved_rbs_even_with_full_capacity():
    env, ue = arrival_env()
    hold(env, 8)
    env.radio_rng = RadioDraw(0.0)
    assert cross(env, ue) == (0, 0, 8)
    result = env.last_handovers[0]
    assert result["available_rbs"] == 8
    assert result["allocated_rbs"] == 0
    assert not result["radio_ok"]
    assert env.metrics.used_reserved_rbs == 0


def test_partial_admission_can_succeed_while_full_demand_qos_fails():
    env, ue = arrival_env(demand=9)
    hold(env, 5)
    env.allocations[1] = 7
    assert cross(env, ue) == (1, 0, 0)
    result = env.last_handovers[0]
    assert result["allocated_rbs"] == 5
    assert result["success"] and not result["qos"]
    assert env.metrics.used_reserved_rbs == 5


def test_unused_surplus_is_released_after_success():
    env, ue = arrival_env(demand=6)
    hold(env, 8)
    assert cross(env, ue) == (1, 1, 2)
    assert env.metrics.used_reserved_rbs == 6
    assert env.allocations[1] == 6
    assert env._free(1) == 6
    assert not env.reservations


def test_rejected_activation_never_occupies_capacity():
    env, _ = arrival_env()
    hold(env, 8, active=False)
    env.allocations[1] = 6
    env._activate_due()
    assert env.metrics.activation_attempts == env.metrics.activation_rejects == 1
    assert env.metrics.activated_rbs == 0
    assert env._reservation_totals().sum() == 0
    assert not env.reservations
    assert env.allocations[1] == 6


def test_expiry_holds_through_inclusive_expiry_interval():
    env, _ = arrival_env()
    r = hold(env, 8)
    env.t = r.expires_at - 1
    assert env._expire() == 0
    assert env._reservation_totals()[1] == 8
    env.t += 1
    assert env._expire() == 8
    assert not env.reservations
    assert env.metrics.used_reserved_rbs == 0


def test_serial_activation_admits_only_capacity_feasible_requests():
    env, _ = arrival_env(capacity=10)
    first = hold(env, 6, active=False)
    second = Reservation(2, 1, 1, 6, 0, 0, 8, 1)
    env.reservations.append(second)
    env._activate_due()
    assert env.reservations == [first]
    assert first.active
    assert env._reservation_totals()[1] == 6
    assert env.metrics.activation_rejects == 1


def test_latency_proxy_matches_declared_load_and_shortfall_units():
    env, ue = arrival_env(demand=10)
    env.allocations[1] = 6
    cross(env, ue)
    # uRLLC base 2 ms, load 6/12, available 6/10 demand.
    expected = 2 * (1 + 0.75 * 0.5**2 + 3 * 0.4)
    assert env.last_handovers[0]["latency_ms"] == pytest.approx(expected)
    assert np.isfinite(expected)
