from dataclasses import replace

import pytest

from proran.actions import decode_action
from proran.config import SimConfig, TrainConfig


@pytest.mark.parametrize(
    "changes",
    [
        {"cell_capacity": 10.5},
        {"cell_capacity": 0},
        {"n_ues": -1},
        {"horizon": 0},
        {"dt_s": 0},
        {"dt_s": float("nan")},
        {"prediction_noise": -0.1},
        {"urllc_burst_prob": 1.1},
        {"centers": ()},
        {"centers": ((1.0, 1.0), (1.0, 1.0))},
        {"centers": ((float("nan"), 1.0),)},
        {"centers": ((101.0, 1.0),)},
    ],
)
def test_invalid_model_domains_are_rejected(changes):
    with pytest.raises(ValueError):
        SimConfig(**changes)


def test_small_explicit_topology_and_variable_slice_demands_are_supported():
    cfg = SimConfig(world_size=20, centers=((5.0, 5.0), (15.0, 15.0)), cell_capacity=3)
    # An infeasible minimum is a legitimate overloaded experiment, not a
    # configuration error: admission should fail rather than invent resources.
    assert cfg.total_capacity == 6
    slices = tuple(replace(s, demand_std=0) for s in cfg.slices)
    assert SimConfig(slices=slices).slices == slices


@pytest.mark.parametrize("arguments", [(-1, 9, 2), (7, 9, 2), (1, 0, 2), (2, 9, 0)])
def test_decoder_rejects_out_of_domain_actions(arguments):
    with pytest.raises(ValueError):
        decode_action(*arguments)


def test_integer_rounding_preserves_action_bounds_even_for_one_rb():
    for demand in range(1, 15):
        for arrival in range(1, 9):
            for action in range(1, 7):
                amount, delay = decode_action(action, demand, arrival)
                assert 1 <= amount <= demand
                assert 0 <= delay < arrival
    assert decode_action(0, 0, 0) == (0, 0)


@pytest.mark.parametrize(
    "changes",
    [
        {"gamma": 1.01},
        {"gae_lambda": -0.1},
        {"clip": 0},
        {"clip": 1},
        {"actor_lr": float("inf")},
        {"critic_lr": 0},
        {"entropy_coef": -0.1},
        {"episodes": 0},
        {"update_epochs": 0},
        {"imitation_episodes": -1},
    ],
)
def test_invalid_training_domains_are_rejected(changes):
    with pytest.raises(ValueError):
        TrainConfig(**changes)


def test_discount_and_lambda_endpoints_and_no_imitation_are_legal():
    assert TrainConfig(gamma=1, gae_lambda=0, imitation_episodes=0).gamma == 1
