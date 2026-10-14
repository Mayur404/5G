"""Independent mathematical and causality checks for the learned forecaster."""

import numpy as np
import pytest

from proran.config import SimConfig
from proran.neural import HISTORY, HORIZON, NeuralPredictor, dataset, features
from proran.simulator import RANHandoverEnv


def test_neural_backpropagation_matches_finite_differences():
    model = NeuralPredictor(73)
    rng = np.random.default_rng(102)
    x = rng.normal(size=(3, model.architecture[0]))
    y = rng.normal(size=(3, model.architecture[-1]))
    _, gradients = model.gradients(x, y)
    for layer in range(3):
        for parameter, gradient in zip(
            (model.weights[layer], model.biases[layer]), gradients[layer]
        ):
            # Check interior and corner entries of every layer, including biases.
            for index in [tuple(0 for _ in parameter.shape), tuple(s - 1 for s in parameter.shape)]:
                value = parameter[index]
                parameter[index] = value + 1e-6
                plus = np.mean((model.forward(x) - y) ** 2)
                parameter[index] = value - 1e-6
                minus = np.mean((model.forward(x) - y) ** 2)
                parameter[index] = value
                assert gradient[index] == pytest.approx((plus - minus) / 2e-6, abs=1e-8)


def test_forecast_does_not_read_future_rng_or_traffic():
    model = NeuralPredictor()
    env = RANHandoverEnv(SimConfig(n_ues=3))
    before = repr(env.rng.bit_generator.state)
    before_predictor = repr(env.pred_rng.bit_generator.state)
    position, demand = model.forecast(env, env.ues[0])
    assert position.shape == (HORIZON, 2)
    assert demand.shape == (HORIZON,)
    assert demand[0] == env.ues[0].demand
    assert np.all(demand >= 1)
    assert repr(env.rng.bit_generator.state) == before
    assert repr(env.pred_rng.bit_generator.state) == before_predictor


def test_forecast_cache_is_invalidated_by_reset():
    model = NeuralPredictor()
    env = RANHandoverEnv(SimConfig(n_ues=3))
    model.forecast(env, env.ues[0])
    env.reset(999)
    actual = model.forecast(env, env.ues[0])
    expected = NeuralPredictor().forecast(env, env.ues[0])
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_dataset_labels_have_correct_interval_alignment():
    steps, n_ues, seed = 20, 2, 3111
    x, y = dataset([seed], steps=steps, n_ues=n_ues)
    env = RANHandoverEnv(SimConfig(n_ues=n_ues, episode_steps=steps))
    obs = env.reset(seed)
    frames = [env.observation_history[-1].copy()]
    for _ in range(steps):
        result = env.step(0, obs)
        obs = result.next_state, result.next_mask, result.info["next_context"]
        frames.append(env.observation_history[-1].copy())
    t = HISTORY - 1
    np.testing.assert_allclose(
        x[:n_ues],
        features(np.array(frames[:HISTORY]), np.array([u.slice_id for u in env.ues]), t, 100),
    )
    output = y[:n_ues].reshape(n_ues, HORIZON, 3)
    np.testing.assert_allclose(output[:, :, 2] * 20, np.array(frames)[t : t + HORIZON, :, 4].T)
    actual_positions = (
        output[:, :, :2] * 10
        + np.array(frames)[t, :, None, :2]
        + np.array(frames)[t, :, None, 2:4] * np.arange(1, HORIZON + 1)[None, :, None]
    )
    np.testing.assert_allclose(
        actual_positions, np.array(frames)[t + 1 : t + HORIZON + 1, :, :2].transpose(1, 0, 2)
    )


def test_saved_neural_forecasts_match_in_memory(tmp_path):
    model = NeuralPredictor()
    model.save(tmp_path / "predictor.npz", {"purpose": "roundtrip"})
    loaded = NeuralPredictor.load(tmp_path / "predictor.npz")
    x = np.random.default_rng(1).normal(size=(4, 35))
    np.testing.assert_array_equal(model.forward(x), loaded.forward(x))


def test_time_to_trigger_is_consistent_in_forecast_and_motion():
    env = RANHandoverEnv(
        SimConfig(n_ues=1, handover_mode="qos_aware", time_to_trigger=3, mobility_noise=0)
    )
    ue = env.ues[0]
    ue.cell = 0
    ue.pos[:] = [49, 25]
    ue.vel[:] = [0, 0]
    ue.demand = 10
    env.allocations[:] = [60, 0, 0, 0]
    env.ue_allocations = {ue.uid: 0}
    pred = env._predict(ue)
    # The noisy synthetic forecaster perturbs its ETA; the shared trigger itself
    # must wait for three observations and must not mutate real trigger state.
    assert pred is not None and not env.switch_candidates
    assert env._move() == []
    assert env._move() == []
    events = env._move()
    assert len(events) == 1 and events[0][1:] == (0, 1)
    assert ue.uid not in env.switch_candidates


def test_whole_network_service_metrics_use_current_interval_demand():
    env = RANHandoverEnv(SimConfig(n_ues=1, research_metrics=True, mobility_noise=0))
    ue = env.ues[0]
    ue.cell = 0
    ue.pos[:] = [25, 25]
    ue.vel[:] = [0, 0]
    ue.demand = 6
    env._allocate_incumbents(count_metrics=False)
    env.step(0, env.observe())
    metrics = env.metrics.summary(0.1)
    assert env.metrics.offered_rbs == env.metrics.served_rbs == 6
    assert metrics["full_service_rate"] == 1
    assert metrics["served_demand_fraction"] == 1


def test_unperturbed_kinematic_forecast_respects_time_to_trigger():
    env = RANHandoverEnv(
        SimConfig(
            n_ues=1, handover_mode="qos_aware", time_to_trigger=3, prediction_mode="kinematic"
        )
    )
    ue = env.ues[0]
    ue.cell = 0
    ue.pos[:] = [49, 25]
    ue.vel[:] = 0
    ue.demand = 10
    env.allocations[:] = [60, 0, 0, 0]
    env.ue_allocations = {ue.uid: 0}
    prediction = env._predict(ue)
    assert prediction.target == 1
    assert prediction.time_to_handover == 3
    assert not env.switch_candidates


@pytest.mark.parametrize("corruption", ["shape", "scale", "score"])
def test_corrupted_predictor_metadata_is_rejected(tmp_path, corruption):
    path = tmp_path / "bad.npz"
    NeuralPredictor().save(path, {})
    with np.load(path) as original:
        data = {key: original[key].copy() for key in original.files}
    if corruption == "shape":
        data["w0"] = np.ones((34, 64))
    elif corruption == "scale":
        data["std"][0] = 0
    else:
        data["forecast_score"][0] = np.nan
    np.savez(path, **data)
    with pytest.raises(ValueError):
        NeuralPredictor.load(path)
