import numpy as np
import pytest

from proran.config import SimConfig
from proran.neural import NeuralPredictor, features, proactive
from proran.simulator import RANHandoverEnv


def test_neural_backprop_matches_finite_differences():
    rng = np.random.default_rng(5)
    model = NeuralPredictor()
    x = rng.normal(size=(4, 35))
    y = rng.normal(size=(4, 24))
    _, gradients = model.gradients(x, y)
    for layer in range(3):
        for parameter, gradient in zip(
            (model.weights[layer], model.biases[layer]), gradients[layer], strict=True
        ):
            index = tuple(0 for _ in parameter.shape)
            original = parameter[index]
            eps = 1e-6
            parameter[index] = original + eps
            plus = np.mean((model.forward(x) - y) ** 2)
            parameter[index] = original - eps
            minus = np.mean((model.forward(x) - y) ** 2)
            parameter[index] = original
            assert gradient[index] == pytest.approx((plus - minus) / (2 * eps), abs=1e-8)


def test_features_use_only_passed_history_and_are_finite():
    history = np.ones((2, 3, 5))
    slices = np.array([0, 1, 2])
    x = features(history, slices, 10, 100)
    assert x.shape == (3, 35) and np.isfinite(x).all()
    assert np.array_equal(x[:, :5], x[:, 5:10])
    assert np.array_equal(x[:, 30:33], np.eye(3))


def test_predictor_save_load_preserves_output(tmp_path):
    model = NeuralPredictor()
    x = np.ones((3, 35))
    path = tmp_path / "model.npz"
    model.save(path, {"test": True})
    restored = NeuralPredictor.load(path)
    assert np.array_equal(model.forward(x), restored.forward(x))


def test_qos_target_prefers_serviceable_neighbor_over_loaded_nearest():
    env = RANHandoverEnv(SimConfig(n_ues=1, handover_mode="qos_aware", cell_capacity=20))
    ue = env.ues[0]
    ue.pos = np.array([49.0, 25.0])
    ue.cell = 0
    ue.demand = 10
    env.allocations[:] = [20, 0, 0, 0]
    env.ue_allocations = {ue.uid: 0}
    assert env._nearest_cell(ue.pos) == 0
    assert env.target_cell(ue, ue.pos) == 1
    assert env.signal_quality(ue.pos, 1) >= env.cfg.minimum_signal


def test_hysteresis_retains_serving_cell_on_small_score_difference():
    env = RANHandoverEnv(SimConfig(n_ues=1, handover_mode="qos_aware"))
    ue = env.ues[0]
    ue.cell = 0
    ue.pos = np.array([50.01, 25.0])
    ue.demand = 5
    env.allocations[:] = 0
    env.ue_allocations = {}
    assert env._nearest_cell(ue.pos) == 1
    assert env.target_cell(ue, ue.pos) == 0


def test_common_radio_and_mobility_streams_are_identical_between_modes():
    left = RANHandoverEnv(SimConfig(seed=9, common_radio_stream=True))
    right = RANHandoverEnv(SimConfig(seed=9, common_radio_stream=True, handover_mode="qos_aware"))
    observations = [left.observe(), right.observe()]
    for _ in range(25):
        for i, env in enumerate((left, right)):
            result = env.step(0, observations[i])
            observations[i] = (result.next_state, result.next_mask, result.info["next_context"])
        assert np.array_equal(left.interval_radio, right.interval_radio)
        for a, b in zip(left.ues, right.ues, strict=True):
            assert np.array_equal(a.pos, b.pos) and a.demand == b.demand


def test_neural_reservation_rule_respects_mask():
    class Pred:
        probability = 0.8

    mask = np.array([True, False, False, True, False, False, False])
    assert proactive(None, mask, {"prediction": Pred()}) == 3
