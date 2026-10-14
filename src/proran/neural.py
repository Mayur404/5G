"""Causal multi-step mobility/demand prediction with a two-hidden-layer MLP.

All training uses synthetic observations, never controller actions or realized
future load as inputs. NumPy keeps the CPU experiment reproducible and small.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HISTORY = 6
HORIZON = 8


def features(history: np.ndarray, slices: np.ndarray, time: int, world: float) -> np.ndarray:
    """history: past/current frames x UEs x (x,y,vx,vy,demand)."""
    if len(history) < HISTORY:
        history = np.concatenate([np.repeat(history[:1], HISTORY - len(history), axis=0), history])
    values = history[-HISTORY:].copy()
    values[:, :, :2] /= world
    values[:, :, 2:4] /= 4
    values[:, :, 4] /= 20
    temporal = values.transpose(1, 0, 2).reshape(len(slices), -1)
    phase = np.tile(
        [np.sin(2 * np.pi * time / 47), np.cos(2 * np.pi * time / 47)], (len(slices), 1)
    )
    return np.concatenate([temporal, np.eye(3)[slices], phase], axis=1)


def kinematic_base(frame: np.ndarray, world: float) -> np.ndarray:
    return (
        frame[:, None, :2] + frame[:, None, 2:4] * np.arange(1, HORIZON + 1)[None, :, None]
    ) / world


def dataset(seeds, steps=240, n_ues=32):
    from .config import SimConfig
    from .simulator import RANHandoverEnv

    xs, ys = [], []
    for seed in seeds:
        env = RANHandoverEnv(SimConfig(seed=seed, n_ues=n_ues, episode_steps=steps))
        obs = env.reset(seed)
        frames = [env.observation_history[-1].copy()]
        for _ in range(steps):
            result = env.step(0, obs)
            obs = (result.next_state, result.next_mask, result.info["next_context"])
            frames.append(env.observation_history[-1].copy())
        frames = np.array(frames)
        slices = np.array([ue.slice_id for ue in env.ues])
        for t in range(HISTORY - 1, len(frames) - HORIZON):
            xs.append(
                features(frames[max(0, t - HISTORY + 1) : t + 1], slices, t, env.cfg.world_size)
            )
            future = frames[t + 1 : t + HORIZON + 1, :, :2].transpose(1, 0, 2) / env.cfg.world_size
            # Learn corrections to constant-velocity projection; scale small
            # positional residuals so demand does not dominate the objective.
            residual = (future - kinematic_base(frames[t], env.cfg.world_size)) / 0.1
            demands = frames[t : t + HORIZON, :, 4].T[:, :, None] / 20
            ys.append(np.concatenate([residual, demands], axis=2).reshape(n_ues, -1))
    return np.concatenate(xs), np.concatenate(ys)


class NeuralPredictor:
    architecture = (35, 64, 32, 24)

    def __init__(self, seed=73):
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.normal(0, np.sqrt(1 / a), (a, b))
            for a, b in zip(self.architecture[:-1], self.architecture[1:], strict=True)
        ]
        self.biases = [np.zeros(b) for b in self.architecture[1:]]
        self.mean = np.zeros(self.architecture[0])
        self.std = np.ones(self.architecture[0])
        self.forecast_score = 0.7
        self._cache_env = None
        self._cache_time = None
        self._cache_frame = None

    def forward(self, x, cache=False):
        activations = [(x - self.mean) / self.std]
        for index, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            value = activations[-1] @ w + b
            activations.append(np.tanh(value) if index < len(self.weights) - 1 else value)
        return (activations[-1], activations) if cache else activations[-1]

    def gradients(self, x, y):
        prediction, activations = self.forward(x, True)
        delta = 2 * (prediction - y) / prediction.size
        grads = []
        for index in reversed(range(len(self.weights))):
            grads.append((activations[index].T @ delta, delta.sum(axis=0)))
            if index:
                delta = (delta @ self.weights[index].T) * (1 - activations[index] ** 2)
        return float(np.mean((prediction - y) ** 2)), list(reversed(grads))

    def fit(self, x, y, validation, epochs=30, batch_size=512, seed=73):
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 0.05)
        params = [p for pair in zip(self.weights, self.biases, strict=True) for p in pair]
        moments = [np.zeros_like(p) for p in params]
        variances = [np.zeros_like(p) for p in params]
        rng = np.random.default_rng(seed)
        step = 0
        history = []
        best = np.inf
        best_params = None
        for epoch in range(epochs):
            order = rng.permutation(len(x))
            losses = []
            for start in range(0, len(x), batch_size):
                ids = order[start : start + batch_size]
                loss, gradient = self.gradients(x[ids], y[ids])
                losses.append(loss)
                step += 1
                for i, (parameter, grad) in enumerate(
                    zip(params, [g for pair in gradient for g in pair], strict=True)
                ):
                    moments[i] = 0.9 * moments[i] + 0.1 * grad
                    variances[i] = 0.999 * variances[i] + 0.001 * grad**2
                    parameter -= (
                        0.001
                        * (moments[i] / (1 - 0.9**step))
                        / (np.sqrt(variances[i] / (1 - 0.999**step)) + 1e-8)
                    )
            val = float(np.mean((self.forward(validation[0]) - validation[1]) ** 2))
            history.append(
                {"epoch": epoch + 1, "training_mse": float(np.mean(losses)), "validation_mse": val}
            )
            if val < best:
                best = val
                best_params = [p.copy() for p in params]
            print(
                f"Neural predictor epoch {epoch + 1:02d}: train {np.mean(losses):.6f}, validation {val:.6f}",
                flush=True,
            )
        for parameter, best_parameter in zip(params, best_params, strict=True):
            parameter[:] = best_parameter
        residual = self.forward(validation[0]).reshape(-1, HORIZON, 3) - validation[1].reshape(
            -1, HORIZON, 3
        )
        position_rmse = float(np.sqrt(np.mean(np.sum((residual[:, :, :2] * 10) ** 2, axis=2))))
        self.forecast_score = float(np.clip(np.exp(-position_rmse / 10), 0.1, 0.95))
        return history, {
            "position_rmse_world_units": position_rmse,
            "demand_mae_rbs": float(np.mean(np.abs(residual[:, :, 2] * 20))),
            "forecast_score": self.forecast_score,
            "score_is_calibrated_probability": False,
        }

    def forecast(self, env, ue):
        if env.cfg.horizon != HORIZON:
            raise ValueError("This saved predictor supports exactly eight forecast intervals")
        # The frame identity also changes on reset, where env and time can be
        # identical to the previous episode. Never replay a stale forecast.
        if (
            self._cache_env is not env
            or self._cache_time != env.t
            or self._cache_frame is not env.observation_history[-1]
        ):
            frame = env.observation_history[-1]
            x = features(
                np.array(env.observation_history),
                np.array([u.slice_id for u in env.ues]),
                env.t,
                env.cfg.world_size,
            )
            output = self.forward(x).reshape(len(env.ues), HORIZON, 3)
            self._positions = np.clip(
                (kinematic_base(frame, env.cfg.world_size) + output[:, :, :2] * 0.1)
                * env.cfg.world_size,
                0,
                env.cfg.world_size,
            )
            self._demands = np.maximum(1, np.rint(output[:, :, 2] * 20)).astype(int)
            self._demands[:, 0] = frame[:, 4].astype(int)
            self._cache_env = env
            self._cache_time = env.t
            self._cache_frame = frame
        return self._positions[ue.uid], self._demands[ue.uid]

    def save(self, path, metadata):
        np.savez(
            path,
            **{f"w{i}": w for i, w in enumerate(self.weights)},
            **{f"b{i}": b for i, b in enumerate(self.biases)},
            mean=self.mean,
            std=self.std,
            forecast_score=np.array([self.forecast_score]),
            metadata=np.array(json.dumps(metadata)),
        )

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path, allow_pickle=False) as data:
            model.weights = [data[f"w{i}"].copy() for i in range(3)]
            model.biases = [data[f"b{i}"].copy() for i in range(3)]
            model.mean = data["mean"].copy()
            model.std = data["std"].copy()
            model.forecast_score = float(data["forecast_score"][0])
        if not all(
            np.isfinite(p).all() for p in [*model.weights, *model.biases, model.mean, model.std]
        ):
            raise ValueError("Neural model parameters must be finite")
        if any(
            w.shape != (a, b) or bias.shape != (b,)
            for w, bias, a, b in zip(
                model.weights,
                model.biases,
                model.architecture[:-1],
                model.architecture[1:],
                strict=True,
            )
        ):
            raise ValueError("Neural model has incompatible layer dimensions")
        if (
            model.mean.shape != (35,)
            or model.std.shape != (35,)
            or np.any(model.std <= 0)
            or not 0 <= model.forecast_score <= 1
        ):
            raise ValueError("Neural model normalization and forecast score are invalid")
        for i, (a, b) in enumerate(zip(model.architecture[:-1], model.architecture[1:])):
            if model.weights[i].shape != (a, b) or model.biases[i].shape != (b,):
                raise ValueError("Neural model parameters have incompatible shapes")
        if (
            model.mean.shape != (model.architecture[0],)
            or model.std.shape != model.mean.shape
            or np.any(model.std <= 0)
        ):
            raise ValueError("Neural normalization must have valid shapes and positive scales")
        if not np.isfinite(model.forecast_score) or not 0 <= model.forecast_score <= 1:
            raise ValueError("Neural forecast score must be finite and between zero and one")
        return model


def proactive(state, mask, context):
    """Fixed full/half midpoint reservation rule, shared by predictor ablations."""
    pred = context.get("prediction")
    if pred is None or pred.probability < 0.5:
        return 0
    return next((a for a in (4, 3, 6, 5, 2, 1) if mask[a]), 0)


def train(output="results/deep", epochs=30):
    import csv

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    training_seeds = list(range(3000, 3024))
    validation_seeds = list(range(4000, 4006))
    print("Generating independent synthetic training trajectories", flush=True)
    x, y = dataset(training_seeds)
    validation = dataset(validation_seeds)
    model = NeuralPredictor()
    history, metrics = model.fit(x, y, validation, epochs=epochs)
    metadata = {
        "architecture": list(model.architecture),
        "hidden_activation": "tanh",
        "history_steps": HISTORY,
        "horizon_steps": HORIZON,
        "training_seeds": training_seeds,
        "validation_seeds": validation_seeds,
        "test_seeds": list(range(9000, 9020)),
        "training_examples": len(x),
        "validation_examples": len(validation[0]),
        "validation": metrics,
        "training_seed": 73,
        "epochs": epochs,
        "uses_future_inputs": False,
    }
    model.save(destination / "predictor.npz", metadata)
    (destination / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    with (destination / "predictor_training.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return model, metadata


if __name__ == "__main__":
    train()
