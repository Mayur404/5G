"""Independent forecast and whole-network service validation.

This command does not select a model or tune any parameters on test seeds.
It complements event-level benchmark statistics with all-UE service measures
and checks whether the learned forecast improves on simple predictors.
"""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from .config import SimConfig
from .deep_experiment import METHODS, evaluate, write_csv
from .neural import HISTORY, HORIZON, NeuralPredictor, features, kinematic_base
from .simulator import RANHandoverEnv


def forecast_accuracy(model, seeds, steps=320):
    rows = []
    for seed in seeds:
        env = RANHandoverEnv(SimConfig(seed=seed, episode_steps=steps))
        obs = env.reset(seed)
        frames = [env.observation_history[-1].copy()]
        for _ in range(steps):
            result = env.step(0, obs)
            obs = result.next_state, result.next_mask, result.info["next_context"]
            frames.append(env.observation_history[-1].copy())
        frames = np.array(frames)
        slices = np.array([u.slice_id for u in env.ues])
        times = range(HISTORY - 1, len(frames) - HORIZON)
        x = np.concatenate(
            [features(frames[t - HISTORY + 1 : t + 1], slices, t, 100) for t in times]
        )
        output = model.forward(x).reshape(len(times), len(slices), HORIZON, 3)
        current = frames[list(times)]
        base = np.array([kinematic_base(frame, 100) * 100 for frame in current])
        neural_positions = np.clip(base + output[:, :, :, :2] * 10, 0, 100)
        truth_positions = np.array(
            [frames[t + 1 : t + HORIZON + 1, :, :2].transpose(1, 0, 2) for t in times]
        )
        truth_demands = np.array([frames[t : t + HORIZON, :, 4].T for t in times])
        neural_demands = np.maximum(1, np.rint(output[:, :, :, 2] * 20))
        neural_demands[:, :, 0] = current[:, :, 4]
        persistence = np.repeat(current[:, :, None, 4], HORIZON, axis=2)
        positions = {"Neural": neural_positions, "Constant velocity": np.clip(base, 0, 100)}
        demands = {"Neural": neural_demands, "Persistence": persistence}
        for horizon in range(HORIZON):
            for sid, name in enumerate(("eMBB", "uRLLC", "mMTC")):
                members = slices == sid
                if not np.any(members):
                    continue
                for method, predictions in positions.items():
                    errors = predictions[:, members, horizon] - truth_positions[:, members, horizon]
                    rows.append(
                        {
                            "seed": seed,
                            "slice": name,
                            "lead_intervals": horizon + 1,
                            "quantity": "position",
                            "method": method,
                            "n": int(errors.shape[0] * errors.shape[1]),
                            "error": float(np.sqrt(np.mean(np.sum(errors**2, axis=-1)))),
                        }
                    )
                for method, predictions in demands.items():
                    error = np.abs(
                        predictions[:, members, horizon] - truth_demands[:, members, horizon]
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "slice": name,
                            "lead_intervals": horizon + 1,
                            "quantity": "demand",
                            "method": method,
                            "n": int(error.size),
                            "error": float(error.mean()),
                        }
                    )
        print(f"Verified held-out forecast seed {seed}", flush=True)
    return rows


def paired_service_intervals(runs):
    proposed = "QoS-aware / neural reservation"
    metrics = (
        "full_service_rate",
        "sla_service_rate",
        "served_demand_fraction",
        "handovers_per_ue_second",
        "pingpong_fraction",
        "jain_served_demand_fairness",
    )
    left = {r["seed"]: r for r in runs if r["method"] == proposed}
    rows = []
    for reference in list(METHODS)[:-1]:
        right = {r["seed"]: r for r in runs if r["method"] == reference}
        for metric in metrics:
            differences = np.array([left[s][metric] - right[s][metric] for s in sorted(left)])
            differences = differences[np.isfinite(differences)]
            half = 2.093024054408263 * differences.std(ddof=1) / np.sqrt(len(differences))
            rows.append(
                {
                    "reference": reference,
                    "metric": metric,
                    "n": len(differences),
                    "mean_difference": float(differences.mean()),
                    "ci95_low": float(differences.mean() - half),
                    "ci95_high": float(differences.mean() + half),
                }
            )
    return rows


def main():
    source_names = ("simulator.py", "neural.py", "models.py", "config.py")
    source_hashes = {
        name: hashlib.sha256(Path("src/proran", name).read_bytes()).hexdigest()
        for name in source_names
    }
    destination = Path("results/research_validation")
    destination.mkdir(parents=True, exist_ok=True)
    model_path = Path("results/deep/predictor.npz")
    model = NeuralPredictor.load(model_path)
    seeds = list(range(9000, 9020))
    steps = 320
    forecast = forecast_accuracy(model, seeds, steps)
    write_csv(destination / "forecast_errors_by_seed_slice_horizon.csv", forecast)
    cfg = SimConfig(common_radio_stream=True, research_metrics=True, prediction_mode="kinematic")
    aggregates = []
    runs = []
    for method in METHODS:
        print("Whole-network service validation: " + method, flush=True)
        aggregate, individual, _ = evaluate(method, cfg, seeds, steps, model)
        aggregates.append(aggregate)
        runs.extend(individual)
    intervals = paired_service_intervals(runs)
    current_hashes = {
        name: hashlib.sha256(Path("src/proran", name).read_bytes()).hexdigest()
        for name in source_names
    }
    if current_hashes != source_hashes:
        raise RuntimeError(
            "Simulator source changed during validation; rerun against the final source before using these results"
        )
    write_csv(destination / "service_metrics.csv", aggregates)
    write_csv(destination / "service_metrics_by_seed.csv", runs)
    write_csv(destination / "paired_service_intervals.csv", intervals)
    report = {
        "status": "passed",
        "test_seeds": seeds,
        "steps_per_run": steps,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "source_sha256": source_hashes,
        "simulation": cfg.as_dict(),
        "predictor_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "forecast_rows": len(forecast),
        "network_runs": len(runs),
        "metrics": aggregates,
        "paired_service_intervals": intervals,
        "definitions": {
            "full_service_rate": "Covered UE intervals with granted RBs >= observed interval demand / all UE intervals",
            "sla_service_rate": "Covered UE intervals with granted RBs >= slice SLA RB floor / all UE intervals",
            "served_demand_fraction": "Covered granted RBs / offered RBs, over all UE intervals",
            "handovers_per_ue_second": "Attempts / (UE intervals times dt)",
            "pingpong_fraction": "Returns to the immediately previous cell within 10 intervals / all attempts",
        },
        "statistical_scope": "Fixed model; unadjusted descriptive paired t intervals; one training initialization",
    }
    (destination / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
