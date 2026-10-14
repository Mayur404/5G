"""Exploratory frozen-model workload checks on new, disjoint trajectory seeds."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import SimConfig
from .deep_experiment import evaluate, write_csv
from .neural import NeuralPredictor

WORKLOADS = {
    "light": {"n_ues": 16},
    "dense": {"n_ues": 48},
    "bursty": {"n_ues": 32, "urllc_burst_prob": 0.25},
    "scarce": {"n_ues": 32, "cell_capacity": 40},
}
METHODS = (
    "Nearest / no reservation",
    "QoS-aware / kinematic reservation",
    "QoS-aware / neural reservation",
)


def main():
    directory = Path("results/robustness_validation")
    directory.mkdir(parents=True, exist_ok=True)
    model_path = Path("results/deep/predictor.npz")
    model = NeuralPredictor.load(model_path)
    source_names = ("simulator.py", "neural.py", "models.py", "config.py")
    source_hashes = {
        n: hashlib.sha256(Path("src/proran", n).read_bytes()).hexdigest() for n in source_names
    }
    seeds = list(range(15000, 15008))
    summaries = []
    runs = []
    paired = []
    for workload, overrides in WORKLOADS.items():
        cfg = SimConfig(
            common_radio_stream=True,
            research_metrics=True,
            prediction_mode="kinematic",
            **overrides,
        )
        local = []
        for method in METHODS:
            print(f"Frozen-model workload {workload}: {method}", flush=True)
            aggregate, individual, _ = evaluate(method, cfg, seeds, 240, model)
            summaries.append({"workload": workload, **aggregate})
            local.extend(individual)
            runs.extend({"workload": workload, **r} for r in individual)
        proposed = {r["seed"]: r for r in local if r["method"] == METHODS[-1]}
        for reference in METHODS[:-1]:
            comparison = {r["seed"]: r for r in local if r["method"] == reference}
            for metric in (
                "handover_success_rate",
                "qos_rate",
                "full_service_rate",
                "served_demand_fraction",
                "reservation_occupancy",
                "pingpong_fraction",
            ):
                delta = np.array([proposed[s][metric] - comparison[s][metric] for s in seeds])
                assert np.isfinite(delta).all()
                half = 2.364624251 * delta.std(ddof=1) / np.sqrt(8)
                paired.append(
                    {
                        "workload": workload,
                        "reference": reference,
                        "metric": metric,
                        "n": 8,
                        "mean_difference": float(delta.mean()),
                        "ci95_low": float(delta.mean() - half),
                        "ci95_high": float(delta.mean() + half),
                    }
                )
        write_csv(directory / "workload_metrics.csv", summaries)
        write_csv(directory / "workload_metrics_by_seed.csv", runs)
        write_csv(directory / "paired_workload_intervals.csv", paired)
    if source_hashes != {
        n: hashlib.sha256(Path("src/proran", n).read_bytes()).hexdigest() for n in source_names
    }:
        raise RuntimeError("Source changed during workload evaluation; rerun before citing results")
    report = {
        "status": "passed",
        "workloads": WORKLOADS,
        "test_seeds": seeds,
        "steps_per_run": 240,
        "runs": len(runs),
        "predictor_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "metrics": summaries,
        "paired": paired,
        "scope": "Exploratory eight-seed frozen-model out-of-training-workload checks; unadjusted paired t intervals, df=7",
    }
    (directory / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
