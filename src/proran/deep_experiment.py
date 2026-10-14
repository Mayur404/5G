"""Versioned factorial evaluation of prediction, reservation and target choice."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import SimConfig
from .neural import NeuralPredictor, proactive
from .policies import no_reservation
from .simulator import RANHandoverEnv

METHODS = {
    "Nearest / no reservation": ("nearest", False, False),
    "Nearest / kinematic reservation": ("nearest", False, True),
    "Nearest / neural reservation": ("nearest", True, True),
    "QoS-aware / no reservation": ("qos_aware", False, False),
    "QoS-aware / kinematic reservation": ("qos_aware", False, True),
    "QoS-aware / neural reservation": ("qos_aware", True, True),
}


def evaluate(name, cfg, seeds, steps, model):
    mode, neural, reserve = METHODS[name]
    rows = []
    counts = []
    latencies = []
    for seed in seeds:
        env = RANHandoverEnv(replace(cfg, seed=seed, handover_mode=mode, episode_steps=steps))
        env.neural_predictor = model if neural else None
        observation = env.observe()
        full_service = minimum_service = weak_samples = unmet_rbs = total_demand = ping_pongs = 0
        last_crossing = {}
        for _ in range(steps):
            action = (proactive if reserve else no_reservation)(*observation)
            result = env.step(action, observation)
            observation = (result.next_state, result.next_mask, result.info["next_context"])
            assert np.all(env.allocations + env._reservation_totals() <= cfg.cell_capacity)
            assert np.all(env.allocations >= 0)
            for ue in env.ues:
                allocated = env.ue_allocations.get(ue.uid, 0)
                demand = env.interval_demands[ue.uid]
                full_service += allocated >= demand
                minimum_service += allocated >= env.cfg.slices[ue.slice_id].sla_rbs
                weak_samples += env.signal_quality(ue.pos, ue.cell) < 0.4
                unmet_rbs += max(0, demand - allocated)
                total_demand += demand
            for event in result.info["handover_details"]:
                previous = last_crossing.get(event["ue"])
                ping_pongs += bool(
                    previous
                    and env.t - previous[0] <= 10
                    and previous[1] == event["target"]
                    and previous[2] == event["source"]
                )
                last_crossing[event["ue"]] = (env.t, event["source"], event["target"])
        for r in list(env.reservations):
            env._finish_contract(r, 0, True)
            env.reservations.remove(r)
        summary = env.metrics.summary(cfg.dt_s)
        summary.update(
            handover_attempts=env.metrics.handovers,
            successful_handovers=env.metrics.successes,
            qos_handovers=env.metrics.qos_successes,
        )
        summary.update(
            full_demand_service_rate=full_service / (steps * cfg.n_ues),
            minimum_service_rate=minimum_service / (steps * cfg.n_ues),
            weak_signal_fraction=weak_samples / (steps * cfg.n_ues),
            unmet_demand_fraction=unmet_rbs / total_demand,
            ping_pong_fraction=ping_pongs / env.metrics.handovers
            if env.metrics.handovers
            else float("nan"),
        )
        rows.append({"method": name, "seed": seed, **summary})
        counts.append({"method": name, "seed": seed, "handovers": env.metrics.handovers})
        latencies.extend(
            {"method": name, "seed": seed, "slice_id": sid, "latency_ms": latency}
            for sid, latency in zip(
                env.metrics.latency_slice_ids, env.metrics.latency_ms, strict=True
            )
        )
    aggregate = {"method": name, "runs": len(seeds)}
    for key in rows[0]:
        if key in {"method", "seed"}:
            continue
        values = np.array([row[key] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        aggregate[key] = float(values.mean()) if len(values) else None
        aggregate[key + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
    return aggregate, rows, latencies


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired(rows):
    names = list(METHODS)
    comparisons = []
    proposed = names[-1]
    for reference in names[:-1]:
        left = {row["seed"]: row for row in rows if row["method"] == proposed}
        right = {row["seed"]: row for row in rows if row["method"] == reference}
        for metric in (
            "handover_success_rate",
            "qos_rate",
            "reservation_occupancy",
            "rb_utilization",
            "block_rate",
            "handover_attempts",
            "full_demand_service_rate",
            "unmet_demand_fraction",
            "ping_pong_fraction",
        ):
            differences = np.array([left[s][metric] - right[s][metric] for s in sorted(left)])
            assert np.isfinite(differences).all()
            half = 2.093024054408263 * differences.std(ddof=1) / np.sqrt(len(differences))
            comparisons.append(
                {
                    "proposed": proposed,
                    "reference": reference,
                    "metric": metric,
                    "n": len(differences),
                    "mean_difference": float(differences.mean()),
                    "ci95_low": float(differences.mean() - half),
                    "ci95_high": float(differences.mean() + half),
                }
            )
    return comparisons


def main(output="results/deep"):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "predictor.npz"
    model = NeuralPredictor.load(model_path)
    cfg = SimConfig(common_radio_stream=True, prediction_mode="kinematic")
    seeds = list(range(10000, 10020))
    steps = 320
    aggregate = []
    runs = []
    latencies = []
    for name in METHODS:
        print("Evaluating " + name, flush=True)
        row, individual, samples = evaluate(name, cfg, seeds, steps, model)
        aggregate.append(row)
        runs.extend(individual)
        latencies.extend(samples)
        print(
            f"  HSR {row['handover_success_rate']:.4f}, QoS {row['qos_rate']:.4f}, handovers {row['handover_attempts']:.1f}",
            flush=True,
        )
    comparisons = paired(runs)
    write_csv(directory / "benchmark_metrics.csv", aggregate)
    write_csv(directory / "per_seed_metrics.csv", runs)
    write_csv(directory / "paired_comparisons.csv", comparisons)
    write_csv(directory / "latency_samples.csv", latencies)
    report = {
        "experiment_version": "deep-v1-final",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "simulation": cfg.as_dict(),
        "evaluation_seeds": seeds,
        "steps_per_run": steps,
        "methods": METHODS,
        "predictor_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "metrics": aggregate,
        "paired": comparisons,
        "confidence_intervals": "unadjusted paired Student t, df=19, descriptive exploratory comparisons",
        "handover_denominators_differ_between_target_policies": True,
    }
    (directory / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Saved new results without replacing the legacy PPO experiment.", flush=True)


if __name__ == "__main__":
    main()
