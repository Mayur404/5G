from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import numpy as np

from .config import SimConfig, TrainConfig
from .policies import BASELINES, SavedPPOPolicy
from .ppo import train_ppo
from .simulator import RANHandoverEnv


def _nanmean(values: list[float]) -> float:
    a = np.asarray(values, dtype=float)
    return float(np.nanmean(a)) if np.any(np.isfinite(a)) else float("nan")


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate_policy_detailed(name: str, policy, cfg: SimConfig, seeds: list[int], steps: int) -> tuple[dict, list[float], list[dict], list[dict]]:
    summaries, latencies, latency_samples = [], [], []
    for seed in seeds:
        run_cfg = SimConfig(**{**cfg.__dict__, "episode_steps": steps})
        env = RANHandoverEnv(run_cfg)
        metrics = env.run(policy, steps=steps, seed=seed)
        summaries.append(metrics.summary(cfg.dt_s))
        latencies.extend(metrics.latency_ms)
        latency_samples.extend({"method": name, "seed": seed, "slice_id": sid,
                                "latency_ms": latency}
                               for sid, latency in zip(metrics.latency_slice_ids,
                                                       metrics.latency_ms, strict=True))
    aggregate = {"method": name, "runs": len(seeds)}
    for key in summaries[0]:
        values = [s[key] for s in summaries]
        aggregate[key] = _nanmean(values)
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        aggregate[f"{key}_std"] = float(np.std(finite)) if finite.size else float("nan")
    return aggregate, latencies, summaries, latency_samples


def evaluate_policy(name: str, policy, cfg: SimConfig, seeds: list[int], steps: int) -> tuple[dict, list[float]]:
    aggregate, latencies, _, _ = evaluate_policy_detailed(name, policy, cfg, seeds, steps)
    return aggregate, latencies


def _threshold_policy(threshold: float):
    def policy(state, mask, context):
        pred = context.get("prediction")
        if pred is None:
            return 0
        score = pred.probability * (1 - pred.target_uncertainty)
        if score < threshold:
            return 0
        for action in (4, 6, 2, 3, 5, 1):
            if mask[action]:
                return action
        return 0
    return policy


def run_experiments(output_dir: str | Path = "results", sim_cfg: SimConfig | None = None,
                    train_cfg: TrainConfig | None = None, retrain: bool = True) -> dict:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = sim_cfg or SimConfig()
    train_cfg = train_cfg or TrainConfig()
    model_path = output_dir / "ppo_model.npz"
    history_path = output_dir / "training_history.csv"
    if retrain or not model_path.exists():
        agent, history = train_ppo(sim_cfg, train_cfg, output_dir)
        ppo_policy = agent.policy(deterministic=True)
    else:
        ppo_policy = SavedPPOPolicy.load(str(model_path))
        with history_path.open(encoding="utf-8") as fh:
            history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]

    methods = dict(BASELINES)
    methods["PPO (proposed)"] = ppo_policy
    seeds = [9000 + i for i in range(20)]
    rows, run_summaries, latency_samples = [], {}, []
    for name, policy in methods.items():
        row, _latencies, summaries, samples = evaluate_policy_detailed(name, policy, sim_cfg, seeds, 320)
        rows.append(row)
        run_summaries[name] = summaries
        latency_samples.extend(samples)

    metric_fields = list(rows[0].keys())
    with (output_dir / "benchmark_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=metric_fields)
        writer.writeheader(); writer.writerows(rows)

    per_seed_rows = []
    for method, summaries in run_summaries.items():
        for seed, summary in zip(seeds, summaries, strict=True):
            per_seed_rows.append({"method": method, "seed": seed, **summary})
    with (output_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=per_seed_rows[0].keys())
        writer.writeheader(); writer.writerows(per_seed_rows)
    with (output_dir / "latency_samples.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "seed", "slice_id", "latency_ms"])
        writer.writeheader(); writer.writerows(latency_samples)
    paired_rows = _paired_comparisons(run_summaries)
    with (output_dir / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=paired_rows[0].keys())
        writer.writeheader(); writer.writerows(paired_rows)

    pareto = []
    pareto_seeds = [12000 + i for i in range(8)]
    for threshold in np.linspace(0.20, 0.82, 9):
        row, _ = evaluate_policy(f"threshold={threshold:.3f}", _threshold_policy(float(threshold)),
                                 sim_cfg, pareto_seeds, 260)
        row["threshold"] = float(threshold)
        pareto.append(row)
    with (output_dir / "pareto_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=pareto[0].keys())
        writer.writeheader(); writer.writerows(pareto)

    from .plotting import plot_convergence, plot_latency_cdfs, plot_pareto, plot_learning_stages
    plot_convergence(history, fig_dir)
    imitation_path = output_dir / "imitation_history.csv"
    if imitation_path.exists():
        with imitation_path.open(encoding="utf-8") as fh:
            imitation = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]
        plot_learning_stages(imitation, history, fig_dir)
    plot_pareto(pareto, rows, fig_dir)
    plot_latency_cdfs(latency_samples, fig_dir, sim_cfg)

    payload = {"simulation": sim_cfg.as_dict(), "training": train_cfg.__dict__,
               "evaluation_seeds": seeds, "metrics": rows}
    with (output_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2, allow_nan=False)
    _write_latex_macros(rows, paired_rows, output_dir / "results_macros.tex")
    return payload


def _paired_comparisons(run_summaries: dict[str, list[dict]]) -> list[dict]:
    rows = []
    proposed = run_summaries["PPO (proposed)"]
    metrics = ("handover_success_rate", "qos_rate", "reservation_occupancy", "block_rate")
    for reference in ("No reservation", "Uncertainty-aware"):
        for metric in metrics:
            differences = np.asarray([p[metric] - r[metric]
                                      for p, r in zip(proposed, run_summaries[reference], strict=True)], dtype=float)
            differences = differences[np.isfinite(differences)]
            mean = float(np.mean(differences))
            # Two-sided 95% paired t interval; t_{.975,19}=2.093 for the fixed 20-run suite.
            half_width = float(2.093 * np.std(differences, ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else float("nan")
            rows.append({"reference": reference, "metric": metric, "n": len(differences),
                         "mean_difference": mean, "ci95_low": mean - half_width,
                         "ci95_high": mean + half_width})
    return rows


def _write_latex_macros(rows: list[dict], paired_rows: list[dict], path: Path) -> None:
    lookup = {r["method"]: r for r in rows}
    aliases = {"NoRes": "No reservation", "Reactive": "Reactive", "Certainty": "Certainty-equivalent",
               "Aware": "Uncertainty-aware", "Oracle": "Projection reference", "PPO": "PPO (proposed)"}
    lines = ["% Auto-generated by proran.experiment; do not hand-edit."]
    metrics = {"HSR": "handover_success_rate", "QoS": "qos_rate",
               "Waste": "unused_reservation_fraction", "Occupancy": "reservation_occupancy",
               "Latency": "mean_latency_ms", "P95Latency": "p95_latency_ms", "Block": "block_rate",
               "EMBBQoS": "embb_qos_rate", "URLLCQoS": "urllc_qos_rate",
               "MMTCQoS": "mmtc_qos_rate"}
    for alias, name in aliases.items():
        row = lookup[name]
        for suffix, key in metrics.items():
            val = row[key]
            rendered = "--" if val is None or not math.isfinite(val) else (f"{100*val:.1f}" if key not in {"mean_latency_ms", "p95_latency_ms"} else f"{val:.2f}")
            lines.append(f"\\newcommand{{\\{alias}{suffix}}}{{{rendered}}}")
    for reference, alias in (("No reservation", "NoRes"),
                             ("Uncertainty-aware", "Aware")):
        for metric, suffix in (("handover_success_rate", "HSR"),
                               ("qos_rate", "QoS")):
            row = next(r for r in paired_rows if r["reference"] == reference
                       and r["metric"] == metric)
            for field, ending in (("mean_difference", "Diff"),
                                  ("ci95_low", "CILow"),
                                  ("ci95_high", "CIHigh")):
                lines.append(f"\\newcommand{{\\{alias}{suffix}{ending}}}"
                             f"{{{100 * row[field]:+.2f}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
