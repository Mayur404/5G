"""Independent checks, held-out prediction errors, figures and paper macros."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from proran.deep_experiment import METHODS, paired
from proran.neural import HORIZON, NeuralPredictor, dataset

ROOT = Path(__file__).resolve().parents[1]
DEEP = ROOT / "results/deep"


def predictions():
    model = NeuralPredictor.load(DEEP / "predictor.npz")
    metadata = json.loads((DEEP / "training_metadata.json").read_text(encoding="utf-8"))
    final_seeds = json.loads((DEEP / "results.json").read_text(encoding="utf-8"))[
        "evaluation_seeds"
    ]
    train = set(metadata["training_seeds"])
    validation = set(metadata["validation_seeds"])
    test = set(final_seeds)
    assert not train & validation and not train & test and not validation & test
    print("Checking held-out trajectory predictions on 20 independent seeds", flush=True)
    x, y = dataset(final_seeds, steps=320)
    truth = y.reshape(-1, HORIZON, 3)
    network = model.forward(x).reshape(-1, HORIZON, 3)
    values = {}
    centers = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    base = x[:, None, 25:27] + x[:, None, 27:29] * 0.04 * np.arange(1, 9)[None, :, None]
    future = base + truth[:, :, :2] * 0.1
    actual_cell = np.argmin(np.sum((future[:, -1, None, :] - centers) ** 2, axis=-1), axis=1)
    current_cell = np.argmin(np.sum((x[:, None, 25:27] - centers) ** 2, axis=-1), axis=1)
    changing = actual_cell != current_cell
    for name, position, demand in [
        (
            "Neural",
            np.clip(base + network[:, :, :2] * 0.1, 0, 1),
            np.maximum(1, np.rint(network[:, :, 2] * 20)),
        ),
        (
            "Constant velocity / demand persistence",
            np.clip(base, 0, 1),
            np.repeat(x[:, 29:30] * 20, 8, axis=1),
        ),
    ]:
        error = (position - future) * 100
        predicted_cell = np.argmin(
            np.sum((position[:, -1, None, :] - centers) ** 2, axis=-1), axis=1
        )
        values[name] = {
            "position_rmse_world_units": float(np.sqrt(np.mean(np.sum(error**2, axis=2)))),
            "horizon_position_rmse_world_units": float(
                np.sqrt(np.mean(np.sum(error[:, -1, :] ** 2, axis=1)))
            ),
            "future_demand_mae_rbs": float(np.mean(np.abs(demand[:, 1:] - truth[:, 1:, 2] * 20))),
            "horizon_cell_accuracy": float(np.mean(predicted_cell == actual_cell)),
            "changed_cell_accuracy": float(
                np.mean(predicted_cell[changing] == actual_cell[changing])
            ),
        }
    report = {
        "test_seeds": final_seeds,
        "examples": len(x),
        "horizon_changed_examples": int(changing.sum()),
        "future_demand_leads": [2, 3, 4, 5, 6, 7, 8],
        "metrics": values,
    }
    (DEEP / "prediction_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def check_saved():
    result = json.loads((DEEP / "results.json").read_text(encoding="utf-8"))
    with (DEEP / "per_seed_metrics.csv").open(encoding="utf-8") as fh:
        rows = [
            {k: v if k == "method" else float(v) for k, v in row.items()}
            for row in csv.DictReader(fh)
        ]
    assert len(rows) == 120
    checked = 0
    for aggregate in result["metrics"]:
        individual = [r for r in rows if r["method"] == aggregate["method"]]
        assert len(individual) == 20 and {int(r["seed"]) for r in individual} == set(
            result["evaluation_seeds"]
        )
        for key, expected in aggregate.items():
            if key in {"method", "runs"}:
                continue
            source = key.removesuffix("_std")
            values = np.array([r[source] for r in individual])
            values = values[np.isfinite(values)]
            actual = (
                float(values.std(ddof=1) if key.endswith("_std") else values.mean())
                if len(values)
                else None
            )
            assert (
                actual is None
                and expected is None
                or math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
            ), (key, actual, expected)
            checked += 1
        for row in individual:
            assert (
                0 <= row["qos_handovers"] <= row["successful_handovers"] <= row["handover_attempts"]
            )
            assert (
                row["handover_success_rate"]
                == row["successful_handovers"] / row["handover_attempts"]
            )
            assert row["qos_rate"] == row["qos_handovers"] / row["handover_attempts"]
    recalculated = paired(rows)
    for actual, stored in zip(recalculated, result["paired"], strict=True):
        for key in ("mean_difference", "ci95_low", "ci95_high"):
            assert math.isclose(actual[key], stored[key], rel_tol=1e-10, abs_tol=1e-12)
            checked += 1
    return result, rows, checked


def paper_assets(result, rows, prediction):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    figdir = DEEP / "figures"
    figdir.mkdir(exist_ok=True)
    names = list(METHODS)
    labels = [
        "Nearest\nNo holds",
        "Nearest\nKinematic",
        "Nearest\nNeural",
        "QoS-aware\nNo holds",
        "QoS-aware\nKinematic",
        "QoS-aware\nNeural",
    ]
    lookup = {row["method"]: row for row in result["metrics"]}
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for axis, metric, title in zip(
        axes,
        ("handover_success_rate", "qos_rate"),
        ("Handover admission", "Handover QoS"),
        strict=True,
    ):
        means = np.array([lookup[n][metric] for n in names]) * 100
        errors = (
            np.array([lookup[n][metric + "_std"] for n in names])
            * 100
            * 2.093024054408263
            / np.sqrt(20)
        )
        axis.bar(
            range(6),
            means,
            yerr=errors,
            capsize=3,
            color=["#9ca3af", "#f59e0b", "#2563eb", "#94a3b8", "#d97706", "#0d9488"],
        )
        axis.set_xticks(range(6), labels, fontsize=7)
        axis.set_ylabel("Mean across seeds (%)")
        axis.set_title(title)
        axis.set_ylim(0, 65)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(figdir / "policy_comparison.pdf")
    figure.savefig(figdir / "policy_comparison.png", dpi=350)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(5, 3.4))
    for i, name in enumerate(names):
        row = lookup[name]
        axis.scatter(
            100 * row["reservation_occupancy"], 100 * row["full_demand_service_rate"], s=55
        )
        axis.annotate(
            labels[i].replace("\n", " / "),
            (100 * row["reservation_occupancy"], 100 * row["full_demand_service_rate"]),
            xytext=(4, (i % 3 - 1) * 10),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Reserved PRB-time / total PRB-time (%)")
    axis.set_ylabel("All-UE full-demand service (%)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figdir / "resource_service.pdf")
    figure.savefig(figdir / "resource_service.png", dpi=350)
    plt.close(figure)
    with (DEEP / "predictor_training.csv").open(encoding="utf-8") as fh:
        history = list(csv.DictReader(fh))
    figure, axis = plt.subplots(figsize=(5, 2.7))
    axis.plot(
        [int(r["epoch"]) for r in history],
        [float(r["training_mse"]) for r in history],
        label="Training",
    )
    axis.plot(
        [int(r["epoch"]) for r in history],
        [float(r["validation_mse"]) for r in history],
        label="Validation",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Scaled multi-output MSE")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figdir / "predictor_learning.pdf")
    figure.savefig(figdir / "predictor_learning.png", dpi=350)
    plt.close(figure)
    aliases = [
        "DeepNoRes",
        "DeepKinematic",
        "DeepNearest",
        "DeepLoadNoRes",
        "DeepLoadKinematic",
        "DeepProposed",
    ]
    fields = {
        "HSR": "handover_success_rate",
        "QoS": "qos_rate",
        "Waste": "unused_reservation_fraction",
        "Occupancy": "reservation_occupancy",
        "FullService": "full_demand_service_rate",
        "Unmet": "unmet_demand_fraction",
        "PingPong": "ping_pong_fraction",
        "WeakSignal": "weak_signal_fraction",
        "Block": "block_rate",
        "EMBBQoS": "embb_qos_rate",
        "URLLCQoS": "urllc_qos_rate",
        "MMTCQoS": "mmtc_qos_rate",
    }
    lines = [
        "% Generated from results/deep/results.json and independent held-out prediction checks."
    ]
    for alias, name in zip(aliases, names, strict=True):
        for suffix, key in fields.items():
            value = lookup[name][key]
            rendered = "--" if value is None else f"{100 * value:.2f}"
            lines.append(f"\\newcommand{{\\{alias}{suffix}}}{{{rendered}}}")
        lines.append(
            f"\\newcommand{{\\{alias}Attempts}}{{{lookup[name]['handover_attempts']:.2f}}}"
        )
    for ref, alias in [
        (names[0], "DeepBase"),
        (names[2], "DeepNearestCompare"),
        (names[4], "DeepKinematicCompare"),
    ]:
        for metric, suffix in [
            ("handover_success_rate", "HSR"),
            ("qos_rate", "QoS"),
            ("full_demand_service_rate", "Service"),
        ]:
            row = next(
                r for r in result["paired"] if r["reference"] == ref and r["metric"] == metric
            )
            for key, ending in [
                ("mean_difference", "Gain"),
                ("ci95_low", "Low"),
                ("ci95_high", "High"),
            ]:
                lines.append(f"\\newcommand{{\\{alias}{suffix}{ending}}}{{{100 * row[key]:+.2f}}}")
    for name, alias in [
        ("Neural", "Neural"),
        ("Constant velocity / demand persistence", "Kinematic"),
    ]:
        for key, suffix in [
            ("position_rmse_world_units", "PositionRMSE"),
            ("future_demand_mae_rbs", "DemandMAE"),
            ("changed_cell_accuracy", "ChangedAccuracy"),
        ]:
            value = prediction["metrics"][name][key] * (100 if key.endswith("accuracy") else 1)
            lines.append(f"\\newcommand{{\\{alias}{suffix}}}{{{value:.2f}}}")
    (DEEP / "paper_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    result, rows, checked = check_saved()
    prediction = predictions()
    paper_assets(result, rows, prediction)
    report = {
        "status": "passed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_summaries": len(rows),
        "independent_metric_checks": checked,
        "train_validation_test_seeds_disjoint": True,
        "held_out_prediction_examples": prediction["examples"],
    }
    (DEEP / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
