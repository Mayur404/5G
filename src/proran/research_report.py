"""Generate tables, plots and TeX fragments directly from validated CSVs."""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/research_validation")


def load(name):
    with (ROOT / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def forecast_summary():
    seed_groups = defaultdict(list)
    for row in load("forecast_errors_by_seed_slice_horizon.csv"):
        seed_groups[
            (row["quantity"], row["method"], int(row["lead_intervals"]), int(row["seed"]))
        ].append(row)
    aggregate = defaultdict(list)
    for (quantity, method, lead, seed), rows in seed_groups.items():
        weights = np.array([int(r["n"]) for r in rows])
        errors = np.array([float(r["error"]) for r in rows])
        error = (
            float(np.sqrt(np.average(errors**2, weights=weights)))
            if quantity == "position"
            else float(np.average(errors, weights=weights))
        )
        aggregate[(quantity, method, lead)].append(error)
    summary = []
    for (quantity, method, lead), values in aggregate.items():
        half = 2.093024054408263 * np.std(values, ddof=1) / np.sqrt(len(values))
        summary.append(
            {
                "quantity": quantity,
                "method": method,
                "lead_intervals": lead,
                "seeds": len(values),
                "mean_error": float(np.mean(values)),
                "ci95_low": float(np.mean(values) - half),
                "ci95_high": float(np.mean(values) + half),
            }
        )
    return summary


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    summary = forecast_summary()
    with (ROOT / "forecast_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for axis, quantity, title in zip(
        axes, ("position", "demand"), ("Position forecast RMSE", "Demand forecast MAE")
    ):
        for method, color in (
            ("Neural", "#2563eb"),
            ("Constant velocity", "#6b7280"),
            ("Persistence", "#6b7280"),
        ):
            rows = sorted(
                (
                    r
                    for r in summary
                    if r["quantity"] == quantity
                    and r["method"] == method
                    and (quantity != "demand" or r["lead_intervals"] > 1)
                ),
                key=lambda r: r["lead_intervals"],
            )
            if not rows:
                continue
            x = [r["lead_intervals"] for r in rows]
            y = [r["mean_error"] for r in rows]
            axis.plot(x, y, "o-", label=method, color=color, markersize=4)
            axis.fill_between(
                x,
                [r["ci95_low"] for r in rows],
                [r["ci95_high"] for r in rows],
                color=color,
                alpha=0.12,
            )
        axis.set_title(title)
        axis.set_xlabel("Lead (0.1 s intervals)")
        axis.set_ylabel("World units" if quantity == "position" else "RBs")
        axis.legend(frameon=False)
        axis.grid(alpha=0.15)
    fig.savefig(ROOT / "forecast_comparison.pdf")
    fig.savefig(ROOT / "forecast_comparison.png", dpi=350)
    plt.close(fig)

    metrics = json.loads((ROOT / "validation.json").read_text())["metrics"]
    abbreviations = [
        "Nearest\nnone",
        "Nearest\nkinematic",
        "Nearest\nneural",
        "QoS\nnone",
        "QoS\nkinematic",
        "QoS\nneural",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for axis, key, title in (
        (axes[0], "qos_rate", "Handover QoS satisfaction"),
        (axes[1], "full_service_rate", "Full service across all UE intervals"),
    ):
        values = np.array([100 * r[key] for r in metrics])
        errors = np.array(
            [100 * 2.093024054408263 * r[key + "_std"] / np.sqrt(20) for r in metrics]
        )
        axis.bar(
            range(6),
            values,
            color=["#9ca3af", "#b7c9e8", "#4a83dc", "#7e9c8f", "#70b2a0", "#14836f"],
            yerr=errors,
            capsize=3,
        )
        axis.set_xticks(range(6), abbreviations, fontsize=8)
        axis.set_title(title)
        axis.set_ylabel("Percent")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.15)
    fig.savefig(ROOT / "event_and_network_service.pdf")
    fig.savefig(ROOT / "event_and_network_service.png", dpi=350)
    plt.close(fig)

    latex = [
        "% Generated from results/research_validation CSV/JSON; do not hand edit.",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & HSR & HO QoS & Full service & Served demand & Holds\\",
        r"\midrule",
    ]
    markdown = [
        "| Method | Handover success | Handover QoS | Full service | Served demand | Held capacity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        values = [
            100 * row[k]
            for k in (
                "handover_success_rate",
                "qos_rate",
                "full_service_rate",
                "served_demand_fraction",
                "reservation_occupancy",
            )
        ]
        latex.append(row["method"] + " & " + " & ".join(f"{v:.2f}" for v in values) + r"\\")
        markdown.append(
            "| " + row["method"] + " | " + " | ".join(f"{v:.2f}%" for v in values) + " |"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (ROOT / "service_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    (ROOT / "service_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    aliases = ("NearNone", "NearKin", "NearNN", "QoSNone", "QoSKin", "QoSNN")
    macros = ["% Generated from measured research validation; do not hand edit."]
    keys = {
        "HSR": "handover_success_rate",
        "QoS": "qos_rate",
        "Full": "full_service_rate",
        "Served": "served_demand_fraction",
        "Holds": "reservation_occupancy",
        "Pingpong": "pingpong_fraction",
    }
    for alias, row in zip(aliases, metrics, strict=True):
        for suffix, key in keys.items():
            macros.append(f"\\newcommand{{\\{alias}{suffix}}}{{{100 * row[key]:.2f}}}")
        macros.append(f"\\newcommand{{\\{alias}Attempts}}{{{row['handover_attempts']:.2f}}}")
    for lead, suffix in ((1, "One"), (4, "Four"), (8, "Eight")):
        for quantity, method, prefix in (
            ("position", "Neural", "NNPosition"),
            ("position", "Constant velocity", "CVPosition"),
            ("demand", "Neural", "NNDemand"),
            ("demand", "Persistence", "PersistDemand"),
        ):
            row = next(
                r
                for r in summary
                if r["quantity"] == quantity
                and r["method"] == method
                and r["lead_intervals"] == lead
            )
            macros.append(f"\\newcommand{{\\{prefix}{suffix}}}{{{row['mean_error']:.3f}}}")
    (ROOT / "summary_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
