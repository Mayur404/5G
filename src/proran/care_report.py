"""Tables, paired statistics, LaTeX macros and figures for the CARE study.

Everything here is derived from saved artifacts in ``results/care``; no
simulation is re-run, so the paper's numbers are traceable to ``runs.json``.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Two-sided 95% Student-t critical values t_{0.975, df} for df = 1..30.
_T975 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093024054408263,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def t_crit(n: int) -> float:
    """95% two-sided t critical value for a mean of ``n`` samples."""
    return _T975[n - 2] if n - 1 <= len(_T975) else 1.96


METHOD_ORDER = (
    "No reservation",
    "Reactive",
    "Certainty-equivalent",
    "Fixed proactive",
    "Floor-JIT",
    "Tuned rule",
    "Slice-tuned rule",
    "CARE",
)
# Fixed identity colors (reference categorical order; gray = no-action reference).
COLORS = {
    "No reservation": "#8a8986",
    "Reactive": "#eb6834",
    "Certainty-equivalent": "#eda100",
    "Fixed proactive": "#e87ba4",
    "Floor-JIT": "#1baf7a",
    "Tuned rule": "#6250d6",
    "Slice-tuned rule": "#008300",
    "CARE": "#2a78d6",
    "CARE single network": "#6fa7e8",
    "CARE-PPO": "#e34948",
    "CARE-PPO w/ on-policy wait": "#52514e",
}
MACRO_NAMES = {
    "No reservation": "NoRes",
    "Reactive": "Reactive",
    "Certainty-equivalent": "Certainty",
    "Fixed proactive": "Fixed",
    "Floor-JIT": "FloorJIT",
    "Tuned rule": "Tuned",
    "Slice-tuned rule": "SliceTuned",
    "CARE": "Care",
    "CARE w/o calibration": "CareUncal",
    "CARE single network": "CareSingle",
    "CARE-PPO": "CarePPO",
    "CARE-PPO w/ on-policy wait": "CarePPOOnPolicy",
}
ABLATIONS = (
    "CARE w/o calibration",
    "CARE single network",
    "CARE-PPO",
    "CARE-PPO w/ on-policy wait",
)
LAMBDA_WORDS = {0.25: "Quarter", 1.0: "One", 4.0: "Four", 16.0: "Sixteen"}
FRONTIER_METHODS = (
    "Floor-JIT",
    "Tuned rule",
    "Slice-tuned rule",
    "CARE",
    "CARE single network",
    "CARE-PPO",
    "CARE-PPO w/ on-policy wait",
)


def hold_share(rows) -> float:
    """Fraction of a method's forecast decisions that committed a hold."""
    committed = decided = 0
    for row in rows:
        counts = np.array(row["action_counts"])
        committed += counts[:, 1:].sum()
        decided += counts.sum()
    return float(committed / decided) if decided else float("nan")


KPIS = {
    "handover_success_rate": ("HSR", 100),
    "qos_rate": ("QoS", 100),
    "rescue_rate": ("Rescue", 100),
    "outage_fraction": ("Outage", 100),
    "full_service_rate": ("Full", 100),
    "served_demand_fraction": ("Served", 100),
    "jain_served_demand_fairness": ("Jain", 1),
    "displaced_rb_fraction": ("Displaced", 100),
    "reservation_occupancy": ("Held", 100),
    "wrong_target_rate": ("Wrong", 100),
    "activation_reject_rate": ("Reject", 100),
    "urllc_qos_rate": ("URLLCQoS", 100),
    "embb_qos_rate": ("EMBBQoS", 100),
    "mmtc_qos_rate": ("MMTCQoS", 100),
}


def utility(row: dict, lam: float) -> float:
    return (
        row["rescue_value_embb"]
        + row["rescue_value_urllc"]
        + row["rescue_value_mmtc"]
        - lam * row["contract_displaced_per_interval"]
    )


def _value(row: dict, key: str) -> float:
    if key.startswith("utility@"):
        return utility(row, float(key.split("@")[1]))
    value = row.get(key)
    return float("nan") if value is None else float(value)


def _select(rows, method, scenario="main"):
    selected = [r for r in rows if r["method"] == method and r["scenario"] == scenario]
    return sorted(selected, key=lambda r: r["seed"])


def mean_ci(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan"), float("nan"))
    return float(v.mean()), float(t_crit(len(v)) * v.std(ddof=1) / math.sqrt(len(v)))


def sign_flip_p(differences, draws: int = 200_000, seed: int = 7) -> float:
    """Two-sided paired randomization test of a zero mean difference."""
    d = np.asarray(differences, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(d.mean())
    total = 0
    for start in range(0, draws, 20_000):
        signs = rng.choice((-1.0, 1.0), size=(min(20_000, draws - start), len(d)))
        total += int(np.sum(np.abs((signs * d).mean(axis=1)) >= observed - 1e-12))
    return (total + 1) / (draws + 1)


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired(rows, left, right, metric, scenario="main") -> dict:
    a = {r["seed"]: _value(r, metric) for r in _select(rows, left, scenario)}
    b = {r["seed"]: _value(r, metric) for r in _select(rows, right, scenario)}
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    d = d[np.isfinite(d)]
    half = t_crit(len(d)) * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {
        "proposed": left,
        "reference": right,
        "scenario": scenario,
        "metric": metric,
        "n": int(len(d)),
        "mean_difference": float(d.mean()),
        "ci95_low": float(d.mean() - half),
        "ci95_high": float(d.mean() + half),
        "p_signflip": sign_flip_p(d),
        "wins": int(np.sum(d > 0)),
    }


# ---------------------------------------------------------------- figures
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5)


def fig_headline(rows: list[dict], path: Path) -> None:
    """Handover admission by slice and outage, main setting (lambda = 1)."""
    # Gray baseline first; blue CARE sits between orange and green so that no
    # adjacent pair is confusable under color-vision deficiency.
    methods = ("No reservation", "Reactive", "CARE", "Slice-tuned rule")
    groups = (
        ("eMBB", "embb_handover_success_rate"),
        ("uRLLC", "urllc_handover_success_rate"),
        ("mMTC", "mmtc_handover_success_rate"),
        ("All", "handover_success_rate"),
    )
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 2.45), gridspec_kw={"width_ratios": [3.0, 1.45]}
    )
    width = 0.2
    x = np.arange(len(groups))
    for i, method in enumerate(methods):
        selected = _select(rows, method)
        stats = [mean_ci([100 * r[key] for r in selected]) for _, key in groups]
        left.bar(
            x + (i - 1.5) * width,
            [m for m, _ in stats],
            width=width - 0.03,
            yerr=[c for _, c in stats],
            color=COLORS[method],
            ecolor="#52514e",
            error_kw={"elinewidth": 0.7, "capsize": 1.5},
            label=method,
        )
    left.set_xticks(x, [g for g, _ in groups])
    left.set_ylabel("Handover admission (%)", fontsize=8)
    left.set_ylim(0, 100)
    left.legend(fontsize=6.8, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.62, 1.0))
    _style(left)
    outage = [mean_ci([100 * r["outage_fraction"] for r in _select(rows, m)]) for m in methods]
    right.bar(
        np.arange(len(methods)),
        [m for m, _ in outage],
        width=0.62,
        yerr=[c for _, c in outage],
        color=[COLORS[m] for m in methods],
        ecolor="#52514e",
        error_kw={"elinewidth": 0.7, "capsize": 1.5},
    )
    for i, (m, c) in enumerate(outage):
        right.text(i, m + c + 0.12, f"{m:.1f}", ha="center", fontsize=6.8, color="#2b2b29")
    right.set_xticks(np.arange(len(methods)), ["None", "Reac-\ntive", "CARE", "Slice-\ntuned"])
    right.set_ylabel("Outage (% of UE-intervals)", fontsize=8)
    _style(right)
    right.tick_params(axis="x", labelsize=6.8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def fig_calibration(calibration: dict, path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    bins = [b for b in calibration["calibrated"]["bins"] if b["count"] >= 20]
    left.plot([0, 1], [0, 1], color="#8a8986", linewidth=1, linestyle=":", label="Ideal")
    left.plot(
        [b["confidence"] for b in bins],
        [b["accuracy"] for b in bins],
        marker="o",
        markersize=4,
        color=COLORS["CARE"],
        linewidth=1.6,
        label="Calibrated",
    )
    const = calibration["uncalibrated_forecast_score"]
    left.scatter(
        [const["bins"][0]["confidence"]],
        [const["base_rate"]],
        marker="s",
        s=28,
        color=COLORS["Reactive"],
        label="Constant score",
        zorder=3,
    )
    left.set_xlabel("Predicted confidence", fontsize=8)
    left.set_ylabel("Observed target accuracy", fontsize=8)
    left.set_xlim(0.2, 1.01)
    left.set_ylim(0.2, 1.01)
    left.legend(fontsize=7, frameon=False, loc="upper left")
    _style(left)
    leads = calibration["by_lead"]
    x = [r["lead"] for r in leads]
    right.plot(
        x,
        [r["accuracy"] for r in leads],
        marker="o",
        markersize=4,
        color="#52514e",
        linewidth=1.6,
        label="Observed",
    )
    right.plot(
        x,
        [r["mean_calibrated"] for r in leads],
        marker="^",
        markersize=4,
        color=COLORS["CARE"],
        linewidth=1.6,
        linestyle="--",
        label="Mean calibrated",
    )
    right.set_xlabel("Forecast lead (intervals of 100 ms)", fontsize=8)
    right.set_ylabel("Target accuracy", fontsize=8)
    right.legend(fontsize=7, frameon=False, loc="lower left")
    _style(right)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def fig_tradeoff(table: dict, path: Path) -> None:
    """Continuity versus incumbent displacement: two panels, one y-scale each."""
    markers = dict(zip(METHOD_ORDER, ("o", "s", "D", "v", "^", "P", "X", "o"), strict=True))
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, key, label in (
        (left, "handover_success_rate", "Handover success (%)"),
        (right, "full_service_rate", "Whole-network full service (%)"),
    ):
        for method in METHOD_ORDER:
            row = table[method]
            ax.errorbar(
                100 * row["displaced_rb_fraction"][0],
                100 * row[key][0],
                yerr=100 * row[key][1],
                xerr=100 * row["displaced_rb_fraction"][1],
                fmt=markers[method],
                markersize=7 if method == "CARE" else 5,
                color=COLORS[method],
                ecolor=COLORS[method],
                elinewidth=0.8,
                capsize=0,
                zorder=4 if method == "CARE" else 3,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=method,
            )
        ax.set_xlabel("Incumbent RBs displaced by holds (% capacity)", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        _style(ax)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def fig_frontier(frontier: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    lams = sorted({r["lambda"] for r in frontier})
    for method, marker in (
        ("Floor-JIT", "s"),
        ("Tuned rule", "D"),
        ("Slice-tuned rule", "v"),
        ("CARE", "o"),
        ("CARE-PPO", "^"),
        ("CARE-PPO w/ on-policy wait", "x"),
    ):
        points = [r for r in frontier if r["method"] == method]
        points.sort(key=lambda r: r["lambda"])
        ax.errorbar(
            [r["lambda"] for r in points],
            [r["utility"] for r in points],
            yerr=[r["ci"] for r in points],
            marker=marker,
            markersize=4.5,
            color=COLORS[method],
            linewidth=1.6,
            capsize=0,
            label=method,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
    ax.axhline(0, color="#8a8986", linewidth=0.8, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(lams)
    ax.set_xticklabels([f"{v:g}" for v in lams])
    ax.set_xlabel(r"Displacement price $\lambda$", fontsize=8)
    ax.set_ylabel("Net utility (RB per interval)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def fig_sweeps(sweeps: dict, path: Path) -> None:
    panels = (
        ("outage", "Re-establishment outage (intervals)"),
        ("ues", "UEs (offered load)"),
        ("prep", "Preparation delay (intervals)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, (key, label) in zip(axes, panels, strict=True):
        for method in ("No reservation", "Floor-JIT", "Tuned rule", "CARE"):
            points = sorted(sweeps[key][method])
            ax.errorbar(
                [p[0] for p in points],
                [p[1] for p in points],
                yerr=[p[2] for p in points],
                marker="o",
                markersize=3.5,
                color=COLORS[method],
                linewidth=1.5,
                capsize=0,
                label=method,
            )
        ax.set_xlabel(label, fontsize=8)
        _style(ax)
    axes[0].set_ylabel("Handover success (%)", fontsize=8)
    axes[0].legend(fontsize=6.3, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def fig_policy(action_shares: dict, path: Path) -> None:
    """Stacked horizontal bars of CARE's committed actions per slice."""
    from .actions import ACTION_CATALOG_V2

    labels = [f"{s.amount}/{s.timing}" for s in ACTION_CATALOG_V2]
    used = [
        i for i in range(len(labels)) if any(action_shares[s][i] > 0.005 for s in action_shares)
    ]
    palette = [
        "#8a8986",
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
        "#52514e",
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    slices = list(action_shares)
    left = np.zeros(len(slices))
    for i in used:
        widths = np.array([100 * action_shares[s][i] for s in slices])
        ax.barh(
            slices,
            widths,
            left=left,
            color=palette[i],
            edgecolor="white",
            linewidth=1.0,
            label="no-op (wait)" if i == 0 else labels[i],
            height=0.6,
        )
        left += widths
    ax.set_xlabel("Share of CARE decisions (%)", fontsize=8)
    ax.set_xlim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, fontsize=6, frameon=False, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def fig_training(directory: Path, lambdas, replica: int, path: Path) -> None:
    """Hold share of the behavior policy during training, per price and learner."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    shades = ("#a9c9f1", "#6fa7e8", "#2a78d6", "#0b3d7a")  # Light to dark = low to high price.
    for ax, (tag, title) in zip(
        axes, (("", r"CARE (fitted Q, $\epsilon$-greedy)"), ("_ppo", "CARE-PPO")), strict=True
    ):
        for shade, lam in zip(shades, lambdas, strict=False):
            file = directory / "models" / f"care_l{lam:g}_r{replica}{tag}.csv"
            if not file.exists():
                continue
            with file.open(encoding="utf-8") as fh:
                share = np.array([float(r["reserve_fraction"]) for r in csv.DictReader(fh)])
            window = 8
            smooth = np.convolve(share, np.ones(window) / window, mode="valid")
            ax.plot(
                np.arange(window, len(share) + 1),
                100 * smooth,
                color=shade,
                linewidth=1.5,
                label=rf"$\lambda={lam:g}$",
            )
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Training episode", fontsize=8)
        ax.set_ylim(0, 102)
        _style(ax)
    axes[0].set_ylabel("Decisions that hold (%)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- report
def _fmt(value: float, digits: int = 1) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def _signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def build_report(directory: Path) -> None:
    directory = Path(directory)
    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = json.loads((directory / "runs.json").read_text(encoding="utf-8"))
    tuning = json.loads((directory / "tuning.json").read_text(encoding="utf-8"))
    calibration = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
    lambdas = sorted(float(k) for k in tuning["rules"])
    main_lambda = 1.0

    metrics = list(KPIS) + [f"utility@{lam:g}" for lam in lambdas]
    table = {}
    aggregate_rows = []
    for method in (*METHOD_ORDER, *ABLATIONS):
        selected = _select(rows, method)
        if not selected:
            continue
        table[method] = {m: mean_ci([_value(r, m) for r in selected]) for m in metrics}
        table[method]["handover_attempts"] = mean_ci([r["handover_attempts"] for r in selected])
        table[method]["mean_commit_lead"] = mean_ci(
            [
                r["mean_commit_lead"] if r["mean_commit_lead"] is not None else np.nan
                for r in selected
            ]
        )
        aggregate_rows.append(
            {
                "method": method,
                "runs": len(selected),
                **{f"{m}_mean": v[0] for m, v in table[method].items()},
                **{f"{m}_ci95": v[1] for m, v in table[method].items()},
            }
        )
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    # Primary family: CARE versus every baseline on four predeclared metrics.
    primary_metrics = (
        f"utility@{main_lambda:g}",
        "handover_success_rate",
        "outage_fraction",
        "full_service_rate",
    )
    comparisons = [
        paired(rows, "CARE", reference, metric)
        for reference in METHOD_ORDER[:-1]
        for metric in primary_metrics
    ]
    comparisons += [
        paired(rows, "CARE", ablation, m) for ablation in ABLATIONS for m in primary_metrics
    ]
    adjusted = holm([c["p_signflip"] for c in comparisons])
    for comparison, p in zip(comparisons, adjusted, strict=True):
        comparison["p_holm"] = p
    with (directory / "paired.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    # Utility frontier over displacement prices.
    def price_rows(method, lam):
        # Floor-JIT has no tunable parameter, so its main-scenario runs serve every price.
        scenario = "main" if lam == main_lambda or method == "Floor-JIT" else f"lambda={lam:g}"
        return _select(rows, method, scenario)

    frontier = []
    for lam in lambdas:
        for method in FRONTIER_METHODS:
            selected = price_rows(method, lam)
            if not selected:
                continue
            m, ci = mean_ci([utility(r, lam) for r in selected])
            frontier.append(
                {
                    "method": method,
                    "lambda": lam,
                    "utility": m,
                    "ci": ci,
                    "hold_share": hold_share(selected),
                    "hsr": mean_ci([r["handover_success_rate"] for r in selected])[0],
                    "displaced": mean_ci([r["displaced_rb_fraction"] for r in selected])[0],
                    "contracts": mean_ci([r["contracts"] for r in selected])[0],
                }
            )
    with (directory / "frontier_points.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(frontier[0]))
        writer.writeheader()
        writer.writerows(frontier)
    frontier_tests = []
    for lam in lambdas:
        care = {r["seed"]: utility(r, lam) for r in price_rows("CARE", lam)}
        for reference in FRONTIER_METHODS:
            if reference == "CARE" or not price_rows(reference, lam):
                continue
            other = {r["seed"]: utility(r, lam) for r in price_rows(reference, lam)}
            d = np.array([care[s] - other[s] for s in sorted(care)])
            half = t_crit(len(d)) * d.std(ddof=1) / math.sqrt(len(d))
            frontier_tests.append(
                {
                    "proposed": "CARE",
                    "reference": reference,
                    "scenario": "main" if lam == main_lambda else f"lambda={lam:g}",
                    "metric": f"utility@{lam:g}",
                    "n": len(d),
                    "mean_difference": float(d.mean()),
                    "ci95_low": float(d.mean() - half),
                    "ci95_high": float(d.mean() + half),
                    "p_signflip": sign_flip_p(d),
                    "wins": int(np.sum(d > 0)),
                }
            )
    for test, p in zip(
        frontier_tests, holm([t["p_signflip"] for t in frontier_tests]), strict=True
    ):
        test["p_holm"] = p
    with (directory / "frontier.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(frontier_tests[0]))
        writer.writeheader()
        writer.writerows(frontier_tests)

    # Sensitivity sweeps (main configuration supplies the reference point).
    base_values = {"outage": 3, "ues": 32, "prep": 1}
    sweeps = {key: {} for key in base_values}
    sweep_rows = []
    for key, base in base_values.items():
        for method in ("No reservation", "Floor-JIT", "Tuned rule", "CARE"):
            points = []
            scenarios = {r["scenario"] for r in rows if r["scenario"].startswith(key + "=")}
            for scenario in sorted(scenarios) + ["main"]:
                value = base if scenario == "main" else int(scenario.split("=")[1])
                selected = _select(rows, method, scenario)
                hsr = mean_ci([100 * r["handover_success_rate"] for r in selected])
                full = mean_ci([100 * r["full_service_rate"] for r in selected])
                util = mean_ci([utility(r, main_lambda) for r in selected])
                points.append((value, hsr[0], hsr[1]))
                sweep_rows.append(
                    {
                        "sweep": key,
                        "value": value,
                        "method": method,
                        "hsr": hsr[0],
                        "hsr_ci": hsr[1],
                        "full_service": full[0],
                        "utility": util[0],
                        "utility_ci": util[1],
                    }
                )
            sweeps[key][method] = points
    with (directory / "sweeps.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)

    # Mechanism ablations with the Floor-JIT rule.
    ablations = {}
    for scenario in ("main", "no-retiming", "free-activation", "single-decision"):
        selected = _select(rows, "Floor-JIT", scenario)
        ablations[scenario] = {
            m: mean_ci([_value(r, m) for r in selected])
            for m in (
                "handover_success_rate",
                "displaced_rb_fraction",
                "full_service_rate",
                "activation_reject_rate",
                f"utility@{main_lambda:g}",
            )
        }
    legacy = {
        method: {
            m: mean_ci([_value(r, m) for r in _select(rows, method, "legacy-mechanics")])
            for m in ("handover_success_rate", "full_service_rate", "outage_fraction")
        }
        for method in ("No reservation", "Fixed proactive")
    }

    # What CARE learned: decision shares by slice at the main price.
    action_shares = {}
    for sid, name in enumerate(("eMBB", "uRLLC", "mMTC")):
        counts = np.sum([np.array(r["action_counts"])[sid] for r in _select(rows, "CARE")], axis=0)
        action_shares[name] = (counts / max(1, counts.sum())).tolist()

    fig_calibration(calibration, figures / "calibration")
    fig_headline(rows, figures / "headline")
    fig_tradeoff(table, figures / "tradeoff")
    fig_frontier(frontier, figures / "frontier")
    fig_sweeps(sweeps, figures / "sweeps")
    fig_policy(action_shares, figures / "policy")
    fig_training(directory, lambdas, 101, figures / "training")

    _write_macros(
        directory / "paper_macros.tex",
        table,
        comparisons,
        frontier_tests,
        ablations,
        legacy,
        calibration,
        tuning,
        sweeps,
        action_shares,
        sweep_rows,
        frontier,
    )
    _write_main_table(directory / "main_table.tex", table)
    _write_frontier_table(directory / "frontier_table.tex", frontier)
    (directory / "report.json").write_text(
        json.dumps(
            {
                "table": table,
                "paired": comparisons,
                "frontier": frontier,
                "frontier_tests": frontier_tests,
                "ablations": ablations,
                "legacy": legacy,
                "action_shares": action_shares,
                "tuning": tuning,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("CARE report written to", directory, flush=True)


def _write_main_table(path: Path, table: dict) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & HSR & QoS & Outage & Full & Served & Displ. & $U_{1}$\\",
        r"\midrule",
    ]
    best_hsr = max(table[m]["handover_success_rate"][0] for m in METHOD_ORDER)
    for method in (*METHOD_ORDER, *ABLATIONS):
        if method not in table:
            continue
        row = table[method]
        cells = [
            _fmt(100 * row["handover_success_rate"][0]),
            _fmt(100 * row["qos_rate"][0]),
            _fmt(100 * row["outage_fraction"][0], 2),
            _fmt(100 * row["full_service_rate"][0]),
            _fmt(100 * row["served_demand_fraction"][0]),
            _fmt(100 * row["displaced_rb_fraction"][0], 2),
            _fmt(row["utility@1"][0], 2),
        ]
        if row["handover_success_rate"][0] == best_hsr:
            cells[0] = r"\textbf{" + cells[0] + "}"
        name = r"\textbf{CARE (proposed)}" if method == "CARE" else method
        if method == ABLATIONS[0]:
            lines.append(r"\midrule")
        lines.append(" & ".join([name, *cells]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_frontier_table(path: Path, frontier: list[dict]) -> None:
    """Utility (and hold share) of every price-specific method at each price."""
    lambdas = sorted({p["lambda"] for p in frontier})
    lines = [
        r"\begin{tabular}{l" + "r" * len(lambdas) + "}",
        r"\toprule",
        "Method & " + " & ".join(rf"$\lambda={lam:g}$" for lam in lambdas) + r"\\",
        r"\midrule",
    ]
    for method in FRONTIER_METHODS:
        cells = []
        for lam in lambdas:
            point = next(
                (p for p in frontier if p["method"] == method and p["lambda"] == lam), None
            )
            best = max(p["utility"] for p in frontier if p["lambda"] == lam)
            if point is None:
                cells.append("--")
                continue
            text = f"{point['utility']:.2f} ({100 * point['hold_share']:.0f}" + r"\%)"
            cells.append(r"\textbf{" + text + "}" if point["utility"] == best else text)
        name = r"\textbf{CARE}" if method == "CARE" else method
        lines.append(" & ".join([name, *cells]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_macros(
    path,
    table,
    comparisons,
    frontier_tests,
    ablations,
    legacy,
    calibration,
    tuning,
    sweeps,
    action_shares,
    sweep_rows,
    frontier,
) -> None:
    out = ["% Auto-generated by proran.care_report from results/care; do not hand-edit."]

    def macro(name: str, value: str) -> None:
        out.append(rf"\newcommand{{\{name}}}{{{value}}}")

    for method, prefix in MACRO_NAMES.items():
        if method not in table:
            continue
        for key, (suffix, scale) in KPIS.items():
            digits = 3 if scale == 1 else (2 if suffix in {"Outage", "Displaced", "Held"} else 1)
            macro(f"{prefix}{suffix}", _fmt(scale * table[method][key][0], digits))
            macro(f"{prefix}{suffix}CI", _fmt(scale * table[method][key][1], digits))
        macro(f"{prefix}Utility", _fmt(table[method]["utility@1"][0], 2))
        macro(f"{prefix}UtilityCI", _fmt(table[method]["utility@1"][1], 2))
        macro(f"{prefix}Attempts", _fmt(table[method]["handover_attempts"][0], 1))
        macro(f"{prefix}Lead", _fmt(table[method]["mean_commit_lead"][0], 2))
    metric_names = {
        "utility@1": "Util",
        "handover_success_rate": "HSR",
        "outage_fraction": "Outage",
        "full_service_rate": "Full",
    }
    for c in comparisons:
        scale = 1 if c["metric"].startswith("utility") else 100
        stem = f"Vs{MACRO_NAMES[c['reference']]}{metric_names[c['metric']]}"
        macro(stem + "Diff", _signed(scale * c["mean_difference"]))
        macro(stem + "Low", _signed(scale * c["ci95_low"]))
        macro(stem + "High", _signed(scale * c["ci95_high"]))
        macro(stem + "P", f"{c['p_holm']:.3g}" if c["p_holm"] >= 1e-4 else r"<10^{-4}")
        macro(stem + "Wins", str(c["wins"]))
    words = LAMBDA_WORDS
    for t in frontier_tests:
        lam = float(t["metric"].split("@")[1])
        stem = f"Front{words.get(lam, 'X')}{MACRO_NAMES[t['reference']]}"
        macro(stem + "Diff", _signed(t["mean_difference"]))
        macro(stem + "Low", _signed(t["ci95_low"]))
        macro(stem + "High", _signed(t["ci95_high"]))
        macro(stem + "Wins", str(t["wins"]))
        macro(stem + "P", f"{t['p_holm']:.3g}" if t["p_holm"] >= 1e-4 else r"<10^{-4}")
    names = {
        "main": "Full",
        "no-retiming": "NoRetime",
        "free-activation": "FreeAct",
        "single-decision": "Single",
    }
    for scenario, values in ablations.items():
        stem = "Abl" + names[scenario]
        macro(stem + "HSR", _fmt(100 * values["handover_success_rate"][0]))
        macro(stem + "Displaced", _fmt(100 * values["displaced_rb_fraction"][0], 2))
        macro(stem + "Full", _fmt(100 * values["full_service_rate"][0]))
        macro(stem + "Reject", _fmt(100 * values["activation_reject_rate"][0]))
        macro(stem + "Utility", _fmt(values["utility@1"][0], 2))
    for method, values in legacy.items():
        stem = "Legacy" + MACRO_NAMES[method]
        macro(stem + "HSR", _fmt(100 * values["handover_success_rate"][0]))
        macro(stem + "Full", _fmt(100 * values["full_service_rate"][0]))
        macro(stem + "Outage", _fmt(100 * values["outage_fraction"][0], 2))
    cal, const = calibration["calibrated"], calibration["uncalibrated_forecast_score"]
    macro("CalBrier", f"{cal['brier']:.3f}")
    macro("CalECE", f"{cal['ece']:.3f}")
    macro("CalAUC", f"{cal['auc']:.3f}")
    macro("ConstBrier", f"{const['brier']:.3f}")
    macro("ConstECE", f"{const['ece']:.3f}")
    macro("CalBaseRate", _fmt(100 * cal["base_rate"]))
    macro("CalFitExamples", f"{calibration['fit_examples']:,}".replace(",", "{,}"))
    macro("CalTestExamples", f"{calibration['test_examples']:,}".replace(",", "{,}"))
    by_lead = {r["lead"]: r for r in calibration["by_lead"]}
    macro("CalAccLeadOne", _fmt(100 * by_lead[1]["accuracy"]))
    macro("CalAccLeadEight", _fmt(100 * by_lead[8]["accuracy"]))
    forecaster = Path(path).parent.parent / "deep" / "training_metadata.json"
    if forecaster.exists():
        meta = json.loads(forecaster.read_text(encoding="utf-8"))
        macro("ForecastPosRMSE", f"{meta['validation']['position_rmse_world_units']:.2f}")
        macro("ForecastDemandMAE", f"{meta['validation']['demand_mae_rbs']:.2f}")
        macro("ForecastScore", f"{meta['validation']['forecast_score']:.2f}")
        macro("ForecastTrainExamples", f"{meta['training_examples']:,}".replace(",", "{,}"))

    def describe(spec) -> str:
        if spec is None:
            return "never hold"
        return "/".join(str(v) for v in spec[:2]) + rf", $\tau={spec[2]}$"

    for lam_key, lam_rules in tuning["rules"].items():
        word = LAMBDA_WORDS[float(lam_key)]
        suffix = "" if float(lam_key) == 1.0 else word
        macro(f"TunedRule{suffix}", describe(lam_rules["uniform"]))
        for sid, name in enumerate(("EMBB", "URLLC", "MMTC")):
            macro(f"SliceRule{name}{suffix}", describe(lam_rules["per_slice"][sid]))
    for point in frontier:
        stem = f"Lam{LAMBDA_WORDS[point['lambda']]}{MACRO_NAMES[point['method']]}"
        macro(stem + "Util", _fmt(point["utility"], 2))
        macro(stem + "UtilCI", _fmt(point["ci"], 2))
        macro(stem + "Hold", _fmt(100 * point["hold_share"]))
        macro(stem + "HSR", _fmt(100 * point["hsr"]))
        macro(stem + "Displaced", _fmt(100 * point["displaced"], 2))
        macro(stem + "Contracts", _fmt(point["contracts"], 1))
    for name, shares in action_shares.items():
        tag = {"eMBB": "EMBB", "uRLLC": "URLLC", "mMTC": "MMTC"}[name]
        macro(f"CareWait{tag}", _fmt(100 * shares[0]))
        macro(f"CareFloor{tag}", _fmt(100 * sum(shares[1:4])))
        macro(f"CareDemand{tag}", _fmt(100 * sum(shares[4:7])))
        macro(f"CareUpper{tag}", _fmt(100 * sum(shares[7:10])))
        macro(f"CareJIT{tag}", _fmt(100 * sum(shares[i] for i in (3, 6, 9))))
    for row in sweep_rows:
        key = {"outage": "Outage", "ues": "Ues", "prep": "Prep"}[row["sweep"]]
        tag = MACRO_NAMES[row["method"]]
        number = {
            0: "Zero",
            1: "One",
            2: "Two",
            3: "Three",
            6: "Six",
            24: "TwentyFour",
            32: "ThirtyTwo",
            40: "Forty",
            48: "FortyEight",
        }[row["value"]]
        macro(f"Sw{key}{number}{tag}HSR", _fmt(row["hsr"]))
        macro(f"Sw{key}{number}{tag}Full", _fmt(row["full_service"]))
        macro(f"Sw{key}{number}{tag}Util", _fmt(row["utility"], 2))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
