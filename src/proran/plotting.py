from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import SimConfig

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "No reservation": "#6c757d",
    "Reactive": "#f0ad4e",
    "Certainty-equivalent": "#9467bd",
    "Uncertainty-aware": "#2ca02c",
    "Projection reference": "#111111",
    "PPO (proposed)": "#d62728",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
        }
    )


def _save(fig, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def _moving(values: np.ndarray, window: int = 12) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_convergence(history: list[dict], directory: Path) -> None:
    _style()
    ep = np.array([h["episode"] for h in history])
    reward = np.array([h["reward"] for h in history])
    loss = np.array([h["value_loss"] for h in history])
    w = min(12, len(ep))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    axes[0].plot(ep, reward, color="#d62728", alpha=0.18, lw=0.7)
    axes[0].plot(ep[w - 1 :], _moving(reward, w), color="#d62728", lw=1.8)
    axes[0].set(xlabel="Training episode", ylabel="Episode return", title="(a) PPO convergence")
    axes[1].semilogy(ep, np.maximum(loss, 1e-5), color="#2878B5", alpha=0.22, lw=0.7)
    axes[1].semilogy(ep[w - 1 :], np.maximum(_moving(loss, w), 1e-5), color="#2878B5", lw=1.8)
    axes[1].set(xlabel="Training episode", ylabel="Value MSE", title="(b) Critic stability")
    _save(fig, directory, "training_convergence")


def plot_learning_stages(imitation: list[dict], history: list[dict], directory: Path) -> None:
    _style()
    bc_epoch = np.array([row["epoch"] for row in imitation])
    bc_loss = np.array([row["cross_entropy"] for row in imitation])
    episode = np.array([row["episode"] for row in history])
    reward = np.array([row["reward"] for row in history])
    window = min(10, len(reward))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    axes[0].plot(bc_epoch, bc_loss, color="#d89a36", lw=1.8)
    axes[0].set(
        xlabel="Behavior-cloning epoch",
        ylabel="Cross-entropy",
        title="(a) Balanced policy initialization",
    )
    axes[1].plot(episode, reward, color="#2aa89a", alpha=0.22, lw=0.7)
    axes[1].plot(episode[window - 1 :], _moving(reward, window), color="#168b80", lw=1.8)
    axes[1].set(
        xlabel="PPO episode", ylabel="Episode return", title="(b) Reinforcement fine-tuning"
    )
    _save(fig, directory, "learning_stages")


def plot_pareto(pareto: list[dict], benchmark: list[dict], directory: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    x = np.array([r["reservation_occupancy"] for r in pareto]) * 100
    y = np.array([r["qos_rate"] for r in pareto]) * 100
    ax.plot(x, y, "o-", color="#2878B5", label="Reliability-threshold sweep", lw=1.5, ms=4)
    for row in benchmark:
        name = row["method"]
        ax.scatter(
            100 * row["reservation_occupancy"],
            100 * row["qos_rate"],
            s=35,
            color=COLORS.get(name),
            marker="*" if "PPO" in name else "s",
            label=name,
            zorder=3,
        )
    ax.set(
        xlabel="Reserved RB occupancy (%)",
        ylabel="Handover QoS satisfaction (%)",
        title="QoS–resource-cost trade-off",
    )
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _save(fig, directory, "qos_resource_pareto")


def plot_latency_cdfs(samples: list[dict], directory: Path, cfg: SimConfig) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.6), sharey=True)
    selected = ["No reservation", "PPO (proposed)"]
    for sid, ax in enumerate(axes):
        for name in selected:
            values = np.sort(
                np.asarray(
                    [
                        row["latency_ms"]
                        for row in samples
                        if row["method"] == name and row["slice_id"] == sid
                    ]
                )
            )
            if not values.size:
                continue
            cdf = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, cdf, lw=1.6, label=name, color=COLORS[name])
        bound = cfg.slices[sid].latency_max_ms
        ax.axvline(bound, color="#555555", lw=1.0, ls=":", label="Slice SLA bound")
        ax.set(
            xlabel="Modeled latency (ms)",
            ylim=(0, 1.01),
            title=f"({chr(97 + sid)}) {cfg.slices[sid].name}",
        )
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Empirical CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    _save(fig, directory, "slice_latency_cdf")
