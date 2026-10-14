from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .config import SimConfig
from .policies import SavedPPOPolicy, uncertainty_aware
from .simulator import RANHandoverEnv


def animate(model_path: str | None = None, save: str | None = None, frames: int = 160) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
    if save:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    cfg = SimConfig(n_ues=44, episode_steps=frames + 2)
    env = RANHandoverEnv(cfg)
    policy = (
        SavedPPOPolicy.load(model_path)
        if model_path and Path(model_path).exists()
        else uncertainty_aware
    )
    observation = env.reset(31415)
    fig, (ax, bars) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1.5, 1]})

    def draw(_frame):
        nonlocal observation
        state, mask, context = observation
        result = env.step(policy(state, mask, context), observation)
        observation = (result.next_state, result.next_mask, result.info["next_context"])
        ax.clear()
        bars.clear()
        ax.axvline(50, color="0.82", lw=1)
        ax.axhline(50, color="0.82", lw=1)
        for c, center in enumerate(env.centers):
            ax.add_patch(Circle(center, 5, color="#222222", alpha=0.12))
            ax.text(*center, f"gNB {c}", ha="center", va="center", weight="bold", fontsize=8)
        for sid, profile in enumerate(cfg.slices):
            points = np.array([u.pos for u in env.ues if u.slice_id == sid])
            if len(points):
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=24,
                    color=profile.color,
                    label=profile.name,
                    edgecolor="white",
                    linewidth=0.3,
                )
        for r in env.reservations:
            ue = env.ues[r.ue_id]
            target = env.centers[r.target]
            ax.plot(
                [ue.pos[0], target[0]],
                [ue.pos[1], target[1]],
                ls="--",
                lw=1.2,
                color="#d62728" if r.active else "#f0ad4e",
                alpha=0.8,
            )
        ax.set(
            xlim=(0, 100),
            ylim=(0, 100),
            aspect="equal",
            xlabel="x (m)",
            ylabel="y (m)",
            title=f"Proactive sliced-RAN handovers  |  t={env.t * cfg.dt_s:.1f}s",
        )
        ax.legend(loc="upper left", fontsize=7, frameon=False)
        x = np.arange(4)
        alloc = result.info["allocations"]
        reserved = result.info["reservations"]
        bars.bar(x, alloc, label="Allocated", color="#2878B5")
        bars.bar(x, reserved, bottom=alloc, label="Reserved", color="#d62728")
        bars.axhline(cfg.cell_capacity, color="black", ls=":", label="Capacity")
        bars.set(
            xticks=x,
            xticklabels=[f"Cell {i}" for i in x],
            ylim=(0, cfg.cell_capacity * 1.18),
            ylabel="Resource blocks",
            title="Target-cell RB ledger",
        )
        bars.legend(fontsize=8, frameon=False)
        events = "\n".join(result.info["events"][-5:]) or "No reservation/handover event"
        bars.text(
            0.02,
            0.98,
            events,
            transform=bars.transAxes,
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=120, blit=False, repeat=False)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        anim.save(save, writer=PillowWriter(fps=8), dpi=120)
        plt.close(fig)
    else:
        plt.show()


def terminal_live(model_path: str | None = None, steps: int = 120, delay: float = 0.12) -> None:
    from rich.live import Live
    from rich.table import Table

    cfg = SimConfig(n_ues=44, episode_steps=steps + 2)
    env = RANHandoverEnv(cfg)
    policy = (
        SavedPPOPolicy.load(model_path)
        if model_path and Path(model_path).exists()
        else uncertainty_aware
    )
    observation = env.reset(2718)

    def render(info=None):
        table = Table(title=f"5G RAN live reservation ledger — t={env.t * cfg.dt_s:.1f}s")
        table.add_column("Cell")
        table.add_column("UEs")
        table.add_column("Allocated")
        table.add_column("Reserved")
        table.add_column("Free")
        held = env._reservation_totals()
        for cell in range(4):
            count = sum(u.cell == cell for u in env.ues)
            free = cfg.cell_capacity - env.allocations[cell] - held[cell]
            table.add_row(
                str(cell),
                str(count),
                f"{env.allocations[cell]} RB",
                f"{held[cell]} RB",
                f"{free} RB",
            )
        if info and info["events"]:
            table.caption = " | ".join(info["events"][-3:])
        return table

    with Live(render(), refresh_per_second=10) as live:
        for _ in range(steps):
            state, mask, context = observation
            result = env.step(policy(state, mask, context), observation)
            observation = (result.next_state, result.next_mask, result.info["next_context"])
            live.update(render(result.info))
            time.sleep(delay)
