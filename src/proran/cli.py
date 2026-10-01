from __future__ import annotations

import argparse
from pathlib import Path

from .config import SimConfig, TrainConfig
from .dashboard import animate, terminal_live
from .experiment import run_experiments
from .ppo import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Proactive 5G RAN handover reservation research artifact")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train masked PPO")
    train.add_argument("--output", default="results")
    train.add_argument("--episodes", type=int, default=60)
    exp = sub.add_parser("experiment", help="Run matched-seed benchmarks and generate figures")
    exp.add_argument("--output", default="results")
    exp.add_argument("--no-retrain", action="store_true")
    show = sub.add_parser("showcase", help="Run the animated 2D showcase")
    show.add_argument("--model", default="results/ppo_model.npz")
    show.add_argument("--save", help="Save GIF instead of opening a GUI")
    show.add_argument("--frames", type=int, default=160)
    show.add_argument("--terminal", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        train_ppo(SimConfig(), TrainConfig(episodes=args.episodes), args.output)
    elif args.command == "experiment":
        run_experiments(args.output, retrain=not args.no_retrain)
    elif args.terminal:
        terminal_live(args.model, args.frames)
    else:
        animate(args.model, args.save, args.frames)


if __name__ == "__main__":
    main()
