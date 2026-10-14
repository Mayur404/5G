"""Recheck stored results and live invariants without retraining or overwriting them.

Run: python -m web.verify_simulation
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from proran.config import SimConfig, SliceProfile
from proran.experiment import _paired_comparisons, evaluate_policy_detailed
from proran.policies import BASELINES, SavedPPOPolicy

from .server import MODEL, ROOT, SCENARIOS, ComparisonSession


def close(actual, expected, label):
    if expected is None or (isinstance(expected, float) and math.isnan(expected)):
        assert actual is None or math.isnan(actual), label
    else:
        assert math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12), (
            label,
            actual,
            expected,
        )


def verify_saved_results():
    published = json.loads((ROOT / "results/results.json").read_text(encoding="utf-8"))
    with (ROOT / "results/per_seed_metrics.csv").open(encoding="utf-8") as fh:
        saved_runs = list(csv.DictReader(fh))
    with (ROOT / "results/paired_comparisons.csv").open(encoding="utf-8") as fh:
        saved_pairs = list(csv.DictReader(fh))
    cfg_data = published["simulation"]
    cfg = SimConfig(**{**cfg_data, "slices": tuple(SliceProfile(**s) for s in cfg_data["slices"])})
    methods = dict(BASELINES)
    methods["PPO (proposed)"] = SavedPPOPolicy.load(str(MODEL))
    recomputed = {}
    checked_values = 0
    for name, policy in methods.items():
        print(f"Recomputing held-out benchmark: {name}", flush=True)
        aggregate, _, runs, _ = evaluate_policy_detailed(
            name, policy, cfg, published["evaluation_seeds"], 320
        )
        stored = next(row for row in published["metrics"] if row["method"] == name)
        for key, value in aggregate.items():
            if key == "method":
                continue
            close(value, stored[key], f"{name}: aggregate {key}")
            checked_values += 1
        for seed, run in zip(published["evaluation_seeds"], runs, strict=True):
            stored_run = next(
                row for row in saved_runs if row["method"] == name and int(row["seed"]) == seed
            )
            for key, value in run.items():
                close(value, float(stored_run[key]), f"{name}: seed {seed} {key}")
                checked_values += 1
        recomputed[name] = runs
    for actual, stored in zip(_paired_comparisons(recomputed), saved_pairs, strict=True):
        assert actual["metric"] == stored["metric"] and actual["reference"] == stored["reference"]
        for key in ("n", "mean_difference", "ci95_low", "ci95_high"):
            close(actual[key], float(stored[key]), f"Paired interval: {key}")
            checked_values += 1
    return {
        "methods": len(methods),
        "seeds": len(published["evaluation_seeds"]),
        "steps_per_run": 320,
        "values_matched": checked_values,
    }


def verify_live_scenarios():
    steps = 240
    seed_count = 8
    counts = defaultdict(int)
    for scenario, preset in SCENARIOS.items():
        print(f"Checking paired live invariants: {scenario}", flush=True)
        seeds = [preset["seed"], *range(20000, 20000 + seed_count - 1)]
        for seed in seeds:
            session = ComparisonSession(seed, scenario)
            for _ in range(steps):
                snap = session.step()
                normal, ppo = snap["normal"], snap["proposed"]
                assert normal["step"] == ppo["step"]
                assert normal["metrics"]["handovers"] == ppo["metrics"]["handovers"]
                for left, right in zip(normal["ues"], ppo["ues"], strict=True):
                    for key in ("id", "x", "y", "vx", "vy", "cell", "demand", "next_demand"):
                        assert left[key] == right[key], (scenario, seed, key)
                    if session.proposed.domain is not None:
                        position = np.array([right["x"], right["y"]])
                        domain = session.proposed.domain
                        assert domain.contains(position), (scenario, seed, "outside hex cluster")
                        assert np.all(
                            np.abs((position - domain.centers[right["cell"]]) @ domain.normals.T)
                            <= domain.apothem + 1e-7
                        )
                        counts["hex_memberships"] += 1
                for network in (normal, ppo):
                    for cell in network["cells"]:
                        assert (
                            cell["allocated"] + cell["reserved"] + cell["free"] == cell["capacity"]
                        )
                        assert min(cell["allocated"], cell["reserved"], cell["free"]) >= 0
                        assert sum(s["allocated"] for s in cell["slices"]) == cell["allocated"]
                        assert sum(s["reserved"] for s in cell["slices"]) == cell["reserved"]
                        assert (
                            sum(
                                ue["allocated"] for ue in network["ues"] if ue["cell"] == cell["id"]
                            )
                            == cell["allocated"]
                        )
                        counts["cell_ledgers"] += 1
                    metrics = network["metrics"]
                    assert (
                        0
                        <= metrics["qos_successes"]
                        <= metrics["successes"]
                        <= metrics["handovers"]
                    )
                    for key in (
                        "handover_success_rate",
                        "qos_rate",
                        "reservation_occupancy",
                        "settled_unused_reservation_fraction",
                    ):
                        assert metrics[key] is None or 0 <= metrics[key] <= 1, key
                    assert len(network["handover_details"]) == network["instant"]["handovers"]
                    for event in network["handover_details"]:
                        expected_success = (
                            event["radio_ok"] and event["available_rbs"] >= event["minimum_rbs"]
                        )
                        expected_qos = (
                            expected_success
                            and event["available_rbs"] >= event["demand_rbs"]
                            and event["latency_ms"] <= event["latency_limit_ms"]
                        )
                        assert event["success"] == expected_success and event["qos"] == expected_qos
                        assert event["allocated_rbs"] == (
                            event["available_rbs"] if event["success"] else 0
                        )
                        counts["handover_records"] += 1
                decision = ppo["decision"]
                close(
                    sum(a["probability"] for a in decision["distribution"]),
                    1.0,
                    "Policy normalization",
                )
                assert all(a["probability"] >= 0 for a in decision["distribution"])
                assert decision["action"] in {a["action"] for a in decision["distribution"]}
                counts["paired_intervals"] += 1
    return {
        "scenarios": len(SCENARIOS),
        "seeds_per_scenario": seed_count,
        "steps_per_run": steps,
        **counts,
    }


def main():
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": verify_saved_results(),
        "live": verify_live_scenarios(),
        "status": "passed",
    }
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    destination = ROOT / "results/verification.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
