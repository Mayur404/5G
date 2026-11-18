"""End-to-end CARE study: calibrate, train, tune baselines, evaluate, report.

Seed partitions (never reused across roles):
  forecaster training 3000-3023, forecaster validation 4000-4005 (existing model),
  calibration fit 4100-4119, CARE training 20000+, rule tuning 5000-5009,
  held-out test 9000-9019.

Usage: python -m proran.care_experiment [stage ...]
stages: calibrate train tune evaluate report (default: all)
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .actions import ACTION_CATALOG_V2
from .calibration import LogisticCalibrator, collect, reliability
from .care import CareConfig, CareEnsemblePolicy, CarePolicy, contract_reward, train_care
from .config import SimConfig
from .neural import NeuralPredictor
from .simulator import RANHandoverEnv

OUT = Path("results/care")
PREDICTOR = Path("results/deep/predictor.npz")
CALIBRATION_SEEDS = list(range(4100, 4120))
TUNING_SEEDS = list(range(5000, 5010))
TEST_SEEDS = list(range(9000, 9020))
TRAIN_SEED_BASE = 20000
STEPS = 320
LAMBDAS = (0.25, 1.0, 4.0, 16.0)
MAIN_LAMBDA = 1.0
TRAINING_REPLICAS = (101, 102, 103)
# Learner variants: name -> (learner, wait backup, model-file tag, training episodes).
# "fqi" is CARE; the PPO variants are ablations of the learner.
VARIANTS = {
    "fqi": ("fqi", "stopping", "", 240),
    "ppo": ("ppo", "stopping", "_ppo", 160),
    "ppo_onpolicy": ("ppo", "on_policy", "_ppo_onpolicy", 160),
}
VARIANT_LAMBDAS = {"fqi": LAMBDAS, "ppo": LAMBDAS, "ppo_onpolicy": (1.0, 16.0)}

BASE = SimConfig(
    common_radio_stream=True,
    prediction_mode="kinematic",
    research_metrics=True,
    action_set="v2",
    max_decisions=32,
    failure_outage=3,
    activation_rule="floor_protected",
    sla_capped_by_demand=True,
    preparation_delay=1,
    retime_pending=True,
    episode_steps=STEPS,
)
LEGACY = replace(
    BASE,
    max_decisions=1,
    activation_rule="free",
    retime_pending=False,
    preparation_delay=0,
)

AMOUNTS = ("floor", "demand", "upper")
TIMINGS = ("now", "mid", "jit")
THRESHOLDS = (0.0, 0.6, 0.8, 0.9, 0.95)


def action_index(amount: str, timing: str) -> int:
    return next(
        i
        for i, spec in enumerate(ACTION_CATALOG_V2)
        if (spec.amount, spec.timing) == (amount, timing)
    )


class Rule:
    """Reserve ``amount``/``timing`` when calibrated confidence >= threshold.

    ``per_slice`` optionally maps slice id -> (amount, timing, threshold) or None.
    """

    def __init__(self, amount="floor", timing="jit", threshold=0.0, max_lead=8, per_slice=None):
        self.default = (amount, timing, threshold)
        self.max_lead = max_lead
        self.per_slice = per_slice

    def __call__(self, state, mask, context):
        prediction = context.get("prediction")
        if prediction is None:
            return 0
        # "prep" means the latest lead at which a hold can still beat preparation.
        limit = (
            context["env"].cfg.preparation_delay + 1 if self.max_lead == "prep" else self.max_lead
        )
        if prediction.time_to_handover > limit:
            return 0
        params = (
            self.per_slice[context["ue"].slice_id] if self.per_slice is not None else self.default
        )
        if params is None:
            return 0
        amount, timing, threshold = params
        if prediction.probability < threshold:
            return 0
        action = action_index(amount, timing)
        return action if mask[action] else 0


def _load_models():
    predictor = NeuralPredictor.load(PREDICTOR)
    calibrator = LogisticCalibrator.load(OUT / "calibrator.npz")
    return predictor, calibrator


def make_policy(spec: dict):
    kind = spec["kind"]
    if kind == "none":
        return lambda s, m, c: 0
    if kind == "rule":
        if "default" in spec and spec["default"] is None:
            return lambda s, m, c: 0  # The tuned choice was to never hold.
        return Rule(
            *spec.get("default", ("floor", "jit", 0.0)),
            max_lead=spec.get("max_lead", 8),
            per_slice=spec.get("per_slice"),
        )
    if kind == "care":
        return CarePolicy.load(spec["path"])
    if kind == "care_ensemble":
        return CareEnsemblePolicy.load(spec["paths"])
    raise ValueError(kind)


def run_once(job: dict) -> dict:
    """One matched-seed run. Picklable for process pools."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    predictor, calibrator = _load_models()
    cfg = replace(SimConfig.from_dict(job["cfg"]), seed=job["seed"])
    env = RANHandoverEnv(cfg)
    env.neural_predictor = predictor
    env.confidence_model = calibrator if job.get("calibrated", True) else None
    policy = make_policy(job["policy"])
    actions = np.zeros((3, len(ACTION_CATALOG_V2)), dtype=int)
    leads = []

    def tracked(state, mask, context):
        action = int(policy(state, mask, context))
        if cfg.action_set == "v2" and context.get("prediction") is not None:
            actions[context["ue"].slice_id, action] += 1
            if action:
                leads.append(context["prediction"].time_to_handover)
        return action

    started = time.perf_counter()
    if cfg.max_decisions > 1:
        env.run_many(tracked, seed=job["seed"])
    else:
        env.run(tracked, seed=job["seed"])
    summary = env.metrics.summary(cfg.dt_s)
    rescue_value = np.zeros(3)
    displaced = 0.0
    zero_cost = CareConfig(displacement_weight=0.0, reward_scale=1.0)
    for outcome in env.contract_outcomes.values():
        rescue_value[outcome["slice_id"]] += contract_reward(outcome, cfg.failure_outage, zero_cost)
        displaced += outcome["displaced"]
    row = {
        "method": job["method"],
        "scenario": job.get("scenario", "main"),
        "seed": job["seed"],
        **{
            k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in summary.items()
        },
        "handover_attempts": env.metrics.handovers,
        "decisions": env.metrics.decisions,
        "contracts": len(env.contract_outcomes),
        "rescue_value_embb": rescue_value[0] / env.metrics.steps,
        "rescue_value_urllc": rescue_value[1] / env.metrics.steps,
        "rescue_value_mmtc": rescue_value[2] / env.metrics.steps,
        "contract_displaced_per_interval": displaced / env.metrics.steps,
        "mean_commit_lead": float(np.mean(leads)) if leads else None,
        "action_counts": actions.tolist(),
        "seconds": time.perf_counter() - started,
    }
    return row


def utility(row: dict, lam: float) -> float:
    """Net slice-weighted utility per interval (RB units) at displacement price lam."""
    return (
        row["rescue_value_embb"]
        + row["rescue_value_urllc"]
        + row["rescue_value_mmtc"]
        - lam * row["contract_displaced_per_interval"]
    )


def _pool_map(function, jobs, workers=None):
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, jobs, chunksize=1))


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v
                    for k, v in row.items()
                }
            )


# --------------------------------------------------------------------- stages
def stage_calibrate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictor = NeuralPredictor.load(PREDICTOR)
    fit_x, fit_y, _, _ = collect(BASE, CALIBRATION_SEEDS, predictor, STEPS)
    calibrator = LogisticCalibrator()
    calibrator.fit(fit_x, fit_y)
    test_x, test_y, test_lead, raw = collect(BASE, TEST_SEEDS, predictor, STEPS)
    calibrated = calibrator.predict(test_x)
    constant = np.full(len(test_y), predictor.forecast_score)
    by_lead = [
        {
            "lead": int(k),
            "count": int((test_lead == k).sum()),
            "accuracy": float(test_y[test_lead == k].mean()),
            "mean_calibrated": float(calibrated[test_lead == k].mean()),
        }
        for k in range(1, BASE.horizon + 1)
        if (test_lead == k).any()
    ]
    report = {
        "fit_seeds": CALIBRATION_SEEDS,
        "test_seeds": TEST_SEEDS,
        "fit_examples": int(len(fit_y)),
        "test_examples": int(len(test_y)),
        "fit_base_rate": float(fit_y.mean()),
        "calibrated": reliability(calibrated, test_y),
        "uncalibrated_forecast_score": reliability(constant, test_y),
        "by_lead": by_lead,
        "weights": calibrator.weights.tolist(),
    }
    calibrator.save(OUT / "calibrator.npz", {k: report[k] for k in ("fit_seeds", "fit_examples")})
    (OUT / "calibration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "Calibration Brier {:.4f} (constant {:.4f}), ECE {:.4f}, AUC {:.3f}".format(
            report["calibrated"]["brier"],
            report["uncalibrated_forecast_score"]["brier"],
            report["calibrated"]["ece"],
            report["calibrated"]["auc"],
        ),
        flush=True,
    )


def _train_job(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    predictor, calibrator = _load_models()
    learner, backup, _, episodes = VARIANTS[job["variant"]]
    care_cfg = CareConfig(
        episodes=job.get("episodes", episodes),
        displacement_weight=job["lam"],
        seed=job["replica"],
        learner=learner,
        wait_backup=backup,
    )
    started = time.perf_counter()
    agent, history = train_care(
        SimConfig.from_dict(job["cfg"]),
        predictor,
        calibrator if job["calibrated"] else None,
        care_cfg,
        seed_base=TRAIN_SEED_BASE + 1000 * (job["replica"] - 100),
        log=lambda _message: None,
    )
    agent.save(job["path"], {"care": asdict(care_cfg), "simulation": job["cfg"]})
    with Path(job["path"]).with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return {
        **{k: job[k] for k in ("variant", "lam", "replica", "calibrated", "path")},
        "seconds": time.perf_counter() - started,
    }


def model_path(lam: float, replica: int, calibrated: bool = True, variant: str = "fqi") -> str:
    tag = ("" if calibrated else "_uncal") + VARIANTS[variant][2]
    return str(OUT / "models" / f"care_l{lam:g}_r{replica}{tag}.npz")


def _trained_models() -> list[tuple[str, float, bool]]:
    """(variant, price, calibrated) combinations trained by ``stage_train``."""
    models = [(v, lam, True) for v, lams in VARIANT_LAMBDAS.items() for lam in lams]
    return models + [("fqi", MAIN_LAMBDA, False)]


def replica_key(variant: str, lam: float, calibrated: bool = True) -> str:
    return f"{variant}{'' if calibrated else '_uncal'}@{lam:g}"


def stage_train() -> None:
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "variant": variant,
            "lam": lam,
            "replica": r,
            "calibrated": calibrated,
            "cfg": asdict(BASE),
            "path": model_path(lam, r, calibrated, variant),
        }
        for variant, lam, calibrated in _trained_models()
        for r in TRAINING_REPLICAS
    ]
    # Longest jobs first keeps the process pool busy until the end.
    jobs.sort(key=lambda job: -VARIANTS[job["variant"]][3])
    for done in _pool_map(_train_job, jobs):
        print("trained", done, flush=True)


def _replica_scores(variant: str, lam: float, calibrated: bool = True) -> dict[int, float]:
    """Tuning-seed utility of each training replica (never test seeds)."""
    jobs = [
        {
            "method": f"r{r}",
            "seed": seed,
            "cfg": asdict(BASE),
            "calibrated": calibrated,
            "policy": {"kind": "care", "path": model_path(lam, r, calibrated, variant)},
        }
        for r in TRAINING_REPLICAS
        for seed in TUNING_SEEDS
    ]
    rows = _pool_map(run_once, jobs)
    return {
        r: float(np.mean([utility(row, lam) for row in rows if row["method"] == f"r{r}"]))
        for r in TRAINING_REPLICAS
    }


def stage_tune() -> None:
    grid = list(itertools.product(AMOUNTS, TIMINGS, THRESHOLDS))
    jobs = [
        {
            "method": f"{a}|{t}|{th}",
            "seed": seed,
            "cfg": asdict(BASE),
            "policy": {"kind": "rule", "default": (a, t, th)},
        }
        for a, t, th in grid
        for seed in TUNING_SEEDS
    ]
    rows = _pool_map(run_once, jobs)
    _write_csv(OUT / "tuning_runs.csv", rows)
    tuned = {}
    for lam in LAMBDAS:
        means = {}
        slice_means = {sid: {} for sid in range(3)}
        for a, t, th in grid:
            selected = [row for row in rows if row["method"] == f"{a}|{t}|{th}"]
            means[(a, t, th)] = float(np.mean([utility(row, lam) for row in selected]))
            # Per-slice attribution: the slice's rescue value net of displacement
            # caused by that slice's holds is approximated by apportioning the
            # run's displacement in proportion to the slice's committed actions.
            for sid, name in enumerate(("embb", "urllc", "mmtc")):
                values = []
                for row in selected:
                    counts = np.array(row["action_counts"])
                    committed = counts[:, 1:].sum()
                    share = counts[sid, 1:].sum() / committed if committed else 0.0
                    values.append(
                        row[f"rescue_value_{name}"]
                        - lam * share * row["contract_displaced_per_interval"]
                    )
                slice_means[sid][(a, t, th)] = float(np.mean(values))
        # Never holding (no contracts) has utility exactly 0 and is a valid choice.
        means[None] = 0.0
        best = max(means, key=means.get)
        per_slice = []
        for sid in range(3):
            candidate = max(slice_means[sid], key=slice_means[sid].get)
            per_slice.append(list(candidate) if slice_means[sid][candidate] > 0 else None)
        tuned[str(lam)] = {
            "uniform": None if best is None else list(best),
            "uniform_tuning_utility": means[best],
            "per_slice": per_slice,
        }
    replicas, selection = {}, []
    for variant, lam, calibrated in _trained_models():
        scores = _replica_scores(variant, lam, calibrated)
        replicas[replica_key(variant, lam, calibrated)] = max(scores, key=scores.get)
        selection += [
            {
                "variant": variant,
                "lam": lam,
                "calibrated": calibrated,
                "replica": r,
                "tuning_utility": u,
            }
            for r, u in scores.items()
        ]
    (OUT / "tuning.json").write_text(
        json.dumps(
            {"rules": tuned, "care_replicas": replicas, "replica_scores": selection}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(tuned, indent=1), replicas, flush=True)


def care_spec(
    tuning: dict, lam: float, variant: str = "fqi", calibrated: bool = True, ensemble=None
) -> dict:
    """CARE uses the Q-ensemble of all replicas; PPO variants use the best replica."""
    if ensemble is None:
        ensemble = variant == "fqi"
    if ensemble:
        paths = [model_path(lam, r, calibrated, variant) for r in TRAINING_REPLICAS]
        return {"policy": {"kind": "care_ensemble", "paths": paths}}
    replica = tuning["care_replicas"][replica_key(variant, lam, calibrated)]
    return {"policy": {"kind": "care", "path": model_path(lam, replica, calibrated, variant)}}


def _methods(tuning: dict, lam: float = MAIN_LAMBDA) -> dict[str, dict]:
    rules = tuning["rules"][str(lam)]
    return {
        "No reservation": {"policy": {"kind": "none"}},
        "Reactive": {
            "policy": {"kind": "rule", "default": ("demand", "now", 0.0), "max_lead": "prep"}
        },
        "Certainty-equivalent": {"policy": {"kind": "rule", "default": ("demand", "now", 0.0)}},
        "Fixed proactive": {"policy": {"kind": "rule", "default": ("demand", "mid", 0.5)}},
        "Floor-JIT": {"policy": {"kind": "rule", "default": ("floor", "jit", 0.0)}},
        "Tuned rule": {
            "policy": {
                "kind": "rule",
                "default": None if rules["uniform"] is None else tuple(rules["uniform"]),
            }
        },
        "Slice-tuned rule": {"policy": {"kind": "rule", "per_slice": rules["per_slice"]}},
        "CARE": care_spec(tuning, lam),
    }


def _learner_ablations(tuning: dict, lam: float) -> dict[str, dict]:
    ablations = {
        "CARE single network": care_spec(tuning, lam, ensemble=False),
        "CARE-PPO": care_spec(tuning, lam, "ppo"),
    }
    if lam in VARIANT_LAMBDAS["ppo_onpolicy"]:
        ablations["CARE-PPO w/ on-policy wait"] = care_spec(tuning, lam, "ppo_onpolicy")
    return ablations


def stage_evaluate() -> None:
    tuning = json.loads((OUT / "tuning.json").read_text(encoding="utf-8"))
    jobs = []

    def add(method, spec, cfg, scenario, calibrated=True):
        jobs.extend(
            {
                "method": method,
                "seed": seed,
                "cfg": asdict(cfg),
                "scenario": scenario,
                "calibrated": calibrated,
                **spec,
            }
            for seed in TEST_SEEDS
        )

    methods = _methods(tuning)
    for name, spec in methods.items():
        add(name, spec, BASE, "main")
    # Utility frontier: every price-specific method trained or tuned for each price.
    for lam in LAMBDAS:
        scenario = "main" if lam == MAIN_LAMBDA else f"lambda={lam:g}"
        if lam != MAIN_LAMBDA:
            lam_methods = _methods(tuning, lam)
            for name in ("Tuned rule", "Slice-tuned rule", "CARE"):
                add(name, lam_methods[name], BASE, scenario)
        for name, spec in _learner_ablations(tuning, lam).items():
            add(name, spec, BASE, scenario)
    # Calibration ablation at the main price: the policy sees the constant score.
    uncal = care_spec(tuning, MAIN_LAMBDA, calibrated=False)
    add("CARE w/o calibration", uncal, BASE, "main", calibrated=False)
    floor = methods["Floor-JIT"]
    add("Floor-JIT", floor, replace(BASE, retime_pending=False), "no-retiming")
    add("Floor-JIT", floor, replace(BASE, activation_rule="free"), "free-activation")
    add("Floor-JIT", floor, replace(BASE, max_decisions=1), "single-decision")
    add("No reservation", methods["No reservation"], LEGACY, "legacy-mechanics")
    add("Fixed proactive", methods["Fixed proactive"], LEGACY, "legacy-mechanics")
    # Generalization sweeps without retraining or retuning.
    sweep_methods = ("No reservation", "Floor-JIT", "Tuned rule", "CARE")
    sweeps = [
        ("outage", "failure_outage", (0, 1, 6)),
        ("ues", "n_ues", (24, 40, 48)),
        ("prep", "preparation_delay", (0, 2, 3)),
    ]
    for label, field, values in sweeps:
        for value in values:
            for name in sweep_methods:
                add(name, methods[name], replace(BASE, **{field: value}), f"{label}={value}")
    print(f"Evaluating {len(jobs)} matched runs", flush=True)
    rows = _pool_map(run_once, jobs)
    _write_csv(OUT / "runs.csv", rows)
    (OUT / "runs.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")


def stage_report() -> None:
    from .care_report import build_report

    build_report(OUT)


def provenance() -> dict:
    sources = sorted(Path(__file__).parent.glob("*.py"))
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "predictor_sha256": hashlib.sha256(PREDICTOR.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest(),
        "base_config": asdict(BASE),
        "seeds": {
            "calibration": CALIBRATION_SEEDS,
            "tuning": TUNING_SEEDS,
            "test": TEST_SEEDS,
            "care_training_base": TRAIN_SEED_BASE,
        },
        "lambdas": LAMBDAS,
        "training_replicas": TRAINING_REPLICAS,
        "variants": VARIANTS,
        "variant_lambdas": VARIANT_LAMBDAS,
    }


def main(argv=None) -> None:
    stages = (argv if argv is not None else sys.argv[1:]) or [
        "calibrate",
        "train",
        "tune",
        "evaluate",
        "report",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        started = time.perf_counter()
        {
            "calibrate": stage_calibrate,
            "train": stage_train,
            "tune": stage_tune,
            "evaluate": stage_evaluate,
            "report": stage_report,
        }[stage]()
        print(f"[{stage}] {time.perf_counter() - started:.0f}s", flush=True)
    (OUT / "provenance.json").write_text(
        json.dumps(provenance(), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
