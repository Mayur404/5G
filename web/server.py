from __future__ import annotations

import csv
import json
import math
import os
import secrets
from dataclasses import replace
from pathlib import Path
from threading import Lock

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from waitress import serve

from proran.actions import ACTION_CATALOG, decode_action
from proran.config import SimConfig
from proran.neural import NeuralPredictor, proactive
from proran.policies import SavedPPOPolicy, no_reservation
from proran.simulator import RANHandoverEnv

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"
MODEL = ROOT / "results" / "ppo_model.npz"
RESULTS = ROOT / "results" / "results.json"
DEEP_MODEL = ROOT / "results/deep/predictor.npz"
CONTROLLERS = {
    "linear_ppo": "Legacy linear PPO",
    "deep_nearest": "Deep forecast / nearest cell",
    "deep_qos": "Deep forecast / QoS-aware cell",
}
HEX_RADIUS = 19.0
HEX_SPACING = math.sqrt(3) * HEX_RADIUS


SCENARIOS = {
    "urban": {
        "label": "Predict a uRLLC car handover",
        "n_ues": 6,
        "capacity": 18,
        "speed": 1.0,
        "seed": 21,
        "focus_uid": 1,
        "description": "A car crosses from C2 to C0; PPO holds 8 RBs before it arrives.",
    },
    "crossing": {
        "label": "Fast crossing into a busy cell",
        "n_ues": 6,
        "capacity": 20,
        "speed": 1.0,
        "seed": 118,
        "focus_uid": 2,
        "description": "A shorter six-device trace with a contested uRLLC arrival.",
    },
    "hex_cluster": {
        "label": "Seven-cell cluster",
        "n_ues": 10,
        "capacity": 16,
        "speed": 0.65,
        "seed": 42,
        "focus_uid": 5,
        "hex_radius": HEX_RADIUS,
        "centers": [
            [50.0, 50.0],  # C0: Central Hub
            [50.0, 50.0 - HEX_SPACING],
            [50.0 + 1.5 * HEX_RADIUS, 50.0 - HEX_SPACING / 2],
            [50.0 + 1.5 * HEX_RADIUS, 50.0 + HEX_SPACING / 2],
            [50.0, 50.0 + HEX_SPACING],
            [50.0 - 1.5 * HEX_RADIUS, 50.0 + HEX_SPACING / 2],
            [50.0 - 1.5 * HEX_RADIUS, 50.0 - HEX_SPACING / 2],
        ],
        "description": "Ten devices share seven capacity-limited cells. Compare advance reservations with admission on arrival, using identical movement and traffic.",
    },
    "five_cells": {
        "label": "Five-cell overlap",
        "n_ues": 7,
        "capacity": 18,
        "speed": 1.0,
        "seed": 42,
        "focus_uid": 1,
        "centers": [[25.0, 25.0], [75.0, 25.0], [25.0, 75.0], [75.0, 75.0], [50.0, 50.0]],
        "description": "Seven devices move among five sites. Nearest-site geometry selects the target; PPO controls reservation amount and timing.",
    },
}


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ComparisonSession:
    def __init__(self, seed: int, scenario: str, controller: str = "linear_ppo"):
        if controller not in CONTROLLERS:
            raise ValueError("Unknown controller")
        if controller == "linear_ppo" and not MODEL.exists():
            raise FileNotFoundError(f"Saved PPO model is required for the comparison: {MODEL}")
        preset = SCENARIOS.get(scenario, SCENARIOS["hex_cluster"])
        centers = preset.get("centers")
        cfg = SimConfig(
            seed=seed,
            n_ues=preset["n_ues"],
            cell_capacity=preset["capacity"],
            centers=tuple(tuple(float(coord) for coord in c) for c in centers) if centers else None,
            hex_radius=preset.get("hex_radius"),
            common_radio_stream=controller != "linear_ppo",
            prediction_mode="synthetic" if controller == "linear_ppo" else "kinematic",
            urllc_burst_prob=preset.get("burst_prob", 0.08),
            episode_steps=100_000,
        )
        self.scenario = scenario if scenario in SCENARIOS else "hex_cluster"
        self.seed = seed
        self.controller_kind = controller
        self.normal = RANHandoverEnv(cfg)
        self.proposed = RANHandoverEnv(
            replace(cfg, handover_mode="qos_aware" if controller == "deep_qos" else "nearest")
        )
        self.normal_obs = self.normal.reset(seed)
        self.proposed_obs = self.proposed.reset(seed)
        if preset["speed"] != 1.0:
            for env in (self.normal, self.proposed):
                for ue in env.ues:
                    ue.vel *= preset["speed"]
            self.normal_obs = self.normal.observe()
            self.proposed_obs = self.proposed.observe()
        self.policy = SavedPPOPolicy.load(str(MODEL)) if controller == "linear_ppo" else proactive
        self.controller_name = (
            "BC-initialized masked PPO" if controller == "linear_ppo" else CONTROLLERS[controller]
        )
        if controller != "linear_ppo":
            self.proposed.neural_predictor = NeuralPredictor.load(DEEP_MODEL)
            self.proposed_obs = self.proposed.observe()
        self.critic_w = self.critic_b = None
        if controller == "linear_ppo":
            with np.load(MODEL) as model:
                self.critic_w = model["critic_w"].copy()
                self.critic_b = float(model["critic_b"][0])
        self.focus_id = int(preset["focus_uid"])
        self.last_normal = None
        self.last_proposed = None
        self.history = {"normal": [], "proposed": []}
        self.handover_history = {"normal": [], "proposed": []}

    def _entity_type(self, uid: int) -> str:
        # Visual taxonomy only: mobility and radio models are identical across icons.
        if uid == self.focus_id:
            return "car"
        return ("drone", "robot", "car")[(uid + self.seed) % 3]

    def _advance(self, env, observation, policy):
        state, mask, context = observation
        prediction = context.get("prediction")
        distribution = []
        value_estimate = None
        if isinstance(policy, SavedPPOPolicy):
            probabilities = policy._probabilities(state, mask, policy.weights, policy.bias)
            for action_id, probability in enumerate(probabilities):
                if mask[action_id]:
                    if action_id == 0:
                        label = "No reservation"
                    else:
                        amount, delay = decode_action(
                            action_id, prediction.demand_rbs, prediction.time_to_handover
                        )
                        fraction = (
                            "half" if ACTION_CATALOG[action_id].amount_fraction == 0.5 else "full"
                        )
                        timing = "now" if delay == 0 else f"in {delay * env.cfg.dt_s:.1f} s"
                        label = f"{amount} RB ({fraction}) · {timing}"
                    distribution.append(
                        {"action": action_id, "label": label, "probability": float(probability)}
                    )
            distribution.sort(key=lambda item: item["probability"], reverse=True)
            if self.critic_w is not None:
                value_estimate = float(state @ self.critic_w + self.critic_b)
        action = int(policy(state, mask, context))
        if policy is proactive:
            for action_id in np.flatnonzero(mask):
                amount, delay = (
                    decode_action(
                        int(action_id), prediction.demand_rbs, prediction.time_to_handover
                    )
                    if prediction
                    else (0, 0)
                )
                distribution.append(
                    {
                        "action": int(action_id),
                        "label": "No reservation"
                        if action_id == 0
                        else f"{amount} RB · in {delay * env.cfg.dt_s:.1f} s",
                        "probability": float(action_id == action),
                    }
                )
        result = env.step(action, observation)
        next_obs = (result.next_state, result.next_mask, result.info["next_context"])
        decision = {
            "step": env.t - 1,
            "action": action,
            "label": "No reservation"
            if action == 0
            else f"{ACTION_CATALOG[action].amount_fraction:.0%} demand · {ACTION_CATALOG[action].timing}",
            "ue": prediction.ue_id if prediction else None,
            "target": prediction.target if prediction else None,
            "probability": prediction.probability if prediction else None,
            "uncertainty": prediction.target_uncertainty if prediction else None,
            "predicted_rbs": prediction.demand_rbs if prediction else None,
            "eta_steps": prediction.time_to_handover if prediction else None,
            "distribution": distribution,
            "value_estimate": value_estimate,
        }
        if prediction:
            amount, delay = decode_action(
                action, prediction.demand_rbs, prediction.time_to_handover
            )
            decision.update(
                {
                    "amount_rbs": amount,
                    "delay_steps": delay,
                    "eta_s": prediction.time_to_handover * env.cfg.dt_s,
                    "delay_s": delay * env.cfg.dt_s,
                }
            )
        return result, next_obs, decision

    def step(self, count: int = 1) -> dict:
        count = max(1, min(count, 12))
        normal_decision = proposed_decision = None
        for _ in range(count):
            self.last_normal, self.normal_obs, normal_decision = self._advance(
                self.normal, self.normal_obs, no_reservation
            )
            self.last_proposed, self.proposed_obs, proposed_decision = self._advance(
                self.proposed, self.proposed_obs, self.policy
            )
            self._record_history()
        return self.snapshot(normal_decision, proposed_decision)

    def _metrics(self, env: RANHandoverEnv) -> dict:
        raw = env.metrics.summary(env.cfg.dt_s)
        # Live holds are outstanding commitments, not completed waste.
        active = int(env._reservation_totals().sum())
        settled = env.metrics.activated_rbs - active
        raw["settled_unused_reservation_fraction"] = (
            (settled - env.metrics.used_reserved_rbs) / settled if settled else None
        )
        return {key: _finite(value) for key, value in raw.items()}

    def _record_history(self) -> None:
        for name, env in (("normal", self.normal), ("proposed", self.proposed)):
            metrics = self._metrics(env)
            self.history[name].append(
                {
                    "t": env.t,
                    "time": env.t * env.cfg.dt_s,
                    "hsr": metrics["handover_success_rate"],
                    "qos": metrics["qos_rate"],
                }
            )
            self.history[name] = self.history[name][-90:]
            for event in env.last_handovers:
                ue = env.ues[event["ue"]]
                self.handover_history[name].append(
                    {
                        **event,
                        "step": env.t,
                        "time": env.t * env.cfg.dt_s,
                        "x": float(ue.pos[0]),
                        "y": float(ue.pos[1]),
                    }
                )
            self.handover_history[name] = self.handover_history[name][-100:]

    def _network(self, env: RANHandoverEnv, result, decision) -> dict:
        held = env._reservation_totals()
        ues = []
        for ue in env.ues:
            speed = float((ue.vel[0] ** 2 + ue.vel[1] ** 2) ** 0.5)
            # An illustrative distance-to-serving-site indicator, not NR RSRP/SINR.
            signal_quality = env.signal_quality(ue.pos, ue.cell)
            ues.append(
                {
                    "id": ue.uid,
                    "x": float(ue.pos[0]),
                    "y": float(ue.pos[1]),
                    "vx": float(ue.vel[0]),
                    "vy": float(ue.vel[1]),
                    "speed": speed,
                    "slice": env.cfg.slices[ue.slice_id].name,
                    "slice_id": ue.slice_id,
                    "cell": ue.cell,
                    "demand": env.interval_demands[ue.uid],
                    "next_demand": ue.demand,
                    "focus": ue.uid == self.focus_id,
                    "entity_type": self._entity_type(ue.uid),
                    "signal_quality": signal_quality,
                    "allocated": env.ue_allocations.get(ue.uid, 0),
                }
            )
        cells = []
        for cell in range(len(env.centers)):
            alloc = int(env.allocations[cell])
            reserved = int(held[cell])
            slices = []
            for slice_id, profile in enumerate(env.cfg.slices):
                members = [ue for ue in env.ues if ue.cell == cell and ue.slice_id == slice_id]
                slices.append(
                    {
                        "id": slice_id,
                        "name": profile.name,
                        "ues": len(members),
                        "demand": sum(env.interval_demands[ue.uid] for ue in members),
                        "allocated": sum(env.ue_allocations.get(ue.uid, 0) for ue in members),
                        "reserved": sum(
                            r.activated_amount
                            for r in env.reservations
                            if r.active
                            and r.target == cell
                            and env.ues[r.ue_id].slice_id == slice_id
                        ),
                        "sla_rbs_per_ue": profile.sla_rbs,
                        "latency_max_ms": profile.latency_max_ms,
                    }
                )
            cells.append(
                {
                    "id": cell,
                    "x": float(env.centers[cell, 0]),
                    "y": float(env.centers[cell, 1]),
                    "capacity": env.cfg.cell_capacity,
                    "allocated": alloc,
                    "reserved": reserved,
                    "free": max(0, env.cfg.cell_capacity - alloc - reserved),
                    "load": (alloc + reserved) / env.cfg.cell_capacity,
                    "slices": slices,
                }
            )
        contracts = [
            {
                "ue": r.ue_id,
                "target": r.target,
                "amount": r.amount,
                "active": r.active,
                "activate_at": r.activate_at,
                "expires_at": r.expires_at,
            }
            for r in env.reservations
        ]
        info = result.info if result else {"events": [], "handovers": 0, "successes": 0, "qos": 0}
        metrics = self._metrics(env)
        metrics["handovers"] = env.metrics.handovers
        metrics["successes"] = env.metrics.successes
        metrics["qos_successes"] = env.metrics.qos_successes
        return {
            "time": env.t * env.cfg.dt_s,
            "step": env.t,
            "ues": ues,
            "cells": cells,
            "contracts": contracts,
            "events": info["events"],
            "handover_details": info.get("handover_details", []),
            "recent_handovers": self.handover_history[
                "proposed" if env is self.proposed else "normal"
            ],
            "contract_events": info.get("contract_events", []),
            "instant": {
                "handovers": info.get("handovers", 0),
                "successes": info.get("successes", 0),
                "qos": info.get("qos", 0),
            },
            "decision": decision,
            "metrics": metrics,
        }

    def snapshot(self, normal_decision=None, proposed_decision=None) -> dict:
        return {
            "seed": self.seed,
            "scenario": self.scenario,
            "focus_id": self.focus_id,
            "dt_s": self.normal.cfg.dt_s,
            "horizon_steps": self.normal.cfg.horizon,
            "world_size": self.normal.cfg.world_size,
            "cell_polygons": self.normal.domain.polygons.tolist() if self.normal.domain else None,
            "domain_area": (
                len(self.normal.centers) * 3 * math.sqrt(3) / 2 * self.normal.cfg.hex_radius**2
                if self.normal.domain
                else self.normal.cfg.world_size**2
            ),
            "controller": self.controller_name,
            "controller_kind": self.controller_kind,
            "handover_mode": self.proposed.cfg.handover_mode,
            "signal_radius": self.proposed.cfg.signal_radius,
            "minimum_signal": self.proposed.cfg.minimum_signal,
            "time_to_trigger": self.proposed.cfg.time_to_trigger,
            "switch_margin": self.proposed.cfg.switch_margin,
            "target_scores": {
                str(ue.uid): [_finite(float(v)) for v in self.proposed.cell_scores(ue, ue.pos)]
                for ue in self.proposed.ues
            }
            if self.controller_kind == "deep_qos"
            else {},
            "normal": self._network(self.normal, self.last_normal, normal_decision),
            "proposed": self._network(self.proposed, self.last_proposed, proposed_decision),
            "history": self.history,
        }


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
    sessions: dict[str, ComparisonSession] = {}
    lock = Lock()

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
        if request.path.startswith("/api/") or request.path == "/":
            response.headers["Cache-Control"] = "no-store"
        elif request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, max-age=0, must-revalidate"
        return response

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/paper.tex")
    def manuscript():
        return send_from_directory(ROOT, "paper.tex", mimetype="text/plain")

    @app.get("/references.bib")
    def bibliography():
        return send_from_directory(ROOT, "references.bib", mimetype="text/plain")

    @app.get("/api/meta")
    def meta():
        published = None
        if RESULTS.exists():
            published = json.loads(RESULTS.read_text(encoding="utf-8"))
        deep_results = ROOT / "results/deep/results.json"
        deep_published = (
            json.loads(deep_results.read_text(encoding="utf-8")) if deep_results.exists() else None
        )
        learning = {"imitation": [], "ppo": []}
        for key, filename in (
            ("imitation", "imitation_history.csv"),
            ("ppo", "training_history.csv"),
        ):
            path = ROOT / "results" / filename
            if path.exists():
                with path.open(encoding="utf-8") as fh:
                    learning[key] = [
                        {name: float(value) for name, value in row.items()}
                        for row in csv.DictReader(fh)
                    ]
        paired = []
        paired_path = ROOT / "results" / "paired_comparisons.csv"
        if paired_path.exists():
            with paired_path.open(encoding="utf-8") as fh:
                paired = [
                    {
                        name: (value if name in {"reference", "metric"} else float(value))
                        for name, value in row.items()
                    }
                    for row in csv.DictReader(fh)
                ]
        return jsonify(
            {
                "scenarios": SCENARIOS,
                "published": published,
                "learning": learning,
                "deep_published": deep_published,
                "controllers": CONTROLLERS,
                "deep_model_available": DEEP_MODEL.exists(),
                "paired": paired,
                "model_available": MODEL.exists(),
            }
        )

    @app.post("/api/session")
    def new_session():
        body = request.get_json(silent=True)
        if body is None:
            body = {}
        if not isinstance(body, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400
        scenario = str(body.get("scenario", "urban"))
        controller = str(body.get("controller", "linear_ppo"))
        if controller not in CONTROLLERS:
            return jsonify({"error": "Unknown controller"}), 400
        try:
            seed_value = body.get("seed", SCENARIOS.get(scenario, SCENARIOS["urban"])["seed"])
            if (
                isinstance(seed_value, bool)
                or isinstance(seed_value, float)
                and not seed_value.is_integer()
            ):
                raise ValueError
            seed = int(seed_value)
        except (TypeError, ValueError):
            return jsonify({"error": "Seed must be an integer"}), 400
        if scenario not in SCENARIOS:
            return jsonify({"error": f"Unknown scenario: {scenario}"}), 400
        if not 0 <= seed <= 2**32 - 1:
            return jsonify({"error": "Seed must be between 0 and 4294967295"}), 400
        if controller == "linear_ppo" and not MODEL.exists():
            return jsonify(
                {"error": "Saved PPO model is missing; run the experiment before opening the demo"}
            ), 503
        if controller != "linear_ppo" and not DEEP_MODEL.exists():
            return jsonify(
                {"error": "Trained neural predictor is missing; run python -m proran.neural"}
            ), 503
        token = secrets.token_urlsafe(12)
        with lock:
            sessions[token] = ComparisonSession(seed, scenario, controller)
            if len(sessions) > 24:
                sessions.pop(next(iter(sessions)))
            snapshot = sessions[token].snapshot()
        return jsonify({"session": token, "snapshot": snapshot})

    @app.post("/api/step")
    def step():
        body = request.get_json(silent=True)
        if body is None:
            body = {}
        if not isinstance(body, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400
        token = str(body.get("session", ""))
        with lock:
            session = sessions.get(token)
            if session is None:
                return jsonify({"error": "Unknown or expired simulation session"}), 404
            try:
                count_value = body.get("count", 1)
                if (
                    isinstance(count_value, bool)
                    or isinstance(count_value, float)
                    and not count_value.is_integer()
                ):
                    raise ValueError
                count = int(count_value)
            except (TypeError, ValueError):
                return jsonify({"error": "Step count must be an integer"}), 400
            snapshot = session.step(count)
        return jsonify(snapshot)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model": MODEL.exists(), "results": RESULTS.exists()})

    return app


app = create_app()


def main() -> None:
    host = os.environ.get("PRORAN_HOST", "127.0.0.1")
    port = int(os.environ.get("PRORAN_PORT", "8080"))
    print(f"ProRAN research UI: http://{host}:{port}")
    serve(app, host=host, port=port, threads=6)


if __name__ == "__main__":
    main()
