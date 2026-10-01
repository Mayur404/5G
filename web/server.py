from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Lock
import csv
import math
import os
import secrets
import numpy as np

from flask import Flask, jsonify, request, send_from_directory
from waitress import serve

from proran.actions import ACTION_CATALOG
from proran.config import SimConfig
from proran.policies import SavedPPOPolicy, no_reservation
from proran.simulator import RANHandoverEnv


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"
MODEL = ROOT / "results" / "ppo_model.npz"
RESULTS = ROOT / "results" / "results.json"


SCENARIOS = {
    "urban": {"label": "Predict a uRLLC car handover", "n_ues": 6,
              "capacity": 18, "speed": 1.0, "seed": 21, "focus_uid": 1,
              "description": "A car crosses from C2 to C0; PPO holds 8 RBs before it arrives."},
    "crossing": {"label": "Fast crossing into a busy cell", "n_ues": 6,
                 "capacity": 20, "speed": 1.0, "seed": 118, "focus_uid": 2,
                 "description": "A shorter six-device trace with a contested uRLLC arrival."},
}


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ComparisonSession:
    def __init__(self, seed: int, scenario: str):
        if not MODEL.exists():
            raise FileNotFoundError(f"Saved PPO model is required for the comparison: {MODEL}")
        preset = SCENARIOS.get(scenario, SCENARIOS["urban"])
        cfg = SimConfig(seed=seed, n_ues=preset["n_ues"], cell_capacity=preset["capacity"],
                        urllc_burst_prob=preset.get("burst_prob", 0.08),
                        episode_steps=100_000)
        self.scenario = scenario if scenario in SCENARIOS else "urban"
        self.seed = seed
        self.normal = RANHandoverEnv(cfg)
        self.proposed = RANHandoverEnv(cfg)
        self.normal_obs = self.normal.reset(seed)
        self.proposed_obs = self.proposed.reset(seed)
        if preset["speed"] != 1.0:
            for env in (self.normal, self.proposed):
                for ue in env.ues:
                    ue.vel *= preset["speed"]
            self.normal_obs = self.normal.observe()
            self.proposed_obs = self.proposed.observe()
        self.policy = SavedPPOPolicy.load(str(MODEL))
        self.controller_name = "BC-initialized masked PPO"
        self.critic_w = self.critic_b = None
        with np.load(MODEL) as model:
            self.critic_w = model["critic_w"].copy()
            self.critic_b = float(model["critic_b"][0])
        self.focus_id = int(preset["focus_uid"])
        self.last_normal = None
        self.last_proposed = None
        self.history = {"normal": [], "proposed": []}

    def _entity_type(self, uid: int) -> str:
        # Visual taxonomy only: mobility and radio models are identical across icons.
        if uid == self.focus_id:
            return "car"
        return ("drone", "robot", "car")[(uid + self.seed) % 3]

    def _advance(self, env, observation, policy):
        state, mask, context = observation
        distribution = []
        value_estimate = None
        if isinstance(policy, SavedPPOPolicy):
            probabilities = policy._probabilities(state, mask, policy.weights, policy.bias)
            for action_id, probability in enumerate(probabilities):
                if mask[action_id]:
                    spec = ACTION_CATALOG[action_id]
                    label = "No reservation" if action_id == 0 else f"{spec.amount_fraction:.0%} · {spec.timing}"
                    distribution.append({"action": action_id, "label": label,
                                         "probability": float(probability)})
            distribution.sort(key=lambda item: item["probability"], reverse=True)
            if self.critic_w is not None:
                value_estimate = float(state @ self.critic_w + self.critic_b)
        action = int(policy(state, mask, context))
        prediction = context.get("prediction")
        result = env.step(action, observation)
        next_obs = (result.next_state, result.next_mask, result.info["next_context"])
        decision = {
            "action": action,
            "label": "No reservation" if action == 0 else f"{ACTION_CATALOG[action].amount_fraction:.0%} demand · {ACTION_CATALOG[action].timing}",
            "ue": prediction.ue_id if prediction else None,
            "target": prediction.target if prediction else None,
            "probability": prediction.probability if prediction else None,
            "uncertainty": prediction.target_uncertainty if prediction else None,
            "predicted_rbs": prediction.demand_rbs if prediction else None,
            "eta_steps": prediction.time_to_handover if prediction else None,
            "distribution": distribution,
            "value_estimate": value_estimate,
        }
        return result, next_obs, decision

    def step(self, count: int = 1) -> dict:
        count = max(1, min(count, 12))
        normal_decision = proposed_decision = None
        for _ in range(count):
            self.last_normal, self.normal_obs, normal_decision = self._advance(
                self.normal, self.normal_obs, no_reservation)
            self.last_proposed, self.proposed_obs, proposed_decision = self._advance(
                self.proposed, self.proposed_obs, self.policy)
            self._record_history()
        return self.snapshot(normal_decision, proposed_decision)

    def _metrics(self, env: RANHandoverEnv) -> dict:
        raw = env.metrics.summary(env.cfg.dt_s)
        return {key: _finite(value) for key, value in raw.items()}

    def _record_history(self) -> None:
        for name, env in (("normal", self.normal), ("proposed", self.proposed)):
            metrics = self._metrics(env)
            self.history[name].append({"t": env.t, "hsr": metrics["handover_success_rate"] or 0,
                                       "qos": metrics["qos_rate"] or 0})
            self.history[name] = self.history[name][-90:]

    def _network(self, env: RANHandoverEnv, result, decision) -> dict:
        held = env._reservation_totals()
        ues = []
        for ue in env.ues:
            speed = float((ue.vel[0] ** 2 + ue.vel[1] ** 2) ** 0.5)
            # An illustrative distance-to-serving-site indicator, not NR RSRP/SINR.
            distance = float(np.linalg.norm(ue.pos - env.centers[ue.cell]))
            signal_quality = float(np.clip(1.0 - distance / 48.0, 0.0, 1.0))
            ues.append({"id": ue.uid, "x": float(ue.pos[0]), "y": float(ue.pos[1]),
                        "vx": float(ue.vel[0]), "vy": float(ue.vel[1]), "speed": speed,
                        "slice": env.cfg.slices[ue.slice_id].name, "slice_id": ue.slice_id,
                        "cell": ue.cell, "demand": env.interval_demands[ue.uid],
                        "next_demand": ue.demand, "focus": ue.uid == self.focus_id,
                        "entity_type": self._entity_type(ue.uid),
                        "signal_quality": signal_quality,
                        "allocated": env.ue_allocations.get(ue.uid, 0)})
        cells = []
        for cell in range(4):
            alloc = int(env.allocations[cell]); reserved = int(held[cell])
            slices = []
            for slice_id, profile in enumerate(env.cfg.slices):
                members = [ue for ue in env.ues if ue.cell == cell and ue.slice_id == slice_id]
                slices.append({"id": slice_id, "name": profile.name, "ues": len(members),
                               "demand": sum(env.interval_demands[ue.uid] for ue in members),
                               "allocated": sum(env.ue_allocations.get(ue.uid, 0) for ue in members),
                               "reserved": sum(r.activated_amount for r in env.reservations
                                               if r.active and r.target == cell
                                               and env.ues[r.ue_id].slice_id == slice_id),
                               "sla_rbs_per_ue": profile.sla_rbs,
                               "latency_max_ms": profile.latency_max_ms})
            cells.append({"id": cell, "x": float(env.centers[cell, 0]), "y": float(env.centers[cell, 1]),
                          "capacity": env.cfg.cell_capacity, "allocated": alloc, "reserved": reserved,
                          "free": max(0, env.cfg.cell_capacity - alloc - reserved),
                          "load": (alloc + reserved) / env.cfg.cell_capacity,
                          "slices": slices})
        contracts = [{"ue": r.ue_id, "target": r.target, "amount": r.amount,
                      "active": r.active, "activate_at": r.activate_at, "expires_at": r.expires_at}
                     for r in env.reservations]
        info = result.info if result else {"events": [], "handovers": 0, "successes": 0, "qos": 0}
        metrics = self._metrics(env)
        metrics["handovers"] = env.metrics.handovers
        metrics["successes"] = env.metrics.successes
        metrics["qos_successes"] = env.metrics.qos_successes
        return {"time": env.t * env.cfg.dt_s, "step": env.t, "ues": ues, "cells": cells,
                "contracts": contracts, "events": info["events"][-6:], "instant": {
                    "handovers": info.get("handovers", 0), "successes": info.get("successes", 0),
                    "qos": info.get("qos", 0)}, "decision": decision, "metrics": metrics}

    def snapshot(self, normal_decision=None, proposed_decision=None) -> dict:
        return {"seed": self.seed, "scenario": self.scenario, "focus_id": self.focus_id,
                "controller": self.controller_name,
                "normal": self._network(self.normal, self.last_normal, normal_decision),
                "proposed": self._network(self.proposed, self.last_proposed, proposed_decision),
                "history": self.history}


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
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
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
            import json
            published = json.loads(RESULTS.read_text(encoding="utf-8"))
        learning = {"imitation": [], "ppo": []}
        for key, filename in (("imitation", "imitation_history.csv"), ("ppo", "training_history.csv")):
            path = ROOT / "results" / filename
            if path.exists():
                with path.open(encoding="utf-8") as fh:
                    learning[key] = [{name: float(value) for name, value in row.items()}
                                     for row in csv.DictReader(fh)]
        paired = []
        paired_path = ROOT / "results" / "paired_comparisons.csv"
        if paired_path.exists():
            with paired_path.open(encoding="utf-8") as fh:
                paired = [{name: (value if name in {"reference", "metric"} else float(value))
                           for name, value in row.items()} for row in csv.DictReader(fh)]
        return jsonify({"scenarios": SCENARIOS, "published": published, "learning": learning,
                        "paired": paired,
                        "model_available": MODEL.exists()})

    @app.post("/api/session")
    def new_session():
        body = request.get_json(silent=True) or {}
        scenario = str(body.get("scenario", "urban"))
        try:
            seed = int(body.get("seed", 2026))
        except (TypeError, ValueError):
            return jsonify({"error": "Seed must be an integer"}), 400
        if scenario not in SCENARIOS:
            return jsonify({"error": f"Unknown scenario: {scenario}"}), 400
        if not 0 <= seed <= 2**32 - 1:
            return jsonify({"error": "Seed must be between 0 and 4294967295"}), 400
        if not MODEL.exists():
            return jsonify({"error": "Saved PPO model is missing; run the experiment before opening the demo"}), 503
        token = secrets.token_urlsafe(12)
        with lock:
            sessions[token] = ComparisonSession(seed, scenario)
            if len(sessions) > 24:
                sessions.pop(next(iter(sessions)))
            snapshot = sessions[token].snapshot()
        return jsonify({"session": token, "snapshot": snapshot})

    @app.post("/api/step")
    def step():
        body = request.get_json(silent=True) or {}
        token = str(body.get("session", ""))
        with lock:
            session = sessions.get(token)
            if session is None:
                return jsonify({"error": "Unknown or expired simulation session"}), 404
            try:
                count = int(body.get("count", 1))
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
