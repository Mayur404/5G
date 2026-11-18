from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .actions import ACTION_CATALOG, ACTION_CATALOG_V2, decode_action, decode_action_v2
from .config import SimConfig
from .geometry import HexDomain
from .models import UE, Prediction, Reservation, RunMetrics, StepResult

PolicyFn = Callable[[np.ndarray, np.ndarray, dict], int]


class RANHandoverEnv:
    """Causal RB-accounting simulator for mobility-driven sliced handovers.

    One interval is ordered as: decide -> activate -> allocate incumbents -> move
    -> resolve handovers -> release/expire. Reservations are hard admission-control
    holds, not physical-layer throughput guarantees.
    """

    state_dim = 14
    n_actions = len(ACTION_CATALOG)
    state_dim_v2 = 17

    def __init__(self, config: SimConfig | None = None):
        self.cfg = config or SimConfig()
        if self.cfg.action_set == "v2":
            # Instance attributes shadow the legacy class-level dimensions.
            self.state_dim = self.state_dim_v2
            self.n_actions = len(ACTION_CATALOG_V2)
        self.confidence_model = None
        default_centers = np.array([[25.0, 25.0], [75.0, 25.0], [25.0, 75.0], [75.0, 75.0]])
        self.centers = (
            np.array(self.cfg.centers, dtype=float)
            if self.cfg.centers is not None
            else default_centers
        )
        self.domain = (
            HexDomain(self.centers, self.cfg.hex_radius)
            if self.cfg.hex_radius is not None
            else None
        )
        self.reset(self.cfg.seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        root_seed = self.cfg.seed if seed is None else seed
        streams = np.random.SeedSequence(root_seed).spawn(3)
        self.rng = np.random.default_rng(streams[0])  # topology, traffic, mobility
        self.pred_rng = np.random.default_rng(streams[1])  # forecasts only
        self.radio_rng = np.random.default_rng(streams[2])  # signaling/radio outcomes
        self.t = 0
        self.neural_predictor = getattr(self, "neural_predictor", None)
        if self.neural_predictor is not None:
            self.neural_predictor._cache_env = None
        self.observation_history = []
        self.switch_candidates = {}
        self.previous_switches = {}
        self.next_rid = 1
        self.outage_until: dict[int, int] = {}
        self.contract_outcomes: dict[int, dict] = {}
        self.cell_unmet = np.zeros(len(self.centers), dtype=int)
        self.reservations: list[Reservation] = []
        self.metrics = RunMetrics()
        self.last_events: list[str] = []
        self.last_handovers: list[dict] = []
        self.last_contract_events: list[dict] = []
        self.ues: list[UE] = []
        slice_probs = np.array([0.42, 0.28, 0.30])
        for uid in range(self.cfg.n_ues):
            sid = int(self.rng.choice(3, p=slice_probs))
            pos = self.rng.uniform(5, self.cfg.world_size - 5, size=2)
            if self.domain is not None:
                # Sample within a tile; all equal-area cells are equally likely.
                center = self.centers[int(self.rng.integers(len(self.centers)))]
                while True:
                    delta = self.rng.uniform(-self.domain.radius, self.domain.radius, size=2)
                    if np.all(np.abs(delta @ self.domain.normals.T) <= self.domain.apothem):
                        pos = center + delta
                        break
            angle = self.rng.uniform(0, 2 * np.pi)
            speed = self.rng.uniform(0.9, 2.6) * (1.15 if sid == 0 else 1.0)
            vel = speed * np.array([np.cos(angle), np.sin(angle)])
            cell = self._nearest_cell(pos)
            self.ues.append(UE(uid, sid, pos, vel, cell))
        self.allocations = np.zeros(len(self.centers), dtype=int)
        self.ue_allocations: dict[int, int] = {}
        self._sample_demands()
        self.interval_demands = {ue.uid: ue.demand for ue in self.ues}
        self._allocate_incumbents(count_metrics=False)
        self._record_observation()
        return self.observe()

    def _record_observation(self) -> None:
        self.observation_history.append(np.array([[*u.pos, *u.vel, u.demand] for u in self.ues]))
        self.observation_history = self.observation_history[-6:]

    def signal_quality(self, position, cell: int) -> float:
        distance = math.hypot(
            float(position[0] - self.centers[cell, 0]), float(position[1] - self.centers[cell, 1])
        )
        return max(0.0, min(1.0, 1 - distance / self.cfg.signal_radius))

    def cell_scores(self, ue: UE, position, demand: int | None = None) -> np.ndarray:
        """Observable capacity/signal/latency scores; no future congestion is used."""
        demand = ue.demand if demand is None else demand
        profile = self.cfg.slices[ue.slice_id]
        scores = np.full(len(self.centers), -np.inf)
        # Reservations do not mutate while a candidate is scored. Reuse their
        # ledger instead of rebuilding it twice for every prospective cell.
        held_totals = self._reservation_totals()
        for cell in range(len(self.centers)):
            signal = self.signal_quality(position, cell)
            if signal < self.cfg.minimum_signal:
                continue
            own = self.ue_allocations.get(ue.uid, 0) if cell == ue.cell else 0
            held = sum(
                r.activated_amount
                for r in self.reservations
                if r.active and r.ue_id == ue.uid and r.target == cell
            )
            free = max(
                0, self.cfg.cell_capacity - int(self.allocations[cell]) - int(held_totals[cell])
            )
            available = min(demand, free + own + held)
            load = (self.allocations[cell] + held_totals[cell]) / self.cfg.cell_capacity
            latency = (8.0, 2.0, 22.0)[ue.slice_id] * (
                1 + 0.75 * load**2 + 3 * max(0, 1 - available / max(1, demand))
            )
            scores[cell] = (
                0.45 * available / max(1, demand)
                + 0.25 * signal
                + 0.20 * max(0, 1 - latency / profile.latency_max_ms)
                + 0.10 * (available >= profile.sla_rbs)
            )
        return scores

    def target_cell(self, ue: UE, position, demand: int | None = None) -> int:
        if self.cfg.handover_mode == "nearest":
            return self._nearest_cell(position)
        scores = self.cell_scores(ue, position, demand)
        if not np.isfinite(scores).any():
            return self._nearest_cell(position)
        best = int(np.argmax(scores))
        if np.isfinite(scores[ue.cell]) and scores[best] < scores[ue.cell] + self.cfg.switch_margin:
            return ue.cell
        return best

    def _trigger_target(self, ue, position, pending, demand=None):
        """Apply the same hysteresis and time-to-trigger in forecasts and motion."""
        target = self.target_cell(ue, position, demand)
        if self.cfg.handover_mode != "qos_aware" or target == ue.cell:
            return target, None
        previous, count = pending if pending else (target, 0)
        count = count + 1 if previous == target else 1
        if (
            count < self.cfg.time_to_trigger
            and self.signal_quality(position, ue.cell) >= self.cfg.minimum_signal
        ):
            return ue.cell, (target, count)
        return target, None

    def _nearest_cell(self, pos: np.ndarray) -> int:
        return int(np.argmin(np.sum((self.centers - pos) ** 2, axis=1)))

    def _sample_demands(self) -> None:
        phase = 1.0 + 0.16 * math.sin(2 * math.pi * self.t / 47.0)
        for ue in self.ues:
            profile = self.cfg.slices[ue.slice_id]
            burst = (
                1.55 if ue.slice_id == 1 and self.rng.random() < self.cfg.urllc_burst_prob else 1.0
            )
            ue.demand = max(
                1,
                int(
                    round(self.rng.normal(profile.demand_mean * phase * burst, profile.demand_std))
                ),
            )

    def _reservation_totals(self) -> np.ndarray:
        totals = np.zeros(len(self.centers), dtype=int)
        for r in self.reservations:
            if r.active:
                totals[r.target] += r.activated_amount
        return totals

    def in_outage(self, ue: UE) -> bool:
        """A UE recovering from a failed handover has no radio connection."""
        return self.outage_until.get(ue.uid, -1) >= self.t

    def admission_floor(self, ue: UE, demand: int | None = None) -> int:
        demand = ue.demand if demand is None else demand
        floor = self.cfg.slices[ue.slice_id].sla_rbs
        return min(floor, demand) if self.cfg.sla_capped_by_demand else floor

    def _allocate_incumbents(self, count_metrics: bool = True) -> dict[int, int]:
        held = self._reservation_totals()
        self.allocations[:] = 0
        self.cell_unmet[:] = 0
        ue_alloc: dict[int, int] = {}
        for cell in range(len(self.centers)):
            budget = max(0, self.cfg.cell_capacity - int(held[cell]))
            members = [u for u in self.ues if u.cell == cell and not self.in_outage(u)]
            members.sort(key=lambda u: (-self.cfg.slices[u.slice_id].priority, u.uid))
            for ue in members:
                granted = min(ue.demand, budget)
                ue_alloc[ue.uid] = granted
                budget -= granted
                self.allocations[cell] += granted
                self.cell_unmet[cell] += ue.demand - granted
                if count_metrics:
                    self.metrics.admission_requests += 1
                    if granted < self.admission_floor(ue):
                        self.metrics.blocked_requests += 1
            if count_metrics and held[cell] and self.cell_unmet[cell]:
                # Under the memoryless per-interval scheduler, removing the
                # holds would have granted exactly this many more incumbent RBs.
                displaced = min(int(held[cell]), int(self.cell_unmet[cell]))
                self.metrics.displaced_rb_intervals += displaced
                for r in self.reservations:
                    if r.active and r.target == cell:
                        r.displaced += displaced * r.activated_amount / int(held[cell])
        self.ue_allocations = ue_alloc
        return ue_alloc

    def _predict(self, ue: UE) -> Prediction | None:
        if self.neural_predictor is not None:
            from .neural import HORIZON

            if self.cfg.horizon != HORIZON:
                raise ValueError("Neural predictor requires the eight-interval training horizon")
            positions, demands = self.neural_predictor.forecast(self, ue)
            pending = self.switch_candidates.get(ue.uid)
            for k, (position, demand) in enumerate(zip(positions, demands, strict=True), 1):
                target, pending = self._trigger_target(ue, position, pending, int(demand))
                if target != ue.cell:
                    quality = self.neural_predictor.forecast_score
                    if self.cfg.action_set == "v2" or self.confidence_model is not None:
                        return self._v2_prediction(ue, target, k, int(demand), position, quality)
                    return Prediction(
                        ue.uid,
                        ue.cell,
                        target,
                        quality,
                        k,
                        int(demand),
                        1 - quality,
                        abs(int(demand) - ue.demand) / self.cfg.cell_capacity,
                        target,
                    )
            return None
        if self.cfg.prediction_mode == "kinematic":
            pending = self.switch_candidates.get(ue.uid)
            for k in range(1, self.cfg.horizon + 1):
                position = np.clip(ue.pos + ue.vel * k, 0, self.cfg.world_size)
                target, pending = self._trigger_target(ue, position, pending)
                if target != ue.cell:
                    if self.cfg.action_set == "v2" or self.confidence_model is not None:
                        return self._v2_prediction(ue, target, k, ue.demand, position, 0.77)
                    # Fixed controller score above the shared rule threshold;
                    # no injected error or oracle future observations.
                    return Prediction(
                        ue.uid, ue.cell, target, 0.77, k, ue.demand, 0.23, 0.0, target
                    )
            return None
        actual_target = ue.cell
        tth = self.cfg.horizon
        pending = self.switch_candidates.get(ue.uid)
        for k in range(1, self.cfg.horizon + 1):
            projected = ue.pos + ue.vel * k
            target, pending = self._trigger_target(ue, projected, pending)
            if target != ue.cell:
                actual_target, tth = target, k
                break
        if actual_target == ue.cell:
            return None

        regime = 0.5 + 0.5 * math.sin(2 * math.pi * (self.t + ue.uid) / 71.0)
        uncertainty = float(np.clip(0.05 + self.cfg.prediction_noise * regime, 0.02, 0.40))
        predicted_target = actual_target
        if self.pred_rng.random() < uncertainty:
            choices = [c for c in range(len(self.centers)) if c != ue.cell]
            predicted_target = int(self.pred_rng.choice(choices))
        noisy_tth = int(
            np.clip(
                round(tth + self.pred_rng.normal(0, 0.7 + 2 * uncertainty)), 1, self.cfg.horizon
            )
        )
        profile = self.cfg.slices[ue.slice_id]
        pred_demand = max(
            1,
            int(
                round(
                    ue.demand
                    + self.pred_rng.normal(0, profile.demand_std * (0.7 + 2 * uncertainty))
                )
            ),
        )
        probability = float(np.clip(0.94 - 0.055 * noisy_tth - 0.85 * uncertainty, 0.08, 0.96))
        demand_uncertainty = float(abs(pred_demand - ue.demand) / self.cfg.cell_capacity)
        return Prediction(
            ue.uid,
            ue.cell,
            predicted_target,
            probability,
            noisy_tth,
            pred_demand,
            uncertainty,
            demand_uncertainty,
            actual_target,
        )

    def prediction_features(self, ue: UE, target: int, lead: int, position) -> tuple[float, ...]:
        """Observable descriptors of one forecast, used only for confidence calibration."""
        serving_now = self.signal_quality(ue.pos, ue.cell)
        margin = self.signal_quality(position, target) - self.signal_quality(position, ue.cell)
        to_target = self.centers[target] - ue.pos
        distance = float(np.linalg.norm(to_target))
        speed = float(np.linalg.norm(ue.vel))
        heading = float(ue.vel @ to_target) / max(1e-9, distance * speed) if speed > 0 else 0.0
        # Distance to the source/target bisector, in units of current speed.
        midpoint = 0.5 * (self.centers[target] + self.centers[ue.cell])
        normal = self.centers[target] - self.centers[ue.cell]
        normal = normal / max(1e-9, float(np.linalg.norm(normal)))
        gap = float((midpoint - ue.pos) @ normal)
        closing = float(ue.vel @ normal)
        return (
            lead / self.cfg.horizon,
            float(margin),
            serving_now,
            speed / 4.0,
            heading,
            gap / 10.0,
            closing / 4.0,
        )

    def _v2_prediction(
        self, ue: UE, target: int, lead: int, demand: int, position, default_score: float
    ) -> Prediction:
        features = self.prediction_features(ue, target, lead, position)
        score = (
            float(self.confidence_model(features))
            if self.confidence_model is not None
            else default_score
        )
        std = self.cfg.slices[ue.slice_id].demand_std
        upper = max(demand, int(math.ceil(demand + std)))
        return Prediction(
            ue.uid,
            ue.cell,
            target,
            score,
            lead,
            max(1, int(demand)),
            1.0 - score,
            abs(int(demand) - ue.demand) / self.cfg.cell_capacity,
            target,
            features,
            upper,
        )

    def predictions(self) -> list[Prediction]:
        """Every uncontracted, connected UE with a forecast cell change, by urgency."""
        contracted = {r.ue_id for r in self.reservations}
        predictions = [
            p
            for u in self.ues
            if u.uid not in contracted
            and not self.in_outage(u)
            and (p := self._predict(u)) is not None
        ]
        predictions.sort(key=lambda p: (p.time_to_handover, -p.probability, p.ue_id))
        return predictions

    def _candidate(self) -> Prediction | None:
        predictions = self.predictions()
        return predictions[0] if predictions else None

    def activation_capacity(self, cell: int) -> int:
        """RBs a new hold may claim at a cell under the configured activation rule."""
        if self.cfg.activation_rule == "free":
            return self._free(cell)
        floors = sum(
            self.admission_floor(u) for u in self.ues if u.cell == cell and not self.in_outage(u)
        )
        return max(0, self.cfg.cell_capacity - int(self._reservation_totals()[cell]) - int(floors))

    def _free(self, cell: int) -> int:
        return max(
            0,
            self.cfg.cell_capacity
            - int(self.allocations[cell])
            - int(self._reservation_totals()[cell]),
        )

    def _encode_v2(self, pred: Prediction | None) -> tuple[np.ndarray, np.ndarray, dict]:
        cap = self.cfg.cell_capacity
        state = np.zeros(self.state_dim_v2, dtype=float)
        mask = np.zeros(len(ACTION_CATALOG_V2), dtype=bool)
        mask[0] = True
        context: dict = {"prediction": pred, "env": self}
        if pred is None:
            state[-1] = 1.0
            return state, mask, context
        ue = self.ues[pred.ue_id]
        profile = self.cfg.slices[ue.slice_id]
        totals = self._reservation_totals()
        free = self._free(pred.target)
        claimable = self.activation_capacity(pred.target)
        onehot = np.zeros(3)
        onehot[ue.slice_id] = 1.0
        floor = self.admission_floor(ue, pred.demand_rbs)
        state[:] = [
            pred.probability,
            pred.time_to_handover / self.cfg.horizon,
            pred.demand_rbs / cap,
            floor / cap,
            free / cap,
            claimable / cap,
            self.allocations[pred.target] / cap,
            self.cell_unmet[pred.target] / cap,
            totals[pred.target] / cap,
            *onehot,
            pred.features[1] if pred.features else 0.0,
            np.linalg.norm(ue.vel) / 4.0,
            self.cfg.failure_outage / 10.0,
            ue.demand / cap,
            1.0,
        ]
        scheduled = sum(
            r.amount for r in self.reservations if r.target == pred.target and not r.active
        )
        for a in range(1, len(ACTION_CATALOG_V2)):
            amount, delay = decode_action_v2(
                a, pred.demand_rbs, pred.time_to_handover, floor, pred.demand_upper
            )
            current_ok = delay > 0 or self.cfg.preparation_delay > 0 or amount <= claimable
            if current_ok and amount <= cap - scheduled - int(totals[pred.target]):
                mask[a] = True
        context.update(
            {
                "ue": ue,
                "profile": profile,
                "free": free,
                "claimable": claimable,
                "floor": floor,
                "actual_target": pred.actual_target_hint,
            }
        )
        return state, mask, context

    def observe_many(self) -> list[tuple[np.ndarray, np.ndarray, dict]]:
        """Up to ``max_decisions`` independent decisions for the next interval."""
        encode = self._encode_v2 if self.cfg.action_set == "v2" else self._encode
        return [encode(p) for p in self.predictions()[: self.cfg.max_decisions]]

    def _encode(self, pred: Prediction | None) -> tuple[np.ndarray, np.ndarray, dict]:
        if self.cfg.action_set == "v2":
            return self._encode_v2(pred)
        state = np.zeros(self.state_dim, dtype=float)
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[0] = True
        context: dict = {"prediction": pred, "env": self}
        if pred is None:
            state[-1] = 1.0
            return state, mask, context
        ue = self.ues[pred.ue_id]
        profile = self.cfg.slices[ue.slice_id]
        free = self._free(pred.target)
        load = self.allocations[pred.target] / self.cfg.cell_capacity
        onehot = np.zeros(3)
        onehot[ue.slice_id] = 1.0
        speed = np.linalg.norm(ue.vel) / 4.0
        state[:] = [
            pred.probability,
            pred.time_to_handover / self.cfg.horizon,
            pred.demand_rbs / self.cfg.cell_capacity,
            free / self.cfg.cell_capacity,
            load,
            pred.target_uncertainty,
            pred.demand_uncertainty,
            *onehot,
            profile.sla_rbs / self.cfg.cell_capacity,
            profile.latency_max_ms / 100.0,
            speed,
            1.0,
        ]
        has_contract = any(r.ue_id == ue.uid for r in self.reservations)
        if not has_contract and pred.probability > 0 and pred.demand_rbs > 0:
            scheduled = sum(
                r.amount for r in self.reservations if r.target == pred.target and not r.active
            )
            for a in range(1, self.n_actions):
                amount, delay = decode_action(a, pred.demand_rbs, pred.time_to_handover)
                current_ok = delay > 0 or amount <= free
                if (
                    current_ok
                    and delay <= pred.time_to_handover
                    and amount <= self.cfg.cell_capacity - scheduled
                ):
                    mask[a] = True
        context.update(
            {"ue": ue, "profile": profile, "free": free, "actual_target": pred.actual_target_hint}
        )
        return state, mask, context

    def observe(self) -> tuple[np.ndarray, np.ndarray, dict]:
        return self._encode(self._candidate())

    def _create_reservation(self, action: int, pred: Prediction | None) -> int | None:
        if action == 0 or pred is None:
            return None
        if self.cfg.action_set == "v2":
            ue = self.ues[pred.ue_id]
            amount, delay = decode_action_v2(
                action,
                pred.demand_rbs,
                pred.time_to_handover,
                self.admission_floor(ue, pred.demand_rbs),
                pred.demand_upper,
            )
            # Handover preparation (request/admission signaling) bounds the
            # earliest activation; a hold can never become active sooner.
            delay = max(delay, self.cfg.preparation_delay)
        else:
            amount, delay = decode_action(action, pred.demand_rbs, pred.time_to_handover)
        reservation = Reservation(
            self.next_rid,
            pred.ue_id,
            pred.target,
            amount,
            self.t,
            self.t + delay,
            self.t + self.cfg.horizon,
            pred.target,
        )
        if self.cfg.action_set == "v2":
            reservation.timing = ACTION_CATALOG_V2[action].timing
        self.reservations.append(reservation)
        self._contract_event("requested", reservation)
        self.next_rid += 1
        self.last_events.append(f"reserve UE{pred.ue_id} {amount}RB -> C{pred.target} (+{delay})")
        return reservation.rid

    def _record_outcome(self, r: Reservation, status: str, **values) -> None:
        """Per-contract attribution used for learning and auditing, never as policy input."""
        self.contract_outcomes[r.rid] = {
            "rid": r.rid,
            "ue": r.ue_id,
            "target": r.target,
            "amount": r.amount,
            "status": status,
            "displaced": float(r.displaced),
            "rescued": False,
            "qos_rescued": False,
            "arrival_demand": 0,
            "slice_id": self.ues[r.ue_id].slice_id if r.ue_id < len(self.ues) else 0,
            "timing": r.timing,
            **values,
        }

    def _contract_event(self, kind: str, reservation: Reservation, used: int = 0) -> None:
        self.last_contract_events.append(
            {
                "kind": kind,
                "reservation": reservation.rid,
                "ue": reservation.ue_id,
                "target": reservation.target,
                "amount_rbs": reservation.amount,
                "used_rbs": used,
                "active": reservation.active,
                "activate_at": reservation.activate_at,
                "expires_at": reservation.expires_at,
            }
        )

    def _retime_pending(self) -> None:
        """Re-anchor pending holds to the newest forecast of the same target.

        The request time plus preparation delay still bounds activation, and the
        target is never changed; only the activation instant tracks the lead.
        """
        for r in self.reservations:
            if r.active or not r.timing or r.ue_id >= len(self.ues):
                continue
            prediction = self._predict(self.ues[r.ue_id])
            if prediction is None or prediction.target != r.target:
                continue
            lead = prediction.time_to_handover
            delay = {"now": 0, "mid": lead // 2, "jit": lead - 1}[r.timing]
            r.activate_at = max(r.created_at + self.cfg.preparation_delay, self.t + delay)
            r.expires_at = max(r.expires_at, self.t + lead + 1)

    def _activate_due(self) -> None:
        for r in sorted(self.reservations, key=lambda x: (x.activate_at, x.ue_id, x.target)):
            if r.active or r.activate_at > self.t:
                continue
            self.metrics.activation_attempts += 1
            free = self.activation_capacity(r.target)
            if self.outage_until.get(r.ue_id, -1) >= self.t:
                free = -1  # A disconnected UE cannot activate a handover hold.
            if r.amount <= free:
                r.active = True
                r.activated_amount = r.amount
                self.metrics.activated_rbs += r.amount
                self._contract_event("activated", r)
                self.last_events.append(f"activate R{r.rid}: {r.amount}RB at C{r.target}")
            else:
                self.metrics.activation_rejects += 1
                self._contract_event("rejected", r)
                self.last_events.append(f"reject R{r.rid}: C{r.target} lacks {r.amount}RB")
                self._record_outcome(r, "rejected")
                self.reservations.remove(r)

    def _move(self) -> list[tuple[UE, int, int]]:
        events = []
        for ue in self.ues:
            ue.vel += self.rng.normal(0, self.cfg.mobility_noise, size=2)
            speed = np.linalg.norm(ue.vel)
            if speed > 4.0:
                ue.vel *= 4.0 / speed
            if self.domain is not None:
                ue.pos, ue.vel = self.domain.move(ue.pos, ue.vel)
            else:
                ue.pos += ue.vel
                for dim in (0, 1):
                    if ue.pos[dim] < 0 or ue.pos[dim] > self.cfg.world_size:
                        ue.vel[dim] *= -1
                        ue.pos[dim] = np.clip(ue.pos[dim], 0, self.cfg.world_size)
            new_cell, pending = self._trigger_target(ue, ue.pos, self.switch_candidates.get(ue.uid))
            if pending:
                self.switch_candidates[ue.uid] = pending
            else:
                self.switch_candidates.pop(ue.uid, None)
            if new_cell != ue.cell:
                if self.in_outage(ue):
                    # A disconnected UE re-establishes at its current best cell;
                    # this is not a prepared handover attempt.
                    ue.cell = new_cell
                    continue
                events.append((ue, ue.cell, new_cell))
                ue.cell = new_cell
        return events

    def _finish_contract(self, r: Reservation, used: int, wrong: bool) -> int:
        if r.active:
            self.metrics.used_reserved_rbs += used
            self.metrics.completed_contracts += 1
            self.metrics.wrong_target_contracts += int(wrong)
            return r.activated_amount - used
        return 0

    def _resolve_handovers(self, events: list[tuple[UE, int, int]]) -> tuple[int, int, int]:
        successes = qos = waste = 0
        # Concurrent arrivals follow the same slice priority as incumbent
        # scheduling; UID provides deterministic ordering within a slice.
        for ue, source, target in sorted(
            events, key=lambda event: (-self.cfg.slices[event[0].slice_id].priority, event[0].uid)
        ):
            # Release this UE's source allocation before any destination admission.
            # Each event then sees the capacity consumed by earlier admissions.
            source_rbs = self.ue_allocations.pop(ue.uid, 0)
            self.allocations[source] -= source_rbs
            self.metrics.handovers += 1
            self.metrics.handovers_by_slice[ue.slice_id] += 1
            profile = self.cfg.slices[ue.slice_id]
            matching = [
                r
                for r in self.reservations
                if r.ue_id == ue.uid and r.target == target and r.active
            ]
            match_rbs = sum(r.activated_amount for r in matching)
            held = self._reservation_totals()
            free = max(
                0, self.cfg.cell_capacity - int(self.allocations[target]) - int(held[target])
            )
            serviceable = min(ue.demand, match_rbs + free)
            radio_ok = (
                self.interval_radio[ue.uid]
                if self.cfg.common_radio_stream
                else self.radio_rng.random()
            ) > 0.01
            floor = self.admission_floor(ue)
            success = serviceable >= floor and radio_ok
            load = (self.allocations[target] + held[target]) / self.cfg.cell_capacity
            shortfall = max(0.0, 1.0 - serviceable / max(ue.demand, 1))
            base_latency = (8.0, 2.0, 22.0)[ue.slice_id]
            latency = base_latency * (1.0 + 0.75 * load**2 + 3.0 * shortfall)
            qos_ok = success and serviceable >= ue.demand and latency <= profile.latency_max_ms
            # One-interval counterfactual without this UE's matching holds: the
            # memoryless scheduler would first give those RBs to unmet incumbents.
            free_without = free + max(0, match_rbs - int(self.cell_unmet[target]))
            serviceable_without = min(ue.demand, free_without)
            success_without = serviceable_without >= floor and radio_ok
            load_without = (
                load
                - (match_rbs - min(match_rbs, int(self.cell_unmet[target])))
                / self.cfg.cell_capacity
            )
            latency_without = base_latency * (
                1.0
                + 0.75 * load_without**2
                + 3.0 * max(0.0, 1.0 - serviceable_without / max(ue.demand, 1))
            )
            qos_without = (
                success_without
                and serviceable_without >= ue.demand
                and latency_without <= profile.latency_max_ms
            )
            rescued = bool(success and not success_without)
            qos_rescued = bool(qos_ok and not qos_without)
            self.metrics.rescued_handovers += int(rescued)
            if not success and self.cfg.failure_outage:
                # Failed admission triggers radio-link re-establishment.
                self.outage_until[ue.uid] = self.t + self.cfg.failure_outage
            successes += int(success)
            qos += int(qos_ok)
            self.metrics.successes += int(success)
            self.metrics.qos_successes += int(qos_ok)
            self.metrics.successes_by_slice[ue.slice_id] += int(success)
            self.metrics.qos_by_slice[ue.slice_id] += int(qos_ok)
            previous = self.previous_switches.get(ue.uid)
            if previous and previous[0] == target and self.t - previous[1] <= 10:
                self.metrics.pingpong_handovers += 1
            self.previous_switches[ue.uid] = (source, self.t)
            self.metrics.latency_ms.append(latency)
            self.metrics.latency_slice_ids.append(ue.slice_id)
            if match_rbs:
                self.metrics.active_coverage += 1
                earliest = min(r.activate_at for r in matching)
                self.metrics.lead_intervals.append(max(0, self.t - earliest))
            ue_contracts = [r for r in self.reservations if r.ue_id == ue.uid]
            remaining_use = min(ue.demand, match_rbs) if success else 0
            for r in ue_contracts:
                used = min(r.activated_amount, remaining_use) if r.target == target else 0
                remaining_use -= used
                self._contract_event("settled" if r.active else "cancelled", r, used)
                waste += self._finish_contract(r, used, r.target != target)
                matched = r.active and r.target == target
                self._record_outcome(
                    r,
                    ("settled" if r.target == target else "wrong_target")
                    if r.active
                    else "cancelled",
                    rescued=rescued and matched,
                    qos_rescued=qos_rescued and matched,
                    arrival_demand=int(ue.demand),
                    used=int(used),
                    slice_id=ue.slice_id,
                )
                self.reservations.remove(r)
            if success:
                self.allocations[target] += serviceable
                self.ue_allocations[ue.uid] = serviceable
            assert np.all(self.allocations + self._reservation_totals() <= self.cfg.cell_capacity)
            status = "SUCCESS" if success else "FAIL"
            self.last_handovers.append(
                {
                    "ue": ue.uid,
                    "source": source,
                    "target": target,
                    "success": bool(success),
                    "qos": bool(qos_ok),
                    "available_rbs": serviceable,
                    "allocated_rbs": serviceable if success else 0,
                    "demand_rbs": ue.demand,
                    "minimum_rbs": profile.sla_rbs,
                    "reserved_rbs": match_rbs,
                    "radio_ok": bool(radio_ok),
                    "latency_ms": float(latency),
                    "latency_limit_ms": profile.latency_max_ms,
                }
            )
            self.last_events.append(
                f"HO UE{ue.uid} C{source}->C{target} {status} ({serviceable}/{ue.demand}RB)"
            )
        return successes, qos, waste

    def _expire(self) -> int:
        waste = 0
        for r in list(self.reservations):
            if self.t >= r.expires_at:
                self._contract_event("expired", r)
                self._record_outcome(r, "expired" if r.active else "cancelled")
                waste += self._finish_contract(r, 0, True)
                self.reservations.remove(r)
                self.last_events.append(f"expire R{r.rid}")
        return waste

    def step(
        self, action: int, observation: tuple[np.ndarray, np.ndarray, dict] | None = None
    ) -> StepResult:
        state, mask, context = observation if observation is not None else self.observe()
        if action < 0 or action >= self.n_actions or not mask[action]:
            raise ValueError(f"inadmissible action {action}")
        reward, info = self._advance([(action, context.get("prediction"))])
        next_obs = self.observe()
        info["next_context"] = next_obs[2]
        done = self.t >= self.cfg.episode_steps
        return StepResult(state, mask, action, reward, next_obs[0], next_obs[1], done, info)

    def step_many(
        self, actions: list[int], observations: list[tuple[np.ndarray, np.ndarray, dict]]
    ) -> tuple[float, bool, dict]:
        """Apply one action per observation from :meth:`observe_many`, then advance."""
        if len(actions) != len(observations):
            raise ValueError("one action is required per observation")
        decisions = []
        for action, (_state, mask, context) in zip(actions, observations, strict=True):
            if action < 0 or action >= len(mask) or not mask[action]:
                raise ValueError(f"inadmissible action {action}")
            decisions.append((int(action), context.get("prediction")))
        reward, info = self._advance(decisions)
        return reward, self.t >= self.cfg.episode_steps, info

    def _advance(self, decisions: list[tuple[int, object]]) -> tuple[float, dict]:
        self.last_events = []
        self.last_handovers = []
        self.last_contract_events = []
        reservation_ids = []
        if self.cfg.retime_pending:
            self._retime_pending()
        for action, prediction in decisions:
            self.metrics.decisions += int(prediction is not None)
            reservation_ids.append(self._create_reservation(action, prediction))
        self._activate_due()
        self.interval_demands = {ue.uid: ue.demand for ue in self.ues}
        self._allocate_incumbents(count_metrics=True)
        if self.cfg.common_radio_stream:
            self.interval_radio = self.radio_rng.random(self.cfg.n_ues)
        held_during_interval = int(self._reservation_totals().sum())
        events = self._move()
        successes, qos, waste = self._resolve_handovers(events)
        waste += self._expire()

        if self.cfg.research_metrics:
            for ue in self.ues:
                grant = self.ue_allocations.get(ue.uid, 0)
                covered = self.signal_quality(ue.pos, ue.cell) >= self.cfg.minimum_signal
                effective_grant = grant if covered else 0
                self.metrics.service_ue_intervals += 1
                self.metrics.outage_ue_intervals += int(self.in_outage(ue))
                self.metrics.full_service_intervals += int(effective_grant >= ue.demand)
                self.metrics.sla_service_intervals += int(
                    effective_grant >= self.cfg.slices[ue.slice_id].sla_rbs
                )
                self.metrics.offered_rbs += ue.demand
                self.metrics.served_rbs += effective_grant
                sid = ue.slice_id
                self.metrics.service_ue_intervals_by_slice[sid] += 1
                self.metrics.full_service_intervals_by_slice[sid] += int(
                    effective_grant >= ue.demand
                )
                self.metrics.sla_service_intervals_by_slice[sid] += int(
                    effective_grant >= self.cfg.slices[sid].sla_rbs
                )
                self.metrics.offered_rbs_by_slice[sid] += ue.demand
                self.metrics.served_rbs_by_slice[sid] += effective_grant
                self.metrics.offered_rbs_by_ue[ue.uid] = (
                    self.metrics.offered_rbs_by_ue.get(ue.uid, 0) + ue.demand
                )
                self.metrics.served_rbs_by_ue[ue.uid] = (
                    self.metrics.served_rbs_by_ue.get(ue.uid, 0) + effective_grant
                )

        self.metrics.occupancy_rb_intervals += held_during_interval
        self.metrics.allocated_rb_intervals += int(self.allocations.sum())
        self.metrics.capacity_rb_intervals += self.cfg.total_capacity
        h_term = successes / len(events) if events else 0.0
        q_term = qos / len(events) if events else 0.0
        reward = (
            self.cfg.reward_success * h_term
            + self.cfg.reward_qos * q_term
            - self.cfg.reward_waste * waste / self.cfg.total_capacity
            - self.cfg.reward_occupancy * held_during_interval / self.cfg.total_capacity
        )
        self.metrics.reward_sum += reward
        self.metrics.steps += 1
        self.t += 1
        self._sample_demands()
        self._record_observation()
        info = {
            "events": list(self.last_events),
            "handovers": len(events),
            "successes": successes,
            "handover_details": list(self.last_handovers),
            "contract_events": list(self.last_contract_events),
            "qos": qos,
            "waste_rbs": waste,
            "held_rbs": held_during_interval,
            "allocations": self.allocations.copy(),
            "reservations": self._reservation_totals().copy(),
            "reservation_ids": reservation_ids,
        }
        return reward, info

    def run(
        self, policy: PolicyFn, steps: int | None = None, seed: int | None = None
    ) -> RunMetrics:
        observation = self.reset(seed)
        n_steps = steps or self.cfg.episode_steps
        for _ in range(n_steps):
            state, mask, context = observation
            action = int(policy(state, mask, context))
            result = self.step(action, observation)
            observation = (result.next_state, result.next_mask, result.info["next_context"])
        self.settle_open_contracts()
        return self.metrics

    def settle_open_contracts(self) -> None:
        """Settle active contracts as censored expiries to avoid hiding terminal waste."""
        for r in list(self.reservations):
            self._record_outcome(r, "censored")
            self._finish_contract(r, 0, True)
            self.reservations.remove(r)

    def run_many(
        self, policy: PolicyFn, steps: int | None = None, seed: int | None = None
    ) -> RunMetrics:
        """Run with up to ``max_decisions`` independent policy calls per interval."""
        self.reset(seed)
        for _ in range(steps or self.cfg.episode_steps):
            observations = self.observe_many()
            actions = [int(policy(*observation)) for observation in observations]
            self.step_many(actions, observations)
        self.settle_open_contracts()
        return self.metrics
