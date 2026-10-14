# Simulation review — 4 October 2026

The browser presents the implemented capacity experiment without claims of zero-drop handovers, overload avoidance, measured signal strength, optical signaling, or guaranteed slice isolation. Following the requested UI revamp, the interface is independent of `design.md`: a colorful viewport-sized map workspace, native Bahnschrift/Segoe UI typography, enlarged SVG icons, compact controls, and a tabbed inspector. An expanded map mode makes background controls inert and restores focus on Escape. Wheel, double-click, and real touch pinch/pan gestures support 50–800% zoom.

## What was verified

Pastel cell regions, blue allocation, amber reservations, green/red handover results, and compact per-tower allocation cards explain the current sample. Cards use a smaller format on narrow maps and can be hidden through the Tower RBs layer. Device signal bars and their text badge use the existing distance proxy, with explicit thresholds and a visible explanation; they do not imply a physical radio measurement or a signal-based handover trigger.

Final validation: **84 Python tests passed**. Edge checks cover seven viewport sizes from 360×800 to 1920×1080, all four inspector tabs, expanded-mode focus containment, Escape restoration, zoom bounds, cursor anchoring, and actual two-finger pinch and subsequent single-finger panning. Updated previews include `results/lab_workspace.png`, `results/lab_expanded.png`, `results/lab_mobile_map.png`, and the full-page desktop/mobile/teaching screenshots.

- The saved six-policy, 20-seed, 320-interval benchmark was recomputed without retraining. All **2,546** checked aggregate values, per-seed metrics, and paired interval quantities matched the stored results within numerical tolerance.
- All four browser scenarios were checked on eight seeds and 240 intervals: **7,680 paired intervals**, **76,800 cell-ledger checks**, and **3,828 handover records**. Movement, demand, nearest-site assignment, and attempt counts remained identical across the baseline and PPO worlds.
- The seven-cell scenario now uses a true seven-hexagon domain with exact lattice spacing. Polygon centroids equal the site coordinates, areas are equal, and 18 exposed boundary edges reflect movement while shared edges permit handovers. Tests check speed preservation, containment, concave-notch collisions, and matching movement between policies. The benchmark continues to use its original square domain.
- Every device has a continuous current-cell label and serving-site line, including devices with zero allocated RBs (a distinct dotted association line). Handover history retains original times across batched steps, supplies all-device map highlights and result chips, and avoids duplicate handover entries on later snapshots.
- A selectable live association roster keeps every device/tower pair readable even when devices overlap. Map captions use collision-aware placement, tower markers align exactly with site coordinates, and curved handover arrows terminate outside destination icons. Browser checks verify roster contents and arrowhead clearance. The index is served without storage, and static UI assets require revalidation on refresh.
- Independent reservation tests cover wrong targets, pending holds at early arrival, failed radio draws, partial admission, surplus release, rejected activation, serial simultaneous activation, and inclusive expiry.
- The PPO actor gradient, including clipping and entropy, and the critic mean-square gradient were checked against central finite differences. Extreme negative legal logits remain normalized; illegal actions have zero probability. Masked softmax now uses negative infinity rather than a finite sentinel.
- The saved latency sample CSV independently reproduces the mean and 95th-percentile latency for all 120 method/seed runs. The benchmark CSV and JSON agree, including undefined values.
- Edge checks cover all four scenarios, replay with the same seed, per-device outcome history, cell-region area conservation and nearest-site membership, zoom/pan/tower selection, a pause during an in-flight response, exact paused sample positions, and mobile layout with every detail panel expanded.

The machine-readable extended audit is `results/verification.json`. Run it again with:

```powershell
.\.venv\Scripts\python.exe -m web.verify_simulation
.\.venv\Scripts\python.exe -m pytest -q
node tests/browser_smoke.mjs
```

The browser check needs the local server running and Microsoft Edge installed. The audit reads the original results and writes only its verification report.

## Implemented mathematical contract

**Capacity.** For each cell and snapshot, allocated RBs + active reserved RBs + free RBs = cell capacity. Pending contracts consume zero capacity. Allocation and reservation totals are integers; invalid fractional capacities are rejected at configuration time. Activation serializes due contracts and rechecks currently uncommitted capacity.

**Action.** The catalog is no-op plus half/full predicted demand activated immediately, at the midpoint, or one interval before forecast arrival. Half amounts use Python’s nearest-integer rounding, bounded below by one and above by predicted demand; for tiny demands distinct categorical actions can decode to the same amount. The mask protects current immediate capacity and a conservative total of pending requests. It does not certify future incumbent demand or all overlapping active holds; runtime activation is the final admission check.

**Admission.** At an actual target, potentially serviceable RBs = min(interval demand, matching active reservation + currently free RBs). Admission additionally requires the slice RB minimum and a successful independent radio draw. A failed attempt allocates zero destination RBs and uses zero reservation RBs. A successful attempt consumes min(demand, matching active held RBs) from the reservation, then releases surplus. Wrong-target holds cannot serve the arrival.

**QoS.** Handover QoS = admitted AND full interval RB demand met AND modeled latency ≤ the slice bound. Full demand by itself is not proof of QoS. Latency is a declared synthetic proxy in milliseconds:

```text
latency = slice_base_ms × (1 + 0.75 × target_load² + 3 × demand_shortfall)
target_load = (existing target allocation + active holds) / target capacity
demand_shortfall = max(0, 1 − serviceable RBs / interval demand)
slice_base_ms = 8 (eMBB), 2 (uRLLC), 22 (mMTC)
```

**Reward.** Per-interval reward is 2 × admission fraction + 3 × QoS fraction − 1.5 × settled unused RBs / total network capacity − 0.25 × active held RBs / total network capacity. Event fractions are zero in intervals without handovers. These are dimensionless quantities; no physical throughput measurement is included.

**Time.** One interval is 0.1 seconds; the forecast horizon is eight intervals (0.8 seconds), not eight seconds. Playback runs slowly for inspection. The clock, reservation activation time, forecast lead, and chart tooltips use the same simulation-time conversion. Events are displayed at the end of their completed interval; decisions refer to its start.

**Geometry.** Cell assignment is squared-Euclidean nearest-site selection. The map clips the world square by the same site bisectors to draw Voronoi regions, rather than imposing decorative hexagons. Site connections are an optional illustration. Motion can be interpolated between samples for display; allocation and admission facts belong to completed simulator intervals.

**Rates.** Admission and QoS denominators are actual handover attempts. Zero denominators remain undefined in JSON and display as a dash, including chart gaps. PPO and baseline can be equal or either can lead. Percentage-point differences are labeled `pp`.

**Live waste.** The offline `unused_reservation_fraction` settles remaining active contracts as unused at the evaluation boundary. In live playback, outstanding holds are not yet waste. The UI therefore uses `settled_unused_reservation_fraction`, whose denominator excludes still-active RBs, and separately displays cumulative reservation occupancy.

## Scope and remaining model limitations

`PROJECT_EQUATIONS.md` is a source-based proposed controller specification, not a claim that all of its proposed interfaces exist in this simulator. The implementation uses a heuristic forecast score instead of a calibrated next-target probability, a configured target-noise rate instead of matured Brier error, and capacity-normalized deviation from current demand instead of an empirical future-demand error history. It does not implement the proposed per-slice hard floors, learned LSTM predictor, probability calibration, or bit/s throughput mapping. These differences are deliberate synthetic abstractions and are now explicit in the UI and documentation.

The one-candidate rule sorts by estimated arrival, then descending heuristic score, then device ID. This differs from the specification’s suggested arrival/ID-only tie break. The implementation permits at most one active or pending contract per UE, which is stricter than one contract per UE/target pair.

Mobility is exogenous and cannot be steered by PPO. Coordinates are synthetic world units; velocities are displacement per interval. Interpreting the 100-unit world as 100 meters is a modeling assumption, not a calibrated mobility experiment. Priority scheduling can starve lower-priority devices. A failed handover still updates the geometric cell; the next interval reschedules it. There is no persistent disconnected-session state, packet buffer, measured packet loss, A3 event, beamforming, or NR PHY/MAC.

Reservation occupancy samples holds before motion and conversion; allocation utilization samples allocations after handover. They are different phases of an interval, so their cumulative percentages must not be added as if they formed a simultaneous capacity ledger. Snapshot allocated/held/free values do share a common phase and conserve capacity exactly.

The legacy `wrong_target_rate` includes active expiry and terminal censoring as no-successful-target outcomes; it is not just a count of wrong cell IDs. End-of-run settlement contributes offline waste accounting, but terminal outstanding waste is not retroactively added to the PPO rollout reward. Changing these evaluation/training conventions would require a new versioned experiment and a freshly evaluated model. The current review preserves the published benchmark and saved actor.

The paired 95% intervals use the fixed 20-run suite’s t critical value, 2.093. They summarize those synthetic runs and do not establish deployment reliability. The default seven-cell browser scenario is outside the saved actor’s four-cell training topology; it is an explanatory demonstration, not a new held-out performance guarantee.
