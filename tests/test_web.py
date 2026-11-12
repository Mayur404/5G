import math

import pytest

from web.server import CARE_MECHANICS, ComparisonSession, care_available, create_app


def test_health_and_static_index():
    client = create_app().test_client()
    assert client.get("/health").get_json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200
    assert b"Predictive handover lab" in page.data
    assert b"Network overview" in page.data
    assert b"Pause" in page.data
    assert b"zoomIn" in page.data
    assert b"zoomOut" in page.data
    assert b"deviceSelect" in page.data
    assert page.headers["Cache-Control"] == "no-store"
    assert "must-revalidate" in client.get("/static/lab.js").headers["Cache-Control"]
    assert client.get("/paper.tex").status_code == 200
    assert client.get("/references.bib").status_code == 200
    meta = client.get("/api/meta").get_json()
    assert [row["name"] for row in meta["slices"]] == ["eMBB", "uRLLC", "mMTC"]
    assert set(meta["controllers"]) == {"care", "deep_qos", "deep_nearest"}


def test_comparison_api_uses_paired_mobility_and_safe_capacity():
    client = create_app().test_client()
    created = client.post("/api/session", json={"scenario": "crossing", "seed": 118})
    assert created.status_code == 200
    payload = created.get_json()
    token = payload["session"]
    for _ in range(8):
        response = client.post("/api/step", json={"session": token, "count": 3})
        assert response.status_code == 200
        snapshot = response.get_json()
        normal = snapshot["normal"]
        proposed = snapshot["proposed"]
        assert len(normal["ues"]) == len(proposed["ues"]) == 6
        for left, right in zip(normal["ues"], proposed["ues"], strict=True):
            assert left["id"] == right["id"]
            assert left["x"] == right["x"]
            assert left["y"] == right["y"]
        for network in (normal, proposed):
            for cell in network["cells"]:
                assert cell["allocated"] + cell["reserved"] <= cell["capacity"]
                assert cell["free"] >= 0
                assert sum(row["allocated"] for row in cell["slices"]) == cell["allocated"]
                assert sum(row["reserved"] for row in cell["slices"]) == cell["reserved"]
                assert sum(row["ues"] for row in cell["slices"]) == sum(
                    ue["cell"] == cell["id"] for ue in network["ues"]
                )
                assert sum(row["demand"] for row in cell["slices"]) == sum(
                    ue["demand"] for ue in network["ues"] if ue["cell"] == cell["id"]
                )
            assert {ue["entity_type"] for ue in network["ues"]} == {"car", "drone", "robot"}
            assert all(0 <= ue["signal_quality"] <= 1 for ue in network["ues"])
            assert network["metrics"]["successes"] <= network["metrics"]["handovers"]
        decision = proposed["decision"]
        if decision and decision["ue"] is not None:
            values = [item["value"] for item in decision["distribution"]]
            assert values == sorted(values, reverse=True)
            assert decision["committed"] <= decision["candidates"]


def test_unknown_session_is_reported():
    client = create_app().test_client()
    response = client.post("/api/step", json={"session": "missing"})
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_teaching_scenarios_are_six_device_capacity_cases():
    assert ComparisonSession(21, "urban").normal.cfg.cell_capacity == 18
    assert ComparisonSession(118, "crossing").normal.cfg.cell_capacity == 20
    assert all(
        ComparisonSession(seed, scenario).normal.cfg.n_ues == 6
        for scenario, seed in (("urban", 21), ("crossing", 118))
    )


def test_curated_trace_shows_care_rescuing_a_urllc_handover():
    session = ComparisonSession(21, "urban")
    snap = session.step()
    first = snap["proposed"]["decision"]
    assert first["ue"] == 1 and first["target"] == 0 and first["amount_rbs"] == 5
    assert first["action"] != 0 and "admission floor" in first["label"]
    snap = session.step(3)
    target = snap["proposed"]["cells"][0]
    assert target["reserved"] == 5
    assert any(
        c["ue"] == 1 and c["target"] == 0 and c["active"] for c in snap["proposed"]["contracts"]
    )
    snap = session.step(3)
    assert any("HO UE1 C2->C0 SUCCESS (5/7RB)" in e for e in snap["proposed"]["events"])
    assert any("HO UE1 C2->C0 FAIL (3/7RB)" in e for e in snap["normal"]["events"])


def test_six_device_paired_trace_and_capacity_invariants_across_seeds():
    for scenario, seed in (("urban", 21), ("urban", 65), ("crossing", 118), ("crossing", 99)):
        session = ComparisonSession(seed, scenario)
        for _ in range(75):
            snap = session.step()
            for left, right in zip(snap["normal"]["ues"], snap["proposed"]["ues"], strict=True):
                assert (left["x"], left["y"], left["cell"], left["demand"]) == (
                    right["x"],
                    right["y"],
                    right["cell"],
                    right["demand"],
                )
            assert (
                snap["normal"]["metrics"]["handovers"] == snap["proposed"]["metrics"]["handovers"]
            )
            for network in (snap["normal"], snap["proposed"]):
                assert len(network["ues"]) == 6
                for cell in network["cells"]:
                    assert cell["allocated"] + cell["reserved"] + cell["free"] == cell["capacity"]
                    assert cell["allocated"] >= 0 and cell["reserved"] >= 0


def test_second_six_device_trace_has_measured_early_gain():
    session = ComparisonSession(118, "crossing")
    session.step(2)
    result = session.step()
    assert any("HO UE2 C2->C0 FAIL (4/6RB)" in event for event in result["normal"]["events"])
    assert any("HO UE2 C2->C0 SUCCESS (5/6RB)" in event for event in result["proposed"]["events"])
    assert result["normal"]["metrics"]["successes"] == 0
    assert result["proposed"]["metrics"]["successes"] == 1


def test_web_refuses_to_run_care_without_trained_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("web.server.CARE_DIR", tmp_path)
    client = create_app().test_client()
    response = client.post("/api/session", json={"scenario": "urban", "controller": "care"})
    assert response.status_code == 503
    assert "CARE" in response.get_json()["error"]
    assert client.post("/api/session", json={"controller": "deep_qos"}).status_code == 200


def test_bad_inputs_return_client_errors_and_security_headers():
    client = create_app().test_client()
    response = client.post("/api/session", json={"scenario": "unknown", "seed": -1})
    assert response.status_code == 400
    response = client.post("/api/session", json={"scenario": "urban", "seed": "not-a-number"})
    assert response.status_code == 400
    page = client.get("/")
    assert page.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in page.headers["Content-Security-Policy"]


def test_history_preserves_undefined_rates_and_time_units():
    session = ComparisonSession(21, "urban")
    snap = session.step()
    assert snap["dt_s"] == 0.1
    assert snap["history"]["proposed"][0]["time"] == 0.1
    assert snap["history"]["proposed"][0]["hsr"] is None
    assert snap["history"]["normal"][0]["qos"] is None
    decision = snap["proposed"]["decision"]
    assert decision["eta_s"] == decision["eta_steps"] * snap["dt_s"]
    assert decision["delay_s"] == decision["delay_steps"] * snap["dt_s"]


def test_live_waste_excludes_still_active_holds():
    session = ComparisonSession(21, "urban")
    snap = session.step(4)
    assert snap["proposed"]["cells"][0]["reserved"] == 5
    assert snap["proposed"]["metrics"]["settled_unused_reservation_fraction"] is None
    snap = session.step(3)
    assert snap["proposed"]["metrics"]["settled_unused_reservation_fraction"] == 0


def test_handover_records_separate_admission_allocation_and_qos():
    snap = ComparisonSession(21, "urban").step(7)
    base = next(e for e in snap["normal"]["handover_details"] if e["ue"] == 1)
    care = next(e for e in snap["proposed"]["handover_details"] if e["ue"] == 1)
    assert base["available_rbs"] == 3 and base["allocated_rbs"] == 0
    assert not base["success"] and not base["qos"]
    # The held admission floor admits the arrival; at 5 of 7 RB its modeled
    # latency exceeds the uRLLC limit, so admission and QoS are reported separately.
    assert care["reserved_rbs"] == care["allocated_rbs"] == care["minimum_rbs"] == 5
    assert care["success"] and not care["qos"]
    assert care["latency_ms"] > care["latency_limit_ms"]
    settled = next(
        e for e in snap["proposed"]["contract_events"] if e["kind"] == "settled" and e["ue"] == 1
    )
    assert settled["used_rbs"] == settled["amount_rbs"] == 5


def test_handover_history_survives_batch_steps_and_carries_original_time():
    snap = ComparisonSession(21, "urban").step(12)
    event = next(e for e in snap["proposed"]["recent_handovers"] if e["ue"] == 1)
    assert event["source"] == 2 and event["target"] == 0
    assert event["step"] == 7 and event["time"] == pytest.approx(0.7)
    assert event["success"] and event["allocated_rbs"] == 5
    assert "x" in event and "y" in event


def test_api_rejects_non_objects_and_fractional_seeds():
    client = create_app().test_client()
    for body in (["urban"], "urban", 42):
        for route in ("/api/session", "/api/step"):
            response = client.post(route, json=body)
            assert response.status_code == 400
            assert "JSON object" in response.get_json()["error"]
    for value in (True, 1.5):
        assert client.post("/api/session", json={"seed": value}).status_code == 400
    default = client.post("/api/session", json={"scenario": "urban"}).get_json()
    assert default["snapshot"]["seed"] == 21
    assert default["snapshot"]["controller_kind"] == "care"


@pytest.mark.skipif(not care_available(), reason="trained CARE artifacts are not present")
@pytest.mark.parametrize("scenario", ["hex_cluster", "benchmark"])
def test_care_session_matches_paper_mechanics_and_preserves_capacity(scenario):
    client = create_app().test_client()
    created = client.post(
        "/api/session", json={"scenario": scenario, "seed": 9000, "controller": "care"}
    )
    assert created.status_code == 200
    token = created.get_json()["session"]
    decisions = 0
    for _ in range(20):
        snapshot = client.post("/api/step", json={"session": token, "count": 3}).get_json()
        assert snapshot["controller_kind"] == "care"
        for name in ("normal", "proposed"):
            network = snapshot[name]
            # Matched mobility: both networks place every device identically.
            for cell in network["cells"]:
                assert cell["allocated"] + cell["reserved"] + cell["free"] == cell["capacity"]
        assert [u["x"] for u in snapshot["normal"]["ues"]] == [
            u["x"] for u in snapshot["proposed"]["ues"]
        ]
        decision = snapshot["proposed"]["decision"]
        if decision:
            decisions += 1
            values = {item["action"]: item["value"] for item in decision["distribution"]}
            assert all(math.isfinite(v) for v in values.values())
            # The shown decision is the greedy action of the Q-ensemble.
            assert values[decision["action"]] == max(values.values())
            assert decision["committed"] <= decision["candidates"]
    assert decisions > 0
    session = ComparisonSession(9000, scenario, "care")
    for env in (session.normal, session.proposed):
        for key, value in CARE_MECHANICS.items():
            assert getattr(env.cfg, key) == value
    assert session.proposed.confidence_model is not None
    assert session.normal.confidence_model is None
    meta = client.get("/api/meta").get_json()
    assert meta["care_available"] is True
