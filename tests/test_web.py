from web.server import ComparisonSession, create_app
from proran.config import SimConfig
from proran.policies import SavedPPOPolicy, no_reservation
from proran.simulator import RANHandoverEnv
from web.server import MODEL


def test_health_and_static_index():
    client = create_app().test_client()
    assert client.get("/health").get_json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200
    assert b"Six-device handover lab" in page.data
    assert b"One shared world" in page.data
    assert b"Pause" in page.data
    assert b"zoomIn" in page.data
    assert b"zoomOut" in page.data
    assert b"stepBtn" not in page.data
    assert b"resetBtn" not in page.data
    assert client.get("/paper.tex").status_code == 200
    assert client.get("/references.bib").status_code == 200
    meta = client.get("/api/meta").get_json()
    assert meta["learning"]["imitation"]
    assert meta["learning"]["ppo"]
    assert len(meta["paired"]) == 8


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
                    ue["cell"] == cell["id"] for ue in network["ues"])
                assert sum(row["demand"] for row in cell["slices"]) == sum(
                    ue["demand"] for ue in network["ues"] if ue["cell"] == cell["id"])
            assert {ue["entity_type"] for ue in network["ues"]} == {"car", "drone", "robot"}
            assert all(0 <= ue["signal_quality"] <= 1 for ue in network["ues"])
            assert network["metrics"]["successes"] <= network["metrics"]["handovers"]
        decision = proposed["decision"]
        if decision and decision["ue"] is not None:
            assert decision["distribution"]
            assert abs(sum(item["probability"] for item in decision["distribution"]) - 1.0) < 1e-9


def test_unknown_session_is_reported():
    client = create_app().test_client()
    response = client.post("/api/step", json={"session": "missing"})
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_teaching_scenarios_are_six_device_capacity_cases():
    assert ComparisonSession(21, "urban").normal.cfg.cell_capacity == 18
    assert ComparisonSession(118, "crossing").normal.cfg.cell_capacity == 20
    assert all(ComparisonSession(seed, scenario).normal.cfg.n_ues == 6
               for scenario, seed in (("urban", 21), ("crossing", 118)))


def test_urban_web_trace_matches_offline_policies():
    session = ComparisonSession(21, "urban")
    session.step(12)
    session.step(12)
    session.step(12)
    session.step(4)
    cfg = SimConfig(seed=21, n_ues=6, cell_capacity=18, episode_steps=40)
    baseline = RANHandoverEnv(cfg).run(no_reservation, steps=40, seed=21)
    learned = RANHandoverEnv(cfg).run(SavedPPOPolicy.load(str(MODEL)), steps=40, seed=21)
    assert session.normal.metrics.handovers == baseline.handovers
    assert session.normal.metrics.successes == baseline.successes
    assert session.proposed.metrics.successes == learned.successes
    assert session.proposed.metrics.occupancy_rb_intervals == learned.occupancy_rb_intervals


def test_curated_trace_shows_real_proactive_handover_gain():
    session = ComparisonSession(21, "urban")
    reservations = []
    for step in range(1, 8):
        snap = session.step()
        reservations.extend(event for event in snap["proposed"]["events"] if event.startswith("reserve "))
        if step == 4:
            target = snap["proposed"]["cells"][0]
            assert target["reserved"] == 8
            assert target["allocated"] + target["reserved"] == target["capacity"]
            assert any(c["ue"] == 1 and c["target"] == 0 and c["active"]
                       for c in snap["proposed"]["contracts"])
    assert any("UE1" in event and "C0" in event for event in reservations)
    assert any("HO UE1 C2->C0 SUCCESS (7/7RB)" in event for event in snap["proposed"]["events"])
    assert any("HO UE1 C2->C0 FAIL (3/7RB)" in event for event in snap["normal"]["events"])
    assert snap["proposed"]["metrics"]["qos_successes"] == 1
    assert snap["normal"]["metrics"]["qos_successes"] == 0


def test_six_device_paired_trace_and_capacity_invariants_across_seeds():
    for scenario, seed in (("urban", 21), ("urban", 65), ("crossing", 118), ("crossing", 99)):
        session = ComparisonSession(seed, scenario)
        for _ in range(75):
            snap = session.step()
            for left, right in zip(snap["normal"]["ues"], snap["proposed"]["ues"], strict=True):
                assert (left["x"], left["y"], left["cell"], left["demand"]) == (
                    right["x"], right["y"], right["cell"], right["demand"])
            assert snap["normal"]["metrics"]["handovers"] == snap["proposed"]["metrics"]["handovers"]
            for network in (snap["normal"], snap["proposed"]):
                assert len(network["ues"]) == 6
                for cell in network["cells"]:
                    assert cell["allocated"] + cell["reserved"] + cell["free"] == cell["capacity"]
                    assert cell["allocated"] >= 0 and cell["reserved"] >= 0


def test_second_six_device_trace_has_measured_early_gain():
    session = ComparisonSession(118, "crossing")
    session.step(2)
    result = session.step()
    assert any("HO UE2 C2->C0 FAIL (4/6RB)" in event
               for event in result["normal"]["events"])
    assert any("HO UE2 C2->C0 SUCCESS (6/6RB)" in event
               for event in result["proposed"]["events"])
    assert result["normal"]["metrics"]["successes"] == 0
    assert result["proposed"]["metrics"]["successes"] == 1


def test_web_refuses_to_claim_ppo_when_saved_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("web.server.MODEL", tmp_path / "missing_model.npz")
    client = create_app().test_client()
    response = client.post("/api/session", json={"scenario": "urban", "seed": 21})
    assert response.status_code == 503
    assert "Saved PPO model is missing" in response.get_json()["error"]


def test_bad_inputs_return_client_errors_and_security_headers():
    client = create_app().test_client()
    response = client.post("/api/session", json={"scenario": "unknown", "seed": -1})
    assert response.status_code == 400
    response = client.post("/api/session", json={"scenario": "urban", "seed": "not-a-number"})
    assert response.status_code == 400
    page = client.get("/")
    assert page.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in page.headers["Content-Security-Policy"]
