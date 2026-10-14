import math

import numpy as np
import pytest

from proran.config import SimConfig
from proran.geometry import HexDomain
from web.server import HEX_RADIUS, SCENARIOS, ComparisonSession


def test_seven_hexagons_have_centered_sites_equal_area_and_shared_edges():
    domain = HexDomain(np.array(SCENARIOS["hex_cluster"]["centers"]), HEX_RADIUS)
    assert len(domain.boundary) == 18
    expected_area = 3 * math.sqrt(3) * HEX_RADIUS**2 / 2
    for center, polygon in zip(domain.centers, domain.polygons, strict=True):
        assert np.allclose(polygon.mean(axis=0), center, atol=1e-12)
        assert np.allclose(np.linalg.norm(polygon - center, axis=1), HEX_RADIUS)
        area = (
            abs(
                sum(
                    a[0] * b[1] - a[1] * b[0]
                    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
                )
            )
            / 2
        )
        assert area == pytest.approx(expected_area)


def test_exposed_edge_reflects_remaining_motion_and_preserves_speed():
    domain = HexDomain(np.array([[50.0, 50.0]]), HEX_RADIUS)
    normal = domain.normals[0]
    start = domain.centers[0] + (domain.apothem - 1) * normal
    end, velocity = domain.move(start, 2 * normal)
    assert np.allclose(end, start, atol=1e-8)
    assert np.allclose(velocity, -2 * normal)
    assert domain.contains(end)


def test_internal_cell_edge_is_a_crossing_without_reflection():
    domain = HexDomain(np.array(SCENARIOS["hex_cluster"]["centers"]), HEX_RADIUS)
    start = np.array([50.0, 50.0 - domain.apothem + 1.0])
    velocity = np.array([0.0, -4.0])
    end, result_velocity = domain.move(start, velocity)
    assert np.array_equal(end, start + velocity)
    assert np.array_equal(result_velocity, velocity)
    assert np.argmin(np.linalg.norm(domain.centers - end, axis=1)) == 1


def test_motion_cannot_cut_across_an_exposed_concave_notch():
    domain = HexDomain(np.array(SCENARIOS["hex_cluster"]["centers"]), HEX_RADIUS)
    corner = np.array([50.0 + HEX_RADIUS, 50.0 - math.sqrt(3) * HEX_RADIUS])
    start, endpoint = corner + [-1.0, -1.0], corner + [2.0, 1.0]
    velocity = endpoint - start
    assert domain.contains(start) and domain.contains(endpoint)
    assert not domain.contains(start + 0.25 * velocity)
    end, result_velocity = domain.move(start, velocity)
    assert domain.contains(end)
    assert not np.allclose(end, endpoint)
    assert np.linalg.norm(result_velocity) == pytest.approx(np.linalg.norm(velocity))


def test_cluster_mobility_stays_inside_its_true_serving_hexagons_and_is_paired():
    for seed in (42, 7, 20001):
        session = ComparisonSession(seed, "hex_cluster")
        domain = session.proposed.domain
        for _ in range(160):
            snap = session.step()
            for left, right in zip(snap["normal"]["ues"], snap["proposed"]["ues"], strict=True):
                assert (left["x"], left["y"], left["cell"]) == (
                    right["x"],
                    right["y"],
                    right["cell"],
                )
                pos = np.array([right["x"], right["y"]])
                assert domain.contains(pos)
                delta = pos - domain.centers[right["cell"]]
                assert np.all(np.abs(delta @ domain.normals.T) <= domain.apothem + 1e-7)


@pytest.mark.parametrize(
    "changes",
    [
        {"hex_radius": 17},
        {"hex_radius": -1, "centers": ((50.0, 50.0),)},
        {"hex_radius": 17, "centers": ((5.0, 5.0),)},
        {"hex_radius": 17, "centers": ((50.0, 50.0), (70.0, 50.0))},
    ],
)
def test_invalid_hex_domains_are_rejected(changes):
    with pytest.raises(ValueError):
        SimConfig(**changes)
