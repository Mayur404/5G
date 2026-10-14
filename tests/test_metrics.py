import math

from proran.models import RunMetrics


def test_resource_ratios_use_consistent_rb_units():
    metrics = RunMetrics(
        steps=2,
        activated_rbs=7,
        used_reserved_rbs=2,
        occupancy_rb_intervals=4,
        allocated_rb_intervals=10,
        capacity_rb_intervals=20,
    )
    summary = metrics.summary(0.1)
    assert math.isclose(summary["unused_reservation_fraction"], 5 / 7)
    assert math.isclose(summary["reservation_occupancy"], 0.2)
    assert math.isclose(summary["rb_utilization"], 0.5)
