"""Step 6 — Path A via the tangent (order=1) for integral/profile families.

The planar arcs triangle_pulse_arc and gaussian_arc are defined by a curvature
envelope (not a closed-form position). We express their unit tangent
T(x) = [0, sin(theta), cos(theta)] with theta the analytic integral of the
(scaled) envelope, and hand it to qurveros with order=1 -- validating the other
half of Path A. Checks: total turning integral(kappa) == turning_angle (the gate
rotation), the curvature peak agrees with curvecontroltoolbox, and the endpoint
curvature reproduces each family's signature (triangle ramps from 0; the
truncated Gaussian is nonzero at the ends).
"""

import numpy as np
import pytest

import pulse_shape_novera  # noqa: F401 -- applies the qurveros patch
from pulse_shape_novera import (
    make_triangle_pulse_spacecurve,
    make_gaussian_arc_spacecurve,
)
from qurveros import frametools

TURNING_ANGLES = [np.pi, np.pi / 2, 2.0 * np.pi / 3]


def _curvature(sc):
    sc.evaluate_frenet_dict()
    return np.asarray(sc.frenet_dict["curvature"]), sc.frenet_dict


@pytest.mark.parametrize("maker", [make_triangle_pulse_spacecurve, make_gaussian_arc_spacecurve])
@pytest.mark.parametrize("turning_angle", TURNING_ANGLES)
def test_total_turning_equals_turning_angle(maker, turning_angle):
    """integral(kappa ds) == turning_angle: the arc realizes X(turning_angle)."""
    sc = maker(turning_angle)
    _, fd = _curvature(sc)
    total = float(frametools.calculate_total_curvature(fd))
    assert total == pytest.approx(turning_angle, rel=1e-4)


def test_triangle_endpoint_curvature_is_zero():
    """The triangle-pulse arc ramps curvature from 0 at both ends."""
    kappa, _ = _curvature(make_triangle_pulse_spacecurve(np.pi))
    assert abs(kappa[0]) < 1e-6 and abs(kappa[-1]) < 1e-6


def test_gaussian_endpoint_curvature_is_nonzero():
    """The truncated-Gaussian arc has small but nonzero endpoint curvature."""
    kappa, _ = _curvature(make_gaussian_arc_spacecurve(np.pi))
    assert kappa[0] > 1e-3


@pytest.mark.parametrize("maker,name", [
    (make_triangle_pulse_spacecurve, "triangle_pulse_arc"),
    (make_gaussian_arc_spacecurve, "gaussian_arc"),
])
def test_peak_curvature_matches_cct(maker, name):
    """qurveros' peak curvature agrees with curvecontroltoolbox's arc."""
    curve_families = pytest.importorskip("curvecontroltoolbox.curve_families")
    inverse = pytest.importorskip("curvecontroltoolbox.inverse")

    kappa, _ = _curvature(maker(np.pi))
    nc = curve_families.make_curve(name, turning_angle=np.pi)
    scc = inverse.make_space_curve(nc.parameter_values, nc.positions)
    v, a = scc.velocities, scc.accelerations
    kc = np.linalg.norm(np.cross(v, a), axis=1) / np.linalg.norm(v, axis=1) ** 3
    assert kappa.max() == pytest.approx(kc[2:-2].max(), rel=2e-2)
