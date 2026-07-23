"""Step 1 — validate the Path-A adapter for the ``circle_arc`` family.

A circular arc is the analytic "hello world" of SCQC: constant curvature means a
constant-amplitude pulse with ``Omega == kappa == turning_angle / duration``. We
check that

1. qurveros (autodiff) recovers that constant curvature and builds control fields;
2. the constant equals the analytic value for several turning angles;
3. curvecontroltoolbox's independent geometry pipeline agrees on the same curve.

Cross-library note: we intentionally cross-check via ``inverse.make_space_curve``
(which uses ``np.gradient``) and compute curvature by hand, rather than calling
``inverse.curve_to_control``. The latter hits ``np.trapz``, removed in NumPy 2.x,
and this env runs numpy 2.x (see ``_dev_logs/step01_pathA_circle.md``).
"""

import numpy as np
import pytest

from pulse_shape_novera import make_circle_arc_spacecurve


TURNING_ANGLES = [np.pi, np.pi / 2, 2.0 * np.pi / 3]


def _qurveros_curvature(turning_angle, duration=1.0):
    sc = make_circle_arc_spacecurve(turning_angle, duration)
    sc.evaluate_frenet_dict()
    return sc, np.asarray(sc.frenet_dict["curvature"])


@pytest.mark.parametrize("turning_angle", TURNING_ANGLES)
def test_qurveros_curvature_matches_analytic(turning_angle):
    """qurveros autodiff gives a constant curvature == turning_angle / duration."""
    duration = 1.0
    kappa = turning_angle / duration
    _, curvature = _qurveros_curvature(turning_angle, duration)

    assert np.allclose(curvature, kappa, atol=1e-4), \
        f"curvature not constant == {kappa}: {curvature.min()}..{curvature.max()}"


def test_qurveros_control_fields_extracted():
    """Path A yields SCQC control fields: Omega == curvature, constant for a circle."""
    turning_angle = np.pi
    sc, _ = _qurveros_curvature(turning_angle)
    sc.evaluate_control_dict("XY")
    cd = sc.control_dict

    for key in ("times", "omega", "phi"):
        assert key in cd, f"missing control field {key!r}"

    omega = np.asarray(cd["omega"])
    assert np.allclose(omega, turning_angle, atol=1e-4)


@pytest.mark.parametrize("turning_angle", TURNING_ANGLES)
def test_cross_library_geometry_agrees(turning_angle):
    """curvecontroltoolbox's finite-diff geometry agrees with qurveros and analytic.

    Avoids ``curve_to_control`` (NumPy-2.x ``np.trapz`` bug); computes curvature
    directly from the sampled curve's velocity/acceleration.
    """
    curve_families = pytest.importorskip("curvecontroltoolbox.curve_families")
    inverse = pytest.importorskip("curvecontroltoolbox.inverse")

    kappa = turning_angle
    nc = curve_families.make_curve("circle_arc", turning_angle=turning_angle)
    scc = inverse.make_space_curve(nc.parameter_values, nc.positions)

    v, a = scc.velocities, scc.accelerations
    cct_curvature = np.linalg.norm(np.cross(v, a), axis=1) / np.linalg.norm(v, axis=1) ** 3

    # Interior points only: one-sided finite differences at the ends are noisier.
    assert np.allclose(cct_curvature[2:-2], kappa, atol=1e-2)

    _, qurveros_curvature = _qurveros_curvature(turning_angle)
    assert np.allclose(qurveros_curvature.mean(), cct_curvature[2:-2].mean(), atol=1e-3)
