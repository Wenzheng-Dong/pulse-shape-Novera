"""Step 2 — end-to-end gate fidelity for the circle_arc pulse.

Closes the loop from step 1 (geometry -> control fields) to the actual quantum
gate: propagate the circle_arc control fields through qurveros' qutip simulator
and check the implemented unitary IS the intended rotation.

Physics: a planar circular arc has constant curvature and zero torsion, so the
SCQC pulse is a constant-amplitude resonant drive with phase Phi == 0 (about x)
and area ``int Omega dt == turning_angle``. Hence circle_arc(Theta) must realize
``R_x(Theta)`` (e.g. Theta = pi -> X gate). We verify this three ways:
qurveros simulation, the analytic rotation, and curvecontroltoolbox's inverse map.
"""

import numpy as np
import pytest

from pulse_shape_novera import gate_fidelity, make_circle_arc_spacecurve, rotation


TURNING_ANGLES = [np.pi, np.pi / 2, 2.0 * np.pi / 3]


@pytest.mark.parametrize("turning_angle", TURNING_ANGLES)
def test_circle_arc_implements_x_rotation(turning_angle):
    """circle_arc(Theta) implements R_x(Theta) at unit fidelity."""
    sc = make_circle_arc_spacecurve(turning_angle, duration=1.0)
    result = gate_fidelity(sc, rotation("x", turning_angle))

    assert result["fidelity"] > 1.0 - 1e-6, \
        f"fidelity to R_x({turning_angle}) = {result['fidelity']}"


def test_half_circle_is_x_gate_unitary():
    """The Theta = pi unitary equals X up to a global phase (== R_x(pi) = -iX)."""
    sc = make_circle_arc_spacecurve(np.pi, duration=1.0)
    u = np.asarray(gate_fidelity(sc, rotation("x", np.pi))["u_final"])

    expected = np.array([[0.0, -1j], [-1j, 0.0]])  # -i * sigma_x
    assert np.allclose(u, expected, atol=1e-6), f"u_final=\n{np.round(u, 4)}"


def test_orthogonal_gate_fidelity_floor():
    """Sanity on the metric: X vs an orthogonal gate hits the 1/3 avg-fidelity floor."""
    sc = make_circle_arc_spacecurve(np.pi, duration=1.0)
    assert gate_fidelity(sc, rotation("y", np.pi))["fidelity"] == pytest.approx(1 / 3, abs=1e-6)


@pytest.mark.parametrize("turning_angle", TURNING_ANGLES)
def test_cross_library_intended_gate(turning_angle):
    """curvecontroltoolbox's inverse map independently reports axis=x, angle=Theta."""
    curve_families = pytest.importorskip("curvecontroltoolbox.curve_families")
    inverse = pytest.importorskip("curvecontroltoolbox.inverse")

    nc = curve_families.make_curve("circle_arc", turning_angle=turning_angle)
    res = inverse.curve_to_control(inverse.make_space_curve(nc.parameter_values, nc.positions))

    assert np.allclose(np.abs(res.gate.rotation_axis), [1.0, 0.0, 0.0], atol=1e-3)
    assert res.gate.rotation_angle == pytest.approx(turning_angle, abs=1e-3)
