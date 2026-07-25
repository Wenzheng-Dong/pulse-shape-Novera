"""Step 6b — RCP z-error families (rcp_lemniscate, rcp_petal) via order=1.

These are the *good* ansaetze for a robust gate (see _plan.md claim 2): closed,
dephasing-robust, and natively implementing the target gate angle. We wrap them
exactly from curvecontroltoolbox's stored Omega-sine-series coefficients (Bezier
reconstruction is unstable for them). Checks: they implement X(pi) at unit
fidelity, and rcp_lemniscate (zero-area) already has near-zero 2nd-order
dephasing cost -- i.e. it is already ~optimal for static dephasing.
"""

import numpy as np
import pytest

import pulse_shape_novera  # noqa: F401 -- applies the qurveros patch
from pulse_shape_novera import make_rcp_spacecurve, gate_fidelity, rotation
from qurveros import losses


@pytest.mark.parametrize("family", ["rcp_lemniscate", "rcp_petal"])
def test_rcp_implements_x_gate(family):
    """rcp_lemniscate / rcp_petal at gate angle pi implement the X gate."""
    sc = make_rcp_spacecurve(family, np.pi)
    assert gate_fidelity(sc, rotation("x", np.pi))["fidelity"] > 1.0 - 1e-5


def test_rcp_lemniscate_is_already_dephasing_robust():
    """The zero-area rcp_lemniscate already has ~zero 2nd-order dephasing cost
    (curve zero-area), i.e. it is essentially optimal for static dephasing."""
    sc = make_rcp_spacecurve("rcp_lemniscate", np.pi)
    sc.evaluate_frenet_dict()
    curve_area = float(np.sum(losses.curve_zero_area_loss(sc.frenet_dict)))
    assert curve_area < 1e-3, curve_area


@pytest.mark.parametrize("angle", [np.pi / 2, np.pi / 4])
def test_rcp_other_angles_implement_negative_x_rotation(angle):
    """The pi/2 and pi/4 members implement R_x(-angle) in qurveros' convention.

    curvecontroltoolbox and qurveros use OPPOSITE rotation-sign conventions for
    the SCQC gate (invisible at pi, since R_x(pi) = R_x(-pi) up to phase): cct
    reports +angle, qurveros gives -angle for the same physical curve. So these
    members are genuine pi/2 and pi/4 X-rotations (magnitude matches gate_angle),
    just the -x direction here. This locks in that understanding.
    """
    sc = make_rcp_spacecurve("rcp_lemniscate", angle)
    assert gate_fidelity(sc, rotation("x", -angle))["fidelity"] > 1.0 - 1e-4   # R_x(-angle)
    assert gate_fidelity(sc, rotation("x", +angle))["fidelity"] < 0.9          # NOT R_x(+angle)
