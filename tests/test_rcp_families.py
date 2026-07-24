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

# Note: only the pi (X-gate) member is used as the robust-X ansatz and validated
# here. Other stored gate angles (pi/2, pi/4) also wrap, but the identification
# of their (tangent) gate_angle with a specific qubit rotation is not asserted --
# out of scope for the robust-X study.
