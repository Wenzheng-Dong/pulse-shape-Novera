"""Step 10 — pulse-energy proxy and the no-free-lunch ansatz tradeoff.

Adds the pulse-energy leakage proxy E = integral Omega^2 dt and checks:
1. it is analytically correct (constant-amplitude circle: E = kappa^2 * T);
2. the ansatz-quality tradeoff -- a more dephasing-robust ansatz (rcp_lemniscate)
   has lower dephasing cost but HIGHER pulse energy than the open circle_arc, i.e.
   robustness is bought with control cost (no free lunch), consistently across
   proxies.
"""

import numpy as np

import pulse_shape_novera  # noqa: F401 -- applies the qurveros patch
from pulse_shape_novera import (
    make_circle_arc_spacecurve, make_rcp_spacecurve,
    pulse_energy_loss, gate_fidelity, rotation,
)
from qurveros import losses


def _energy(sc):
    sc.evaluate_frenet_dict()
    return float(pulse_energy_loss(sc.frenet_dict))


def _dephasing(sc):
    sc.evaluate_frenet_dict()
    return float(np.sum(losses.curve_zero_area_loss(sc.frenet_dict)))


def test_pulse_energy_matches_analytic_circle():
    """circle_arc(pi): constant Omega=pi over unit time -> E = pi^2."""
    e = _energy(make_circle_arc_spacecurve(np.pi, duration=1.0))
    assert e == np.pi ** 2 or abs(e - np.pi ** 2) < 1e-3, e


def test_robustness_costs_energy_no_free_lunch():
    """The dephasing-robust good ansatz beats the open one on dephasing but pays
    more pulse energy -- a genuine tradeoff, not a free improvement."""
    rcp = make_rcp_spacecurve("rcp_lemniscate", np.pi)
    circle = make_circle_arc_spacecurve(np.pi)

    # both implement X
    assert gate_fidelity(rcp, rotation("x", np.pi))["fidelity"] > 1 - 1e-5
    assert gate_fidelity(circle, rotation("x", np.pi))["fidelity"] > 1 - 1e-5

    # good ansatz: far better dephasing ...
    assert _dephasing(rcp) < 1e-2 * _dephasing(circle)
    # ... but higher pulse energy (the cost of robustness)
    assert _energy(rcp) > _energy(circle)
