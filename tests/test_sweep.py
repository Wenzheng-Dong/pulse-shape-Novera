"""Step 11d -- invariants of the weight-sweep machinery (sweep.py).

The normalization contract (see ``results/README.md``) has two properties we can
check cheaply and that guard against a mis-wired reference:

1. Every reference ``C_i^ref`` on the naive pulse is finite and > 0 (no
   division-by-zero in ``Chat_i``), and matches the analytic values of the
   constant-amplitude pi-pulse (energy = pi^2, max_amp = pi, T_g = 1).
2. The naive pulse *is* the bad ansatz, so a Bezier-fitted ``circle_arc(pi)``
   evaluates to ``Chat_i ~ 1`` on every term -- the baseline sits at 1 by
   construction, which is exactly what makes the good/bad comparison fair.
"""

import numpy as np
import qutip
import pytest

import pulse_shape_novera  # noqa: F401 -- applies qurveros patch
from pulse_shape_novera import (make_circle_arc_spacecurve,
                                 fit_bezier_control_points)
from pulse_shape_novera import sweep

ANGLE = np.pi


@pytest.fixture(scope="module")
def reference():
    return sweep.compute_references(make_circle_arc_spacecurve(ANGLE))


def test_references_finite_positive(reference):
    refs = reference["refs"]
    for name, val in refs.items():
        assert np.isfinite(val) and val > 0.0, f"{name} ref must be finite > 0"


def test_references_match_analytic_pi_pulse(reference):
    # Constant-amplitude pi-pulse over T_g = 1: Omega = pi.
    # energy = int Omega^2 dt = pi^2 ; max_amp = T_g * Omega_max = pi ; T_g = 1.
    refs = reference["refs"]
    assert reference["tg_ref"] == pytest.approx(1.0, abs=1e-6)
    assert refs["max_amp"] == pytest.approx(np.pi, rel=1e-4)
    assert refs["energy"] == pytest.approx(np.pi ** 2, rel=1e-3)


def test_bad_ansatz_sits_at_baseline(reference):
    # The Bezier-fitted circle_arc IS the naive pulse -> Chat_i ~ 1 everywhere,
    # and it already realizes the gate (infidelity ~ 0) with T_g ~ 1.
    sc = make_circle_arc_spacecurve(ANGLE)
    sc.evaluate_frenet_dict(400)
    W0 = fit_bezier_control_points(np.asarray(sc.frenet_dict["curve"]),
                                   sweep.BEZIER_DEGREE,
                                   param=np.asarray(sc.frenet_dict["x_values"]))
    r = sweep.run_single(W0, {"closure": 1.0}, reference, qutip.sigmax(),
                         n_iter=1, checkpoint_every=1)
    c0 = {k: v[0] for k, v in r["chat_history"].items()}
    for name, chat in c0.items():
        assert chat == pytest.approx(1.0, rel=0.05), f"{name} Chat should be ~1, got {chat}"
    assert r["gate_infidelity"][0] < 1e-6
    assert r["tg"][0] == pytest.approx(1.0, abs=1e-2)
