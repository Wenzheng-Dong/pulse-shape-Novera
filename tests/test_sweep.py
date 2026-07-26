"""Step 11d -- invariants of the weight-sweep machinery (sweep.py).

The normalization contract (see ``results/README.md``) has properties we can
check cheaply and that guard against a mis-wired reference or scaling:

1. Every reference ``C_i^ref`` on the naive pulse is finite and > 0, and matches
   the analytic constant-amplitude pi-pulse (energy = pi^2, max_amp = pi, L = 1).
2. The naive pulse *is* the bad ansatz, so the exact ``circle_arc(pi)`` evaluates
   to ``Chat_i == 1`` on every term -- the baseline sits at 1 by construction,
   which is what makes the good/bad comparison fair.
3. The good ansatz (rcp_lemniscate) is far more dephasing-robust than the naive
   pulse (``Chat_dephasing << 1``) but pays for it in energy (``Chat_energy > 1``)
   -- the no-free-lunch signature, and a check that the scale-invariant forms
   behave.
"""

import numpy as np
import pytest

import pulse_shape_novera  # noqa: F401 -- applies qurveros patch
from pulse_shape_novera import make_circle_arc_spacecurve, make_rcp_spacecurve
from pulse_shape_novera import sweep

ANGLE = np.pi


@pytest.fixture(scope="module")
def reference():
    return sweep.compute_references(make_circle_arc_spacecurve(ANGLE))


def test_references_finite_positive(reference):
    for name, val in reference["refs"].items():
        assert np.isfinite(val) and val > 0.0, f"{name} ref must be finite > 0"


def test_references_match_analytic_pi_pulse(reference):
    # Constant-amplitude pi-pulse over L = 1: Omega = pi.
    # energy = int Omega^2 dt = pi^2 ; max_amp = L * Omega_max = pi ; L = 1.
    refs = reference["refs"]
    assert reference["tg_ref"] == pytest.approx(1.0, abs=1e-6)
    assert refs["max_amp"] == pytest.approx(np.pi, rel=1e-4)
    assert refs["energy"] == pytest.approx(np.pi ** 2, rel=1e-3)


def test_bad_ansatz_sits_at_baseline(reference):
    # The naive pulse IS the bad ansatz -> Chat_i == 1 on every term.
    chat = sweep.evaluate_chat(make_circle_arc_spacecurve(ANGLE), reference)
    for name, val in chat.items():
        assert val == pytest.approx(1.0, rel=1e-3), f"{name} Chat should be 1, got {val}"


def test_good_ansatz_no_free_lunch(reference):
    # rcp good ansatz: far more dephasing-robust than naive, but costlier in energy.
    chat = sweep.evaluate_chat(make_rcp_spacecurve("rcp_lemniscate", ANGLE), reference)
    assert chat["closure"] < 1e-3, f"good closure Chat should be << 1, got {chat['closure']}"
    assert chat["curve_area"] < 1e-2, f"good curve_area Chat should be << 1, got {chat['curve_area']}"
    assert chat["energy"] > 1.0, f"good energy Chat should exceed naive, got {chat['energy']}"


def test_seeded_barq_keeps_gate_and_robustness(reference):
    # The step-11e linchpin: seeding the rcp good ansatz into BARQ must keep the
    # gate exact (TTC fidelity ~ 1, via PGF) AND preserve its dephasing robustness.
    from pulse_shape_novera.barq import barq_gate_fidelity
    free = sweep.seed_free_points_from_curve(make_rcp_spacecurve("rcp_lemniscate", ANGLE))
    barq = sweep.make_seeded_barq(free)
    assert barq_gate_fidelity(barq) > 0.99, "seeded BARQ gate must stay exact (PGF)"
    chat = sweep.evaluate_chat(barq, reference)
    assert chat["curve_area"] < 1e-2, \
        f"seeded good curve should stay dephasing-robust, got {chat['curve_area']}"
