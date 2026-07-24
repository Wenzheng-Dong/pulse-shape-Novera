"""Step 8 — BARQ baseline: gate is hard-fixed; optimization only buys robustness.

BARQ's defining property is that point gate-fixing makes the target gate exact
for any parameters, so gate fidelity is 1 at initialization and stays 1 through
optimization, which only reduces the robustness cost (tantrix zero-area =
first-order amplitude-error robustness). This is the no-prior-ansatz arm for the
step-9 ansatz-quality comparison.
"""

import numpy as np
import pytest

import pulse_shape_novera  # noqa: F401 -- applies the qurveros patch
from pulse_shape_novera import make_barq_xgate, optimize_barq_robustness, barq_gate_fidelity
from qurveros import losses


def _tantrix_area(barq):
    barq.evaluate_frenet_dict()
    return float(np.sum(np.asarray(losses.tantrix_zero_area_loss(barq.frenet_dict))))


def test_barq_gate_is_hard_fixed_at_init():
    """Without any optimization, the BARQ curve already realizes X at fidelity 1."""
    barq = make_barq_xgate()
    assert barq_gate_fidelity(barq) > 1.0 - 1e-6


def test_barq_optimization_improves_robustness_at_fixed_gate():
    """Optimization keeps fidelity == 1 while driving the tantrix area way down."""
    barq = make_barq_xgate()
    area_init = _tantrix_area(barq)

    optimize_barq_robustness(barq, max_iter=500)

    assert barq_gate_fidelity(barq) > 1.0 - 1e-6          # gate still exact
    assert _tantrix_area(barq) < 0.1 * area_init          # robustness improved a lot
