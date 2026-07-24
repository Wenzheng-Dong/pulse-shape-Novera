"""Step 3 — qurveros optimization runs end-to-end, and the scale-gauge lesson.

Validates two things on the frequency-circle demo vehicle:
1. the qurveros optimize loop actually runs and moves the parameter;
2. minimizing gate time alone collapses ``omega -> 0`` (trivial zero control),
   while a ``-log10(omega)`` anchor stabilizes it at the analytic balance
   ``omega* = 1 / (2*pi*ln 10)`` (see _plan.md Q1).
"""

import numpy as np
import pytest

from pulse_shape_novera.optimize import optimize_gate_time


def test_optimization_runs_and_moves_param():
    """The optimize loop records a full history and actually updates omega."""
    res = optimize_gate_time(init_omega=1.0, anchor_weight=0.0, max_iter=200)
    assert len(res["omega_history"]) == 201          # max_iter + initial
    assert res["omega_history"][0] == pytest.approx(1.0)
    assert abs(res["omega_final"] - 1.0) > 0.1        # it moved


def test_unanchored_collapses_to_zero_control():
    """Minimizing gate time alone drives omega (and the pulse) to ~0."""
    res = optimize_gate_time(init_omega=1.0, anchor_weight=0.0, max_iter=400)
    assert abs(res["omega_final"]) < 0.1, res["omega_final"]
    assert res["tg_final"] < 0.1 * (2 * np.pi)        # gate time collapsed


def test_anchor_prevents_collapse_at_analytic_balance():
    """The -log10 anchor stops the collapse at omega* = 1/(2*pi*ln10)."""
    res = optimize_gate_time(init_omega=1.0, anchor_weight=1.0, max_iter=400)
    omega_star = 1.0 / (2.0 * np.pi * np.log(10.0))    # ~0.06911
    assert res["omega_final"] == pytest.approx(omega_star, abs=1e-3)
    assert abs(res["omega_final"]) > 0.03              # did NOT collapse
