"""BARQ: the automated, no-prior-ansatz baseline optimizer.

BARQ (Bezier Ansatz for Robust Quantum control) parameterizes the space curve by
Bezier control points and, via point gate-fixing (PGF), constructs them so the
TARGET GATE is satisfied exactly for any parameter values. Optimization therefore
never touches the gate (fidelity stays 1) and only shapes robustness -- there is
no gate-vs-robustness tradeoff. This is our "no prior ansatz" comparison arm for
the ansatz-quality study (the other arm is Path-A curve_family ansaetze).

The scale (``norm_value``) is frozen during optimization -- the scale-gauge
lesson from step 3 (otherwise the curve could collapse).
"""

from __future__ import annotations

import jax
import optax
import qutip
from qurveros import losses, barqtools
from qurveros.optspacecurve import BarqCurve
from qurveros.qubit_bench import quantumtools, simulator


@jax.jit
def xgate_pgf_mod(pgf_params, input_points):
    """PGF modifier that ties the free gate-fixing knobs to a single ``norm_value``
    for an X-gate boundary condition (from qurveros' Xgate BARQ example)."""
    p = pgf_params.copy()
    n = p["norm_value"]
    for key in ("left_tangent_fix", "left_tangent_aux", "left_binormal_fix",
                "right_binormal_fix", "right_tangent_aux", "right_tangent_fix"):
        p[key] = n
    p["left_binormal_aux"] = p["right_binormal_aux"]
    return p


def make_barq_xgate(*, n_free_points: int = 10, seed: int = 4531469,
                    norm_value: float = 0.25) -> BarqCurve:
    """Build and initialize a BARQ curve whose target gate is X (sigma_x)."""
    adj_target = quantumtools.calculate_adj_rep(qutip.sigmax())
    barq = BarqCurve(adj_target=adj_target, n_free_points=n_free_points, pgf_mod=xgate_pgf_mod)
    init_pgf = barqtools.get_default_pgf_params_dict()
    init_pgf["norm_value"] = norm_value
    barq.initialize_parameters(seed=seed, init_pgf_params=init_pgf)
    return barq


def optimize_barq_robustness(barq: BarqCurve, *, max_iter: int = 1500, lr: float = 1e-3,
                             amp_weight: float = 1e-2) -> BarqCurve:
    """Optimize robustness (tantrix zero-area + max-amp) with the scale frozen."""
    labels = jax.tree.map(lambda _: True, barq.params)
    labels["pgf_params"]["norm_value"] = False          # freeze scale (gauge)
    optimizer = optax.multi_transform(
        {True: optax.adam(lr), False: optax.set_to_zero()}, param_labels=labels)
    barq.prepare_optimization_loss(
        [losses.tantrix_zero_area_loss, 1.0], [losses.max_amp_loss, amp_weight])
    barq.optimize(optimizer, max_iter=max_iter)
    return barq


def barq_gate_fidelity(barq: BarqCurve) -> float:
    """Average gate fidelity of the BARQ curve to its X-gate target.

    Uses BARQ's own (TTC) control mode, not the resonant 'XY' one.
    """
    barq.evaluate_frenet_dict()
    barq.evaluate_control_dict()
    sim = simulator.simulate_control_dict(barq.get_control_dict(), qutip.sigmax())
    return float(sim["avg_gate_fidelity"])
