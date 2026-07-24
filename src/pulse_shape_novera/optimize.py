"""Optimization mechanics + the scale-gauge lesson (see _plan.md Q1).

This module is a **demonstration vehicle**, not a gate ansatz: it uses the
frequency-parametrized circle ``[cos(w x), sin(w x), 0]`` on ``[0, 2*pi]`` whose
gate time is ``Tg = 2*pi*|w|``. Minimizing gate time alone drives ``w -> 0``:
the curve shrinks and the control field vanishes (the trivial "do nothing"
solution _plan.md Q1 warns about). Anchoring the scale with ``-log10(w)``
prevents the collapse. We use gate time (not curvature/max-amp) because
differentiating qurveros' Frenet frame through a *planar* curve's torsion (0/0)
produces NaN gradients — a real gotcha; ``total_time_loss`` only touches speed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from qurveros import losses
from qurveros.optspacecurve import OptimizableSpaceCurve


def frequency_circle_curve(x, omega):
    """Circle traversed at frequency ``omega`` over ``[0, 2*pi]`` (demo vehicle).

    Arc length (hence gate time) is ``2*pi*|omega|``, so ``omega`` acts as the
    scale/strength knob whose collapse to zero we study.
    """
    return [jnp.cos(omega * x), jnp.sin(omega * x), 0.0 * x]


def scale_anchor_loss(frenet_dict):
    """``-log10(omega)`` — penalizes the trivial ``omega -> 0`` (zero-control) sink."""
    return -jnp.log10(frenet_dict["params"][0])


def optimize_gate_time(init_omega: float = 1.0, *, anchor_weight: float = 0.0,
                       max_iter: int = 400) -> dict:
    """Minimize gate time on the frequency-circle, optionally scale-anchored.

    Parameters
    ----------
    init_omega : float
        Initial frequency (default 1.0).
    anchor_weight : float
        Weight of ``scale_anchor_loss``. ``0`` (default) is unanchored and
        collapses to ``omega ~ 0``; a positive weight stabilizes ``omega``.
    max_iter : int
        Gradient-descent iterations.

    Returns
    -------
    dict
        ``{"omega_history": ndarray, "omega_final": float, "tg_final": float}``.
    """
    sc = OptimizableSpaceCurve(
        curve=frequency_circle_curve, order=0,
        interval=[0.0, 2.0 * np.pi], params=init_omega,
    )
    sc.initialize_parameters(init_omega)

    loss_terms = [[losses.total_time_loss, 1.0]]
    if anchor_weight > 0.0:
        loss_terms.append([scale_anchor_loss, anchor_weight])
    sc.prepare_optimization_loss(*loss_terms)

    sc.optimize(max_iter=max_iter)

    history = np.array([float(np.asarray(p).ravel()[0]) for p in sc.get_params_history()])
    omega_final = float(history[-1])
    return {
        "omega_history": history,
        "omega_final": omega_final,
        "tg_final": 2.0 * np.pi * abs(omega_final),
    }
