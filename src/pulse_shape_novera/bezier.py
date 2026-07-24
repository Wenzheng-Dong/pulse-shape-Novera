"""Path B: fit an ansatz curve to Bezier control points and reconstruct it.

This is the bridge to BARQ, whose optimization variable is a cloud of Bezier
control points. To seed BARQ with one of our physically-motivated ansatz curves
we must first express that curve in the Bezier basis -- so the key question is
how faithfully a degree-n Bezier reproduces the ansatz, measured not just on
positions but on the control fields and the resulting gate (curvature is a
second derivative, so it amplifies fit error). See _plan.md Q2.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import comb
from qurveros.spacecurve import SpaceCurve
from qurveros.beziertools import bezier_curve_vec


def bernstein_matrix(u: np.ndarray, degree: int) -> np.ndarray:
    """Return the ``(len(u), degree+1)`` Bernstein basis matrix on ``u in [0, 1]``."""
    u = np.asarray(u, dtype=float)
    i = np.arange(degree + 1)
    return comb(degree, i)[None, :] * (u[:, None] ** i[None, :]) * ((1.0 - u)[:, None] ** (degree - i)[None, :])


def fit_bezier_control_points(positions: np.ndarray, degree: int,
                              param: np.ndarray | None = None) -> np.ndarray:
    """Least-squares fit sampled ``positions`` (M, 3) to ``degree+1`` control points.

    ``param`` are the curve parameters of the samples; defaults to a uniform grid
    on ``[0, 1]``. Returns the control-point array ``W`` of shape ``(degree+1, 3)``.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (M, 3).")
    if param is None:
        u = np.linspace(0.0, 1.0, len(positions))
    else:
        param = np.asarray(param, dtype=float)
        u = (param - param[0]) / (param[-1] - param[0])
    basis = bernstein_matrix(u, degree)
    control_points, *_ = np.linalg.lstsq(basis, positions, rcond=None)
    return control_points


def make_bezier_spacecurve(control_points: np.ndarray) -> SpaceCurve:
    """Wrap Bezier control points as a qurveros ``SpaceCurve`` (order 0, [0, 1])."""
    return SpaceCurve(curve=bezier_curve_vec, order=0, interval=[0.0, 1.0],
                      params=jnp.asarray(np.asarray(control_points, dtype=float)))


def reconstruct_spacecurve(spacecurve: SpaceCurve, degree: int, *, n_samples: int = 400):
    """Sample any ansatz ``SpaceCurve``, fit a degree-n Bezier, and rebuild it.

    Works for both position (order 0) and tangent (order 1) ansaetze: the sampled
    positions come from ``frenet_dict['curve']``. Returns ``(control_points,
    reconstructed_spacecurve)``.
    """
    spacecurve.evaluate_frenet_dict(n_samples)
    positions = np.asarray(spacecurve.frenet_dict["curve"])
    x_values = np.asarray(spacecurve.frenet_dict["x_values"])
    control_points = fit_bezier_control_points(positions, degree, param=x_values)
    return control_points, make_bezier_spacecurve(control_points)
