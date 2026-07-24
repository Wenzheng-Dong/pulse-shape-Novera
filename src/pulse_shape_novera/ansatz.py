"""Path-A ansatz adapters: express a curve family as a JAX ``curve(x, params)``
and hand it to qurveros directly (no Bezier projection).

This is the "Path A" route from ``_plan.md`` Q2: rather than fitting an existing
curve onto Bezier control points, we re-express the (analytic) curve family in
JAX closed form so qurveros can autodiff the Frenet frame from it. ``circle_arc``
is the first and simplest adapter; further families follow the same shape
(a ``curve(x, params) -> [x, y, z]`` callable plus the parameter interval).

SCQC convention (see ``curvecontroltoolbox`` foundations): the curve is unit
speed, so the parameter is arc length = evolution time, curvature is the Rabi
rate ``Omega`` and torsion is ``d(phi)/dt``.
"""

from __future__ import annotations

import jax.numpy as jnp
from qurveros.spacecurve import SpaceCurve


def circle_arc_curve(x, kappa):
    """Unit-speed circular arc in the ``yz`` plane, matching curvecontroltoolbox's
    ``circle_arc`` family.

    Parameters
    ----------
    x : float or jax array
        Curve parameter (arc length), traversed over ``[0, duration]``.
    kappa : float
        Constant curvature = ``turning_angle / duration``. Under the SCQC map this
        is the (constant) Rabi rate ``Omega`` of the resulting pulse.

    Returns
    -------
    list
        ``[x_comp, y_comp, z_comp]`` position vector. The initial tangent is ``+z``
        and the curve is unit speed, so ``x`` doubles as evolution time.
    """
    return [jnp.zeros_like(x),
            (1.0 - jnp.cos(kappa * x)) / kappa,
            jnp.sin(kappa * x) / kappa]


def make_circle_arc_spacecurve(turning_angle: float, duration: float = 1.0) -> SpaceCurve:
    """Build a qurveros ``SpaceCurve`` for the ``circle_arc`` family via Path A.

    Parameters
    ----------
    turning_angle : float
        Total tangent turning over the arc, in radians (the gate knob; e.g. ``pi``
        for the half-circle ``X(pi)`` baseline).
    duration : float, optional
        Arc length / gate time ``T_g`` (default 1.0). With unit speed, curvature is
        ``kappa = turning_angle / duration``.

    Returns
    -------
    qurveros.spacecurve.SpaceCurve
        Configured but not yet evaluated. Call ``evaluate_frenet_dict()`` and
        ``evaluate_control_dict('XY')`` to obtain the geometry and control fields.
    """
    if duration <= 0.0:
        raise ValueError("duration must be positive.")
    kappa = turning_angle / duration
    return SpaceCurve(
        curve=circle_arc_curve,
        order=0,
        interval=[0.0, duration],
        params=kappa,
    )


def helix_curve(x, params):
    """Unit-speed helix with constant curvature and torsion, matching
    curvecontroltoolbox's ``helix`` family (``_build_constant_frenet_helix``).

    Parameters
    ----------
    x : float or jax array
        Curve parameter (arc length).
    params : array-like of shape (2,)
        ``(curvature, torsion)``. Under the SCQC map curvature is the Rabi rate
        ``Omega`` and torsion is ``d(Phi)/dt`` -- so unlike the planar families
        this one exercises a nonzero (constant) drive-phase rate.

    Returns
    -------
    list
        ``[x, y, z]`` position. With ``c = sqrt(k^2 + t^2)``, radius ``a = k/c^2``
        and slope ``b = t/c^2`` the curve ``[a cos(cx), a sin(cx), b c x]`` is
        unit speed with curvature ``k`` and torsion ``t`` (rigid orientation is
        irrelevant to those invariants, so we omit cct's initial-frame rotation).
    """
    kappa, tau = params[0], params[1]
    c = jnp.sqrt(kappa ** 2 + tau ** 2)
    radius = kappa / c ** 2
    slope = tau / c ** 2
    return [radius * jnp.cos(c * x), radius * jnp.sin(c * x), slope * c * x]


def make_helix_spacecurve(curvature: float, torsion: float,
                          duration: float = 1.0) -> SpaceCurve:
    """Build a qurveros ``SpaceCurve`` for the ``helix`` family via Path A.

    Parameters
    ----------
    curvature, torsion : float
        Constant curvature and torsion (both nonzero -> a genuine 3D curve).
    duration : float, optional
        Arc length / gate time ``T_g`` (default 1.0); the helix is unit speed.
    """
    if duration <= 0.0:
        raise ValueError("duration must be positive.")
    if curvature == 0.0 and torsion == 0.0:
        raise ValueError("curvature and torsion cannot both be zero.")
    return SpaceCurve(
        curve=helix_curve,
        order=0,
        interval=[0.0, duration],
        params=jnp.array([float(curvature), float(torsion)]),
    )
