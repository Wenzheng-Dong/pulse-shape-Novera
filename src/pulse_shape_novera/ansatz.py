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


# --------------------------------------------------------------------------
# Fixed-form analytic 3D families (default "study members" from
# curvecontroltoolbox). These are non-unit-speed closed curves; the constants
# below reproduce cct's builder defaults exactly (verified by positions match).
# The ``params`` argument is accepted (qurveros calls curve(x, params)) but
# unused here -- we wrap the specific member, not a parameterized shape.
# --------------------------------------------------------------------------

def lissajous_curve(x, params=None):
    """cct ``lissajous_3d`` / ``lissajous_simple_3d`` default member, over [0, 2*pi]."""
    wx = 0.72
    wy = -1.0 / (2.0 * wx)
    return [jnp.cos(x) + wx * jnp.cos(2.0 * x),
            jnp.sin(x) + wy * jnp.sin(2.0 * x),
            0.28 * jnp.sin(3.0 * x + 0.55)]


def alpha_curve(x, params=None):
    """cct ``alpha_3d`` explicit curve, over [0, 2*pi]."""
    s2 = jnp.sqrt(2.0)
    return [0.25 * (s2 * jnp.cos(2.0 * x) - 2.0 * jnp.cos(x)),
            0.25 * (-s2 * jnp.sin(2.0 * x) - 2.0 * jnp.sin(x)),
            0.5 * jnp.sqrt(s2 * jnp.cos(3.0 * x) + 2.5)]


def zeng_clifford_curve(x, params=None):
    """cct ``zeng_clifford_3d`` (Zeng PRA19) curve, over [0, 1] with q = 1.6054."""
    q = 1.6054
    cq, sq = jnp.cos(q), jnp.sin(q)
    pref = jnp.sqrt(2.0) * jnp.sin(jnp.pi * x)
    sin_sq = jnp.sin(0.5 * jnp.pi * x) ** 2
    cos_sq = jnp.cos(0.5 * jnp.pi * x) ** 2
    r1 = [0.0 * x, pref * sin_sq, pref * cos_sq]
    # r2_base = pref * [sin_sq, cos_sq, 0], then row-vector @ R_z(q).
    b0, b1 = pref * sin_sq, pref * cos_sq
    r2 = [cq * b0 + sq * b1, -sq * b0 + cq * b1, 0.0 * x]
    return [(1.0 - x) * r1[i] + x * r2[i] for i in range(3)]


# name -> (curve function, parameter interval). Circle/helix have their own
# parameterized factories above; these are the fixed-form members.
ANALYTIC_FAMILY_CURVES = {
    "lissajous_3d": (lissajous_curve, [0.0, 2.0 * float(jnp.pi)]),
    "alpha_3d": (alpha_curve, [0.0, 2.0 * float(jnp.pi)]),
    "zeng_clifford_3d": (zeng_clifford_curve, [0.0, 1.0]),
}


def make_analytic_spacecurve(name: str) -> SpaceCurve:
    """Build a qurveros ``SpaceCurve`` for a fixed-form analytic family (Path A)."""
    if name not in ANALYTIC_FAMILY_CURVES:
        raise KeyError(f"unknown analytic family {name!r}; have {sorted(ANALYTIC_FAMILY_CURVES)}")
    curve_fn, interval = ANALYTIC_FAMILY_CURVES[name]
    return SpaceCurve(curve=curve_fn, order=0, interval=interval, params=0.0)
