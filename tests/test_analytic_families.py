"""Step 5b — complete the analytic-family tier: lissajous_3d, alpha_3d,
zeng_clifford_3d (fixed-form 3D members, non-unit-speed).

Two independent "correct & accurate" checks per family:
1. our JAX closed form reproduces curvecontroltoolbox's sampled positions exactly
   (adapter really is that family);
2. qurveros' autodiff curvature/torsion agree with an independent numpy
   finite-difference computation on the same samples (geometry extraction is
   right, even for these varying-curvature curves).
"""

import numpy as np
import pytest

import pulse_shape_novera  # noqa: F401 -- applies the qurveros patch
from pulse_shape_novera import ANALYTIC_FAMILY_CURVES, make_analytic_spacecurve

FAMILIES = sorted(ANALYTIC_FAMILY_CURVES)


def _finite_diff_curvature_torsion(positions, t):
    v = np.gradient(positions, t, axis=0, edge_order=2)
    a = np.gradient(v, t, axis=0, edge_order=2)
    j = np.gradient(a, t, axis=0, edge_order=2)
    vxa = np.cross(v, a)
    vxa_norm = np.linalg.norm(vxa, axis=1)
    curvature = vxa_norm / np.linalg.norm(v, axis=1) ** 3
    torsion = np.sum(vxa * j, axis=1) / (vxa_norm ** 2)
    return curvature, torsion


@pytest.mark.parametrize("name", FAMILIES)
def test_jax_curve_matches_cct_positions(name):
    """The JAX closed form reproduces cct's family positions to machine precision."""
    curve_families = pytest.importorskip("curvecontroltoolbox.curve_families")
    curve_fn, interval = ANALYTIC_FAMILY_CURVES[name]

    nc = curve_families.make_curve(name)
    t = np.asarray(nc.parameter_values)
    mine = np.array([np.asarray(curve_fn(ti)) for ti in t])

    assert np.allclose(mine, nc.positions, atol=1e-9), np.abs(mine - nc.positions).max()


@pytest.mark.parametrize("name", FAMILIES)
def test_qurveros_geometry_matches_finite_diff(name):
    """qurveros autodiff curvature/torsion == independent numpy finite-diff."""
    curve_fn, interval = ANALYTIC_FAMILY_CURVES[name]

    sc = make_analytic_spacecurve(name)
    sc.evaluate_frenet_dict()
    kq = np.asarray(sc.frenet_dict["curvature"])
    tq = np.asarray(sc.frenet_dict["torsion"])
    n = len(kq)

    t = np.linspace(interval[0], interval[1], n)
    pos = np.array([np.asarray(curve_fn(ti)) for ti in t])
    kf, tf = _finite_diff_curvature_torsion(pos, t)

    sl = slice(5, -5)  # drop noisier endpoints of the finite-diff reference
    assert np.allclose(kq[sl], kf[sl], rtol=2e-2, atol=1e-2), "curvature mismatch"
    assert np.allclose(tq[sl], tf[sl], rtol=2e-2, atol=1e-2), "torsion mismatch"
