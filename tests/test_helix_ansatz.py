"""Step 5 — Path-A adapter for the ``helix`` family (first nonzero torsion).

The helix is the simplest genuine 3D curve: constant curvature ``kappa`` AND
constant torsion ``tau``. This is the first family that exercises torsion, i.e.
the drive-phase rate ``d(Phi)/dt = tau`` (the planar families all had tau = 0).
We check curvature and torsion are correct & accurate, cross-checked three ways:
qurveros autodiff, the analytic constants, and curvecontroltoolbox's finite-diff
inverse map.
"""

import numpy as np
import pytest

import pulse_shape_novera  # noqa: F401 -- applies the qurveros patch
from pulse_shape_novera import make_helix_spacecurve


CASES = [(1.5, 0.5), (1.0, 1.0), (2.0, 0.3)]  # (curvature, torsion)


@pytest.mark.parametrize("kappa,tau", CASES)
def test_qurveros_curvature_and_torsion_match_analytic(kappa, tau):
    """qurveros autodiff recovers constant curvature == kappa and torsion == tau."""
    sc = make_helix_spacecurve(kappa, tau, duration=1.2)
    sc.evaluate_frenet_dict()
    curvature = np.asarray(sc.frenet_dict["curvature"])
    torsion = np.asarray(sc.frenet_dict["torsion"])

    assert np.allclose(curvature, kappa, atol=1e-4), (curvature.min(), curvature.max())
    assert np.allclose(torsion, tau, atol=1e-4), (torsion.min(), torsion.max())


@pytest.mark.parametrize("kappa,tau", CASES)
def test_cross_library_curvature_and_torsion(kappa, tau):
    """curvecontroltoolbox's inverse map agrees on curvature and torsion."""
    curve_families = pytest.importorskip("curvecontroltoolbox.curve_families")
    inverse = pytest.importorskip("curvecontroltoolbox.inverse")

    nc = curve_families.make_curve("helix", curvature=kappa, torsion=tau, duration=1.2)
    res = inverse.curve_to_control(
        inverse.make_space_curve(nc.parameter_values, nc.positions)
    )
    # Interior points: finite differences are noisier at the endpoints.
    assert np.mean(res.curvature[3:-3]) == pytest.approx(kappa, abs=1e-2)
    assert np.mean(res.torsion[3:-3]) == pytest.approx(tau, abs=2e-2)
