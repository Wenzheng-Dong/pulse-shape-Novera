"""Step 4 — regression guard: planar-curve robustness-loss gradients are finite.

qurveros' Frenet computation produces NaN gradients on exactly-planar curves
(e.g. a circle: r''' is parallel to T, so |T x r'''| = 0 and vector_norm's
gradient is 0/0). Our package patches qurveros.frametools.vector_norm with a
safe norm at import (see _patches.py). This test locks that in: without the
patch these gradients are NaN (and qurveros' debug_nans would even raise).

Importing pulse_shape_novera applies the patch.
"""

import numpy as np
import jax.numpy as jnp
import pytest

import pulse_shape_novera  # noqa: F401 -- import applies the qurveros patch
from qurveros.optspacecurve import OptimizableSpaceCurve
from qurveros import losses


def _planar_circle(x, kappa):
    # Exactly planar (x-component identically zero) -> triggers the degeneracy.
    return [jnp.zeros_like(x), (1.0 - jnp.cos(kappa * x)) / kappa, jnp.sin(kappa * x) / kappa]


def _grad(loss_fn, kappa=2.0):
    sc = OptimizableSpaceCurve(curve=_planar_circle, order=0, interval=[0.0, 1.0], params=kappa)
    sc.initialize_parameters(kappa)
    sc.prepare_optimization_loss([loss_fn, 1.0])
    return float(np.asarray(sc.loss_grad(sc.params)).ravel()[0])


@pytest.mark.parametrize("loss_fn", [
    losses.total_curvature_loss,
    losses.max_amp_loss,
    losses.tantrix_zero_area_loss,
    losses.curve_zero_area_loss,
])
def test_planar_robustness_loss_gradient_is_finite(loss_fn):
    """All curvature/frame-based losses give a finite gradient on a planar curve."""
    g = _grad(loss_fn)
    assert np.isfinite(g), f"{loss_fn.__name__} gradient not finite: {g}"


def test_planar_gradients_have_correct_values():
    """Beyond finite: the recovered gradients match the analytic values.

    For the unit-speed circle over [0, 1] with curvature kappa: total length is
    1, so total_curvature = kappa and max_amp = kappa, both with d/dkappa = 1.
    """
    assert _grad(losses.total_curvature_loss) == pytest.approx(1.0, abs=1e-6)
    assert _grad(losses.max_amp_loss) == pytest.approx(1.0, abs=1e-6)
