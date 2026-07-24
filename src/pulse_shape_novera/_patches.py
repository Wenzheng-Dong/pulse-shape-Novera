"""Runtime patches applied to upstream libraries when this package is imported.

Keeping the patch here (rather than editing the qurveros clone) means it is
version-controlled with our code, travels with the repo, and is traceable.

------------------------------------------------------------------------------
PATCH: safe vector norm in qurveros.frametools  (planar-curve NaN gradient)
------------------------------------------------------------------------------
Symptom
    Optimizing an *exactly planar* curve (e.g. any single-qubit X-rotation,
    which is planar in SCQC) with a curvature/frame-based loss
    (``max_amp_loss``, ``tantrix_zero_area_loss``, ``curve_zero_area_loss``,
    ``total_curvature_loss``) yields a NaN gradient and, since qurveros enables
    jax_debug_nans, raises FloatingPointError. Forward simulation is unaffected.

Root cause (verified 2026-07-23; see _dev_logs/step04_planar_nan_fix.md)
    qurveros stacks the norms |T x r^(k)| of tangent-vs-higher-derivative cross
    products. For a planar curve some of these are identically zero -- e.g. a
    circle has r''' = -k^2 r' (parallel to the tangent T), so |T x r'''| = 0.
    qurveros' ``vector_norm(v) = sqrt(sum(v**2))`` has gradient v/|v| = 0/0 = NaN
    at v = 0. Even though that array element is unused downstream, reverse-mode
    multiplies its NaN gradient by a zero cotangent (0 * NaN = NaN), poisoning
    the whole gradient. This is an autodiff artifact, NOT an intrinsic geometric
    singularity: the planar Frenet frame is smooth.

Fix
    Replace ``vector_norm`` with a double-``where`` "safe norm" whose gradient at
    the origin is 0 instead of NaN. Numerically identical for nonzero vectors;
    at zero it returns 0 with a 0 gradient. Verified to make all four losses'
    gradients finite AND correct on an exactly-planar circle (e.g. the analytic
    tantrix-area gradient, recovered as the eps->0 limit of a small 3D lift).

Must be applied BEFORE qurveros traces its jitted frame functions (i.e. before
the first curve evaluation). We call apply_qurveros_patches() at package import.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import qurveros.frametools as _frametools

_PATCHED = False


@jax.jit
def _safe_vector_norm(vector):
    """Euclidean norm with a well-defined (zero) gradient at the origin.

    Identical to ``sqrt(sum(v**2))`` for nonzero ``v``; at ``v == 0`` returns 0
    with gradient 0 rather than the NaN produced by ``v / |v|``.
    """
    sq = jnp.sum(vector ** 2)
    sq_safe = jnp.where(sq > 0.0, sq, 1.0)         # keep the sqrt branch off 0
    return jnp.where(sq > 0.0, jnp.sqrt(sq_safe), 0.0)


def apply_qurveros_patches() -> None:
    """Idempotently install our qurveros patches. Called at package import."""
    global _PATCHED
    if _PATCHED:
        return
    _frametools.vector_norm = _safe_vector_norm
    _PATCHED = True
