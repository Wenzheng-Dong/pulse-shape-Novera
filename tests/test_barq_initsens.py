"""Step 11 — BARQ init-sensitivity (the clean version of the ansatz-quality study).

With the gate hard-fixed (F == 1 always, no weighted-sum drift), BARQ optimizing
[curve_area + lambda*energy] reliably reaches a low dephasing cost regardless of
its random free-point seed -- i.e. the *primary* robustness objective is
init-insensitive. (Its energy landing is more seed-dependent; the good analytic
ansatz rcp_lemniscate instead gives a balanced solution with no optimization --
see _dev_logs/step11_barq_initsens.md and the figure.)

Kept to two seeds / short optimization to bound test time.
"""

import numpy as np

import pulse_shape_novera  # noqa: F401 -- applies the qurveros patch
from pulse_shape_novera import make_barq_xgate, optimize_barq_robustness, barq_gate_fidelity, pulse_energy_loss
from qurveros import losses


def _optimize(seed, lam=0.05, iters=800):
    bc = make_barq_xgate(seed=seed)
    optimize_barq_robustness(
        bc, loss_terms=[[losses.curve_zero_area_loss, 1.0], [pulse_energy_loss, lam]],
        max_iter=iters)
    bc.evaluate_frenet_dict()
    area = float(np.sum(losses.curve_zero_area_loss(bc.frenet_dict)))
    return barq_gate_fidelity(bc), area


def test_barq_dephasing_is_init_insensitive_gate_fixed():
    """Two seeds: gate stays exactly X, and both reach a low, similar dephasing."""
    f1, a1 = _optimize(seed=11)
    f2, a2 = _optimize(seed=3003)

    assert f1 > 1 - 1e-6 and f2 > 1 - 1e-6          # gate hard-fixed throughout
    assert a1 < 3e-2 and a2 < 3e-2                   # dephasing reliably reduced
    assert max(a1, a2) < 5 * min(a1, a2)             # ... to a similar level (init-insensitive)
