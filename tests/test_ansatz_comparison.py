"""Step 9 — the ansatz-quality finding (dephasing-vs-amplitude tradeoff).

Core, cheap assertions of the corrected claim-2 thesis (the full Pareto sweep
lives in _dev_logs/step09_plot.py):

- A *good* ansatz for robust X is closed + zero-area (rcp_lemniscate): it
  implements X AND already has near-zero dephasing cost (already ~optimal for
  static dephasing).
- An *open* ansatz (circle_arc) also implements X but is NOT dephasing-robust:
  its curve zero-area cost is orders of magnitude larger -> it is dominated.

"Good ansatz" therefore means "satisfies the geometric robustness conditions",
not merely "implements the gate".
"""

import numpy as np

import pulse_shape_novera  # noqa: F401 -- applies the qurveros patch
from pulse_shape_novera import make_rcp_spacecurve, make_circle_arc_spacecurve, gate_fidelity, rotation
from qurveros import losses


def _x_fid_and_dephasing(sc):
    fid = gate_fidelity(sc, rotation("x", np.pi))["fidelity"]
    sc.evaluate_frenet_dict()
    area = float(np.sum(losses.curve_zero_area_loss(sc.frenet_dict)))
    return fid, area


def test_both_ansaetze_implement_x():
    for sc in (make_rcp_spacecurve("rcp_lemniscate", np.pi), make_circle_arc_spacecurve(np.pi)):
        fid, _ = _x_fid_and_dephasing(sc)
        assert fid > 1.0 - 1e-5


def test_good_ansatz_is_dephasing_robust_bad_one_is_not():
    """Closed+zero-area rcp_lemniscate dominates the open circle on dephasing cost."""
    _, rcp_area = _x_fid_and_dephasing(make_rcp_spacecurve("rcp_lemniscate", np.pi))
    _, circle_area = _x_fid_and_dephasing(make_circle_arc_spacecurve(np.pi))

    assert rcp_area < 1e-3            # good ansatz already ~optimal for dephasing
    assert circle_area > 1e-2         # open ansatz is not dephasing-robust
    assert circle_area > 100 * rcp_area   # orders-of-magnitude domination
