"""Step 11d -- ansatz-quality weight sweep (COMPUTE).

For each weight combination ``w`` we compare three routes to the X(pi) gate and
record which minimizes the normalized cost ``C = sum_i w_i Chat_i``:

  * **good** -- rcp_lemniscate used as is (fixed; gate-correct + dephasing-robust),
  * **naive** -- the open constant-amplitude arc used as is (fixed),
  * **optimize** -- BARQ run under that same ``w`` (PGF keeps the gate exact).

The good/naive costs are evaluated once (they are fixed curves); only BARQ is
optimized per cell, so the whole sweep is cheap. See ``sweep.py`` for the
normalization contract and why this replaced the earlier soft-gate design.

Robustness floor: dephasing costs (closure, curve area) below ``DEPH_FLOOR`` are
clamped -- a curve 1000x better-closed than the naive pulse is already perfectly
dephasing-robust, and rewarding further (physically meaningless) reduction would
let a machine-zero optimizer nominally "beat" an already-robust free ansatz.

Outputs (retained, tracked): ``results/data/sweep_*.npz`` + ``.json``.
Figures are produced separately by ``plot_sweep.py``.

Run (background, ~10 min):
    conda run -n curve python results/scripts/run_sweep.py
"""

import json
import os

import numpy as np

import pulse_shape_novera  # noqa: F401 -- applies qurveros patch
from pulse_shape_novera import make_rcp_spacecurve, make_circle_arc_spacecurve
from pulse_shape_novera import sweep

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
ANGLE = np.pi
BARQ_ITER = 900
DEPH_FLOOR = 1e-3
ROUTES = ["good (rcp, as is)", "naive (arc, as is)", "optimize (BARQ)"]

reference = sweep.compute_references(make_circle_arc_spacecurve(ANGLE))
GOOD_CHAT = sweep.evaluate_chat(make_rcp_spacecurve("rcp_lemniscate", ANGLE), reference)
NAIVE_CHAT = sweep.evaluate_chat(make_circle_arc_spacecurve(ANGLE), reference)


def _floored(chat):
    """Clamp the dephasing terms at DEPH_FLOOR (robust-enough is robust-enough)."""
    c = dict(chat)
    for t in ("closure", "curve_area"):
        c[t] = max(c[t], DEPH_FLOOR)
    return c


def cost(chat, weights):
    return sweep.total_cost(_floored(chat), weights)


def weights_for(axis_x, vx, axis_y, vy, base):
    """base weights overridden by two swept axes; 'dephasing' sets closure+curve_area."""
    w = dict(base)
    for axis, v in ((axis_x, vx), (axis_y, vy)):
        if axis == "dephasing":
            w["closure"] = w["curve_area"] = v
        else:
            w[axis] = v
    return w


def run_grid(axis_x, xs, axis_y, ys, base, keep_traj_cells):
    ny, nx = len(ys), len(xs)
    winner = np.full((ny, nx), -1, dtype=int)
    costs = {r: np.full((ny, nx), np.nan) for r in ("good", "naive", "barq")}
    barq_chat = {t: np.full((ny, nx), np.nan) for t in sweep.INV_TERMS}
    traj = {}

    for iy, vy in enumerate(ys):
        for ix, vx in enumerate(xs):
            w = weights_for(axis_x, vx, axis_y, vy, base)
            bq = sweep.optimize_barq(w, reference, max_iter=BARQ_ITER, cost_fn=cost)
            c_good = cost(GOOD_CHAT, w)
            c_naive = cost(NAIVE_CHAT, w)
            c_barq = cost(bq["final_chat"], w)
            trio = [c_good, c_naive, c_barq]
            winner[iy, ix] = int(np.argmin(trio))
            costs["good"][iy, ix], costs["naive"][iy, ix], costs["barq"][iy, ix] = trio
            for t in sweep.INV_TERMS:
                barq_chat[t][iy, ix] = bq["final_chat"][t]
            if (ix, iy) in keep_traj_cells:
                traj[f"{ix}_{iy}"] = {
                    "vx": float(vx), "vy": float(vy),
                    "steps": bq["steps"], "barq_cost": bq["cost_history"],
                    "c_good": c_good, "c_naive": c_naive,
                }
            print(f"[{axis_y}={vy:g} x {axis_x}={vx:g}] winner={ROUTES[winner[iy, ix]]:20s} "
                  f"| good={c_good:.3g} naive={c_naive:.3g} barq={c_barq:.3g}")

    return {"axis_x": axis_x, "xs": list(map(float, xs)),
            "axis_y": axis_y, "ys": list(map(float, ys)),
            "winner": winner, "costs": costs, "barq_chat": barq_chat, "traj": traj}


def save_grid(name, g):
    arrays = {"winner": g["winner"]}
    for r, arr in g["costs"].items():
        arrays[f"cost_{r}"] = arr
    for t, arr in g["barq_chat"].items():
        arrays[f"barq_{t}"] = arr
    np.savez_compressed(os.path.join(DATA, f"sweep_{name}.npz"), **arrays)
    meta = {k: g[k] for k in ("axis_x", "xs", "axis_y", "ys")}
    meta.update(routes=ROUTES, deph_floor=DEPH_FLOOR, barq_iter=BARQ_ITER,
                references=reference["refs"], good_chat=GOOD_CHAT, naive_chat=NAIVE_CHAT,
                traj=g["traj"])
    with open(os.path.join(DATA, f"sweep_{name}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved sweep_{name}.npz / .json")


if __name__ == "__main__":
    w_deph = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]     # 6
    w_energy = [0.0, 0.03, 0.1, 0.3, 1.0]         # 5
    w_tantrix = [0.0, 0.1, 0.3, 1.0]              # 4
    w_maxamp = [0.0, 0.03, 0.1, 0.3]              # 4

    # MAIN: dephasing x energy. Keep trajectory cells: dephasing-dominated,
    # balanced/mixed, energy-dominated.
    main = run_grid("dephasing", w_deph, "energy", w_energy,
                    base={}, keep_traj_cells={(5, 0), (3, 2), (1, 4)})
    save_grid("main", main)

    # SLICE1: tantrix x energy at fixed dephasing weight = 1.
    s1 = run_grid("tantrix", w_tantrix, "energy", w_energy,
                  base={"closure": 1.0, "curve_area": 1.0}, keep_traj_cells=set())
    save_grid("tantrix_energy", s1)

    # SLICE2: max_amp x energy at fixed dephasing weight = 1.
    s2 = run_grid("max_amp", w_maxamp, "energy", w_energy,
                  base={"closure": 1.0, "curve_area": 1.0}, keep_traj_cells=set())
    save_grid("maxamp_energy", s2)

    print("ALL DONE")
