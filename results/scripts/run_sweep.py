"""Step 11f -- BARQ initialization-quality sweep (COMPUTE, one-shot).

Honest framing (see _dev_logs/step11d_sweep.md): BARQ's point gate-fixing keeps the
gate exact but does NOT faithfully preserve a seeded ansatz -- it mangles the
curve (e.g. rcp's energy inflates from 33x to ~600x), and can even invert the
good/bad ordering. So this is NOT an ansatz-quality study. It asks a narrower,
honest question:

    Does WARM-STARTING BARQ from a simple geometric curve beat its random default?

Starting points, all seeded into BARQ (gate exact via PGF):
  * 4 structured seeds -- rcp_lemniscate (good, closed) and three open arcs
    (circle / triangle / gaussian);
  * an ENSEMBLE of random BARQ defaults (5 seeds) -- the no-prior baseline as a
    distribution, not a single (possibly unlucky) draw.

Energy weight is always >= 0.001: verified to prevent the pure-dephasing energy
runaway (energy -> ~2900x, gate -> 0.95) and keep the gate exact -- this small
always-on penalty is the physical realism floor.

Records the full per-25-step history (cost, all Chat_i, control params) locally;
a curated summary + control-solution library + figures go to results/.

Run (one-shot, ~15 min):
    conda run -n curve python results/scripts/run_sweep.py
"""

import json
import os

import numpy as np

import pulse_shape_novera  # noqa: F401 -- applies qurveros patch
from pulse_shape_novera import (make_rcp_spacecurve, make_circle_arc_spacecurve,
                                 make_triangle_pulse_spacecurve, make_gaussian_arc_spacecurve)
from pulse_shape_novera import sweep
from pulse_shape_novera.barq import barq_gate_fidelity

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "..", "data")
HISTORY = os.path.join(ROOT, "_dev_logs", "sweep_history")
ANGLE = np.pi
BARQ_ITER = 400
CKPT = 25
DEPH_FLOOR = 1e-3
ENSEMBLE_SEEDS = [101, 202, 303, 404, 505]

STRUCTURED = {
    "rcp (good, closed)": lambda: make_rcp_spacecurve("rcp_lemniscate", ANGLE),
    "circle (open)":      lambda: make_circle_arc_spacecurve(ANGLE),
    "triangle (open)":    lambda: make_triangle_pulse_spacecurve(ANGLE),
    "gaussian (open)":    lambda: make_gaussian_arc_spacecurve(ANGLE),
}

reference = sweep.compute_references(make_circle_arc_spacecurve(ANGLE))


def _floored_cost(chat, weights):
    c = dict(chat)
    for t in ("closure", "curve_area"):
        c[t] = max(c[t], DEPH_FLOOR)
    return sweep.total_cost(c, weights)


def build_structured_seeds():
    return {name: sweep.seed_free_points_from_curve(factory())
            for name, factory in STRUCTURED.items()}


def report_seeds(seeds):
    """Log what each structured seed BECOMES inside BARQ (mangling is real)."""
    print("=== structured seeds after entering BARQ (pre-optimization) ===")
    for name, free in seeds.items():
        b = sweep.make_seeded_barq(free); b.evaluate_frenet_dict(400)
        chat = sweep.evaluate_chat(b, reference)
        print(f"  {name:22s} gateF={barq_gate_fidelity(b):.4f} "
              f"energy={chat['energy']:8.1f} curve_area={chat['curve_area']:.2e}", flush=True)
    print()


def weights_for(vdeph, venergy):
    return {"closure": vdeph, "curve_area": vdeph, "energy": venergy}


def run_grid(name, deph_vals, energy_vals, struct_seeds, keep_traj_cells):
    ny, nx = len(energy_vals), len(deph_vals)
    ns, ne = len(struct_seeds), len(ENSEMBLE_SEEDS)
    nfree = sweep.N_FREE_POINTS
    ncheck = len(range(0, BARQ_ITER + 1, CKPT))
    steps_axis = list(range(0, BARQ_ITER + 1, CKPT))
    struct_names = list(struct_seeds.keys())

    s_cost = np.full((ny, nx, ns), np.nan)
    s_gate = np.full((ny, nx, ns), np.nan)
    s_chat = {t: np.full((ny, nx, ns), np.nan) for t in sweep.INV_TERMS}
    s_free = np.full((ny, nx, ns, nfree, 3), np.nan)
    e_cost = np.full((ny, nx, ne), np.nan)
    e_gate = np.full((ny, nx, ne), np.nan)
    e_chat = {t: np.full((ny, nx, ne), np.nan) for t in sweep.INV_TERMS}
    s_cost_hist = np.full((ny, nx, ns, ncheck), np.nan)
    s_params_hist = np.full((ny, nx, ns, ncheck, nfree, 3), np.nan)
    traj = {}

    for iy, ve in enumerate(energy_vals):
        for ix, vd in enumerate(deph_vals):
            w = weights_for(vd, ve)
            # structured seeds (full history)
            s_runs = []
            for k, free in enumerate(struct_seeds.values()):
                r = sweep.optimize_barq(w, reference, init_free_points=free,
                                        max_iter=BARQ_ITER, checkpoint_every=CKPT,
                                        cost_fn=_floored_cost)
                s_runs.append(r)
                s_cost[iy, ix, k] = r["cost_history"][-1]
                s_gate[iy, ix, k] = r["final_gate"]
                for t in sweep.INV_TERMS:
                    s_chat[t][iy, ix, k] = r["final_chat"][t]
                s_cost_hist[iy, ix, k, :] = r["cost_history"]
                s_params_hist[iy, ix, k, :, :, :] = np.stack(r["params_history"])
                s_free[iy, ix, k] = r["final_free_points"]
            # random-default ensemble (final only + trajectory for traj cells)
            e_runs = []
            for j, sd in enumerate(ENSEMBLE_SEEDS):
                r = sweep.optimize_barq(w, reference, init_free_points=None, seed=sd,
                                        max_iter=BARQ_ITER, checkpoint_every=CKPT,
                                        cost_fn=_floored_cost)
                e_runs.append(r)
                e_cost[iy, ix, j] = r["cost_history"][-1]
                e_gate[iy, ix, j] = r["final_gate"]
                for t in sweep.INV_TERMS:
                    e_chat[t][iy, ix, j] = r["final_chat"][t]
            if (ix, iy) in keep_traj_cells:
                traj[f"{ix}_{iy}"] = {
                    "vdeph": float(vd), "venergy": float(ve),
                    "steps": s_runs[0]["steps"],
                    "struct_cost": [r["cost_history"] for r in s_runs],
                    "ens_cost": [r["cost_history"] for r in e_runs]}
            print(f"[energy={ve:g} deph={vd:g}] struct_final {np.round(s_cost[iy,ix],2)} "
                  f"ens_final[min,med,max] "
                  f"[{np.min(e_cost[iy,ix]):.2f},{np.median(e_cost[iy,ix]):.2f},{np.max(e_cost[iy,ix]):.2f}] "
                  f"gate[min] s={np.min(s_gate[iy,ix]):.3f} e={np.min(e_gate[iy,ix]):.3f}", flush=True)

    summary = {"struct_final_cost": s_cost, "struct_gate": s_gate, "struct_free_points": s_free,
               "ens_final_cost": e_cost, "ens_gate": e_gate}
    for t in sweep.INV_TERMS:
        summary[f"struct_chat_{t}"] = s_chat[t]
        summary[f"ens_chat_{t}"] = e_chat[t]
    np.savez_compressed(os.path.join(DATA, f"sweep_{name}.npz"), **summary)
    meta = {"deph_vals": list(map(float, deph_vals)), "energy_vals": list(map(float, energy_vals)),
            "struct_names": struct_names, "n_ensemble": ne, "ensemble_seeds": ENSEMBLE_SEEDS,
            "barq_iter": BARQ_ITER, "ckpt": CKPT, "n_free_points": nfree,
            "references": reference["refs"], "steps_axis": steps_axis, "traj": traj}
    with open(os.path.join(DATA, f"sweep_{name}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    hist = {"struct_cost_hist": s_cost_hist, "struct_params_hist": s_params_hist,
            "steps_axis": np.array(steps_axis)}
    np.savez_compressed(os.path.join(HISTORY, f"{name}_history.npz"), **hist)
    print(f"saved sweep_{name} (curated + history)", flush=True)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(HISTORY, exist_ok=True)
    seeds = build_structured_seeds()
    report_seeds(seeds)
    deph_vals = [0.3, 1.0, 3.0]
    energy_vals = [0.001, 0.01, 0.1]
    run_grid("initquality", deph_vals, energy_vals, seeds,
             keep_traj_cells={(2, 0), (2, 2)})   # (deph=3,energy=0.001) and (deph=3,energy=0.1)
    print("ALL DONE", flush=True)
