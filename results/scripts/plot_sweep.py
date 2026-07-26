"""Step 11f -- render BARQ initialization-quality figures from swept data (PLOT).

Reads results/data/ and writes two outsider-legible figures to results/figures/.

  fig1_warmstart_vs_random.png
      Final optimized cost from each structured warm-start (rcp / circle /
      triangle / gaussian, seeded into BARQ) vs the random-default ensemble
      (gray band = min-max, tick = median), across weight regimes. Shows which
      starting point BARQ reaches the best solution from.

  fig2_convergence.png
      Cost vs optimization step from the four structured warm-starts and the
      random ensemble (band), in two regimes.

Note (honest): BARQ mangles a seeded curve (it does not preserve the ansatz), so
this compares WARM-START quality, not ansatz quality. See ../README.md.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")
HISTORY = os.path.join(ROOT, "_dev_logs", "sweep_history")

STRUCT_COLORS = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]  # rcp, circle, triangle, gaussian
ENS_COLOR = "#7f7f7f"


def _load(name):
    d = np.load(os.path.join(DATA, f"sweep_{name}.npz"))
    meta = json.load(open(os.path.join(DATA, f"sweep_{name}.json")))
    return d, meta


def fig_warmstart_vs_random():
    d, meta = _load("initquality")
    sc = d["struct_final_cost"]   # [ny(energy), nx(deph), ns]
    ec = d["ens_final_cost"]      # [ny, nx, ne]
    sg = d["struct_gate"]         # [ny, nx, ns]  physical (TTC) gate fidelity
    dvals, evals = meta["deph_vals"], meta["energy_vals"]
    names = meta["struct_names"]
    ny, nx, ns = sc.shape

    combos = [(iy, ix) for iy in range(ny) for ix in range(nx)]
    fig, ax = plt.subplots(figsize=(13, 5))
    xpos = np.arange(len(combos))
    for c, (iy, ix) in enumerate(combos):
        # ensemble range (min-max) + median
        lo, md, hi = ec[iy, ix].min(), np.median(ec[iy, ix]), ec[iy, ix].max()
        ax.vlines(c, lo, hi, color=ENS_COLOR, lw=6, alpha=0.35,
                  label="random default (ensemble min-max)" if c == 0 else None)
        ax.plot(c, md, "_", color=ENS_COLOR, ms=14, mew=2,
                label="random default (median)" if c == 0 else None)
        # structured warm-starts; filled = physical gate F>=0.99, hollow x = gate deficient
        for k in range(ns):
            gate_ok = sg[iy, ix, k] >= 0.99
            ax.plot(c + (k - 1.5) * 0.12, sc[iy, ix, k],
                    "o" if gate_ok else "x", ms=7 if gate_ok else 8,
                    mfc=STRUCT_COLORS[k] if gate_ok else "none",
                    mec=STRUCT_COLORS[k], color=STRUCT_COLORS[k],
                    label=names[k] if c == 0 else None)
    # legend note for the hollow-x convention
    ax.plot([], [], "x", color="k", label="(hollow ×: physical gate F<0.99 — not a valid gate)")
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"deph={dvals[ix]:g}\nenergy={evals[iy]:g}" for (iy, ix) in combos],
                       fontsize=7)
    ax.set_ylabel("final optimized cost  (lower = better)", fontsize=10)
    ax.set_xlabel("weight regime", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.set_title("Warm-starting BARQ from a simple pulse vs its random default\n"
                 "(where a marker sits below the gray band, that warm-start beats "
                 "optimizing from scratch)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig1_warmstart_vs_random.png")
    fig.savefig(out, dpi=140); print("saved", out)


def fig_convergence():
    _, meta = _load("initquality")
    traj = meta["traj"]
    names = meta["struct_names"]
    keys = sorted(traj.keys(), key=lambda k: (traj[k]["venergy"], traj[k]["vdeph"]))
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4.4), sharey=False)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        t = traj[key]
        steps = t["steps"]
        ens = np.array(t["ens_cost"])   # [ne, ncheck]
        ax.fill_between(steps, ens.min(0), ens.max(0), color=ENS_COLOR, alpha=0.25,
                        label="random default (ensemble)")
        for k, cost in enumerate(t["struct_cost"]):
            ax.semilogy(steps, np.maximum(cost, 1e-12), "-", color=STRUCT_COLORS[k],
                        lw=1.8, label=names[k])
        ax.set_title(f"dephasing wt = {t['vdeph']:g},  energy wt = {t['venergy']:g}", fontsize=10)
        ax.set_xlabel("optimization step", fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("total weighted cost  (lower = better)", fontsize=9)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Convergence from four structured warm-starts vs the random-default "
                 "ensemble\n(rcp & circle keep the physical gate exact; triangle/gaussian "
                 "degrade in energy-weighted regimes — see fig1)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig2_convergence.png")
    fig.savefig(out, dpi=140); print("saved", out)


def fig_waveform_evolution(iy=1, ix=1):
    """Control waveform Omega(t), Phi(t) at start / middle / end of optimization,
    for the four structured warm-starts, rebuilt from the saved free-point history.

    Default cell (iy=1, ix=1) = dephasing wt 1, energy wt 0.01 (gate-clean for all
    four seeds). Random-default ensemble is intentionally omitted (too arbitrary).
    """
    import pulse_shape_novera  # noqa: F401
    from pulse_shape_novera import sweep

    h = np.load(os.path.join(HISTORY, "initquality_history.npz"))
    meta = json.load(open(os.path.join(DATA, "sweep_initquality.json")))
    ph = h["struct_params_hist"]         # [ny,nx,ns,ncheck,nfree,3]
    steps = h["steps_axis"]
    names = meta["struct_names"]
    ns = ph.shape[2]
    nchk = ph.shape[3]
    stages = [(0, "start"), (nchk // 2, "middle"), (nchk - 1, "end")]
    stage_colors = {"start": "#cccccc", "middle": "#ff7f0e", "end": "#1f77b4"}

    def waveform(free):
        b = sweep.make_seeded_barq(free)
        b.evaluate_frenet_dict(300)
        fd = b.frenet_dict
        x = np.asarray(fd["x_values"])
        kappa = np.asarray(fd["curvature"])          # Omega (Rabi rate)
        speed = np.asarray(fd["speed"])
        tau = np.asarray(fd["torsion"])              # dPhi/dt
        phi = np.concatenate([[0.0], np.cumsum(0.5 * (tau[1:] * speed[1:] +
                              tau[:-1] * speed[:-1]) * np.diff(x))])  # cumulative phase
        return x, kappa, phi

    fig, axes = plt.subplots(2, ns, figsize=(4.2 * ns, 6.4), sharex=True)
    for k in range(ns):
        for cidx, label in stages:
            x, kappa, phi = waveform(ph[iy, ix, k, cidx])
            axes[0, k].plot(x, kappa, color=stage_colors[label], lw=1.8,
                            label=f"step {int(steps[cidx])} ({label})")
            axes[1, k].plot(x, phi, color=stage_colors[label], lw=1.8)
        axes[0, k].set_title(names[k], fontsize=10)
        axes[1, k].set_xlabel("normalized time  t / T_gate", fontsize=9)
        for row in (0, 1):
            axes[row, k].grid(alpha=0.3)
    axes[0, 0].set_ylabel("drive amplitude  Ω(t)  [= curvature]", fontsize=9)
    axes[1, 0].set_ylabel("drive phase  Φ(t)", fontsize=9)
    axes[0, 0].legend(fontsize=8, loc="best")
    dvals, evals = meta["deph_vals"], meta["energy_vals"]
    fig.suptitle("Control-pulse evolution during optimization "
                 f"(dephasing wt = {dvals[ix]:g}, energy wt = {evals[iy]:g})\n"
                 "each warm-start: start (gray) → middle (orange) → end (blue)",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig3_waveform_evolution.png")
    fig.savefig(out, dpi=140); print("saved", out)


def fig_radar():
    """One radar (spider) chart per weight regime: the final optimized costs of the
    four structured warm-starts across metrics (all axes 'lower = better',
    per-axis normalized within the regime; smaller polygon = better)."""
    d, meta = _load("initquality")
    dvals, evals = meta["deph_vals"], meta["energy_vals"]
    names = meta["struct_names"]
    ny, nx = len(evals), len(dvals)
    ns = len(names)
    # metric name -> array [ny,nx,ns], all lower = better
    metrics = [
        ("gate\ninfidelity", 1.0 - d["struct_gate"]),
        ("1st-order\ndephasing", d["struct_chat_closure"]),
        ("2nd-order\ndephasing", d["struct_chat_curve_area"]),
        ("amplitude\nerror", d["struct_chat_tantrix"]),
        ("pulse\nenergy", d["struct_chat_energy"]),
        ("peak\namplitude", d["struct_chat_max_amp"]),
    ]
    labels = [m[0] for m in metrics]
    nm = len(metrics)
    angles = np.linspace(0, 2 * np.pi, nm, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    fig, axes = plt.subplots(ny, nx, figsize=(4.2 * nx, 4.2 * ny),
                             subplot_kw=dict(polar=True))
    for iy in range(ny):
        for ix in range(nx):
            ax = axes[iy, ix]
            # per-axis min-max normalization across the four seeds
            for k in range(ns):
                vals = []
                for _, arr in metrics:
                    col = arr[iy, ix]            # length ns
                    lo, hi = np.nanmin(col), np.nanmax(col)
                    v = 0.0 if (hi - lo) < 1e-12 else (arr[iy, ix, k] - lo) / (hi - lo)
                    vals.append(v)
                vals_closed = np.concatenate([vals, vals[:1]])
                ax.plot(angles_closed, vals_closed, color=STRUCT_COLORS[k], lw=1.6,
                        label=names[k] if (iy == 0 and ix == 0) else None)
                ax.fill(angles_closed, vals_closed, color=STRUCT_COLORS[k], alpha=0.08)
            ax.set_xticks(angles)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_yticks([])
            ax.set_title(f"deph={dvals[ix]:g}, energy={evals[iy]:g}", fontsize=9, pad=14)
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Final optimized cost profile per weight regime "
                 "(each axis normalized within the regime; smaller polygon = better)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = os.path.join(FIGS, "fig4_radar_cost_profile.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    fig_warmstart_vs_random()
    fig_convergence()
    fig_waveform_evolution()
    fig_radar()
    print("PLOTS DONE")
