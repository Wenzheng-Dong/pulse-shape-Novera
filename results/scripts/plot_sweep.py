"""Step 11d -- render the curated ansatz-quality figures from swept data (PLOT).

Reads ``results/data/`` (from ``run_sweep.py``) and writes two outsider-legible
figures to ``results/figures/``. No optimization here, so figures re-render fast.

  fig1_winner_map.png
      Three phase diagrams over weight combinations. Each cell is colored by the
      best route to the gate: use the good ansatz as is, use the naive pulse as
      is, or run the optimizer. Good wins where dephasing robustness dominates;
      the naive pulse wins where low energy / low peak dominates and robustness
      is cheap; optimization wins the mixed middle where no ready-made curve is
      good enough.

  fig2_when_to_optimize.png
      In three weight regimes, the optimizer's cost vs step, with the two
      ready-made curves as horizontal lines. Shows whether (and after how many
      steps) optimizing beats simply grabbing the good or the naive pulse.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")

# 0 = good ansatz, 1 = naive pulse, 2 = optimize (BARQ).
ROUTE_COLORS = ["#2ca02c", "#ff7f0e", "#1f77b4"]
ROUTE_LABELS = ["use good ansatz (free)", "use naive pulse (free)", "run optimizer"]

AXIS_NAME = {
    "dephasing": "dephasing-robustness weight",
    "energy": "pulse-energy weight  (leakage proxy)",
    "tantrix": "amplitude-error-robustness weight",
    "max_amp": "peak-amplitude weight",
}


def _load(name):
    d = np.load(os.path.join(DATA, f"sweep_{name}.npz"))
    meta = json.load(open(os.path.join(DATA, f"sweep_{name}.json")))
    return d, meta


def draw_winner(ax, name, title):
    d, meta = _load(name)
    win = d["winner"]
    xs, ys = meta["xs"], meta["ys"]
    ax.pcolormesh(np.arange(len(xs) + 1), np.arange(len(ys) + 1), win,
                  cmap=ListedColormap(ROUTE_COLORS), vmin=0, vmax=2,
                  edgecolors="white", linewidth=0.6)
    ax.set_xticks(np.arange(len(xs)) + 0.5); ax.set_xticklabels([f"{v:g}" for v in xs], fontsize=8)
    ax.set_yticks(np.arange(len(ys)) + 0.5); ax.set_yticklabels([f"{v:g}" for v in ys], fontsize=8)
    ax.set_xlabel(AXIS_NAME.get(meta["axis_x"], meta["axis_x"]), fontsize=9)
    ax.set_ylabel(AXIS_NAME.get(meta["axis_y"], meta["axis_y"]), fontsize=9)
    ax.set_title(title, fontsize=10)


def fig_winner_maps():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    draw_winner(axes[0], "main", "Dephasing vs pulse energy")
    draw_winner(axes[1], "tantrix_energy", "Amplitude-error vs pulse energy")
    draw_winner(axes[2], "maxamp_energy", "Peak amplitude vs pulse energy")
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(ROUTE_COLORS, ROUTE_LABELS)],
               loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Best route to the gate across weight regimes\n(the good geometric "
                 "prior is the right free choice only where dephasing robustness "
                 "dominates)", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(FIGS, "fig1_winner_map.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)


def fig_when_to_optimize():
    _, meta = _load("main")
    traj = meta["traj"]
    keys = sorted(traj.keys(), key=lambda k: (traj[k]["vy"], traj[k]["vx"]))
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.3), sharey=False)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        t = traj[key]
        ax.semilogy(t["steps"], np.maximum(t["barq_cost"], 1e-12), "-", color=ROUTE_COLORS[2],
                    lw=2, label="run optimizer (BARQ)")
        ax.axhline(max(t["c_good"], 1e-12), color=ROUTE_COLORS[0], ls="--", lw=1.8,
                   label="use good ansatz (free)")
        ax.axhline(max(t["c_naive"], 1e-12), color=ROUTE_COLORS[1], ls="--", lw=1.8,
                   label="use naive pulse (free)")
        ax.set_title(f"dephasing wt = {t['vx']:g},  energy wt = {t['vy']:g}", fontsize=10)
        ax.set_xlabel("optimization step", fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("total weighted cost  (lower = better)", fontsize=9)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("When is optimizing worth it? Optimizer cost vs the two ready-made "
                 "curves\n(dephasing-dominated: the good ansatz already wins for free; "
                 "mixed: optimizing pays off)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig2_when_to_optimize.png")
    fig.savefig(out, dpi=140); print("saved", out)


if __name__ == "__main__":
    fig_winner_maps()
    fig_when_to_optimize()
    print("PLOTS DONE")
