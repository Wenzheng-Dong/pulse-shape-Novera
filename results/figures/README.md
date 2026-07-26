# Figures — ansatz-quality weight sweep

These figures ask, for a single-qubit X(π) gate, **which route to the gate is
best across different design priorities**: use a good geometric prior (a closed,
zero-area curve) as is, use the naive constant-amplitude pulse as is, or run a
no-prior optimizer (BARQ). The priorities are encoded as weights on a normalized
cost `C = Σ_i w_i · Ĉ_i`; see `../README.md` for the cost terms and normalization.

All three routes produce an **exact** gate (the good/naive curves by
construction; BARQ via point gate-fixing), so the comparison is purely about
robustness and pulse cost.

Reproduce from a clean clone:

```bash
conda run -n curve python results/scripts/run_sweep.py    # compute -> results/data/*.npz,*.json
conda run -n curve python results/scripts/plot_sweep.py   # data     -> results/figures/*.png
```

(`run_sweep.py` optimizes only the BARQ arm per weight cell — the good/naive
curves are fixed — so it is fast, ~10 min. `plot_sweep.py` only reads saved data.)

## fig1_winner_map.png — best route across weight regimes

Three phase diagrams over weight combinations. Each cell is colored by the route
with the lowest cost:

- **use good ansatz (free)** — wins where **dephasing robustness dominates**: the
  good curve is already robust and needs no optimization.
- **use naive pulse (free)** — wins where **low pulse energy / low peak amplitude
  dominates** and robustness is weighted lightly: the plain pulse is cheapest.
- **run optimizer** — wins the **mixed middle**, where you need both robustness
  and a moderate pulse cost and neither ready-made curve is good enough.

Panels: dephasing-vs-energy (main), amplitude-error-vs-energy, peak-amplitude-vs-energy.

## fig2_when_to_optimize.png — is optimizing worth it?

In three weight regimes, the optimizer's cost vs step (solid), with the two
ready-made curves as horizontal lines (dashed). Where the good-ansatz line sits
below the optimizer's whole trajectory, you should just grab it; where the
optimizer descends below both lines, optimizing pays off.

---
*Data: `../data/sweep_main.{npz,json}`, `sweep_tantrix_energy.*`,
`sweep_maxamp_energy.*` — each stores the per-cell winner, the three route costs,
the BARQ optimum's normalized cost vector, and a few BARQ cost trajectories, plus
the fixed good/naive cost vectors and references in the JSON.*
