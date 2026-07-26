# Figures — BARQ warm-start quality

For a single-qubit X(π) gate, the same robustness objective `C = Σ_i w_i · Ĉ_i`
is optimized from several **warm-starts**, all with the gate held exact by BARQ's
point gate-fixing:

- four **structured** seeds — `rcp_lemniscate` (closed) and three open arcs
  (`circle`, `triangle`, `gaussian`), each seeded into BARQ;
- a **random-default ensemble** (5 random BARQ inits) — the no-prior baseline as
  a distribution.

Energy weight is always ≥ 0.001 (a physical floor that prevents the
pure-dephasing energy runaway). See `../README.md` for the cost terms and the
important honest caveat: BARQ mangles a seeded curve, so this is **warm-start
quality**, not ansatz quality.

Reproduce from a clean clone:

```bash
conda run -n curve python results/scripts/run_sweep.py    # optimize -> data/ (+ local full history)
conda run -n curve python results/scripts/plot_sweep.py   # data     -> figures/
```

## fig1_warmstart_vs_random.png — which start reaches the best solution

For each weight regime, the final optimized cost of each structured warm-start
(colored markers) against the random-default ensemble (gray band = min–max, tick
= median). A marker below the band means that warm-start beats optimizing from
scratch.

## fig2_convergence.png — convergence traces

Cost vs optimization step from the four structured warm-starts and the random
ensemble (shaded band), in two regimes (dephasing-dominated and energy-weighted).

---
*Data: `../data/sweep_initquality.{npz,json}` — per-cell final cost, final
normalized costs, final gate fidelity (qutip), and optimized control parameters
(`struct_free_points`) for every warm-start; a few convergence traces in the
JSON. Full per-25-step history is in `_dev_logs/sweep_history/` (untracked).*
