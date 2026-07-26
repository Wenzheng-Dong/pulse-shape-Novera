# Results — robust pulse design via geometric space curves

Curated, shareable results for the *pulse-shape-Novera* project: the
optimization objective we use, the cost terms and their physical meaning, and
the figures/data from the ansatz-quality study. (Internal scratch lives in the
untracked `_dev_logs/`; this folder is tracked and meant to be viewed on GitHub
or after a clone. A fuller `docs/` will supersede parts of this later.)

## 1. The optimization objective

We shape a single control pulse `Omega(t), Phi(t)` for a target gate by
optimizing a geometric space curve. The cost is a weighted sum of terms:

```
C(curve) = sum_i  w_i * Chat_i
```

- `w_i >= 0` are **relative-importance weights** we sweep.
- `Chat_i = C_i / C_i^ref` is each cost term **normalized** by a reference value
  so the weights are comparable (see §3 — the raw `C_i` differ by orders of
  magnitude, so weighting the raw terms would conflate importance with units).

Geometry ↔ control (single-qubit SCQC): the curve is unit speed, so **curvature
= Rabi rate `Omega`** and **torsion = drive-phase rate `dPhi/dt`**; the gate is
set by the curve's boundary conditions, its noise robustness by the curve's
shape.

## 2. Cost terms `C_i` (raw expressions and physical meaning)

| term | raw expression `C_i` | qurveros symbol | physical meaning | robustness order |
|---|---|---|---|---|
| gate error   | `1 - F_avg` (adjoint fidelity) | `calculate_adj_fidelity` | target gate is realized | — |
| closure      | `|r(T) - r(0)|^2` | closed-curve endpoints | static dephasing (δz) robustness | **1st** |
| curve area   | `(signed enclosed area)^2` | `curve_zero_area_loss` | static dephasing robustness | **2nd** |
| tantrix area | `(tangent-indicatrix signed area)^2` | `tantrix_zero_area_loss` | multiplicative amplitude / Rabi error `Ω→(1+ε)Ω` | 1st |
| peak amplitude | `T_g · Ω_max` | `max_amp_loss` | peak drive (hardware / leakage ceiling) | — |
| pulse energy | `∫ Ω^2 dt` | `pulse_energy_loss` (ours) | pulse power → leakage proxy (perturbative leakage ~ Ω²/Δ) | — |
| pulse area (L1) | `∫ |Ω| dt` | `total_curvature` | total pulse area | — |
| gate time    | `T_g` | `total_time_loss` | gate duration | — |
| CFI          | curve-filtering index | `cfi_value_loss` | 1/f² time-correlated dephasing | — |

Notes
- **Dephasing robustness** = closure (1st order) + curve area (2nd order). These
  are exactly the conditions a *good* ansatz already satisfies by construction.
- **Leakage** is not modeled in the 2-level picture; `pulse energy` / `peak
  amplitude` / `pulse area` are **proxies**. A true 3-level (Duffing) leakage
  test is a later, separate validation step.
- **Terms in the weight sweep (fixed 2026-07-25):** gate error (locked, large
  fixed weight — not swept), closure, curve area, pulse energy, tantrix area,
  peak amplitude. The remaining terms (pulse area, gate time, CFI) are recorded
  but not swept in this study.

## 3. On scaling / normalization (important)

The raw terms live at very different magnitudes for a typical X(π) curve, e.g.

| term | typical raw magnitude |
|---|---|
| curve area (dephasing) | ~1e-2 (open) … 1e-5 (good) |
| tantrix area | ~1e1 – 1e2 |
| peak amplitude | ~3 – 30 |
| pulse energy | ~1e1 – 1e3 |

Weighting the **raw** terms would make `w_i` a mix of "importance" and "unit
scale". We therefore optimize **normalized** terms `Chat_i = C_i / C_i^ref`, so a
`w_i` sweep is a clean **importance** sweep. Both the raw `C_i` and the chosen
`C_i^ref` are reported alongside every result.

### The reference `C_i^ref`: the *naive-pulse* baseline (fair by construction)

For a fair comparison the reference must **not** come from any of the three
candidates (good / bad / barq) — otherwise that candidate would sit at
`Chat_i = 1` by construction. We use a **fourth, fixed curve**, scored
identically for all three:

> **`C_i^ref` = the value of term `i` on the *naive pulse*** — the constant-
> amplitude resonant pulse of area Θ that implements the target `X(Θ)`,
> geometrically the **circular arc of turning angle Θ** (a semicircle for X(π)),
> at the common gate time `T_g`.

Why this is the fair, defensible reference:

1. **Ansatz-independent.** Computed once, applied the same way to all candidates
   — nobody gets a home-field `Chat = 1`.
2. **Physically canonical.** It is the "zero-cleverness" pulse: the simplest
   thing that still does the gate. So `Chat_i < 1` reads as *"better than the
   naive pulse"* and `Chat_i > 1` as *"worse"* — legible without project jargon.
3. **No division by zero.** The naive arc is **open** and encloses area, so its
   closure, curve area, tantrix area, energy and peak amplitude are all finite
   and nonzero → every `Chat_i` is well defined. (A *closed* good ansatz cannot
   serve as the reference: its closure ≈ 0 would blow up the ratio.)

Two deliberate exceptions / notes, surfaced not hidden:

- **Gate error is the exception.** The naive pulse itself realizes the gate
  (`1 - F ≈ 0`), so it cannot normalize its own gate error. We use a fixed
  infidelity tolerance **`ε_ref = 1e-4`**: `Chat_gate = (1 - F) / 1e-4` (= 1 at
  99.99% fidelity). The gate weight is kept **large / locked**, not swept — gate
  fidelity is a constraint, not an importance knob.
- **The naive arc coincides with the *bad* ansatz (`circle_arc`) at
  initialization.** This is intentional: the bad ansatz simply *starts at the
  baseline* (`Chat ≡ 1`), and we watch where the optimizer can take it. Sitting
  at 1 is neither reward nor penalty; the *good* ansatz starts **below 1 on
  dephasing** (its genuine advantage) and **above 1 on energy** (its genuine
  cost) — symmetric and honest.

All cost terms are compared at a **fixed gate time `T_g`** (scale gauge frozen;
see `CLAUDE.md`), and the naive baseline is evaluated at the same `T_g`.

## 4. Figures

Curated figures for the ansatz-quality study live in `results/figures/` and are
regenerated by tracked scripts (so anyone can reproduce them from a clean clone):

```bash
conda run -n curve python results/scripts/run_sweep.py    # optimize -> results/data/
conda run -n curve python results/scripts/plot_sweep.py   # data     -> results/figures/
```

- `fig1_winner_map.png` — best route to the gate across weight regimes (use the
  good ansatz as is / use the naive pulse as is / run the optimizer).
- `fig2_when_to_optimize.png` — the optimizer's cost vs the two ready-made curves,
  showing when optimizing is worth it and when a free ansatz already wins.

Each figure is labeled for an outside reader (no project jargon). See
`results/figures/README.md` for the full reading of each.

## 5. The three routes compared

Each is a way to obtain an X(π) pulse; all produce an exact gate.

- **good ansatz (used as is)**: `rcp_lemniscate` — closed + zero-area, natively
  implements the target X-rotation and is already dephasing-robust. No
  optimization: its value is being usable directly.
- **naive pulse (used as is)**: the open constant-amplitude arc (`circle_arc`) —
  implements the gate at minimal energy/peak but is not dephasing-robust. This is
  also the normalization reference (§3), so its cost vector is `Ĉ_i ≡ 1`.
- **optimizer (BARQ)**: qurveros' automated control-point optimizer, run under the
  cell's weights; the gate is hard-fixed by point gate-fixing (`F ≡ 1`), so
  optimization only shapes robustness/cost. This is the "no prior, just optimize"
  baseline.

> A note on scope: optimizing *starting from* the good/naive curves (in raw Bézier
> space) is gate-fragile — the optimizer falls into degenerate wrong-gate minima
> that no finite gate penalty prevents. So the good and naive curves are compared
> **as is** (their whole point is being ready-made), and BARQ's PGF is the arm
> that optimizes with an exact gate. See `_dev_logs/step11d_sweep.md`.
