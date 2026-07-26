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

### What the BARQ optimizer guarantees *by construction* vs what the cost must buy

Not every term above needs to be optimized: BARQ's parameterization already
satisfies some of them for free. Verified on fresh (un-optimized) random BARQ
curves (3 seeds):

| property | built in? | evidence (fresh BARQ) |
|---|---|---|
| **gate / rotation angle** (`F = 1`) | **YES** — point gate-fixing (PGF) hard-locks it for any parameters | TTC gate fidelity = 1.00000 |
| **closure = 1st-order dephasing robustness** | **YES** — the curve is forced to start and end at the origin | closure `Ĉ ≡ 0`, endpoints at origin |
| smooth pulse on/off (zero endpoint curvature) | **YES** — PGF vanishing-envelope condition | — |
| **curve area = 2nd-order dephasing robustness** | **NO — must be in the cost** | `Ĉ ≈ 5e-3–1e-2` (varies with seed); only → 0 when `curve_zero_area` is weighted |
| **tantrix area = amplitude / Rabi-drift robustness** | **NO — must be in the cost** | `Ĉ ≈ 2–6` (varies); step 8: 104 → 7.6e-5 only when weighted |
| pulse energy / peak amplitude / leakage proxy | **NO — must be in the cost** | unconstrained; energy runs to ~2900× if never penalized |

So for BARQ, gate fidelity and 1st-order dephasing are **not tradeoffs** — they
come for free. Weights only ever need to be spent on **2nd-order dephasing,
amplitude-error robustness, and the pulse-cost/leakage terms**. (Caveat: PGF
guarantees the *geometric* gate; the *physical* TTC gate can still degrade for
extreme optimized curves — always check `barq_gate_fidelity`. See
`_dev_logs/LESSONS.md` §2.)

> A *good geometric ansatz* (e.g. rcp_lemniscate) satisfies closure AND curve
> zero-area by construction — i.e. it comes with 2nd-order dephasing robustness
> too, which BARQ does not. That extra built-in robustness is the ansatz's
> advantage; but note it is only realized if the ansatz is carried *faithfully*
> (raw-Bézier), not through BARQ seeding, which mangles it (§5 / LESSONS §3).

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

Curated figures live in `results/figures/`, regenerated by tracked scripts (so
anyone can reproduce them from a clean clone):

```bash
conda run -n curve python results/scripts/run_sweep.py    # optimize -> results/data/ (+ local history)
conda run -n curve python results/scripts/plot_sweep.py   # data     -> results/figures/
```

The experiment (step 11f): for every weight combination, the SAME objective is
optimized from several **warm-starts** — four structured seeds (rcp, and three
open arcs: circle/triangle/gaussian, each seeded into BARQ) and a random-default
**ensemble** — all with the gate held exact by BARQ's point gate-fixing. We ask
which starting point BARQ reaches the best solution from.

- `fig1_warmstart_vs_random.png` — final optimized cost from each structured
  warm-start vs the random-default ensemble (band), across weight regimes.
- `fig2_convergence.png` — cost vs step from the structured warm-starts and the
  random ensemble, in two regimes.

> Honest scope: BARQ's gate-fixing does NOT faithfully preserve a seeded curve —
> it mangles it (e.g. rcp's pulse energy inflates ~18×) and can invert the
> good/bad ordering. So this measures **warm-start quality inside BARQ**, not
> ansatz quality. The faithful ansatz comparison needs a raw-Bézier carrier (see
> `_dev_logs/step11d_sweep.md`).

Each figure is labeled for an outside reader (no project jargon). See
`results/figures/README.md` for the full reading of each.

## 5. The warm-starts compared (all optimized, gate exact)

Every run optimizes the same objective; the starts differ only in the initial
control points. All are seeded into BARQ, whose point gate-fixing keeps the gate
exact (`F ≡ 1`) throughout. Energy weight is always ≥ 0.001 (a physical-realism
floor that prevents the pure-dephasing energy runaway).

- **structured warm-starts**: `rcp_lemniscate` (closed) and three open arcs
  (`circle_arc`, `triangle_pulse`, `gaussian_arc`), each fitted and seeded into
  BARQ's free points.
- **random default (ensemble)**: 5 random BARQ initializations — the no-prior
  baseline as a *distribution*, not a single (possibly unlucky) draw.

What each seed becomes inside BARQ (pre-optimization; the mangling is real and
reported): rcp → pulse energy ~608× (curve_area preserved ~1e-4); the open arcs →
~23–33×. So BARQ does not preserve the ansatz — the closed "good" rcp actually
maps to the *highest-energy* start. This is why the study is framed as warm-start
quality, not ansatz quality.

Solution library: `struct_free_points` in `results/data/sweep_initquality.npz`
stores the optimized control parameters for every weight combination and
warm-start — rebuild the pulse with `sweep.make_seeded_barq(free_points)`.

> For the faithful ansatz-quality comparison (which distinguishes the open arcs by
> their true energy 1.0/1.33/1.58 and rcp's 33), use a raw-Bézier carrier — it
> reproduces rcp to RMS 7.7e-4, whereas BARQ mangles it to RMS 0.23. See
> `_dev_logs/step11d_sweep.md`.
