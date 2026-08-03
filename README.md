# pulse-shape-Novera

Robust single-qubit gate pulses designed as **geometric space curves** (SCQC) and
optimized with [`qurveros`](https://github.com/evpiliouras/qurveros) (JAX).

The point of this README is one question that every reader asks first:

> **In a qurveros optimization, what is held fixed and what is actually varied?**

Sections [2](#2-locked-vs-optimized--the-core-tables) and
[3](#3-the-cost-terms-that-are-optimized) answer it with explicit expressions.
Cost-term rationale, normalization policy and the figures live in
[`results/README.md`](results/README.md); the research plan lives in `_plan.md`.

---

## 1. Geometry ↔ control dictionary

A **unit-speed** space curve $\mathbf{r}(s)\in\mathbb{R}^3$, $s\in[0,L]$, maps to a
single-qubit drive $H(t)=\tfrac{\Omega(t)}{2}\big[\cos\Phi\,\sigma_x+\sin\Phi\,\sigma_y\big]+\tfrac{\Delta(t)}{2}\sigma_z$:

| curve object | control object | expression |
|---|---|---|
| curvature $\kappa(s)$ | Rabi rate | $\Omega(t)=\kappa(t)$ |
| torsion $\tau(s)$ | drive-phase rate | $\dot\Phi(t)=\tau(t)$ |
| total arc length $L$ | gate time | $T_g=L$ |
| boundary data $\{\mathbf{r},\mathbf{T},\mathbf{B}\}$ at $s=0,L$ | the gate $U(T_g)$ | fixed by construction (see §2) |
| closure $\mathbf{r}(L)=\mathbf{r}(0)$ | 1st-order dephasing robustness | $\int_0^{T_g}\!\hat{\mathbf{n}}(t)\,dt=0$ |
| enclosed area $\oint \mathbf{r}\times d\mathbf{r}$ | 2nd-order dephasing robustness | see §3 |
| tantrix area $\oint \mathbf{T}\times d\mathbf{T}$ | amplitude-error robustness ($\Omega\to(1+\epsilon)\Omega$) | see §3 |

Everything below is written in these variables.

---

## 2. Locked vs optimized — the core tables

Our optimizer arm is **BARQ** (*Bezier Ansatz for Robust Quantum control*), i.e.
`qurveros.optspacecurve.BarqCurve` driven by `optimize_barq_robustness`
(`src/pulse_shape_novera/barq.py`) or `optimize_barq` (`sweep.py`). BARQ's key
property: several physical requirements are satisfied **structurally**, by how the
Bezier control points are *built*, not by a penalty that the optimizer has to
trade against anything else.

### 2A. LOCKED — enforced exactly, never traded

| # | locked quantity | mathematical statement | how it is locked | consequence |
|---|---|---|---|---|
| 1 | **target gate** | $U(T_g)=X(\pi)$, i.e. $\mathrm{Adj}\,[U]=\mathrm{Adj}[\sigma_x]$ | point gate-fixing (PGF): the boundary control points are *derived* from the free ones so the gate holds for **any** parameter values (`BarqCurve(adj_target=…)`, `xgate_pgf_mod`) | $1-F\equiv 0$ throughout; no gate-vs-robustness tradeoff |
| 2 | **curve closure** | $\mathbf{r}(0)=\mathbf{r}(T_g)=\mathbf{0}$ | PGF control-point layout `[0, gate_fix, free_points, gate_fix, 0]` | 1st-order dephasing robustness **free** (measured $\hat C_{\rm closure}=0.0$ on a fresh, un-optimized BARQ curve). The closure entry in §3 is therefore **metric-only** for BARQ — it can never be "optimized towards" |
| 3 | **pulse on/off** | $\Omega(0)=\Omega(T_g)=0$ (vanishing endpoint curvature) | PGF vanishing-envelope condition | smooth, hardware-realizable envelope |
| 4 | **scale (gauge)** | $n\equiv$ `pgf_params["norm_value"]` $=0.25$, gradient zeroed | `optax.multi_transform({True: adam, False: set_to_zero})` with `labels["pgf_params"]["norm_value"] = False` | forbids the trivial sink $\mathbf{r}\to 0$ ($\Omega\to0$, "do nothing"); see §4 |
| 5 | **boundary knobs tied to the gauge** | `left_tangent_fix = left_tangent_aux = left_binormal_fix = right_binormal_fix = right_tangent_aux = right_tangent_fix` $= n$, and `left_binormal_aux = right_binormal_aux` | `xgate_pgf_mod` (barq.py) | reduces the PGF boundary d.o.f. to what an $X$ gate needs |
| 6 | **gate time / physical units** | $T_g$ | *not* an optimization variable: every swept cost term is scale-invariant (§4), so $T_g$ is chosen **after** optimization by rescaling; waveforms are exported dimensionless as $T_g\Omega$ vs $t/T_g$ | "fix $T_g$, minimize amplitude" and "fix amplitude, minimize $T_g$" are the *same* optimization |
| 7 | **detuning (resonant arm)** | $\Delta(t)\equiv 0$ | export uses the resonant `XY` control mode; for the TTC-mode candidate the residual detuning is pushed down by a *soft* term (§3, `barq_detuning_loss`) | strictly resonant $x/y$ waveform for the lab |
| 8 | **hyperparameters** | $n_{\rm free}=10$ (waveform generation) / $14$ (weight sweep); Frenet samples $400$–$1200$; Adam $\eta=10^{-3}$; $1200$–$1500$ steps | module constants (`barq.py`, `sweep.py`, `generate_waveforms.py`) | reproducibility; all runs are seeded (`seed=4531469`) |
| 9 | **cost references** $C_i^{\rm ref}$ | the naive constant-amplitude $X(\pi)$ arc, $L=1$ (see §5) | `sweep.compute_references` | weights are *importance*, not unit conversion |

### 2B. OPTIMIZED — the actual free parameters

| variable | symbol | shape | what it controls |
|---|---|---|---|
| internal Bezier control points | `params["free_points"]` $=W$ | $(n_{\rm free},3)$ = $(10,3)$ or $(14,3)$ | the curve shape. Rows `[2:]` **are** the internal control points; rows `[:2]` feed PGF, which derives the gate-fixing points from them |
| BARQ angle | $\theta_B=$ `pgf_params["barq_angle"]` | scalar (init $\pi$) | the frame angle at $x=1$ that sets the TTC control's detuning |
| free binormal aux | `pgf_params["right_binormal_aux"]` (`left_` copies it) | scalar | remaining boundary freedom left open by `xgate_pgf_mod` |

**Total: $3n_{\rm free}+2$ real numbers** (32 for $n_{\rm free}=10$). Nothing else is
varied — in particular *not* the gate, *not* the closure, *not* the scale.

---

## 3. The cost terms

The objective is a weighted sum of **normalized** terms

$$C(\text{curve})=\sum_i w_i\,\hat C_i,\qquad \hat C_i=\frac{C_i}{C_i^{\rm ref}},\qquad w_i\ge 0 .$$

**Read the "role" column first — it is what reconciles this table with §2A.** The same
term library serves two jobs:

* **loss** — a term that actually drives the optimizer (nonzero gradient w.r.t. the free
  parameters of §2B);
* **metric only** — the identical expression evaluated as a *score*, on curves that are
  **not** produced by PGF: the good/naive ansätze compared in the sweep
  (`sweep.evaluate_chat`) and raw-Bezier carriers. There it is generally nonzero and
  informative.

Two terms are **locked by construction for the BARQ arm** (§2A rows 1–2) and therefore
sit at their optimum with zero gradient — closure and gate error. `run_sweep.py` does
hand a nonzero closure weight to `optimize_barq`, but on a PGF curve
$\hat C_{\rm closure}\equiv0$, so that term contributes exactly nothing to the BARQ loss
or its gradient; it is carried in the objective so that the *same* $C$ can be applied to
the non-PGF candidates, where closure is a real, active cost. Nothing in this repo ever
optimizes a BARQ curve *towards* closure — it starts closed and stays closed.

| term | raw expression | scale-invariant form used | code symbol | role for the BARQ arm | physics it buys | order |
|---|---|---|---|---|---|---|
| closure | $\lVert\mathbf{r}(T_g)-\mathbf{r}(0)\rVert^2$ | $\lVert\Delta\mathbf{r}\rVert^{2}/L^{2}$ | `sweep.closure_term` | **locked (§2A-2) → inert**; metric only | static dephasing $\delta_z$ | 1st |
| gate error | $1-F_{\rm avg}$, normalized by $\varepsilon_{\rm ref}=10^{-4}$ | — | `calculate_adj_fidelity` | **locked (§2A-1) → not in the cost**; metric only | target gate realized | — |
| curve area | $\big\lVert \mathbf{A}_r\big\rVert^2,\ \ \mathbf{A}_r=\frac{1}{L^{2}}\!\int_0^{L}\!\mathbf{r}\times\mathbf{r}'\,ds$ | (see note ▼) | `losses.curve_zero_area_loss` → `sweep.curve_area_term` | **active loss** | static dephasing $\delta_z$ | 2nd |
| tantrix area | $\big\lVert \mathbf{A}_T\big\rVert^2,\ \ \mathbf{A}_T=\int_0^{L}\!\mathbf{T}\times\mathbf{T}'\,ds=\int_0^{T_g}\!\Omega(t)\,\mathbf{B}\,dt$ | already invariant | `losses.tantrix_zero_area_loss` | **active loss** | multiplicative amplitude error $\Omega\to(1+\epsilon)\Omega$ | 1st |
| pulse energy | $E=\int_0^{T_g}\!\Omega^2\,dt=\int \kappa^2 v\,dx$ | $L\cdot E$ | `proxies.pulse_energy_loss` → `sweep.energy_term` | **active loss** | leakage proxy (perturbative leakage $\sim\Omega^2/\Delta_{\rm anh}$) | — |
| peak amplitude | $T_g\,\Omega_{\max}=L\max_s\kappa(s)$ | already invariant | `losses.max_amp_loss` | **active loss** | hardware / leakage ceiling | — |
| TTC detuning | $\delta_{\rm TTC}(\theta_B)^2$ | invariant | `losses.barq_detuning_loss` | **active loss** (refinement stage only) | drives the TTC solution to a **resonant** $x/y$ pulse | — |
| pulse area | $\int_0^{T_g}\!\lvert\Omega\rvert\,dt$ | invariant | `losses.total_curvature_loss` | available, not used | total turning / L1 drive cost | — |
| gate time | $T_g=L$ | — | `losses.total_time_loss` | available, not used (meaningless with the scale frozen — §4) | speed | — |
| CFI | $\int \lVert\mathbf{r}\rVert^{2}\,ds$ (normalized) | — | `losses.cfi_value_loss` | available, not used | $1/f^2$ time-correlated dephasing | — |

**Typical weights actually used** for the exported robust candidate
(`generate_waveforms.py`): $w_{\rm tantrix}=1$, $w_{\rm energy}=0.01$,
$w_{\rm max\,amp}=0.01$, plus $w_{\delta_{\rm TTC}}=0.01$ in a short refinement stage.

> ▼ **Note / known caveat (measured, not hidden).** `qurveros`' `calculate_curve_area`
> **already** divides by $T_g^2$, so `curve_zero_area_loss` $=\lVert\mathbf{A}_r\rVert^2$ is
> scale-invariant on its own. `sweep.curve_area_term` divides by $L^4$ once more, i.e. it
> evaluates $\lVert\int\mathbf{r}\times d\mathbf{r}\rVert^2/L^{8}$ and therefore scales as
> $L^{-4}$ (verified: ×1/16 per doubling, §4). Since $C^{\rm ref}$ is taken at $L=1$ and a
> fresh BARQ curve has $L\approx1.05$, this biases $\hat C_{\rm curve\,area}$ by
> $\approx(1/1.05)^4\approx0.82$ — small, but it means this one term is *not* strictly
> length-neutral across candidates.

### What BARQ gives for free vs what the weights must buy

| property | free? | evidence on a fresh, un-optimized BARQ curve ($n_{\rm free}=10$, default seed) |
|---|:--:|---|
| gate fidelity | ✅ | TTC gate fidelity $=0.999999999998$ ($1-F\approx2\times10^{-12}$) |
| closure (1st-order dephasing) | ✅ | $\hat C_{\rm closure}=0.0$ (exactly) |
| smooth pulse on/off | ✅ | $\Omega(0)=\Omega(T_g)=0$ by PGF |
| curve area (2nd-order dephasing) | ❌ | $\hat C\approx2.5\times10^{-2}$; falls orders of magnitude only when weighted |
| tantrix area (amplitude robustness) | ❌ | $\hat C\approx1.1\times10^{1}$; $\to\ll1$ only when weighted |
| pulse energy / peak amplitude | ❌ | $\hat C\approx1.9\times10^{2}$ / $3.6\times10^{2}$ — runs away if never penalized |

So weights are only ever spent on **2nd-order dephasing, amplitude-error robustness,
and the pulse-cost/leakage terms**. (Caveat: PGF guarantees the *geometric* gate; always
re-check the *physical* gate with `barq_gate_fidelity` after optimization.)

---

## 4. Why the scale is locked (it is a gauge freedom)

Under a uniform rescaling $\mathbf{r}\to s\,\mathbf{r}$:

$$L\to sL,\qquad \kappa\to\kappa/s,\qquad \Omega\to\Omega/s,\qquad T_g\to sT_g,$$

so $T_g\Omega_{\max}$, $\int\Omega\,dt$, $\mathbf{A}_T$, $\mathbf{A}_r/L^2$ and
$\lVert\Delta\mathbf{r}\rVert/L$ are **unchanged**, while $T_g$ and $\Omega$ separately
are not. Two consequences:

1. *"Fix $T_g$, minimize the amplitude"* and *"fix the amplitude, minimize $T_g$"* are the
   **same** optimization (both minimize $T_g\Omega_{\max}$). Physical units are set by one
   final rescaling.
2. If the scale is left free while a $T_g$-type cost is minimized, the optimizer slides
   down the gauge direction to $\Omega\equiv0$. `optimize.optimize_gate_time` demonstrates
   exactly this collapse and its cure (a $-\log_{10}\omega$ anchor); BARQ instead simply
   **freezes** `norm_value`.

**Numerical check** (semicircle $X(\pi)$, radius scaled by $s$; script:
`sweep` terms evaluated at $s=1,2,4$):

| $s$ | $L$ | closure | curve area | energy | tantrix | max amp |
|---|---|---|---|---|---|---|
| 1 | 3.1416 | 4.0528e-01 | 1.0402e-03 | 9.8696 | 9.8696 | 3.1416 |
| 2 | 6.2832 | 4.0528e-01 | 6.5010e-05 | 9.8696 | 9.8696 | 3.1416 |
| 4 | 12.566 | 4.0528e-01 | 4.0631e-06 | 9.8696 | 9.8696 | 3.1416 |

Four of five terms are invariant to all printed digits; `curve_area` falls as $L^{-4}$ —
the over-normalization flagged in §3.

---

## 5. The normalization reference $C_i^{\rm ref}$

$C_i^{\rm ref}$ is evaluated on a **fourth, fixed curve** — the naive constant-amplitude
resonant $X(\pi)$ pulse, geometrically the unit-speed semicircle of arc length $L=1$ and
curvature $\kappa=\pi$. It belongs to none of the compared candidates, so nobody gets a
home-field $\hat C_i=1$, and all its values are analytic:

| term | analytic $C_i^{\rm ref}$ | value | computed |
|---|---|---|---|
| closure | $4/\pi^2$ | 0.405285 | 0.405285 |
| curve area | $1/\pi^2$ | 0.101321 | 0.101321 |
| energy | $\pi^2$ | 9.869604 | 9.869604 |
| tantrix area | $\pi^2$ | 9.869604 | 9.869604 |
| peak amplitude | $\pi$ | 3.141593 | 3.141593 |

Agreement to all printed digits is a standing sanity check on the whole cost stack. Full
policy discussion: [`results/README.md`](results/README.md) §3.

---

## 6. The second optimizer: the resonant composite arm

The `1qb-DB` experiment also ships a non-geometric candidate — a BB1-like four-segment
composite pulse optimized directly against $\pm3\%$ amplitude error over 60 dressed-basis
cycles (`proposals/1qb-DB/X_gate_pulse_generation/generate_waveforms.py`). Same question,
different answer:

| | quantity |
|---|---|
| **locked** | segment rotation angles $(\pi,\pi,2\pi,\pi)$; phase pattern $(0,\varphi_1,\varphi_2,\varphi_1)$; $\sin^2$ envelope; segment durations $\propto$ angle; $\Delta\equiv0$; error grid $\epsilon\in\{-3\%,\dots,+3\%\}$ (13 points); $N_{\rm cycles}=60$ |
| **optimized** | **two scalars** $\varphi_1(=\varphi_3)$ and $\varphi_2$ |
| **objective** | $\max_\epsilon\big[1-\min_k P_0^{(k)}(\epsilon)\big]\;+\;10\cdot\max_\epsilon\big[1-F(\epsilon)\big]$ (minimax, `differential_evolution` → Nelder–Mead) |

---

## 7. Repository map

| path | contents |
|---|---|
| `src/pulse_shape_novera/` | `ansatz.py` (curve families), `bezier.py` (ansatz → Bezier), `barq.py` (BARQ + gate fidelity), `proxies.py` (pulse-energy loss), `sweep.py` (scale-invariant terms, references, weight sweep), `optimize.py` (scale-gauge demo), `_patches.py` (safe `vector_norm`: fixes NaN gradients for **planar** curves) |
| `results/` | curated figures + data + regeneration scripts + the cost-term/normalization write-up |
| `proposals/1qb-DB/` | dressed-basis experiment: waveform generation, exported CSVs, demo notebook |
| `tests/` | physics-level regression tests (analytic families, gate fidelity, scale gauge, planar-NaN fix, Bezier reconstruction) |
| `CLAUDE.md`, `_plan.md` | working conventions / research plan |

Environment (`environment.yml`, conda env `curve`, Python ≥ 3.10):

```bash
conda run -n curve pytest                                       # regression tests
conda run -n curve python results/scripts/run_sweep.py          # optimize → results/data/
conda run -n curve python results/scripts/plot_sweep.py         # data → results/figures/
```

`import pulse_shape_novera` applies the qurveros patches automatically — do it before
anything else touches qurveros.
