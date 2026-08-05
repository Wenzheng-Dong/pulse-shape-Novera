"""Verify that the AWG-rate export reproduces the ideal composite X(pi).

The 1qb-DB experiment drives at 1 GSa/s, so a 100 ns gate is 100 samples --
far coarser than the 2501-point normalized grid in ``waveforms/*.csv``. This
script checks that nothing is lost by treating the coarse table as a
zero-order-hold (staircase) waveform, which is what an AWG actually plays.

Checks, in order:

1. accumulated rotation angle per segment, for midpoint and endpoint sampling;
2. the staircase propagator against the analytic four-rotation composite;
3. average gate fidelity across the +/-3% amplitude-error grid;
4. the 60-cycle drive-back population trace the pulse was optimized for;
5. how far the sample rate can be lowered before the gate actually degrades.

Output is a printed table only; no figures are produced.

Everything here is a 2x2 propagator product, so no simulator dependency is
needed; ``generate_waveforms.py`` remains the qutip-based cross-check on the
fine grid.

Run from the repository root::

    conda run -n curve python \
        proposals/1qb-DB/X_gate_pulse_generation/verify_coarse_waveform.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from export_coarse_waveform import (  # noqa: E402
    COMPOSITE_PARAMS_FILE,
    load_composite_ansatz,
    sample_composite,
)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
N_CYCLES = 60
EPS_GRID = np.linspace(-0.03, 0.03, 13)


def rot(angle, phase):
    """Resonant rotation by ``angle`` about the equatorial axis at ``phase``."""
    axis = np.cos(phase) * X + np.sin(phase) * Y
    return np.cos(angle / 2) * I2 - 1j * np.sin(angle / 2) * axis


def analytic_unitary(angles, phases, eps):
    """Ideal composite: one exact rotation per segment, amplitude error eps."""
    u = I2.copy()
    for a, p in zip(angles, phases):
        u = rot(a * (1 + eps), p) @ u
    return u


def zoh_unitary(grid, eps):
    """Exact propagator of the piecewise-constant waveform an AWG plays."""
    dt = grid["dt_ns"]
    u = I2.copy()
    for om, ph in zip(grid["omega"], grid["phase"]):
        u = rot(om * (1 + eps) * dt, ph) @ u
    return u


def avg_gate_fidelity(u, target):
    """Average gate fidelity for a qubit (d=2)."""
    return (abs(np.trace(target.conj().T @ u)) ** 2 + 2.0) / 6.0


def db_trace(u, n=N_CYCLES):
    """Ground-state population after each drive-back cycle (a gate pair)."""
    cycle = u @ u
    state = np.array([1.0, 0.0], dtype=complex)
    vals = [1.0]
    for _ in range(n):
        state = cycle @ state
        vals.append(abs(state[0]) ** 2)
    return np.asarray(vals)


angles, phases = load_composite_ansatz(COMPOSITE_PARAMS_FILE)
target = rot(np.pi, 0.0)

grids = {
    "coarse-midpoint (100 @ 1GSa/s)": sample_composite(
        angles, phases, 100.0, 1.0, "midpoint"),
    "coarse-endpoint (100 @ 1GSa/s)": sample_composite(
        angles, phases, 100.0, 1.0, "endpoint"),
    "fine reference (100000 @ 1000GSa/s)": sample_composite(
        angles, phases, 100.0, 1000.0, "midpoint"),
}

report = {}
print("=" * 72)
print("1. rotation area per segment")
for name, g in grids.items():
    seg = g["segment_samples"]
    edges = np.concatenate(([0], np.cumsum(seg)))
    areas = [float(np.sum(g["omega"][edges[i]:edges[i + 1]]) * g["dt_ns"])
             for i in range(len(seg))]
    tgt = angles
    print(f"  {name}")
    print(f"    areas/pi = {[round(a / np.pi, 12) for a in areas]}"
          f"   target {[round(a / np.pi, 3) for a in tgt]}")
    print(f"    max |dev| = {max(abs(np.array(areas) - tgt)):.3e} rad")
    report.setdefault(name, {})["segment_area_max_dev_rad"] = float(
        max(abs(np.array(areas) - tgt)))

print("=" * 72)
print("2. ZOH unitary vs analytic composite unitary (Frobenius dist, eps=0)")
for name, g in grids.items():
    d = np.linalg.norm(zoh_unitary(g, 0.0) - analytic_unitary(angles, phases, 0.0))
    print(f"  {name:38s} ||dU|| = {d:.3e}")
    report[name]["unitary_dist_vs_analytic"] = float(d)

print("=" * 72)
print("3. average gate fidelity vs amplitude error")
fid = {}
for name, g in grids.items():
    fid[name] = np.array([avg_gate_fidelity(zoh_unitary(g, e), target)
                          for e in EPS_GRID])
fid["analytic composite"] = np.array(
    [avg_gate_fidelity(analytic_unitary(angles, phases, e), target)
     for e in EPS_GRID])
hdr = "  eps    " + "".join(f"{k.split(' (')[0][:22]:>24s}" for k in fid)
print(hdr)
for i, e in enumerate(EPS_GRID):
    print(f"  {e:+.3f}" + "".join(f"{fid[k][i]:>24.12f}" for k in fid))
for k, v in fid.items():
    report.setdefault(k, {})["min_fidelity_over_eps"] = float(v.min())

print("=" * 72)
print(f"4. DB {N_CYCLES}-cycle population, P0 statistics")
traces = {}
for e in (-0.03, 0.0, 0.03):
    for name, g in grids.items():
        tr = db_trace(zoh_unitary(g, e))
        traces[(name, e)] = tr
        print(f"  eps={e:+.2f}  {name:38s} "
              f"P0_final={tr[-1]:.9f}  dP0={np.ptp(tr):.3e}")
    tr = db_trace(analytic_unitary(angles, phases, e))
    traces[("analytic composite", e)] = tr
    print(f"  eps={e:+.2f}  {'analytic composite':38s} "
          f"P0_final={tr[-1]:.9f}  dP0={np.ptp(tr):.3e}")
    print()

print("=" * 72)
print("5. margin: how coarse can the grid get before the gate degrades?")
print("   (segments are 20/20/40/20% of Tg, so N must be a multiple of 5)")
print(f"  {'N samples':>10s} {'rate GSa/s':>11s} {'peak MHz':>10s} "
      f"{'||dU||':>11s} {'1-F(eps=3%)':>14s} {'dP0(60,3%)':>12s}")
margin = {}
to_mhz = 1e3 / 2 / np.pi
for n_s in (5, 10, 20, 25, 50, 100, 200):
    g = sample_composite(angles, phases, 100.0, n_s / 100.0, "midpoint")
    u0 = zoh_unitary(g, 0.0)
    d = np.linalg.norm(u0 - analytic_unitary(angles, phases, 0.0))
    u3 = zoh_unitary(g, 0.03)
    inf3 = 1 - avg_gate_fidelity(u3, target)
    dp = np.ptp(db_trace(u3))
    margin[n_s] = dict(unitary_dist=float(d), infidelity_eps3=float(inf3),
                       dP0=float(dp))
    print(f"  {n_s:>10d} {n_s / 100:>11.2f} "
          f"{np.max(g['omega']) * to_mhz:>10.3f} "
          f"{d:>11.2e} {inf3:>14.3e} {dp:>12.3e}")
report["coarse_grid_margin"] = margin
