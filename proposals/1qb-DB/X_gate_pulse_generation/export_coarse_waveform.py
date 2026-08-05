"""Export the resonant composite X(pi) on an AWG sample grid.

The waveforms shipped in ``waveforms/*.csv`` use a *normalized* time column
(``time_over_Tg`` in [0, 1]) on a deliberately fine 2501-point grid. That grid
is an integration/plotting resolution, not a sample rate: it says nothing about
the gate duration. An AWG needs the same pulse resampled onto its own clock.

This script emits the selected robust candidate on a uniform AWG grid, in
physical units, for a caller-specified gate duration and sample rate. The
default is the 1qb-DB experiment configuration: ``Tg = 100 ns`` at
``1 GSa/s``, i.e. 100 samples.

Why so few samples are enough
-----------------------------
The candidate is an analytic four-segment composite with rotation angles
``(pi, pi, 2pi, pi)``, segment durations proportional to the angles, a
``sin^2`` envelope per segment, and a piecewise-constant phase. Two facts make
a coarse grid exact rather than approximate:

1. Within one segment the phase is constant, so the Hamiltonians at different
   times commute. The segment unitary therefore depends on the drive *only*
   through the accumulated area ``int Omega dt``, not on the envelope shape.
2. A uniform sum of ``sin^2`` over a whole number ``N`` of samples reproduces
   that area exactly, for *any* sampling offset ``a`` in the bin::

       sum_{n=0}^{N-1} sin^2(pi (n + a) / N) = N / 2   (exact, any a)

   because ``sum_n cos(2 pi (n + a) / N) = 0`` for ``N > 1``. This matches
   ``int_0^L sin^2(pi t / L) dt = L / 2``.

So the only real requirement is that every segment span a whole number of
samples. For ``Tg = 100 ns`` at ``1 GSa/s`` the segments are 20/20/40/20
samples, that holds, and the zero-order-hold staircase implements the ideal
composite unitary to machine precision -- for every amplitude error, since a
common error only rescales ``Omega``. (It stays exact down to absurdly coarse
grids; 100 samples is margin against hardware filtering, not against the
math.)

``--sampling`` selects where inside each bin the analytic pulse is read.
``midpoint`` is the default and the one to ship: it gives a time-symmetric
staircase and does not spend a sample on the envelope zero at ``t=0``.
``endpoint`` is retained only for comparison -- by fact 2 it carries the same
area, so it produces the same gate here.

Run from the repository root::

    conda run -n curve python \
        proposals/1qb-DB/X_gate_pulse_generation/export_coarse_waveform.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "waveforms"
COMPOSITE_PARAMS_FILE = OUTPUT / "resonant_composite_robust_params.json"

DEFAULT_GATE_NS = 100.0
DEFAULT_SAMPLE_RATE_GSPS = 1.0


def load_composite_ansatz(path: Path = COMPOSITE_PARAMS_FILE):
    """Return ``(angles, phases)`` of the saved composite candidate.

    Parameters
    ----------
    path : Path
        JSON file written by ``generate_waveforms.py``.

    Returns
    -------
    tuple of ndarray
        Per-segment rotation angles [rad] and drive phases [rad].
    """
    with path.open(encoding="utf-8") as handle:
        params = json.load(handle)
    ansatz = params["ansatz"]
    if ansatz["segment_envelope"] != "sin^2":
        raise ValueError(
            f"unsupported envelope {ansatz['segment_envelope']!r}; "
            "this exporter assumes sin^2 segments"
        )
    if float(ansatz["detuning"]) != 0.0:
        raise ValueError("this exporter assumes a strictly resonant pulse")
    angles = np.asarray(ansatz["rotation_angles_rad"], dtype=float)
    phases = np.asarray(ansatz["phases_rad"], dtype=float)
    return angles, phases


def sample_composite(angles, phases, gate_ns, sample_rate_gsps,
                     sampling="midpoint"):
    """Sample the composite pulse on a uniform AWG grid, in physical units.

    Parameters
    ----------
    angles, phases : array_like
        Per-segment rotation angles and drive phases [rad].
    gate_ns : float
        Total gate duration :math:`T_g` [ns].
    sample_rate_gsps : float
        AWG sample rate [GSa/s]; the sample period is its reciprocal [ns].
    sampling : {'midpoint', 'endpoint'}
        Where inside each sample bin the analytic pulse is evaluated. A
        zero-order-hold AWG holds one value across a whole bin, so 'midpoint'
        is the natural (and time-symmetric) choice. Both conventions carry the
        exact segment area -- see the module docstring -- so 'endpoint' exists
        for comparison, not because it is wrong.

    Returns
    -------
    dict
        ``t_ns`` (bin start times), ``omega`` (envelope, rad/ns),
        ``phase`` (rad), ``omega_x`` / ``omega_y`` (quadratures, rad/ns),
        ``dt_ns``, and ``segment_samples``.

    Notes
    -----
    ``omega`` is the angular Rabi rate: the drive Hamiltonian is
    :math:`H = \\tfrac12 [\\Omega_x(t) X + \\Omega_y(t) Y]`, so a segment
    rotates by :math:`\\int \\Omega\\,dt`. The Rabi *frequency* in GHz is
    :math:`\\Omega / 2\\pi`.
    """
    angles = np.asarray(angles, dtype=float)
    phases = np.asarray(phases, dtype=float)

    dt_ns = 1.0 / sample_rate_gsps
    n_total = gate_ns * sample_rate_gsps
    if abs(n_total - round(n_total)) > 1e-9:
        raise ValueError(
            f"gate of {gate_ns} ns is not a whole number of samples at "
            f"{sample_rate_gsps} GSa/s"
        )
    n_total = int(round(n_total))

    # Segment durations are proportional to the rotation angles. Require them
    # to land on sample boundaries, otherwise the exact-area argument fails.
    fractions = angles / angles.sum()
    seg_samples_float = fractions * n_total
    seg_samples = np.round(seg_samples_float).astype(int)
    if np.any(np.abs(seg_samples_float - seg_samples) > 1e-9):
        raise ValueError(
            "segment boundaries do not fall on sample boundaries: "
            f"{seg_samples_float} samples per segment. Choose a gate/rate "
            "pair whose sample count is divisible accordingly."
        )
    if seg_samples.sum() != n_total:
        raise ValueError("segment sample counts do not sum to the gate length")

    offset = 0.5 if sampling == "midpoint" else 0.0
    if sampling not in ("midpoint", "endpoint"):
        raise ValueError(f"unknown sampling convention {sampling!r}")

    omega = np.empty(n_total)
    phase = np.empty(n_total)
    start = 0
    for angle, seg_phase, n_seg in zip(angles, phases, seg_samples):
        seg_ns = n_seg * dt_ns
        # Peak angular rate of a sin^2 segment carrying `angle` in `seg_ns`.
        peak = 2.0 * angle / seg_ns
        local = (np.arange(n_seg) + offset) / n_seg
        omega[start:start + n_seg] = peak * np.sin(np.pi * local) ** 2
        phase[start:start + n_seg] = seg_phase
        start += n_seg

    t_ns = np.arange(n_total) * dt_ns
    return {
        "t_ns": t_ns,
        "dt_ns": dt_ns,
        "omega": omega,
        "phase": phase,
        "omega_x": omega * np.cos(phase),
        "omega_y": omega * np.sin(phase),
        "segment_samples": seg_samples,
    }


def write_csv(path: Path, grid, peak_norm=None):
    """Write the AWG table with normalized and physical amplitude columns.

    ``I_norm``/``Q_norm`` are scaled so that the envelope peaks at 1.0, which
    is what an AWG amplitude setting expects; the ``*_MHz`` columns give the
    same waveform as an ordinary Rabi frequency :math:`\\Omega / 2\\pi`.
    """
    omega = grid["omega"]
    peak = float(np.max(np.abs(omega))) if peak_norm is None else peak_norm
    to_mhz = 1e3 / (2.0 * np.pi)  # rad/ns -> MHz
    table = np.column_stack(
        [
            np.arange(len(omega)),
            grid["t_ns"],
            grid["omega_x"] / peak,
            grid["omega_y"] / peak,
            grid["omega_x"] * to_mhz,
            grid["omega_y"] * to_mhz,
            omega * to_mhz,
            grid["phase"],
        ]
    )
    header = (
        "sample_index,t_ns,I_norm,Q_norm,"
        "rabi_x_MHz,rabi_y_MHz,rabi_envelope_MHz,phase_rad"
    )
    np.savetxt(
        path,
        table,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%.6f", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e",
             "%.12e"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-ns", type=float, default=DEFAULT_GATE_NS)
    parser.add_argument(
        "--sample-rate-gsps", type=float, default=DEFAULT_SAMPLE_RATE_GSPS
    )
    parser.add_argument(
        "--sampling", choices=("midpoint", "endpoint"), default="midpoint"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    angles, phases = load_composite_ansatz()
    grid = sample_composite(
        angles, phases, args.gate_ns, args.sample_rate_gsps, args.sampling
    )
    n = len(grid["omega"])
    out = args.out or (
        OUTPUT
        / f"resonant_composite_robust_x_pi_{n}samples_"
          f"{args.sample_rate_gsps:g}GSa.csv"
    )
    write_csv(out, grid)

    area = float(np.sum(grid["omega"]) * grid["dt_ns"])
    print(f"wrote {out} ({n} samples, dt = {grid['dt_ns']} ns)")
    print(f"  segment samples : {grid['segment_samples'].tolist()}")
    print(f"  peak Rabi       : {np.max(grid['omega']) * 1e3 / 2 / np.pi:.4f}"
          " MHz")
    print(f"  total area      : {area / np.pi:.12f} pi  (target 5 pi)")


if __name__ == "__main__":
    main()
