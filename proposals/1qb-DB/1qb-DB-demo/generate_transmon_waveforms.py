"""Optimize resonant, amplitude-robust X(pi) candidates for a transmon.

The waveforms exported by ``X_gate_pulse_generation`` were selected for the
two-level DB experiment, where peak drive costs nothing.  On a transmon it
costs leakage, and the gauge choice of :mod:`db_transmon` turns a large
normalized peak ``Tg*Omega_max`` into a long gate.  This script re-runs the
BARQ arm to look for a *resonant* pulse that is genuinely first-order robust to
amplitude error, and sweeps the peak-amplitude weight so the frontier between
the two is measured rather than asserted.

What makes the resonant arm hard, both measured here:

* ``tantrix_zero_area_loss`` is the **squared** area ``|A_T|^2`` normalized by
  ``pi^2``, so the error-curve closure is ``pi*sqrt(C_tantrix_hat)`` and the
  robustness term fades out of a weighted sum exactly as it starts to succeed.
  It needs a weight of order ``1e3``, not ``1``.
* ``|A_T|`` is the robustness of BARQ's own **TTC** control.  It transfers to
  the resonant ``resTTC`` export only when ``Tg*Delta_TTC`` is genuinely small.
  A cold start with a large tantrix weight reaches ``|A_T| = 9e-5`` while
  ``(Tg*Delta)^2`` runs to 7.6, and the exported resonant pulse then closes at
  2.5 rad -- no robustness at all.

The recipe that works is a **warm start followed by an escalating resonance
penalty** (:data:`DETUNING_SCHEDULE`).  Starting from the step08 TTC curve,
which is already amplitude-robust but badly detuned, and raising ``w_delta``
through 0.01 -> 1 -> 100 walks the curve to resonance while the large tantrix
weight holds the geometry.  Jumping straight to a large ``w_delta`` lands in a
bad basin instead; the escalation is the point.

Every candidate is re-checked by propagating its exported CSV, because PGF
fixes the gate for the TTC control, not for the resonant export.

Artifacts land in ``transmon_waveforms/`` and are reused unless
``--reoptimize`` is given.  Run from the repository root:

    conda run -n curve python \\
        proposals/1qb-DB/1qb-DB-demo/generate_transmon_waveforms.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pulse_shape_novera  # noqa: F401  (applies the qurveros patches)
from pulse_shape_novera import (
    make_barq_xgate,
    make_circle_arc_spacecurve,
    optimize_barq_robustness,
    sweep,
)
from qurveros import controltools, losses

import db_helpers
import db_transmon


OUTPUT = db_transmon.TRANSMON_DIR
FORMAT_VERSION = 1
TARGET_ANGLE = float(np.pi)

# Same BARQ initialization as the generator, so the only difference between
# these candidates and the shipped resonant screening pulse is the weighting.
N_FREE_POINTS = 10
SEED = 4531469
NORM_VALUE = 0.25
OPTIMIZER_LR = 1e-3
N_FRENET = 1200
N_SAMPLES = 2501

#: Warm start: the step08 TTC-mode BARQ curve.  It is already amplitude robust
#: (``|A_T| = 8.7e-3``) but needs ``Tg*Delta = 3.1``, which is exactly the
#: trade this script has to undo.
WARM_START_PARAMS = (
    HERE.parent
    / "X_gate_pulse_generation/waveforms/barq_amplitude_robust_params.npz"
)

#: Weight on the (squared, ``pi^2``-normalized) tantrix area.  Large on purpose;
#: see the module docstring.
TANTRIX_WEIGHT = 1000.0

#: Resonance penalty, escalated in place.  Each entry is ``(w_delta, steps)``.
DETUNING_SCHEDULE = ((0.01, 800), (1.0, 800), (100.0, 800))

#: Peak-amplitude / energy weight.  This is the axis of the sweep: it trades
#: ``Tg*Omega_max`` (leakage, gate time) against the closure.
AMP_WEIGHTS = (0.01, 0.1, 1.0)

#: A candidate only counts as first-order amplitude robust below this closure
#: distance |R(Tg)-R(0)| in radians (the Gaussian sits at pi).
CLOSURE_TARGET = 1e-2


def _stem(amp_weight):
    return f"barq_resonant_robust_a{amp_weight:g}"


def _display_name(amp_weight):
    return f"BARQ resonant robust (w_amp={amp_weight:g})"


def _normalized(loss_fn, reference):
    return lambda frenet: loss_fn(frenet) / reference


def _save_params(barq, path: Path, weights) -> None:
    arrays = {
        "format_version": np.asarray(FORMAT_VERSION),
        "n_free_points": np.asarray(N_FREE_POINTS),
        "seed": np.asarray(SEED),
        "norm_value": np.asarray(NORM_VALUE),
        "tantrix_weight": np.asarray(TANTRIX_WEIGHT),
        "amp_weight": np.asarray(weights),
        "free_points": np.asarray(barq.params["free_points"]),
    }
    for name, value in barq.params["pgf_params"].items():
        arrays[f"pgf__{name}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def _load_params(path: Path):
    saved = np.load(path)
    if int(saved["format_version"]) != FORMAT_VERSION:
        raise ValueError(f"unsupported parameter format in {path.name}")
    if int(saved["n_free_points"]) != N_FREE_POINTS:
        raise ValueError(f"{path.name} uses a different number of free points")
    barq = make_barq_xgate(
        n_free_points=N_FREE_POINTS, seed=SEED, norm_value=NORM_VALUE
    )
    params = barq.params.copy()
    params["free_points"] = jnp.asarray(saved["free_points"])
    pgf = params["pgf_params"].copy()
    for name in pgf:
        pgf[name] = jnp.asarray(saved[f"pgf__{name}"])
    params["pgf_params"] = pgf
    barq.set_params(params)
    return barq


def _optimize(amp_weight, references, verbose=True):
    """Warm-start BARQ and walk it to resonance under an escalating penalty.

    The tantrix weight stays at :data:`TANTRIX_WEIGHT` throughout; only the
    resonance penalty moves.  Returns the optimized curve together with the
    per-stage trace, which is worth keeping: it is the evidence that the
    escalation, not the final weight, is what makes the recipe work.
    """
    cost = [
        [_normalized(losses.tantrix_zero_area_loss, references["tantrix"]),
         TANTRIX_WEIGHT],
        [_normalized(sweep.energy_term, references["energy"]), amp_weight],
        [_normalized(losses.max_amp_loss, references["max_amp"]), amp_weight],
    ]
    barq = _load_params(WARM_START_PARAMS)

    trace = []
    for detuning_weight, steps in DETUNING_SCHEDULE:
        optimize_barq_robustness(
            barq,
            max_iter=steps,
            lr=OPTIMIZER_LR,
            loss_terms=cost + [[losses.barq_detuning_loss, detuning_weight]],
        )
        barq.evaluate_frenet_dict(N_FRENET)
        stage = {
            "detuning_weight": detuning_weight,
            "steps": steps,
            "tantrix_area_norm": float(
                np.sqrt(losses.tantrix_zero_area_loss(barq.frenet_dict))
            ),
            "TTC_detuning_squared": float(
                losses.barq_detuning_loss(barq.frenet_dict)
            ),
        }
        trace.append(stage)
        if verbose:
            print(
                f"    w_delta={detuning_weight:<6g} "
                f"|A_T|={stage['tantrix_area_norm']:.3e}  "
                f"(Tg*D)^2={stage['TTC_detuning_squared']:.3e}",
                flush=True,
            )
    return barq, trace


def _export_csv(stem, control):
    """Write the normalized CSV in the convention used by every other pulse."""
    phase = np.unwrap(np.asarray(control["phi"]))
    omega = np.asarray(control["omega"])
    table = np.column_stack([
        np.asarray(control["times"]),
        omega * np.cos(phase),
        omega * np.sin(phase),
        omega,
        phase,
        np.asarray(control["delta"]),
    ])
    header = (
        "time_over_Tg,Tg_omega_x,Tg_omega_y,"
        "Tg_omega_envelope,phase_rad,Tg_delta"
    )
    np.savetxt(
        OUTPUT / f"{stem}.csv",
        table,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.12e",
    )


def build_candidates(reoptimize=False, weights=AMP_WEIGHTS, verbose=True):
    """Optimize (or reuse) one candidate per amplitude weight, export the CSVs.

    Returns
    -------
    dict
        ``{display name: summary dict}`` in weight order.  The summary carries
        the normalized geometry costs, the residual TTC detuning, the
        escalation trace and the properties of the *exported resonant*
        waveform, which is the thing the notebook actually propagates.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    naive = make_circle_arc_spacecurve(TARGET_ANGLE)
    reference = sweep.compute_references(naive, n_frenet=N_FRENET)

    summaries = {}
    for amp_weight in weights:
        stem = _stem(amp_weight)
        params_file = OUTPUT / f"{stem}_params.npz"
        trace = None
        if params_file.exists() and not reoptimize:
            barq = _load_params(params_file)
            reused = True
        else:
            if verbose:
                print(f"optimizing {stem} ...", flush=True)
            barq, trace = _optimize(amp_weight, reference["refs"], verbose)
            _save_params(barq, params_file, amp_weight)
            reused = False

        barq.evaluate_frenet_dict(N_FRENET)
        control = controltools.calculate_control_dict(
            barq.frenet_dict, "resTTC", N_SAMPLES
        )
        _export_csv(stem, control)

        # Re-check on the exported resonant waveform, not on the geometry:
        # resTTC drops the detuning it needed, and PGF does not cover that.
        pulse = db_transmon.load_transmon_waveform(stem)
        metrics = db_transmon.pulse_metrics(pulse)
        summaries[_display_name(amp_weight)] = {
            "stem": stem,
            "tantrix_weight": TANTRIX_WEIGHT,
            "amp_weight": amp_weight,
            "detuning_schedule": [list(item) for item in DETUNING_SCHEDULE],
            "escalation_trace": trace,
            "reused_saved_parameters": reused,
            "normalized_geometry_costs": sweep.evaluate_chat(
                barq, reference, n_frenet=N_FRENET
            ),
            # sqrt of the raw tantrix loss, i.e. |A_T| in radians.  This must
            # equal the exported waveform's amplitude error-curve closure --
            # qurveros' geometric loss and db_helpers' Hamiltonian-level curve
            # are two independent implementations of the same quantity.
            "tantrix_area_norm": float(
                np.sqrt(losses.tantrix_zero_area_loss(barq.frenet_dict))
            ),
            "TTC_detuning_squared": float(
                losses.barq_detuning_loss(barq.frenet_dict)
            ),
            "exported_resonant": metrics,
        }
    _write_settings(summaries, weights)
    return summaries


def _write_settings(summaries, weights):
    settings = {
        "format_version": FORMAT_VERSION,
        "purpose": (
            "resonant, first-order amplitude-robust X(pi) candidates for the "
            "three-level transmon comparison"
        ),
        "barq": {
            "n_free_points": N_FREE_POINTS,
            "seed": SEED,
            "norm_value": NORM_VALUE,
            "learning_rate": OPTIMIZER_LR,
            "warm_start": str(WARM_START_PARAMS.name),
            "detuning_schedule": [list(item) for item in DETUNING_SCHEDULE],
            "control_mode": "resTTC",
            "samples": N_SAMPLES,
        },
        "cost": (
            f"{TANTRIX_WEIGHT}*C_tantrix_hat "
            "+ w_amp*(C_energy_hat + C_max_amp_hat) "
            "+ w_delta*(Tg*Delta_TTC)^2, w_delta escalated in place"
        ),
        "amp_weights": list(weights),
        "closure_target": CLOSURE_TARGET,
        "candidates": summaries,
    }
    with (OUTPUT / "settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)


def select_candidate(summaries, closure_target=CLOSURE_TARGET):
    """Pick the lowest-peak candidate that is still first-order robust.

    Returns ``(name, summary)``, or ``(None, None)`` when no candidate closes
    its amplitude error curve well enough -- a real possible outcome, since the
    peak-amplitude term is exactly what fights the tantrix term.
    """
    passing = [
        (name, summary)
        for name, summary in summaries.items()
        if summary["exported_resonant"]["closure"] <= closure_target
    ]
    if not passing:
        return None, None
    return min(
        passing, key=lambda item: item[1]["exported_resonant"]["peak_Tg_omega"]
    )


def load_candidates(weights=AMP_WEIGHTS):
    """Load the exported candidate CSVs without touching the optimizer."""
    return {
        _display_name(weight): db_transmon.load_transmon_waveform(_stem(weight))
        for weight in weights
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reoptimize",
        action="store_true",
        help="ignore saved BARQ parameters and optimize from scratch",
    )
    args = parser.parse_args()

    summaries = build_candidates(reoptimize=args.reoptimize)
    print(f"\nwrote {OUTPUT}")
    print(
        f"{'candidate':44s}{'Tg*Om_max':>11s}{'|R(T)-R(0)|':>14s}"
        f"{'|A_T| (geom)':>14s}{'(Tg*D_TTC)^2':>14s}{'1-F(eps=0)':>12s}"
        f"{'Tg [ns]':>10s}{'P2 (exact)':>13s}"
    )
    for name, summary in summaries.items():
        metrics = summary["exported_resonant"]
        print(
            f"{name:44s}{metrics['peak_Tg_omega']:11.3f}"
            f"{metrics['closure']:14.3e}"
            f"{summary['tantrix_area_norm']:14.3e}"
            f"{summary['TTC_detuning_squared']:14.3e}"
            f"{metrics['zero_error_infidelity']:12.2e}"
            f"{metrics['gate_time_ns']:10.2f}"
            f"{metrics['leakage_P2']:13.3e}"
        )
    selected, _ = select_candidate(summaries)
    print(f"\nselected: {selected or 'none passes the closure target'}")


if __name__ == "__main__":
    main()
