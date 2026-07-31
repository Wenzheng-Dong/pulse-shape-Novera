"""Generate normalized X(pi) waveforms for the 1qb-DB experiment.

The recommended robust candidate is strictly resonant and optimized directly
for +/-3% amplitude error over 60 DB cycles. Saved parameters are reused unless
``--reoptimize`` is supplied. Legacy BARQ candidates are retained only for
comparison.

Run from the repository root:

    conda run -n curve python \
        proposals/1qb-DB/X_gate_pulse_generation/generate_waveforms.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import qutip
from scipy.optimize import differential_evolution, minimize

import pulse_shape_novera
from pulse_shape_novera import (
    make_barq_xgate,
    make_circle_arc_spacecurve,
    make_gaussian_arc_spacecurve,
    optimize_barq_robustness,
    sweep,
)
from qurveros import controltools, losses
from qurveros.qubit_bench import simulator


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
OUTPUT = HERE / "waveforms"
TTC_PARAMS_FILE = OUTPUT / "barq_amplitude_robust_params.npz"
RESONANT_PARAMS_FILE = OUTPUT / "barq_resonant_screening_params.npz"
COMPOSITE_PARAMS_FILE = OUTPUT / "resonant_composite_robust_params.json"
SOURCE_FILES = (
    REPO_ROOT / "environment.yml",
    REPO_ROOT / "src/pulse_shape_novera/_patches.py",
    REPO_ROOT / "src/pulse_shape_novera/ansatz.py",
    REPO_ROOT / "src/pulse_shape_novera/barq.py",
    REPO_ROOT / "src/pulse_shape_novera/proxies.py",
    REPO_ROOT / "src/pulse_shape_novera/sweep.py",
)
FORMAT_VERSION = 1
TARGET_ANGLE = float(np.pi)
GAUSSIAN_DURATION = 1.0
GAUSSIAN_SIGMA_FRACTION = 0.18
BARQ_N_FREE_POINTS = 10
BARQ_SEED = 4531469
BARQ_NORM_VALUE = 0.25
OPTIMIZER_LR = 1e-3
PRIMARY_STEPS = 1200
REFINEMENT_STEPS = 400
TANTRIX_WEIGHT = 1.0
ENERGY_WEIGHT = 0.01
MAX_AMP_WEIGHT = 0.01
TTC_DETUNING_WEIGHT = 0.01
TTC_ROBUST_STEPS = 1500
TTC_MAX_AMP_WEIGHT = 0.01
N_FRENET = 1200
N_SAMPLES = 2501
ERRORS = (-0.03, 0.0, 0.03)
N_DB_CYCLES = 60
PHASE_OPT_SEED = 20260730
PHASE_OPT_GATE_WEIGHT = 10.0
PHASE_OPT_GRID = np.linspace(-0.03, 0.03, 13)


def _normalized(loss_fn, reference):
    return lambda frenet: loss_fn(frenet) / reference


def _save_barq_params(barq, path: Path) -> None:
    arrays = {
        "format_version": np.asarray(FORMAT_VERSION),
        "n_free_points": np.asarray(BARQ_N_FREE_POINTS),
        "seed": np.asarray(BARQ_SEED),
        "norm_value": np.asarray(BARQ_NORM_VALUE),
        "free_points": np.asarray(barq.params["free_points"]),
    }
    for name, value in barq.params["pgf_params"].items():
        arrays[f"pgf__{name}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def _load_barq_params(path: Path):
    saved = np.load(path)
    saved_version = int(saved["format_version"])
    saved_n_free = int(saved["n_free_points"])
    if saved_version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported parameter format {saved_version}; "
            f"expected {FORMAT_VERSION}"
        )
    if saved_n_free != BARQ_N_FREE_POINTS:
        raise ValueError(
            f"saved candidate uses {saved_n_free} free points; "
            f"generator expects {BARQ_N_FREE_POINTS}"
        )
    barq = make_barq_xgate(
        n_free_points=BARQ_N_FREE_POINTS,
        seed=BARQ_SEED,
        norm_value=BARQ_NORM_VALUE,
    )
    params = barq.params.copy()
    params["free_points"] = jnp.asarray(saved["free_points"])
    pgf = params["pgf_params"].copy()
    for name in pgf:
        pgf[name] = jnp.asarray(saved[f"pgf__{name}"])
    params["pgf_params"] = pgf
    barq.set_params(params)
    return barq


def _build_resonant_screening_candidate(reoptimize: bool):
    if RESONANT_PARAMS_FILE.exists() and not reoptimize:
        return _load_barq_params(RESONANT_PARAMS_FILE), False

    naive = make_circle_arc_spacecurve(TARGET_ANGLE)
    references = sweep.compute_references(naive)["refs"]
    primary_cost = [
        [
            _normalized(
                losses.tantrix_zero_area_loss, references["tantrix"]
            ),
            TANTRIX_WEIGHT,
        ],
        [
            _normalized(sweep.energy_term, references["energy"]),
            ENERGY_WEIGHT,
        ],
        [
            _normalized(losses.max_amp_loss, references["max_amp"]),
            MAX_AMP_WEIGHT,
        ],
    ]

    barq = make_barq_xgate(
        n_free_points=BARQ_N_FREE_POINTS,
        seed=BARQ_SEED,
        norm_value=BARQ_NORM_VALUE,
    )
    optimize_barq_robustness(
        barq,
        max_iter=PRIMARY_STEPS,
        lr=OPTIMIZER_LR,
        loss_terms=primary_cost,
    )

    # A short refinement suppresses the TTC detuning so that the exported
    # resTTC waveform is a genuinely resonant x/y control.
    optimize_barq_robustness(
        barq,
        max_iter=REFINEMENT_STEPS,
        lr=OPTIMIZER_LR,
        loss_terms=primary_cost
        + [[losses.barq_detuning_loss, TTC_DETUNING_WEIGHT]],
    )
    _save_barq_params(barq, RESONANT_PARAMS_FILE)
    return barq, True


def _build_ttc_robust_candidate(reoptimize: bool):
    if TTC_PARAMS_FILE.exists() and not reoptimize:
        return _load_barq_params(TTC_PARAMS_FILE), False

    barq = make_barq_xgate(
        n_free_points=BARQ_N_FREE_POINTS,
        seed=BARQ_SEED,
        norm_value=BARQ_NORM_VALUE,
    )
    optimize_barq_robustness(
        barq,
        max_iter=TTC_ROBUST_STEPS,
        lr=OPTIMIZER_LR,
        loss_terms=[
            [losses.tantrix_zero_area_loss, 1.0],
            [losses.max_amp_loss, TTC_MAX_AMP_WEIGHT],
        ],
    )
    _save_barq_params(barq, TTC_PARAMS_FILE)
    return barq, True


def _rotation(angle, phase, epsilon):
    identity = np.eye(2, dtype=complex)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    half_angle = 0.5 * angle * (1.0 + epsilon)
    axis = np.cos(phase) * pauli_x + np.sin(phase) * pauli_y
    return np.cos(half_angle) * identity - 1.0j * np.sin(half_angle) * axis


def _composite_unitary(phases, epsilon):
    angles = (np.pi, np.pi, 2.0 * np.pi, np.pi)
    sequence_phases = (0.0, phases[0], phases[1], phases[0])
    unitary = np.eye(2, dtype=complex)
    for angle, phase in zip(angles, sequence_phases):
        unitary = _rotation(angle, phase, epsilon) @ unitary
    return unitary


def _exact_composite_losses(phases, epsilon):
    unitary = _composite_unitary(phases, epsilon)
    target = _rotation(np.pi, 0.0, 0.0)
    fidelity = (abs(np.trace(target.conj().T @ unitary)) ** 2 + 2.0) / 6.0
    cycle = unitary @ unitary
    state = np.array([1.0, 0.0], dtype=complex)
    min_p0 = 1.0
    for _ in range(N_DB_CYCLES):
        state = cycle @ state
        min_p0 = min(min_p0, float(abs(state[0]) ** 2))
    return 1.0 - float(fidelity), 1.0 - min_p0


def _phase_objective(phases):
    losses_by_error = [
        _exact_composite_losses(phases, epsilon)
        for epsilon in PHASE_OPT_GRID
    ]
    worst_gate = max(item[0] for item in losses_by_error)
    worst_db = max(item[1] for item in losses_by_error)
    return worst_db + PHASE_OPT_GATE_WEIGHT * worst_gate


def _optimize_composite_phases():
    global_result = differential_evolution(
        _phase_objective,
        bounds=[(0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi)],
        seed=PHASE_OPT_SEED,
        popsize=20,
        maxiter=300,
        tol=1e-10,
        polish=False,
    )
    local_result = minimize(
        _phase_objective,
        global_result.x,
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-14, "fatol": 1e-16},
    )
    phase_1, phase_2 = map(float, local_result.x)
    params = {
        "format_version": FORMAT_VERSION,
        "candidate": "finite-error optimized resonant composite X(pi)",
        "ansatz": {
            "rotation_angles_rad": [
                float(np.pi),
                float(np.pi),
                float(2.0 * np.pi),
                float(np.pi),
            ],
            "phases_rad": [0.0, phase_1, phase_2, phase_1],
            "segment_envelope": "sin^2",
            "segment_durations": "proportional to rotation angle",
            "detuning": 0.0,
        },
        "optimization": {
            "warm_start_family": "BB1-like four-segment composite pulse",
            "optimized_variables": [
                "phase_1_equals_phase_3",
                "phase_2",
            ],
            "epsilon_grid": PHASE_OPT_GRID.tolist(),
            "db_cycles": N_DB_CYCLES,
            "objective": (
                "max_db_population_loss "
                f"+ {PHASE_OPT_GATE_WEIGHT:g}*max_single_gate_infidelity"
            ),
            "global_optimizer": "scipy.optimize.differential_evolution",
            "global_seed": PHASE_OPT_SEED,
            "global_popsize": 20,
            "global_maxiter": 300,
            "global_tol": 1e-10,
            "polish": "Nelder-Mead",
        },
    }
    with COMPOSITE_PARAMS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)
    return params


def _load_composite_params(reoptimize):
    if reoptimize or not COMPOSITE_PARAMS_FILE.exists():
        return _optimize_composite_phases(), True
    with COMPOSITE_PARAMS_FILE.open(encoding="utf-8") as handle:
        return json.load(handle), False


def _composite_control(params):
    angles = np.asarray(params["ansatz"]["rotation_angles_rad"], dtype=float)
    phases = np.asarray(params["ansatz"]["phases_rad"], dtype=float)
    durations = angles / np.sum(angles)
    edges = np.concatenate(([0.0], np.cumsum(durations)))
    times = np.linspace(0.0, 1.0, N_SAMPLES)
    omega = np.zeros(N_SAMPLES)
    phase = np.zeros(N_SAMPLES)
    for index, (angle, segment_phase) in enumerate(zip(angles, phases)):
        if index == len(angles) - 1:
            mask = (times >= edges[index]) & (times <= edges[index + 1])
        else:
            mask = (times >= edges[index]) & (times < edges[index + 1])
        local_time = (times[mask] - edges[index]) / durations[index]
        omega[mask] = (
            2.0
            * angle
            / durations[index]
            * np.sin(np.pi * local_time) ** 2
        )
        phase[mask] = segment_phase
    return {
        "times": times,
        "omega": omega,
        "phi": phase,
        "delta": np.zeros_like(times),
    }


def _gaussian_control():
    gaussian = make_gaussian_arc_spacecurve(
        TARGET_ANGLE,
        duration=GAUSSIAN_DURATION,
        sigma_fraction=GAUSSIAN_SIGMA_FRACTION,
    )
    gaussian.evaluate_frenet_dict(N_FRENET)
    return controltools.calculate_control_dict(
        gaussian.frenet_dict,
        "XY",
        N_SAMPLES,
    )


def _barq_control(barq, mode):
    barq.evaluate_frenet_dict(N_FRENET)
    return controltools.calculate_control_dict(
        barq.frenet_dict,
        mode,
        N_SAMPLES,
    )


def _copy_with_error(control, epsilon):
    copied = {
        name: value.copy() if hasattr(value, "copy") else value
        for name, value in control.items()
    }
    copied["omega"] = (1.0 + epsilon) * copied["omega"]
    return copied


def _fidelity(control, epsilon):
    noisy = _copy_with_error(control, epsilon)
    result = simulator.simulate_control_dict(noisy, qutip.sigmax())
    return float(result["avg_gate_fidelity"])


def _db_summary(control, epsilon):
    noisy = _copy_with_error(control, epsilon)
    unitary = np.asarray(
        simulator.simulate_control_dict(noisy, qutip.sigmax())["u_final"]
    )
    cycle = unitary @ unitary
    state = np.array([1.0, 0.0], dtype=complex)
    values = [1.0]
    for _ in range(N_DB_CYCLES):
        state = cycle @ state
        values.append(float(abs(state[0]) ** 2))
    values = np.asarray(values)
    return {
        "cycles": N_DB_CYCLES,
        "P0_at_final_cycle": float(values[-1]),
        "P0_min": float(values.min()),
        "P0_max": float(values.max()),
        "delta_P0_max_minus_min": float(np.ptp(values)),
    }


def _export_csv(name, control):
    phase = np.unwrap(np.asarray(control["phi"]))
    omega = np.asarray(control["omega"])
    table = np.column_stack(
        [
            np.asarray(control["times"]),
            omega * np.cos(phase),
            omega * np.sin(phase),
            omega,
            phase,
            np.asarray(control["delta"]),
        ]
    )
    header = (
        "time_over_Tg,Tg_omega_x,Tg_omega_y,"
        "Tg_omega_envelope,phase_rad,Tg_delta"
    )
    np.savetxt(
        OUTPUT / f"{name}.csv",
        table,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.12e",
    )


def _pulse_metrics(control):
    time = np.asarray(control["times"])
    omega = np.asarray(control["omega"])
    return {
        "zero_error_fidelity": _fidelity(control, 0.0),
        "peak_Tg_omega": float(np.max(np.abs(omega))),
        "dimensionless_energy": float(np.trapezoid(omega**2, time)),
        "fidelity_by_epsilon": {
            f"{epsilon:+.2f}": _fidelity(control, epsilon)
            for epsilon in ERRORS
        },
        f"DB_trace_over_cycles_0_to_{N_DB_CYCLES}": {
            f"{epsilon:+.2f}": _db_summary(control, epsilon)
            for epsilon in (-0.03, 0.03)
        },
    }


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions():
    names = (
        "pulse-shape-novera",
        "qurveros",
        "jax",
        "jaxlib",
        "optax",
        "numpy",
        "scipy",
        "qutip",
    )
    return {
        name: importlib.metadata.version(name)
        for name in names
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reoptimize",
        action="store_true",
        help=(
            "Ignore the saved resonant-composite phases and optimize a new "
            "candidate. Legacy BARQ parameters are always reused."
        ),
    )
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    # These two BARQ pulses are historical comparisons. Do not spend time
    # reoptimizing them when the selected resonant composite is reoptimized.
    ttc_barq, ttc_optimized_now = _build_ttc_robust_candidate(False)
    resonant_barq, resonant_optimized_now = (
        _build_resonant_screening_candidate(False)
    )
    composite_params, composite_optimized_now = _load_composite_params(
        args.reoptimize
    )
    gaussian = _gaussian_control()
    composite_robust = _composite_control(composite_params)
    ttc_robust = _barq_control(ttc_barq, "TTC")
    resonant_screening = _barq_control(resonant_barq, "resTTC")

    _export_csv("gaussian_x_pi", gaussian)
    _export_csv("resonant_composite_robust_x_pi", composite_robust)
    _export_csv("barq_amplitude_robust_x_pi", ttc_robust)
    _export_csv("barq_resonant_screening_x_pi", resonant_screening)

    naive = make_circle_arc_spacecurve(TARGET_ANGLE)
    reference = sweep.compute_references(naive)
    ttc_barq.evaluate_frenet_dict(N_FRENET)
    resonant_barq.evaluate_frenet_dict(N_FRENET)
    artifacts = (
        TTC_PARAMS_FILE,
        RESONANT_PARAMS_FILE,
        COMPOSITE_PARAMS_FILE,
        OUTPUT / "gaussian_x_pi.csv",
        OUTPUT / "resonant_composite_robust_x_pi.csv",
        OUTPUT / "barq_amplitude_robust_x_pi.csv",
        OUTPUT / "barq_resonant_screening_x_pi.csv",
    )
    metadata = {
        "format_version": FORMAT_VERSION,
        "generator": {
            "path": str(Path(__file__).relative_to(REPO_ROOT)),
            "sha256": _sha256(Path(__file__)),
            "package_versions": _package_versions(),
            "source_sha256": {
                str(path.relative_to(REPO_ROOT)): _sha256(path)
                for path in SOURCE_FILES
            },
        },
        "normalization": {
            "time": "s=t/Tg",
            "fields": "Tg*Omega and Tg*Delta",
            "hamiltonian": (
                "Tg H(s) = 0.5 * "
                "[Tg*omega_x(s)*X + Tg*omega_y(s)*Y "
                "+ Tg*delta(s)*Z]"
            ),
        },
        "samples": N_SAMPLES,
        "gaussian_baseline": {
            "target_angle_rad": TARGET_ANGLE,
            "duration": GAUSSIAN_DURATION,
            "sigma_fraction": GAUSSIAN_SIGMA_FRACTION,
            "control_mode": "XY",
        },
        "robust_candidate": {
            "status": "selected for the resonant +/-3%, 60-cycle experiment",
            "method": "finite-error optimized four-segment composite X(pi)",
            "control_mode": "resonant XY",
            "detuning": 0.0,
            "reused_saved_parameters": not composite_optimized_now,
            "parameter_file": COMPOSITE_PARAMS_FILE.name,
            "parameters": composite_params,
        },
        "legacy_ttc_candidate": {
            "status": (
                "unsupported by the resonant-only experiment; retained for "
                "comparison"
            ),
            "method": "BARQ with total-torsion compensation (TTC)",
            "control_mode": "TTC",
            "reused_saved_parameters": not ttc_optimized_now,
            "constant_Tg_delta_squared": float(
                losses.barq_detuning_loss(ttc_barq.frenet_dict)
            ),
        },
        "resonant_screening_candidate": {
            "status": (
                "rejected for the 60-cycle DB target; retained for comparison"
            ),
            "method": "BARQ with resonant total-torsion compensation",
            "control_mode": "resTTC",
            "reused_saved_parameters": not resonant_optimized_now,
            "initialization": {
                "n_free_points": BARQ_N_FREE_POINTS,
                "seed": BARQ_SEED,
                "norm_value": BARQ_NORM_VALUE,
            },
            "optimizer": {
                "algorithm": "Adam",
                "learning_rate": OPTIMIZER_LR,
            },
            "optimization": [
                {
                    "steps": PRIMARY_STEPS,
                    "cost": (
                        f"{TANTRIX_WEIGHT}*C_tantrix_hat "
                        f"+ {ENERGY_WEIGHT}*C_energy_hat "
                        f"+ {MAX_AMP_WEIGHT}*C_max_amp_hat"
                    ),
                },
                {
                    "steps": REFINEMENT_STEPS,
                    "cost": (
                        "previous cost "
                        f"+ {TTC_DETUNING_WEIGHT}*(Tg*Delta_TTC)^2"
                    ),
                },
            ],
            "normalized_geometry_costs": sweep.evaluate_chat(
                resonant_barq,
                reference,
                n_frenet=N_FRENET,
            ),
            "TTC_detuning_squared": float(
                losses.barq_detuning_loss(resonant_barq.frenet_dict)
            ),
        },
        "pulses": {
            "gaussian_x_pi": _pulse_metrics(gaussian),
            "resonant_composite_robust_x_pi": _pulse_metrics(
                composite_robust
            ),
            "barq_amplitude_robust_x_pi": _pulse_metrics(ttc_robust),
            "barq_resonant_screening_x_pi": _pulse_metrics(
                resonant_screening
            ),
        },
        "artifact_sha256": {
            path.name: _sha256(path)
            for path in artifacts
        },
    }
    with (OUTPUT / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote waveforms and metadata to {OUTPUT}")
    print(json.dumps(metadata["pulses"], indent=2))


if __name__ == "__main__":
    main()
