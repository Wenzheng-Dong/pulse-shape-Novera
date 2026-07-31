"""Numerical helpers for the 1qb-DB waveform demo.

The amplitude error is injected into the sampled Hamiltonian before the
time-ordered gate and DB trace are calculated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
U_TARGET = -1j * X

WAVEFORM_DIR = (
    Path(__file__).resolve().parents[1]
    / "X_gate_pulse_generation"
    / "waveforms"
)


def load_experiment_inputs():
    """Load the selected waveform CSVs and their saved metadata."""
    with (WAVEFORM_DIR / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    def load_csv(filename):
        return np.genfromtxt(
            WAVEFORM_DIR / filename,
            delimiter=",",
            names=True,
        )

    pulses = {
        "Gaussian": load_csv("gaussian_x_pi.csv"),
        "Resonant composite": load_csv(
            "resonant_composite_robust_x_pi.csv"
        ),
    }
    return WAVEFORM_DIR, metadata, pulses


def plot_controls(pulses):
    """Plot the two quadratures of each imported waveform."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for axis, (name, pulse) in zip(axes, pulses.items()):
        axis.plot(
            pulse["time_over_Tg"],
            pulse["Tg_omega_x"],
            label=r"$T_g\Omega_x$",
        )
        axis.plot(
            pulse["time_over_Tg"],
            pulse["Tg_omega_y"],
            label=r"$T_g\Omega_y$",
        )
        axis.set(title=name, ylabel="normalized drive")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel(r"normalized time $t/T_g$")
    figure.tight_layout()
    return figure


def injected_midpoint_hamiltonians(pulse, epsilon):
    """Construct H_epsilon at the midpoint of every sampled interval."""
    time = pulse["time_over_Tg"]
    omega_x = (1.0 + epsilon) * pulse["Tg_omega_x"]
    omega_y = (1.0 + epsilon) * pulse["Tg_omega_y"]
    delta = pulse["Tg_delta"]

    midpoint_x = 0.5 * (omega_x[:-1] + omega_x[1:])
    midpoint_y = 0.5 * (omega_y[:-1] + omega_y[1:])
    midpoint_delta = 0.5 * (delta[:-1] + delta[1:])
    hamiltonians = 0.5 * (
        midpoint_x[:, None, None] * X
        + midpoint_y[:, None, None] * Y
        + midpoint_delta[:, None, None] * Z
    )
    return np.diff(time), hamiltonians


def time_ordered_gate(pulse, epsilon=0.0):
    """Calculate U_epsilon(Tg) as the ordered product of exp(-i H dt)."""
    time_steps, hamiltonians = injected_midpoint_hamiltonians(
        pulse,
        epsilon,
    )
    unitary = I2.copy()
    for dt, hamiltonian in zip(time_steps, hamiltonians):
        unitary = expm(-1j * hamiltonian * dt) @ unitary
    return unitary


def calculate_gates(pulses, errors):
    """Calculate the zero-error and requested noisy gates."""
    return {
        name: {
            epsilon: time_ordered_gate(pulse, epsilon)
            for epsilon in (0.0, *errors)
        }
        for name, pulse in pulses.items()
    }


def average_gate_fidelity(unitary, target=U_TARGET):
    return float(
        (abs(np.trace(target.conj().T @ unitary)) ** 2 + 2.0) / 6.0
    )


def db_trace(unitary, cycles):
    """Apply the calculated gate pair repeatedly to |0>."""
    pair = unitary @ unitary
    state = np.array([1.0, 0.0], dtype=complex)
    values = [1.0]
    for _ in range(cycles):
        state = pair @ state
        values.append(float(abs(state[0]) ** 2))
    return np.asarray(values)


def calculate_db_results(pulses, gates, errors, cycles):
    traces = {
        name: {
            epsilon: db_trace(gates[name][epsilon], cycles)
            for epsilon in errors
        }
        for name in pulses
    }
    results = {
        name: {
            epsilon: {
                "fidelity": average_gate_fidelity(gates[name][epsilon]),
                "P0_final": float(traces[name][epsilon][-1]),
                "delta_P0": float(np.ptp(traces[name][epsilon])),
            }
            for epsilon in errors
        }
        for name in pulses
    }
    return traces, results


def plot_db_comparison(traces, cycles):
    """Overlay both pulses; use connected markers when error signs overlap."""
    import matplotlib.pyplot as plt

    colors = {
        "Gaussian": "tab:blue",
        "Resonant composite": "tab:orange",
    }
    styles = {
        -0.03: {
            "linestyle": "-",
            "linewidth": 1.4,
            "marker": "o",
            "markevery": 4,
            "markersize": 6,
            "markerfacecolor": "white",
            "markeredgewidth": 1.5,
            "zorder": 3,
        },
        0.03: {
            "linestyle": "-",
            "linewidth": 3.0,
            "alpha": 0.60,
            "zorder": 2,
        },
    }

    figure, axis = plt.subplots(figsize=(10, 5.2))
    cycle_numbers = np.arange(cycles + 1)
    for name, pulse_traces in traces.items():
        for epsilon, values in pulse_traces.items():
            axis.plot(
                cycle_numbers,
                values,
                color=colors[name],
                label=f"{name}, epsilon={epsilon:+.0%}",
                **styles[epsilon],
            )
    axis.set(
        title="1qb-DB: waveform comparison",
        xlabel="XX cycle count n",
        ylabel=r"return probability $P_0$",
        ylim=(-0.03, 1.015),
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    return figure


def print_gate_report(gates, errors):
    """Print the actual noisy gate matrices and their unitarity errors."""
    for name in gates:
        print(f"\n{name}")
        for epsilon in errors:
            gate = gates[name][epsilon]
            error = np.linalg.norm(gate.conj().T @ gate - I2)
            print(
                f"epsilon={epsilon:+.0%}, "
                f"||U_dagger U-I||={error:.2e}"
            )
            print(np.array2string(gate, precision=7, suppress_small=True))


def print_results_table(results, errors):
    print(
        f"{'pulse':20s} {'error':>7s} {'F_gate':>14s} "
        f"{'P0(final)':>14s} {'Delta P0':>14s}"
    )
    print("-" * 75)
    for name in results:
        for epsilon in errors:
            result = results[name][epsilon]
            print(
                f"{name:20s} {epsilon:+7.0%} "
                f"{result['fidelity']:14.12f} "
                f"{result['P0_final']:14.9f} "
                f"{result['delta_P0']:14.6e}"
            )


def validate_results(metadata, gates, results, errors, cycles):
    """Compare the independent notebook calculation with saved metadata."""
    metadata_names = {
        "Gaussian": "gaussian_x_pi",
        "Resonant composite": "resonant_composite_robust_x_pi",
    }
    db_key = f"DB_trace_over_cycles_0_to_{cycles}"

    for name, metadata_name in metadata_names.items():
        saved = metadata["pulses"][metadata_name]
        assert (
            abs(
                average_gate_fidelity(gates[name][0.0])
                - saved["zero_error_fidelity"]
            )
            < 2e-8
        )
        for epsilon in errors:
            saved_db = saved[db_key][f"{epsilon:+.2f}"]
            current = results[name][epsilon]
            assert (
                abs(
                    current["P0_final"]
                    - saved_db["P0_at_final_cycle"]
                )
                < 2e-5
            )
            assert (
                abs(
                    current["delta_P0"]
                    - saved_db["delta_P0_max_minus_min"]
                )
                < 2e-5
            )

    assert max(
        results["Gaussian"][epsilon]["delta_P0"]
        for epsilon in errors
    ) > 0.99
    assert max(
        results["Resonant composite"][epsilon]["delta_P0"]
        for epsilon in errors
    ) < 1e-5
