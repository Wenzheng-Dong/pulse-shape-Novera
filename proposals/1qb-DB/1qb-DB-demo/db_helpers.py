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
PAULIS = np.stack([X, Y, Z])
U_TARGET = -1j * X

PULSE_COLORS = {
    "Gaussian": "tab:blue",
    "Resonant composite": "tab:orange",
    "BARQ resonant (smooth)": "tab:green",
}

WAVEFORM_DIR = (
    Path(__file__).resolve().parents[1]
    / "X_gate_pulse_generation"
    / "waveforms"
)


def load_waveform(filename):
    """Load one waveform CSV from the saved waveform directory."""
    return np.genfromtxt(
        WAVEFORM_DIR / filename,
        delimiter=",",
        names=True,
    )


def load_experiment_inputs():
    """Load the two selected waveform CSVs and their saved metadata."""
    with (WAVEFORM_DIR / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    pulses = {
        "Gaussian": load_waveform("gaussian_x_pi.csv"),
        "Resonant composite": load_waveform(
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


def ideal_propagators(pulse):
    """Calculate the error-free propagator U_0 at every sampled time."""
    time_steps, hamiltonians = injected_midpoint_hamiltonians(pulse, 0.0)
    propagators = np.empty((len(time_steps) + 1, 2, 2), dtype=complex)
    propagators[0] = I2
    for index, (dt, hamiltonian) in enumerate(
        zip(time_steps, hamiltonians)
    ):
        propagators[index + 1] = (
            expm(-1j * hamiltonian * dt) @ propagators[index]
        )
    return propagators


def error_curve(pulse, noise="amplitude"):
    """Calculate the first-order error curve R(t) of one waveform.

    An error Hamiltonian H_err(t)=epsilon*G(t) enters the first Magnus
    term as -i*epsilon*int_0^Tg U_0^dag(t) G(t) U_0(t) dt.  Writing the
    toggling-frame operator as U_0^dag G U_0 = g(t).sigma/2 defines the
    error curve

        R(t) = int_0^t g(s) ds,

    whose geometry carries the robustness of the pulse:

    - ``closure`` = |R(Tg)-R(0)| is the first-order error rotation angle
      per unit epsilon, so it vanishes exactly for a first-order robust
      pulse (a closed curve);
    - ``area`` = (1/2) int R x dR is the signed area vector, whose three
      components are the areas of the projections onto the yz, zx and xy
      planes.  It controls the leading second-order term.  The open path
      is closed by a straight chord back to the origin, which adds no
      area because R is parallel to dR along it.

    Parameters
    ----------
    pulse : numpy structured array
        One imported waveform, in the normalized CSV convention.
    noise : {"amplitude", "dephasing"}
        Which error the curve describes.  ``"amplitude"`` uses G=H_c,
        the multiplicative drive error that this demo injects; the curve
        then has speed Tg*Omega(t) and total length equal to the total
        rotation angle in radians.  ``"dephasing"`` uses G=Z/2, the
        standard SCQC error curve, which is unit speed with length Tg.

    Returns
    -------
    dict
        ``time``, ``curve`` (N, 3), ``closure``, ``area`` (3,) and
        ``length``.  All lengths and areas are in radians (radians^2),
        because the waveforms are normalized to Tg=1.
    """
    time = pulse["time_over_Tg"]
    if noise == "amplitude":
        generator = np.stack(
            [
                pulse["Tg_omega_x"],
                pulse["Tg_omega_y"],
                np.zeros_like(time),
            ],
            axis=1,
        )
    elif noise == "dephasing":
        generator = np.zeros((len(time), 3))
        generator[:, 2] = 1.0
    else:
        raise ValueError(f"unknown noise channel: {noise!r}")

    propagators = ideal_propagators(pulse)
    generator_operator = np.einsum("na,aij->nij", generator, PAULIS)
    toggled = np.einsum(
        "nji,njk,nkl->nil",
        propagators.conj(),
        generator_operator,
        propagators,
    )
    # g_b = (1/2) Tr[sigma_b U_0^dag (c.sigma) U_0], real by hermiticity.
    tangent = 0.5 * np.einsum("aij,nji->na", PAULIS, toggled).real

    increments = 0.5 * (tangent[:-1] + tangent[1:]) * np.diff(time)[:, None]
    curve = np.vstack([np.zeros(3), np.cumsum(increments, axis=0)])
    midpoints = 0.5 * (curve[:-1] + curve[1:])
    area = 0.5 * np.sum(np.cross(midpoints, np.diff(curve, axis=0)), axis=0)
    return {
        "time": time,
        "curve": curve,
        "closure": float(np.linalg.norm(curve[-1] - curve[0])),
        "area": area,
        "length": float(np.sum(np.linalg.norm(increments, axis=1))),
    }


def calculate_error_curves(pulses, noise="amplitude"):
    """Calculate the error curve of every imported waveform."""
    return {
        name: error_curve(pulse, noise)
        for name, pulse in pulses.items()
    }


def plot_error_curves(curves):
    """Show the error curves in 3D together with their plane projections.

    The wide translucent line, the thin line and the dashed line let the
    curves stay visible where they overlap, as in the DB comparison plot.
    All panels use one common isotropic scale, so a curve that stays in a
    plane also looks flat instead of having its numerical noise magnified.

    Dots mark equally spaced times, so their spacing shows |dR/dt|=Omega(t):
    a smooth envelope shows up as a smooth density along the path, whereas
    the *shape* only bends where the drive phase turns.  A composite pulse
    with piecewise-constant phase therefore gives a polygon whose corners
    are traversed at Omega=0, however smooth its envelope is.
    """
    import matplotlib.pyplot as plt

    # (horizontal axis, vertical axis, area component carried by the plane)
    projections = ((0, 1, "xy"), (1, 2, "yz"), (2, 0, "zx"))
    labels = ("x", "y", "z")
    styles = (
        {"linewidth": 3.5, "alpha": 0.55, "zorder": 2},
        {"linewidth": 1.6, "zorder": 3},
        {"linewidth": 1.4, "linestyle": (0, (5, 2)), "zorder": 4},
    )

    time_dots = 80  # equally spaced samples in time, not in arc length
    points = np.vstack([result["curve"] for result in curves.values()])
    center = 0.5 * (points.max(axis=0) + points.min(axis=0))
    half_span = 0.55 * np.ptp(points, axis=0).max()
    limits = np.stack([center - half_span, center + half_span], axis=1)

    figure = plt.figure(figsize=(11, 8))
    space_axis = figure.add_subplot(2, 2, 1, projection="3d")
    for index, (name, result) in enumerate(curves.items()):
        curve = result["curve"]
        color = PULSE_COLORS[name]
        style = styles[index % len(styles)]
        label = f"{name}, $|R(T)-R(0)|$={result['closure']:.1e}"
        space_axis.plot(*curve.T, color=color, label=label, **style)
        space_axis.scatter(
            *curve[0], color=color, marker="o", s=45, depthshade=False
        )
        space_axis.scatter(
            *curve[-1],
            color=color,
            marker="s",
            s=45,
            facecolor="white",
            depthshade=False,
        )
    space_axis.set(
        title="error curves (circle: start, square: end)",
        xlabel="$R_x$",
        ylabel="$R_y$",
        zlabel="$R_z$",
        xlim=limits[0],
        ylim=limits[1],
        zlim=limits[2],
    )
    space_axis.set_box_aspect((1, 1, 1))
    space_axis.legend(loc="upper left", fontsize=8)

    for index, (i, j, plane) in enumerate(projections):
        axis = figure.add_subplot(2, 2, index + 2)
        for order, (name, result) in enumerate(curves.items()):
            curve = result["curve"]
            color = PULSE_COLORS[name]
            style = styles[order % len(styles)]
            axis.plot(
                curve[:, i], curve[:, j], color=color, label=name, **style
            )
            stride = max(1, len(curve) // time_dots)
            axis.plot(
                curve[::stride, i],
                curve[::stride, j],
                color=color,
                linestyle="none",
                marker=".",
                markersize=3,
                zorder=4,
            )
            axis.plot(*curve[0, [i, j]], color=color, marker="o")
            axis.plot(
                *curve[-1, [i, j]],
                color=color,
                marker="s",
                markerfacecolor="white",
            )
        # Annotate the area this plane carries, in the colour of its curve.
        for row, (name, result) in enumerate(curves.items()):
            axis.text(
                0.02,
                0.96 - 0.07 * row,
                f"$A_{{{plane}}}$={result['area'][(i + 2) % 3]:+.2e}",
                transform=axis.transAxes,
                color=PULSE_COLORS[name],
                fontsize=8,
                verticalalignment="top",
            )
        axis.set(
            title=f"{plane} projection (signed area $A_{{{plane}}}$)",
            xlabel=f"$R_{labels[i]}$",
            ylabel=f"$R_{labels[j]}$",
            xlim=limits[i],
            ylim=limits[j],
        )
        axis.set_aspect("equal")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def print_error_curve_report(curves, errors=None):
    """Print the closure distance and projected areas of each error curve."""
    print(
        f"{'pulse':24s} {'length/pi':>10s} {'|R(T)-R(0)|':>13s} "
        f"{'A_yz':>12s} {'A_zx':>12s} {'A_xy':>12s} {'|A|':>12s}"
    )
    print("-" * 100)
    for name, result in curves.items():
        area = result["area"]
        print(
            f"{name:24s} {result['length'] / np.pi:10.6f} "
            f"{result['closure']:13.6e} "
            f"{area[0]:12.4e} {area[1]:12.4e} {area[2]:12.4e} "
            f"{np.linalg.norm(area):12.4e}"
        )
    if errors is None:
        return
    # |R(Tg)-R(0)| is the error rotation angle per unit epsilon, so this is
    # the first-order prediction for the angle error of a single gate.
    print("\nfirst-order error angle per gate, epsilon*|R(T)-R(0)|:")
    for name, result in curves.items():
        angles = "  ".join(
            f"epsilon={epsilon:+.0%}: {abs(epsilon) * result['closure']:.3e} rad"
            for epsilon in errors
        )
        print(f"{name:24s} {angles}")


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
                color=PULSE_COLORS[name],
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
