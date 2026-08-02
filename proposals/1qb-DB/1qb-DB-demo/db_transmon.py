"""Transmon layer for the 1qb-DB experiment.

The demo notebook (``1qb-DB-demo.ipynb``) works entirely in the normalized
two-level picture, where a waveform is a dimensionless pair
``(t/Tg, Tg*Omega)``.  A transmon adds two things that picture cannot see:

* **leakage** to the second excited state, set by the peak drive measured
  against the anharmonicity ``alpha``;
* **decoherence**, set by the gate duration measured against ``T1``/``T2``.

Those two costs pull in opposite directions along the one gauge freedom the
normalized waveform still has.  Rescaling a curve leaves every dimensionless
quantity ``Tg*Omega_max``, ``int Omega dt``, tantrix area, ... unchanged while
trading ``Tg`` against ``Omega`` (see the repository README section 4).  This
module fixes that gauge the way a laboratory does: **every pulse is driven at
the same peak Rabi rate** ``DEVICE.rabi_max``, so each waveform gets the gate
time its own shape demands.  Leakage is then held roughly level across the
comparison and the price of robustness shows up as gate time.

Section 1 of the notebook is coherent and perturbative in the leakage coupling
(:func:`leakage_amplitude`, checked against the exact unitary propagation in
:func:`three_level_propagation`).  Section 2 adds the frozen error model --
Lindblad ``T1``/``T2`` plus a static detuning and the fixed multiplicative
control error -- and compresses one gate into a single 9x9 quantum channel
(:func:`gate_channel`), which is what makes the DB sequence of section 3 cheap:
``n`` cycles is a matrix power, not a re-integration.

The normalized-waveform loading, ideal propagators and error-curve geometry are
reused from :mod:`db_helpers` rather than reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import db_helpers


HERE = Path(__file__).resolve().parent
TRANSMON_DIR = HERE / "transmon_waveforms"

TWO_PI = 2.0 * np.pi


# --- device ------------------------------------------------------------------


@dataclass(frozen=True)
class TransmonDevice:
    """Fixed hardware parameters, in laboratory units.

    Frequencies are ordinary frequencies in GHz (so ``0.05`` is 50 MHz) and
    times are in ns.  Angular rates used by the Hamiltonian are obtained from
    the ``*_rate`` properties, which multiply by ``2*pi``.

    Attributes
    ----------
    anharmonicity : float
        ``alpha/2pi`` in GHz, negative for a transmon.  Sets the 1-2 detuning
        in the frame rotating with a drive resonant on 0-1.
    rabi_max : float
        ``Omega_max/2pi`` in GHz, the peak drive every pulse is normalized to.
    t1, t2_echo : float
        Relaxation and echo coherence times in ns.
    sample_rate : float
        AWG sample rate in GS/s, i.e. samples per ns.
    static_detuning : float
        ``delta_z/2pi`` in GHz: a fixed, unknown-to-the-pulse offset between the
        drive and the 0-1 frequency.  This is the *dephasing* channel section 1
        measures geometrically (``G = Z/2``); Markovian ``T2`` cannot stand in
        for it, because it is quasi-static and therefore partly echoed by the
        repeated-``X`` sequence.
    control_error : float
        ``epsilon`` in the multiplicative drive error
        ``(Omega_x, Omega_y) -> (1+epsilon)(Omega_x, Omega_y)``.  Fixed at 3%
        and never swept.
    """

    anharmonicity: float = -0.200
    rabi_max: float = 0.050
    t1: float = 60_000.0
    t2_echo: float = 60_000.0
    sample_rate: float = 2.4
    static_detuning: float = 1.0e-4
    control_error: float = 0.03

    @property
    def anharmonicity_rate(self) -> float:
        """``alpha`` in rad/ns (negative)."""
        return TWO_PI * self.anharmonicity

    @property
    def rabi_max_rate(self) -> float:
        """``Omega_max`` in rad/ns."""
        return TWO_PI * self.rabi_max

    @property
    def static_detuning_rate(self) -> float:
        """``delta_z`` in rad/ns."""
        return TWO_PI * self.static_detuning

    @property
    def gamma1(self) -> float:
        """Relaxation rate ``1/T1`` of the 1-0 transition, in 1/ns."""
        return 1.0 / self.t1

    @property
    def gamma_phi(self) -> float:
        """Pure dephasing rate ``1/T2 - 1/(2 T1)``, in 1/ns.

        The split is the usual one: ``T2`` already contains the coherence loss
        that relaxation alone causes, and only the remainder is a genuine
        dephasing channel.
        """
        rate = 1.0 / self.t2_echo - 0.5 / self.t1
        if rate < -1e-15:
            raise ValueError(
                f"T2 = {self.t2_echo} ns exceeds the 2*T1 bound for "
                f"T1 = {self.t1} ns"
            )
        return max(rate, 0.0)


DEVICE = TransmonDevice()


# --- normalized waveforms -----------------------------------------------------

#: Waveforms exported by ``X_gate_pulse_generation``.  ``barq_amplitude_robust``
#: is deliberately absent: it carries a constant detuning and the experiment is
#: resonant-only.
IMPORTED_WAVEFORMS = {
    "Gaussian": "gaussian_x_pi.csv",
    "Resonant composite": "resonant_composite_robust_x_pi.csv",
    "BARQ resonant (smooth)": "barq_resonant_screening_x_pi.csv",
}


def load_imported_waveforms(names=None):
    """Load the resonant waveforms exported by the generator.

    Parameters
    ----------
    names : iterable of str, optional
        Subset of :data:`IMPORTED_WAVEFORMS` keys.  Defaults to all of them.

    Returns
    -------
    dict
        ``{display name: structured array}`` in the normalized CSV convention
        ``time_over_Tg, Tg_omega_x, Tg_omega_y, Tg_omega_envelope, phase_rad,
        Tg_delta``.
    """
    selected = IMPORTED_WAVEFORMS if names is None else {
        name: IMPORTED_WAVEFORMS[name] for name in names
    }
    return {
        name: db_helpers.load_waveform(filename)
        for name, filename in selected.items()
    }


def load_transmon_waveform(stem):
    """Load a waveform generated by this module from ``transmon_waveforms/``."""
    return np.genfromtxt(
        TRANSMON_DIR / f"{stem}.csv",
        delimiter=",",
        names=True,
    )


# --- gauge fixing: normalized -> physical -------------------------------------


def to_physical(pulse, device=DEVICE):
    """Place a normalized waveform on the device at fixed peak Rabi rate.

    The gate time follows from the shape,

    .. math:: T_g = \\max_s |T_g\\Omega(s)| / \\Omega_{\\max},

    so a waveform whose normalized peak is large (a robust pulse) becomes a
    long gate rather than an over-driven one.  This is the single gauge choice
    that the whole transmon comparison rests on.

    Returns
    -------
    dict
        ``time`` (ns), ``omega_x``/``omega_y``/``omega``/``delta`` (rad/ns),
        ``phase`` (rad), ``gate_time`` (ns) and ``peak_tg_omega``
        (dimensionless, the invariant that set the gate time).
    """
    peak = float(np.max(np.abs(pulse["Tg_omega_envelope"])))
    gate_time = peak / device.rabi_max_rate
    return {
        "time": pulse["time_over_Tg"] * gate_time,
        "omega_x": pulse["Tg_omega_x"] / gate_time,
        "omega_y": pulse["Tg_omega_y"] / gate_time,
        "omega": pulse["Tg_omega_envelope"] / gate_time,
        "phase": pulse["phase_rad"],
        "delta": pulse["Tg_delta"] / gate_time,
        "gate_time": gate_time,
        "peak_tg_omega": peak,
    }


# --- first-order leakage ------------------------------------------------------


def leakage_amplitude(pulse, device=DEVICE):
    """First-order leakage amplitude ``c_2`` out of ``|0>`` at the gate end.

    Split the three-level rotating-frame Hamiltonian into the driven qubit
    block, which is treated exactly, and the coupling to ``|2>``, which is
    treated to first order.  With the drive resonant on 0-1 the second level
    sits at ``alpha`` and the ladder matrix element carries the usual
    ``sqrt(2)``, so

    .. math::
        c_2 = -\\tfrac{i}{\\sqrt2}\\int_0^{T_g}\\!\\Omega(t)\\,
              e^{i\\Phi(t)}e^{i\\alpha t}\\,\\langle 1|U_{01}(t)|0\\rangle\\,dt .

    Written in normalized time the integrand depends on the waveform only
    through the exported dimensionless fields and the single number
    ``alpha*Tg`` -- which is exactly where the gate-time/leakage trade-off
    enters.  No DRAG correction is applied: the ``|2>`` amplitude reported here
    is what the bare waveform produces.

    Returns
    -------
    dict
        ``amplitude`` (complex ``c_2``), ``population`` (``|c_2|^2``),
        ``alpha_Tg`` and ``drive_ratio`` = ``Omega_max/|alpha|``.
    """
    physical = to_physical(pulse, device)
    alpha_tg = device.anharmonicity_rate * physical["gate_time"]

    time = pulse["time_over_Tg"]
    drive = pulse["Tg_omega_x"] + 1j * pulse["Tg_omega_y"]
    # <1|U_01(t)|0>, the qubit amplitude the leakage coupling acts on.
    u10 = db_helpers.ideal_propagators(pulse)[:, 1, 0]

    integrand = drive * np.exp(1j * alpha_tg * time) * u10
    amplitude = -1j / np.sqrt(2.0) * np.trapezoid(integrand, time)
    return {
        "amplitude": complex(amplitude),
        "population": float(abs(amplitude) ** 2),
        "alpha_Tg": float(alpha_tg),
        "drive_ratio": device.rabi_max_rate / abs(device.anharmonicity_rate),
    }


def three_level_propagation(pulse, device=DEVICE, coupling=1.0, stride=1):
    """Propagate ``|0>`` through the exact three-level RWA Hamiltonian.

    In the frame rotating with a drive resonant on 0-1, and after the RWA,

    .. math::
        H = \\alpha|2\\rangle\\langle2| + \\tfrac12\\big[
            (\\Omega_x - i\\Omega_y)\\,a + \\mathrm{h.c.}\\big],\\qquad
        a = |0\\rangle\\langle1| + \\sqrt2\\,|1\\rangle\\langle2| ,

    integrated as an ordered product of midpoint exponentials, exactly as the
    two-level demo does.  No DRAG correction and no decoherence: this is the
    leakage the bare waveform produces.

    Parameters
    ----------
    coupling : float
        Scales the 1-2 ladder element only.  ``coupling -> 0`` recovers the
        perturbative limit and is how :func:`leakage_amplitude` is verified.
    stride : int
        Subsample the waveform before propagating.  The exported grids carry
        2501 points; ``stride=4`` changes the results in the fourth digit.

    Returns
    -------
    dict
        ``populations`` (3,) at the gate end and ``leakage`` = ``P2``.
    """
    from scipy.linalg import expm

    physical = to_physical(pulse, device)
    time = physical["time"][::stride]
    omega_x = physical["omega_x"][::stride]
    omega_y = physical["omega_y"][::stride]

    lower = np.diag([1.0, coupling * np.sqrt(2.0)], 1)
    bare = np.diag([0.0, 0.0, device.anharmonicity_rate]).astype(complex)

    state = np.array([1.0, 0.0, 0.0], dtype=complex)
    mid_x = 0.5 * (omega_x[:-1] + omega_x[1:])
    mid_y = 0.5 * (omega_y[:-1] + omega_y[1:])
    for step, (x_value, y_value) in zip(np.diff(time), zip(mid_x, mid_y)):
        drive = 0.5 * (
            (x_value - 1j * y_value) * lower
            + (x_value + 1j * y_value) * lower.conj().T
        )
        state = expm(-1j * (bare + drive) * step) @ state
    populations = np.abs(state) ** 2
    return {
        "populations": populations,
        "leakage": float(populations[2]),
    }


def drive_spectrum(pulse, device=DEVICE, span=0.6, points=1201):
    """Complex-envelope spectrum of the drive, in laboratory frequency units.

    Returns the magnitude of ``int Omega(t) e^{i Phi(t)} e^{i 2 pi f t} dt``
    over ``f`` in ``[-span, span]`` GHz.  This is the textbook leakage picture:
    weight at ``f = alpha/2pi`` drives the 1-2 transition.  It omits the
    ``<1|U_01|0>`` factor kept by :func:`leakage_amplitude`, so it is a reading
    aid, not the quantitative estimate.

    Returns
    -------
    dict
        ``frequency`` (GHz) and ``weight`` (rad, i.e. an angle).
    """
    physical = to_physical(pulse, device)
    time = physical["time"]
    drive = physical["omega_x"] + 1j * physical["omega_y"]
    frequency = np.linspace(-span, span, points)
    kernel = np.exp(1j * TWO_PI * np.outer(frequency, time))
    weight = np.abs(np.trapezoid(kernel * drive, time, axis=1))
    return {"frequency": frequency, "weight": weight}


# --- per-pulse summary --------------------------------------------------------


def pulse_metrics(pulse, device=DEVICE, leakage_stride=4):
    """Dimensionless geometry plus the physical cost of running the pulse.

    The dimensionless block is gauge invariant and reproduces the demo
    notebook; the physical block is what the choice of :func:`to_physical`
    turns that geometry into on this device.  Both the perturbative and the
    exact three-level leakage are reported, because at ``Omega_max/|alpha| =
    0.25`` first order is only a screen (see the notebook).
    """
    physical = to_physical(pulse, device)
    # Both noise channels, both orders.  "amplitude" is the multiplicative
    # control error this experiment injects, (Omega_x, Omega_y) ->
    # (1+eps)(Omega_x, Omega_y); "dephasing" is the static detuning channel.
    # Closure carries the first order, the signed area the second.
    curve = db_helpers.error_curve(pulse, noise="amplitude")
    dephasing = db_helpers.error_curve(pulse, noise="dephasing")
    leakage = leakage_amplitude(pulse, device)
    exact = three_level_propagation(pulse, device, stride=leakage_stride)
    gate = db_helpers.time_ordered_gate(pulse, 0.0)

    time = pulse["time_over_Tg"]
    envelope = np.abs(pulse["Tg_omega_envelope"])
    gate_time = physical["gate_time"]

    return {
        # gauge-invariant geometry
        "peak_Tg_omega": physical["peak_tg_omega"],
        "pulse_area": float(np.trapezoid(envelope, time)),
        "energy": float(np.trapezoid(envelope**2, time)),
        "closure": curve["closure"],
        "area_norm": float(np.linalg.norm(curve["area"])),
        "dephasing_closure": dephasing["closure"],
        "dephasing_area_norm": float(np.linalg.norm(dephasing["area"])),
        "zero_error_infidelity": 1.0 - db_helpers.average_gate_fidelity(gate),
        # physical, at fixed peak Rabi rate
        "gate_time_ns": gate_time,
        "drive_over_alpha": leakage["drive_ratio"],
        "leakage_P2_first_order": leakage["population"],
        "leakage_P2": exact["leakage"],
        "Tg_over_T1": gate_time / device.t1,
        "Tg_over_T2": gate_time / device.t2_echo,
    }


#: ``(group title, [(metric key, column title, format)])``.  The grouping is the
#: point of the table: robustness has two independent noise channels and two
#: orders in each, and a pulse can be excellent in one cell and useless in the
#: neighbouring one.
METRIC_GROUPS = (
    ("drive cost", (
        ("peak_Tg_omega", "Tg*Om_max", "{:11.3f}"),
        ("pulse_area", "area/pi", "{:9.3f}"),
    )),
    ("control error", (
        ("closure", "1st |R(T)-R(0)|", "{:17.3e}"),
        ("area_norm", "2nd |A|", "{:12.3e}"),
    )),
    ("dephasing error", (
        ("dephasing_closure", "1st |R(T)-R(0)|", "{:17.3e}"),
        ("dephasing_area_norm", "2nd |A|", "{:12.3e}"),
    )),
    ("on this device", (
        ("zero_error_infidelity", "1-F(eps=0)", "{:12.2e}"),
        ("gate_time_ns", "Tg [ns]", "{:10.2f}"),
        ("leakage_P2", "P2 (exact)", "{:13.3e}"),
    )),
)

METRIC_COLUMNS = tuple(
    column for _, columns in METRIC_GROUPS for column in columns
)


def print_metrics_table(pulses, device=DEVICE):
    """Print the section-1 summary table and return the raw metric dict.

    Columns are grouped: what the drive costs, how the pulse survives a
    multiplicative *control* error, how it survives *dephasing*, and what the
    device charges for it.  Closure is the first-order measure of each channel
    and the signed area the second-order one; a pulse that closes its curve but
    encloses a large area is robust only to leading order.
    """
    metrics = {
        name: pulse_metrics(pulse, device)
        for name, pulse in pulses.items()
    }
    label_width = max(len(name) for name in metrics) + 2

    print(
        f"peak drive fixed at Omega_max/2pi = {device.rabi_max * 1e3:.0f} MHz,"
        f"  alpha/2pi = {device.anharmonicity * 1e3:.0f} MHz,"
        f"  T1 = T2echo = {device.t1 / 1e3:.0f} us"
    )
    print(
        "control error = (Om_x,Om_y) -> (1+eps)(Om_x,Om_y);"
        "  dephasing error = static delta_z."
        "  Closure is 1st order, signed area 2nd."
    )

    widths = [
        sum(len(fmt.format(0.0)) for _, _, fmt in columns)
        for _, columns in METRIC_GROUPS
    ]
    group_row = f"{'':{label_width}s}" + "".join(
        f"{title[:width - 2]:^{width}s}"
        for (title, _), width in zip(METRIC_GROUPS, widths)
    )
    header = f"{'pulse':{label_width}s}" + "".join(
        f"{title:>{len(fmt.format(0.0))}s}"
        for _, title, fmt in METRIC_COLUMNS
    )
    print(group_row)
    print(header)
    print("-" * len(header))
    for name, values in metrics.items():
        row = f"{name:{label_width}s}"
        for key, _, fmt in METRIC_COLUMNS:
            value = values[key]
            if key == "pulse_area":
                value = value / np.pi
            row += fmt.format(value)
        print(row)
    return metrics


#: Compact geometry table for a presentation: one column per geometric cost, in
#: the order they are usually explained -- what the drive costs, then how the
#: pulse survives each noise channel at first and second order.  A subset of
#: :data:`METRIC_GROUPS`, which is the full diagnostic version.
GEOMETRY_COLUMNS = (
    ("gate_time_ns", "Tg [ns]", "{:9.1f}"),
    ("peak_Tg_omega", "Tg*Om_max", "{:11.2f}"),
    ("pulse_area", "int|Om|dt / pi", "{:16.2f}"),
    ("closure", "ctrl closure", "{:14.2e}"),
    ("area_norm", "ctrl |A|", "{:11.2e}"),
    ("dephasing_closure", "deph closure", "{:14.2e}"),
    ("leakage_P2", "P2", "{:10.1e}"),
)


def print_geometry_table(pulses, device=DEVICE, metrics=None,
                         columns=GEOMETRY_COLUMNS):
    """Print the geometric costs of each waveform, one row per pulse.

    The short form of :func:`print_metrics_table`, for readers who want the
    costs rather than the full two-channel, two-order diagnostic.  Everything
    except ``Tg`` and ``P2`` is dimensionless and gauge invariant: rescaling a
    space curve trades ``Tg`` against ``Omega`` and leaves the rest untouched.
    """
    metrics = metrics or {
        name: pulse_metrics(pulse, device) for name, pulse in pulses.items()
    }
    label_width = max(len(name) for name in metrics) + 2
    print(
        f"peak drive fixed at Omega_max/2pi = {device.rabi_max * 1e3:.0f} MHz,"
        f"  alpha/2pi = {device.anharmonicity * 1e3:.0f} MHz"
    )
    header = f"{'pulse':{label_width}s}" + "".join(
        f"{title:>{len(fmt.format(0.0))}s}" for _, title, fmt in columns
    )
    print(header)
    print("-" * len(header))
    for name, values in metrics.items():
        row = f"{name:{label_width}s}"
        for key, _, fmt in columns:
            value = values[key]
            if key == "pulse_area":
                value = value / np.pi
            row += fmt.format(value)
        print(row)
    return metrics


# --- noise model: Lindblad + static detuning ----------------------------------

#: Ladder operators of the three-level transmon.  ``LOWER`` is the truncated
#: annihilation operator, so the 2-1 matrix element carries the ``sqrt(2)`` that
#: makes the 2-1 relaxation rate twice the 1-0 one.
LOWER = np.diag([1.0, np.sqrt(2.0)], 1)
NUMBER = np.diag([0.0, 1.0, 2.0])

#: Drive quadratures, ``H_drive = (Omega_x * DRIVE_X + Omega_y * DRIVE_Y)``.
#: In the qubit block they reduce to ``X/2`` and ``Y/2``, the convention
#: :mod:`db_helpers` uses.
DRIVE_X = 0.5 * (LOWER + LOWER.T)
DRIVE_Y = 0.5j * (LOWER.T - LOWER)

#: Sign convention for a detuning.  A drive detuned by ``delta`` below the 0-1
#: frequency puts level ``n`` at ``-n*delta`` in the rotating frame, which in the
#: qubit block is ``+delta*Z/2`` -- exactly the generator of the ``"dephasing"``
#: error curve of :func:`db_helpers.error_curve`.
DETUNING = -NUMBER

#: Pauli basis of the qubit subspace, embedded in three levels.  Used for the
#: leakage-aware average gate fidelity, where ``{I, X, Y, Z}`` is the orthogonal
#: operator basis Nielsen's formula needs.
QUBIT_PAULIS = np.array([
    np.diag([1.0, 1.0, 0.0]).astype(complex),
    *(
        np.pad(pauli, (0, 1))
        for pauli in (db_helpers.X, db_helpers.Y, db_helpers.Z)
    ),
])


def collapse_operators(device=DEVICE):
    """Lindblad collapse operators of the three-level transmon.

    Two channels, both in rad-free units of 1/ns:

    * relaxation ``sqrt(Gamma1) * a``, whose ladder structure gives the 2-1
      transition twice the 1-0 rate;
    * pure dephasing ``sqrt(2 Gamma_phi) * n``, normalized so that the 0-1
      coherence decays at ``Gamma_phi`` -- i.e. the two-level limit reproduces
      ``1/T2 = 1/(2 T1) + Gamma_phi``.

    Returns
    -------
    list of qutip.Qobj
    """
    import qutip as qt

    return [
        qt.Qobj(np.sqrt(device.gamma1) * LOWER),
        qt.Qobj(np.sqrt(2.0 * device.gamma_phi) * NUMBER),
    ]


def frame_rotation(phase):
    """Frame rotation ``exp(-i phase * n)``, i.e. a virtual Z on the ladder.

    A virtual Z costs nothing in hardware -- it is a phase offset applied to
    every later drive pulse -- so it is a calibration knob, not an error.
    """
    return np.diag(np.exp(-1j * phase * np.array([0.0, 1.0, 2.0])))


def _calibration_terms(device, calibration):
    """Unpack a calibration dict into ``(scale, detuning rate, frame phase)``."""
    calibration = calibration or {}
    detuning_rate = device.static_detuning_rate + TWO_PI * calibration.get(
        "detuning", 0.0
    )
    return (
        calibration.get("scale", 1.0),
        detuning_rate,
        calibration.get("frame_phase", 0.0),
    )


def liouvillian_parts(device=DEVICE, dissipation=True, detuning_rate=None):
    """Pre-assemble the Liouvillian as a constant plus three control terms.

    The Liouvillian is linear in the Hamiltonian, and the Hamiltonian is linear
    in ``(Omega_x, Omega_y, delta)``, so the whole time dependence of one gate
    is four fixed 9x9 matrices and three scalars per time step.  That is what
    keeps :func:`gate_channel` cheap enough to call for every waveform and every
    error setting.

    Parameters
    ----------
    detuning_rate : float, optional
        Constant detuning in rad/ns.  Defaults to the device's static offset;
        :func:`calibrate` passes the static offset *plus* its own drive-frequency
        correction.

    Returns
    -------
    dict of ndarray
        ``constant`` (anharmonicity + static detuning + dissipator) and
        ``omega_x``, ``omega_y``, ``delta`` (the per-unit control Liouvillians),
        all in the column-stacking vectorization qutip uses.
    """
    import qutip as qt

    if detuning_rate is None:
        detuning_rate = device.static_detuning_rate
    bare = (
        np.diag([0.0, 0.0, device.anharmonicity_rate]).astype(complex)
        + detuning_rate * DETUNING
    )
    c_ops = collapse_operators(device) if dissipation else []
    return {
        "constant": qt.liouvillian(qt.Qobj(bare), c_ops).full(),
        "omega_x": qt.liouvillian(qt.Qobj(DRIVE_X)).full(),
        "omega_y": qt.liouvillian(qt.Qobj(DRIVE_Y)).full(),
        "delta": qt.liouvillian(qt.Qobj(DETUNING)).full(),
    }


def _integrate_channel(pulse, epsilon=0.0, device=DEVICE, stride=1,
                       dissipation=True, initial_state=None, calibration=None):
    """Ordered product of midpoint Liouvillian exponentials over one gate.

    Shared implementation of :func:`gate_channel` and
    :func:`channel_trajectory`; the two differ only in what they keep.
    """
    from scipy.linalg import expm

    scale, detuning_rate, frame_phase = _calibration_terms(device, calibration)
    physical = to_physical(pulse, device)
    time = physical["time"][::stride]
    omega_x = scale * (1.0 + epsilon) * physical["omega_x"][::stride]
    omega_y = scale * (1.0 + epsilon) * physical["omega_y"][::stride]
    # The waveform's own detuning field; zero for the resonant pulses, but the
    # static offset above is not the only place a detuning can come from.
    delta = physical["delta"][::stride]

    parts = liouvillian_parts(device, dissipation, detuning_rate)
    frame = frame_rotation(frame_phase)
    frame_channel = np.kron(frame.conj(), frame)
    steps = np.diff(time)
    mid_x = 0.5 * (omega_x[:-1] + omega_x[1:])
    mid_y = 0.5 * (omega_y[:-1] + omega_y[1:])
    mid_delta = 0.5 * (delta[:-1] + delta[1:])

    channel = np.eye(9, dtype=complex)
    trajectory = None
    if initial_state is not None:
        trajectory = np.empty((len(time), 3, 3), dtype=complex)
        trajectory[0] = initial_state

    for index, (step, x_value, y_value, delta_value) in enumerate(
        zip(steps, mid_x, mid_y, mid_delta)
    ):
        generator = (
            parts["constant"]
            + x_value * parts["omega_x"]
            + y_value * parts["omega_y"]
            + delta_value * parts["delta"]
        )
        channel = expm(generator * step) @ channel
        if trajectory is not None:
            # Recorded before the final frame rotation, which is diagonal and
            # therefore leaves every population untouched.
            trajectory[index + 1] = apply_channel(channel, initial_state)
    return {
        "channel": frame_channel @ channel,
        "gate_time": physical["gate_time"],
        "time": time,
        "trajectory": trajectory,
    }


def gate_channel(pulse, epsilon=0.0, device=DEVICE, stride=1, dissipation=True,
                 calibration=None):
    """One gate as a 9x9 quantum channel on the three-level density matrix.

    The Lindblad master equation is integrated as an ordered product of
    midpoint exponentials on the exported sample grid -- the same discretization
    the coherent propagators use, so the coherent limit is comparable digit by
    digit.  Because the result is the *channel* and not a trajectory, the DB
    sequence of section 3 is a matrix power of it.

    Parameters
    ----------
    epsilon : float
        Multiplicative control error on top of the *calibrated* amplitude --
        that is what an unknown residual control error means in an experiment.
        The static detuning and the dissipator are deliberately not scaled by
        it.
    dissipation : bool
        Switch ``T1``/``T2`` off to recover a unitary channel; used for
        validation and for the error budget.
    calibration : dict, optional
        Output of :func:`calibrate`: ``scale``, ``detuning`` (GHz) and
        ``frame_phase``.  Omitting it runs the bare waveform, which on a
        three-level device is *not* an X gate (see :func:`calibrate`).

    Returns
    -------
    ndarray, shape (9, 9)
        Column-stacking (qutip) vectorization, so
        ``vec(rho') = channel @ vec(rho)``.
    """
    return _integrate_channel(
        pulse, epsilon, device, stride, dissipation, None, calibration
    )["channel"]


def channel_trajectory(pulse, epsilon=0.0, device=DEVICE, stride=1,
                       dissipation=True, initial_state=None, calibration=None):
    """Populations along one gate, for plotting.

    Same integration as :func:`gate_channel`, but the state is recorded at every
    sample.  ``initial_state`` defaults to ``|0><0|``.

    Returns
    -------
    dict
        ``time`` (ns), ``populations`` (N, 3) and ``coherence`` = ``|rho_01|``.
    """
    if initial_state is None:
        initial_state = np.diag([1.0, 0.0, 0.0]).astype(complex)
    result = _integrate_channel(
        pulse, epsilon, device, stride, dissipation, initial_state, calibration
    )
    states = result["trajectory"]
    return {
        "time": result["time"],
        "gate_time": result["gate_time"],
        "populations": np.einsum("nii->ni", states).real,
        "coherence": np.abs(states[:, 0, 1]),
    }


def as_qobj(channel):
    """Wrap a 9x9 channel as a qutip superoperator, for qutip's own tools."""
    import qutip as qt

    return qt.Qobj(
        channel, dims=[[[3], [3]], [[3], [3]]], superrep="super"
    )


def apply_channel(channel, operator):
    """Act with a column-stacking superoperator on a 3x3 operator."""
    flat = np.asarray(operator, dtype=complex).ravel(order="F")
    return (channel @ flat).reshape(3, 3, order="F")


def subspace_gate_fidelity(channel, target=db_helpers.U_TARGET):
    """Leakage-aware average gate fidelity of a three-level channel.

    Nielsen's formula for a channel ``Lambda`` and target ``U`` on ``d = 2``
    levels,

    .. math::
        F = \\frac{d^2 + \\sum_k \\mathrm{tr}
            \\big[U P_k U^\\dagger\\,\\Lambda(P_k)\\big]}{d^2 (d+1)},

    with ``{P_k} = {I, X, Y, Z}`` embedded in the three-level space and the
    output projected back onto the qubit block.  Population that ends in
    ``|2>`` therefore leaves the trace of the projected output below one and is
    charged as an error, which is the behaviour we want: leakage is a loss, not
    a relabelling.  For a unitary two-level channel this reduces to the
    ``(|tr U^dag V|^2 + 2)/6`` used by :mod:`db_helpers` (checked in
    :func:`verify_noise_model`).
    """
    total = 0.0
    for pauli in QUBIT_PAULIS:
        output = apply_channel(channel, pauli)[:2, :2]
        rotated = target @ pauli[:2, :2] @ target.conj().T
        total += np.trace(rotated @ output).real
    return float((total + 4.0) / 12.0)


def channel_populations(channel, initial_state=None):
    """Final three-level populations of a channel, from ``|0>`` by default."""
    if initial_state is None:
        initial_state = np.diag([1.0, 0.0, 0.0]).astype(complex)
    return np.einsum("ii->i", apply_channel(channel, initial_state)).real


# --- calibration --------------------------------------------------------------


def three_level_gate(pulse, epsilon=0.0, device=DEVICE, stride=1,
                     calibration=None):
    """Coherent three-level propagator of one gate, as a 3x3 matrix.

    The same discretization as :func:`gate_channel` with the dissipator off,
    but in Hilbert space rather than Liouville space -- nine times smaller, and
    therefore cheap enough to sit inside the calibration optimizer.
    """
    from scipy.linalg import expm

    scale, detuning_rate, frame_phase = _calibration_terms(device, calibration)
    physical = to_physical(pulse, device)
    time = physical["time"][::stride]
    omega_x = scale * (1.0 + epsilon) * physical["omega_x"][::stride]
    omega_y = scale * (1.0 + epsilon) * physical["omega_y"][::stride]
    delta = physical["delta"][::stride]

    bare = (
        np.diag([0.0, 0.0, device.anharmonicity_rate]).astype(complex)
        + detuning_rate * DETUNING
    )
    unitary = np.eye(3, dtype=complex)
    for step, x_value, y_value, delta_value in zip(
        np.diff(time),
        0.5 * (omega_x[:-1] + omega_x[1:]),
        0.5 * (omega_y[:-1] + omega_y[1:]),
        0.5 * (delta[:-1] + delta[1:]),
    ):
        hamiltonian = (
            bare
            + x_value * DRIVE_X
            + y_value * DRIVE_Y
            + delta_value * DETUNING
        )
        unitary = expm(-1j * hamiltonian * step) @ unitary
    return frame_rotation(frame_phase) @ unitary


def block_gate_fidelity(gate, target=db_helpers.U_TARGET,
                        leakage_corrected=False):
    """:func:`subspace_gate_fidelity` for a coherent gate, without the channel.

    Same formula with ``Lambda(P) = W P W^dag`` and ``W`` the qubit block of
    ``gate``; ``W`` is sub-unitary when the pulse leaks, and the missing weight
    is charged as error.

    Parameters
    ----------
    leakage_corrected : bool
        Rescale ``W`` to unit average norm first, which removes the loss of
        population to ``|2>`` and leaves only the rotation error inside the
        qubit subspace.  This is the quantity :func:`calibrate` minimizes: a
        laboratory calibrates amplitude and frequency against a Rabi/Ramsey
        signal in the computational subspace, and letting the optimizer see
        leakage instead makes it chase the narrow spectral nulls of section 1.4,
        which are not reproducible.
    """
    block = gate[:2, :2]
    if leakage_corrected:
        norm = np.sqrt(0.5 * np.trace(block.conj().T @ block).real)
        block = block / max(norm, 1e-12)
    total = 0.0
    for pauli in QUBIT_PAULIS:
        small = pauli[:2, :2]
        rotated = target @ small @ target.conj().T
        total += np.trace(rotated @ block @ small @ block.conj().T).real
    return float((total + 4.0) / 12.0)


def best_frame_phase(gate, target=db_helpers.U_TARGET, coarse=181, refine=61,
                     leakage_corrected=False):
    """Best virtual-Z for one gate, by a coarse scan plus a local refinement.

    Returns ``(phase, fidelity)``.  The scan is over the free frame angle only,
    so this costs a few hundred 2x2 products, not a re-integration.
    """
    def scan(angles):
        values = [
            block_gate_fidelity(
                frame_rotation(angle) @ gate, target, leakage_corrected
            )
            for angle in angles
        ]
        best = int(np.argmax(values))
        return angles[best], values[best]

    grid = np.linspace(-np.pi, np.pi, coarse)
    centre, _ = scan(grid)
    span = grid[1] - grid[0]
    phase, fidelity = scan(np.linspace(centre - span, centre + span, refine))
    return float(phase), float(fidelity)


def calibrate(pulse, device=DEVICE, stride=4, target=db_helpers.U_TARGET,
              scale_window=(0.8, 1.25), detuning_window=(-0.030, 0.030)):
    """Calibrate one waveform against the three-level device.

    The waveforms were designed in the two-level picture, where they are exact
    ``X(pi)`` gates.  On a transmon at ``Omega_max/|alpha| = 0.25`` they are
    not: the ``|2>`` level Stark-shifts the 0-1 transition during the pulse, and
    the resulting rotation axis is tilted out of the equator by of order
    ``0.2`` rad -- several times the ``3%`` control error the experiment
    injects.  Left uncorrected it would dominate every later number, and it is
    not a property of the waveform's robustness but of a missing calibration.

    So we calibrate exactly what a laboratory calibrates, and nothing else:

    * ``scale`` -- the drive amplitude (a Rabi calibration).  The injected
      control error is applied *on top* of it, so ``epsilon`` keeps its meaning
      of an unknown residual;
    * ``detuning`` -- the drive frequency, which absorbs the Stark shift;
    * ``frame_phase`` -- a virtual Z, which is free in hardware.

    Both knobs are bounded: the amplitude to ``scale_window`` and the drive
    frequency to ``detuning_window`` (GHz).  Without bounds the optimizer will
    happily leave the fixed-peak-drive gauge of section 1 -- there is another
    ``pi`` point at ``1.6`` times the amplitude -- and a laboratory Rabi
    calibration does not do that either.  A pulse that ends up *on* a bound is
    reported as such: it means two knobs cannot turn that waveform into an X
    gate at its design amplitude.

    No DRAG and no pulse reshaping: the waveform itself is untouched.  The
    objective is the *leakage-corrected* rotation error inside the qubit
    subspace (see :func:`block_gate_fidelity`), so the knobs fix the rotation
    and are not allowed to chase leakage; what is left over afterwards is a
    cost of the waveform, not of the calibration.

    Returns
    -------
    dict
        ``scale``, ``detuning`` (GHz), ``frame_phase`` (rad), the calibration
        objective before and after (``raw_rotation_error``,
        ``rotation_error``), the full coherent residual ``infidelity`` at
        ``epsilon = 0`` and ``leakage`` (``P2`` of the calibrated gate).
    """
    from scipy.optimize import minimize

    # The static detuning is unknown noise, so the calibration is not allowed to
    # see it.  It is 40 times smaller than the Stark shift the calibration does
    # remove, so this is a matter of principle rather than of numbers.
    device = _configured(device, detuning=False)

    def rotation_error(calibration):
        gate = three_level_gate(pulse, 0.0, device, stride, calibration)
        return 1.0 - best_frame_phase(
            gate, target, leakage_corrected=True
        )[1]

    result = minimize(
        lambda values: rotation_error(
            {"scale": values[0], "detuning": values[1] * 1e-3}
        ),
        np.array([1.0, 0.0]),
        method="Nelder-Mead",
        bounds=(scale_window, tuple(1e3 * value for value in detuning_window)),
        options={"xatol": 1e-6, "fatol": 1e-12, "maxfev": 600},
    )
    calibration = {
        "scale": float(result.x[0]),
        "detuning": float(result.x[1] * 1e-3),
    }
    gate = three_level_gate(pulse, 0.0, device, stride, calibration)
    calibration["frame_phase"] = best_frame_phase(
        gate, target, leakage_corrected=True
    )[0]
    calibrated = three_level_gate(pulse, 0.0, device, stride, calibration)
    calibration["rotation_error"] = 1.0 - block_gate_fidelity(
        calibrated, target, leakage_corrected=True
    )
    calibration["raw_rotation_error"] = rotation_error(None)
    calibration["infidelity"] = 1.0 - block_gate_fidelity(calibrated, target)
    calibration["leakage"] = float(abs(calibrated[2, 0]) ** 2)
    calibration["at_bound"] = bool(
        min(abs(calibration["scale"] - edge) for edge in scale_window) < 1e-3
        or min(
            abs(calibration["detuning"] - edge) for edge in detuning_window
        ) < 1e-6
    )
    return calibration


def calibrate_all(pulses, device=DEVICE, stride=4):
    """Calibrate every waveform; returns ``{name: calibration}``."""
    return {
        name: calibrate(pulse, device, stride)
        for name, pulse in pulses.items()
    }


def print_calibration(pulses, device=DEVICE, calibrations=None, stride=4):
    """Print what calibration had to move, and what it could not fix."""
    calibrations = calibrations or calibrate_all(pulses, device, stride)
    label_width = max(len(name) for name in calibrations) + 2
    print(
        "amplitude, drive frequency and virtual Z calibrated against the "
        "three-level device;\nno DRAG, no reshaping.  The objective is the "
        "rotation error inside the qubit\nsubspace, with leakage divided out; "
        "1-F is the full coherent residual at eps = 0."
    )
    header = (
        f"{'pulse':{label_width}s}{'scale':>9s}{'d_cal [MHz]':>13s}"
        f"{'virtual Z':>11s}{'rot. err before':>17s}{'after':>10s}"
        f"{'1-F':>11s}{'P2':>11s}"
    )
    print(header)
    print("-" * len(header))
    for name, values in calibrations.items():
        print(
            f"{name:{label_width}s}{values['scale']:9.4f}"
            f"{'*' if values.get('at_bound') else ' '}"
            f"{values['detuning'] * 1e3:12.3f}"
            f"{values['frame_phase']:11.4f}"
            f"{values['raw_rotation_error']:17.2e}"
            f"{values['rotation_error']:10.2e}"
            f"{values['infidelity']:11.2e}"
            f"{values['leakage']:11.2e}"
        )
    if any(values.get("at_bound") for values in calibrations.values()):
        print(
            "* on a calibration bound: two knobs cannot make this waveform an "
            "X gate at its design amplitude."
        )
    return calibrations


def print_noise_model(device=DEVICE):
    """Print the frozen error model of section 2.

    Nothing in this table is swept: one number per row, chosen once, and every
    later result is conditional on it.
    """
    rows = (
        ("anharmonicity", f"alpha/2pi = {device.anharmonicity * 1e3:+.0f} MHz",
         "sets the 1-2 detuning, i.e. leakage"),
        ("peak drive", f"Omega_max/2pi = {device.rabi_max * 1e3:.0f} MHz",
         "gauge fixing: every pulse runs at this peak"),
        ("relaxation", f"T1 = {device.t1 / 1e3:.0f} us",
         f"Gamma1 = {device.gamma1 * 1e3:.3e} / us"),
        ("dephasing", f"T2echo = {device.t2_echo / 1e3:.0f} us",
         f"Gamma_phi = 1/T2 - 1/2T1 = {device.gamma_phi * 1e3:.3e} / us"),
        ("static detuning",
         f"delta_z/2pi = {device.static_detuning * 1e3:.3f} MHz",
         "quasi-static, partly echoed by the XX sequence"),
        ("control error", f"epsilon = {device.control_error:+.0%}",
         "(Om_x,Om_y) -> (1+eps)(Om_x,Om_y), both signs"),
        ("sample rate", f"{device.sample_rate:.1f} GS/s",
         "waveform grid; not the integration grid"),
    )
    width = max(len(row[1]) for row in rows) + 2
    print(f"{'quantity':18s}{'value':{width}s}role")
    print("-" * (18 + width + 44))
    for label, value, role in rows:
        print(f"{label:18s}{value:{width}s}{role}")
    print()
    print(
        "collapse operators:  sqrt(Gamma1) * a   with  a = |0><1| + sqrt2 |1><2|"
    )
    print(
        "                     sqrt(2 Gamma_phi) * n   with  n = diag(0,1,2)"
    )


#: Error-budget configurations.  Each turns one term of the model on or off, so
#: the columns of :func:`print_error_budget` add up to a decomposition rather
#: than a list of unrelated numbers.
BUDGET_CONFIGS = (
    ("coherent, eps=0", dict(epsilon=0.0, dissipation=False, static=False)),
    ("T1/T2 only", dict(epsilon=0.0, dissipation=True, static=False)),
    ("delta_z only", dict(epsilon=0.0, dissipation=False, static=True)),
    ("full, eps=0", dict(epsilon=0.0, dissipation=True, static=True)),
    ("full, eps=+3%", dict(epsilon=+0.03, dissipation=True, static=True)),
    ("full, eps=-3%", dict(epsilon=-0.03, dissipation=True, static=True)),
)


def _configured(device, detuning=True):
    """Copy of ``device`` with the static detuning switched off if asked."""
    from dataclasses import replace

    return device if detuning else replace(device, static_detuning=0.0)


def budget_channels(pulses, device=DEVICE, stride=1, configs=BUDGET_CONFIGS,
                    calibrations=None):
    """Build one gate channel per waveform per configuration.

    The calibration is held fixed across configurations -- it is done once, at
    ``epsilon = 0`` and without dissipation, exactly as in an experiment -- so
    the columns differ only by the error term they switch on.

    Returns
    -------
    dict
        ``{waveform: {configuration label: channel}}``.
    """
    calibrations = calibrations or calibrate_all(pulses, device)
    return {
        name: {
            label: gate_channel(
                pulse,
                epsilon=settings["epsilon"],
                device=_configured(device, settings["static"]),
                stride=stride,
                dissipation=settings["dissipation"],
                calibration=calibrations[name],
            )
            for label, settings in configs
        }
        for name, pulse in pulses.items()
    }


def print_error_budget(pulses, device=DEVICE, stride=1, channels=None,
                       configs=BUDGET_CONFIGS, calibrations=None):
    """Print ``1 - F`` per waveform with one term of the model at a time.

    The first column is the coherent gate itself, where the only error is
    leakage; the next two add ``T1``/``T2`` and the static detuning
    separately; the last three are the full model at the three control-error
    settings the experiment uses.  Comparing ``full, eps=+-3%`` against
    ``full, eps=0`` isolates what section 1's control-error geometry buys, and
    comparing ``T1/T2 only`` across rows shows what its gate time costs.
    """
    channels = channels or budget_channels(
        pulses, device, stride, configs, calibrations
    )
    labels = [label for label, _ in configs]
    width = max(len(label) for label in labels) + 3
    label_width = max(len(name) for name in channels) + 2

    print("1 - F_avg on the qubit subspace (leakage counted as loss)")
    header = f"{'pulse':{label_width}s}{'Tg [ns]':>9s}" + "".join(
        f"{label:>{width}s}" for label in labels
    )
    print(header)
    print("-" * len(header))
    budget = {}
    for name, pulse in pulses.items():
        gate_time = to_physical(pulse, device)["gate_time"]
        row = f"{name:{label_width}s}{gate_time:9.1f}"
        budget[name] = {"gate_time_ns": gate_time}
        for label in labels:
            infidelity = 1.0 - subspace_gate_fidelity(channels[name][label])
            budget[name][label] = infidelity
            row += f"{infidelity:>{width}.2e}"
        print(row)
    return budget


def verify_dephasing_channel(pulses, device=DEVICE, stride=2,
                             detunings=(0.05e-3, 0.1e-3, 0.2e-3),
                             calibrations=None):
    """Tie the static-detuning channel back to section 1's error-curve closure.

    A first-order-in-``delta_z`` argument predicts the whole effect from the
    geometry: the error rotation angle is ``closure * Tg * delta_z`` with
    ``closure`` the *dephasing* error-curve closure of section 1 (dimensionless,
    normalized to ``Tg = 1``), and a small rotation by ``theta`` costs
    ``1 - F = theta^2/6``.  Measuring the infidelity increment at several
    ``delta_z`` therefore tests two things at once: the quadratic scaling, and
    whether the geometric closure really is the coefficient.

    Dissipation is off throughout, so the increment isolates the static channel.
    The prediction is a two-level statement, so it only applies where leakage is
    small; pulses whose coherent residual is dominated by ``|2>`` are printed
    anyway, and flagged.
    """
    calibrations = calibrations or calibrate_all(pulses, device)
    print(
        "static-detuning channel against the section-1 dephasing closure;\n"
        "prediction  1-F = (closure * Tg * delta_z)^2 / 6,  dissipation off."
    )
    header = (
        f"{'pulse':26s}{'closure':>10s}{'P2':>10s}"
        + "".join(
            f"{f'meas/pred @{d * 1e6:.0f}kHz':>21s}" for d in detunings
        )
    )
    print(header)
    print("-" * len(header))
    results = {}
    for name, pulse in pulses.items():
        calibration = calibrations[name]
        closure = db_helpers.error_curve(pulse, noise="dephasing")["closure"]
        gate_time = to_physical(pulse, device)["gate_time"]
        reference = gate_channel(
            pulse, 0.0, _configured(device, detuning=False), stride,
            dissipation=False, calibration=calibration,
        )
        baseline = 1.0 - subspace_gate_fidelity(reference)
        leakage = channel_populations(reference)[2]
        row = f"{name:26s}{closure:10.3e}{leakage:10.1e}"
        ratios = []
        for detuning in detunings:
            from dataclasses import replace

            shifted = gate_channel(
                pulse, 0.0, replace(device, static_detuning=detuning), stride,
                dissipation=False, calibration=calibration,
            )
            measured = 1.0 - subspace_gate_fidelity(shifted) - baseline
            angle = closure * gate_time * TWO_PI * detuning
            predicted = angle**2 / 6.0
            ratios.append(measured / predicted)
            row += f"{measured:12.2e}/{predicted:8.1e}"
        results[name] = {"closure": closure, "ratios": ratios}
        print(row + ("   <- leakage-dominated" if leakage > 1e-3 else ""))
    return results


def verify_noise_model(pulses, device=DEVICE, strides=(1, 2, 4),
                       idle_time=2_000.0, calibrations=None):
    """Check the channel integrator: right, then accurate.

    Four independent checks, each isolating one thing that could be wrong:

    1. **Fidelity formula.** A two-level unitary channel built by hand from
       :func:`db_helpers.time_ordered_gate` must give the same number through
       :func:`subspace_gate_fidelity` as through the closed-form
       ``(|tr U^dag V|^2 + 2)/6``.  This also pins the vectorization
       convention.
    2. **Coherent limit.** With the dissipator and the static detuning off, the
       channel must reproduce :func:`three_level_propagation`, which was written
       independently as a state propagation.
    3. **Idle decay.** With no drive, the analytic answers are known:
       ``P1(t) = e^{-t/T1}``, ``|rho_01(t)| = e^{-t/T2}`` and
       ``P2(t) = e^{-2 t/T1}``.
    4. **Hilbert space against Liouville space.** The cheap 3x3 propagator that
       the calibrator uses must agree with the 9x9 channel with its dissipator
       switched off.
    5. **Discretization.** The gate infidelity at the working error setting must
       be stable against subsampling the grid.
    6. **Complete positivity** and trace preservation of the full channel.

    Every check prints its residual; nothing is asserted silently.
    """
    import qutip as qt
    from scipy.linalg import expm

    calibrations = calibrations or calibrate_all(pulses, device)

    print("1. average gate fidelity formula vs the two-level closed form")
    print(f"{'pulse':26s}{'F (channel)':>14s}{'F (closed form)':>17s}"
          f"{'difference':>13s}")
    for name, pulse in pulses.items():
        gate = db_helpers.time_ordered_gate(pulse, device.control_error)
        embedded = np.pad(gate, (0, 1))
        embedded[2, 2] = 1.0
        # vec(U rho U^dag) = (U* kron U) vec(rho) in column stacking.
        unitary_channel = np.kron(embedded.conj(), embedded)
        from_channel = subspace_gate_fidelity(unitary_channel)
        closed_form = db_helpers.average_gate_fidelity(gate)
        print(f"{name:26s}{from_channel:14.10f}{closed_form:17.10f}"
              f"{from_channel - closed_form:13.2e}")

    print()
    print("2. coherent limit vs the independent three-level propagation")
    print(f"{'pulse':26s}{'P2 (channel)':>15s}{'P2 (state)':>13s}"
          f"{'rel. diff':>12s}{'|tr-1|':>10s}")
    coherent_device = _configured(device, detuning=False)
    for name, pulse in pulses.items():
        channel = gate_channel(
            pulse, 0.0, coherent_device, stride=1, dissipation=False
        )
        populations = channel_populations(channel)
        exact = three_level_propagation(pulse, device, stride=1)
        trace_error = abs(populations.sum() - 1.0)
        reference = exact["leakage"]
        relative = abs(populations[2] - reference) / max(reference, 1e-300)
        print(f"{name:26s}{populations[2]:15.6e}{reference:13.6e}"
              f"{relative:12.2e}{trace_error:10.1e}")

    print()
    print(f"3. idle decay over {idle_time:.0f} ns, no drive, against the "
          "analytic Lindblad solution")
    parts = liouvillian_parts(device, dissipation=True)
    idle = expm(parts["constant"] * idle_time)
    p1 = apply_channel(idle, np.diag([0.0, 1.0, 0.0]).astype(complex))[1, 1].real
    p2 = apply_channel(idle, np.diag([0.0, 0.0, 1.0]).astype(complex))[2, 2].real
    superposition = 0.5 * np.array(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=complex
    )
    coherence = abs(apply_channel(idle, superposition)[0, 1])
    for label, value, reference in (
        ("P1 from |1>", p1, np.exp(-idle_time / device.t1)),
        ("P2 from |2>", p2, np.exp(-2.0 * idle_time / device.t1)),
        ("|rho_01|", 2.0 * coherence, np.exp(-idle_time / device.t2_echo)),
    ):
        print(f"{label:14s}{value:14.10f}  analytic {reference:14.10f}"
              f"  diff {value - reference:9.2e}")

    print()
    print("4. calibrated coherent gate: 3x3 propagator vs the 9x9 channel")
    print(f"{'pulse':26s}{'1-F (3x3)':>14s}{'1-F (9x9)':>14s}"
          f"{'difference':>13s}")
    for name, pulse in pulses.items():
        calibration = calibrations[name]
        gate = three_level_gate(pulse, 0.0, device, 1, calibration)
        channel = gate_channel(
            pulse, 0.0, device, stride=1, dissipation=False,
            calibration=calibration,
        )
        small = 1.0 - block_gate_fidelity(gate)
        large = 1.0 - subspace_gate_fidelity(channel)
        print(f"{name:26s}{small:14.6e}{large:14.6e}{small - large:13.2e}")

    print()
    print(f"5. discretization: 1 - F at eps = {device.control_error:+.0%}, "
          "full model, calibrated")
    header = f"{'pulse':26s}" + "".join(
        f"{f'stride={s}':>15s}" for s in strides
    )
    print(header)
    for name, pulse in pulses.items():
        row = f"{name:26s}"
        for stride in strides:
            channel = gate_channel(
                pulse, device.control_error, device, stride=stride,
                calibration=calibrations[name],
            )
            row += f"{1.0 - subspace_gate_fidelity(channel):15.6e}"
        print(row)

    print()
    print("6. complete positivity and trace preservation of the full channel")
    print(f"{'pulse':26s}{'min eig(Choi)':>15s}{'|tr-1| from |0>':>17s}")
    for name, pulse in pulses.items():
        channel = gate_channel(
            pulse, device.control_error, device,
            calibration=calibrations[name],
        )
        choi = qt.to_choi(as_qobj(channel)).full()
        eigenvalues = np.linalg.eigvalsh(0.5 * (choi + choi.conj().T))
        trace_error = abs(channel_populations(channel).sum() - 1.0)
        print(f"{name:26s}{eigenvalues.min():15.2e}{trace_error:17.1e}")


# --- DB sequence --------------------------------------------------------------

#: Cycles of the DB sequence.  One cycle is the gate *pair* ``XX``, so an ideal
#: sequence returns to ``|0>`` after every cycle and the readout is the return
#: probability ``P_0(n)``.  Kept at the two-level demo's value so the traces can
#: be compared against it directly.
N_CYCLES = 60

#: Sequence configurations.  ``coherent`` isolates the control-error
#: oscillation: the dissipator and the static detuning are off, so the only
#: thing left that moves ``P_0`` is ``epsilon`` (leakage cannot be switched off
#: -- the device has three levels in every configuration).  ``full, eps=0`` is
#: the decay envelope the same sequence has *without* any control error, and the
#: gap between the two ``full, eps=+-3%`` traces and that envelope is what a DB
#: experiment actually reads out.
DB_CONFIGS = (
    ("coherent, eps=0", dict(epsilon=0.0, dissipation=False, static=False)),
    ("coherent, eps=+3%", dict(epsilon=+0.03, dissipation=False, static=False)),
    ("coherent, eps=-3%", dict(epsilon=-0.03, dissipation=False, static=False)),
    ("full, eps=0", dict(epsilon=0.0, dissipation=True, static=True)),
    ("full, eps=+3%", dict(epsilon=+0.03, dissipation=True, static=True)),
    ("full, eps=-3%", dict(epsilon=-0.03, dissipation=True, static=True)),
)

#: Labels :func:`print_db_table` and :func:`plot_db_traces` read out of
#: :data:`DB_CONFIGS`.  Both ``eps = 0`` rows are baselines to be divided out,
#: not results: the coherent one carries the accumulated leakage, which drifts
#: ``P_0`` down on its own and would otherwise be counted as a control-error
#: signal.
#: A DB signal saturates: once the accumulated rotation error passes ``pi/2`` the
#: trace covers the full range of ``P_0`` and stops ordering the waveforms.  The
#: cycle count at which the signal first crosses this threshold does not
#: saturate, and is reported alongside it.
DB_SIGNAL_THRESHOLD = 0.05

DB_LEAKAGE_FLOOR = "coherent, eps=0"
DB_ENVELOPE = "full, eps=0"
DB_FULL = ("full, eps=+3%", "full, eps=-3%")
DB_COHERENT = ("coherent, eps=+3%", "coherent, eps=-3%")


def db_sequence(channel, cycles=N_CYCLES, initial_state=None):
    """Three-level populations after ``n`` DB cycles, for ``n = 0 ... cycles``.

    A cycle is the gate pair, so the sequence channel is ``(channel @ channel)``
    raised to the ``n``-th power; nothing is re-integrated.  The gate channel
    already carries the virtual-Z frame update as a left multiplication, and
    ``(Z U)^n`` is exactly what an experiment does when it shifts the phase of
    every subsequent drive, so the matrix power is the physical sequence and not
    an approximation of it.

    Parameters
    ----------
    channel : ndarray, shape (9, 9)
        One calibrated gate, from :func:`gate_channel`.
    initial_state : ndarray, optional
        Defaults to ``|0><0|``.

    Returns
    -------
    ndarray, shape (cycles + 1, 3)
        Populations ``(P_0, P_1, P_2)``, row ``n`` after ``n`` cycles.  Row 0 is
        the input state, so the trace starts at ``P_0 = 1``.
    """
    if initial_state is None:
        initial_state = np.diag([1.0, 0.0, 0.0]).astype(complex)
    pair = channel @ channel
    state = np.asarray(initial_state, dtype=complex)
    populations = np.empty((cycles + 1, 3))
    populations[0] = np.einsum("ii->i", state).real
    for index in range(cycles):
        state = apply_channel(pair, state)
        populations[index + 1] = np.einsum("ii->i", state).real
    return populations


def db_traces(pulses, device=DEVICE, cycles=N_CYCLES, stride=1,
              configs=DB_CONFIGS, calibrations=None):
    """Run the DB sequence for every waveform in every configuration.

    The calibration is done once per waveform -- at ``epsilon = 0``, without
    dissipation, exactly as in an experiment -- and then held fixed, so the
    configurations differ only by the error term they switch on.

    Returns
    -------
    dict
        ``{waveform: {"gate_time_ns", "cycles", "elapsed_us",
        <configuration label>: populations}}``, with ``elapsed_us`` the physical
        duration ``2 n T_g`` of the sequence in microseconds.
    """
    calibrations = calibrations or calibrate_all(pulses, device)
    cycle_numbers = np.arange(cycles + 1)
    traces = {}
    for name, pulse in pulses.items():
        gate_time = to_physical(pulse, device)["gate_time"]
        record = {
            "gate_time_ns": gate_time,
            "cycles": cycle_numbers,
            "elapsed_us": 2.0 * cycle_numbers * gate_time / 1e3,
        }
        for label, settings in configs:
            channel = gate_channel(
                pulse,
                epsilon=settings["epsilon"],
                device=_configured(device, settings["static"]),
                stride=stride,
                dissipation=settings["dissipation"],
                calibration=calibrations[name],
            )
            record[label] = db_sequence(channel, cycles)
        traces[name] = record
    return traces


def _cycles_to_threshold(signals, threshold=DB_SIGNAL_THRESHOLD):
    """Earliest cycle at which any of ``signals`` reaches ``threshold``.

    ``nan`` if none of them ever does, i.e. the sequence never resolved the
    error at this length.
    """
    crossings = [
        int(found[0])
        for found in (
            np.flatnonzero(np.asarray(signal) >= threshold)
            for signal in signals
        )
        if found.size
    ]
    return float(min(crossings)) if crossings else float("nan")


def db_readout(trace):
    """The two readouts of one waveform's DB run, plus what they are read against.

    A transmon DB trace is not a single number: decoherence pulls ``P_0`` down
    whether or not the control is robust, so the coherent signal has to be
    separated from the envelope it rides on.

    Returns
    -------
    dict
        ``oscillation``
            Largest ``|P_0(n, +-3%) - P_0(n, 0)|`` over the sequence in the
            *coherent* configuration: the control-error signal alone, with the
            incoherent terms switched off and the accumulated leakage divided
            out.  This is the transmon counterpart of the two-level demo's
            ``Delta P0``.
        ``cycles_to_osc``, ``cycles_to_signal``
            First cycle at which that signal, and the measurable one below,
            reach :data:`DB_SIGNAL_THRESHOLD`.  A DB signal saturates once the
            accumulated error passes ``pi/2``, and at ``N = 60`` most of these
            waveforms are saturated; these two columns keep ordering them after
            the amplitudes no longer do.  ``nan`` means the sequence never
            resolved the error at this length.
        ``envelope_loss``
            ``1 - P_0`` at the last cycle with ``epsilon = 0`` and the full
            model on: decoherence, static detuning and leakage, no control
            error.  This is the floor the signal has to be seen against.
        ``delta_p0``
            Peak-to-peak of the *full* ``P_0(n)``, which mixes the two above and
            is what a raw trace shows.
        ``signal``
            ``|P_0(N, +-3%) - P_0(N, 0)|`` at the last cycle -- the envelope
            divided out at a single point, which is the actual DB discriminator.
        ``leakage``
            ``P_2`` at the last cycle, full model.
        All of them are the worst case over the two error signs.
    """
    cycles = len(trace["cycles"]) - 1
    envelope = trace[DB_ENVELOPE][:, 0]
    floor = trace[DB_LEAKAGE_FLOOR][:, 0]
    coherent = [np.abs(trace[label][:, 0] - floor) for label in DB_COHERENT]
    full = [np.abs(trace[label][:, 0] - envelope) for label in DB_FULL]
    return {
        "gate_time_ns": trace["gate_time_ns"],
        "elapsed_us": float(trace["elapsed_us"][-1]),
        "oscillation": max(float(signal.max()) for signal in coherent),
        "cycles_to_osc": _cycles_to_threshold(coherent),
        "envelope_loss": float(1.0 - envelope[-1]),
        "delta_p0": max(
            float(np.ptp(trace[label][:, 0])) for label in DB_FULL
        ),
        "signal": max(float(signal[cycles]) for signal in full),
        "cycles_to_signal": _cycles_to_threshold(full),
        "leakage": max(
            float(trace[label][cycles, 2]) for label in DB_FULL
        ),
    }


def print_db_table(traces):
    """Print the DB readouts, one row per waveform.

    ``oscillation`` and ``envelope loss`` are the two independent readings; the
    remaining columns say how they combine in a raw trace.  Comparing
    ``oscillation`` down a column reproduces the ordering of section 1's
    control-error closure, and ``envelope loss`` the ordering of the gate times.
    """
    readouts = {name: db_readout(trace) for name, trace in traces.items()}
    cycles = len(next(iter(traces.values()))["cycles"]) - 1
    label_width = max(len(name) for name in traces) + 2

    percent = f"{DB_SIGNAL_THRESHOLD:.0%}"

    def cycle_count(value):
        return "  -" if np.isnan(value) else f"{int(value):3d}"

    print(f"DB readout after {cycles} XX cycles ({2 * cycles} gates)")
    header = (
        f"{'pulse':{label_width}s}{'Tg [ns]':>9s}{'seq [us]':>10s}"
        f"{'oscillation':>14s}{f'n@{percent} coh':>12s}"
        f"{'envelope loss':>15s}"
        f"{'Delta P0':>11s}{'signal at N':>13s}{f'n@{percent} full':>13s}"
        f"{'P2(N)':>10s}"
    )
    print(header)
    print("-" * len(header))
    for name, readout in readouts.items():
        print(
            f"{name:{label_width}s}{readout['gate_time_ns']:9.1f}"
            f"{readout['elapsed_us']:10.2f}"
            f"{readout['oscillation']:14.3e}"
            f"{cycle_count(readout['cycles_to_osc']):>12s}"
            f"{readout['envelope_loss']:15.3e}"
            f"{readout['delta_p0']:11.3e}{readout['signal']:13.3e}"
            f"{cycle_count(readout['cycles_to_signal']):>13s}"
            f"{readout['leakage']:10.2e}"
        )
    print()
    print(
        "oscillation   = max_n |P0(n,+-3%) - P0(n,0)| with the dissipator and\n"
        "                delta_z off, so only the control error is left;\n"
        f"n@{percent}         = first cycle at which that signal reaches "
        f"{percent}; it keeps ordering\n"
        "                the waveforms after the amplitudes have saturated "
        "('-' = never);\n"
        "envelope loss = 1 - P0(N) at eps = 0 with the full model on;\n"
        "Delta P0      = peak-to-peak of the raw full trace, as in the "
        "two-level demo;\n"
        "signal at N   = |P0(N, +-3%) - P0(N, 0)|, i.e. the envelope divided out."
    )
    return readouts


def verify_db_sequence(pulses, device=DEVICE, traces=None, cycles=N_CYCLES,
                       strides=(1, 2), check_cycles=(1, 2, 4, 8),
                       calibrations=None):
    """Check the sequence layer: right, then accurate.

    Five checks, each isolating one thing the sequence layer could get wrong:

    1. **Vectorization and the power itself.** ``qutip`` is asked to raise its
       own superoperator to the same power and propagate a ``Qobj`` state; it
       carries its own ``dims``/``superrep`` bookkeeping, so agreeing with it
       tests the convention independently of this module.
    2. **Coherent limit against the two-level demo.** With the dissipator and
       the static detuning off, the trace should reproduce the demo's
       ``Delta P0`` up to what three levels and the calibration add.  Not
       asserted -- the two models genuinely differ -- but the ordering and the
       order of magnitude must survive.
    3. **How each channel accumulates.** ``1 - F`` of the whole sequence
       (target ``X^{2n} = I``) is fitted to a power law in ``n`` for the two
       error terms separately.  An incoherent term adds in probability and gives
       slope 1; a coherent one adds in amplitude and gives slope 2; a term the
       ``XX`` train echoes away gives slope 0.  This is the sequence-level test
       of section 2.5's claim that the static detuning is the coherent one.
    4. **Sequence against single gate.** The coherent signal must be section 2's
       single-gate rotation error added in amplitude, ``sin^2(n theta_g)`` with
       ``theta_g`` read off the per-gate infidelity increment.  This is the one
       check that crosses layers instead of testing the sequence against itself.
    5. **Discretization.** The readouts must be stable against subsampling the
       integration grid.

    Every check prints its residual; nothing is asserted silently.
    """
    import qutip as qt

    calibrations = calibrations or calibrate_all(pulses, device)
    traces = traces or db_traces(
        pulses, device, cycles, calibrations=calibrations
    )

    print(f"1. matrix power vs qutip's own superoperator power, n = {cycles}")
    print(f"{'pulse':26s}{'P0 (numpy)':>14s}{'P0 (qutip)':>14s}"
          f"{'difference':>13s}")
    initial = qt.basis(3, 0) * qt.basis(3, 0).dag()
    for name, pulse in pulses.items():
        channel = gate_channel(
            pulse, device.control_error, device,
            calibration=calibrations[name],
        )
        ours = db_sequence(channel, cycles)[-1, 0]
        superoperator = as_qobj(channel @ channel) ** cycles
        theirs = qt.vector_to_operator(
            superoperator * qt.operator_to_vector(initial)
        ).full()[0, 0].real
        print(f"{name:26s}{ours:14.10f}{theirs:14.10f}{ours - theirs:13.2e}")

    print()
    print("2. coherent limit vs the two-level demo's Delta P0")
    print(f"{'pulse':26s}{'ptp (3 level)':>15s}{'ptp (2 level)':>15s}"
          f"{'ratio':>10s}")
    for name, pulse in pulses.items():
        ours = max(
            float(np.ptp(traces[name][label][:, 0])) for label in DB_COHERENT
        )
        two_level = max(
            float(
                np.ptp(
                    db_helpers.db_trace(
                        db_helpers.time_ordered_gate(pulse, sign * device.control_error),
                        cycles,
                    )
                )
            )
            for sign in (+1, -1)
        )
        ratio = ours / two_level if two_level > 0 else float("nan")
        print(f"{name:26s}{ours:15.3e}{two_level:15.3e}{ratio:10.2e}")
    print(
        "   A ratio far above one is not a numerical discrepancy: on three "
        "levels the injected\n   epsilon also rescales the Stark shift, i.e. it "
        "adds an epsilon-dependent Z term that\n   the two-level control-error "
        "curve does not contain and cannot be robust against."
    )

    print()
    print("3. accumulation over the sequence: fitted slope of the 1-F "
          f"*increment* vs n at n = {check_cycles}")
    print("   (1 = incoherent, adds in probability;  2 = coherent, adds in "
          "amplitude;  0 = echoed away)")
    print("   The leakage-only sequence of the same length is subtracted, "
          "otherwise every\n   slope is a mixture of the term under test and "
          "the leakage floor.  The XX train\n   echoes delta_z, so its residual "
          "need not be first order in delta_z and its slope\n   need not be 2; "
          "a negative increment (slope printed as nan) means the echoed\n   "
          "sequence is *better* than the leakage floor at those n.")
    identity = np.eye(2, dtype=complex)
    terms = (
        ("T1/T2 only", dict(epsilon=0.0, dissipation=True, static=False)),
        ("delta_z only", dict(epsilon=0.0, dissipation=False, static=True)),
        ("eps=+3% only", dict(epsilon=+0.03, dissipation=False, static=False)),
    )
    baseline_settings = dict(epsilon=0.0, dissipation=False, static=False)

    def sequence_infidelities(pulse, calibration, settings):
        """``1 - F`` of the ``2n``-gate sequence, target ``X^{2n} = I``."""
        channel = gate_channel(
            pulse, settings["epsilon"],
            _configured(device, settings["static"]),
            stride=1, dissipation=settings["dissipation"],
            calibration=calibration,
        )
        pair = channel @ channel
        return np.array([
            1.0 - subspace_gate_fidelity(
                np.linalg.matrix_power(pair, count), target=identity
            )
            for count in check_cycles
        ])

    header = f"{'pulse':26s}" + "".join(
        f"{f'{label}: slope':>21s}{f'inc(n={check_cycles[-1]})':>14s}"
        for label, _ in terms
    )
    print(header)
    for name, pulse in pulses.items():
        calibration = calibrations[name]
        baseline = sequence_infidelities(pulse, calibration, baseline_settings)
        row = f"{name:26s}"
        for _, settings in terms:
            increment = sequence_infidelities(pulse, calibration, settings)
            increment = increment - baseline
            if increment.min() <= 0.0:
                slope = float("nan")
            else:
                slope = np.polyfit(
                    np.log(check_cycles), np.log(increment), 1
                )[0]
            row += f"{slope:21.2f}{increment[-1]:14.2e}"
        print(row)

    print()
    print("4. the sequence signal against the single-gate error of section 2")
    print("   A per-gate over-rotation theta_g costs 1-F = theta_g^2/6, and the "
          "XX pair leaves\n   2 theta_g behind, so after n cycles "
          "P0 = cos^2(n theta_g) and the signal is\n   sin^2(n theta_g).  "
          "theta_g is taken from the coherent single-gate infidelity\n   "
          "increment; the prediction is exact only while the error is a pure "
          "over-rotation\n   about the drive axis, which is why the ratio drifts "
          "where epsilon also tilts the axis.")
    header = (
        f"{'pulse':26s}{'theta_g [rad]':>14s}"
        + "".join(
            f"{f'meas/pred n={count}':>19s}" for count in check_cycles
        )
    )
    print(header)
    for name, pulse in pulses.items():
        calibration = calibrations[name]
        coherent = _configured(device, detuning=False)
        infidelities = [
            1.0 - subspace_gate_fidelity(
                gate_channel(
                    pulse, epsilon, coherent, stride=1, dissipation=False,
                    calibration=calibration,
                )
            )
            for epsilon in (0.0, device.control_error)
        ]
        theta = np.sqrt(6.0 * max(infidelities[1] - infidelities[0], 0.0))
        floor = traces[name][DB_LEAKAGE_FLOOR][:, 0]
        signal = np.abs(traces[name][DB_COHERENT[0]][:, 0] - floor)
        row = f"{name:26s}{theta:14.4f}"
        for count in check_cycles:
            predicted = np.sin(count * theta) ** 2
            ratio = signal[count] / predicted if predicted > 0 else float("nan")
            row += f"{ratio:19.2f}"
        print(row)

    print()
    print("5. discretization: the two readouts against the integration stride")
    header = f"{'pulse':26s}" + "".join(
        f"{f'osc (s={s})':>14s}{f'env (s={s})':>14s}" for s in strides
    )
    print(header)
    for name, pulse in pulses.items():
        row = f"{name:26s}"
        for stride in strides:
            single = db_traces(
                {name: pulse}, device, cycles, stride=stride,
                calibrations={name: calibrations[name]},
            )[name]
            readout = db_readout(single)
            row += f"{readout['oscillation']:14.4e}"
            row += f"{readout['envelope_loss']:14.4e}"
        print(row)


# --- plots --------------------------------------------------------------------


#: Anharmonicities to sweep when reporting leakage, in GHz.  A single value is
#: not safe to quote: the drive spectrum has deep, narrow nulls, and whether
#: ``alpha`` happens to land in one is an accident of the numbers.
ANHARMONICITY_SWEEP = (-0.170, -0.185, -0.200, -0.215, -0.230, -0.250)


def leakage_vs_anharmonicity(pulses, device=DEVICE,
                             anharmonicities=ANHARMONICITY_SWEEP, stride=4):
    """Exact three-level leakage across a band of anharmonicities.

    The peak Rabi rate stays fixed, so each pulse keeps its gate time and only
    the 1-2 detuning moves.  Reporting the band rather than one number is the
    honest form: individual values swing by two orders of magnitude as
    ``alpha`` slides across a spectral null, while the *ordering* between
    waveforms does not.

    Returns
    -------
    dict
        ``{name: {"values": array, "min": float, "max": float}}``.
    """
    results = {}
    for name, pulse in pulses.items():
        values = np.array([
            three_level_propagation(
                pulse,
                TransmonDevice(
                    anharmonicity=alpha,
                    rabi_max=device.rabi_max,
                    t1=device.t1,
                    t2_echo=device.t2_echo,
                    sample_rate=device.sample_rate,
                ),
                stride=stride,
            )["leakage"]
            for alpha in anharmonicities
        ])
        results[name] = {
            "values": values,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return results


def print_leakage_band(pulses, device=DEVICE,
                       anharmonicities=ANHARMONICITY_SWEEP):
    """Print :func:`leakage_vs_anharmonicity` as a table."""
    results = leakage_vs_anharmonicity(pulses, device, anharmonicities)
    header = f"{'pulse':26s}" + "".join(
        f"{alpha * 1e3:>13.0f}" for alpha in anharmonicities
    )
    print("exact three-level P2, alpha/2pi [MHz] across the top")
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        print(f"{name:26s}" + "".join(f"{v:13.3e}" for v in result["values"]))
    print()
    print(f"{'pulse':26s}{'worst case P2':>15s}")
    for name, result in results.items():
        print(f"{name:26s}{result['max']:15.3e}")
    return results


def verify_leakage_formula(pulses, device=DEVICE, couplings=(1.0, 0.3, 0.1, 0.03),
                           stride=4):
    """Check :func:`leakage_amplitude` against the exact propagation.

    Scaling the 1-2 ladder element by ``lambda`` makes the exact leakage
    ``lambda^2 |c_2|^2 + O(lambda^4)``, so ``P2(lambda)/lambda^2`` must
    approach the perturbative value.  This isolates "is the formula right"
    from "is first order accurate here", which are different questions at
    ``Omega_max/|alpha| = 0.25``.
    """
    print(
        f"{'pulse':26s}{'|c2|^2':>15s}"
        + "".join(f"{f'P2/l^2, l={c:g}':>16s}" for c in couplings)
    )
    for name, pulse in pulses.items():
        row = f"{name:26s}{leakage_amplitude(pulse, device)['population']:15.6e}"
        for coupling in couplings:
            exact = three_level_propagation(
                pulse, device, coupling=coupling, stride=stride
            )
            row += f"{exact['leakage'] / coupling**2:16.6e}"
        print(row)


def plot_physical_controls(pulses, device=DEVICE, colors=None):
    """Overlay the quadratures of every pulse on one laboratory time axis.

    Everything is drawn in MHz (``Omega/2pi``) against ns, with the shared peak
    drive marked, so the comparison reads directly as "same amplitude,
    different duration".
    """
    import matplotlib.pyplot as plt

    colors = colors or color_cycle(pulses)
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 6), sharex=True)
    for name, pulse in pulses.items():
        physical = to_physical(pulse, device)
        time = physical["time"]
        color = colors[name]
        axes[0].plot(
            time,
            physical["omega_x"] / TWO_PI * 1e3,
            color=color,
            label=f"{name}, $T_g$={physical['gate_time']:.1f} ns",
        )
        axes[1].plot(
            time,
            physical["omega_y"] / TWO_PI * 1e3,
            color=color,
            label=name,
        )
    for axis, quadrature in zip(axes, ("x", "y")):
        axis.axhline(0.0, color="0.6", linewidth=0.8)
        axis.set_ylabel(rf"$\Omega_{quadrature}/2\pi$ [MHz]")
        axis.grid(alpha=0.25)
    for sign in (+1, -1):
        axes[0].axhline(
            sign * device.rabi_max * 1e3,
            color="0.3",
            linestyle=":",
            linewidth=1.0,
        )
    axes[0].set_title(
        "control waveforms at fixed peak drive "
        rf"($\Omega_\mathrm{{max}}/2\pi$ = {device.rabi_max * 1e3:.0f} MHz,"
        " dotted)"
    )
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time [ns]")
    figure.tight_layout()
    return figure


def plot_drive_spectra(pulses, device=DEVICE, colors=None, span=0.6):
    """Show where each drive puts spectral weight relative to ``alpha``."""
    import matplotlib.pyplot as plt

    colors = colors or color_cycle(pulses)
    figure, axis = plt.subplots(figsize=(9.5, 4.6))
    peak = 0.0
    for name, pulse in pulses.items():
        spectrum = drive_spectrum(pulse, device, span=span)
        exact = three_level_propagation(pulse, device, stride=4)
        peak = max(peak, spectrum["weight"].max())
        axis.semilogy(
            spectrum["frequency"] * 1e3,
            spectrum["weight"],
            color=colors[name],
            linewidth=1.2,
            label=f"{name}, $P_2$={exact['leakage']:.1e}",
        )
        # The one value the reader is here for: the weight sitting on the 1-2
        # transition.  The nulls elsewhere are sinc structure, not physics.
        at_alpha = np.interp(
            device.anharmonicity, spectrum["frequency"], spectrum["weight"]
        )
        axis.plot(
            device.anharmonicity * 1e3,
            at_alpha,
            marker="o",
            markersize=7,
            color=colors[name],
            markeredgecolor="white",
            zorder=5,
        )
    axis.axvline(
        device.anharmonicity * 1e3,
        color="0.3",
        linestyle="--",
        linewidth=1.2,
    )
    # Deep sinc nulls would otherwise stretch the log axis over ten decades.
    axis.set_ylim(peak * 1e-4, peak * 3)
    axis.text(
        device.anharmonicity * 1e3,
        peak * 2,
        r"  $\alpha/2\pi$ (1-2 transition)",
        color="0.3",
        fontsize=9,
        verticalalignment="top",
    )
    axis.set(
        title="drive spectrum; the marked weight at the anharmonicity is what leaks",
        xlabel="detuning from the qubit drive [MHz]",
        ylabel=r"$|\int\Omega e^{i\Phi}e^{i2\pi ft}dt|$ [rad]",
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8, loc="upper left", framealpha=0.9)
    figure.tight_layout()
    return figure


def plot_gate_dynamics(pulses, device=DEVICE, epsilon=None, colors=None,
                       stride=2, calibrations=None):
    """Populations through one gate under the full error model.

    Left: the transferred population ``P_1(t)``, which ends short of one by the
    total gate error.  Right: ``P_2(t)`` on a log axis -- the leakage builds up
    and partly returns, and where it lands at ``t = T_g`` is what the error
    budget charges.  Time is in units of each pulse's own ``T_g`` so that
    waveforms of different duration can be overlaid.
    """
    import matplotlib.pyplot as plt

    epsilon = device.control_error if epsilon is None else epsilon
    calibrations = calibrations or calibrate_all(pulses, device)
    colors = colors or color_cycle(pulses)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, pulse in pulses.items():
        trajectory = channel_trajectory(
            pulse, epsilon, device, stride=stride,
            calibration=calibrations[name],
        )
        fraction = trajectory["time"] / trajectory["gate_time"]
        populations = trajectory["populations"]
        axes[0].plot(
            fraction,
            populations[:, 1],
            color=colors[name],
            label=f"{name}, $1-P_1$={1 - populations[-1, 1]:.1e}",
        )
        axes[1].semilogy(
            fraction,
            np.maximum(populations[:, 2], 1e-12),
            color=colors[name],
            label=f"{name}, $P_2(T_g)$={populations[-1, 2]:.1e}",
        )
    axes[0].set(
        title=rf"population transfer, $\epsilon$={epsilon:+.0%}",
        xlabel="$t/T_g$",
        ylabel="$P_1$",
    )
    axes[1].set(
        title="leakage during the gate",
        xlabel="$t/T_g$",
        ylabel="$P_2$",
    )
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    figure.suptitle(
        "one gate under the frozen error model "
        f"($T_1$=$T_2$={device.t1 / 1e3:.0f} us, "
        rf"$\delta_z/2\pi$={device.static_detuning * 1e3:.1f} MHz)",
        fontsize=10,
    )
    figure.tight_layout()
    return figure


def plot_error_budget(budget, configs=BUDGET_CONFIGS, colors=None):
    """Bar chart of :func:`print_error_budget`, one group per configuration."""
    import matplotlib.pyplot as plt

    labels = [label for label, _ in configs]
    names = list(budget)
    colors = colors or color_cycle({name: None for name in names})
    positions = np.arange(len(labels))
    width = 0.8 / len(names)

    figure, axis = plt.subplots(figsize=(10, 4.4))
    for index, name in enumerate(names):
        offset = (index - 0.5 * (len(names) - 1)) * width
        axis.bar(
            positions + offset,
            [max(budget[name][label], 1e-12) for label in labels],
            width=width,
            color=colors[name],
            label=f"{name}, $T_g$={budget[name]['gate_time_ns']:.0f} ns",
        )
    axis.set_yscale("log")
    axis.set_xticks(positions, labels, fontsize=9)
    axis.set(
        title="single-gate error budget, one term of the model at a time",
        ylabel=r"$1-F_\mathrm{avg}$ (qubit subspace)",
    )
    axis.grid(alpha=0.25, axis="y", which="both")
    # Headroom, so the legend never sits on top of the tallest bar.
    lower, upper = axis.get_ylim()
    axis.set_ylim(lower, upper * 60)
    axis.legend(fontsize=7.5, ncol=3, loc="upper left", framealpha=0.95)
    figure.tight_layout()
    return figure


#: Line styles of the two error signs in the DB panels, kept identical to the
#: two-level demo so the two notebooks' figures can be read side by side: the
#: open circles are ``-3%`` and stay visible where the two signs overlap.
DB_STYLES = {
    "full, eps=+3%": dict(linestyle="-", linewidth=2.6, alpha=0.60, zorder=2),
    "full, eps=-3%": dict(
        linestyle="-", linewidth=1.2, marker="o", markevery=6, markersize=5,
        markerfacecolor="white", markeredgewidth=1.2, zorder=3,
    ),
}


def plot_db_traces(traces, device=DEVICE, colors=None, layout="grid"):
    """The four views of one DB run.

    (a) ``P_0`` against cycle number, the standard DB readout, with the
    ``epsilon = 0`` decay envelope dashed underneath each pair of traces.
    (b) the same curves against the sequence's physical duration -- each
    waveform has its own ``T_g``, so a shared second axis on (a) would be
    meaningless and this is the honest form of it: it is where the gate-time
    price of robustness becomes visible.
    (c) leakage accumulated over the sequence.
    (d) the control-error signal with the envelope divided out,
    ``|P_0(n, +-3%) - P_0(n, 0)|``, on a log axis -- the only panel in which
    five orders of magnitude of robustness fit at once.

    ``layout`` is ``"grid"`` for the 2x2 arrangement or ``"tall"`` for three
    rows, which gives the two ``P_0`` panels the full width; the tall form is
    less crowded once several waveforms are overlaid.
    """
    import matplotlib.pyplot as plt

    colors = colors or color_cycle(traces)
    if layout == "tall":
        figure = plt.figure(figsize=(11.5, 11.5))
        grid = figure.add_gridspec(3, 2)
        top_left = figure.add_subplot(grid[0, :])
        top_right = figure.add_subplot(grid[1, :])
        bottom_left = figure.add_subplot(grid[2, 0])
        bottom_right = figure.add_subplot(grid[2, 1])
        axes = np.array([top_left, top_right, bottom_left, bottom_right])
    elif layout == "grid":
        figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
        (top_left, top_right), (bottom_left, bottom_right) = axes
    else:
        raise ValueError(f"unknown layout {layout!r}; use 'grid' or 'tall'")

    for name, trace in traces.items():
        color = colors[name]
        cycle_numbers = trace["cycles"]
        elapsed = trace["elapsed_us"]
        envelope = trace[DB_ENVELOPE][:, 0]
        top_left.plot(
            cycle_numbers, envelope, color=color, linestyle="--",
            linewidth=1.0, alpha=0.9, zorder=1,
        )
        top_right.plot(
            elapsed, envelope, color=color, linestyle="--", linewidth=1.0,
            alpha=0.9, zorder=1,
        )
        for label in DB_FULL:
            populations = trace[label]
            top_left.plot(
                cycle_numbers, populations[:, 0], color=color,
                label=f"{name}, {label.split(', ')[1]}", **DB_STYLES[label],
            )
            top_right.plot(
                elapsed, populations[:, 0], color=color, **DB_STYLES[label]
            )
            # The log panels start at n = 1: at n = 0 nothing has been applied
            # yet and both quantities are identically zero, which would drag the
            # axis down by ten decades.
            bottom_right.semilogy(
                cycle_numbers[1:],
                np.maximum(np.abs(populations[1:, 0] - envelope[1:]), 1e-16),
                color=color, **DB_STYLES[label],
            )
        bottom_left.semilogy(
            cycle_numbers[1:],
            np.maximum(trace[DB_FULL[0]][1:, 2], 1e-12),
            color=color, linewidth=1.4,
            label=f"{name}, $T_g$={trace['gate_time_ns']:.0f} ns",
        )

    top_left.set(
        title="(a) DB readout: return probability against cycle count",
        xlabel="XX cycle count $n$", ylabel="$P_0$", ylim=(-0.03, 1.02),
    )
    top_right.set(
        title="(b) the same traces against physical sequence duration",
        xlabel=r"elapsed time $2nT_g$ [$\mu$s]", ylabel="$P_0$",
        ylim=(-0.03, 1.02),
    )
    bottom_left.set(
        title="(c) leakage accumulated over the sequence",
        xlabel="XX cycle count $n$", ylabel=r"$P_2$",
    )
    bottom_right.set(
        title="(d) control-error signal, envelope divided out",
        xlabel="XX cycle count $n$",
        ylabel=r"$|P_0(n,\pm3\%)-P_0(n,0)|$",
    )
    # Both axes logarithmic: the signal grows as n^2 before it saturates, so the
    # ranking of the waveforms is the horizontal offset between parallel lines at
    # small n, and that is where it is readable.
    bottom_right.set_xscale("log")
    for axis in axes.ravel():
        axis.grid(alpha=0.25, which="both")
    top_left.legend(fontsize=7, ncol=2, loc="lower left", framealpha=0.95)
    bottom_left.legend(fontsize=7, loc="lower right", framealpha=0.95)
    figure.suptitle(
        "1qb-DB on a transmon: dashed = decay envelope at "
        r"$\epsilon=0$, thick = $+3\%$, circles = $-3\%$  "
        f"($T_1$=$T_2$={device.t1 / 1e3:.0f} us, "
        rf"$\delta_z/2\pi$={device.static_detuning * 1e3:.1f} MHz)",
        fontsize=10,
    )
    figure.tight_layout()
    return figure


#: Colours for waveforms the demo does not name.  Compared as normalized RGBA
#: against ``db_helpers.PULSE_COLORS``, because "tab:blue" and "#1f77b4" are the
#: same colour written two ways and a string comparison would let it repeat.
SPARE_COLORS = (
    "tab:red", "tab:purple", "tab:brown",
    "tab:pink", "tab:olive", "tab:cyan",
)


def color_cycle(pulses):
    """Reuse the demo colours where they exist, extend them where they do not."""
    from matplotlib.colors import to_rgba

    used = {to_rgba(color) for color in db_helpers.PULSE_COLORS.values()}
    spare = (color for color in SPARE_COLORS if to_rgba(color) not in used)
    colors = {}
    for name in pulses:
        color = db_helpers.PULSE_COLORS.get(name)
        if color is None:
            color = next(spare)
            used.add(to_rgba(color))
        colors[name] = color
    return colors
