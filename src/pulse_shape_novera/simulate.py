"""Thin simulation helpers: propagate a space curve's SCQC control fields to a
gate and score it.

We reuse qurveros' qutip-based single-qubit simulator rather than rolling our
own propagator (see CLAUDE.md: prefer mature tools). The simulator integrates

    H(t) = (1/2)[Omega cos(Phi) X + Omega sin(Phi) Y + Delta Z]

and returns the implemented unitary and its average gate fidelity to a target.
"""

from __future__ import annotations

import numpy as np
import qutip
from qurveros.qubit_bench import simulator


def gate_fidelity(spacecurve, u_target, *, control_mode: str = "XY") -> dict:
    """Propagate a qurveros ``SpaceCurve`` and score the resulting gate.

    Parameters
    ----------
    spacecurve : qurveros.spacecurve.SpaceCurve
        A configured space curve. Its Frenet/control dicts are (re)evaluated
        here, so the caller does not need to prime them.
    u_target : qutip.Qobj or array
        Target gate to score against. Average gate fidelity is phase-insensitive,
        so ``qutip.sigmax()`` and ``R_x(pi) = -i X`` give the same score.
    control_mode : str, optional
        qurveros control mode (default ``"XY"``, resonant drive).

    Returns
    -------
    dict
        ``{"fidelity": float, "u_final": (2, 2) complex ndarray}``.
    """
    spacecurve.evaluate_frenet_dict()
    spacecurve.evaluate_control_dict(control_mode)

    if not isinstance(u_target, qutip.Qobj):
        u_target = qutip.Qobj(np.asarray(u_target))

    sim = simulator.simulate_control_dict(spacecurve.get_control_dict(), u_target)
    return {"fidelity": float(sim["avg_gate_fidelity"]), "u_final": sim["u_final"]}


def rotation(axis: str, angle: float) -> qutip.Qobj:
    """Single-qubit rotation ``R_axis(angle) = exp(-i angle/2 * sigma_axis)``."""
    pauli = {"x": qutip.sigmax(), "y": qutip.sigmay(), "z": qutip.sigmaz()}[axis]
    return (-1j * (angle / 2.0) * pauli).expm()
