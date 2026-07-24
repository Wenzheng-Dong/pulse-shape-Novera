"""Step 7 — Bezier reconstruction fidelity (the bridge to BARQ seeding).

BARQ optimizes Bezier control points, so seeding it with one of our ansatz
curves requires fitting that curve into the Bezier basis. This test quantifies
how faithfully a degree-n Bezier reproduces an ansatz -- at the gate level, not
just positions (curvature is a 2nd derivative and amplifies fit error). It
answers _plan.md Q2 with numbers: smooth ansaetze reconstruct to machine
precision at modest degree, but low degree distorts (so degree must be chosen).
"""

import numpy as np
import qutip
import pytest

import pulse_shape_novera  # noqa: F401 -- applies the qurveros patch
from pulse_shape_novera import (
    make_circle_arc_spacecurve,
    make_analytic_spacecurve,
    reconstruct_spacecurve,
    gate_fidelity,
    rotation,
)


def _recon_gate_fidelity(sc, target, degree):
    _, recon = reconstruct_spacecurve(sc, degree)
    return gate_fidelity(recon, target)["fidelity"]


def test_circle_reconstruction_converges_to_x_gate():
    """circle_arc -> Bezier reproduces the X gate to machine precision by degree 8."""
    target = rotation("x", np.pi)
    fid_low = _recon_gate_fidelity(make_circle_arc_spacecurve(np.pi), target, degree=3)
    fid_high = _recon_gate_fidelity(make_circle_arc_spacecurve(np.pi), target, degree=8)

    assert fid_high > 1.0 - 1e-6, fid_high
    assert fid_low < fid_high  # degree matters: low degree distorts the gate


def test_position_error_decreases_with_degree():
    """Reconstruction position error shrinks monotonically as the degree grows."""
    sc = make_circle_arc_spacecurve(np.pi)
    sc.evaluate_frenet_dict(400)
    original = np.asarray(sc.frenet_dict["curve"])

    errors = []
    for degree in (3, 6, 10, 14):
        _, recon = reconstruct_spacecurve(make_circle_arc_spacecurve(np.pi), degree)
        recon.evaluate_frenet_dict(400)
        errors.append(np.abs(np.asarray(recon.frenet_dict["curve"]) - original).max())

    assert all(b < a for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < 1e-6


def _self_gate_fidelity(name, degree):
    """Fidelity of the degree-n Bezier reconstruction to the family's OWN gate."""
    original_u = gate_fidelity(make_analytic_spacecurve(name), qutip.qeye(2))["u_final"]
    return _recon_gate_fidelity(make_analytic_spacecurve(name), qutip.Qobj(original_u), degree=degree)


@pytest.mark.parametrize("name", ["lissajous_3d", "zeng_clifford_3d"])
def test_smooth_families_reconstruct_at_moderate_degree(name):
    """These families reconstruct to (near) unit gate fidelity by degree 20."""
    assert _self_gate_fidelity(name, degree=20) > 1.0 - 1e-3


def test_reconstruction_difficulty_is_ansatz_dependent():
    """Not every ansatz Bezier-seeds equally: at the same degree, alpha_3d's gate
    is far less faithful than lissajous_3d -- the Q2 caveat, quantified. (alpha's
    position error still shrinks with degree; gate fidelity is what lags, because
    curvature amplifies fit error.)"""
    fid_easy = _self_gate_fidelity("lissajous_3d", degree=20)
    fid_hard = _self_gate_fidelity("alpha_3d", degree=20)
    assert fid_easy > 1.0 - 1e-3          # lissajous is essentially exact
    assert fid_hard < 1.0 - 1e-3          # alpha is not yet faithful at this degree
    assert fid_easy - fid_hard > 1e-3     # a clear, ansatz-dependent gap
