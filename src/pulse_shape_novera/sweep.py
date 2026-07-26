"""Weight-sweep machinery for the ansatz-quality study (step 11d).

We ask, across a grid of weight combinations ``w_i``, which of three routes to a
single-qubit gate minimizes the normalized cost ``C = sum_i w_i * Chat_i``:

  * **good ansatz** -- the closed, zero-area rcp_lemniscate curve, used *as is*
    (already gate-correct and dephasing-robust; the whole point of a good prior
    is that you don't optimize it);
  * **naive / bad ansatz** -- the open constant-amplitude arc, used *as is*
    (implements the gate but is not robust);
  * **optimize (BARQ)** -- the no-prior automated optimizer, run under that same
    ``w``; its gate is hard-fixed by point gate-fixing (PGF), so ``F == 1`` and
    optimization only shapes robustness.

The winner map therefore reads as a decision: *grab the good ansatz for free*,
*grab the naive pulse for free*, or *you must run the optimizer*.

Normalization contract (see ``results/README.md``):

* **Fair reference.** ``Chat_i = C_i / C_i^ref`` with ``C_i^ref`` = the term on
  the *naive pulse* (the constant-amplitude arc of area ``theta`` implementing
  ``X(theta)``). It is a fixed fourth curve scored identically for everyone, so
  no candidate sits at ``Chat_i = 1`` by construction; it is open and encloses
  area, so every ``C_i^ref`` is finite and nonzero.

* **Scale invariance instead of a scale anchor.** Closure, curve area and pulse
  energy are not scale invariant on their own, so comparing curves of different
  arc length ``L`` would be meaningless. We use their scale-*invariant* forms
  (divide by the appropriate power of ``L``), so every curve -- good, naive, and
  BARQ (whose ``L`` differs) -- is compared on equal footing with no gauge
  fixing. At ``L = 1`` (the naive pulse) the invariant forms equal the raw ones,
  so the references keep their clean analytic values (energy ``pi^2``,
  ``max_amp`` ``pi``).

Historical note: an earlier design optimized all three inits in a common raw
Bezier space under a soft gate penalty. It was abandoned -- the raw-Bezier
optimizer falls into degenerate wrong-gate minima that no finite gate weight
prevents (see ``_dev_logs/step11d_sweep.md``). BARQ's PGF is the only mechanism
that keeps the gate exact under optimization, hence the design above.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import qutip

from qurveros import losses, frametools, barqtools
from qurveros.optspacecurve import BarqCurve

from pulse_shape_novera.proxies import pulse_energy_loss
from pulse_shape_novera.barq import make_barq_xgate, xgate_pgf_mod, barq_gate_fidelity
from pulse_shape_novera.bezier import fit_bezier_control_points

N_FRENET = 400    # Frenet sampling resolution
N_FREE_POINTS = 14   # BARQ free points; internal block (n-2) carries the seeded shape


# --- scale-invariant cost terms ------------------------------------------------
# Each returns a scale-invariant functional of the curve (frenet_dict), so curves
# of different arc length L are directly comparable. At L = 1 these equal the raw
# qurveros losses.
def _length(frenet_dict):
    return frametools.calculate_total_length(frenet_dict)


def closure_term(frenet_dict):
    """Fractional squared endpoint gap ``|r(T)-r(0)|^2 / L^2`` (1st-order dephasing)."""
    curve = frenet_dict["curve"]
    return jnp.sum((curve[-1] - curve[0]) ** 2) / _length(frenet_dict) ** 2


def curve_area_term(frenet_dict):
    """Scale-invariant squared enclosed area ``(area/L^2)^2`` (2nd-order dephasing)."""
    return losses.curve_zero_area_loss(frenet_dict) / _length(frenet_dict) ** 4


def energy_term(frenet_dict):
    """Scale-invariant pulse energy ``L * integral Omega^2 dt`` (leakage proxy)."""
    return pulse_energy_loss(frenet_dict) * _length(frenet_dict)


# tantrix area and peak amplitude (T_g * Omega_max) are already scale invariant.
INV_TERMS = {
    "closure": closure_term,                       # 1st-order dephasing
    "curve_area": curve_area_term,                 # 2nd-order dephasing
    "energy": energy_term,                         # leakage proxy, ~ integral Omega^2
    "tantrix": losses.tantrix_zero_area_loss,      # amplitude / Rabi-error robustness
    "max_amp": losses.max_amp_loss,                # peak drive T_g * Omega_max
}


# --- references + per-curve evaluation -----------------------------------------
def compute_references(naive_spacecurve, n_frenet=N_FRENET):
    """Evaluate ``C_i^ref`` on the naive-pulse baseline (all finite, > 0)."""
    naive_spacecurve.evaluate_frenet_dict(n_frenet)
    fd = naive_spacecurve.frenet_dict
    refs = {name: float(fn(fd)) for name, fn in INV_TERMS.items()}
    return {"refs": refs, "tg_ref": float(_length(fd))}


def evaluate_chat(spacecurve, reference, n_frenet=N_FRENET):
    """Normalized cost vector ``{term: Chat_i}`` for an arbitrary space curve."""
    spacecurve.evaluate_frenet_dict(n_frenet)
    fd = spacecurve.frenet_dict
    refs = reference["refs"]
    return {name: float(fn(fd)) / refs[name] for name, fn in INV_TERMS.items()}


def total_cost(chat, weights):
    """``sum_i w_i * Chat_i`` over the swept terms."""
    return sum(float(weights.get(name, 0.0)) * chat[name] for name in INV_TERMS)


# --- BARQ seeding: inject an ansatz curve as the optimizer's initial point -----
def seed_free_points_from_curve(spacecurve, *, n_free_points=N_FREE_POINTS,
                                n_frenet=600):
    """Fit an ansatz curve to BARQ ``free_points`` so BARQ starts *at* that ansatz.

    BARQ builds its Bezier control points as
    ``[0, gate_fix[:3], free_points[2:], gate_fix[3:], 0]`` -- the internal control
    points ARE ``free_points[2:]`` and PGF derives the gate-fixing points from
    ``free_points[:2]`` (keeping the gate exact for any values). We therefore fit
    the ansatz to a degree ``n_free_points+5`` Bezier curve and read the internal
    block straight into ``free_points[2:]`` (with the two near-boundary points
    seeding ``free_points[:2]``). The resulting BARQ curve starts close to the
    ansatz in shape, with the gate exact by construction.
    """
    spacecurve.evaluate_frenet_dict(n_frenet)
    pos = np.asarray(spacecurve.frenet_dict["curve"])
    x = np.asarray(spacecurve.frenet_dict["x_values"])
    degree = n_free_points + 5
    W = np.asarray(fit_bezier_control_points(pos, degree, param=x))  # (n_free+6, 3)
    internal = W[4:4 + (n_free_points - 2)]              # the shape-carrying block
    return np.vstack([W[1:3], internal])                # (n_free_points, 3)


def make_seeded_barq(init_free_points=None, *, n_free_points=N_FREE_POINTS,
                     seed=4531469, norm_value=0.25):
    """Build a BARQ X-gate curve, optionally seeded at a given ``free_points``.

    ``init_free_points=None`` -> BARQ's own random init (the no-prior baseline).
    """
    barq = BarqCurve(adj_target=quantum_x_adj(), n_free_points=n_free_points,
                     pgf_mod=xgate_pgf_mod)
    init_pgf = barqtools.get_default_pgf_params_dict()
    init_pgf["norm_value"] = norm_value
    if init_free_points is None:
        barq.initialize_parameters(seed=seed, init_pgf_params=init_pgf)
    else:
        barq.initialize_parameters(init_free_points=jnp.asarray(init_free_points),
                                   init_pgf_params=init_pgf)
    return barq


# --- the optimizer arm: BARQ (PGF, gate exact) under a normalized objective ----
# Note on the gate: BARQ's gate lives in its TTC control mode; the Frenet-frame
# adjoint gives a frame-convention value (1/3 for X), NOT the gate. Use
# ``barq_gate_fidelity`` (TTC + simulator) for the actual gate. PGF fixes it
# exactly, so we verify it once at the optimum rather than per checkpoint.
def _normalized(fn, ref):
    return lambda fd: fn(fd) / ref


def optimize_barq(weights, reference, *, init_free_points=None,
                  max_iter=1200, lr=1e-3, n_free_points=N_FREE_POINTS,
                  seed=4531469, norm_value=0.25, checkpoint_every=25,
                  n_frenet=N_FRENET, cost_fn=None):
    """Optimize a (optionally seeded) BARQ curve under ``sum_i w_i Chat_i``.

    ``cost_fn(chat, weights)`` (default :func:`total_cost`) computes the logged
    total cost. Records the FULL per-checkpoint history so the run can later serve
    as a ready-to-use control solution.

    Returns a dict with, at each of ``steps`` checkpoints:
        ``cost_history``   : total cost (via ``cost_fn``),
        ``chat_history``   : {term: [Chat_i per checkpoint]},
        ``params_history`` : list of ``free_points`` arrays (the control solution),
    plus ``final_chat``, ``final_free_points``, and ``final_gate`` (TTC fidelity;
    PGF keeps it ~1, verified once at the optimum).
    """
    if cost_fn is None:
        cost_fn = total_cost
    barq = make_seeded_barq(init_free_points, n_free_points=n_free_points,
                            seed=seed, norm_value=norm_value)

    refs = reference["refs"]
    terms = [[_normalized(INV_TERMS[name], refs[name]), float(w)]
             for name, w in weights.items() if float(w) > 0.0]
    if not terms:  # degenerate all-zero weights: nothing to optimize
        terms = [[_normalized(INV_TERMS["curve_area"], refs["curve_area"]), 0.0]]

    # Freeze the scale (norm_value) as in step 3 / barq.py to avoid collapse.
    labels = jax.tree.map(lambda _: True, barq.params)
    labels["pgf_params"]["norm_value"] = False
    optimizer = optax.multi_transform(
        {True: optax.adam(lr), False: optax.set_to_zero()}, param_labels=labels)
    barq.prepare_optimization_loss(*terms)
    barq.optimize(optimizer, max_iter=max_iter)

    hist = barq.get_params_history()
    steps = list(range(0, len(hist), checkpoint_every))
    if steps[-1] != len(hist) - 1:
        steps.append(len(hist) - 1)
    cost_history, params_history = [], []
    chat_history = {name: [] for name in INV_TERMS}
    for k in steps:
        barq.update_params_from_opt_history(k)
        barq.evaluate_frenet_dict(n_frenet)
        fd = barq.frenet_dict
        chat = {name: float(fn(fd)) / refs[name] for name, fn in INV_TERMS.items()}
        for name in INV_TERMS:
            chat_history[name].append(chat[name])
        cost_history.append(cost_fn(chat, weights))
        params_history.append(np.asarray(barq.params["free_points"]))
    final_chat = {name: chat_history[name][-1] for name in INV_TERMS}
    final_gate = barq_gate_fidelity(barq)   # TTC-mode fidelity; PGF keeps it ~1
    return {"steps": steps, "cost_history": cost_history,
            "chat_history": chat_history, "params_history": params_history,
            "final_chat": final_chat, "final_free_points": params_history[-1],
            "final_gate": final_gate}


def quantum_x_adj():
    """Adjoint (SO(3)) representation of the X gate target (cached-friendly)."""
    from qurveros.qubit_bench import quantumtools
    return quantumtools.calculate_adj_rep(qutip.sigmax())
