"""Systematic weight-sweep machinery for the ansatz-quality study (step 11d).

We compare three initializations -- a *good* ansatz (closed, zero-area, native
gate), a *bad* ansatz (open arc), and a no-prior BARQ-default -- optimized under
the same objective

    C(curve) = sum_i  w_i * Chat_i ,   Chat_i = C_i / C_i^ref ,

over a grid of weight combinations ``w_i``. See ``results/README.md`` for the
cost-term table and the normalization contract; the short version:

* **Fair reference.** ``C_i^ref`` is the value of term ``i`` on the *naive
  pulse* -- the constant-amplitude resonant arc of area ``theta`` that implements
  ``X(theta)`` (a semicircle for ``X(pi)``). It is a fixed fourth curve, scored
  identically for all three candidates, so nobody sits at ``Chat_i = 1`` by
  construction. The naive arc is open and encloses area, so every ``C_i^ref`` is
  finite and nonzero (no division by zero). Gate error is the exception: the
  naive pulse realizes the gate, so it uses a fixed tolerance ``eps_ref``.

* **Scale gauge.** Closure, curve area and pulse energy are *not* scale
  invariant, so a bare comparison across curves of different length is
  meaningless. We freeze the gauge with a soft anchor ``(T_g - T_g^ref)^2`` (not
  a swept term) that pins every curve to the naive pulse's gate time ``T_g^ref``.
  With the gauge fixed, all ``C_i`` are directly comparable.

* **Gate lock.** ``1 - F`` carries a large fixed weight (``W_GATE``) so ``F ~ 1``
  throughout; the gate is a constraint, not an importance knob (cf. step 11b,
  where too small a gate weight let the gate drift).

The reusable pieces (cost terms, references, single optimization run) live here;
the grid orchestration and figures live in the tracked ``results/`` scripts.
"""

import numpy as np
import jax.numpy as jnp
import optax
import qutip

from qurveros.optspacecurve import OptimizableSpaceCurve
from qurveros.beziertools import bezier_curve_vec
from qurveros import losses, frametools
from qurveros.qubit_bench import quantumtools

from pulse_shape_novera.proxies import pulse_energy_loss

# --- fixed knobs (gauge fixers, not swept) -------------------------------------
W_GATE = 50.0     # gate-lock weight (holds F ~ 1; matches step 11b)
W_SCALE = 5.0     # scale-anchor weight (holds T_g ~ T_g^ref)
EPS_REF = 1e-4    # gate-error reference: Chat_gate = 1 at 99.99% fidelity
BEZIER_DEGREE = 15
N_FRENET = 400    # Frenet sampling resolution


# --- cost terms ----------------------------------------------------------------
def closure_loss(frenet_dict):
    """Squared endpoint gap ``|r(T) - r(0)|^2`` (1st-order dephasing robustness).

    qurveros ships a curve *area* loss but not a closure loss; a closed curve is
    the geometric condition for first-order static-dephasing robustness.
    """
    curve = frenet_dict["curve"]
    return jnp.sum((curve[-1] - curve[0]) ** 2)


def _adj_target(target_gate):
    """Adjoint (SO(3)) representation of the target single-qubit gate."""
    return quantumtools.calculate_adj_rep(target_gate)


def make_gate_infidelity(target_gate):
    """Return ``fd -> 1 - F_avg`` against ``target_gate`` (adjoint fidelity)."""
    adj = _adj_target(target_gate)

    def gate_infidelity(frenet_dict):
        frame_gate = frametools.calculate_frenet_adj(frenet_dict, 0.0)
        return 1.0 - frametools.calculate_adj_fidelity(frame_gate, adj)

    return gate_infidelity


def make_scale_anchor(tg_ref):
    """Return ``fd -> (T_g - tg_ref)^2`` -- the soft gauge-fixer."""
    def scale_anchor(frenet_dict):
        return (frametools.calculate_total_length(frenet_dict) - tg_ref) ** 2

    return scale_anchor


# Swept cost terms: name -> raw loss functional C_i(frenet_dict).
# (gate error and the scale anchor are handled separately -- they are locked,
#  not swept.)
SWEPT_TERMS = {
    "closure": closure_loss,                       # 1st-order dephasing
    "curve_area": losses.curve_zero_area_loss,     # 2nd-order dephasing
    "energy": pulse_energy_loss,                   # leakage proxy, int Omega^2 dt
    "tantrix": losses.tantrix_zero_area_loss,      # amplitude / Rabi-error robustness
    "max_amp": losses.max_amp_loss,                # peak drive T_g * Omega_max
}


# --- references ----------------------------------------------------------------
def compute_references(naive_spacecurve, n_frenet=N_FRENET):
    """Evaluate ``C_i^ref`` (and ``T_g^ref``) on the naive-pulse baseline.

    Parameters
    ----------
    naive_spacecurve : qurveros SpaceCurve
        The naive pulse -- a constant-amplitude arc implementing the target gate
        (e.g. ``make_circle_arc_spacecurve(theta)``).

    Returns
    -------
    dict with keys:
        ``refs``   : {term_name: C_i^ref} for every swept term (all finite, > 0),
        ``tg_ref`` : the naive pulse's gate time (arc length),
        ``eps_ref``: the gate-error reference tolerance.
    """
    naive_spacecurve.evaluate_frenet_dict(n_frenet)
    fd = naive_spacecurve.frenet_dict
    refs = {name: float(fn(fd)) for name, fn in SWEPT_TERMS.items()}
    tg_ref = float(frametools.calculate_total_length(fd))
    return {"refs": refs, "tg_ref": tg_ref, "eps_ref": EPS_REF}


# --- single optimization run ---------------------------------------------------
def _normalized(fn, ref):
    """Wrap a raw loss so it returns the normalized ``C_i / C_i^ref``."""
    return lambda fd: fn(fd) / ref


def run_single(W0, weights, reference, target_gate,
               n_iter=1500, lr=5e-3, checkpoint_every=50, threshold=None):
    """Optimize one initialization under ``C = sum_i w_i Chat_i`` and log history.

    Parameters
    ----------
    W0 : (deg+1, 3) array
        Initial Bezier control points (the ansatz, fitted to control points).
    weights : dict
        ``{term_name: w_i}`` over ``SWEPT_TERMS``; missing terms default to 0.
    reference : dict
        Output of :func:`compute_references` (``refs``/``tg_ref``/``eps_ref``).
    target_gate : qutip.Qobj
        Target single-qubit gate for the fidelity (adjoint) term.
    threshold : float or None
        If given, also report the first checkpoint step at which the *total*
        normalized swept cost (sum_i w_i Chat_i, gate/anchor excluded) drops
        below ``threshold`` -- the early-stop / steps-to-target metric.

    Returns
    -------
    dict with keys:
        ``steps``            : checkpoint step indices,
        ``chat_history``     : {term: [Chat_i at each checkpoint]},
        ``gate_infidelity``  : [1 - F at each checkpoint],
        ``tg``               : [T_g at each checkpoint],
        ``final_chat``       : {term: Chat_i at the last checkpoint},
        ``steps_to_threshold``: int or None.
    """
    refs, tg_ref = reference["refs"], reference["tg_ref"]
    gate_inf = make_gate_infidelity(target_gate)
    scale_anchor = make_scale_anchor(tg_ref)

    # Build the objective term list for qurveros: [[fn, weight], ...].
    # Gate + scale are locked; swept terms enter normalized with their weight.
    terms = [[gate_inf, W_GATE], [scale_anchor, W_SCALE]]
    for name, fn in SWEPT_TERMS.items():
        w = float(weights.get(name, 0.0))
        if w > 0.0:
            terms.append([_normalized(fn, refs[name]), w])

    sc = OptimizableSpaceCurve(curve=bezier_curve_vec, order=0,
                               interval=[0.0, 1.0], params=jnp.asarray(W0))
    sc.initialize_parameters(jnp.asarray(W0))
    sc.prepare_optimization_loss(*terms)
    sc.optimize(optax.adam(lr), n_iter)

    hist = sc.get_params_history()
    steps = list(range(0, len(hist), checkpoint_every))
    if steps[-1] != len(hist) - 1:
        steps.append(len(hist) - 1)

    chat_history = {name: [] for name in SWEPT_TERMS}
    gate_hist, tg_hist, total_swept = [], [], []
    for k in steps:
        sc.update_params_from_opt_history(k)
        sc.evaluate_frenet_dict(N_FRENET)
        fd = sc.frenet_dict
        tot = 0.0
        for name, fn in SWEPT_TERMS.items():
            chat = float(fn(fd)) / refs[name]
            chat_history[name].append(chat)
            tot += float(weights.get(name, 0.0)) * chat
        total_swept.append(tot)
        gate_hist.append(float(gate_inf(fd)))
        tg_hist.append(float(frametools.calculate_total_length(fd)))

    steps_to_threshold = None
    if threshold is not None:
        for s, tot in zip(steps, total_swept):
            if tot < threshold:
                steps_to_threshold = s
                break

    return {
        "steps": steps,
        "chat_history": chat_history,
        "total_swept": total_swept,
        "gate_infidelity": gate_hist,
        "tg": tg_hist,
        "final_chat": {name: chat_history[name][-1] for name in SWEPT_TERMS},
        "steps_to_threshold": steps_to_threshold,
    }
