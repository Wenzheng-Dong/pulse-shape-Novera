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
from pulse_shape_novera.barq import make_barq_xgate, xgate_pgf_mod

N_FRENET = 400    # Frenet sampling resolution


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


# --- the "optimize" arm: BARQ under a normalized weighted objective ------------
def _normalized(fn, ref):
    return lambda fd: fn(fd) / ref


def optimize_barq(weights, reference, *, max_iter=1200, lr=1e-3,
                  n_free_points=10, seed=4531469, norm_value=0.25,
                  checkpoint_every=50, n_frenet=N_FRENET, cost_fn=None):
    """Run BARQ (PGF, gate exact) minimizing ``sum_i w_i * Chat_i`` and log history.

    ``cost_fn(chat, weights)`` (default :func:`total_cost`) computes the total cost
    recorded in ``cost_history`` -- pass the caller's cost (e.g. one that applies a
    robustness floor) so the logged trajectory matches the decision metric.

    Returns
    -------
    dict with:
        ``final_chat``   : {term: Chat_i} at the optimum,
        ``steps``        : checkpoint step indices,
        ``cost_history`` : total cost (via ``cost_fn``) at each checkpoint,
        ``gate_locked``  : True (PGF fixes the gate exactly by construction).
    """
    if cost_fn is None:
        cost_fn = total_cost
    adj_target = quantum_x_adj()
    barq = BarqCurve(adj_target=adj_target, n_free_points=n_free_points,
                     pgf_mod=xgate_pgf_mod)
    init_pgf = barqtools.get_default_pgf_params_dict()
    init_pgf["norm_value"] = norm_value
    barq.initialize_parameters(seed=seed, init_pgf_params=init_pgf)

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
    cost_history = []
    for k in steps:
        barq.update_params_from_opt_history(k)
        barq.evaluate_frenet_dict(n_frenet)
        chat = {name: float(fn(barq.frenet_dict)) / refs[name]
                for name, fn in INV_TERMS.items()}
        cost_history.append(cost_fn(chat, weights))
    final_chat = {name: float(fn(barq.frenet_dict)) / refs[name]
                  for name, fn in INV_TERMS.items()}
    return {"final_chat": final_chat, "steps": steps,
            "cost_history": cost_history, "gate_locked": True}


def quantum_x_adj():
    """Adjoint (SO(3)) representation of the X gate target (cached-friendly)."""
    from qurveros.qubit_bench import quantumtools
    return quantumtools.calculate_adj_rep(qutip.sigmax())
