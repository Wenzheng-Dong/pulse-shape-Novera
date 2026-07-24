"""pulse-shape-Novera: robust gate pulse design via geometric space curves.

See ``_plan.md`` for the research plan and ``CLAUDE.md`` for working
conventions. The package couples two upstream libraries:

- ``qurveros`` — the JAX optimization / SCQC engine,
- ``curvecontroltoolbox`` — curve families, two-qubit iSWAP, noise/leakage.
"""

# Apply upstream patches (e.g. qurveros safe vector norm) BEFORE anything below
# imports/traces qurveros. See _patches.py for the full rationale.
from pulse_shape_novera._patches import apply_qurveros_patches

apply_qurveros_patches()

from pulse_shape_novera.ansatz import (
    circle_arc_curve,
    make_circle_arc_spacecurve,
    helix_curve,
    make_helix_spacecurve,
    lissajous_curve,
    alpha_curve,
    zeng_clifford_curve,
    ANALYTIC_FAMILY_CURVES,
    make_analytic_spacecurve,
    triangle_pulse_tangent,
    gaussian_arc_tangent,
    make_triangle_pulse_spacecurve,
    make_gaussian_arc_spacecurve,
)
from pulse_shape_novera.simulate import gate_fidelity, rotation
from pulse_shape_novera.optimize import optimize_gate_time
from pulse_shape_novera.bezier import (
    fit_bezier_control_points,
    make_bezier_spacecurve,
    reconstruct_spacecurve,
)

__all__ = [
    "circle_arc_curve",
    "make_circle_arc_spacecurve",
    "helix_curve",
    "make_helix_spacecurve",
    "lissajous_curve",
    "alpha_curve",
    "zeng_clifford_curve",
    "ANALYTIC_FAMILY_CURVES",
    "make_analytic_spacecurve",
    "triangle_pulse_tangent",
    "gaussian_arc_tangent",
    "make_triangle_pulse_spacecurve",
    "make_gaussian_arc_spacecurve",
    "gate_fidelity",
    "rotation",
    "optimize_gate_time",
    "fit_bezier_control_points",
    "make_bezier_spacecurve",
    "reconstruct_spacecurve",
]
