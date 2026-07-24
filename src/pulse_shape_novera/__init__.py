"""pulse-shape-Novera: robust gate pulse design via geometric space curves.

See ``_plan.md`` for the research plan and ``CLAUDE.md`` for working
conventions. The package couples two upstream libraries:

- ``qurveros`` — the JAX optimization / SCQC engine,
- ``curvecontroltoolbox`` — curve families, two-qubit iSWAP, noise/leakage.
"""

from pulse_shape_novera.ansatz import circle_arc_curve, make_circle_arc_spacecurve
from pulse_shape_novera.simulate import gate_fidelity, rotation

__all__ = [
    "circle_arc_curve",
    "make_circle_arc_spacecurve",
    "gate_fidelity",
    "rotation",
]
