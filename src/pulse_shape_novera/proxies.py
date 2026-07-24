"""Leakage / control-cost proxies computed within the 2-level SCQC picture.

Per the project's proxy-first strategy (see _plan.md), we quantify "leakage cost"
with cheap, differentiable 2-level quantities before ever simulating a real
3-level system. qurveros already provides two:

- ``max_amp_loss``       = Tg * Omega_max         (peak drive)
- ``total_curvature``    = integral |Omega| dt    (total turning / L1 of Omega)

This module adds the **pulse energy** proxy

    E = integral Omega^2 dt = integral speed * curvature^2 dx,

which is the more physically motivated leakage proxy (perturbative leakage to a
third level scales with the drive power, ~Omega^2/Delta), and weights peaked
pulses more heavily than the L1 proxy. It is an ordinary frenet_dict loss, so it
can be added to any qurveros optimization.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def pulse_energy_loss(frenet_dict):
    r"""Pulse energy ``integral Omega^2 dt = integral speed * curvature^2 dx``."""
    integrand = frenet_dict["speed"] * frenet_dict["curvature"] ** 2
    return jnp.trapezoid(integrand, frenet_dict["x_values"])
