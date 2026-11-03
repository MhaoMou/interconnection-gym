"""Ground truth from the theory: equilibria, efficiency, and their bounds.

These functions return what the paper proves, so that learned or observed
behavior can be scored against it.
"""

from __future__ import annotations

import numpy as np

from .fees import FlatFee
from .game import BatchGame
from .region import Region
from .rules import ProportionalRule, UniformRule
from .waterfill import max_min_fair_point


def stages(region: Region, weights=None):
    """Levels ``tau_k`` and blocked sets ``B_k`` of unbounded water-filling."""
    _, st = max_min_fair_point(region, weights=weights, return_stages=True)
    return st


def price_of_anarchy(region: Region, served) -> float:
    """Planner's best total over the total of ``served``."""
    best, _ = region.max_total()
    total = float(np.sum(served))
    return best / total if total > 0 else np.inf


def poa_bound(region: Region) -> float:
    """``(1 + nR)^2 / (4nR)`` with ``R`` the spread of hosting capacities (paper: price of anarchy)."""
    h = region.hosting_capacities()
    R, n = h.max() / h.min(), region.n
    return (1 + n * R) ** 2 / (4 * n * R)


def proportional_inflation(n: int, value: float, kappa: float, beta: float) -> float:
    """Request inflation ``rho`` of the proportional rule on one limit (paper: proportional rule)."""
    rho = max((1 - 1 / n) * value / kappa, 1.0)
    return min(rho, 1 / beta) if beta > 0 else rho


def predicted_equilibrium(game: BatchGame) -> dict:
    """What the theory predicts for this game, or ``{'known': False}``.

    * uniform (or weighted) rule with a per-MW cost below the value: the unique
      equilibrium is the (weighted) max-min fair point, served in full;
    * uniform rule with a flat fee: every profile in ``[w, w/beta]`` is an
      equilibrium, all serving ``w``;
    * proportional rule on one limit: ``r_i = rho b / (n a_i)``.
    """
    rule, fee, region = game.rule, game.fee, game.region
    kappa = fee.marginal_request_cost()
    V = game.value
    if isinstance(rule, UniformRule):
        w = max_min_fair_point(region, weights=rule.weights)
        if isinstance(fee, FlatFee):
            hi = w / rule.beta if rule.beta > 0 else np.full(game.n, np.inf)
            return {"known": True, "kind": "band", "served": w, "low": w, "high": hi}
        if np.all(0 < kappa) and np.all(kappa < V) and not getattr(fee, "forfeit_rate", 0.0):
            return {"known": True, "kind": "unique", "requests": w, "served": w}
    if isinstance(rule, ProportionalRule) and getattr(region, "m", 0) == 1 and np.allclose(V, V[0]) and kappa > 0:
        a, b = region.A[0], region.b[0]
        rho = proportional_inflation(game.n, float(V[0]), kappa, rule.beta)
        served = b / (game.n * a)
        return {"known": True, "kind": "equilibrium", "requests": rho * served, "served": served, "rho": rho}
    return {"known": False}
