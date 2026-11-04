"""Energy charged at locational marginal prices (LMPs).

:func:`lmp` solves a period's least-cost DCOPF with the new loads and returns
the price at every bus from the LP's duals. :class:`EnergyCharge` is a fee
that charges each applicant its served energy at its bus's price in every
period, weighted by the period weights. Combine it with a commitment by
:class:`FeeSum`. The theory says the uniform rule's max-min fair point is then
an epsilon-equilibrium, with epsilon set by how far one applicant can move
its own price (paper: locational prices).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from .fees import Fee
from .network import DCNetwork


def lmp(net: DCNetwork, buses, served, period: int, active: set | None = None) -> np.ndarray:
    """Prices ($/MWh) at every bus of ``net`` in ``period`` with loads ``served`` at ``buses``.

    Least-cost redispatch of the existing generators (``net.gen_cost``) with
    branch limits added lazily; the price at bus ``k`` is the balance dual plus
    the branch duals times the PTDF column of ``k``.
    """
    if net.gen_cost is None:
        raise ValueError("the network has no generator costs (marginal_cost_c1_per_mwh)")
    prob = net.period_problem(period, buses)
    x = np.asarray(served, float)
    G = len(prob.gen_bounds)
    rows = set() if active is None else set(active)
    tol = 1e-7 * max(1.0, float(prob.limit.max()))
    rhs = prob.base + prob.flow_site @ x
    for _ in range(len(prob.limit) + 1):
        r = np.array(sorted(rows), dtype=int)
        A_ub = np.vstack([prob.flow_gen[r], -prob.flow_gen[r]]) if r.size else None
        b_ub = np.concatenate([prob.limit[r] + rhs[r], prob.limit[r] - rhs[r]]) if r.size else None
        res = linprog(net.gen_cost, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, G)), b_eq=[prob.total_demand + x.sum()],
                      bounds=prob.gen_bounds, method="highs")
        if res.status != 0:
            raise ValueError(f"period {period}: the loads cannot be served ({res.message})")
        flow = prob.flow_gen @ res.x - rhs
        bad = set(np.where(np.abs(flow) > prob.limit + tol)[0].tolist()) - rows
        if not bad:
            break
        rows |= bad
    price = np.full(len(net.buses), float(res.eqlin.marginals[0]))
    if r.size:
        m = res.ineqlin.marginals
        price += (m[:r.size] - m[r.size:]) @ net.ptdf[r]
    if active is not None:
        active |= rows
    return price


@dataclass
class EnergyCharge(Fee):
    """Served energy charged at the LMP of the applicant's bus, ``sum_t w_t LMP_t s_i``.

    ``period_weights`` converts one MW in period ``t`` into weighted MWh (hours
    times the represented days, times any annuity); ``share`` scales the charge
    (the paper's ``chi``). Prices depend on everyone's served load, so each new
    profile solves one LP per period (cached).
    """

    net: DCNetwork
    buses: list
    period_weights: np.ndarray
    share: float = 1.0
    _cache: dict = field(default_factory=dict, repr=False)
    _rows: dict = field(default_factory=dict, repr=False)

    def prices(self, served) -> np.ndarray:
        """Price at each applicant's bus in every period, ``(periods, n)``."""
        key = tuple(np.round(np.asarray(served, float), 9))
        if key not in self._cache:
            cols = [self.net.buses.index(str(b)) for b in self.buses]
            out = np.empty((self.net.periods, len(cols)))
            for t in range(self.net.periods):
                rows = self._rows.setdefault(t, set())
                out[t] = lmp(self.net, self.buses, served, t, active=rows)[cols]
            self._cache[key] = out
        return self._cache[key]

    def cost(self, requests, served):
        """``share * sum_t w_t LMP_t(served) * served_i`` for each applicant."""
        s = np.asarray(served, float)
        return self.share * (np.asarray(self.period_weights, float) @ self.prices(s)) * s

    @property
    def has_energy_charge(self) -> bool:
        """Marks fees whose cost depends on prices (the theory then gives an epsilon-equilibrium)."""
        return True


@dataclass
class FeeSum(Fee):
    """Several fees paid together, for example a commitment plus energy at the LMP."""

    fees: tuple

    def cost(self, requests, served):
        """The sum of the components' costs."""
        return sum(f.cost(requests, served) for f in self.fees)

    def marginal_request_cost(self):
        """The sum of the components' marginal request costs."""
        return sum(f.marginal_request_cost() for f in self.fees)

    @property
    def has_energy_charge(self) -> bool:
        """Whether any component charges energy at prices."""
        return any(getattr(f, "has_energy_charge", False) for f in self.fees)

    @property
    def forfeit_rate(self) -> float:
        return max((getattr(f, "forfeit_rate", 0.0) for f in self.fees), default=0.0)
