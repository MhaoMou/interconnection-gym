"""The batch request game: applicants choose requests, the rule serves them.

Applicant ``i`` earns ``V_i s_i - cost_i(r, s)``, where ``s`` is the rule's
allocation of the requests ``r`` on the region. This is the reduced game of the
paper (one site per applicant at a fixed bus, constant value per MW).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from .fees import Fee, PerMWCommitment
from .region import Region
from .rules import AllocationRule, UniformRule
from .waterfill import water_fill


@dataclass
class BatchGame:
    region: Region
    rule: AllocationRule = field(default_factory=UniformRule)
    fee: Fee = field(default_factory=lambda: PerMWCommitment(0.4))
    value: float | np.ndarray = 1.0

    def __post_init__(self) -> None:
        self.value = np.broadcast_to(np.asarray(self.value, dtype=float), (self.region.n,)).copy()

    @property
    def n(self) -> int:
        return self.region.n

    def allocate(self, requests) -> np.ndarray:
        return self.rule.allocate(self.region, requests)

    def payoffs(self, requests, served=None) -> np.ndarray:
        r = np.asarray(requests, dtype=float)
        s = self.allocate(r) if served is None else served
        return self.value * s - self.fee.cost(r, s)

    def payoff(self, i: int, x: float, requests) -> float:
        r = np.array(requests, dtype=float)
        r[i] = x
        return float(self.payoffs(r)[i])

    def predicted_equilibrium(self) -> dict:
        """The theory's answer for this game (see :func:`icgym.predicted_equilibrium`)."""
        from .oracles import predicted_equilibrium

        return predicted_equilibrium(self)

    # ---------------------------------------------------------------- best responses
    def _exact_best_response(self, i: int, requests) -> float | None:
        """Closed form for the uniform rule without minimum allocation (paper: best-response dynamics)."""
        rule = self.rule
        if not (isinstance(rule, UniformRule) and rule.beta == 0.0):
            return None
        if not (0.0 < self.fee.marginal_request_cost() < self.value[i]):
            return None
        if getattr(self.fee, "forfeit_rate", 0.0):
            return None
        r = np.array(requests, dtype=float)
        r[i] = np.inf
        return float(water_fill(self.region, r, weights=rule.weights)[i])

    def _search_upper(self, i: int, requests) -> float:
        h = self.region.hosting_capacities()
        r = np.asarray(requests, dtype=float)
        beta = getattr(self.rule, "beta", 0.0)
        reach = h[i] / beta if beta > 0 else h[i]
        if getattr(self.region, "is_implicit", False):
            return float(min(max(h[i], reach, r.max()), 20 * h[i]) * 1.05)
        A = self.region.A
        rows = A[:, i] > 0
        # a request can matter up to the point where it overtakes every rival in
        # the cut order, or pushes the service fraction below beta
        rival = (A[rows] * r).max() / A[rows, i].min() if rows.any() else 0.0
        return float(min(max(h[i], reach, rival, r.max()), 20 * h[i]) * 1.05)

    def _kinks(self, i: int, requests) -> list[float]:
        """Requests at which the payoff can jump, and points just either side of them.

        The minimum allocation cuts an applicant when its service fraction falls
        below ``beta`` and drops the largest request or loading first, so the
        payoff jumps where a limit's common fraction reaches ``beta`` and where
        the applicant's request or loading ties a rival's. A grid alone can miss
        the narrow windows these create.
        """
        beta = getattr(self.rule, "beta", 0.0)
        if beta <= 0 or getattr(self.region, "is_implicit", False):
            return []
        A, b = self.region.A, self.region.b
        r = np.asarray(requests, dtype=float)
        others = np.delete(np.arange(self.n), i)
        pts = list(r[others])
        for c in np.where(A[:, i] > 0)[0]:
            rest = float(A[c, others] @ r[others])
            pts.append((b[c] / beta - rest) / A[c, i])
            pts.append(b[c] / A[c, i])
            pts += list(A[c, others] * r[others] / A[c, i])
        return [x * f for x in pts if x > 0 for f in (1 - 1e-9, 1.0, 1 + 1e-9)]

    def best_response(self, i: int, requests, method: str = "auto", grid: int = 400) -> tuple[float, float]:
        """Best request of applicant ``i`` against the others' requests.

        ``method='exact'`` uses the closed form when it applies (uniform rule,
        no minimum allocation, per-MW cost); ``'search'`` maximizes the payoff
        numerically over ``[0, upper]``, which also finds deviations that get a
        rival cut. Returns ``(request, payoff)``. Under the minimum allocation
        the supremum may not be attained; the search returns the best point found.
        """
        if method in ("auto", "exact"):
            x = self._exact_best_response(i, requests)
            if x is not None:
                return x, self.payoff(i, x, requests)
            if method == "exact":
                raise ValueError("no closed-form best response for this rule and fee")
        upper = self._search_upper(i, requests)
        xs = np.linspace(0.0, upper, grid)
        cands = list(xs) + [float(requests[i]), float(self.region.hosting_capacities()[i])]
        cands += [x for x in self._kinks(i, requests) if 0.0 <= x <= upper]
        vals = [self.payoff(i, x, requests) for x in cands]
        order = np.argsort(vals)[::-1][:4]
        best_x, best_v = cands[order[0]], vals[order[0]]
        step = upper / (grid - 1)
        for k in order:
            lo, hi = max(0.0, cands[k] - step), min(upper, cands[k] + step)
            res = minimize_scalar(lambda x: -self.payoff(i, x, requests), bounds=(lo, hi),
                                  method="bounded", options={"xatol": 1e-9 * max(1.0, upper)})
            if -res.fun > best_v + 1e-12:
                best_x, best_v = float(res.x), float(-res.fun)
        return best_x, best_v

    def exploitability(self, requests, method: str = "auto", grid: int = 400) -> np.ndarray:
        """Each applicant's gain from its best unilateral deviation (0 at an equilibrium)."""
        base = self.payoffs(requests)
        return np.array([max(0.0, self.best_response(i, requests, method, grid)[1] - base[i])
                         for i in range(self.n)])

    def best_response_dynamics(self, requests, max_rounds: int = 50, order: str = "cyclic",
                               tol: float = 1e-7, method: str = "auto", seed: int | None = None,
                               grid: int = 400, gain_tol: float | None = None):
        """Applicants best-respond in rounds, each once per round.

        Returns ``(requests, history, rounds)``, where ``rounds`` counts the rounds
        in which some request changed, or ``None`` if the profile did not settle
        within ``max_rounds``. A round settles only if no request moved by more
        than ``tol`` and no applicant could gain more than ``gain_tol`` (in payoff
        units; default ``1e-9 * max |V h|``) by deviating: under the minimum
        allocation a best response can be a vanishing undercut of a rival with a
        large gain, which is not an equilibrium.
        """
        rng = np.random.default_rng(seed)
        r = np.array(requests, dtype=float)
        history = [r.copy()]
        if gain_tol is None:
            gain_tol = 1e-9 * float(np.max(np.abs(self.value * self.region.hosting_capacities())))
        for k in range(1, max_rounds + 1):
            prev = r.copy()
            gain = 0.0
            idx = rng.permutation(self.n) if order == "random" else range(self.n)
            for i in idx:
                before = self.payoff(i, r[i], r)
                x, v = self.best_response(i, r, method, grid)
                gain = max(gain, v - before)
                r[i] = x
            history.append(r.copy())
            if np.allclose(r, prev, atol=tol, rtol=0) and gain <= gain_tol:
                return r, history, k - 1
        return r, history, None
