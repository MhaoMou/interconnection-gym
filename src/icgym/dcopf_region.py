"""An implicit batch region: the DCOPF itself, queried by linear programs.

:class:`DCOPFRegion` answers the questions the rules ask (water-filling levels,
hosting capacities, the planner's optimum, the proportional rule's common
fraction, the utilitarian allocation) by solving LPs over the served loads and
every period's generator redispatch, instead of listing facets. It is exact for
any number of applicants, at the cost of seconds per allocation rather than
microseconds; use :func:`icgym.network.export_region` for small batches that
need speed.

Two things keep the LPs small on large networks, without changing their
optimum: periods enter lazily (the LP starts with the periods found binding so
far, and any period the solution violates is added), and so do branch limits
within each period.

The region is the exact DCOPF region ``H``: relief that one new load gives
another (counterflow) is counted. The theory assumes its largest
downward-closed subset; where counterflow binds, the two can differ (compare
with an exported region).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

from .network import DCNetwork, PeriodProblem
from .waterfill import Stage

TOL = 1e-7


@dataclass
class _Extra:
    """Linear constraints and bounds on ``y = (x, t)`` shared by all periods."""

    A: np.ndarray | None = None
    b: np.ndarray | None = None
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None
    bounds: list | None = None


class DCOPFRegion:
    """Batch region of new loads at ``buses``, defined implicitly by the DCOPF."""

    is_implicit = True

    def __init__(self, net: DCNetwork, buses, applicant_names=None, initial_periods: int = 3):
        self.net = net
        self.buses = [str(b) for b in buses]
        self.applicant_names = tuple(applicant_names or (f"bus_{b}" for b in self.buses))
        self._problems: dict[int, PeriodProblem] = {}
        peak = np.argsort(-net.demand.sum(axis=1))[:initial_periods]
        self.binding: set[int] = set(int(t) for t in peak)
        self.lp_count = 0
        self._h = None

    @property
    def n(self) -> int:
        return len(self.buses)

    def _period(self, t: int) -> PeriodProblem:
        if t not in self._problems:
            self._problems[t] = self.net.period_problem(t, self.buses)
        return self._problems[t]

    # ------------------------------------------------------------------ the LP
    def optimize(self, c, extra: _Extra) -> tuple[float, np.ndarray] | None:
        """``max c'y`` over ``y = (x, t)`` with ``extra`` and every period feasible.

        Returns ``(value, x)``, or ``None`` if infeasible.
        """
        n = self.n
        c = np.asarray(c, dtype=float)
        for _ in range(self.net.periods + 1):
            periods = sorted(self.binding)
            res = self._joint_lp(c, extra, periods)
            if res is None:
                return None
            val, y, ps = res
            x = y[:n]
            changed = False
            for t, p in zip(periods, ps):  # branch rows within the included periods
                prob = self._period(t)
                flow = prob.flow_gen @ p - prob.flow_site @ x - prob.base
                tol = TOL * max(1.0, float(prob.limit.max()))
                bad = set(np.where(np.abs(flow) > prob.limit + tol)[0].tolist()) - prob.active
                if bad:
                    prob.active |= bad
                    changed = True
            if changed:
                continue
            for t in range(self.net.periods):  # periods left out
                if t not in self.binding and not self._feasible_lazy(t, x):
                    self.binding.add(t)
                    changed = True
            if not changed:
                return val, x
        raise RuntimeError("lazy period generation did not converge")

    def _feasible_lazy(self, t: int, x) -> bool:
        prob = self._period(t)
        G = len(prob.gen_bounds)
        tol = TOL * max(1.0, float(prob.limit.max()))
        for _ in range(len(prob.limit) + 1):
            rows = np.array(sorted(prob.active), dtype=int)
            rhs = prob.base[rows] + prob.flow_site[rows] @ x if rows.size else None
            A_ub = np.vstack([prob.flow_gen[rows], -prob.flow_gen[rows]]) if rows.size else None
            b_ub = np.concatenate([prob.limit[rows] + rhs, prob.limit[rows] - rhs]) if rows.size else None
            res = linprog(np.zeros(G), A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, G)),
                          b_eq=[prob.total_demand + float(np.sum(x))], bounds=prob.gen_bounds, method="highs")
            self.lp_count += 1
            if res.status != 0:
                return False
            flow = prob.flow_gen @ res.x - prob.flow_site @ x - prob.base
            bad = set(np.where(np.abs(flow) > prob.limit + tol)[0].tolist()) - prob.active
            if not bad:
                return True
            prob.active |= bad
        raise RuntimeError("lazy branch generation did not converge")

    def _joint_lp(self, c, extra: _Extra, periods):
        n, ny = self.n, self.n + 1
        Gs = [len(self._period(t).gen_bounds) for t in periods]
        nvar = ny + sum(Gs)
        cols = np.concatenate([[0], np.cumsum(Gs)]) + ny
        A_rows, b_rows, E_rows, e_rows = [], [], [], []
        if extra.A is not None:
            A_rows.append(csr_matrix(np.hstack([extra.A, np.zeros((extra.A.shape[0], nvar - ny))])))
            b_rows.append(extra.b)
        if extra.A_eq is not None:
            E_rows.append(csr_matrix(np.hstack([extra.A_eq, np.zeros((extra.A_eq.shape[0], nvar - ny))])))
            e_rows.append(extra.b_eq)
        bounds = list(extra.bounds) if extra.bounds is not None else [(0, None)] * n + [(None, None)]
        for k, t in enumerate(periods):
            prob = self._period(t)
            G = Gs[k]
            rows = np.array(sorted(prob.active), dtype=int)
            if rows.size:
                blk = np.zeros((rows.size, nvar))
                blk[:, :n] = -prob.flow_site[rows]
                blk[:, cols[k]:cols[k] + G] = prob.flow_gen[rows]
                A_rows.append(csr_matrix(np.vstack([blk, -blk])))
                b_rows.append(np.concatenate([prob.limit[rows] + prob.base[rows], prob.limit[rows] - prob.base[rows]]))
            bal = np.zeros((1, nvar))
            bal[0, :n] = -1.0
            bal[0, cols[k]:cols[k] + G] = 1.0
            E_rows.append(csr_matrix(bal))
            e_rows.append([prob.total_demand])
            bounds += prob.gen_bounds
        obj = np.zeros(nvar)
        obj[:ny] = -c
        A_ub = vstack(A_rows).tocsr() if A_rows else None
        b_ub = np.concatenate(b_rows) if b_rows else None
        res = None
        for method in ("highs", "highs-ds", "highs-ipm"):
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=vstack(E_rows).tocsr(), b_eq=np.concatenate(e_rows),
                          bounds=bounds, method=method)
            self.lp_count += 1
            if res.status in (0, 2):
                break
        if res.status == 2:
            return None
        if res.status != 0:
            raise RuntimeError(res.message)
        y = res.x[:ny]
        ps = [res.x[cols[k]:cols[k] + Gs[k]] for k in range(len(periods))]
        return float(-res.fun), y, ps

    # ------------------------------------------------------------------ queries
    def _bounds(self, fixed: dict, t_upper=None):
        b = []
        for i in range(self.n):
            b.append((fixed[i], fixed[i]) if i in fixed else (0.0, None))
        b.append((0.0, t_upper))
        return b

    def hosting_capacities(self) -> np.ndarray:
        if self._h is None:
            h = []
            for i in range(self.n):
                c = np.zeros(self.n + 1)
                c[i] = 1.0
                fixed = {j: 0.0 for j in range(self.n) if j != i}
                h.append(self.optimize(c, _Extra(bounds=self._bounds(fixed)))[0])
            self._h = np.array(h)
        return self._h

    def max_total(self) -> tuple[float, np.ndarray]:
        c = np.concatenate([np.ones(self.n), [0.0]])
        val, x = self.optimize(c, _Extra(bounds=self._bounds({})))
        return val, x

    def contains(self, s, tol: float = 1e-6) -> bool:
        s = np.asarray(s, dtype=float)
        return all(self._feasible_lazy(t, s * (1 - tol)) for t in range(self.net.periods))

    def max_common_fraction(self, r) -> float:
        """Largest ``alpha <= 1`` with ``alpha r`` feasible (proportional rule)."""
        r = np.asarray(r, dtype=float)
        A_eq = np.zeros((self.n, self.n + 1))
        A_eq[:, :self.n] = np.eye(self.n)
        A_eq[:, -1] = -r
        res = self.optimize(np.concatenate([np.zeros(self.n), [1.0]]),
                            _Extra(A_eq=A_eq, b_eq=np.zeros(self.n), bounds=self._bounds({}, 1.0)))
        return 0.0 if res is None else min(1.0, res[0])

    def utilitarian(self, r) -> np.ndarray:
        """Largest total with ``s <= r``; ties broken by the largest smallest fraction."""
        r = np.asarray(r, dtype=float)
        bounds = [(0.0, float(u)) for u in r] + [(0.0, 0.0)]
        total, x = self.optimize(np.concatenate([np.ones(self.n), [0.0]]), _Extra(bounds=bounds))
        idx = np.where(r > TOL)[0]
        if idx.size == 0:
            return np.zeros(self.n)
        A = np.zeros((idx.size + 1, self.n + 1))
        A[np.arange(idx.size), idx] = -1.0
        A[np.arange(idx.size), -1] = r[idx]
        A[-1, :self.n] = -1.0
        b = np.concatenate([np.zeros(idx.size), [-(total - 1e-7 * max(1.0, total))]])
        res = self.optimize(np.concatenate([np.zeros(self.n), [1.0]]),
                            _Extra(A=A, b=b, bounds=[(0.0, float(u)) for u in r] + [(0.0, 1.0)]))
        return x if res is None else res[1]

    def water_fill(self, requests, weights=None, active=None, return_stages: bool = False):
        """Water-filling ``U(r)`` by LPs: raise the level, then find who is blocked."""
        n = self.n
        r = np.asarray(requests, dtype=float).reshape(n)
        lam = np.ones(n) if weights is None else np.asarray(weights, dtype=float).reshape(n)
        rising = np.ones(n, bool) if active is None else np.asarray(active, bool).copy()
        rising &= r > TOL
        s = np.zeros(n)
        fixed = {i: 0.0 for i in range(n) if not rising[i]}
        stages = []
        scale = max(1.0, float(np.max(self.hosting_capacities())))
        while rising.any():
            idx = np.where(rising)[0]
            cap = float(np.min(r[idx] / lam[idx]))
            A_eq = np.zeros((idx.size, n + 1))
            A_eq[np.arange(idx.size), idx] = 1.0
            A_eq[:, -1] = -lam[idx]
            res = self.optimize(np.concatenate([np.zeros(n), [1.0]]),
                                _Extra(A_eq=A_eq, b_eq=np.zeros(idx.size),
                                       bounds=self._bounds(fixed, cap if np.isfinite(cap) else None)))
            if res is None:
                raise RuntimeError("level LP infeasible")
            level = res[0]
            s[idx] = lam[idx] * level
            stop = rising & (s >= r - TOL * scale)
            for i in idx:
                if stop[i]:
                    continue
                c = np.zeros(n + 1)
                c[i] = 1.0
                others = {j: float(s[j]) for j in range(n) if j != i}
                own = self.optimize(c, _Extra(bounds=self._bounds(others)))
                if own is None or own[0] <= s[i] + 1e-6 * scale:
                    stop[i] = True
            if not stop.any():
                raise RuntimeError("no applicant blocked at the level")
            if return_stages:
                stages.append(Stage(level, tuple(np.where(stop)[0].tolist()), ()))
            for i in np.where(stop)[0]:
                fixed[int(i)] = float(s[i])
            rising &= ~stop
        s = np.minimum(s, r)
        return (s, stages) if return_stages else s
