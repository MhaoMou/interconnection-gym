"""Devices dispatched over the network's real day: generators, batteries, and curtailment.

:class:`DeviceRegion` is the exact batch region of served loads when the
applicants also operate devices at their buses, every hour of every day:

* a per-period **load scale** ``m[t, i]`` multiplies applicant ``i``'s served
  load in period ``t``: ``1 - phi_i`` in the peak periods for a curtailable
  fraction ``phi_i``, or ``0`` for non-firm load that is only served off peak;
* **on-site generators** of capacity ``G_i`` run each hour within a daily
  energy budget of ``run_hours * G_i`` MWh, never export, and (``split=
  'proportional'``, the gross rule) run in proportion to their capacities, so
  the capacity they free is shared;
* **batteries** of power ``P_i`` and ``storage_hours`` of energy cycle within
  each day (charged when the network has room, discharged when it is tight).

Applicant ``i``'s net load in period ``t`` is ``m[t, i] x_i - g[t, i] -
d[t, i] + c[t, i] >= 0``, and the net loads must fit that period's DCOPF after
the existing generators are redispatched. Devices couple the hours of a day, so
days, not periods, enter the LPs lazily, and branch limits enter lazily within
each period. The region plugs into every rule and game, like
:class:`icgym.DCOPFRegion`. Unlike the constant-theta peak model of
:mod:`icgym.devices`, the capacity a generator frees per MW follows the real
daily load shape.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

from .dcopf_region import TOL, DCOPFRegion, _Extra
from .network import DCNetwork


class DeviceRegion(DCOPFRegion):
    """Batch region of new loads at ``buses`` with on-site devices dispatched hour by hour.

    ``load_scale`` is a ``(periods, n)`` array (default all ones);
    ``generators`` the capacities ``G_i`` (MW); ``run_hours`` the daily energy
    budget in hours (``None``: no budget); ``storage_power`` the battery powers
    ``P_i`` (MW) with ``storage_hours`` of energy. Days are ``day_length``
    consecutive periods.
    """

    def __init__(self, net: DCNetwork, buses, load_scale=None, generators=None, run_hours: float | None = None,
                 split: str = "proportional", storage_power=None, storage_hours: float = 4.0,
                 day_length: int = 24, applicant_names=None, initial_days: int = 1):
        super().__init__(net, buses, applicant_names=applicant_names, initial_periods=0)
        n, T = self.n, net.periods
        if T % day_length:
            raise ValueError(f"{T} periods are not whole days of {day_length}")
        if split not in ("proportional", "free"):
            raise ValueError("split must be 'proportional' or 'free'")
        self.scale = np.ones((T, n)) if load_scale is None else np.asarray(load_scale, float).reshape(T, n)
        self.G = np.zeros(n) if generators is None else np.asarray(generators, float).reshape(n)
        self.P = np.zeros(n) if storage_power is None else np.asarray(storage_power, float).reshape(n)
        self.run_hours, self.split, self.eta, self.L = run_hours, split, float(storage_hours), day_length
        self.days = T // day_length
        peak_day = int(np.argmax(net.demand.sum(axis=1)) // day_length)
        order = [peak_day] + [d for d in range(self.days) if d != peak_day]
        self.binding_days: set[int] = set(order[:initial_days])

    # ------------------------------------------------------------------ one day's LP block
    def _day_block(self, d: int, nvar0: int):
        """Columns and bounds of day ``d``'s variables, placed after ``nvar0`` columns.

        Returns ``(ncols, layout, bounds)``: the column count so far, one dict per
        period with the column ranges of the redispatch ``p`` and of the devices
        (``g``, ``c``, ``d``, ``soc``), and the variables' bounds.
        """
        n = self.n
        has_g, has_s = bool(np.any(self.G > 0)), bool(np.any(self.P > 0))
        periods = list(range(d * self.L, (d + 1) * self.L))
        col = nvar0
        layout = []
        bounds = []
        for t in periods:
            prob = self._period(t)
            G = len(prob.gen_bounds)
            lay = {"t": t, "p": (col, col + G)}
            col += G
            bounds += prob.gen_bounds
            if has_g:
                lay["g"] = (col, col + n)
                col += n
                bounds += [(0.0, float(self.G[i])) for i in range(n)]
            if has_s:
                lay["c"] = (col, col + n)
                col += n
                lay["d"] = (col, col + n)
                col += n
                lay["soc"] = (col, col + n)
                col += n
                bounds += [(0.0, float(self.P[i])) for i in range(n)] * 2
                bounds += [(0.0, float(self.eta * self.P[i])) for i in range(n)]
            layout.append(lay)
        return col, layout, bounds

    def _rows(self, layout, nvar):
        """Inequality and equality rows of the included periods (sparse), using active branches."""
        n = self.n
        has_g, has_s = "g" in layout[0], "c" in layout[0]
        A, b, E, e = [], [], [], []
        for lay in layout:
            t = lay["t"]
            prob = self._period(t)
            m = self.scale[t]
            p0, p1 = lay["p"]
            rows = np.array(sorted(prob.active), dtype=int)
            if rows.size:
                blk = np.zeros((rows.size, nvar))
                blk[:, :n] = -prob.flow_site[rows] * m[None, :]
                blk[:, p0:p1] = prob.flow_gen[rows]
                if has_g:
                    blk[:, lay["g"][0]:lay["g"][1]] = prob.flow_site[rows]
                if has_s:
                    blk[:, lay["d"][0]:lay["d"][1]] = prob.flow_site[rows]
                    blk[:, lay["c"][0]:lay["c"][1]] = -prob.flow_site[rows]
                A.append(csr_matrix(np.vstack([blk, -blk])))
                b.append(np.concatenate([prob.limit[rows] + prob.base[rows], prob.limit[rows] - prob.base[rows]]))
            bal = np.zeros((1, nvar))
            bal[0, :n] = -m
            bal[0, p0:p1] = 1.0
            if has_g:
                bal[0, lay["g"][0]:lay["g"][1]] = 1.0
            if has_s:
                bal[0, lay["d"][0]:lay["d"][1]] = 1.0
                bal[0, lay["c"][0]:lay["c"][1]] = -1.0
            E.append(csr_matrix(bal))
            e.append([prob.total_demand])
            if has_g or has_s:   # no export: net load >= 0
                ne = np.zeros((n, nvar))
                ne[np.arange(n), np.arange(n)] = -m
                if has_g:
                    ne[np.arange(n), lay["g"][0] + np.arange(n)] = 1.0
                if has_s:
                    ne[np.arange(n), lay["d"][0] + np.arange(n)] = 1.0
                    ne[np.arange(n), lay["c"][0] + np.arange(n)] = -1.0
                A.append(csr_matrix(ne))
                b.append(np.zeros(n))
        # day-level constraints, for each day in the layout
        for d0 in range(0, len(layout), self.L):
            day = layout[d0:d0 + self.L]
            if has_g and self.run_hours is not None:
                bud = np.zeros((n, nvar))
                for lay in day:
                    bud[np.arange(n), lay["g"][0] + np.arange(n)] = 1.0
                A.append(csr_matrix(bud))
                b.append(self.run_hours * self.G)
            if has_g and self.split == "proportional":
                act = np.where(self.G > 0)[0]
                for lay in day:
                    for j in act[1:]:
                        row = np.zeros((1, nvar))
                        row[0, lay["g"][0] + act[0]] = self.G[j]
                        row[0, lay["g"][0] + j] = -self.G[act[0]]
                        E.append(csr_matrix(row))
                        e.append([0.0])
            if has_s:
                for k, lay in enumerate(day):
                    nxt = day[(k + 1) % len(day)]
                    soc = np.zeros((n, nvar))
                    soc[np.arange(n), nxt["soc"][0] + np.arange(n)] = 1.0
                    soc[np.arange(n), lay["soc"][0] + np.arange(n)] = -1.0
                    soc[np.arange(n), lay["c"][0] + np.arange(n)] = -1.0
                    soc[np.arange(n), lay["d"][0] + np.arange(n)] = 1.0
                    E.append(csr_matrix(soc))
                    e.append(np.zeros(n))
        return A, b, E, e

    def _solve(self, days, c_y, extra: _Extra, fuel_weights=None):
        """One LP over ``y = (x, t)`` and the devices and redispatch of ``days``.

        ``c_y`` is the objective on ``y`` (maximized); ``fuel_weights`` (per
        period) instead minimizes the weighted generator output. Returns
        ``(value, x, layout, solution)`` or ``None`` if infeasible.
        """
        n, ny = self.n, self.n + 1
        col, layout, dev_bounds = ny, [], []
        for d in sorted(days):
            col, lay, bnd = self._day_block(d, col)
            layout += lay
            dev_bounds += bnd
        nvar = col
        A, b, E, e = self._rows(layout, nvar)
        if extra.A is not None:
            A.append(csr_matrix(np.hstack([extra.A, np.zeros((extra.A.shape[0], nvar - ny))])))
            b.append(extra.b)
        if extra.A_eq is not None:
            E.append(csr_matrix(np.hstack([extra.A_eq, np.zeros((extra.A_eq.shape[0], nvar - ny))])))
            e.append(extra.b_eq)
        bounds = (list(extra.bounds) if extra.bounds is not None else [(0, None)] * n + [(None, None)]) + dev_bounds
        obj = np.zeros(nvar)
        if fuel_weights is None:
            obj[:ny] = -np.asarray(c_y, float)
        else:
            for lay in layout:
                if "g" in lay:
                    obj[lay["g"][0]:lay["g"][1]] = fuel_weights[lay["t"]]
        A_ub = vstack(A).tocsr() if A else None
        b_ub = np.concatenate(b) if b else None
        res = None
        for method in ("highs", "highs-ds", "highs-ipm"):
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=vstack(E).tocsr(), b_eq=np.concatenate(e),
                          bounds=bounds, method=method)
            self.lp_count += 1
            if res.status in (0, 2):
                break
        if res.status == 2:
            return None
        if res.status != 0:
            raise RuntimeError(res.message)
        val = float(-res.fun) if fuel_weights is None else float(res.fun)
        return val, res.x[:n], layout, res.x

    def _add_violated_branches(self, layout, sol) -> bool:
        n = self.n
        x = sol[:n]
        changed = False
        for lay in layout:
            t = lay["t"]
            prob = self._period(t)
            y = self.scale[t] * x
            if "g" in lay:
                y = y - sol[lay["g"][0]:lay["g"][1]]
            if "c" in lay:
                y = y - sol[lay["d"][0]:lay["d"][1]] + sol[lay["c"][0]:lay["c"][1]]
            p = sol[lay["p"][0]:lay["p"][1]]
            flow = prob.flow_gen @ p - prob.flow_site @ y - prob.base
            tol = TOL * max(1.0, float(prob.limit.max()))
            bad = set(np.where(np.abs(flow) > prob.limit + tol)[0].tolist()) - prob.active
            if bad:
                prob.active |= bad
                changed = True
        return changed

    def _day_feasible(self, d: int, x) -> bool:
        x = np.asarray(x, float)
        bounds = [(float(v), float(v)) for v in x] + [(0.0, 0.0)]
        for _ in range(200):
            res = self._solve({d}, np.zeros(self.n + 1), _Extra(bounds=bounds))
            if res is None:
                return False
            if not self._add_violated_branches(res[2], res[3]):
                return True
        raise RuntimeError("lazy branch generation did not converge")

    def _optimize(self, c, extra: _Extra):
        for _ in range(self.days * 400):
            res = self._solve(self.binding_days, c, extra)
            if res is None:
                return None
            val, x, layout, sol = res
            if self._add_violated_branches(layout, sol):
                continue
            changed = False
            for d in range(self.days):
                if d not in self.binding_days and not self._day_feasible(d, x):
                    self.binding_days.add(d)
                    changed = True
            if not changed:
                return val, x
        raise RuntimeError("lazy day generation did not converge")

    def contains(self, s, tol: float = 1e-6) -> bool:
        """Whether served loads ``s`` can be hosted every day, with the devices dispatched."""
        s = np.asarray(s, float) * (1 - tol)
        return all(self._day_feasible(d, s) for d in range(self.days))

    # ------------------------------------------------------------------ dispatch
    def dispatch(self, s, period_weights=None) -> dict:
        """Device dispatch serving loads ``s`` with the least weighted generator output.

        Returns per-period arrays ``generation``, ``charge``, ``discharge``
        (``(periods, n)``) and each applicant's weighted generator energy
        ``fuel_energy`` (``sum_t w_t g[t, i]``, MWh weighted; ``w`` defaults to 1).
        """
        T, n = self.net.periods, self.n
        w = np.ones(T) if period_weights is None else np.asarray(period_weights, float)
        s = np.asarray(s, float)
        g, ch, dis = np.zeros((T, n)), np.zeros((T, n)), np.zeros((T, n))
        bounds = [(float(v), float(v)) for v in s] + [(0.0, 0.0)]
        for d in range(self.days):
            for _ in range(200):
                res = self._solve({d}, None, _Extra(bounds=bounds), fuel_weights=w)
                if res is None:
                    raise ValueError(f"served loads are infeasible on day {d}")
                if not self._add_violated_branches(res[2], res[3]):
                    break
            for lay in res[2]:
                t = lay["t"]
                if "g" in lay:
                    g[t] = res[3][lay["g"][0]:lay["g"][1]]
                if "c" in lay:
                    ch[t] = res[3][lay["c"][0]:lay["c"][1]]
                    dis[t] = res[3][lay["d"][0]:lay["d"][1]]
        return {"generation": g, "charge": ch, "discharge": dis, "fuel_energy": w @ g}


def peak_periods(net: DCNetwork, per_day: int, day_length: int = 24) -> list[int]:
    """The ``per_day`` periods of highest total demand in each day."""
    total = net.demand.sum(axis=1)
    out = []
    for d in range(net.periods // day_length):
        hours = np.arange(d * day_length, (d + 1) * day_length)
        out.extend(int(t) for t in hours[np.argsort(-total[hours], kind="stable")[:per_day]])
    return sorted(out)


def curtailment_scale(net: DCNetwork, n: int, peak, phi) -> np.ndarray:
    """Load scale for curtailable fractions ``phi`` (per applicant) in the ``peak`` periods."""
    m = np.ones((net.periods, n))
    m[np.asarray(peak, int)] = 1.0 - np.asarray(phi, float)[None, :]
    return m


def with_added_load(net: DCNetwork, buses, mw, periods=None) -> DCNetwork:
    """A copy of ``net`` whose demand includes ``mw`` at ``buses`` (in ``periods``, default all).

    Used to fix one tier of load (for example the firm loads) while another is allocated.
    """
    import dataclasses

    demand = net.demand.copy()
    rows = slice(None) if periods is None else np.asarray(periods, int)
    for b, x in zip(buses, np.asarray(mw, float)):
        demand[rows, net.buses.index(str(b))] += x
    return dataclasses.replace(net, demand=demand, root="", source_hash="", source=net.source + "+load")


# ---------------------------------------------------------------------- games on the network
from .devices import TwoPartGame  # noqa: E402
from .rules import UniformRule  # noqa: E402
from .waterfill import water_fill  # noqa: E402


class NetworkGenerationGame(TwoPartGame):
    """Requests and on-site generators dispatched over the network's real days; ``X[i] = (r_i, G_i)``.

    Generators (``G_i <= gamma r_i``) run within ``run_hours`` per day. Rules:

    * ``'gross'``: one common level on served load; generators run in proportion
      to their capacities, so the capacity they free is shared;
    * ``'credit'``: the extra total load all generators unlock together is
      split among their owners in proportion to capacity, and the uniform rule
      serves the requests net of these credits.

    Payoff: ``V s_i - kappa r_i - capital G_i - fuel_cost * fuel_i``, where
    ``fuel_i`` is the period-weighted energy of ``i``'s generator in the
    least-fuel dispatch that serves ``s``.
    """

    def __init__(self, net: DCNetwork, buses, value: float, kappa: float, gamma: float = 0.5,
                 run_hours: float | None = 2.0, capital: float = 0.0, fuel_cost: float = 0.0,
                 period_weights=None, rule: str = "gross", beta: float = 0.0, day_length: int = 24):
        if rule not in ("gross", "credit"):
            raise ValueError("rule must be 'gross' or 'credit'")
        self.net, self.buses = net, [str(b) for b in buses]
        self.value, self.kappa, self.gamma, self.run_hours = value, kappa, gamma, run_hours
        self.capital, self.fuel_cost, self.rule, self.beta, self.L = capital, fuel_cost, rule, beta, day_length
        self.weights = np.ones(net.periods) if period_weights is None else np.asarray(period_weights, float)
        self._regions: dict = {}
        self._cache: dict = {}
        self.base_region = self._region(np.zeros(self.n))

    @property
    def n(self) -> int:
        """Number of applicants."""
        return len(self.buses)

    def _region(self, G) -> DeviceRegion:
        key = tuple(np.round(np.asarray(G, float), 6))
        if key not in self._regions:
            self._regions[key] = DeviceRegion(self.net, self.buses, generators=np.array(key),
                                              run_hours=self.run_hours, day_length=self.L)
        return self._regions[key]

    def _clip(self, strategies):
        X = np.maximum(np.array(strategies, float).reshape(self.n, 2), 0.0)
        X[:, 1] = np.minimum(X[:, 1], self.gamma * X[:, 0])
        return X

    def credits(self, G) -> np.ndarray:
        """Each owner's credit: the jointly unlocked total load split by capacity."""
        G = np.asarray(G, float)
        if G.sum() <= 0:
            return np.zeros(self.n)
        gain = max(self._region(G).max_total()[0] - self.base_region.max_total()[0], 0.0)
        return gain * G / G.sum()

    def allocate(self, strategies) -> np.ndarray:
        """Served loads under the rule."""
        X = self._clip(strategies)
        key = tuple(np.round(X.ravel(), 6))
        if key in self._cache:
            return self._cache[key][0]
        r, G = X[:, 0], X[:, 1]
        if self.rule == "gross":
            s = UniformRule(beta=self.beta).allocate(self._region(G), r)
        else:
            c = self.credits(G)
            net_r = np.maximum(r - c, 0.0)
            share = water_fill(self.base_region, net_r) if net_r.sum() > 0 else np.zeros(self.n)
            s = np.minimum(r, c + share)
        fuel = np.zeros(self.n)
        if np.any(G > 0) and np.any(s > 0) and self.fuel_cost:
            if self.rule == "gross":
                fuel = self._region(G).dispatch(s, self.weights)["fuel_energy"]
            else:   # under crediting, each generator covers at least its own credit
                fuel = self._region(G).dispatch(np.minimum(s, r), self.weights)["fuel_energy"]
        self._cache[key] = (s, fuel)
        return s

    def payoffs(self, strategies) -> np.ndarray:
        """``V s_i - kappa r_i - capital G_i - fuel_cost * fuel_i`` for each applicant."""
        X = self._clip(strategies)
        s = self.allocate(X)
        fuel = self._cache[tuple(np.round(X.ravel(), 6))][1]
        return self.value * s - self.kappa * X[:, 0] - self.capital * X[:, 1] - self.fuel_cost * fuel

    def unlocking_ratio(self, i: int = 0, size: float = 100.0) -> float:
        """MW of extra hosting capacity per MW of generator, for applicant ``i`` alone (``theta``)."""
        G = np.zeros(self.n)
        G[i] = size
        return (self._region(G).hosting_capacities()[i] - self.base_region.hosting_capacities()[i]) / size

    def predicted_equilibrium(self) -> dict:
        """No closed form on a real network (the unlocking ratio varies with load); use grids."""
        return {"known": False, "reason": "network dispatch: the unlocking ratio varies with load"}

    def _upper(self, i, X):
        return float(2.0 * max(self.base_region.hosting_capacities()[i], np.asarray(X)[i, 0]))

    def _second_options(self, i, x1, X, grid):
        return np.linspace(0.0, self.gamma * x1, grid)

    def _second_fraction(self, i, x1, x2, X):
        cap = self.gamma * x1
        return x2 / cap if cap > 0 else 0.0

    def _second_from_fraction(self, i, x1, frac, X):
        return frac * self.gamma * x1

    def _second_from_action(self, i, x1, a1, request_cap):
        return a1 * self.gamma * x1

    def _fraction_bounds(self, i, x1, X, grid):
        return 0.0, 1.0


class NetworkNonFirmGame(TwoPartGame):
    """Firm and non-firm service on the network, with peak periods taken from the load shape.

    ``peak`` lists the peak periods (see :func:`peak_periods`); ``value`` is
    ``V`` and ``peak_value`` is ``V^p``, the value of the load in the peak
    periods. Rules:

    * ``'two_tier'``: ``X[i] = (firm request, non-firm request)``; firm requests
      are water-filled over all periods, non-firm requests over the off-peak
      periods with the firm loads fixed;
    * ``'gross'``: ``X[i] = (request, curtailable fraction phi_i <= gamma)``; one
      common level, net loads ``(1 - phi_i) s_i`` in the peak periods.
    """

    def __init__(self, net: DCNetwork, buses, peak, value: float, peak_value: float, kappa: float,
                 rule: str = "two_tier", gamma: float = 0.05, day_length: int = 24):
        if rule not in ("two_tier", "gross"):
            raise ValueError("rule must be 'two_tier' or 'gross'")
        self.net, self.buses, self.peak = net, [str(b) for b in buses], sorted(int(t) for t in peak)
        self.value, self.peak_value, self.kappa, self.rule, self.gamma, self.L = (
            value, peak_value, kappa, rule, gamma, day_length)
        self.firm_region = DeviceRegion(net, self.buses, day_length=day_length)
        self._regions: dict = {}
        self._cache: dict = {}

    @property
    def n(self) -> int:
        """Number of applicants."""
        return len(self.buses)

    @property
    def offpeak_value(self) -> float:
        """Value of the load outside the peak periods, ``V - V^p``."""
        return self.value - self.peak_value

    def _offpeak_scale(self, n):
        m = np.ones((self.net.periods, n))
        m[self.peak] = 0.0
        return m

    def nonfirm_region(self, firm) -> DeviceRegion:
        """Region of non-firm loads (served off peak only) with the firm loads fixed."""
        key = tuple(np.round(np.asarray(firm, float), 6))
        if key not in self._regions:
            # firm loads at their limit leave the binding period marginally infeasible;
            # back them off by a relative 1e-9 (micro-MW) so the residual region is well defined
            net2 = with_added_load(self.net, self.buses, np.asarray(firm, float) * (1 - 1e-9))
            self._regions[key] = DeviceRegion(net2, self.buses, load_scale=self._offpeak_scale(self.n),
                                              day_length=self.L)
        return self._regions[key]

    def gross_region(self, phi) -> DeviceRegion:
        """Served loads with curtailable fractions ``phi`` in the peak periods."""
        key = ("gross",) + tuple(np.round(np.asarray(phi, float), 6))
        if key not in self._regions:
            self._regions[key] = DeviceRegion(self.net, self.buses,
                                              load_scale=curtailment_scale(self.net, self.n, self.peak, phi),
                                              day_length=self.L)
        return self._regions[key]

    def _clip(self, strategies):
        X = np.maximum(np.array(strategies, float).reshape(self.n, 2), 0.0)
        if self.rule == "gross":
            X[:, 1] = np.minimum(X[:, 1], self.gamma)
        return X

    def allocate(self, strategies):
        """Served loads: ``(firm, non-firm)`` per applicant for the two-tier rule, ``s`` for the gross rule."""
        X = self._clip(strategies)
        key = tuple(np.round(X.ravel(), 6))
        if key not in self._cache:
            if self.rule == "two_tier":
                sf = water_fill(self.firm_region, X[:, 0])
                sn = water_fill(self.nonfirm_region(sf), X[:, 1]) if X[:, 1].sum() > 0 else np.zeros(self.n)
                self._cache[key] = np.column_stack([sf, sn])
            else:
                self._cache[key] = water_fill(self.gross_region(X[:, 1]), X[:, 0])
        return self._cache[key]

    def payoffs(self, strategies) -> np.ndarray:
        """Each applicant's payoff under the two-tier or the gross rule."""
        X = self._clip(strategies)
        s = self.allocate(X)
        if self.rule == "two_tier":
            return self.value * s[:, 0] + self.offpeak_value * s[:, 1] - self.kappa * X.sum(axis=1)
        return self.value * s - self.peak_value * X[:, 1] * s - self.kappa * X[:, 0]

    def predicted_equilibrium(self) -> dict:
        """Two-tier rule: firm ``w^p`` (all periods) and non-firm ``w^n`` (off peak, firm fixed)."""
        if self.rule != "two_tier":
            return {"known": False, "reason": "gross rule on a network: use grids"}
        from .waterfill import max_min_fair_point

        wp = max_min_fair_point(self.firm_region)
        wn = max_min_fair_point(self.nonfirm_region(wp)) if self.offpeak_value > self.kappa else np.zeros(self.n)
        return {"known": True, "kind": "tiers", "strategies": np.column_stack([wp, wn])}

    def _upper(self, i, X):
        return float(2.0 * max(self.firm_region.hosting_capacities()[i], np.asarray(X)[i, 0]))

    def _second_options(self, i, x1, X, grid):
        if self.rule == "gross":
            return np.linspace(0.0, self.gamma, grid)
        return np.linspace(0.0, max(self._upper(i, X) / 2, np.asarray(X)[i, 1]) * 2, grid)

    def _second_from_action(self, i, x1, a1, request_cap):
        if self.rule == "gross":
            return a1 * self.gamma
        return a1 * request_cap * self.hosting_capacity(i)
