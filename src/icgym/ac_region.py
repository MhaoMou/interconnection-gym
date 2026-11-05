"""An implicit batch region under AC power flow: the ACOPF, queried by Ipopt.

:class:`ACOPFRegion` answers the same questions as
:class:`~icgym.dcopf_region.DCOPFRegion` (water-filling levels, hosting
capacities, the planner's optimum, the proportional rule's common fraction, the
utilitarian allocation), but a period hosts the new loads only if some
redispatch of the existing generators, *in active and reactive power*, meets the
full AC power-flow equations with

* apparent-power limits on both ends of every branch,
* bus voltage magnitudes within their limits, and
* generator active and reactive limits.

New loads draw reactive power at a fixed power factor (lagging). Losses, voltage
limits, and reactive power make the region nonconvex, and it need not be
downward closed (the theory's Assumption 1); :func:`downward_closure_check`
tests that assumption on samples of the region's boundary.

The nonlinear programs are solved by Ipopt through CasADi (the ``ac`` extra:
``pip install "interconnection-gym[ac]"``). Ipopt finds *local* optima. Every
query here is a feasible problem (no new load is always feasible) maximizing a
linear function of the new loads, and the region is usually star-shaped along
the directions the rules use, but a returned level can in principle fall short
of the global one. ``multistart`` re-solves from extra starting points.

Periods enter lazily, as in the DC region: the joint program starts with the
peak periods, and any period the solution cannot host is added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dcopf_region import DCOPFRegion, _Extra

try:  # optional dependency
    import casadi as ca
except ImportError:  # pragma: no cover
    ca = None


def _require_casadi():
    if ca is None:
        raise ImportError('ACOPFRegion needs CasADi: pip install "interconnection-gym[ac]"')


def _matrix(text: str, name: str) -> np.ndarray:
    body = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", text, re.S)
    if body is None:
        raise ValueError(f"mpc.{name} not found in the case file")
    rows = [line.split("%")[0].strip().rstrip(";") for line in body.group(1).strip().splitlines()]
    return np.array([[float(x) for x in r.split()] for r in rows if r])


@dataclass
class ACNetwork:
    """An AC network over periods: MATPOWER-style data in per unit of ``base_mva``.

    ``pd``/``qd`` are the base loads (periods x buses, MW and MVAr). Branch
    ``rate`` is the apparent-power limit in MVA (``inf`` for none).
    """

    buses: list[str]
    bus_type: np.ndarray
    gs: np.ndarray                  # shunt conductance (MW at 1 p.u.)
    bs: np.ndarray                  # shunt susceptance (MVAr at 1 p.u.)
    vmin: np.ndarray
    vmax: np.ndarray
    f: np.ndarray                   # branch from-bus index
    t: np.ndarray                   # branch to-bus index
    r: np.ndarray
    x: np.ndarray
    b: np.ndarray
    tap: np.ndarray                 # off-nominal ratio (1 for lines)
    shift: np.ndarray               # phase shift (radians)
    rate: np.ndarray                # MVA
    gen_bus: np.ndarray
    pmin: np.ndarray
    pmax: np.ndarray
    qmin: np.ndarray
    qmax: np.ndarray
    pd: np.ndarray                  # periods x buses
    qd: np.ndarray
    base_mva: float = 100.0
    source: str = ""
    branch_names: list = field(default_factory=list)

    @property
    def demand(self) -> np.ndarray:
        """Active base load, periods x buses (MW)."""
        return self.pd

    @property
    def periods(self) -> int:
        """Number of periods."""
        return self.pd.shape[0]

    @classmethod
    def from_matpower(cls, case_file, rate=None, period_multipliers=None, pmin=None, source: str = ""):
        """Read a MATPOWER ``.m`` case.

        ``rate`` maps ``"f-t"`` to an MVA limit (overriding ``rateA``; a zero
        ``rateA`` means no limit). ``period_multipliers`` scales the base active
        and reactive loads, one period per multiplier (default: one period).
        ``pmin`` maps a generator's bus to its minimum output (MW).
        """
        text = Path(case_file).read_text()
        base = float(re.search(r"mpc\.baseMVA\s*=\s*([\d.eE+-]+)", text).group(1))
        bus, gen, br = _matrix(text, "bus"), _matrix(text, "gen"), _matrix(text, "branch")
        ids = [str(int(v)) for v in bus[:, 0]]
        idx = {int(v): i for i, v in enumerate(bus[:, 0])}
        br = br[br[:, 10] > 0]
        gen = gen[gen[:, 7] > 0]
        names = [f"{int(a)}-{int(c)}" for a, c in br[:, :2]]
        rate_arr = np.array([float(rate[nm]) if rate and nm in rate else (v if v > 0 else np.inf)
                             for nm, v in zip(names, br[:, 5])])
        tap = np.where(br[:, 8] == 0, 1.0, br[:, 8])
        mult = np.ones(1) if period_multipliers is None else np.asarray(period_multipliers, float)
        p_min = gen[:, 9].copy()
        if pmin:
            for k, gb in enumerate(gen[:, 0]):
                if str(int(gb)) in pmin:
                    p_min[k] = float(pmin[str(int(gb))])
        return cls(buses=ids, bus_type=bus[:, 1].astype(int), gs=bus[:, 4], bs=bus[:, 5],
                   vmin=bus[:, 12], vmax=bus[:, 11],
                   f=np.array([idx[int(v)] for v in br[:, 0]]), t=np.array([idx[int(v)] for v in br[:, 1]]),
                   r=br[:, 2], x=br[:, 3], b=br[:, 4], tap=tap, shift=np.deg2rad(br[:, 9]), rate=rate_arr,
                   gen_bus=np.array([idx[int(v)] for v in gen[:, 0]]), pmin=p_min, pmax=gen[:, 8],
                   qmin=gen[:, 4], qmax=gen[:, 3],
                   pd=np.outer(mult, bus[:, 2]), qd=np.outer(mult, bus[:, 3]), base_mva=base,
                   source=source or Path(case_file).name, branch_names=names)

    # ------------------------------------------------------------------ physics
    def admittances(self):
        """Branch two-port admittances ``(Yff, Yft, Ytf, Ytt)`` as complex arrays (p.u.)."""
        ys = 1.0 / (self.r + 1j * self.x)
        tau = self.tap * np.exp(1j * self.shift)
        bc = 1j * self.b / 2
        return (ys + bc) / (tau * np.conj(tau)), -ys / np.conj(tau), -ys / tau, ys + bc

    def branch_flows(self, vm, va):
        """Complex power entering each branch at its from and to ends (p.u.), as arrays."""
        V = vm * np.exp(1j * va)
        yff, yft, ytf, ytt = self.admittances()
        If = yff * V[self.f] + yft * V[self.t]
        It = ytf * V[self.f] + ytt * V[self.t]
        return V[self.f] * np.conj(If), V[self.t] * np.conj(It)

    def mismatch(self, vm, va, pg, qg, pd, qd):
        """Bus power mismatch (p.u.) of an operating point: injections minus generation plus load."""
        sf, st = self.branch_flows(vm, va)
        nb = len(self.buses)
        inj = np.zeros(nb, complex)
        np.add.at(inj, self.f, sf)
        np.add.at(inj, self.t, st)
        inj += (self.gs - 1j * self.bs) / self.base_mva * vm ** 2
        gen = np.zeros(nb, complex)
        np.add.at(gen, self.gen_bus, (pg + 1j * qg) / self.base_mva)
        return inj - gen + (pd + 1j * qd) / self.base_mva


class _Period:
    """CasADi expressions of one period's ACOPF constraints, given its variables."""

    def __init__(self, net: ACNetwork, cols, tanphi):
        self.net, self.cols, self.tanphi = net, np.asarray(cols), np.asarray(tanphi)
        nb, ng = len(net.buses), len(net.gen_bus)
        self.nb, self.ng = nb, ng
        self.nv = 2 * nb + 2 * ng
        yff, yft, ytf, ytt = net.admittances()
        self.y = [(a.real, a.imag) for a in (yff, yft, ytf, ytt)]
        self.limited = np.where(np.isfinite(net.rate))[0]
        ref = np.where(net.bus_type == 3)[0]
        if ref.size != 1:
            raise ValueError("the case needs exactly one reference bus (type 3)")
        self.ref = int(ref[0])
        nl = len(net.f)
        # sparse incidence: branch ends, generators, and new loads onto buses
        self.Cf = ca.DM.triplet([int(v) for v in net.f], list(range(nl)), ca.DM.ones(nl), nb, nl)
        self.Ct = ca.DM.triplet([int(v) for v in net.t], list(range(nl)), ca.DM.ones(nl), nb, nl)
        self.Cg = ca.DM.triplet([int(v) for v in net.gen_bus], list(range(ng)), ca.DM.ones(ng), nb, ng)
        k = len(self.cols)
        self.Cs = ca.DM.triplet([int(v) for v in self.cols], list(range(k)), ca.DM.ones(k), nb, k)

    def split(self, z):
        nb, ng = self.nb, self.ng
        return z[:nb], z[nb:2 * nb], z[2 * nb:2 * nb + ng], z[2 * nb + ng:]

    def constraints(self, z, load, pd, qd):
        """``(g, lbg, ubg)`` of the period: power balance, branch limits, and the reference angle."""
        net, base = self.net, self.net.base_mva
        va, vm, pg, qg = self.split(z)
        f, t = net.f, net.t
        (gff, bff), (gft, bft), (gtf, btf), (gtt, btt) = self.y
        vf, vt, af, at = vm[f], vm[t], va[f], va[t]
        cft, sft = ca.cos(af - at), ca.sin(af - at)
        pf = vf ** 2 * gff + vf * vt * (gft * cft + bft * sft)
        qf = -vf ** 2 * bff + vf * vt * (gft * sft - bft * cft)
        pt = vt ** 2 * gtt + vf * vt * (gtf * cft - btf * sft)
        qt = -vt ** 2 * btt + vf * vt * (-gtf * sft - btf * cft)
        nb = self.nb
        Cf, Ct, Cg, Cs = self.Cf, self.Ct, self.Cg, self.Cs
        pnew = ca.mtimes(Cs, load) / base
        qnew = ca.mtimes(Cs, load * ca.DM(self.tanphi)) / base
        pbal = (ca.mtimes(Cf, pf) + ca.mtimes(Ct, pt) + (net.gs / base) * vm ** 2
                - ca.mtimes(Cg, pg) / base + pd / base + pnew)
        qbal = (ca.mtimes(Cf, qf) + ca.mtimes(Ct, qt) - (net.bs / base) * vm ** 2
                - ca.mtimes(Cg, qg) / base + qd / base + qnew)
        L = self.limited
        rate2 = (net.rate[L] / base) ** 2
        g = [pbal, qbal, pf[L] ** 2 + qf[L] ** 2 - rate2, pt[L] ** 2 + qt[L] ** 2 - rate2, va[self.ref]]
        lb = [np.zeros(nb), np.zeros(nb), -np.inf * np.ones(L.size), -np.inf * np.ones(L.size), [0.0]]
        ub = [np.zeros(nb), np.zeros(nb), np.zeros(L.size), np.zeros(L.size), [0.0]]
        return ca.vertcat(*g), np.concatenate(lb), np.concatenate(ub)

    def bounds(self):
        net = self.net
        lb = np.concatenate([-np.pi * np.ones(self.nb), net.vmin, net.pmin, net.qmin])
        ub = np.concatenate([np.pi * np.ones(self.nb), net.vmax, net.pmax, net.qmax])
        return lb, ub

    def start(self):
        net = self.net
        return np.concatenate([np.zeros(self.nb), np.ones(self.nb), (net.pmin + net.pmax) / 2,
                               (net.qmin + net.qmax) / 2])


_IPOPT = {"print_time": 0, "ipopt.print_level": 0, "ipopt.sb": "yes", "ipopt.tol": 1e-9,
          "ipopt.constr_viol_tol": 1e-9, "ipopt.max_iter": 3000}
_OK = ("Solve_Succeeded", "Solved_To_Acceptable_Level")
_RESCUE = 8          # random restarts after Ipopt fails from the warm and flat starts


class ACOPFRegion(DCOPFRegion):
    """Batch region of new loads at ``buses``, defined implicitly by the multi-period ACOPF.

    ``power_factor`` (one number or one per applicant, lagging) sets the
    reactive power each new load draws. Queries return MW of new active load.
    """

    is_implicit = True

    def __init__(self, net: ACNetwork, buses, power_factor=0.95, applicant_names=None, initial_periods: int = 3,
                 multistart: int = 0, seed: int = 0, certify: float = 1e-3):
        _require_casadi()
        self.net = net
        self.buses = [str(v) for v in buses]
        self.applicant_names = tuple(applicant_names or (f"bus_{v}" for v in self.buses))
        pf = np.broadcast_to(np.asarray(power_factor, float), (len(self.buses),))
        if np.any((pf <= 0) | (pf > 1)):
            raise ValueError("power factors must be in (0, 1]")
        self.power_factor = pf.copy()
        cols = [net.buses.index(v) for v in self.buses]
        self._per = _Period(net, cols, np.sqrt(1 - pf ** 2) / pf)
        peak = np.argsort(-net.pd.sum(axis=1))[:initial_periods]
        self.binding: set[int] = set(int(t) for t in peak)
        self.lp_count = 0          # NLP solves (name kept from the DC region)
        self._h = None
        self.multistart, self._rng = multistart, np.random.default_rng(seed)
        self._warm: dict[int, np.ndarray] = {}
        self._solvers: dict = {}
        self._feas = None
        self.certify = certify                     # relative step of the feasibility checks in water-filling
        self.unblocked_stages: list[dict] = []     # water-filling stages where no applicant was blocked alone
        self.bisected_stages = 0                   # water-filling levels found by bisection after Ipopt failed

    # ------------------------------------------------------------------ NLPs
    def _joint_solver(self, K: int, mA: int, mE: int):
        key = (K, mA, mE)
        if key in self._solvers:
            return self._solvers[key]
        n, nb, per = self.n, self._per.nb, self._per
        y = ca.SX.sym("y", n + 1)
        zs = [ca.SX.sym(f"z{k}", per.nv) for k in range(K)]
        P = ca.SX.sym("P", 2 * nb * K + (n + 1) + mA * (n + 2) + mE * (n + 2))
        # parameters: each period's (pd, qd), then c, A, bA, E, bE
        sizes = [n + 1, mA * (n + 1), mA, mE * (n + 1), mE]
        cuts = np.cumsum([2 * nb * K] + sizes)
        c, Af, bA, Ef, bE = (P[int(a):int(b)] for a, b in zip(cuts[:-1], cuts[1:]))
        A, E = ca.reshape(Af, mA, n + 1), ca.reshape(Ef, mE, n + 1)
        parts = []
        for k in range(K):
            o = 2 * nb * k
            parts.append(per.constraints(zs[k], y[:n], P[o:o + nb], P[o + nb:o + 2 * nb]))
        if mA:
            parts.append((ca.mtimes(A, y) - bA, -np.inf * np.ones(mA), np.zeros(mA)))
        if mE:
            parts.append((ca.mtimes(E, y) - bE, np.zeros(mE), np.zeros(mE)))
        g, lb, ub = (list(v) for v in zip(*parts))
        nlp = {"x": ca.vertcat(y, *zs), "f": -ca.dot(c, y), "g": ca.vertcat(*g), "p": P}
        solver = ca.nlpsol(f"acopf{K}_{mA}_{mE}", "ipopt", nlp, _IPOPT)
        self._solvers[key] = (solver, np.concatenate(lb), np.concatenate(ub))
        return self._solvers[key]

    def _solve(self, periods, c, extra: _Extra, start=None):
        n, per = self.n, self._per
        K = len(periods)
        A = np.zeros((0, n + 1)) if extra.A is None else np.asarray(extra.A, float)
        E = np.zeros((0, n + 1)) if extra.A_eq is None else np.asarray(extra.A_eq, float)
        bA = np.zeros(0) if extra.b is None else np.asarray(extra.b, float)
        bE = np.zeros(0) if extra.b_eq is None else np.asarray(extra.b_eq, float)
        solver, lbg, ubg = self._joint_solver(K, A.shape[0], E.shape[0])
        p = np.concatenate([np.concatenate([self.net.pd[t], self.net.qd[t]]) for t in periods]
                           + [np.asarray(c, float), A.T.ravel(), bA, E.T.ravel(), bE])
        ybounds = extra.bounds if extra.bounds is not None else [(0, None)] * n + [(None, None)]
        big = 10.0 * max(1.0, float(self.net.pmax.sum()))
        ylb = np.array([-big if lo is None else lo for lo, _ in ybounds], float)
        yub = np.array([big if hi is None else hi for _, hi in ybounds], float)
        zl, zu = per.bounds()
        lbx = np.concatenate([ylb] + [zl] * K)
        ubx = np.concatenate([yub] + [zu] * K)
        starts = []
        y0 = np.clip(np.zeros(n + 1), ylb, yub)
        starts.append(np.concatenate([y0] + [self._warm.get(t, per.start()) for t in periods]))
        if start is not None:
            starts.insert(0, start)
        starts.append(np.concatenate([y0] + [per.start()] * K))
        starts += [np.concatenate([y0] + [self._random_start() for _ in periods]) for _ in range(self.multistart)]
        best = None

        def attempt(x0):
            nonlocal best
            sol = solver(x0=np.clip(x0, lbx, ubx), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
            self.lp_count += 1
            if solver.stats()["return_status"] not in _OK:
                return False
            val = -float(sol["f"])
            if best is None or val > best[0] + 1e-9 * max(1.0, abs(val)):
                best = (val, np.array(sol["x"]).ravel())
            return True

        for x0 in starts:
            if attempt(x0) and self.multistart == 0:      # the first success; extra starts only on request
                break
        # Every query has a feasible point (no new load), so a failure is numerical: retry from random starts.
        for _ in range(_RESCUE if best is None else 0):
            if attempt(np.concatenate([y0] + [self._random_start() for _ in periods])):
                break
        if best is None:
            return None
        val, xs = best
        zs = xs[n + 1:].reshape(K, per.nv)
        for t, z in zip(periods, zs):
            self._warm[t] = z
        return val, xs[:n + 1], zs

    def _random_start(self) -> np.ndarray:
        """A random starting point for one period: small angles, voltages near 1, outputs within limits."""
        per, net, rng = self._per, self.net, self._rng
        lo_v, hi_v = np.maximum(net.vmin, 0.95), np.minimum(net.vmax, 1.05)
        q_lo, q_hi = np.maximum(net.qmin, -net.pmax.sum()), np.minimum(net.qmax, net.pmax.sum())
        return np.concatenate([rng.uniform(-0.2, 0.2, per.nb), rng.uniform(np.minimum(lo_v, hi_v), hi_v),
                               rng.uniform(net.pmin, net.pmax), rng.uniform(q_lo, q_hi)])

    def _feasible_lazy(self, t: int, x) -> bool:
        """Whether period ``t`` hosts new loads ``x``: the largest ``a <= 1`` with ``a x`` hosted is 1."""
        x = np.asarray(x, float)
        if self._feas is None:
            per, n, nb = self._per, self.n, self._per.nb
            a = ca.SX.sym("a")
            z = ca.SX.sym("z", per.nv)
            P = ca.SX.sym("P", 2 * nb + n)
            g, lb, ub = per.constraints(z, a * P[2 * nb:], P[:nb], P[nb:2 * nb])
            nlp = {"x": ca.vertcat(a, z), "f": -a, "g": g, "p": P}
            self._feas = (ca.nlpsol("acfeas", "ipopt", nlp, _IPOPT), lb, ub)
        solver, lbg, ubg = self._feas
        zl, zu = self._per.bounds()
        p = np.concatenate([self.net.pd[t], self.net.qd[t], x])
        # "hosted" is certified by an operating point; "not hosted" needs the warm and the flat start to agree
        starts = [z for z in (self._warm.get(t), self._per.start()) if z is not None]
        solved = 0
        for k, z0 in enumerate(starts + [self._random_start() for _ in range(_RESCUE)]):
            if k >= len(starts) and solved:
                break
            sol = solver(x0=np.concatenate([[1.0], np.clip(z0, zl, zu)]), lbx=np.concatenate([[0.0], zl]),
                         ubx=np.concatenate([[1.0], zu]), lbg=lbg, ubg=ubg, p=p)
            self.lp_count += 1
            if solver.stats()["return_status"] in _OK:
                solved += 1
                if float(np.array(sol["x"])[0, 0]) >= 1 - 1e-7:
                    self._warm[t] = np.array(sol["x"]).ravel()[1:]
                    return True
        if solved:
            return False
        raise RuntimeError(f"Ipopt failed on period {t} (the base case may be infeasible)")

    def _optimize(self, c, extra: _Extra):
        """``max c'y`` over ``y = (x, t)`` with ``extra`` and every period hosted (a local optimum)."""
        c = np.asarray(c, float)
        for _ in range(self.net.periods + 1):
            periods = sorted(self.binding)
            res = self._solve(periods, c, extra)
            if res is None:
                return None
            val, y, _ = res
            x = y[:self.n]
            order = np.argsort(-self.net.pd.sum(axis=1))
            new = [int(t) for t in order if int(t) not in self.binding and not self._feasible_lazy(int(t), x)]
            if not new:
                return val, x
            self.binding |= set(new[:3])      # add a few at a time: the worst periods usually repeat
        raise RuntimeError("lazy period generation did not converge")

    def water_fill(self, requests, weights=None, active=None, return_stages: bool = False):
        """Water-filling ``U(r)`` by NLPs, each step checked by feasibility tests.

        A hosted point is a certificate (Ipopt returns an operating point that
        meets every limit), while a maximum can be a local one. So the level is
        pushed while a point ``certify`` (relative) further along the ray is
        hosted, and an applicant counts as blocked only if growing it alone by
        ``certify`` is not hosted.
        """
        from .waterfill import Stage

        n = self.n
        r = np.asarray(requests, dtype=float).reshape(n)
        lam = np.ones(n) if weights is None else np.asarray(weights, dtype=float).reshape(n)
        rising = np.ones(n, bool) if active is None else np.asarray(active, bool).copy()
        rising &= r > 1e-9
        s = np.zeros(n)
        fixed = {i: 0.0 for i in range(n) if not rising[i]}
        stages = []
        scale = max(1.0, float(np.max(self.hosting_capacities())))
        eps = self.certify
        while rising.any():
            idx = np.where(rising)[0]
            cap = float(np.min(r[idx] / lam[idx]))
            A_eq = np.zeros((idx.size, n + 1))
            A_eq[np.arange(idx.size), idx] = 1.0
            A_eq[:, -1] = -lam[idx]

            def level_from(lo, idx=idx, cap=cap, A_eq=A_eq):
                bounds = self._bounds(fixed, cap)
                bounds[-1] = (lo, cap)
                res = self._optimize(np.concatenate([np.zeros(n), [1.0]]),
                                     _Extra(A_eq=A_eq, b_eq=np.zeros(idx.size), bounds=bounds))
                return None if res is None else res[0]

            level = level_from(0.0)
            if level is None:                         # Ipopt failed: bisect on hosted points instead
                level = self._bisect_level(s, idx, lam, cap)
                self.bisected_stages += 1
            for _ in range(50):                       # push while the next point on the ray is hosted
                nxt = min(cap, level + eps * max(level, eps * scale))
                trial = s.copy()
                trial[idx] = lam[idx] * nxt
                if nxt <= level or not self.contains(trial):
                    break
                level = max(nxt, level_from(nxt) or nxt)
            s[idx] = lam[idx] * level
            stop = rising & (s >= r - 1e-7 * scale)
            room = {}
            for i in idx:
                if stop[i]:
                    continue
                step = eps * max(s[i], eps * scale)
                trial = s.copy()
                trial[i] += step
                if self.contains(trial):
                    room[int(i)] = step
                else:
                    stop[i] = True
            if not stop.any():
                stop = self._unblocked_stage(s, rising, room, scale)
            if return_stages:
                stages.append(Stage(level, tuple(np.where(stop)[0].tolist()), ()))
            for i in np.where(stop)[0]:
                fixed[int(i)] = float(s[i])
            rising &= ~stop
        s = np.minimum(s, r)
        return (s, stages) if return_stages else s

    def _bisect_level(self, s, idx, lam, cap, rel: float = 1e-4) -> float:
        """Largest level on the ray whose point is hosted, by bisection on feasibility checks.

        Every hosted point is a certificate, so the returned level is hosted;
        the true level lies within ``rel`` (relative) above it unless the
        region is not star-shaped along the ray.
        """
        def point(t):
            x = s.copy()
            x[idx] = lam[idx] * t
            return x

        if self.contains(point(cap)):
            return cap
        lo, hi = 0.0, cap
        while hi - lo > rel * max(hi, 1e-9):
            mid = 0.5 * (lo + hi)
            if self.contains(point(mid)):
                lo = mid
            else:
                hi = mid
        return lo

    def _unblocked_stage(self, s, rising, room, scale):
        """Every rising applicant can still grow alone, though the common level cannot rise.

        On a nonconvex region this can be genuine (one load relieving another,
        so the region is not downward closed there). The applicant frozen is the
        first in order, and the stage is recorded in ``unblocked_stages``.
        """
        stop = np.zeros_like(rising)
        i = min(room)
        stop[i] = True
        self.unblocked_stages.append({"level_point": s.tolist(), "frozen": int(i),
                                      "room": {int(k): float(v) for k, v in room.items()}})
        return stop

    def contains(self, s, tol: float = 1e-6) -> bool:
        """Whether served loads ``s`` are hosted in every period, within ``tol``."""
        s = np.asarray(s, float) * (1 - tol)
        # binding periods first, then by peak load: an infeasible point usually fails within a few solves
        rest = [int(t) for t in np.argsort(-self.net.pd.sum(axis=1)) if int(t) not in self.binding]
        return all(self._feasible_lazy(t, s) for t in sorted(self.binding) + rest)

    def operating_point(self, s, period: int) -> dict:
        """An AC operating point hosting loads ``s`` in ``period``: voltages, angles, generation, flows."""
        s = np.asarray(s, float)
        bounds = [(float(v), float(v)) for v in s] + [(0.0, 0.0)]
        res = self._solve([period], np.zeros(self.n + 1), _Extra(bounds=bounds))
        if res is None:
            raise ValueError(f"loads {s} are not hosted in period {period}")
        va, vm, pg, qg = (np.asarray(v) for v in self._per.split(res[2][0]))
        sf, st = self.net.branch_flows(vm, va)
        return {"va": va, "vm": vm, "pg": pg, "qg": qg, "sf_mva": np.abs(sf) * self.net.base_mva,
                "st_mva": np.abs(st) * self.net.base_mva}


def downward_closure_check(region, samples: int = 20, seed: int = 0, cuts=(0.0, 0.5), tol: float = 1e-5) -> dict:
    """Test the theory's downward-closure assumption on the region's boundary.

    Each sample is a boundary point: water-filling with random weights. For
    each applicant and each factor in ``cuts``, its load is reduced to that
    factor and the point is tested. Returns the number of tests and the
    violations (points inside the region whose reduction is outside).
    """
    rng = np.random.default_rng(seed)
    n = region.n
    h = region.hosting_capacities()
    tests, bad = 0, []
    for _ in range(samples):
        w = rng.uniform(0.2, 1.0, n)
        s = region.water_fill(h * 2.0, weights=w)
        for i in range(n):
            for c in cuts:
                s2 = s.copy()
                s2[i] *= c
                tests += 1
                if not region.contains(s2 * (1 - tol)):
                    bad.append({"point": s.tolist(), "reduced": i, "factor": c})
    return {"tests": tests, "violations": bad}
