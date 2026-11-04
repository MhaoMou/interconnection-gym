"""Batch regions from a DC power-flow study.

A period ``t`` can host the new loads ``x`` at a set of buses if some
redispatch of the existing generators serves the base load plus ``x`` within
branch and generator limits (lossless DCOPF in PTDF form). The batch region is
the intersection over periods of these per-period regions.

:func:`export_region` computes the region exactly, period by period: it grows
an inner approximation by the maximizers of support LPs until every facet of
its convex hull is confirmed by an LP (a projection by support functions). The
facets of all periods together describe the region; their positive parts
describe its largest downward-closed subset (counterflow between new loads
treated as non-firm), which is the :class:`~icgym.region.Region` returned.

Input: a scenario directory with ``bus.csv``, ``branch_limits.csv``,
``generator_limits.csv``, ``demand_bus.csv`` and ``dcline.csv`` in the format
of the ERCOT connection game packages.
"""

from __future__ import annotations

import csv
import hashlib
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from .region import Region

INPUT_FILES = ("bus.csv", "branch_limits.csv", "generator_limits.csv", "demand_bus.csv", "dcline.csv")


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass
class DCNetwork:
    buses: list[str]
    ptdf: np.ndarray            # branches x buses
    limit: np.ndarray           # branch thermal limits (MW)
    gen_bus: np.ndarray         # bus index of each generator
    gen_min: np.ndarray
    gen_max: np.ndarray
    demand: np.ndarray          # periods x buses (MW)
    branch_names: list[str]
    source_hash: str = ""
    source: str = ""            # folder name, safe to publish
    root: str = ""              # full path, used only to reload in worker processes
    gen_cost: np.ndarray | None = None   # marginal cost of each generator ($/MWh), for prices

    @classmethod
    def from_csv_dir(cls, directory: str | Path, horizon: int | None = None) -> DCNetwork:
        """Read a DC power-flow scenario folder (bus, branch, generator, demand, and DC-line CSV files)."""
        root = Path(directory)
        bus_rows = _rows(root / "bus.csv")
        buses = [r["bus_id"] for r in bus_rows]
        index = {b: i for i, b in enumerate(buses)}
        ref = [r["bus_id"] for r in bus_rows if r["type"] == "3"]
        if len(ref) != 1:
            raise ValueError("bus.csv must have exactly one reference bus (type 3)")
        branches = [r for r in _rows(root / "branch_limits.csv") if r["status"] == "1"]
        nb, nl = len(buses), len(branches)
        inc = np.zeros((nl, nb))
        sus = np.zeros(nl)
        for k, r in enumerate(branches):
            inc[k, index[r["from_bus_id"]]] = 1.0
            inc[k, index[r["to_bus_id"]]] = -1.0
            sus[k] = 1.0 / float(r["reactance_pu"])
        bbus = inc.T @ (sus[:, None] * inc)
        keep = [i for i in range(nb) if buses[i] != ref[0]]
        ptdf = np.zeros((nl, nb))
        ptdf[:, keep] = (sus[:, None] * inc[:, keep]) @ np.linalg.inv(bbus[np.ix_(keep, keep)])
        gens = [r for r in _rows(root / "generator_limits.csv") if r["status"] == "1"]
        if any(r.get("status") == "1" for r in _rows(root / "dcline.csv")):
            raise NotImplementedError("active DC lines are not supported yet")
        demand_rows = _rows(root / "demand_bus.csv")
        if horizon is not None:
            demand_rows = demand_rows[:horizon]
        demand = np.array([[float(r[b]) for b in buses] for r in demand_rows])
        digest = hashlib.sha256()
        for name in INPUT_FILES:
            digest.update(name.encode())
            digest.update((root / name).read_bytes())
        return cls(
            buses=buses,
            ptdf=ptdf,
            limit=np.array([float(r["thermal_limit_mva"]) for r in branches]),
            gen_bus=np.array([index[r["bus_id"]] for r in gens], dtype=int),
            gen_min=np.array([float(r["p_min_mw"]) for r in gens]),
            gen_max=np.array([float(r["p_max_mw"]) for r in gens]),
            demand=demand,
            branch_names=[f"{r['from_bus_id']}-{r['to_bus_id']}" for r in branches],
            source_hash=digest.hexdigest(),
            source=root.name,
            root=str(root.resolve()),
            gen_cost=np.array([float(r.get("marginal_cost_c1_per_mwh") or 0.0) for r in gens]),
        )

    @property
    def periods(self) -> int:
        """Number of periods."""
        return self.demand.shape[0]

    def period_problem(self, t: int, buses) -> PeriodProblem:
        """The redispatch problem of period ``t`` with new loads at ``buses``."""
        cols = [self.buses.index(str(b)) for b in buses]
        return PeriodProblem(
            flow_site=self.ptdf[:, cols],
            flow_gen=self.ptdf[:, self.gen_bus],
            base=self.ptdf @ self.demand[t],
            limit=self.limit,
            total_demand=float(self.demand[t].sum()),
            gen_bounds=list(zip(self.gen_min, self.gen_max)),
            n=len(cols),
        )


def _solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds):
    """HiGHS with fallbacks: numerical trouble ('unknown' status) is retried
    with a normalized objective and the dual simplex, then interior point."""
    res = None
    scale = max(1e-12, float(np.abs(c).max()))
    for method in ("highs", "highs-ds", "highs-ipm"):
        res = linprog(c / scale, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
        if res.status in (0, 2, 3):
            break
    if res.status == 0:
        res.fun = res.fun * scale
    return res


@dataclass
class PeriodProblem:
    """One period's DCOPF: ``|flow_gen p - flow_site x - base| <= limit``,
    ``sum p - sum x = total demand``, generator bounds, ``x >= 0``.

    Branch limits enter lazily: an LP is solved with the branches found binding
    so far, and violated branches are added until none remain, which gives the
    exact optimum with small LPs on large networks.
    """

    flow_site: np.ndarray
    flow_gen: np.ndarray
    base: np.ndarray
    limit: np.ndarray
    total_demand: float
    gen_bounds: list
    n: int
    active: set = field(default_factory=set)

    def support(self, direction, zero=None) -> tuple[float, np.ndarray]:
        """``max d'x`` over the period's region (``x >= 0``) and a maximizer.

        ``zero`` lists coordinates held at zero.
        """
        d = np.asarray(direction, dtype=float)
        G = len(self.gen_bounds)
        c = np.concatenate([-d, np.zeros(G)])
        zero = set() if zero is None else set(zero)
        bounds = [(0.0, 0.0) if i in zero else (0.0, None) for i in range(self.n)] + self.gen_bounds
        A_eq = np.concatenate([-np.ones(self.n), np.ones(G)])[None, :]
        tol = 1e-7 * max(1.0, float(self.limit.max()))
        for _ in range(len(self.limit) + 1):
            rows = np.array(sorted(self.active), dtype=int)
            if rows.size:
                A = np.hstack([-self.flow_site[rows], self.flow_gen[rows]])
                A_ub = np.vstack([A, -A])
                b_ub = np.concatenate([self.limit[rows] + self.base[rows], self.limit[rows] - self.base[rows]])
            else:
                A_ub, b_ub = None, None
            res = _solve_lp(c, A_ub, b_ub, A_eq, [self.total_demand], bounds)
            if res.status == 2:
                raise ValueError("the base case is infeasible in this period")
            if res.status == 3:
                raise ValueError("unbounded: a bus outside every binding limit (add more periods or branches)")
            if res.status != 0:
                raise RuntimeError(res.message)
            x, p = res.x[: self.n], res.x[self.n:]
            flow = self.flow_gen @ p - self.flow_site @ x - self.base
            violated = set(np.where(np.abs(flow) > self.limit + tol)[0].tolist()) - self.active
            if not violated:
                return float(-res.fun), np.asarray(x)
            self.active |= violated
        raise RuntimeError("lazy branch generation did not converge")

    def feasible(self, x) -> bool:
        """Whether loads ``x`` can be hosted, with every branch limit (no lazy rows)."""
        x = np.asarray(x, dtype=float)
        G = len(self.gen_bounds)
        rhs = self.base + self.flow_site @ x
        res = linprog(np.zeros(G), A_ub=np.vstack([self.flow_gen, -self.flow_gen]),
                      b_ub=np.concatenate([self.limit + rhs, self.limit - rhs]),
                      A_eq=np.ones((1, G)), b_eq=[self.total_demand + x.sum()],
                      bounds=self.gen_bounds, method="highs")
        return res.status == 0


def period_facets(problem: PeriodProblem, tol: float = 1e-7, max_iter: int = 200, gap: float | None = None):
    """Facets ``(G, h)`` with ``G x <= h`` of one period's region.

    With ``gap=None`` the facets are exact (to ``tol``). With a relative
    ``gap``, enumeration stops once the support of the true region exceeds
    every facet of the current hull by at most ``gap`` times the period's
    largest hosting capacity; the result is the hull of feasible points, so it
    lies inside the true region. Coordinates with zero hosting capacity are
    fixed at zero by a row ``x_i <= 0``. Returns the facets other than ``x >= 0``.
    """
    accept = tol if gap is None else max(gap, tol)
    n = problem.n
    caps = np.array([problem.support(np.eye(n)[i])[0] for i in range(n)])
    scale = max(1.0, float(caps.max()))
    live = np.where(caps > tol * scale)[0]
    rows = [np.eye(n)[i] for i in range(n) if i not in live]
    rhs = [0.0] * len(rows)
    k = live.size
    if k == 0:
        return np.array(rows).reshape(-1, n), np.array(rhs)

    def lift(g):
        full = np.zeros(n)
        full[live] = g
        return full

    if k == 1:
        rows.append(lift(np.ones(1)))
        rhs.append(float(caps[live[0]]))
        return np.array(rows), np.array(rhs)

    # seed with feasible points only: the origin, each single-bus capacity
    # (other loads at zero), and the maximizers of the coordinate directions
    points = [np.zeros(k)]
    for i in live:
        others = [j for j in range(n) if j != i]
        points.append(problem.support(np.eye(n)[i], zero=others)[1][live])
        points.append(problem.support(np.eye(n)[i])[1][live])
    points.append(problem.support(lift(np.ones(k)))[1][live])
    confirmed: dict[tuple, tuple[np.ndarray, float]] = {}
    for _ in range(max_iter):
        hull = ConvexHull(np.array(points), qhull_options="Qt")
        added = False
        seen = set()
        for eq in hull.equations:
            g, off = eq[:-1], -eq[-1]
            norm = np.linalg.norm(g)
            g, off = g / norm, off / norm
            key = tuple(np.round(np.concatenate([g, [off / scale]]), 7))
            if key in seen or key in confirmed:
                continue
            seen.add(key)
            if np.all(g <= tol) and abs(off) <= tol * scale:   # x_i >= 0 facets
                confirmed[key] = (g, off)
                continue
            val, x = problem.support(lift(g))
            if val > off + accept * scale:
                points.append(x[live])
                added = True
            else:
                # exact: the support itself; approximate: the hull (inner) offset
                confirmed[key] = (g, max(val, off) if gap is None else off)
        if not added:
            break
    else:
        raise RuntimeError("facet enumeration did not converge")
    for g, off in confirmed.values():
        if np.all(g <= tol) and abs(off) <= tol * scale:
            continue
        rows.append(lift(g))
        rhs.append(off)
    return np.array(rows), np.array(rhs)


_NET: DCNetwork | None = None


def _init_worker(root: str, horizon: int) -> None:
    global _NET
    _NET = DCNetwork.from_csv_dir(root, horizon=horizon)


def _period_worker(args):
    t, buses, tol, gap = args
    return period_facets(_NET.period_problem(t, buses), tol=tol, gap=gap)


def _prune(A: np.ndarray, b: np.ndarray, tol: float = 1e-7):
    """Drop duplicate and redundant rows of ``{x >= 0 : A x <= b}`` (``A >= 0``)."""
    scale = np.where(b > tol, b, 1.0)
    An, bn = A / scale[:, None], b / scale
    _, idx = np.unique(np.round(np.hstack([An, bn[:, None]]), 9), axis=0, return_index=True)
    A, b = A[np.sort(idx)], b[np.sort(idx)]
    keep = np.ones(len(b), bool)
    for c in range(len(b)):
        others = keep.copy()
        others[c] = False
        if not others.any():
            continue
        res = linprog(-A[c], A_ub=A[others], b_ub=b[others], bounds=[(0, None)] * A.shape[1], method="highs")
        if res.status == 0 and -res.fun <= b[c] * (1 + 1e-9) + tol:
            keep[c] = False
    return A[keep], b[keep]


@dataclass
class ExportedRegion:
    region: Region               # largest downward-closed subset (A >= 0)
    signed_A: np.ndarray         # facets of the exact region, negative entries = counterflow
    signed_b: np.ndarray
    metadata: dict

    @property
    def has_counterflow(self) -> bool:
        """Whether any exact facet has a negative coefficient (relief between new loads)."""
        return bool(np.any(self.signed_A < -1e-9))


def export_region(net: DCNetwork, buses, periods=None, workers: int = 1, tol: float = 1e-7,
                  names=None, gap: float | None = None) -> ExportedRegion:
    """Batch region of new loads at ``buses`` over ``periods`` (default all).

    ``gap`` trades exactness for speed in many dimensions (see :func:`period_facets`).
    """
    buses = [str(b) for b in buses]
    periods = list(range(net.periods)) if periods is None else list(periods)
    jobs = [(t, buses, tol, gap) for t in periods]
    if workers > 1:
        # each worker loads the network once instead of receiving it per job
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker,
                                          initargs=(net.root, net.periods)) as pool:
            parts = pool.map(_period_worker, jobs)
    else:
        parts = [period_facets(net.period_problem(t, buses), tol=tol, gap=gap) for t in periods]
    ex = assemble_region(net, buses, periods, parts, tol=tol, names=names)
    ex.metadata["gap"] = gap
    return ex


def assemble_region(net: DCNetwork, buses, periods, parts, tol: float = 1e-7, names=None) -> ExportedRegion:
    """Combine per-period facets ``parts`` (from :func:`period_facets`) into a region."""
    buses = [str(b) for b in buses]
    periods = list(periods)
    G = np.vstack([p[0] for p in parts])
    h = np.concatenate([p[1] for p in parts])
    signed_A, signed_b = _prune_signed(G, h, tol)
    A_plus, b_plus = _prune(np.clip(signed_A, 0.0, None), signed_b, tol)
    names = tuple(names) if names else tuple(f"bus_{b}" for b in buses)
    region = Region(A_plus, b_plus, applicant_names=names)
    meta = {
        "buses": buses,
        "periods": periods if len(periods) < net.periods else "all",
        "n_periods": len(periods),
        "source": net.source,
        "source_sha256": net.source_hash,
        "method": "per-period support-function projection; positive parts (non-firm counterflow)",
        "tolerance": tol,
    }
    return ExportedRegion(region, signed_A, signed_b, meta)


def _prune_signed(G: np.ndarray, h: np.ndarray, tol: float):
    """Drop duplicate and redundant rows of ``{x >= 0 : G x <= h}``."""
    scale = np.where(h > tol, h, 1.0)
    key = np.round(np.hstack([G / scale[:, None], (h / scale)[:, None]]), 9)
    _, idx = np.unique(key, axis=0, return_index=True)
    G, h = G[np.sort(idx)], h[np.sort(idx)]
    keep = np.ones(len(h), bool)
    for c in range(len(h)):
        others = keep.copy()
        others[c] = False
        if not others.any():
            continue
        res = linprog(-G[c], A_ub=G[others], b_ub=h[others], bounds=[(0, None)] * G.shape[1], method="highs")
        if res.status == 0 and -res.fun <= h[c] + tol * max(1.0, abs(h[c])):
            keep[c] = False
    return G[keep], h[keep]
