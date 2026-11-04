"""Allocation rules an operator can apply to a batch of requests.

Every rule maps requests ``r`` to served loads ``s`` with ``0 <= s <= r`` and
``s`` in the region. The minimum allocation ``beta`` drops an applicant served
less than ``beta * r_i`` and recomputes the allocation for the rest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .region import TOL, Region
from .waterfill import water_fill

CUT_ORDERS = ("largest_request", "largest_loading")


class AllocationRule(ABC):
    """Base class: subclasses implement ``_allocate_active``."""

    beta: float = 0.0
    cut_order: str = "largest_request"

    def allocate(self, region: Region, requests) -> np.ndarray:
        """Served loads for ``requests`` on ``region``, with the minimum allocation applied."""
        r = np.asarray(requests, dtype=float).reshape(region.n)
        if np.any(r < 0):
            raise ValueError("requests must be nonnegative")
        if self.cut_order not in CUT_ORDERS:
            raise ValueError(f"cut_order must be one of {CUT_ORDERS}")
        active = r > TOL
        while True:
            s = np.where(active, self._allocate_active(region, r, active), 0.0)
            short = active & (s < self.beta * r * (1.0 - 1e-9))
            if not short.any():
                return s
            active[self._next_cut(region, r, s, short)] = False

    def _next_cut(self, region: Region, r, s, short) -> int:
        """Index of the short applicant to drop; ties go to the lowest index."""
        if self.cut_order == "largest_loading" and getattr(region, "is_implicit", False):
            raise NotImplementedError("cut order 'largest_loading' needs explicit limits; "
                                      "export the region or use cut_order='largest_request'")
        if self.cut_order == "largest_request":
            key = np.where(short, r, -np.inf)
        else:  # largest loading of the most loaded limit
            load = region.A @ np.where(short | (s > 0), r, 0.0)
            c = int(np.argmax(load / np.where(region.b > TOL, region.b, TOL)))
            key = np.where(short, region.A[c] * r, -np.inf)
        best = key.max()
        return int(np.where(np.isclose(key, best, rtol=1e-12, atol=1e-12) & short)[0][0])

    @abstractmethod
    def _allocate_active(self, region: Region, r: np.ndarray, active: np.ndarray) -> np.ndarray: ...


@dataclass
class UniformRule(AllocationRule):
    """Water-filling with a common level (the paper's uniform rule).

    With ``weights``, applicant ``i`` rises at rate ``weights[i]`` (the weighted
    uniform rule); weights must be fixed before the requests are known.
    """

    beta: float = 0.0
    weights: tuple[float, ...] | None = None
    cut_order: str = "largest_request"

    def _allocate_active(self, region, r, active):
        return water_fill(region, r, weights=self.weights, active=active)


@dataclass
class ProportionalRule(AllocationRule):
    """Scale every active request by one common fraction until a limit binds.

    The minimum allocation drops the applicant that loads the binding limit
    most; dropping by MW instead can break the equilibrium (see the paper).
    """

    beta: float = 0.0
    cut_order: str = "largest_loading"

    def _allocate_active(self, region, r, active):
        ra = np.where(active, r, 0.0)
        if getattr(region, "is_implicit", False):
            return region.max_common_fraction(ra) * ra
        load = region.A @ ra
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(load > TOL, region.b / np.where(load > TOL, load, 1.0), np.inf)
        alpha = min(1.0, float(ratios.min()))
        return alpha * ra


@dataclass
class UtilitarianRule(AllocationRule):
    """Serve the largest total load, ``max 1's`` subject to ``s <= r``.

    Ties among maximizers are broken by maximizing the smallest service
    fraction. The total is what a planner maximizes, but applicants then gain
    by inflating requests at sites that load the limits least.
    """

    beta: float = 0.0
    cut_order: str = "largest_request"

    def _allocate_active(self, region, r, active):
        n = region.n
        ub = np.where(active, r, 0.0)
        if getattr(region, "is_implicit", False):
            return region.utilitarian(ub)
        bounds = [(0.0, float(u)) for u in ub]
        res = linprog(-np.ones(n), A_ub=region.A, b_ub=region.b, bounds=bounds, method="highs")
        if res.status != 0:
            raise RuntimeError(res.message)
        total = -res.fun
        # second stage: max z with s_i >= z r_i, 1's >= total
        idx = np.where(ub > TOL)[0]
        if idx.size == 0:
            return np.zeros(n)
        c = np.zeros(n + 1)
        c[-1] = -1.0
        A_ub = [np.concatenate([region.A, np.zeros((region.m, 1))], axis=1)]
        b_ub = [region.b]
        A_ub.append(np.concatenate([-np.ones((1, n)), np.zeros((1, 1))], axis=1))
        b_ub.append([-(total - 1e-9 * max(1.0, total))])
        rows = np.zeros((idx.size, n + 1))
        rows[np.arange(idx.size), idx] = -1.0
        rows[:, -1] = ub[idx]
        A_ub.append(rows)
        b_ub.append(np.zeros(idx.size))
        res2 = linprog(c, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub),
                       bounds=bounds + [(0.0, 1.0)], method="highs")
        return np.asarray(res2.x[:n]) if res2.status == 0 else np.asarray(res.x)
