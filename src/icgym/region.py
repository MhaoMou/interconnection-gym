"""Batch regions: the served loads a network can host.

A region is the polytope ``{s >= 0 : A s <= b}`` over the applicants' served
loads. Allocation rules and the equilibrium results assume it is downward
closed, which holds exactly when ``A >= 0``. A general polytope is reduced to
its largest downward-closed subset by keeping the positive part of ``A``,
which treats counterflow between new loads as non-firm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

TOL = 1e-9


@dataclass(frozen=True)
class Region:
    """Downward-closed batch region ``{s >= 0 : A s <= b}`` with ``A >= 0``.

    Rows are shared limits (branches, spare generation, import limits);
    columns are applicants.
    """

    A: np.ndarray
    b: np.ndarray
    limit_names: tuple[str, ...] = field(default=())
    applicant_names: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        A = np.atleast_2d(np.asarray(self.A, dtype=float))
        b = np.asarray(self.b, dtype=float).reshape(-1)
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"A has {A.shape[0]} rows but b has {b.shape[0]} entries")
        if np.any(A < -TOL):
            raise ValueError("A has negative entries; use Region.downward_closed() for a general polytope")
        if np.any(b < -TOL):
            raise ValueError("b must be nonnegative (the origin must be feasible)")
        unloaded = np.where(A.max(axis=0) <= TOL)[0]
        if unloaded.size:
            raise ValueError(f"applicants {unloaded.tolist()} load no limit, so the region is unbounded")
        object.__setattr__(self, "A", np.clip(A, 0.0, None))
        object.__setattr__(self, "b", np.clip(b, 0.0, None))
        m, n = A.shape
        if not self.limit_names:
            object.__setattr__(self, "limit_names", tuple(f"limit_{c}" for c in range(m)))
        if not self.applicant_names:
            object.__setattr__(self, "applicant_names", tuple(f"applicant_{i}" for i in range(n)))
        if len(self.limit_names) != m or len(self.applicant_names) != n:
            raise ValueError("name tuples must match the shape of A")

    @classmethod
    def downward_closed(cls, A, b, **names) -> Region:
        """Largest downward-closed subset of ``{x >= 0 : A x <= b}`` (Lemma 4.1).

        Negative entries of ``A`` are relief one new load gives another
        (counterflow); dropping them keeps every single-bus hosting capacity.
        """
        A = np.atleast_2d(np.asarray(A, dtype=float))
        return cls(np.clip(A, 0.0, None), b, **names)

    @classmethod
    def single_limit(cls, loadings, capacity: float, **names) -> Region:
        """One shared limit ``sum_i a_i s_i <= capacity``."""
        return cls(np.asarray(loadings, dtype=float)[None, :], [capacity], **names)

    @property
    def n(self) -> int:
        """Number of applicants (columns of ``A``)."""
        return self.A.shape[1]

    @property
    def m(self) -> int:
        """Number of shared limits (rows of ``A``)."""
        return self.A.shape[0]

    def contains(self, s, tol: float = 1e-7) -> bool:
        """Whether served loads ``s`` lie in the region, within ``tol``."""
        s = np.asarray(s, dtype=float)
        return bool(np.all(s >= -tol) and np.all(self.A @ s <= self.b + tol))

    def hosting_capacities(self) -> np.ndarray:
        """Single-bus hosting capacity ``h_i = max_{s in region} s_i``."""
        with np.errstate(divide="ignore"):
            ratios = np.where(self.A > TOL, self.b[:, None] / np.where(self.A > TOL, self.A, 1.0), np.inf)
        return ratios.min(axis=0)

    def restrict(self, applicants) -> Region:
        """Region seen by a subset of applicants (the others served zero)."""
        idx = list(applicants)
        A = self.A[:, idx]
        keep = A.max(axis=1) > TOL
        return Region(
            A[keep],
            self.b[keep],
            limit_names=tuple(np.array(self.limit_names)[keep]),
            applicant_names=tuple(np.array(self.applicant_names)[idx]),
        )

    def max_total(self) -> tuple[float, np.ndarray]:
        """Planner's optimum: the largest total served load and a maximizer."""
        res = linprog(-np.ones(self.n), A_ub=self.A, b_ub=self.b, bounds=[(0, None)] * self.n, method="highs")
        if res.status != 0:
            raise RuntimeError(f"planner LP failed: {res.message}")
        return float(-res.fun), np.asarray(res.x)
