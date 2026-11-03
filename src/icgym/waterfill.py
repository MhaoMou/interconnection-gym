"""Water-filling on a downward-closed region.

Serve each active applicant ``min(r_i, lambda_i * t)`` as a common level ``t``
rises; an applicant stops when it reaches its request or when a limit it loads
becomes tight. With unit weights this is the uniform rule; with unbounded
requests it returns the (weighted) max-min fair point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .region import TOL, Region


@dataclass(frozen=True)
class Stage:
    """One event of unbounded water-filling: the applicants that stop at a level."""

    level: float
    applicants: tuple[int, ...]
    limits: tuple[int, ...]


def water_fill(region: Region, requests, weights=None, active=None, return_stages: bool = False):
    """Allocation ``U(r)`` of water-filling without the minimum allocation.

    ``requests`` may contain ``np.inf``. ``active`` masks applicants that take
    part (the others are served zero). With ``return_stages`` the stopping
    events are returned too.
    """
    if getattr(region, "is_implicit", False):
        return region.water_fill(requests, weights=weights, active=active, return_stages=return_stages)
    A, b = region.A, region.b
    n = region.n
    r = np.asarray(requests, dtype=float).reshape(n)
    lam = np.ones(n) if weights is None else np.asarray(weights, dtype=float).reshape(n)
    if np.any(lam <= 0):
        raise ValueError("weights must be positive")
    if np.any(r < 0):
        raise ValueError("requests must be nonnegative")
    rising = np.ones(n, bool) if active is None else np.asarray(active, bool).copy()
    rising &= r > TOL
    s = np.zeros(n)
    stages: list[Stage] = []
    scale = max(1.0, float(np.max(b)) if b.size else 1.0)
    while rising.any():
        fixed = A[:, ~rising] @ s[~rising]
        rate = A[:, rising] @ lam[rising]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_limit = np.where(rate > TOL, (b - fixed) / np.where(rate > TOL, rate, 1.0), np.inf)
        t_limit = np.maximum(t_limit, 0.0)
        t_req = np.where(rising, r / lam, np.inf)
        t = min(float(t_limit.min()), float(t_req.min()))
        if not np.isfinite(t):
            raise RuntimeError("water-filling did not stop; the region is unbounded for the active applicants")
        s[rising] = lam[rising] * t
        stop = rising & (t_req <= t * (1 + 1e-12) + TOL)
        tight = np.where((rate > TOL) & (t_limit <= t * (1 + 1e-12) + TOL * scale))[0]
        for c in tight:
            stop |= rising & (A[c] > TOL)
        if not stop.any():  # numerical safety: stop the closest applicant
            stop[np.argmin(np.where(rising, t_req, np.inf))] = True
        if return_stages:
            stages.append(Stage(t, tuple(np.where(stop)[0].tolist()), tuple(int(c) for c in tight)))
        rising &= ~stop
    s = np.minimum(s, r)
    return (s, stages) if return_stages else s


def max_min_fair_point(region: Region, weights=None, return_stages: bool = False):
    """The (weighted) max-min fair point ``w = U(inf * 1)`` and its stages."""
    return water_fill(region, np.full(region.n, np.inf), weights=weights, return_stages=return_stages)
