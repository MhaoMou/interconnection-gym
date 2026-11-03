"""On-site generation, storage, and non-firm service (paper: on-site generation and non-firm service).

The limits bind only in peak periods (``nu`` per day, ``dt`` hours each). A
generator of capacity ``G_i <= gamma r_i`` with a daily run-hour budget ``h``
shaves ``theta G_i`` from every peak period, ``theta = min(1, h / (nu dt))``;
a battery of power ``P_i`` and duration ``eta`` does the same with
``theta_S = min(1, eta / (nu dt))``. Who keeps the capacity a device frees is
the operator's choice:

* ``rule='gross'``: one common level on served load; the net loads
  ``(s - theta G)^+`` must fit the peak region, so the freed room is shared.
* ``rule='credit'``: the uniform rule on net requests ``r - theta G``, plus
  ``theta G_i`` credited to its owner.

Curtailable (non-firm) load is a device with ``theta = 1``, no capital cost,
and a fuel cost equal to the peak value forgone. :class:`NonFirmGame` has the
two-tier rule (firm requests on the peak region, non-firm requests on the
off-peak room left) and the gross rule (a curtailable fraction ``phi_i`` of a
common level).

Every game here has two-part strategies ``X[i] = (request, second)``;
:meth:`TwoPartGame.best_response` searches both parts numerically and
:meth:`predicted_equilibrium` returns the theory's answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .region import TOL, Region
from .rules import UniformRule
from .waterfill import max_min_fair_point, water_fill


def peak_theta(hours: float, nu: int, dt: float = 1.0) -> float:
    """Share of capacity a device can shave from each of ``nu`` peak periods: ``min(1, hours / (nu dt))``."""
    return float(min(1.0, hours / (nu * dt)))


# --------------------------------------------------------------------- water-filling with offsets
def _first_tight(a, fixed, b, r, o, t0):
    """Largest ``t >= t0`` with ``fixed + sum_i a_i (min(t, r_i) - o_i)^+ <= b`` (``inf`` if never exceeded)."""
    load = lambda t: fixed + float(a @ np.maximum(np.minimum(t, r) - o, 0.0))  # noqa: E731
    pts = np.unique(np.concatenate([[t0], o[(o > t0) & np.isfinite(o)], r[(r > t0) & np.isfinite(r)]]))
    pts = np.append(pts, np.inf)
    for p, q in zip(pts[:-1], pts[1:]):
        mid = p + 1.0 if not np.isfinite(q) else 0.5 * (p + q)
        slope = float(a @ ((o < mid) & (r > mid)))
        if slope <= 0:
            continue
        Lp = load(p)
        if not np.isfinite(q) or Lp + slope * (q - p) > b:
            return p + max(b - Lp, 0.0) / slope
    return np.inf


def water_fill_offset(region: Region, requests, offset, active=None) -> np.ndarray:
    """Water-filling on ``{s >= 0 : A (s - offset)^+ <= b}``, the served loads whose net loads fit.

    Served loads rise at a common level, each capped by its request, and an
    applicant stops when a limit it would load further is tight. With a zero
    offset this is :func:`icgym.water_fill`. Exact (breakpoint search per stage).
    """
    A, b = region.A, region.b
    r = np.asarray(requests, dtype=float).reshape(region.n)
    o = np.maximum(np.asarray(offset, dtype=float).reshape(region.n), 0.0)
    act = r > TOL if active is None else (np.asarray(active, bool) & (r > TOL))
    s = np.zeros(region.n)
    rising = act.copy()
    t0 = 0.0
    while rising.any():
        fixed = A[:, ~rising] @ np.maximum(s[~rising] - o[~rising], 0.0)
        idx = np.where(rising)[0]
        T = np.full(region.m, np.inf)
        for c in range(region.m):
            if np.any(A[c, idx] > TOL):
                T[c] = _first_tight(A[c, idx], fixed[c], b[c], r[idx], o[idx], t0)
        tstar = min(float(T.min()), float(r[idx].max()))
        s[idx] = np.minimum(tstar, r[idx])
        eps = 1e-12 * max(1.0, abs(tstar))
        done = rising & (r <= tstar + eps)
        for c in np.where(T <= tstar + eps)[0]:
            done |= rising & (A[c] > TOL) & (o <= tstar + eps)
        if not done.any():  # pragma: no cover - guards against an endless loop
            raise RuntimeError("water-filling with offsets made no progress")
        rising &= ~done
        t0 = tstar
    return s


def _cut_loop(alloc, r, beta):
    """Minimum allocation: drop the largest short request and recompute (as :class:`AllocationRule`)."""
    active = r > TOL
    while True:
        s = np.where(active, alloc(active), 0.0)
        short = active & (s < beta * r * (1.0 - 1e-9))
        if not short.any():
            return s
        key = np.where(short, r, -np.inf)
        active[int(np.argmax(key))] = False


# --------------------------------------------------------------------- shared game machinery
class TwoPartGame:
    """Games whose strategies are ``X[i] = (request, second part)``."""

    n: int

    def payoffs(self, X) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _upper(self, i: int, X) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def _second_options(self, i: int, x1: float, X, grid: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def payoff(self, i: int, x, X) -> float:
        Y = np.array(X, dtype=float)
        Y[i] = x
        return float(self.payoffs(Y)[i])

    def best_response(self, i: int, X, grid: int = 120, second_grid: int = 11,
                      fix_second: bool = False) -> tuple[np.ndarray, float]:
        """Best ``(request, second)`` of applicant ``i`` against ``X``, by grid search and refinement.

        With ``fix_second=True`` only the request moves (the second part stays
        at ``X[i, 1]``).
        """
        X = np.asarray(X, dtype=float)
        upper = self._upper(i, X)
        cands = []
        for x1 in np.append(np.linspace(0.0, upper, grid), X[i, 0]):
            opts = [X[i, 1]] if fix_second else self._second_options(i, x1, X, second_grid)
            cands += [(x1, x2) for x2 in opts]
        cands.append(tuple(X[i]))
        if not fix_second:
            cands += [tuple(c) for c in self._extra_candidates(i, X)]
        vals = np.array([self.payoff(i, c, X) for c in cands])
        top = np.argsort(vals)[::-1][:4]
        best, best_v = np.array(cands[top[0]]), float(vals[top[0]])
        step = upper / (grid - 1)
        for k in top:
            x1, x2 = cands[k]
            frac = x2 if fix_second else self._second_fraction(i, x1, x2, X)
            lo2, hi2 = self._fraction_bounds(i, x1, X, second_grid)
            step2 = (hi2 - lo2) / max(second_grid - 1, 1)
            for _ in range(1 if fix_second else 3):   # alternate: request, then second part
                def f(z, frac=frac):
                    sec = frac if fix_second else self._second_from_fraction(i, z, frac, X)
                    return -self.payoff(i, (z, sec), X)
                res = minimize_scalar(f, bounds=(max(0.0, x1 - step), min(upper, x1 + step)), method="bounded",
                                      options={"xatol": 1e-10 * max(1.0, upper)})
                x1 = float(res.x)
                if not fix_second and step2 > 0:
                    def g(q, x1=x1):
                        return -self.payoff(i, (x1, self._second_from_fraction(i, x1, q, X)), X)
                    res2 = minimize_scalar(g, bounds=(max(lo2, frac - step2), min(hi2, frac + step2)),
                                           method="bounded", options={"xatol": 1e-10 * max(1.0, abs(hi2))})
                    if -res2.fun >= -g(frac):
                        frac = float(res2.x)
                sec = frac if fix_second else self._second_from_fraction(i, x1, frac, X)
                v = self.payoff(i, (x1, sec), X)
                if v > best_v + 1e-12:
                    best, best_v = np.array([x1, sec]), v
        return best, best_v

    def _fraction_bounds(self, i, x1, X, grid):
        """Range of the refined second variable (the second part itself by default)."""
        opts = self._second_options(i, x1, X, grid)
        return float(np.min(opts)), float(np.max(opts))

    def _extra_candidates(self, i: int, X) -> list:
        """Strategies worth trying besides the grid (e.g. a closed-form best response)."""
        return []

    # by default the second part is absolute and independent of the request
    def _second_fraction(self, i, x1, x2, X):
        return x2

    def _second_from_fraction(self, i, x1, frac, X):
        return frac

    def hosting(self, i: int) -> float:
        """Scale of applicant ``i``'s request (its hosting capacity, with its own device at the cap)."""
        return self._upper(i, np.zeros((self.n, 2))) / 2.0

    def strategy_from_action(self, i: int, action, request_cap: float = 2.0) -> np.ndarray:
        """Map an action in ``[0, 1]^2`` to ``(request, second part)``.

        The request is ``action[0] * request_cap * hosting(i)``; the second
        part is ``action[1]`` times its own cap (``gamma r_i`` for a device,
        ``gamma`` for a curtailable fraction, ``request_cap`` times the
        off-peak hosting capacity for a non-firm request).
        """
        a = np.clip(np.asarray(action, dtype=float).reshape(2), 0.0, 1.0)
        x1 = a[0] * request_cap * self.hosting(i)
        return np.array([x1, self._second_from_action(i, x1, a[1], request_cap)])

    def _second_from_action(self, i, x1, a1, request_cap):  # pragma: no cover - abstract
        raise NotImplementedError

    def exploitability(self, X, fix_second: bool = False, **kw) -> np.ndarray:
        """Each applicant's gain from its best unilateral deviation (0 at an equilibrium)."""
        base = self.payoffs(X)
        return np.array([max(0.0, self.best_response(i, X, fix_second=fix_second, **kw)[1] - base[i])
                         for i in range(self.n)])


# --------------------------------------------------------------------- on-site generation and storage
@dataclass(frozen=True)
class Device:
    """A device that shaves ``theta`` of its capacity from every peak period.

    ``gamma`` caps the capacity at ``gamma * request``; ``cost`` is the cost per
    MW of capacity (capital plus peak-period fuel).
    """

    theta: float
    gamma: float
    cost: float
    name: str = "generator"

    @classmethod
    def generator(cls, run_hours: float, nu: int, gamma: float, capital: float, fuel_cost: float = 0.0,
                  peak_weight: float = 0.0, dt: float = 1.0) -> Device:
        """Generator with a daily run-hour budget; fuel ``F = theta * fuel_cost * peak_weight`` per MW."""
        th = peak_theta(run_hours, nu, dt)
        return cls(th, gamma, capital + th * fuel_cost * peak_weight, "generator")

    @classmethod
    def storage(cls, duration: float, nu: int, gamma: float, power_cost: float, energy_cost: float,
                dt: float = 1.0) -> Device:
        """Battery of the given duration charged off-peak (paper: storage remark); cost ``kappa^P + eta kappa^B``."""
        return cls(peak_theta(duration, nu, dt), gamma, power_cost + duration * energy_cost, "storage")


class GenerationGame(TwoPartGame):
    """Requests and on-site devices on a peak-period region; ``X[i] = (r_i, G_i)``.

    Payoff ``V s_i - kappa_C r_i - cost G_i``; ``G_i`` is clipped to
    ``gamma r_i`` unless ``enforce_cap=False`` (the theorem's part 1 sets the
    cap aside).
    """

    def __init__(self, region: Region, device: Device, value: float = 1.0, kappa: float = 0.4,
                 rule: str = "gross", beta: float = 0.0, enforce_cap: bool = True):
        if rule not in ("gross", "credit"):
            raise ValueError("rule must be 'gross' or 'credit'")
        if getattr(region, "is_implicit", False):
            raise NotImplementedError("devices need an explicit region; export it first")
        self.region, self.device, self.value, self.kappa = region, device, value, kappa
        self.rule, self.beta = rule, beta
        self.enforce_cap = enforce_cap

    @property
    def n(self) -> int:
        return self.region.n

    def _clip(self, X):
        X = np.array(X, dtype=float).reshape(self.n, 2)
        X[:, 0] = np.maximum(X[:, 0], 0.0)
        X[:, 1] = np.maximum(X[:, 1], 0.0)
        if self.enforce_cap:
            X[:, 1] = np.minimum(X[:, 1], self.device.gamma * X[:, 0])
        return X

    def allocate(self, X) -> np.ndarray:
        X = self._clip(X)
        r, shave = X[:, 0], self.device.theta * X[:, 1]
        if self.rule == "credit":
            net = np.maximum(r - shave, 0.0)
            return UniformRule(beta=self.beta).allocate(self.region, net) + np.where(r > TOL, shave, 0.0)
        return _cut_loop(lambda act: water_fill_offset(self.region, r, shave, act), r, self.beta)

    def payoffs(self, X) -> np.ndarray:
        X = self._clip(X)
        s = self.allocate(X)
        return self.value * s - self.kappa * X[:, 0] - self.device.cost * X[:, 1]

    def shares_given_devices(self, G) -> np.ndarray:
        """``w(G)``: the unique equilibrium of the requests with devices fixed (cap set aside)."""
        G = np.asarray(G, dtype=float)
        if self.rule == "credit":
            return max_min_fair_point(self.region) + self.device.theta * G
        return water_fill_offset(self.region, np.full(self.n, np.inf), self.device.theta * G)

    # margins of the theorem
    def builder_margin(self) -> float:
        """``mu+ = theta (V - kappa_C) - cost``."""
        return self.device.theta * (self.value - self.kappa) - self.device.cost

    def scrapper_margins(self) -> np.ndarray:
        """``mu-_i = theta (V - kappa_C) a_i / sum_j a_j - cost`` on a single limit."""
        if self.region.m != 1:
            raise ValueError("scrapper margins are defined for a single limit")
        a = self.region.A[0]
        return self.device.theta * (self.value - self.kappa) * a / a.sum() - self.device.cost

    def no_carry(self) -> bool:
        d = self.device
        return d.theta * d.gamma < 1 and d.theta * self.value < self.kappa / d.gamma + d.cost

    def predicted_equilibrium(self) -> dict:
        d, mu = self.device, self.builder_margin()
        if not self.no_carry():
            return {"known": False, "reason": "no-carry condition fails"}
        if self.rule == "credit":
            w = max_min_fair_point(self.region)
            if mu > 0:
                r = w / (1 - d.theta * d.gamma)
                return {"known": True, "kind": "unique", "X": np.column_stack([r, d.gamma * r]), "served": r}
            if mu < 0:
                return {"known": True, "kind": "unique", "X": np.column_stack([w, np.zeros(self.n)]), "served": w}
            return {"known": False, "reason": "zero margin"}
        if self.region.m != 1:
            return {"known": False, "reason": "gross rule: regimes are known for a single limit"}
        a, b = self.region.A[0], self.region.b[0]
        if mu < 0:
            w = np.full(self.n, b / a.sum())
            return {"known": True, "kind": "unique", "X": np.column_stack([w, np.zeros(self.n)]), "served": w}
        if np.all(self.scrapper_margins() > 0):
            w = np.full(self.n, b / ((1 - d.theta * d.gamma) * a.sum()))
            return {"known": True, "kind": "unique", "X": np.column_stack([w, d.gamma * w]), "served": w}
        if mu > 0:
            return {"known": True, "kind": "none", "reason": "mu+ > 0 > mu-_i for some i: no pure equilibrium"}
        return {"known": False, "reason": "zero margin"}

    def _upper(self, i, X):
        h = self.region.hosting_capacities()[i] / max(1e-9, 1 - self.device.theta * self.device.gamma)
        return float(2.0 * max(h, X[i, 0]))

    def _second_options(self, i, x1, X, grid):
        return np.linspace(0.0, self.device.gamma * x1, grid)

    def _second_fraction(self, i, x1, x2, X):
        cap = self.device.gamma * x1
        return x2 / cap if cap > 0 else 0.0

    def _second_from_fraction(self, i, x1, frac, X):
        return frac * self.device.gamma * x1

    def _second_from_action(self, i, x1, a1, request_cap):
        return a1 * self.device.gamma * x1

    def _fraction_bounds(self, i, x1, X, grid):
        return 0.0, 1.0


# --------------------------------------------------------------------- non-firm service
class NonFirmGame(TwoPartGame):
    """Firm and non-firm connections (paper: firm and non-firm connections).

    ``peak`` and ``offpeak`` are the batch regions of the peak and the other
    periods (``offpeak=None``: unconstrained off peak). ``peak_value`` is
    ``V^p``, the value of the load in the peak periods; ``V^o = V - V^p``.

    * ``rule='two_tier'``: ``X[i] = (firm request, non-firm request)``.
    * ``rule='gross'``: ``X[i] = (request, curtailable fraction phi_i <= gamma)``.
    """

    def __init__(self, peak: Region, offpeak: Region | None = None, value: float = 1.0, peak_value: float = 0.2,
                 kappa: float = 0.4, rule: str = "two_tier", gamma: float = 0.5):
        if rule not in ("two_tier", "gross"):
            raise ValueError("rule must be 'two_tier' or 'gross'")
        if rule == "gross" and offpeak is None and gamma >= 1:
            raise ValueError("with an unconstrained off-peak, gamma must be below 1")
        self.peak, self.offpeak, self.value, self.peak_value = peak, offpeak, value, peak_value
        self.kappa, self.rule, self.gamma = kappa, rule, gamma

    @property
    def n(self) -> int:
        return self.peak.n

    @property
    def offpeak_value(self) -> float:
        return self.value - self.peak_value

    def _clip(self, X):
        X = np.maximum(np.array(X, dtype=float).reshape(self.n, 2), 0.0)
        if self.rule == "gross":
            X[:, 1] = np.minimum(X[:, 1], self.gamma)
        return X

    def _residual(self, sf) -> Region | None:
        if self.offpeak is None:
            return None
        b = np.maximum(self.offpeak.b - self.offpeak.A @ sf, 0.0)
        return Region(self.offpeak.A, b, limit_names=self.offpeak.limit_names)

    def gross_region(self, phi) -> Region:
        """Served loads whose peak net loads ``(1 - phi) s`` and off-peak loads ``s`` fit."""
        A = self.peak.A * (1.0 - np.asarray(phi, dtype=float))[None, :]
        b = self.peak.b
        if self.offpeak is not None:
            A, b = np.vstack([A, self.offpeak.A]), np.concatenate([b, self.offpeak.b])
        return Region(A, b)

    def allocate(self, X):
        """Served loads: ``(firm, non-firm)`` per applicant for the two-tier rule, ``s`` for the gross rule."""
        X = self._clip(X)
        if self.rule == "two_tier":
            sf = water_fill(self.peak, X[:, 0])
            R = self._residual(sf)
            sn = X[:, 1].copy() if R is None else water_fill(R, X[:, 1])
            return np.column_stack([sf, sn])
        return water_fill(self.gross_region(X[:, 1]), X[:, 0])

    def payoffs(self, X) -> np.ndarray:
        X = self._clip(X)
        s = self.allocate(X)
        if self.rule == "two_tier":
            return self.value * s[:, 0] + self.offpeak_value * s[:, 1] - self.kappa * X.sum(axis=1)
        return self.value * s - self.peak_value * X[:, 1] * s - self.kappa * X[:, 0]

    def predicted_equilibrium(self) -> dict:
        V, Vp, Vo, k = self.value, self.peak_value, self.offpeak_value, self.kappa
        if self.rule == "two_tier":
            wp = max_min_fair_point(self.peak)
            R = self._residual(wp)
            if Vo > k:
                if R is None:
                    return {"known": False, "reason": "unconstrained off-peak: non-firm requests unbounded"}
                wn = max_min_fair_point(R)
            else:
                wn = np.zeros(self.n)
            return {"known": True, "kind": "tiers", "X": np.column_stack([wp, wn])}
        if self.peak.m != 1 or (self.offpeak is not None and self.offpeak.m != 1):
            return {"known": False, "reason": "gross rule: regimes are known for single limits"}
        a, bp = self.peak.A[0], self.peak.b[0]
        if self.offpeak is not None and not np.allclose(self.offpeak.A[0] / self.offpeak.A[0].sum(), a / a.sum()):
            return {"known": False, "reason": "peak and off-peak limits must have the same loadings"}
        bo = np.inf if self.offpeak is None else self.offpeak.b[0] * a.sum() / self.offpeak.A[0].sum()
        if Vo < k:
            w = np.full(self.n, bp / a.sum())
            return {"known": True, "kind": "unique", "X": np.column_stack([w, np.zeros(self.n)])}
        if np.all((V - k) * a / a.sum() > Vp):
            phi = min(self.gamma, 1 - bp / bo)
            w = np.full(self.n, bp / ((1 - phi) * a.sum()))
            return {"known": True, "kind": "unique", "X": np.column_stack([w, np.full(self.n, phi)])}
        return {"known": True, "kind": "none", "reason": "V^o > kappa and (V - kappa) a_i / sum a < V^p for some i"}

    def _extra_candidates(self, i, X):
        """Two-tier rule: in each tier, request what an unlimited request would be served."""
        if self.rule != "two_tier":
            return []
        Y = self._clip(X)
        Y[i] = [np.inf, 0.0]
        firm = water_fill(self.peak, Y[:, 0])[i]
        Y[i] = [firm, np.inf]
        sf = water_fill(self.peak, Y[:, 0])
        R = self._residual(sf)
        if R is None:
            return [(firm, 0.0)]
        nonfirm = water_fill(R, Y[:, 1])[i]
        return [(firm, nonfirm), (firm, 0.0)]

    def _upper(self, i, X):
        h = self.peak.hosting_capacities()[i]
        if self.rule == "gross":
            h /= max(1e-9, 1 - self.gamma)
        return float(2.0 * max(h, X[i, 0]))

    def _second_options(self, i, x1, X, grid):
        if self.rule == "gross":
            return np.linspace(0.0, self.gamma, grid)
        cap = self.offpeak.hosting_capacities()[i] if self.offpeak is not None else self._upper(i, X)
        return np.linspace(0.0, 2.0 * max(cap, X[i, 1]), grid)

    def _second_from_action(self, i, x1, a1, request_cap):
        if self.rule == "gross":
            return a1 * self.gamma
        cap = self.offpeak.hosting_capacities()[i] if self.offpeak is not None else self.hosting(i)
        return a1 * request_cap * cap
