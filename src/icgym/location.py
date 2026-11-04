"""The location game: applicants choose where to connect, then what to request.

Each applicant picks a placement from its finite menu: a bus, or a vector of
proportions over buses for a load split across sites. Given the placements,
the request stage under the uniform rule with a per-MW commitment below the
value has the unique equilibrium ``w(sigma)``, the max-min fair point of the
placements' batch region (paper: max-min fair equilibrium), so applicant ``i``
earns ``(V - kappa) w_i(sigma)``. The location game is the game over
placements with these payoffs.

On a downward-closed network region (counterflow treated as non-firm) it has a
strong pure equilibrium and every sequence of improving moves is finite: each
move raises the sorted vector of shares lexicographically (paper: location
game). :meth:`LocationGame.improving_moves` and
:meth:`LocationGame.best_response_dynamics` check both on any instance.

A move counts as improving if it raises the mover's share by more than
``tol`` (default 1e-4, in the region's units, MW for network regions), which
keeps solver noise out of the counts.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Hashable, Sequence

import numpy as np

from .region import Region
from .waterfill import max_min_fair_point

Placement = Hashable
Profile = tuple


def _plain(p):
    """Placements as plain hashable Python values (numpy scalars and arrays compare element-wise)."""
    if isinstance(p, np.generic):
        return p.item()
    if isinstance(p, (np.ndarray, list, tuple)):
        return tuple(float(x) for x in np.asarray(p).reshape(-1))
    return p


class LocationGame:
    """Location game over placements, with shares from the request-stage equilibrium.

    ``region_for(profile)`` returns the batch region (a :class:`Region` or an
    implicit region such as :class:`icgym.DCOPFRegion`) whose column ``i``
    is applicant ``i`` at ``profile[i]``. Shares are cached by the multiset of
    placements, so applicants at the same placement are interchangeable;
    ``margin`` is ``V - kappa`` (payoff per MW served).
    """

    def __init__(self, menus: Sequence[Sequence[Placement]], region_for: Callable[[Profile], object] | None = None,
                 margin: float | Sequence[float] = 1.0, table: dict | None = None, scale: float | None = None,
                 names: Sequence[str] | None = None):
        self.menus = [[_plain(p) for p in m] for m in menus]
        if any(len(m) == 0 for m in self.menus):
            raise ValueError("every menu needs at least one placement")
        self.region_for = region_for
        self.margin = np.broadcast_to(np.asarray(margin, dtype=float), (self.n,)).copy()
        self._cache: dict[tuple, np.ndarray] = {}
        if table is not None:
            for prof, w in table.items():
                self._store(tuple(prof), np.asarray(w, dtype=float))
        elif region_for is None:
            raise ValueError("give region_for or a precomputed table")
        self._scale = scale
        self.names = list(names) if names is not None else [f"applicant_{i}" for i in range(self.n)]
        self.evaluations = 0

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_bus_region(cls, region: Region, menus, margin=1.0) -> LocationGame:
        """Placements on a region over buses (columns of ``region`` are buses).

        A placement is a bus index or a vector of proportions over buses
        (summing to one); applicant ``i`` at ``m`` loads limit ``c`` by
        ``(A m)_c`` per MW. Needs ``region`` downward closed over all buses,
        which :class:`Region` is.
        """
        def vec(p):
            if np.ndim(p) == 0:
                e = np.zeros(region.n)
                e[int(p)] = 1.0
                return e
            return np.asarray(p, dtype=float)

        def region_for(profile):
            cols = np.column_stack([region.A @ vec(p) for p in profile])
            keep = cols.max(axis=1) > 0
            return Region(cols[keep], region.b[keep], limit_names=tuple(np.array(region.limit_names)[keep]),
                          applicant_names=tuple(f"at_{p}" if np.ndim(p) == 0 else "mix" for p in profile))

        hashable = [[_plain(p) for p in m] for m in menus]
        def alone(p):  # hosting capacity of one applicant at placement p
            load = region.A @ vec(p)
            return float(np.min(region.b[load > 0] / load[load > 0]))

        scale = max(alone(p) for m in hashable for p in m)
        return cls(hashable, region_for, margin=margin, scale=scale)

    @classmethod
    def from_network(cls, net, menu: Sequence[str], n: int, margin=1.0, **region_kw) -> LocationGame:
        """``n`` applicants choosing buses of ``menu`` on a DC network (exact DCOPF regions).

        Each new multiset of placements costs a few water-filling LPs (seconds
        on a 2,000-bus network); results are cached.
        """
        from .dcopf_region import DCOPFRegion

        return cls([list(map(str, menu))] * n, lambda prof: DCOPFRegion(net, prof, **region_kw), margin=margin)

    @classmethod
    def from_table(cls, menus, table: dict, margin=1.0) -> LocationGame:
        """A game whose shares were computed beforehand: ``table[profile] = w``."""
        return cls(menus, None, margin=margin, table=table, scale=max(float(np.max(w)) for w in table.values()))

    # ------------------------------------------------------------------ shares
    @property
    def n(self) -> int:
        """Number of applicants."""
        return len(self.menus)

    @property
    def scale(self) -> float:
        """A share scale for normalizing rewards (largest single-placement hosting capacity, if known)."""
        return self._scale if self._scale else 1.0

    def _store(self, profile, w) -> None:
        order = sorted(range(len(profile)), key=lambda i: repr(profile[i]))
        key = tuple(profile[i] for i in order)
        self._cache[key] = w[order]

    def shares(self, profile) -> np.ndarray:
        """Served loads ``w(sigma)`` at the request-stage equilibrium."""
        profile = tuple(_plain(p) for p in profile)
        if len(profile) != self.n:
            raise ValueError(f"profile has {len(profile)} placements for {self.n} applicants")
        order = sorted(range(self.n), key=lambda i: repr(profile[i]))
        key = tuple(profile[i] for i in order)
        if key not in self._cache:
            if self.region_for is None:
                raise KeyError(f"profile {profile} is not in the table")
            self._cache[key] = np.asarray(max_min_fair_point(self.region_for(key)), dtype=float)
            self.evaluations += 1
        w = np.empty(self.n)
        w[order] = self._cache[key]
        return w

    def payoffs(self, profile) -> np.ndarray:
        """Each applicant's payoff ``margin_i * w_i(profile)``."""
        return self.margin * self.shares(profile)

    def payoff(self, i: int, x, profile) -> float:
        """Applicant ``i``'s payoff when it takes placement ``x`` and the others keep ``profile``."""
        prof = list(profile)
        prof[i] = x
        return float(self.payoffs(prof)[i])

    def potential(self, profile) -> np.ndarray:
        """Shares sorted in increasing order; every improving move raises it lexicographically."""
        return np.sort(self.shares(profile))

    # ------------------------------------------------------------ equilibrium
    def best_response(self, i: int, profile, tol: float = 1e-4) -> tuple[Placement, float]:
        """Best placement of applicant ``i`` against the others, and its payoff.

        Keeps the current placement unless another raises the share by more than ``tol``.
        """
        profile = list(profile)
        best_p, best_w = profile[i], self.shares(profile)[i]
        for p in self.menus[i]:
            if p == profile[i]:
                continue
            new = profile.copy()
            new[i] = p
            w = self.shares(new)[i]
            if w > best_w + tol:
                best_p, best_w = p, w
        return best_p, float(self.margin[i] * best_w)

    def exploitability(self, profile, tol: float = 1e-4) -> np.ndarray:
        """Each applicant's payoff gain from its best move (0 at a pure equilibrium)."""
        base = self.payoffs(profile)
        return np.array([max(0.0, self.best_response(i, profile, tol)[1] - base[i]) for i in range(self.n)])

    def is_equilibrium(self, profile, tol: float = 1e-4) -> bool:
        """Whether no applicant gains more than ``tol`` (MW of share) by moving."""
        return bool(np.all(self.exploitability(profile, tol) <= tol * np.max(np.abs(self.margin))))

    def profiles(self):
        """Every placement profile (the product of the menus)."""
        return itertools.product(*self.menus)

    def pure_equilibria(self, tol: float = 1e-4) -> list[tuple]:
        """All pure equilibria, by enumerating every profile (``prod |menu_i|`` of them)."""
        return [p for p in self.profiles() if self.is_equilibrium(p, tol)]

    def improving_moves(self, tol: float = 1e-4) -> dict:
        """Every unilateral improving move, and how many fail to raise the sorted shares.

        The theory says ``violations == 0`` on a downward-closed region.
        """
        moves = violations = 0
        for prof in self.profiles():
            w = self.shares(prof)
            for i in range(self.n):
                for p in self.menus[i]:
                    if p == prof[i]:
                        continue
                    new = list(prof)
                    new[i] = p
                    w2 = self.shares(new)
                    if w2[i] > w[i] + tol:
                        moves += 1
                        d = np.sort(w2) - np.sort(w)
                        nz = np.where(np.abs(d) > tol)[0]
                        violations += int(nz.size == 0 or d[nz[0]] < 0)
        return {"moves": moves, "violations": violations}

    def best_response_dynamics(self, start, max_rounds: int = 100, order: str = "cyclic", seed: int | None = None,
                 tol: float = 1e-4):
        """Best-response dynamics in rounds; returns ``(profile, path, rounds)``.

        ``rounds`` is the number of rounds with a move, or ``None`` if the
        profile did not settle within ``max_rounds``.
        """
        rng = np.random.default_rng(seed)
        prof = list(start)
        path = [tuple(prof)]
        for k in range(max_rounds):
            moved = False
            idx = rng.permutation(self.n) if order == "random" else range(self.n)
            for i in idx:
                p, _ = self.best_response(i, prof, tol)
                if p != prof[i]:
                    prof[i] = p
                    moved = True
                    path.append(tuple(prof))
            if not moved:
                return tuple(prof), path, k
        return tuple(prof), path, None


def activsg2000_location(margin: float = 1.0) -> LocationGame:
    """The paper's ACTIVSg2000 location game: three applicants, six buses spanning the hosting-capacity range.

    Shares of all 216 placements were computed with :class:`icgym.DCOPFRegion`
    on the five representative days and ship with the package.
    """
    from .scenarios import reference

    ref = reference("activsg2000", "location_w")
    table = {tuple(k.split("|")): v for k, v in ref["w"].items()}
    return LocationGame.from_table([ref["menu"]] * ref["n"], table, margin=margin)
