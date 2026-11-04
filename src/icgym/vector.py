"""Batched allocation and a vectorized environment, for fast learning.

:func:`allocate_batch` applies a rule to ``E`` request profiles at once (the
uniform and proportional rules are vectorized in numpy, with the minimum
allocation; other rules and implicit regions fall back to a loop).
:class:`VectorBatchRequestEnv` steps ``E`` independent copies of the batch
request game in one call, with numpy arrays in and out:

    env = VectorBatchRequestEnv(game, num_envs=256)
    obs = env.reset(seed=0)                  # (E, n, obs_dim)
    obs, rew, done, info = env.step(acts)    # acts (E, n) in [0, 1]; rew (E, n)

It has the same action map, rewards, and observations as
:class:`icgym.envs.BatchRequestParallelEnv`, and needs neither gymnasium nor
pettingzoo.
"""

from __future__ import annotations

import numpy as np

from .game import BatchGame
from .region import TOL, Region
from .rules import ProportionalRule, UniformRule


def water_fill_batch(region: Region, requests, weights=None, active=None) -> np.ndarray:
    """:func:`icgym.water_fill` on each row of ``requests`` (shape ``(E, n)``)."""
    A, b = region.A, region.b
    R = np.atleast_2d(np.asarray(requests, dtype=float))
    E, n = R.shape
    lam = np.ones(n) if weights is None else np.asarray(weights, dtype=float).reshape(n)
    rising = (R > TOL) if active is None else (np.asarray(active, bool).reshape(E, n) & (R > TOL))
    S = np.zeros((E, n))
    scale = max(1.0, float(np.max(b)) if b.size else 1.0)
    loads = A > TOL                                            # (m, n)
    for _ in range(n + 1):
        live = rising.any(axis=1)
        if not live.any():
            break
        fixed = (S * ~rising) @ A.T                            # (E, m)
        rate = (lam * rising) @ A.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lim = np.where(rate > TOL, (b - fixed) / np.where(rate > TOL, rate, 1.0), np.inf)
        t_lim = np.maximum(t_lim, 0.0)
        t_req = np.where(rising, R / lam, np.inf)
        t = np.minimum(t_lim.min(axis=1), t_req.min(axis=1))   # (E,)
        if np.any(live & ~np.isfinite(t)):
            raise RuntimeError("water-filling did not stop; the region is unbounded for the active applicants")
        t = np.where(live, t, 0.0)
        S = np.where(rising, lam * t[:, None], S)
        stop = rising & (t_req <= t[:, None] * (1 + 1e-12) + TOL)
        tight = (rate > TOL) & (t_lim <= t[:, None] * (1 + 1e-12) + TOL * scale)   # (E, m)
        stop |= rising & ((tight.astype(float) @ loads.astype(float)) > 0)
        stuck = live & ~stop.any(axis=1)
        if stuck.any():  # numerical safety, as in water_fill
            k = np.argmin(np.where(rising, t_req, np.inf), axis=1)
            stop[np.where(stuck)[0], k[stuck]] = True
        rising &= ~stop
    return np.minimum(S, R)


def _proportional_batch(region: Region, R, active):
    Ra = np.where(active, R, 0.0)
    load = Ra @ region.A.T
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(load > TOL, region.b / np.where(load > TOL, load, 1.0), np.inf)
    alpha = np.minimum(1.0, ratio.min(axis=1))
    return alpha[:, None] * Ra


def allocate_batch(rule, region, requests) -> np.ndarray:
    """``rule.allocate`` on each row of ``requests`` (shape ``(E, n)``)."""
    R = np.atleast_2d(np.asarray(requests, dtype=float))
    if getattr(region, "is_implicit", False) or not isinstance(rule, (UniformRule, ProportionalRule)):
        return np.array([rule.allocate(region, r) for r in R])
    if np.any(R < 0):
        raise ValueError("requests must be nonnegative")
    active = R > TOL
    E = R.shape[0]
    rows = np.arange(E)
    for _ in range(R.shape[1] + 1):
        if isinstance(rule, UniformRule):
            S = water_fill_batch(region, R, weights=rule.weights, active=active)
        else:
            S = _proportional_batch(region, R, active)
        S = np.where(active, S, 0.0)
        short = active & (S < rule.beta * R * (1.0 - 1e-9))
        cut = short.any(axis=1)
        if not cut.any():
            return S
        if rule.cut_order == "largest_request":
            key = np.where(short, R, -np.inf)
        else:
            load = np.where(short | (S > 0), R, 0.0) @ region.A.T
            c = np.argmax(load / np.where(region.b > TOL, region.b, TOL), axis=1)
            key = np.where(short, region.A[c] * R, -np.inf)
        best = key.max(axis=1, keepdims=True)
        first = np.argmax(np.isclose(key, best, rtol=1e-12, atol=1e-12) & short, axis=1)
        active[rows[cut], first[cut]] = False
    return S  # pragma: no cover - the loop always ends within n cuts


def payoffs_batch(game: BatchGame, requests) -> np.ndarray:
    """Payoffs of ``E`` request profiles, shape ``(E, n)``."""
    R = np.atleast_2d(np.asarray(requests, dtype=float))
    return _payoffs(game, R, allocate_batch(game.rule, game.region, R))


def _payoffs(game: BatchGame, R, S) -> np.ndarray:
    from .fees import PerMWCommitment

    if isinstance(game.fee, PerMWCommitment):
        cost = game.fee.kappa * R
    else:
        cost = np.array([game.fee.cost(r, s) for r, s in zip(R, S)])
    return game.value * S - cost


class VectorBatchRequestEnv:
    """``num_envs`` copies of :class:`icgym.envs.BatchRequestParallelEnv`, stepped together.

    Each copy runs ``rounds`` batches and then resets itself; ``done[e]`` marks
    the step that ended copy ``e``'s episode, and the returned observation is
    then the first observation of its next episode.
    """

    def __init__(self, game: BatchGame, num_envs: int = 64, rounds: int = 1, request_cap: float = 2.0,
                 observation: str = "private", normalize_rewards: bool = True):
        if observation not in ("private", "public"):
            raise ValueError("observation must be 'private' or 'public'")
        self.game, self.num_envs, self.rounds, self.request_cap = game, num_envs, rounds, request_cap
        self.observation_mode, self.normalize_rewards = observation, normalize_rewards
        self.n = game.n
        self._h = game.region.hosting_capacities()
        self.obs_dim = 3 if observation == "private" else 3 + 2 * self.n

    def _obs(self):
        E, n = self.num_envs, self.n
        r, s = self._r / self._h, self._s / self._h
        t = np.broadcast_to((self._t / max(self.rounds, 1))[:, None, None], (E, n, 1))
        own = np.concatenate([r[:, :, None], s[:, :, None], t], axis=2)
        if self.observation_mode == "public":
            pub = np.concatenate([r, s], axis=1)[:, None, :].repeat(n, axis=1)
            own = np.concatenate([own, pub], axis=2)
        return own.astype(np.float32)

    def reset(self, seed=None):
        """Reset every copy; returns observations of shape ``(E, n, obs_dim)``."""
        self._r = np.zeros((self.num_envs, self.n))
        self._s = np.zeros((self.num_envs, self.n))
        self._t = np.zeros(self.num_envs)
        return self._obs()

    def requests_from_actions(self, actions) -> np.ndarray:
        """Requests from actions in ``[0, 1]``, shape ``(E, n)``."""
        a = np.clip(np.asarray(actions, dtype=float).reshape(self.num_envs, self.n), 0.0, 1.0)
        return a * self.request_cap * self._h

    def step(self, actions):
        """Step every copy; returns ``(obs, rewards, done, info)``."""
        R = self.requests_from_actions(actions)
        S = allocate_batch(self.game.rule, self.game.region, R)
        pay = _payoffs(self.game, R, S)
        if self.normalize_rewards:
            pay = pay / (self.game.value * self._h)
        self._r, self._s, self._t = R, S, self._t + 1
        done = self._t >= self.rounds
        info = {"requests": R, "served": S}
        obs = self._obs()
        if done.any():
            self._r[done] = 0.0
            self._s[done] = 0.0
            self._t[done] = 0.0
            obs = np.where(done[:, None, None], self._obs(), obs)
        return obs, pay.astype(float), done, info
