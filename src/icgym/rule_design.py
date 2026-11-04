"""The operator as a learning agent: choose the rule, its minimum allocation, and the commitment.

Applicants respond strategically, so a rule's parameters have to be judged at
the applicants' equilibrium, not at their truthful requests. These environments
let an operator (a single RL agent, a bandit, or a search) choose them and be
scored by an objective of its choice:

* :class:`OperatorEnv` (Gymnasium): one batch per episode on a network drawn
  from a scenario distribution;
* :class:`SequentialOperatorEnv` (Gymnasium): a queue of batches
  (:class:`icgym.sequential.SequentialBatchGame`), parameters chosen before
  each batch.

Applicants answer with :func:`equilibrium_response`: the theory's equilibrium
where it is known (uniform rule with a per-MW commitment: the max-min fair point
capped at each applicant's need; uniform rule with a flat fee: an end of the
band ``[w, w/beta]``), and best-response dynamics otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fees import FlatFee, PerMWCommitment
from .game import BatchGame
from .region import TOL, Region
from .rules import ProportionalRule, UniformRule, UtilitarianRule
from .scenarios import random_region
from .waterfill import water_fill

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    gym, spaces = None, None

RULE_NAMES = ("uniform", "proportional", "utilitarian")


def _with_needs(region: Region, need) -> Region:
    """``region`` with the extra limits ``s_i <= need_i`` (value stops at the need)."""
    need = np.asarray(need, float)
    if not np.any(np.isfinite(need)):
        return region
    idx = np.where(np.isfinite(need))[0]
    rows = np.zeros((idx.size, region.n))
    rows[np.arange(idx.size), idx] = 1.0
    return Region(np.vstack([region.A, rows]), np.concatenate([region.b, need[idx]]),
                  limit_names=region.limit_names + tuple(f"need_{i}" for i in idx),
                  applicant_names=region.applicant_names)


def make_rule(name: str, beta: float):
    """Rule by name: ``'uniform'``, ``'proportional'`` (cut by loading), or ``'utilitarian'``."""
    return {"uniform": UniformRule, "proportional": ProportionalRule, "utilitarian": UtilitarianRule}[name](beta=beta)


def equilibrium_response(region: Region, rule, fee, value, need=None, flat_fee_end: str = "low",
                         max_rounds: int = 15, grid: int = 80) -> np.ndarray:
    """Applicants' requests at equilibrium on ``region`` under ``rule`` and ``fee``.

    ``need`` caps the MW each applicant can use (``None``: unbounded). The
    uniform rule with a per-MW commitment below every value gives the max-min
    fair point capped at the needs; with a flat fee the band's low end ``w`` or
    high end ``w / beta`` (``flat_fee_end``). Other cases use best-response
    dynamics from the capped max-min fair point. Applicants whose value is at
    most the commitment request nothing.
    """
    n = region.n
    V = np.broadcast_to(np.asarray(value, float), (n,))
    need = np.full(n, np.inf) if need is None else np.asarray(need, float)
    kappa = fee.marginal_request_cost()
    keep = V > kappa + TOL
    r = np.zeros(n)
    if not keep.any():
        return r
    capped = _with_needs(region, need)
    w = water_fill(capped, np.where(keep, need, 0.0))
    if isinstance(rule, UniformRule) and rule.weights is None:
        if isinstance(fee, FlatFee) or kappa <= TOL:
            if flat_fee_end == "high" and rule.beta > 0:
                return np.where(keep, w / rule.beta, 0.0)
            return w
        return w
    game = BatchGame(capped, rule, fee, V)
    start = w.copy()
    r, _, _ = game.best_response_dynamics(start, max_rounds=max_rounds, method="search", grid=grid,
                                          tol=1e-6 * max(1.0, float(np.max(w))))
    return np.where(keep, r, 0.0)


@dataclass
class Objective:
    """Operator's reward: ``efficiency - waste * wasted_share - unfairness * (1 - jain)``.

    * efficiency: total served over the planner's optimum;
    * wasted_share: MW requested but not served, over MW requested;
    * jain: Jain's index of the served shares of hosting capacity (1 = equal).
    """

    waste: float = 0.5
    unfairness: float = 0.0

    def __call__(self, metrics: dict) -> float:
        return metrics["efficiency"] - self.waste * metrics["wasted_share"] - self.unfairness * (1 - metrics["jain"])


def batch_metrics(region: Region, requests, served) -> dict:
    """Efficiency, wasted share, and Jain's fairness index of one batch."""
    r, s = np.asarray(requests, float), np.asarray(served, float)
    planner = region.max_total()[0]
    h = region.hosting_capacities()
    x = s / np.where(h > TOL, h, 1.0)
    jain = float(x.sum() ** 2 / (len(x) * (x ** 2).sum())) if (x ** 2).sum() > 0 else 1.0
    return {"served_mw": float(s.sum()), "planner_mw": float(planner),
            "efficiency": float(s.sum() / planner) if planner > 0 else 0.0,
            "wasted_share": float((r.sum() - s.sum()) / r.sum()) if r.sum() > TOL else 0.0, "jain": jain}


def _decode(action, rules, kappa_max):
    a = np.clip(np.asarray(action, float).reshape(3), 0.0, 1.0)
    name = rules[min(int(a[0] * len(rules)), len(rules) - 1)]
    return name, float(a[1]), float(a[2] * kappa_max)


class OperatorEnv(gym.Env if gym is not None else object):
    """One batch per episode: the operator picks ``(rule, beta, kappa)``, applicants respond at equilibrium.

    Action ``a`` in ``[0, 1]^3``: ``a[0]`` picks the rule from ``rules`` (equal
    bins), ``a[1]`` is the minimum allocation ``beta``, ``a[2] * kappa_max`` is
    the commitment per MW (0 is a flat fee). Observation: the number of
    applicants and of limits over their maxima, the sorted hosting capacities
    over the largest (padded with zeros), the price-of-anarchy bound over 10,
    and the mean value over ``kappa_max``. Reward: ``objective(metrics)``;
    ``info`` has the metrics, the requests and served loads, and the theory's
    reference (the uniform rule with a commitment serves ``1 / PoA`` of the
    planner, with nothing wasted).
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario=None, rules=RULE_NAMES, objective: Objective | None = None, kappa_max: float = 1.0,
                 max_applicants: int = 6, max_limits: int = 6, flat_fee_end: str = "high"):
        if gym is None:
            raise ImportError("install the 'envs' extra for the environments")
        self.rules, self.kappa_max, self.max_n, self.max_m = tuple(rules), kappa_max, max_applicants, max_limits
        self.objective = objective or Objective()
        self.flat_fee_end = flat_fee_end
        self.scenario = scenario or self._random_scenario
        self.observation_space = spaces.Box(0.0, np.float32(np.inf), shape=(4 + max_applicants,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)

    def _random_scenario(self, rng):
        n = int(rng.integers(2, self.max_n + 1))
        m = int(rng.integers(1, self.max_m + 1))
        region = random_region(n, m, seed=int(rng.integers(1 << 31)), skew=float(rng.uniform(1.0, 3.0)))
        return region, rng.uniform(1.0, 2.0, size=n)

    def _obs(self):
        from .oracles import poa_bound
        R, V = self.region, self.values
        h = np.sort(R.hosting_capacities())[::-1]
        hs = np.zeros(self.max_n)
        hs[: R.n] = h[: self.max_n] / h.max()
        return np.concatenate([[R.n / self.max_n, R.m / self.max_m, poa_bound(R) / 10.0,
                                float(np.mean(V)) / self.kappa_max], hs]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Gymnasium API: draw a network and the applicants' values."""
        super().reset(seed=seed)
        self.region, self.values = self.scenario(self.np_random)
        return self._obs(), {}

    def step(self, action):
        """Gymnasium API: apply the operator's choice to the batch and score the equilibrium."""
        name, beta, kappa = _decode(action, self.rules, self.kappa_max)
        rule = make_rule(name, beta)
        fee = PerMWCommitment(kappa) if kappa > TOL else FlatFee(0.0)
        r = equilibrium_response(self.region, rule, fee, self.values, flat_fee_end=self.flat_fee_end)
        s = rule.allocate(self.region, r)
        m = batch_metrics(self.region, r, s)
        info = {**m, "rule": name, "beta": beta, "kappa": kappa, "requests": r, "served": s}
        return self._obs(), float(self.objective(m)), True, False, info


class SequentialOperatorEnv(gym.Env if gym is not None else object):
    """The operator sets ``(rule, beta, kappa)`` before each batch of a queue.

    Built on :class:`icgym.sequential.SequentialBatchGame`; applicants answer
    each batch at the equilibrium of that batch (capped at their remaining
    needs). Observation: queue length over ``max_queue``, remaining capacity
    over the initial, total remaining need over the initial capacity, and the
    episode elapsed. Reward per batch: MW connected over the initial capacity,
    minus ``waste`` times the MW requested but not served over the initial
    capacity.
    """

    metadata = {"render_modes": []}

    def __init__(self, game, rules=RULE_NAMES, kappa_max: float = 1.0, waste: float = 0.5, max_queue: int = 20):
        if gym is None:
            raise ImportError("install the 'envs' extra for the environments")
        self.game, self.rules, self.kappa_max = game, tuple(rules), kappa_max
        self.waste, self.max_queue = waste, max_queue
        self.observation_space = spaces.Box(0.0, np.float32(np.inf), shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)

    def _obs(self):
        g = self.game
        cap0 = float(g.region.b.sum())
        need = sum(a.need for a in g.queue)
        return np.array([len(g.queue) / self.max_queue, g.residual_b().sum() / cap0, need / cap0,
                         g.k / g.batches], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Gymnasium API: start a new queue."""
        super().reset(seed=seed)
        self.game.reset(None if seed is None else int(seed))
        return self._obs(), {}

    def step(self, action):
        """Gymnasium API: study the next batch under the operator's choice."""
        g = self.game
        name, beta, kappa = _decode(action, self.rules, self.kappa_max)
        g.rule = make_rule(name, beta)
        g.fee = PerMWCommitment(kappa) if kappa > TOL else FlatFee(0.0)
        if g.queue:
            need = np.array([a.need for a in g.queue])
            values = np.array([a.value for a in g.queue])
            r = equilibrium_response(g.batch_region(), g.rule, g.fee, values, need=need, flat_fee_end="high")
        else:
            r = np.zeros(0)
        rec = g.step(r)
        cap0 = float(g.region.b.sum())
        served, requested = float(rec.served.sum()), float(rec.requests.sum())
        reward = served / cap0 - self.waste * (requested - served) / cap0
        info = {"rule": name, "beta": beta, "kappa": kappa, "served_mw": served, "requested_mw": requested}
        return self._obs(), reward, g.finished, False, info
