"""Sequential batches: a queue of applicants studied batch after batch.

Real interconnection queues run one batch after another. Here a network is a
:class:`icgym.Region` over **buses** (column ``j`` is one MW of new load at bus
``j``). Before each batch:

* new applicants **arrive** (an :class:`ArrivalProcess`): each has a bus, a
  target size ``D`` (the MW it can use), and a value per MW served;
* the **residual region** is what the network can still host after the loads
  connected in earlier batches and any background load growth.

In the batch, every queued applicant requests capacity; the operator's rule
serves it on the residual region; served MW **connect for good**. An applicant
whose request was not met in full stays in the queue for its remaining need,
paying a waiting cost, until it is connected in full or gives up after
``max_tries`` batches. The default game keeps each applicant at its arrival bus.

Ground truth per batch: with a per-MW commitment below every value, the uniform
rule's unique equilibrium of the batch is the water-filling of the residual
region capped at each applicant's remaining need (paper: max-min fair
equilibrium, with requests bounded by need). :meth:`SequentialBatchGame.myopic_equilibrium`
returns it, so policies can be scored batch by batch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .fees import Fee, PerMWCommitment
from .region import TOL, Region
from .rules import AllocationRule, UniformRule
from .waterfill import water_fill


@dataclass
class Applicant:
    """One queued applicant: its bus, remaining need (MW), value per MW, and history."""

    ident: int
    bus: int
    need: float
    value: float
    arrived: int
    connected: float = 0.0
    tries: int = 0
    paid: float = 0.0


@dataclass
class ArrivalProcess:
    """Random arrivals: ``Poisson(rate)`` applicants per batch at buses of ``menu``.

    Sizes are drawn uniformly from ``size`` (MW) and values from ``value``
    (per MW, in the same units as the fee).
    """

    menu: list
    rate: float = 2.0
    size: tuple = (20.0, 80.0)
    value: tuple = (1.0, 1.0)

    def draw(self, rng, batch: int, start_id: int) -> list[Applicant]:
        """New applicants for ``batch``."""
        k = int(rng.poisson(self.rate))
        out = []
        for j in range(k):
            out.append(Applicant(ident=start_id + j, bus=int(self.menu[rng.integers(len(self.menu))]),
                                 need=float(rng.uniform(*self.size)), value=float(rng.uniform(*self.value)),
                                 arrived=batch))
        return out


@dataclass
class BatchRecord:
    """What happened in one batch."""

    batch: int
    applicants: list
    requests: np.ndarray
    served: np.ndarray
    payoffs: np.ndarray
    equilibrium: np.ndarray
    residual_capacity: float


@dataclass
class SequentialBatchGame:
    """A queue of applicants on a bus region, studied in ``batches`` consecutive batches.

    ``growth`` is background load growth per batch, in MW per limit (subtracted
    from the residual capacity), or ``None``. ``wait_cost`` is charged per MW of
    remaining need for each batch an applicant waits.
    """

    region: Region
    arrivals: ArrivalProcess
    batches: int = 8
    rule: AllocationRule = field(default_factory=UniformRule)
    fee: Fee = field(default_factory=lambda: PerMWCommitment(0.4))
    growth: np.ndarray | None = None
    wait_cost: float = 0.0
    max_tries: int = 4
    initial: list = field(default_factory=list)

    def reset(self, seed=None) -> None:
        """Start a new queue: nothing connected, the initial applicants queued."""
        self.rng = np.random.default_rng(seed)
        self.k = 0
        self.connected_mw = np.zeros(self.region.n)          # per bus
        self.queue: list[Applicant] = [copy.copy(a) for a in self.initial]
        self.next_id = max([a.ident for a in self.queue], default=-1) + 1
        self.done: list[Applicant] = []
        self.history: list[BatchRecord] = []
        self._arrive()

    def _arrive(self):
        new = self.arrivals.draw(self.rng, self.k, self.next_id)
        self.next_id += len(new)
        self.queue += new

    # ------------------------------------------------------------------ regions
    def residual_b(self) -> np.ndarray:
        """Capacity left on each limit after connected loads and growth."""
        b = self.region.b - self.region.A @ self.connected_mw
        if self.growth is not None:
            b = b - np.asarray(self.growth, float) * self.k
        return np.maximum(b, 0.0)

    def batch_region(self, applicants=None) -> Region:
        """The residual region seen by the queued applicants (column ``i`` is applicant ``i``)."""
        apps = self.queue if applicants is None else applicants
        cols = np.column_stack([self.region.A[:, a.bus] for a in apps]) if apps else np.zeros((self.region.m, 0))
        keep = cols.max(axis=1) > TOL if apps else np.ones(self.region.m, bool)
        return Region(cols[keep], self.residual_b()[keep], applicant_names=tuple(f"a{a.ident}" for a in apps))

    def myopic_equilibrium(self) -> np.ndarray:
        """Requests at the batch's equilibrium: water-filling capped at each applicant's need."""
        if not self.queue:
            return np.zeros(0)
        need = np.array([a.need for a in self.queue])
        return water_fill(self.batch_region(), need)

    # ------------------------------------------------------------------ one batch
    def step(self, requests) -> BatchRecord:
        """Study one batch with these requests (one per queued applicant); returns its record."""
        apps = list(self.queue)
        r = np.asarray(requests, float).reshape(len(apps))
        need = np.array([a.need for a in apps])
        eq = self.myopic_equilibrium()
        if apps:
            region = self.batch_region(apps)
            s = np.minimum(self.rule.allocate(region, r), need)
            value = np.array([a.value for a in apps])
            pay = value * s - self.fee.cost(r, s)
        else:
            s = pay = np.zeros(0)
        for a, si, ri in zip(apps, s, r):
            a.connected += si
            a.need -= si
            a.tries += 1
            a.paid += float(ri * self.fee.marginal_request_cost())
            self.connected_mw[a.bus] += si
        wait = np.array([self.wait_cost * a.need for a in apps]) if apps else np.zeros(0)
        pay = pay - wait
        rec = BatchRecord(self.k, [a.ident for a in apps], r, s, pay, eq, float(self.residual_b().sum()))
        self.history.append(rec)
        stay, leave = [], []
        for a in apps:
            (leave if a.need <= TOL or a.tries >= self.max_tries else stay).append(a)
        self.done += leave
        self.queue = stay
        self.k += 1
        if self.k < self.batches:
            self._arrive()
        return rec

    @property
    def finished(self) -> bool:
        return self.k >= self.batches

    def simulate(self, policy=None, seed=None) -> dict:
        """Run every batch; ``policy(game) -> requests`` (default: the myopic equilibrium).

        Returns summary metrics and the per-batch history.
        """
        self.reset(seed)
        policy = policy or (lambda g: g.myopic_equilibrium())
        while not self.finished:
            self.step(policy(self))
        return self.summary()

    def summary(self) -> dict:
        """Connected MW, MW requested but not served, waits, and the deviation from the batch equilibria."""
        everyone = self.done + self.queue
        served = sum(float(np.sum(h.served)) for h in self.history)
        requested = sum(float(np.sum(h.requests)) for h in self.history)
        gap = [float(np.max(np.abs(h.requests - h.equilibrium) / np.maximum(h.equilibrium, 1.0)))
               for h in self.history if h.requests.size]
        waits = [a.tries for a in everyone if a.connected > TOL]
        return {"batches": len(self.history), "connected_mw": served, "requested_mw": requested,
                "unserved_requested_mw": requested - served,
                "applicants": len(everyone), "fully_connected": sum(a.need <= TOL for a in everyone),
                "mean_batches_to_connect": float(np.mean(waits)) if waits else float("nan"),
                "max_gap_to_batch_equilibrium": max(gap) if gap else 0.0,
                "history": self.history}


# ---------------------------------------------------------------------- PettingZoo environment
try:
    from gymnasium import spaces
    from pettingzoo import ParallelEnv
except ImportError:  # pragma: no cover - optional dependency
    spaces, ParallelEnv = None, object


class SequentialBatchEnv(ParallelEnv):
    """Sequential batches as a PettingZoo ``ParallelEnv`` with arriving and leaving agents.

    Agent ``applicant_k`` is the ``k``-th applicant to arrive (up to
    ``max_applicants``); it is in ``env.agents`` while it is queued. Action: its
    request as a fraction of ``request_cap`` times its remaining need.
    Observation (6 numbers): remaining need and connected MW over the original
    size, the residual hosting capacity at its bus over its need, its value over
    the commitment, batches waited over ``max_tries``, and the episode elapsed.
    Reward: its payoff in the batch over ``value * size``. ``info`` carries the
    batch equilibrium request for the agent.
    """

    metadata = {"name": "icgym_sequential_v0", "render_modes": []}

    def __init__(self, game: SequentialBatchGame, max_applicants: int = 64, request_cap: float = 2.0):
        if spaces is None:
            raise ImportError("install the 'envs' extra for the environments")
        self.game, self.max_applicants, self.request_cap = game, max_applicants, request_cap
        self.possible_agents = [f"applicant_{k}" for k in range(max_applicants)]
        self.agents: list[str] = []
        self._obs_space = spaces.Box(0.0, np.float32(np.inf), shape=(6,), dtype=np.float32)
        self._act_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

    def observation_space(self, agent):
        """PettingZoo API: the observation space of ``agent``."""
        return self._obs_space

    def action_space(self, agent):
        """PettingZoo API: the action space of ``agent``."""
        return self._act_space

    def _name(self, a):
        return f"applicant_{a.ident}"

    def _live(self):
        return [a for a in self.game.queue if a.ident < self.max_applicants]

    def _obs(self):
        g = self.game
        kappa = max(g.fee.marginal_request_cost(), 1e-12)
        b = g.residual_b()
        out = {}
        for a in self._live():
            col = g.region.A[:, a.bus]
            hc = float(np.min(b[col > TOL] / col[col > TOL])) if np.any(col > TOL) else 0.0
            size = a.need + a.connected
            out[self._name(a)] = np.array([a.need / size, a.connected / size, hc / max(a.need, 1e-9),
                                           a.value / kappa, a.tries / g.max_tries, g.k / g.batches], dtype=np.float32)
        return out

    def _infos(self):
        g = self.game
        eq = g.myopic_equilibrium()
        return {self._name(a): {"equilibrium_request": float(e), "need": a.need, "bus": a.bus}
                for a, e in zip(g.queue, eq) if a.ident < self.max_applicants}

    def reset(self, seed=None, options=None):
        """PettingZoo API: start a new queue."""
        self.game.reset(seed)
        self.agents = [self._name(a) for a in self._live()]
        return self._obs(), self._infos()

    def step(self, actions):
        """PettingZoo API: study one batch; returns observations, rewards, terminations, truncations, infos."""
        g = self.game
        apps = list(g.queue)
        req = []
        for a in apps:
            act = actions.get(self._name(a))
            frac = 1.0 if act is None else float(np.clip(np.asarray(act).reshape(-1)[0], 0.0, 1.0))
            req.append(frac * self.request_cap * a.need)
        rec = g.step(req)
        rewards = {self._name(a): float(p / (a.value * (a.need + a.connected) + 1e-12))
                   for a, p in zip(apps, rec.payoffs) if a.ident < self.max_applicants}
        still = {self._name(a) for a in self._live()}
        last = g.finished
        terms = {n: (n not in still) or last for n in rewards}
        truncs = {n: False for n in rewards}
        obs, infos = self._obs(), self._infos()
        for n in rewards:            # agents that left get a final observation
            obs.setdefault(n, np.zeros(6, dtype=np.float32))
            infos.setdefault(n, {})
        self.agents = [] if last else sorted(still, key=lambda s: int(s.split("_")[1]))
        new = {self._name(a) for a in self._live()} - set(rewards)
        for n in new:                # arrivals appear with an observation and zero reward
            rewards[n], terms[n], truncs[n] = 0.0, last, False
        return obs, rewards, terms, truncs, infos
