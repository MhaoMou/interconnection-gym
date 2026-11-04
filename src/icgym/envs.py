"""Reinforcement-learning environments for the batch request game.

* :class:`BatchRequestParallelEnv` - PettingZoo ``ParallelEnv``: every
  applicant submits a request each batch; an episode repeats the batch on the
  same region for ``rounds`` batches.
* :class:`SingleApplicantEnv` - Gymnasium ``Env``: one learner against rivals
  that follow fixed policies (for example the equilibrium, or best responses).
* :class:`LocationParallelEnv` - PettingZoo ``ParallelEnv``: applicants choose
  placements; the request stage is played at its unique equilibrium.
* :class:`DeviceParallelEnv` - PettingZoo ``ParallelEnv``: requests plus
  on-site generation, storage, or non-firm service.

Actions are requests as a fraction of the applicant's hosting capacity times
``request_cap`` (so ``1.0`` means ``request_cap * h_i``). Rewards are payoffs
divided by ``V_i h_i`` unless ``normalize_rewards=False``. The ``info`` dict
carries the theory's ground truth (equilibrium requests, exploitability).
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np

from .game import BatchGame
from .oracles import predicted_equilibrium

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    gym = None
    spaces = None

try:
    from pettingzoo import ParallelEnv
except ImportError:  # pragma: no cover - optional dependency
    ParallelEnv = object

OBSERVATION_MODES = ("private", "public")


def _require_envs():
    if gym is None:
        raise ImportError("install the 'envs' extra: pip install "
                          "'interconnection-gym[envs] @ git+https://github.com/MhaoMou/interconnection-gym'")


class BatchRequestParallelEnv(ParallelEnv):
    """Repeated batch request game for several learning applicants.

    Observation of applicant ``i`` (all scaled by ``h_i``), for
    ``observation='private'``: its last request, its last served load, and the
    fraction of the episode elapsed. With ``'public'``, every applicant's last
    request and served load are appended.
    """

    metadata = {"name": "icgym_batch_request_v0", "render_modes": []}

    def __init__(self, game: BatchGame, rounds: int = 1, request_cap: float = 2.0,
                 observation: str = "private", normalize_rewards: bool = True,
                 compute_exploitability: bool = False):
        _require_envs()
        if observation not in OBSERVATION_MODES:
            raise ValueError(f"observation must be one of {OBSERVATION_MODES}")
        self.game = game
        self.rounds = rounds
        self.request_cap = request_cap
        self.observation_mode = observation
        self.normalize_rewards = normalize_rewards
        self.compute_exploitability = compute_exploitability
        self.possible_agents = list(game.region.applicant_names)
        self.agents: list[str] = []
        self._h = game.region.hosting_capacities()
        self._prediction = predicted_equilibrium(game)
        obs_dim = 3 if observation == "private" else 3 + 2 * game.n
        self._obs_space = spaces.Box(0.0, np.float32(max(request_cap, 1.0)), shape=(obs_dim,), dtype=np.float32)
        self._act_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

    def observation_space(self, agent):
        """PettingZoo API: the observation space of ``agent``."""
        return self._obs_space

    def action_space(self, agent):
        """PettingZoo API: the action space of ``agent``."""
        return self._act_space

    def _obs(self):
        t = np.float32(self._t / max(self.rounds, 1))
        out = {}
        for i, a in enumerate(self.possible_agents):
            own = [self._r[i] / self._h[i], self._s[i] / self._h[i], t]
            if self.observation_mode == "public":
                own += list(self._r / self._h) + list(self._s / self._h)
            out[a] = np.asarray(own, dtype=np.float32)
        return out

    def _info(self):
        gain = self.game.exploitability(self._r) if self.compute_exploitability and self._t > 0 else None
        out = {}
        for a in self.possible_agents:
            info = {"requests": self._r.copy(), "served": self._s.copy(),
                    "prediction": copy.deepcopy(self._prediction)}
            if gain is not None:
                info["exploitability"] = gain.copy()
            out[a] = info
        return out

    def reset(self, seed=None, options=None):
        """PettingZoo API: start an episode; returns observations and infos."""
        self.agents = list(self.possible_agents)
        self._t = 0
        self._r = np.zeros(self.game.n)
        self._s = np.zeros(self.game.n)
        return self._obs(), self._info()

    def requests_from_actions(self, actions) -> np.ndarray:
        """Requests in MW from the agents' actions."""
        frac = np.array([float(np.clip(np.asarray(actions[a]).reshape(-1)[0], 0.0, 1.0))
                         for a in self.possible_agents])
        return frac * self.request_cap * self._h

    def step(self, actions):
        """PettingZoo API: play one batch; returns observations, rewards, terminations, truncations, infos."""
        self._r = self.requests_from_actions(actions)
        self._s = self.game.allocate(self._r)
        pay = self.game.payoffs(self._r, self._s)
        if self.normalize_rewards:
            pay = pay / (self.game.value * self._h)
        self._t += 1
        done = self._t >= self.rounds
        rewards = {a: float(pay[i]) for i, a in enumerate(self.possible_agents)}
        terms = {a: done for a in self.possible_agents}
        truncs = {a: False for a in self.possible_agents}
        obs, info = self._obs(), self._info()
        if done:
            self.agents = []
        return obs, rewards, terms, truncs, info


Policy = Callable[[int, np.ndarray, BatchGame], float]


def equilibrium_policy(i: int, last_requests: np.ndarray, game: BatchGame) -> float:
    """Request the predicted equilibrium (falls back to the last request)."""
    pred = predicted_equilibrium(game)
    return float(pred["requests"][i]) if pred.get("known") and "requests" in pred else float(last_requests[i])


def best_response_policy(i: int, last_requests: np.ndarray, game: BatchGame) -> float:
    """Best-respond to the other applicants' last requests."""
    return game.best_response(i, last_requests)[0]


class SingleApplicantEnv(gym.Env if gym is not None else object):
    """One learner (applicant ``learner``) against rivals with fixed policies.

    Each step is one batch: the rivals choose their requests from the last
    profile, the learner's action sets its request, and the learner's reward
    is its payoff.
    """

    metadata = {"render_modes": []}

    def __init__(self, game: BatchGame, learner: int = 0, rival_policy: Policy = equilibrium_policy,
                 rounds: int = 1, request_cap: float = 2.0, normalize_rewards: bool = True):
        _require_envs()
        self.inner = BatchRequestParallelEnv(game, rounds=rounds, request_cap=request_cap,
                                             observation="public", normalize_rewards=normalize_rewards)
        self.game = game
        self.learner = learner
        self.rival_policy = rival_policy
        self.observation_space = self.inner.observation_space(None)
        self.action_space = self.inner.action_space(None)

    def reset(self, seed=None, options=None):
        """Gymnasium API: start an episode."""
        super().reset(seed=seed)
        obs, info = self.inner.reset(seed=seed)
        a = self.inner.possible_agents[self.learner]
        return obs[a], info[a]

    def step(self, action):
        """Gymnasium API: play one batch against the rivals."""
        agents = self.inner.possible_agents
        last = self.inner._r.copy()
        cap = self.inner.request_cap * self.inner._h
        actions = {}
        for i, a in enumerate(agents):
            if i == self.learner:
                actions[a] = action
            else:
                actions[a] = np.array([self.rival_policy(i, last, self.game) / cap[i]], dtype=np.float32)
        obs, rew, term, trunc, info = self.inner.step(actions)
        a = agents[self.learner]
        return obs[a], rew[a], term[a], trunc[a], info[a]


class LocationParallelEnv(ParallelEnv):
    """Repeated location game (PettingZoo ``ParallelEnv``) on a :class:`icgym.location.LocationGame`.

    Each step every applicant picks a placement from its menu (``Discrete``);
    the request stage is solved at its unique equilibrium, so applicant ``i``
    earns ``margin_i * w_i(sigma)``, divided by ``game.scale`` unless
    ``normalize_rewards=False``. The observation is the one-hot vector of every
    applicant's last placement (zeros before the first step), each applicant's
    last share over ``game.scale``, and the fraction of the episode elapsed.
    ``info`` carries the shares, whether the profile is a pure equilibrium,
    and each applicant's gain from its best move (``compute_exploitability=True``).
    """

    metadata = {"name": "icgym_location_v0", "render_modes": []}

    def __init__(self, game, rounds: int = 1, normalize_rewards: bool = True, compute_exploitability: bool = True):
        _require_envs()
        self.game = game
        self.rounds = rounds
        self.normalize_rewards = normalize_rewards
        self.compute_exploitability = compute_exploitability
        self.possible_agents = list(game.names)
        self.agents: list[str] = []
        self._sizes = [len(m) for m in game.menus]
        dim = sum(self._sizes) + game.n + 1
        self._obs_space = spaces.Box(0.0, np.float32(np.inf), shape=(dim,), dtype=np.float32)
        self._act_spaces = {a: spaces.Discrete(k) for a, k in zip(self.possible_agents, self._sizes)}

    def observation_space(self, agent):
        """PettingZoo API: the observation space of ``agent``."""
        return self._obs_space

    def action_space(self, agent):
        """PettingZoo API: the action space of ``agent``."""
        return self._act_spaces[agent]

    def _obs(self):
        onehot = []
        for i, k in enumerate(self._sizes):
            v = np.zeros(k)
            if self._choice is not None:
                v[self._choice[i]] = 1.0
            onehot.append(v)
        vec = np.concatenate(onehot + [self._w / self.game.scale, [self._t / max(self.rounds, 1)]])
        vec = vec.astype(np.float32)
        return {a: vec.copy() for a in self.possible_agents}

    def _info(self):
        out = {}
        prof = self.profile
        gains = self.game.exploitability(prof) if (self.compute_exploitability and prof is not None) else None
        for a in self.possible_agents:
            info = {"profile": prof, "shares": self._w.copy()}
            if gains is not None:
                info["exploitability"] = gains.copy()
                info["is_equilibrium"] = bool(np.all(gains <= 1e-6))
            out[a] = info
        return out

    @property
    def profile(self):
        """The current placement profile, or ``None`` before the first step."""
        if self._choice is None:
            return None
        return tuple(self.game.menus[i][c] for i, c in enumerate(self._choice))

    def reset(self, seed=None, options=None):
        """PettingZoo API: start an episode; returns observations and infos."""
        self.agents = list(self.possible_agents)
        self._t = 0
        self._choice = None
        self._w = np.zeros(self.game.n)
        return self._obs(), self._info()

    def step(self, actions):
        """PettingZoo API: play one batch; returns observations, rewards, terminations, truncations, infos."""
        self._choice = [int(np.asarray(actions[a]).reshape(-1)[0]) for a in self.possible_agents]
        self._w = self.game.shares(self.profile)
        pay = self.game.margin * self._w
        if self.normalize_rewards:
            pay = pay / self.game.scale
        self._t += 1
        done = self._t >= self.rounds
        rewards = {a: float(pay[i]) for i, a in enumerate(self.possible_agents)}
        terms = {a: done for a in self.possible_agents}
        truncs = {a: False for a in self.possible_agents}
        obs, info = self._obs(), self._info()
        if done:
            self.agents = []
        return obs, rewards, terms, truncs, info


class DeviceParallelEnv(ParallelEnv):
    """Repeated two-part game (PettingZoo ``ParallelEnv``): requests plus devices or non-firm service.

    Works with :class:`icgym.devices.GenerationGame` (request, generator or
    battery) and :class:`icgym.devices.NonFirmGame` (firm and non-firm
    requests, or a request and a curtailable fraction). Actions are in
    ``[0, 1]^2``, mapped by ``game.strategy_from_action``; rewards are payoffs
    divided by ``V * hosting_capacity(i)``. Observation: the applicant's last strategy
    and served load over ``hosting_capacity(i)``, every applicant's with
    ``observation='public'``, and the fraction of the episode elapsed. ``info``
    carries the strategies, served loads, the theory's prediction, and
    (``compute_exploitability=True``) each applicant's gain from deviating.
    """

    metadata = {"name": "icgym_devices_v0", "render_modes": []}

    def __init__(self, game, rounds: int = 1, request_cap: float = 2.0, observation: str = "private",
                 normalize_rewards: bool = True, compute_exploitability: bool = False):
        _require_envs()
        if observation not in OBSERVATION_MODES:
            raise ValueError(f"observation must be one of {OBSERVATION_MODES}")
        self.game, self.rounds, self.request_cap = game, rounds, request_cap
        self.observation_mode = observation
        self.normalize_rewards = normalize_rewards
        self.compute_exploitability = compute_exploitability
        self.possible_agents = [f"applicant_{i}" for i in range(game.n)]
        self.agents: list[str] = []
        self._h = np.array([game.hosting_capacity(i) for i in range(game.n)])
        self._prediction = game.predicted_equilibrium()
        self._width = 4 if getattr(game, "rule", "") == "two_tier" else 3
        dim = (self._width if observation == "private" else self._width * game.n) + 1
        self._obs_space = spaces.Box(0.0, np.float32(np.inf), shape=(dim,), dtype=np.float32)
        self._act_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)

    def observation_space(self, agent):
        """PettingZoo API: the observation space of ``agent``."""
        return self._obs_space

    def action_space(self, agent):
        """PettingZoo API: the action space of ``agent``."""
        return self._act_space

    def _rows(self):
        s = self._s if self._s.ndim == 2 else self._s[:, None]
        return np.column_stack([self._X / self._h[:, None], s / self._h[:, None]])

    def _obs(self):
        rows, t = self._rows(), self._t / max(self.rounds, 1)
        out = {}
        for i, a in enumerate(self.possible_agents):
            body = rows[i] if self.observation_mode == "private" else rows.reshape(-1)
            out[a] = np.append(body, t).astype(np.float32)
        return out

    def _info(self):
        gain = self.game.exploitability(self._X) if self.compute_exploitability and self._t > 0 else None
        out = {}
        for a in self.possible_agents:
            info = {"strategies": self._X.copy(), "served": self._s.copy(),
                    "prediction": copy.deepcopy(self._prediction)}
            if gain is not None:
                info["exploitability"] = gain.copy()
            out[a] = info
        return out

    def reset(self, seed=None, options=None):
        """PettingZoo API: start an episode; returns observations and infos."""
        self.agents = list(self.possible_agents)
        self._t = 0
        self._X = np.zeros((self.game.n, 2))
        self._s = np.zeros((self.game.n, 2)) if self._width == 4 else np.zeros(self.game.n)
        return self._obs(), self._info()

    def step(self, actions):
        """PettingZoo API: play one batch; returns observations, rewards, terminations, truncations, infos."""
        self._X = np.array([self.game.strategy_from_action(i, actions[a], self.request_cap)
                            for i, a in enumerate(self.possible_agents)])
        self._s = self.game.allocate(self._X)
        pay = self.game.payoffs(self._X)
        if self.normalize_rewards:
            pay = pay / (self.game.value * self._h)
        self._t += 1
        done = self._t >= self.rounds
        rewards = {a: float(pay[i]) for i, a in enumerate(self.possible_agents)}
        terms = {a: done for a in self.possible_agents}
        truncs = {a: False for a in self.possible_agents}
        obs, info = self._obs(), self._info()
        if done:
            self.agents = []
        return obs, rewards, terms, truncs, info
