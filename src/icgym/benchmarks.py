"""Benchmark tasks with known answers, three baselines, and a runner.

Each :class:`Task` is a game whose equilibrium the theory pins down (or
proves absent), so a learner's final profile can be scored exactly:

* ``exploitability``: the largest gain any applicant has from deviating
  alone, over its value at its hosting capacity (0 at an equilibrium);
* ``distance``: the largest gap to the predicted equilibrium, per applicant,
  over its hosting capacity (served loads for the flat-fee band, where any
  profile in the band is an equilibrium; absent when none exists);
* ``efficiency``: total served over the planner's optimum (batch tasks) or
  the total share over the best total of any placement (location).

Baselines: ``best_response`` (best-response dynamics on the game, an oracle
that sees payoffs), ``hedge`` (full-information multiplicative weights on a
grid of actions; it sees the payoff of every action against the rivals'
sampled play), and ``independent_pg`` (independent policy gradient, which
sees only its own rewards, as an RL agent in the environments would).

    from icgym.benchmarks import TASKS, run, run_suite, format_results
    res = run_suite(["uniform-ieee14"], ["best_response", "independent_pg"], seeds=range(3))
    print(format_results(res))
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .devices import Device, GenerationGame, NonFirmGame, TwoPartGame
from .fees import FlatFee, PerMWCommitment
from .game import BatchGame
from .location import LocationGame, activsg2000_location
from .oracles import predicted_equilibrium
from .region import Region
from .rules import ProportionalRule, UniformRule
from .scenarios import ieee14, random_region
from .vector import allocate_batch, payoffs_batch


# --------------------------------------------------------------------- tasks
@dataclass(frozen=True)
class Task:
    name: str
    description: str
    make: Callable[[], object]
    expected: str          # "unique", "band", "location", or "none"


def _single_limit():
    return Region.single_limit([1.0, 1.5, 2.0], 6.0)


TASKS: dict[str, Task] = {t.name: t for t in [
    Task("uniform-ieee14", "Uniform rule, beta 0.5, IEEE 14-bus buses 9, 13, 14: unique equilibrium w",
         lambda: BatchGame(ieee14((9, 13, 14)), UniformRule(beta=0.5), PerMWCommitment(0.4)), "unique"),
    Task("uniform-random8", "Uniform rule, beta 0.5, 8 applicants on a random 6-limit region: unique equilibrium w",
         lambda: BatchGame(random_region(8, 6, seed=0), UniformRule(beta=0.5), PerMWCommitment(0.4)), "unique"),
    Task("flatfee-ieee14", "Uniform rule with a flat fee, buses 1 and 2: every profile in [w, w/beta] is an "
         "equilibrium", lambda: BatchGame(ieee14((1, 2)), UniformRule(beta=0.5), FlatFee(1.0)), "band"),
    Task("proportional-limit", "Proportional rule, beta 0.5, one limit, 4 applicants: requests inflated by rho",
         lambda: BatchGame(Region.single_limit([1.0, 2.0, 1.5, 1.0], 6.0), ProportionalRule(beta=0.5),
                           PerMWCommitment(0.4)), "unique"),
    Task("location-activsg2000", "Location game, 3 applicants over 6 ACTIVSg2000 buses: 3 pure equilibria",
         activsg2000_location, "location"),
    Task("generation-credit", "Generators credited to their owners, buses 9, 13, 14: unique equilibrium with "
         "generators at the cap", lambda: GenerationGame(ieee14((9, 13, 14)), Device.generator(
             run_hours=4, nu=6, gamma=0.2, capital=0.3), value=1.0, kappa=0.4, rule="credit"), "unique"),
    Task("generation-gross-cycle", "Gross rule between the builder's and scrapper's margins, one limit: no pure "
         "equilibrium", lambda: GenerationGame(_single_limit(), Device(0.5, 0.5, 0.15), value=1.0, kappa=0.4,
                                               rule="gross"), "none"),
    Task("nonfirm-two-tier", "Two-tier firm and non-firm rule, buses 9, 13, 14, off-peak limits 1.6 times the "
         "peak: equilibrium served in full", lambda: (lambda R: NonFirmGame(
             R, Region(R.A, 1.6 * R.b), value=1.0, peak_value=0.2, kappa=0.4, rule="two_tier"))(
                 ieee14((9, 13, 14))), "unique"),
]}


# --------------------------------------------------------------------- adapters
class _BatchAdapter:
    """Batch request game; actions in [0, 1] map to requests ``a * request_cap * h_i``."""

    kind, dim = "box", 1

    def __init__(self, game: BatchGame, request_cap: float = 2.0):
        self.game, self.cap = game, request_cap
        self.n = game.n
        self.h = game.region.hosting_capacities()
        self.planner = game.region.max_total()[0]
        self.pred = predicted_equilibrium(game)

    def to_strategy(self, a):
        return np.clip(np.asarray(a, float).reshape(self.n), 0, 1) * self.cap * self.h

    def rewards(self, A):
        R = np.clip(np.asarray(A, float).reshape(-1, self.n), 0, 1) * self.cap * self.h
        return payoffs_batch(self.game, R) / (self.game.value * self.h)

    def best_response_play(self, rng, rounds=30):
        start = rng.random(self.n) * self.cap * self.h
        r, _, k = self.game.best_response_dynamics(start, max_rounds=rounds, method="search", grid=150,
                                                   tol=1e-6 * self.h.max())
        self.converged = k is not None
        return r

    def evaluate(self, r):
        r = np.asarray(r, float)
        s = allocate_batch(self.game.rule, self.game.region, r[None])[0]
        gain = self.game.exploitability(r, grid=200) / (self.game.value * self.h)
        out = {"exploitability": float(gain.max()), "efficiency": float(s.sum() / self.planner)}
        p = self.pred
        if p.get("kind") == "band":
            out["distance"] = float(np.max(np.abs(s - p["served"]) / self.h))
            out["in_band"] = bool(np.all(r >= p["low"] - 1e-6 * self.h) and np.all(r <= p["high"] + 1e-6 * self.h))
        elif "requests" in p:
            out["distance"] = float(np.max(np.abs(r - p["requests"]) / self.h))
        return out


class _DeviceAdapter:
    """Two-part games; actions in [0, 1]^2 mapped by ``game.strategy_from_action``."""

    kind, dim = "box", 2

    def __init__(self, game: TwoPartGame, request_cap: float = 2.0):
        self.game, self.cap = game, request_cap
        self.n = game.n
        self.h = np.array([game.hosting(i) for i in range(self.n)])
        self.pred = game.predicted_equilibrium()

    def to_strategy(self, a):
        a = np.asarray(a, float).reshape(self.n, 2)
        return np.array([self.game.strategy_from_action(i, a[i], self.cap) for i in range(self.n)])

    def rewards(self, A):
        A = np.asarray(A, float).reshape(-1, self.n, 2)
        return np.array([self.game.payoffs(self.to_strategy(a)) for a in A]) / (self.game.value * self.h)

    def best_response_play(self, rng, rounds=20):
        X = self.to_strategy(rng.random((self.n, 2)))
        self.converged = False
        for _ in range(rounds):
            prev = X.copy()
            for i in range(self.n):
                X[i] = self.game.best_response(i, X, grid=60, second_grid=6)[0]
            if np.allclose(X, prev, atol=1e-6 * self.h.max()):
                self.converged = True
                break
        return X

    def _served(self, X):
        s = self.game.allocate(X)
        return s if s.ndim == 2 else s[:, None]      # per tier for the two-tier rule

    def evaluate(self, X):
        gain = self.game.exploitability(X, grid=80, second_grid=8) / (self.game.value * self.h)
        out = {"exploitability": float(gain.max())}
        if self.pred.get("known") and "X" in self.pred:
            gap = np.abs(self._served(X) - self._served(self.pred["X"])) / self.h[:, None]
            out["distance"] = float(gap.max())
        return out


class _LocationAdapter:
    """Location game; discrete actions index each applicant's menu."""

    kind = "discrete"

    def __init__(self, game: LocationGame):
        self.game = game
        self.n = game.n
        self.sizes = [len(m) for m in game.menus]
        self.best_total = max(float(game.shares(p).sum()) for p in game.profiles())

    def to_strategy(self, a):
        return tuple(self.game.menus[i][int(k)] for i, k in enumerate(np.asarray(a).reshape(self.n)))

    def rewards(self, A):
        A = np.asarray(A).reshape(-1, self.n)
        return np.array([self.game.payoffs(self.to_strategy(a)) for a in A]) / self.game.scale

    def best_response_play(self, rng, rounds=50):
        start = tuple(m[rng.integers(len(m))] for m in self.game.menus)
        prof, _, k = self.game.dynamics(start, max_rounds=rounds)
        self.converged = k is not None
        return prof

    def evaluate(self, prof):
        gains = self.game.gains(prof) / self.game.scale
        return {"exploitability": float(gains.max()), "distance": 0.0 if self.game.is_equilibrium(prof) else 1.0,
                "efficiency": float(self.game.shares(prof).sum() / self.best_total)}


def adapter(game, request_cap: float = 2.0):
    if isinstance(game, BatchGame):
        return _BatchAdapter(game, request_cap)
    if isinstance(game, (GenerationGame, NonFirmGame)):
        return _DeviceAdapter(game, request_cap)
    if isinstance(game, LocationGame):
        return _LocationAdapter(game)
    raise TypeError(f"no benchmark adapter for {type(game).__name__}")


# --------------------------------------------------------------------- baselines
def _grid(ad, k1: int, k2: int):
    """Action grid per applicant: points in [0, 1]^dim, or menu indices."""
    if ad.kind == "discrete":
        return [np.arange(s)[:, None] for s in ad.sizes]
    g1 = np.linspace(0.0, 1.0, k1)
    if ad.dim == 1:
        return [g1[:, None]] * ad.n
    pts = np.array(list(itertools.product(g1, np.linspace(0.0, 1.0, k2))))
    return [pts] * ad.n


def _assemble(ad, actions):
    """Stack per-applicant action rows into the adapter's batch shape."""
    A = np.stack(actions, axis=1)            # (E, n, d)
    return A[..., 0] if (ad.kind == "discrete" or ad.dim == 1) else A


def best_response(ad, rng, iterations=None):
    return ad.best_response_play(rng)


def hedge(ad, rng, iterations: int = 200, samples: int = 4, eta: float = 8.0, k1: int = 41, k2: int = 5):
    """Full-information multiplicative weights; returns each applicant's most likely grid action."""
    if ad.kind == "box" and ad.dim == 2:
        k1 = min(k1, 17)
    grid = _grid(ad, k1, k2)
    logw = [np.zeros(len(g)) for g in grid]
    for _ in range(iterations):
        probs = [np.exp(lw - lw.max()) / np.exp(lw - lw.max()).sum() for lw in logw]
        draws = [g[rng.choice(len(g), size=samples, p=p)] for g, p in zip(grid, probs)]
        for i in range(ad.n):
            K = len(grid[i])
            rows = []
            for j in range(ad.n):
                rows.append(np.repeat(grid[i], samples, axis=0) if j == i else np.tile(draws[j], (K, 1)))
            u = ad.rewards(_assemble(ad, rows))[:, i].reshape(K, samples).mean(axis=1)
            logw[i] = logw[i] + eta * u
    a = [g[int(np.argmax(lw))] for g, lw in zip(grid, logw)]
    return ad.to_strategy(_assemble(ad, [x[None] for x in a])[0])


def independent_pg(ad, rng, iterations: int = 400, batch: int = 128, lr: float = 0.3):
    """Independent policy gradient with a per-applicant baseline, from rewards only.

    Box actions: a Gaussian in logit space, ``a = sigmoid(z)``, updated along
    the natural gradient (the mean moves by ``lr * sigma * E[adv * eps]``, as
    in evolution strategies, so steps shrink with the noise); discrete actions:
    softmax over the menu. Advantages are normalized per applicant and the
    step size decays linearly. Returns the mean (or most likely) action.
    """
    n = ad.n
    if ad.kind == "discrete":
        logits = [np.zeros(s) for s in ad.sizes]
        for it in range(iterations):
            step = lr * (1 - it / iterations) + 0.02
            probs = [np.exp(lg - lg.max()) / np.exp(lg - lg.max()).sum() for lg in logits]
            A = np.stack([rng.choice(len(p), size=batch, p=p) for p in probs], axis=1)
            R = ad.rewards(A)
            for i in range(n):
                adv = R[:, i] - R[:, i].mean()
                adv = adv / (adv.std() + 1e-8)
                onehot = np.eye(len(probs[i]))[A[:, i]]
                logits[i] = logits[i] + 3 * step * ((onehot - probs[i]) * adv[:, None]).mean(axis=0)
        return ad.to_strategy(np.array([int(np.argmax(lg)) for lg in logits]))
    d = ad.dim
    mu = np.zeros((n, d))
    log_sigma = np.zeros((n, d))
    for it in range(iterations):
        step = lr * (1 - it / iterations) + 0.02
        sigma = np.exp(log_sigma)
        z = mu + sigma * rng.standard_normal((batch, n, d))
        A = 1.0 / (1.0 + np.exp(-z))
        R = ad.rewards(A[..., 0] if d == 1 else A)
        for i in range(n):
            adv = R[:, i] - R[:, i].mean()
            adv = adv / (adv.std() + 1e-8)
            eps = (z[:, i] - mu[i]) / sigma[i]
            mu[i] += step * sigma[i] * (adv[:, None] * eps).mean(axis=0)
            log_sigma[i] += 0.5 * step * (adv[:, None] * (eps ** 2 - 1)).mean(axis=0)
        log_sigma = np.clip(log_sigma, -6.0, 1.0)
    a = 1.0 / (1.0 + np.exp(-mu))
    return ad.to_strategy(a[:, 0] if d == 1 else a)


BASELINES = {"best_response": best_response, "hedge": hedge, "independent_pg": independent_pg}


# --------------------------------------------------------------------- runner
def run(task: str, baseline: str, seed: int = 0, **kw) -> dict:
    """Run one baseline on one task; returns the metrics of its final profile."""
    t = TASKS[task]
    ad = adapter(t.make())
    rng = np.random.default_rng(seed)
    t0 = time.time()
    strategy = BASELINES[baseline](ad, rng, **kw)
    seconds = time.time() - t0
    metrics = ad.evaluate(strategy)
    if baseline == "best_response":
        metrics["converged"] = bool(getattr(ad, "converged", False))
    return {"task": task, "baseline": baseline, "seed": seed, "expected": t.expected, "seconds": seconds,
            **metrics, "strategy": np.asarray(strategy, dtype=object).tolist()}


def run_suite(tasks=None, baselines=None, seeds=range(3), progress: bool = False) -> list[dict]:
    out = []
    for task in tasks or list(TASKS):
        for b in baselines or list(BASELINES):
            for s in seeds:
                out.append(run(task, b, s))
                if progress:
                    r = out[-1]
                    print(f"{task:24s} {b:15s} seed {s}: exploitability {r['exploitability']:.4f} "
                          f"({r['seconds']:.1f}s)", flush=True)
    return out


def format_results(results: list[dict]) -> str:
    """Markdown table: mean (and max) over seeds for each task and baseline."""
    keys = sorted({(r["task"], r["baseline"]) for r in results}, key=lambda k: (list(TASKS).index(k[0])
                                                                                 if k[0] in TASKS else 99, k[1]))
    lines = ["| task | baseline | exploitability (mean / max) | distance (mean) | efficiency (mean) | "
             "converged | seconds |", "|---|---|---|---|---|---|---|"]
    for task, b in keys:
        rows = [r for r in results if r["task"] == task and r["baseline"] == b]
        ex = np.array([r["exploitability"] for r in rows])
        dist = [r["distance"] for r in rows if "distance" in r]
        eff = [r["efficiency"] for r in rows if "efficiency" in r]
        cells = [task, b, f"{ex.mean():.4f} / {ex.max():.4f}", f"{np.mean(dist):.4f}" if dist else "-",
                 f"{np.mean(eff):.3f}" if eff else "-",
                 f"{sum(r['converged'] for r in rows)}/{len(rows)}" if "converged" in rows[0] else "-",
                 f"{np.mean([r['seconds'] for r in rows]):.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    """``icgym-bench``: run baselines on the benchmark tasks and print a Markdown table."""
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Run baselines on the icgym benchmark tasks.")
    ap.add_argument("--tasks", default="all", help=f"comma-separated, or 'all': {', '.join(TASKS)}")
    ap.add_argument("--baselines", default="all", help=f"comma-separated, or 'all': {', '.join(BASELINES)}")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--list", action="store_true", help="list the tasks and exit")
    ap.add_argument("--json", help="write every run as JSON")
    args = ap.parse_args(argv)
    if args.list:
        for t in TASKS.values():
            print(f"{t.name:24s} [{t.expected}] {t.description}")
        return
    tasks = list(TASKS) if args.tasks == "all" else [x.strip() for x in args.tasks.split(",")]
    bases = list(BASELINES) if args.baselines == "all" else [x.strip() for x in args.baselines.split(",")]
    bad = [x for x in tasks if x not in TASKS] + [x for x in bases if x not in BASELINES]
    if bad:
        ap.error(f"unknown tasks or baselines: {bad}")
    res = run_suite(tasks, bases, seeds=range(args.seeds), progress=True)
    print()
    print(format_results(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=str))
