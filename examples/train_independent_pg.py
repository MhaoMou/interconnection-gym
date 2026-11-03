"""Independent learners in the PettingZoo environment, scored against the theory.

Each applicant runs its own policy gradient (a Gaussian over its request,
updated from its rewards only) in ``BatchRequestParallelEnv``. The theory says
the unique equilibrium is the max-min fair point ``w``; the script prints how
close the learners get and how much any of them could still gain by deviating.

    pip install "interconnection-gym[envs]"
    python examples/train_independent_pg.py [iterations]
"""

import sys

import numpy as np

from icgym import BatchGame, PerMWCommitment, UniformRule, ieee14, max_min_fair_point
from icgym.envs import BatchRequestParallelEnv

iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 300
game = BatchGame(ieee14((9, 13, 14)), UniformRule(beta=0.5), PerMWCommitment(0.4))
env = BatchRequestParallelEnv(game, rounds=1)          # one batch per episode
agents = env.possible_agents
rng = np.random.default_rng(0)
mu = {a: 0.0 for a in agents}                          # policy: action = sigmoid(mu + sigma * eps)
log_sigma = {a: 0.0 for a in agents}
batch = 64

for it in range(iterations):
    lr = 0.3 * (1 - it / iterations) + 0.02
    eps = {a: rng.standard_normal(batch) for a in agents}
    rewards = {a: np.empty(batch) for a in agents}
    for k in range(batch):
        env.reset(seed=it * batch + k)
        actions = {a: np.array([1 / (1 + np.exp(-(mu[a] + np.exp(log_sigma[a]) * eps[a][k])))], dtype=np.float32)
                   for a in agents}
        _, rew, *_ = env.step(actions)
        for a in agents:
            rewards[a][k] = rew[a]
    for a in agents:                                   # natural-gradient step, from own rewards only
        adv = (rewards[a] - rewards[a].mean()) / (rewards[a].std() + 1e-8)
        mu[a] += lr * np.exp(log_sigma[a]) * np.mean(adv * eps[a])
        log_sigma[a] = float(np.clip(log_sigma[a] + 0.5 * lr * np.mean(adv * (eps[a] ** 2 - 1)), -6, 1))

h = game.region.hosting_capacities()
learned = np.array([1 / (1 + np.exp(-mu[a])) for a in agents]) * env.request_cap * h
w = max_min_fair_point(game.region)
print("learned requests (MW):", learned.round(2))
print("theory (max-min fair point):", w.round(2))
print("largest gap over hosting capacity:", float(np.max(np.abs(learned - w) / h)).__round__(4))
print("exploitability over V h:", (game.exploitability(learned) / h).round(4))
