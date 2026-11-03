import numpy as np
import pytest

from icgym import (
    BatchGame,
    FlatFee,
    PerMWCommitment,
    ProportionalRule,
    UniformRule,
    UtilitarianRule,
    ieee14,
    random_region,
    water_fill,
)
from icgym.vector import VectorBatchRequestEnv, allocate_batch, payoffs_batch, water_fill_batch


@pytest.mark.parametrize("seed", range(5))
def test_batched_allocation_equals_the_loop(seed):
    rng = np.random.default_rng(seed)
    region = random_region(6, 5, seed=seed)
    h = region.hosting_capacities()
    R = rng.random((200, 6)) * 2 * h * (rng.random((200, 6)) > 0.1)
    assert np.allclose(water_fill_batch(region, R), [water_fill(region, r) for r in R], atol=1e-10)
    w = rng.random(6) + 0.5
    assert np.allclose(water_fill_batch(region, R, weights=w), [water_fill(region, r, weights=w) for r in R],
                       atol=1e-10)
    for rule in (UniformRule(beta=0.6), UniformRule(beta=0.6, cut_order="largest_loading"),
                 ProportionalRule(beta=0.5), ProportionalRule(beta=0.5, cut_order="largest_request"),
                 UtilitarianRule(beta=0.3)):
        assert np.allclose(allocate_batch(rule, region, R[:40]), [rule.allocate(region, r) for r in R[:40]],
                           atol=1e-9)


def test_payoffs_batch():
    game = BatchGame(ieee14((9, 13, 14)), UniformRule(beta=0.5), PerMWCommitment(0.4))
    R = np.random.default_rng(0).random((30, 3)) * 100
    assert np.allclose(payoffs_batch(game, R), [game.payoffs(r) for r in R])
    flat = BatchGame(ieee14((1, 2)), UniformRule(beta=0.5), FlatFee(3.0))
    R = np.random.default_rng(1).random((30, 2)) * 400
    assert np.allclose(payoffs_batch(flat, R), [flat.payoffs(r) for r in R])


def test_vector_env_matches_the_parallel_env():
    pytest.importorskip("gymnasium")
    pytest.importorskip("pettingzoo")
    from icgym.envs import BatchRequestParallelEnv

    game = BatchGame(random_region(4, 3, seed=2), UniformRule(beta=0.5), PerMWCommitment(0.4))
    venv = VectorBatchRequestEnv(game, num_envs=16, rounds=2, observation="public")
    env = BatchRequestParallelEnv(game, rounds=2, observation="public")
    rng = np.random.default_rng(0)
    venv.reset()
    A1, A2 = rng.random((16, 4)), rng.random((16, 4))
    o1, r1, d1, _ = venv.step(A1)
    o2, r2, d2, _ = venv.step(A2)
    assert not d1.any() and d2.all() and np.allclose(o2[:, :, :2], 0)   # auto-reset
    for e in (0, 7):
        env.reset()
        obs, rew, *_ = env.step({a: A1[e, i:i + 1] for i, a in enumerate(env.possible_agents)})
        assert np.allclose(r1[e], [rew[a] for a in env.possible_agents])
        assert np.allclose(o1[e], [obs[a] for a in env.possible_agents], atol=1e-6)
        _, rew, *_ = env.step({a: A2[e, i:i + 1] for i, a in enumerate(env.possible_agents)})
        assert np.allclose(r2[e], [rew[a] for a in env.possible_agents])
