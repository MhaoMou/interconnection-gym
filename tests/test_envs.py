import numpy as np
import pytest

from icgym import BatchGame, PerMWCommitment, UniformRule, load_region, max_min_fair_point, random_region, save_region

gym = pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")

from gymnasium.utils.env_checker import check_env  # noqa: E402
from pettingzoo.test import parallel_api_test  # noqa: E402

from icgym.envs import BatchRequestParallelEnv, SingleApplicantEnv  # noqa: E402


def make_game(seed=0):
    return BatchGame(random_region(3, 3, seed=seed), UniformRule(beta=0.5), PerMWCommitment(0.4), 1.0)


def test_parallel_api():
    parallel_api_test(BatchRequestParallelEnv(make_game(), rounds=5, observation="public"), num_cycles=10)


def test_single_agent_api():
    check_env(SingleApplicantEnv(make_game(), rounds=3), skip_render_check=True)


def test_equilibrium_actions_earn_the_equilibrium_payoff():
    game = make_game(1)
    env = BatchRequestParallelEnv(game, rounds=1, normalize_rewards=False, compute_exploitability=True)
    env.reset()
    w = max_min_fair_point(game.region)
    cap = env.request_cap * game.region.hosting_capacities()
    actions = {a: np.array([w[i] / cap[i]], dtype=np.float32) for i, a in enumerate(env.possible_agents)}
    _, rewards, terms, _, info = env.step(actions)
    assert all(terms.values())
    got = np.array([rewards[a] for a in env.possible_agents])
    assert np.allclose(got, (1.0 - 0.4) * w, atol=1e-5)
    assert info[env.possible_agents[0]]["exploitability"].max() < 1e-4


def test_region_json_round_trip(tmp_path):
    region = random_region(4, 3, seed=5)
    path = tmp_path / "region.json"
    save_region(region, path, source="test")
    back = load_region(path)
    assert np.allclose(back.A, region.A) and np.allclose(back.b, region.b)
