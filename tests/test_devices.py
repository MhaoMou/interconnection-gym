import numpy as np
import pytest

from icgym import Region, random_region, water_fill
from icgym.devices import Device, GenerationGame, NonFirmGame, peak_theta, water_fill_offset

LIMIT = Region.single_limit([1.0, 1.5, 2.0], 6.0)


def gen(cost, rule="gross", region=LIMIT, **kw):
    # theta = 0.5 (2 run hours over 4 peak hours), gamma = 0.5, V = 1, kappa_C = 0.4
    return GenerationGame(region, Device(peak_theta(2, 4), 0.5, cost), value=1.0, kappa=0.4, rule=rule, **kw)


def scale(game):
    return float(np.max(np.abs(game.payoffs(game.predicted_equilibrium()["X"])))) + 1.0


# ------------------------------------------------------------------ water-filling with offsets
@pytest.mark.parametrize("seed", range(8))
def test_offset_water_filling(seed):
    rng = np.random.default_rng(seed)
    region = random_region(4, 3, seed=seed)
    r = rng.random(4) * 2 * region.hosting_capacities()
    assert np.allclose(water_fill_offset(region, r, np.zeros(4)), water_fill(region, r), atol=1e-9)
    o = rng.random(4) * region.hosting_capacities() * 0.5
    s = water_fill_offset(region, r, o)
    load = lambda x: region.A @ np.maximum(x - o, 0.0)  # noqa: E731
    assert np.all(load(s) <= region.b + 1e-9) and np.all(s <= r + 1e-12)
    for i in range(4):  # every applicant is at its request or blocked
        if s[i] < r[i] - 1e-9:
            x = s.copy()
            x[i] += 1e-6
            assert np.any(load(x) > region.b + 1e-12)


# ------------------------------------------------------------------ on-site generation
@pytest.mark.parametrize("seed", range(4))
def test_requests_given_generators(seed):
    """Part 1: with generators fixed, w(G) is the unique equilibrium of the requests."""
    rng = np.random.default_rng(seed)
    region = random_region(3, 3, seed=seed)
    game = gen(0.1, region=region, enforce_cap=False)
    G = rng.random(3) * region.hosting_capacities()
    w = game.shares_given_devices(G)
    X = np.column_stack([w, G])
    assert np.allclose(game.allocate(X), w)
    assert game.exploitability(X, fix_second=True).max() < 1e-6 * (1 + w.max())
    # more generation raises the sorted shares lexicographically
    w2 = game.shares_given_devices(G + rng.random(3))
    d = np.sort(w2) - np.sort(w)
    nz = np.where(np.abs(d) > 1e-9)[0]
    assert nz.size == 0 or d[nz[0]] > 0


def test_three_regimes_on_one_limit():
    none = gen(0.35)                       # mu+ < 0: no generation
    assert none.builder_margin() < 0 and none.no_carry()
    p = none.predicted_equilibrium()
    assert np.allclose(p["X"][:, 1], 0) and np.allclose(p["served"], 6.0 / 4.5)
    assert none.exploitability(p["X"]).max() < 1e-6

    cap = gen(0.03)                        # mu-_i > 0 for all: everyone at the cap
    assert np.all(cap.scrapper_margins() > 0)
    p = cap.predicted_equilibrium()
    assert np.allclose(p["served"], 6.0 / (0.75 * 4.5)) and np.allclose(cap.allocate(p["X"]), p["served"])
    assert cap.exploitability(p["X"]).max() < 1e-6

    mid = gen(0.15)                        # mu+ > 0 > mu-_i: no pure equilibrium
    assert mid.predicted_equilibrium()["kind"] == "none"
    for X in (none.predicted_equilibrium()["X"], cap.predicted_equilibrium()["X"]):
        assert mid.exploitability(X).max() > 1e-3


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("cost", [0.03, 0.35])
def test_crediting_rule_has_the_predicted_equilibrium(seed, cost):
    """Part 3: crediting generators to their owners, on any region."""
    game = gen(cost, rule="credit", region=random_region(3, 3, seed=seed))
    p = game.predicted_equilibrium()
    assert p["known"] and np.allclose(game.allocate(p["X"]), p["X"][:, 0])
    assert game.exploitability(p["X"]).max() < 1e-6 * scale(game)


def test_storage_device():
    bat = Device.storage(duration=4, nu=8, gamma=0.5, power_cost=0.01, energy_cost=0.005)
    assert np.isclose(bat.theta, 0.5) and np.isclose(bat.cost, 0.03)
    game = GenerationGame(random_region(3, 2, seed=5), bat, rule="credit")
    p = game.predicted_equilibrium()
    assert np.all(p["X"][:, 1] > 0)
    assert game.exploitability(p["X"]).max() < 1e-6 * scale(game)


# ------------------------------------------------------------------ non-firm service
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("peak_value", [0.2, 0.7])
def test_two_tier_rule(seed, peak_value):
    peak = random_region(3, 3, seed=seed)
    off = Region(peak.A, peak.b * 1.6)
    game = NonFirmGame(peak, off, value=1.0, peak_value=peak_value, kappa=0.4, rule="two_tier")
    p = game.predicted_equilibrium()
    assert np.allclose(game.allocate(p["X"]), p["X"])      # every request served in full
    assert (p["X"][:, 1].sum() > 0) == (game.offpeak_value > game.kappa)
    assert game.exploitability(p["X"]).max() < 1e-6 * scale(game)


def test_gross_nonfirm_regimes():
    off = Region.single_limit([1.0, 1.5, 2.0], 9.0)
    make = lambda vp: NonFirmGame(LIMIT, off, value=1.0, peak_value=vp, kappa=0.4, rule="gross", gamma=0.5)  # noqa: E731
    none, cut, mid = make(0.7), make(0.1), make(0.3)
    p = none.predicted_equilibrium()
    assert np.allclose(p["X"][:, 1], 0) and none.exploitability(p["X"]).max() < 1e-6
    p = cut.predicted_equilibrium()
    assert np.allclose(p["X"][:, 1], 1 / 3) and np.allclose(cut.allocate(p["X"]), p["X"][:, 0])
    assert cut.exploitability(p["X"]).max() < 1e-6
    assert mid.predicted_equilibrium()["kind"] == "none"
    for X in (none.predicted_equilibrium()["X"], cut.predicted_equilibrium()["X"]):
        assert mid.exploitability(X).max() > 1e-3


@pytest.mark.parametrize("make", [
    lambda: gen(0.03),
    lambda: gen(0.03, rule="credit", region=random_region(3, 3, seed=1)),
    lambda: NonFirmGame(LIMIT, Region.single_limit([1.0, 1.5, 2.0], 9.0), rule="gross", peak_value=0.1),
    lambda: NonFirmGame(LIMIT, Region.single_limit([1.0, 1.5, 2.0], 9.0), rule="two_tier"),
])
def test_device_env(make):
    pytest.importorskip("gymnasium")
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    from icgym.envs import DeviceParallelEnv

    game = make()
    parallel_api_test(DeviceParallelEnv(game, rounds=3, observation="public"), num_cycles=6)
    env = DeviceParallelEnv(game, rounds=1, normalize_rewards=False)
    env.reset()
    X = env._prediction["X"]
    acts = {}
    for i, a in enumerate(env.possible_agents):      # invert the action map at the prediction
        x1 = X[i, 0] / (env.request_cap * env._h[i])
        x2 = game.strategy_from_action(i, [x1, 1.0], env.request_cap)[1]
        acts[a] = np.array([x1, X[i, 1] / x2 if x2 > 0 else 0.0], dtype=np.float32)
    _, rew, _, _, _ = env.step(acts)
    assert np.allclose([rew[a] for a in env.possible_agents], game.payoffs(X), rtol=1e-5, atol=1e-6)
