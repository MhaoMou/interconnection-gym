import itertools

import numpy as np
import pytest

from icgym import (
    LocationGame,
    Region,
    activsg2000,
    activsg2000_location,
    bundled,
    max_min_fair_point,
    random_region,
    reference,
)


# ------------------------------------------------------------------ bundled catalog
def test_catalog_is_bundled():
    names = bundled("activsg2000")
    assert sum(n.startswith("pairs/") for n in names) == 150
    assert sum(n.startswith("poa/") for n in names) == 35
    assert sum(n.startswith("location/") for n in names) == 8


def test_bundled_region_reorders_columns():
    a = activsg2000(("2068", "7344", "2027"))
    b = activsg2000(("2027", "2068", "7344"))
    assert np.allclose(max_min_fair_point(a)[[1, 2, 0]], max_min_fair_point(b))
    with pytest.raises(KeyError):
        activsg2000(("1", "2"))


def test_reference_poa_batches():
    ref = reference("activsg2000", "poa_implicit")["poa_batches"]
    assert len(ref) == 25
    assert all(q["poa"] <= q["bound"] + 1e-9 for q in ref)


def test_explicit_and_implicit_location_shares_agree():
    """The exported (downward-closed) regions give the shipped DCOPFRegion shares."""
    game = activsg2000_location()
    for name in bundled("activsg2000"):
        if not name.startswith("location/"):
            continue
        buses = tuple(name.split("buses_")[1].split("_"))
        assert np.allclose(max_min_fair_point(activsg2000(buses)), game.shares(buses), atol=1e-2)


# ------------------------------------------------------------------ the paper's location game
def test_activsg2000_location_game_matches_the_paper():
    game = activsg2000_location()
    eq = game.pure_equilibria()
    assert sorted(eq) == sorted(set(itertools.permutations(("7197", "5361", "5361"))))
    assert game.improving_moves() == {"moves": 1617, "violations": 0}
    rounds = [game.best_response_dynamics(start)[2] for start in game.profiles()]
    assert None not in rounds and max(rounds) == 2


# ------------------------------------------------------------------ the theory on random networks
@pytest.mark.parametrize("seed", range(6))
def test_improving_moves_raise_sorted_shares(seed):
    rng = np.random.default_rng(seed)
    region = random_region(5, 4, seed=seed, skew=2.0)
    mix = tuple(rng.dirichlet(np.ones(5)))
    menus = [list(rng.choice(5, 3, replace=False)) + ([mix] if i == 0 else []) for i in range(3)]
    game = LocationGame.from_bus_region(region, menus)
    assert game.improving_moves()["violations"] == 0
    assert game.pure_equilibria()
    for start in game.profiles():
        prof, path, rounds = game.best_response_dynamics(start)
        assert rounds is not None and game.is_equilibrium(prof)
        pots = [game.potential(p) for p in path]
        for p0, p1 in zip(pots, pots[1:]):
            d = p1 - p0
            nz = np.where(np.abs(d) > 1e-6)[0]
            assert nz.size and d[nz[0]] > 0


def test_single_limit_equilibria_load_the_limit_least():
    region = Region.single_limit([1.0, 0.5, 2.0, 0.5], 4.0)
    game = LocationGame.from_bus_region(region, [[0, 1, 2], [0, 2, 3], [2, 3]])
    eq = game.pure_equilibria()
    assert sorted(eq) == sorted(itertools.product([1], [3], [3]))
    w = game.shares(eq[0])
    assert np.allclose(w, 4.0 / 1.5)


# ------------------------------------------------------------------ environment
def test_location_env_api():
    pytest.importorskip("gymnasium")
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    from icgym.envs import LocationParallelEnv

    env = LocationParallelEnv(activsg2000_location(), rounds=4)
    parallel_api_test(env, num_cycles=8)
    env.reset()
    menu = env.game.menus[0]
    acts = {a: menu.index(b) for a, b in zip(env.possible_agents, ("7197", "5361", "5361"))}
    _, rew, _, _, info = env.step(acts)
    first = info[env.possible_agents[0]]
    assert first["is_equilibrium"] and first["exploitability"].max() == 0
    assert np.isclose(rew[env.possible_agents[1]] * env.game.scale, first["shares"][1])
