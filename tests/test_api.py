"""The public API of 1.x. A failure here means a public name changed: update
CHANGELOG.md and bump the major version, or restore the name."""

import inspect

import icgym
from icgym import BatchGame, GenerationGame, LocationGame, NonFirmGame, Region
from icgym.devices import TwoPartGame

EXPORTS = {
    "AllocationRule", "BatchGame", "DCOPFRegion", "Device", "Fee", "FlatFee", "GenerationGame", "LocationGame",
    "NonFirmGame", "PerMWCommitment", "ProportionalRule", "Region", "SecurityDeposit", "UniformRule",
    "UtilitarianRule", "activsg2000", "activsg2000_location", "bundled", "bundled_region", "compare_rules",
    "flat_fee_example", "ieee14", "load_region", "load_region_record", "max_min_fair_point", "peak_theta",
    "poa_bound", "poa_worst_case", "predicted_equilibrium", "price_of_anarchy", "proportional_inflation",
    "random_region", "reference", "save_region", "stages", "water_fill", "water_fill_offset",
}

# every game answers the same questions with the same method names
GAME_METHODS = {"payoffs", "payoff", "best_response", "exploitability", "best_response_dynamics"}


def test_exports():
    assert set(icgym.__all__) == EXPORTS
    assert all(hasattr(icgym, name) for name in EXPORTS)


def test_games_share_one_vocabulary():
    for cls in (BatchGame, TwoPartGame, LocationGame):
        missing = {m for m in GAME_METHODS if not callable(getattr(cls, m, None))}
        assert not missing, (cls.__name__, missing)
    for cls in (BatchGame, GenerationGame, NonFirmGame):
        assert callable(cls.predicted_equilibrium)
    assert callable(TwoPartGame.hosting_capacity) and callable(Region.hosting_capacities)


def test_dynamics_return_the_same_triple():
    from icgym import Device, PerMWCommitment, UniformRule

    region = Region.single_limit([1.0, 2.0], 4.0)
    out = BatchGame(region, UniformRule(), PerMWCommitment(0.4)).best_response_dynamics([0.5, 0.5])
    assert len(out) == 3 and out[2] is not None
    gen = GenerationGame(region, Device(0.5, 0.5, 0.35), rule="credit")
    X, history, rounds = gen.best_response_dynamics(gen.predicted_equilibrium()["strategies"])
    assert rounds == 0 and len(history) == 2
    loc = LocationGame.from_bus_region(Region.single_limit([1.0, 2.0], 4.0), [[0, 1], [0, 1]])
    prof, path, rounds = loc.best_response_dynamics((1, 1))
    assert prof == (0, 0) and rounds is not None


def test_environment_keywords():
    pytest = __import__("pytest")
    pytest.importorskip("pettingzoo")
    from icgym import envs

    for cls in (envs.BatchRequestParallelEnv, envs.DeviceParallelEnv, envs.LocationParallelEnv):
        params = inspect.signature(cls).parameters
        assert {"game", "rounds", "normalize_rewards", "compute_exploitability"} <= set(params), cls.__name__
