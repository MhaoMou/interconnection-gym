"""Devices dispatched over the network's days, and energy at the LMP, on a small ring network."""

import numpy as np
import pytest
from test_network import write_network

from icgym import BatchGame, DCOPFRegion, PerMWCommitment, UniformRule, max_min_fair_point
from icgym.game import grid_equilibria
from icgym.network import DCNetwork
from icgym.network_devices import (
    DeviceRegion,
    NetworkGenerationGame,
    NetworkNonFirmGame,
    curtailment_scale,
    peak_periods,
    with_added_load,
)
from icgym.prices import EnergyCharge, FeeSum, lmp


@pytest.fixture
def net(tmp_path):
    # two days of four periods, with a daily peak
    rows = [(6, 4, 5), (10, 7, 8), (14, 9, 12), (8, 5, 6)] * 2
    write_network(tmp_path, rows)
    return DCNetwork.from_csv_dir(tmp_path)


def test_no_devices_is_the_plain_region(net):
    for buses in (["2", "3"], ["2", "3", "4"]):
        a = max_min_fair_point(DCOPFRegion(net, buses))
        b = max_min_fair_point(DeviceRegion(net, buses, day_length=4))
        assert np.allclose(a, b, atol=1e-6)
        assert np.allclose(DCOPFRegion(net, buses).hosting_capacities(),
                           DeviceRegion(net, buses, day_length=4).hosting_capacities(), atol=1e-6)


def test_generator_budget_and_unlocking(net):
    base = DeviceRegion(net, ["3"], day_length=4).hosting_capacities()[0]
    none = DeviceRegion(net, ["3"], generators=[10.0], run_hours=0.0, day_length=4).hosting_capacities()[0]
    full = DeviceRegion(net, ["3"], generators=[10.0], run_hours=None, day_length=4).hosting_capacities()[0]
    one = DeviceRegion(net, ["3"], generators=[10.0], run_hours=1.0, day_length=4).hosting_capacities()[0]
    assert np.isclose(none, base, atol=1e-6)
    assert base < one < full + 1e-9 and np.isclose(full, base + 10.0, atol=1e-6)
    R = DeviceRegion(net, ["3"], generators=[10.0], run_hours=1.0, day_length=4)
    d = R.dispatch([one])
    per_day = d["generation"].reshape(2, 4, 1).sum(axis=1)
    assert np.all(per_day <= 10.0 * 1.0 + 1e-6) and np.all(d["generation"] <= 10.0 + 1e-9)
    assert R.contains([one]) and not R.contains([one + 1.0])


def test_storage_and_curtailment_relieve_the_peak(net):
    base = DeviceRegion(net, ["3"], day_length=4).hosting_capacities()[0]
    bat = DeviceRegion(net, ["3"], storage_power=[5.0], storage_hours=1.0, day_length=4).hosting_capacities()[0]
    assert bat >= base - 1e-9
    peak = peak_periods(net, 1, day_length=4)
    assert peak == [2, 6]
    cut = DeviceRegion(net, ["3"], load_scale=curtailment_scale(net, 1, peak, [0.5]), day_length=4)
    assert cut.hosting_capacities()[0] >= base - 1e-9


def test_added_load_fixes_a_tier(net):
    a = DCOPFRegion(net, ["2", "3"])
    w = max_min_fair_point(a)
    rest = DeviceRegion(with_added_load(net, ["2", "3"], w), ["2", "3"], day_length=4)
    assert np.allclose(rest.hosting_capacities(), 0.0, atol=1e-5)


def test_network_games(net):
    gen = NetworkGenerationGame(net, ["2", "3"], value=1.0, kappa=0.4, gamma=0.5, run_hours=1.0, capital=0.05,
                                rule="gross", day_length=4)
    w = max_min_fair_point(gen.base_region)
    assert np.allclose(gen.allocate([[w[0], 0.0], [w[1], 0.0]]), w, atol=1e-6)
    assert gen.unlocking_ratio(0, 5.0) >= 0
    credit = NetworkGenerationGame(net, ["2", "3"], value=1.0, kappa=0.4, rule="credit", day_length=4)
    assert np.all(credit.credits([2.0, 2.0]) >= 0)
    peak = peak_periods(net, 1, day_length=4)
    nf = NetworkNonFirmGame(net, ["2", "3"], peak, value=1.0, peak_value=0.2, kappa=0.4, day_length=4)
    p = nf.predicted_equilibrium()
    assert np.allclose(nf.allocate(p["strategies"]), p["strategies"], atol=1e-5)


def test_lmp_and_energy_charge(net):
    # uncongested at zero new load: one price, the cheapest unit's cost (both units cost 0 here)
    assert np.allclose(lmp(net, ["2"], [0.0], 0), lmp(net, ["2"], [0.0], 0)[0])
    weights = np.ones(net.periods)
    fee = FeeSum((PerMWCommitment(0.4), EnergyCharge(net, ["2", "3"], weights, share=1e-3)))
    R = DCOPFRegion(net, ["2", "3"])
    game = BatchGame(R, UniformRule(), fee, value=10.0)
    assert game.predicted_equilibrium()["kind"] == "epsilon"
    w = max_min_fair_point(R)
    assert np.all(np.isfinite(game.payoffs(w)))


def test_grid_equilibria_recover_w():
    from icgym import Region
    R = Region.single_limit([1.0, 1.0], 2.0)
    game = BatchGame(R, UniformRule(), PerMWCommitment(0.4), value=1.0)
    grid = [np.linspace(0, 2, 9)] * 2
    eqs, table = grid_equilibria(game, grid)
    assert [list(map(float, e)) for e in eqs] == [[1.0, 1.0]] and len(table) == 81
