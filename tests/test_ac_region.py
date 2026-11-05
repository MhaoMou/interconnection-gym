"""The AC batch region (needs the 'ac' extra)."""

import numpy as np
import pytest

pytest.importorskip("casadi")

from icgym import ACNetwork, ACOPFRegion, BatchGame, DCOPFRegion, PerMWCommitment, UniformRule, downward_closure_check
from icgym.datasets import ieee14_ac_network, ieee14_network

CASE3 = """function mpc = case3
mpc.baseMVA = 100;
mpc.bus = [
 1 3 0  0  0 0 1 1 0 230 1 1.05 0.95;
 2 2 20 5  0 0 1 1 0 230 1 1.05 0.95;
 3 1 40 10 0 0 1 1 0 230 1 1.05 0.95;
];
mpc.gen = [
 1 0 0 100 -100 1 100 1 200 0 0 0 0 0 0 0 0 0 0 0 0;
 2 0 0 50 -50 1 100 1 100 0 0 0 0 0 0 0 0 0 0 0 0;
];
mpc.branch = [
 1 2 0.01 0.1 0.02 120 0 0 0 0 1 -360 360;
 1 3 0.01 0.1 0.02 100 0 0 0 0 1 -360 360;
 2 3 0.01 0.1 0.02 60 0 0 0 0 1 -360 360;
];
"""


@pytest.fixture(scope="module")
def case3(tmp_path_factory):
    path = tmp_path_factory.mktemp("ac") / "case3.m"
    path.write_text(CASE3)
    return ACNetwork.from_matpower(path, period_multipliers=[0.8, 1.0, 0.6])


def test_points_satisfy_the_power_flow_equations(case3):
    R = ACOPFRegion(case3, ["2", "3"], power_factor=0.95)
    w = R.water_fill(R.hosting_capacities() * 2)
    for t in range(case3.periods):
        op = R.operating_point(w * 0.999, t)
        pd, qd = case3.pd[t].copy(), case3.qd[t].copy()
        tan = np.sqrt(1 - 0.95 ** 2) / 0.95
        pd[1:] += w * 0.999
        qd[1:] += w * 0.999 * tan
        assert np.abs(case3.mismatch(op["vm"], op["va"], op["pg"], op["qg"], pd, qd)).max() < 1e-7
        assert np.all(op["vm"] <= case3.vmax + 1e-7) and np.all(op["vm"] >= case3.vmin - 1e-7)
        assert np.all(np.maximum(op["sf_mva"], op["st_mva"]) <= case3.rate + 1e-5)


def test_water_filling_ends_on_the_boundary(case3):
    R = ACOPFRegion(case3, ["2", "3"], power_factor=1.0)
    h = R.hosting_capacities()
    w = R.water_fill(h * 2)
    assert R.contains(w)
    assert 2 in R.binding           # the peak period binds
    for i in range(2):
        e = np.eye(2)[i]
        if w[i] < h[i] - 1e-6:      # an applicant stopped below its own capacity is blocked
            assert not R.contains(w + 1e-3 * e)
    assert R.max_total()[0] >= w.sum() - 1e-6


def test_reactive_power_lowers_hosting_capacity(case3):
    h1 = ACOPFRegion(case3, ["3"], power_factor=1.0).hosting_capacities()[0]
    h9 = ACOPFRegion(case3, ["3"], power_factor=0.9).hosting_capacities()[0]
    assert h9 < h1


def test_rules_and_games_run_on_the_ac_region(case3):
    R = ACOPFRegion(case3, ["2", "3"], power_factor=0.95)
    w = R.water_fill(R.hosting_capacities() * 2)
    game = BatchGame(R, UniformRule(), PerMWCommitment(0.4), value=1.0)
    assert np.allclose(game.allocate(w), w, atol=1e-6)
    assert downward_closure_check(R, samples=2)["violations"] == []


def test_ieee14_ac_is_below_dc_and_matches_its_data():
    ac = ieee14_ac_network()
    assert ac.periods == ieee14_network().periods == 120
    A = ACOPFRegion(ac, ["9", "13"], power_factor=1.0)
    D = DCOPFRegion(ieee14_network(), ["9", "13"])
    ha, hd = A.hosting_capacities(), D.hosting_capacities()
    assert np.all(ha < hd)                       # losses and reactive flows use up capacity
    assert np.all(ha > 0.85 * hd)
