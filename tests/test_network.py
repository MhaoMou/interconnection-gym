"""Exported regions agree with the DCOPF they come from."""

import numpy as np
import pytest

from icgym import max_min_fair_point, price_of_anarchy
from icgym.network import DCNetwork, export_region
from icgym.scenarios import ieee14, load_region_record


def write_network(root, demand_rows):
    """Four buses: a generator at bus 1, loads elsewhere, a meshed ring."""
    (root / "bus.csv").write_text("bus_id,type\n1,3\n2,1\n3,1\n4,1\n")
    (root / "branch_limits.csv").write_text(
        "branch_id,from_bus_id,to_bus_id,reactance_pu,thermal_limit_mva,status\n"
        "a,1,2,0.1,60,1\nb,2,3,0.2,25,1\nc,3,4,0.1,40,1\nd,4,1,0.3,50,1\ne,1,3,0.25,30,1\n")
    (root / "generator_limits.csv").write_text(
        "generator_id,bus_id,p_min_mw,p_max_mw,marginal_cost_c1_per_mwh,status,profile_type\n"
        "g1,1,0,200,0,1,thermal\ng2,4,0,30,0,1,thermal\n")
    lines = ["period,1,2,3,4"] + [f"{t},0,{d[0]},{d[1]},{d[2]}" for t, d in enumerate(demand_rows)]
    (root / "demand_bus.csv").write_text("\n".join(lines) + "\n")
    (root / "dcline.csv").write_text("status,from_bus_id,to_bus_id,Pt\n")


@pytest.fixture
def net(tmp_path):
    write_network(tmp_path, [(10, 5, 8), (14, 7, 3), (6, 9, 12)])
    return DCNetwork.from_csv_dir(tmp_path)


def feasible(net, buses, x):
    return all(net.period_problem(t, buses).feasible(x) for t in range(net.periods))


@pytest.mark.parametrize("buses", [("2", "3"), ("2", "4"), ("2", "3", "4")])
def test_exact_region_matches_dcopf_feasibility(net, buses):
    ex = export_region(net, buses)
    h = ex.region.hosting_capacities()
    rng = np.random.default_rng(0)
    for _ in range(150):
        x = rng.random(len(buses)) * h * 1.3
        inside = bool(np.all(ex.signed_A @ x <= ex.signed_b + 1e-6))
        margin = np.min(ex.signed_b - ex.signed_A @ x)
        if abs(margin) < 1e-4:
            continue  # skip points on the boundary
        assert inside == feasible(net, buses, x)


def test_downward_closed_subset_keeps_hosting_capacities(net):
    ex = export_region(net, ("2", "3", "4"))
    for i in range(3):
        caps = [net.period_problem(t, [("2", "3", "4")[i]]).support([1.0])[0] for t in range(net.periods)]
        assert np.isclose(ex.region.hosting_capacities()[i], min(caps), rtol=1e-7)


# ------------------------------------------------------------------ bundled IEEE 14-bus regions
PAPER = {  # max-min fair point (MW) and price of anarchy reported in the paper
    (1, 2): ([235.34, 235.34], 1.0),
    (13, 14): ([25.40, 25.40], 1.0),
    (1, 2, 4): ([156.89, 156.89, 156.89], 1.0),
    (9, 13, 14): ([46.78, 25.40, 25.40], 1.239),
}


@pytest.mark.parametrize("buses", list(PAPER))
def test_ieee14_regions_reproduce_the_paper(buses):
    region = ieee14(buses)
    w = max_min_fair_point(region)
    assert np.allclose(w, PAPER[buses][0], atol=0.01)
    assert np.isclose(price_of_anarchy(region, w), PAPER[buses][1], atol=1e-3)


SCENARIO_SHA256 = {  # the locally built IEEE 14-bus scenario, byte for byte
    "bus.csv": "1b620245ac8b9946cfa60a8de248655d04e35c8b80590f4268471d222888f0f5",
    "branch_limits.csv": "f64146f6d50f8ed6d5abac1105132e406a92ba2f97fd2b0e255cda1320182fbf",
    "generator_limits.csv": "9042bf7641f0ca4f321d449b6ed7edcb57bdc987793767f1c046ba8fc84f51c3",
    "demand_bus.csv": "2df66cc30aadbf7da5b113fd0fa330e940461d3a624eb3f760a85dd33afca96f",
    "dcline.csv": "843b38d7bef71ea519f7734f5af67f688b337423145c4fe6210a6ec0e3f78bde",
}


def test_ieee14_scenario_is_rebuilt_exactly():
    import hashlib

    from icgym.datasets import fetch_ieee14
    root = fetch_ieee14(quiet=True)
    for name, digest in SCENARIO_SHA256.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name


def test_ieee14_counterflow_pairs():
    from icgym.datasets import data_dir
    for buses, expected in [((2, 5), True), ((4, 5), True), ((1, 2), False)]:
        ieee14(buses)
        rec = load_region_record(data_dir() / "ieee14" / "regions" / f"buses_{buses[0]}_{buses[1]}.json")
        assert bool(np.any(rec["signed_A"] < -1e-9)) is expected
        assert rec["metadata"]["has_counterflow"] is expected
        assert "/" not in rec["metadata"]["source"]
    assert ieee14((14, 13)).applicant_names == ieee14((13, 14)).applicant_names[::-1]


# ------------------------------------------------------------------ implicit DCOPF region
@pytest.mark.parametrize("buses", [("2", "3"), ("2", "3", "4")])
def test_implicit_region_agrees_with_the_exact_export(net, buses):
    from icgym import DCOPFRegion, ProportionalRule, UniformRule, UtilitarianRule
    ex = export_region(net, buses)
    D = DCOPFRegion(net, buses)
    assert np.allclose(D.hosting_capacities(), ex.region.hosting_capacities(), rtol=1e-6)
    signed_best = linprog_total(ex.signed_A, ex.signed_b)
    assert np.isclose(D.max_total()[0], signed_best, rtol=1e-6)
    rng = np.random.default_rng(1)
    for _ in range(4):
        r = rng.random(len(buses)) * ex.region.hosting_capacities() * 1.3
        for rule in (UniformRule(beta=0.4), ProportionalRule(cut_order="largest_request"), UtilitarianRule()):
            s = rule.allocate(D, r)
            assert np.all(ex.signed_A @ s <= ex.signed_b + 1e-5)   # inside the exact region
            assert np.all(s <= r + 1e-7)
            if not ex.has_counterflow:
                assert np.allclose(s, rule.allocate(ex.region, r), atol=1e-5)


def linprog_total(A, b):
    from scipy.optimize import linprog
    res = linprog(-np.ones(A.shape[1]), A_ub=A, b_ub=b, bounds=[(0, None)] * A.shape[1], method="highs")
    return -res.fun


def test_location_game_on_a_network(net):
    """Location game with exact DCOPF regions agrees with the exported regions."""
    from icgym import LocationGame
    game = LocationGame.from_network(net, ["2", "3", "4"], n=2)
    assert game.improving_moves()["violations"] == 0
    assert game.pure_equilibria()
    for prof in game.profiles():
        ex = export_region(net, prof)
        if not ex.has_counterflow:
            assert np.allclose(game.shares(prof), max_min_fair_point(ex.region), atol=1e-4)
    assert game.evaluations == 6           # one per multiset of placements


def test_ieee14_first_use_in_parallel(tmp_path, monkeypatch):
    """Several processes building the 14-bus data at once must not trip over each other."""
    import multiprocessing as mp

    from icgym.datasets import data_dir
    src = data_dir() / "ieee14" / "case14.m"
    ieee14((1, 2))                                   # make sure a verified case file exists to copy
    monkeypatch.setenv("ICGYM_DATA_DIR", str(tmp_path))
    (tmp_path / "ieee14").mkdir(parents=True)
    (tmp_path / "ieee14" / "case14.m").write_bytes(src.read_bytes())
    with mp.get_context("spawn").Pool(4) as pool:
        out = pool.map(_first_use, [(13, 14)] * 4)
    assert all(np.allclose(o, out[0]) for o in out)


def _first_use(buses):
    return max_min_fair_point(ieee14(buses))
