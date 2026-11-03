"""The paper's results as tests: if a rule or the engine changes, these say
whether the theory still describes it."""

import numpy as np
import pytest

from icgym import (
    BatchGame,
    FlatFee,
    PerMWCommitment,
    ProportionalRule,
    Region,
    UniformRule,
    UtilitarianRule,
    max_min_fair_point,
    poa_bound,
    poa_worst_case,
    predicted_equilibrium,
    price_of_anarchy,
    random_region,
    stages,
    water_fill,
)

V, KAPPA = 1.0, 0.4


def regions(count, n_max=4, seed=0):
    rng = np.random.default_rng(seed)
    for k in range(count):
        yield random_region(int(rng.integers(2, n_max + 1)), int(rng.integers(1, 5)), seed=seed * 1000 + k)


# ------------------------------------------------------------------ regions and water-filling
def test_downward_closed_subset_keeps_hosting_capacities():
    A = np.array([[1.0, -0.5], [0.2, 1.0]])
    b = np.array([1.0, 1.5])
    region = Region.downward_closed(A, b)
    assert np.all(region.A >= 0)
    h_plus = region.hosting_capacities()
    # hosting capacities of the original polytope along each axis
    h = [min(b[c] / A[c, k] for c in range(2) if A[c, k] > 0) for k in range(2)]
    assert np.allclose(h_plus, h)


@pytest.mark.parametrize("seed", range(3))
def test_water_filling_is_feasible_and_a_fixed_point(seed):
    for region in regions(30, seed=seed):
        rng = np.random.default_rng(seed)
        r = rng.random(region.n) * region.hosting_capacities() * 1.5
        s = water_fill(region, r)
        assert region.contains(s)
        assert np.all(s <= r + 1e-9)
        # an applicant served less than its request loads a tight limit
        tight = region.A @ s >= region.b - 1e-7
        for i in np.where(s < r - 1e-7)[0]:
            assert np.any(tight & (region.A[:, i] > 0))


def test_requests_at_least_w_are_served_w():
    for region in regions(30):
        w = max_min_fair_point(region)
        assert np.allclose(water_fill(region, w * 1.7 + 0.1), w)


# ------------------------------------------------------------------ equilibrium of the uniform rule
@pytest.mark.parametrize("beta", [0.0, 0.5, 0.9])
def test_max_min_fair_point_is_an_equilibrium(beta):
    for region in regions(6, n_max=3, seed=1):
        game = BatchGame(region, UniformRule(beta=beta), PerMWCommitment(KAPPA), V)
        w = max_min_fair_point(region)
        assert np.allclose(game.allocate(w), w)
        assert game.exploitability(w, method="search").max() < 1e-6


def test_best_responses_reach_w_within_two_rounds_per_stage():
    rng = np.random.default_rng(3)
    for region in regions(40, n_max=5, seed=2):
        game = BatchGame(region, UniformRule(), PerMWCommitment(KAPPA), V)
        w, st = max_min_fair_point(region, return_stages=True)
        K = len({round(s.level, 9) for s in st})
        r0 = rng.random(region.n) * region.hosting_capacities() * 2
        r, _, rounds = game.best_response_dynamics(r0, max_rounds=4 * region.n, order="random", seed=1)
        assert rounds is not None and rounds <= 2 * K
        assert np.allclose(r, w, atol=1e-7)


def test_stages_partition_the_applicants():
    region = random_region(5, 4, seed=7)
    blocked = [i for s in stages(region) for i in s.applicants]
    assert sorted(blocked) == list(range(region.n))


# ------------------------------------------------------------------ flat fee
def test_flat_fee_band():
    region = Region.single_limit([1.0, 1.5], 3.0)
    beta = 0.5
    game = BatchGame(region, UniformRule(beta=beta), FlatFee(0.1), V)
    pred = predicted_equilibrium(game)
    assert pred["kind"] == "band"
    w = pred["low"]
    inside = w * np.array([1.3, 1.9])
    assert np.allclose(game.allocate(inside), w)
    assert game.exploitability(inside, method="search").max() < 1e-6
    outside = w * np.array([0.8, 1.0])  # applicant 1 asks for less than its share
    assert game.exploitability(outside, method="search")[0] > 1e-3


# ------------------------------------------------------------------ proportional and weighted rules
@pytest.mark.parametrize("value,beta", [(3.0, 0.5), (1.5, 0.5), (3.0, 0.4), (6.0, 0.5)])
def test_proportional_rule_equilibrium(value, beta):
    region = Region.single_limit([1.0, 2.0], 1.0)
    game = BatchGame(region, ProportionalRule(beta=beta), PerMWCommitment(1.0), value)
    pred = predicted_equilibrium(game)
    r = pred["requests"]
    assert np.allclose(game.allocate(r), pred["served"])
    assert game.exploitability(r, method="search").max() < 1e-6


def test_proportional_rule_cut_by_mw_breaks_the_equilibrium():
    region = Region.single_limit([1.0, 2.0], 1.0)
    rule = ProportionalRule(beta=0.5, cut_order="largest_request")
    game = BatchGame(region, rule, PerMWCommitment(1.0), 3.0)
    r = predicted_equilibrium(BatchGame(region, ProportionalRule(beta=0.5), PerMWCommitment(1.0), 3.0))["requests"]
    gain = game.exploitability(r, method="search")
    assert gain[1] > 0.4  # applicant 2 over-requests to have applicant 1 dropped


def test_weighted_uniform_rule_matches_proportional_service_without_inflation():
    a = np.array([1.0, 2.0, 4.0])
    region = Region.single_limit(a, 1.0)
    game = BatchGame(region, UniformRule(weights=tuple(1 / a)), PerMWCommitment(KAPPA), V)
    w = max_min_fair_point(region, weights=1 / a)
    assert np.allclose(w, 1 / (3 * a))
    assert game.exploitability(w, method="search").max() < 1e-6


def test_utilitarian_rule_maximizes_total_but_invites_inflation():
    region = Region.single_limit([1.0, 2.0], 1.0)
    w = max_min_fair_point(region)
    rule = UtilitarianRule()
    s = rule.allocate(region, np.array([10.0, 10.0]))
    assert np.isclose(s.sum(), region.max_total()[0])
    game = BatchGame(region, rule, PerMWCommitment(KAPPA), V)
    assert game.exploitability(w, method="search")[0] > 1e-3


# ------------------------------------------------------------------ price of anarchy
def test_poa_bound_holds_on_random_regions():
    for region in regions(300, n_max=6, seed=4):
        w = max_min_fair_point(region)
        assert price_of_anarchy(region, w) <= poa_bound(region) * (1 + 1e-9)


@pytest.mark.parametrize("n,R", [(3, 1.0), (2, 1.5), (5, 3.0)])
def test_poa_bound_is_approached(n, R):
    region = poa_worst_case(n, R, eps=1e-6)
    poa = price_of_anarchy(region, max_min_fair_point(region))
    k = int(round((n * R - 1) / (2 * R)))
    assert np.isclose(poa, (n - k) * (k * R + 1) / n, rtol=1e-4)
    if np.isclose((n * R - 1) / (2 * R), k):
        assert np.isclose(poa, poa_bound(region), rtol=1e-4)
