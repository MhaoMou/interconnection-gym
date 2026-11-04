import numpy as np
import pytest

from icgym import PerMWCommitment, ProportionalRule, Region, UniformRule, random_region
from icgym.sequential import Applicant, ArrivalProcess, SequentialBatchGame


def make_game(**kw):
    region = random_region(5, 4, seed=3)          # columns are buses
    arrivals = ArrivalProcess(menu=[0, 1, 2, 3, 4], rate=2.0, size=(0.2, 0.6), value=(1.0, 1.5))
    return SequentialBatchGame(region, arrivals, batches=6, fee=PerMWCommitment(0.4), **kw)


def test_equilibrium_policy_is_served_in_full():
    game = make_game()
    out = game.simulate(seed=0)
    assert out["batches"] == 6 and out["applicants"] > 0
    assert out["unserved_requested_mw"] < 1e-6 and out["max_gap_to_batch_equilibrium"] < 1e-9
    for h in out["history"]:          # the residual capacity never goes negative and only shrinks
        assert h.residual_capacity >= -1e-9
    caps = [h.residual_capacity for h in out["history"]]
    assert all(a >= b - 1e-9 for a, b in zip(caps, caps[1:]))


def test_over_requesting_wastes_commitment():
    game = make_game(rule=ProportionalRule())
    out = game.simulate(policy=lambda g: 2.0 * np.array([a.need for a in g.queue]), seed=0)
    assert out["unserved_requested_mw"] > 0


def test_growth_and_waiting():
    region = Region.single_limit([1.0], 1.0)
    first = [Applicant(ident=0, bus=0, need=2.0, value=1.0, arrived=0)]
    game = SequentialBatchGame(region, ArrivalProcess(menu=[0], rate=0.0), batches=3, initial=first,
                               growth=np.array([0.1]), wait_cost=0.01, max_tries=3)
    out = game.simulate(seed=0)
    h = out["history"]
    assert np.isclose(h[0].served[0], 1.0)            # the limit serves 1 MW in the first batch
    assert len(h) == 3 and all(rec.served.sum() <= 1.0 + 1e-9 for rec in h[1:])
    assert h[0].payoffs[0] < 1.0 * 1.0 - 0.4 * 1.0 + 1e-9   # the waiting cost on the 1 MW still needed


def test_sequential_env_api():
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    from icgym.sequential import SequentialBatchEnv
    env = SequentialBatchEnv(make_game(rule=UniformRule(beta=0.5)), max_applicants=40)
    parallel_api_test(env, num_cycles=6)
    obs, infos = env.reset(seed=1)
    acts = {a: np.array([infos[a]["equilibrium_request"] / (env.request_cap * infos[a]["need"])], dtype=np.float32)
            for a in env.agents}
    _, rew, *_ = env.step(acts)
    assert all(np.isfinite(v) for v in rew.values())
