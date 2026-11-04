import numpy as np
import pytest

from icgym import (
    PerMWCommitment,
    ProportionalRule,
    Region,
    UniformRule,
    poa_bound,
    price_of_anarchy,
    proportional_inflation,
)
from icgym.rule_design import Objective, batch_metrics, equilibrium_response
from icgym.sequential import ArrivalProcess, SequentialBatchGame


def test_uniform_commitment_wastes_nothing():
    R = Region([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], [2.0, 3.0])
    rule = UniformRule(beta=0.5)
    r = equilibrium_response(R, rule, PerMWCommitment(0.4), 1.0)
    s = rule.allocate(R, r)
    m = batch_metrics(R, r, s)
    assert m["wasted_share"] < 1e-9 and np.isclose(m["efficiency"], 1 / price_of_anarchy(R, s))
    assert price_of_anarchy(R, s) <= poa_bound(R)


def test_flat_fee_high_end_and_needs():
    R = Region.single_limit([1.0, 1.0], 2.0)
    rule = UniformRule(beta=0.5)
    from icgym import FlatFee
    r = equilibrium_response(R, rule, FlatFee(0.0), 1.0, flat_fee_end="high")
    assert np.allclose(r, [2.0, 2.0])                       # w / beta
    assert batch_metrics(R, r, rule.allocate(R, r))["wasted_share"] > 0.4
    capped = equilibrium_response(R, rule, PerMWCommitment(0.4), 1.0, need=[0.5, np.inf])
    assert np.allclose(capped, [0.5, 1.5])                   # the need caps one applicant, the other fills


def test_proportional_inflates_on_one_limit():
    R = Region.single_limit([1.0, 2.0, 1.5], 6.0)
    r = equilibrium_response(R, ProportionalRule(beta=0.0), PerMWCommitment(0.4), 1.0)
    s = ProportionalRule().allocate(R, r)
    assert np.allclose(r / s, proportional_inflation(3, 1.0, 0.4, 0.0), rtol=1e-3)
    assert Objective(waste=1.0)(batch_metrics(R, r, s)) < batch_metrics(R, r, s)["efficiency"]


def test_operator_envs():
    pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    from icgym import random_region
    from icgym.rule_design import OperatorEnv, SequentialOperatorEnv
    env = OperatorEnv(rules=("uniform", "proportional"), max_applicants=4, max_limits=3)
    check_env(env, skip_render_check=True)
    env.reset(seed=0)
    _, good, *_ = env.step(np.array([0.1, 0.5, 0.4], dtype=np.float32))     # uniform, commitment 0.4
    env.reset(seed=0)
    _, flat, *_ = env.step(np.array([0.1, 0.5, 0.0], dtype=np.float32))     # uniform, flat fee
    assert good >= flat - 1e-9
    game = SequentialBatchGame(random_region(4, 3, seed=1), ArrivalProcess(menu=[0, 1, 2, 3], rate=2.0,
                                                                            size=(0.2, 0.5)), batches=4)
    senv = SequentialOperatorEnv(game, rules=("uniform", "proportional"))
    check_env(senv, skip_render_check=True)
    senv.reset(seed=2)
    total, done = 0.0, False
    while not done:
        _, rew, done, _, info = senv.step(np.array([0.1, 0.5, 0.4], dtype=np.float32))
        total += rew
    assert total > 0
