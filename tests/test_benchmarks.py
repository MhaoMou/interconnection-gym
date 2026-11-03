import numpy as np

from icgym.benchmarks import BASELINES, TASKS, adapter, format_results, run


def test_every_task_builds_and_scores_its_prediction():
    for name, task in TASKS.items():
        ad = adapter(task.make())
        if task.expected in ("unique", "band"):
            p = ad.pred
            X = p.get("X", p.get("requests", p.get("low")))
            m = ad.evaluate(X)
            assert m["exploitability"] < 1e-4, name
            assert m.get("distance", 0.0) < 1e-9, name


def test_no_equilibrium_task_stays_exploitable():
    ad = adapter(TASKS["generation-gross-cycle"].make())
    assert ad.pred["kind"] == "none"
    r = run("generation-gross-cycle", "best_response", seed=0)
    assert r["exploitability"] > 1e-3 and not r["converged"]


def test_baselines_on_the_uniform_task():
    br = run("uniform-ieee14", "best_response", seed=1)
    assert br["converged"] and br["exploitability"] < 1e-6 and br["distance"] < 1e-6
    pg = run("uniform-ieee14", "independent_pg", seed=1)
    assert pg["exploitability"] < 1e-2 and pg["distance"] < 1e-2
    hd = run("location-activsg2000", "hedge", seed=0)
    assert hd["exploitability"] == 0.0 and hd["distance"] == 0.0
    table = format_results([br, pg])
    assert table.count("\n") == 3 and "uniform-ieee14" in table


def test_registry():
    assert set(BASELINES) == {"best_response", "hedge", "independent_pg"}
    assert {t.expected for t in TASKS.values()} == {"unique", "band", "location", "none"}
    assert np.all([t.description for t in TASKS.values()])
