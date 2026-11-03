import json

import numpy as np

from icgym import FlatFee, PerMWCommitment, Region, ieee14, max_min_fair_point, proportional_inflation
from icgym.cli import compare_main
from icgym.compare import compare_rules, format_table


def test_uniform_row_is_the_max_min_fair_point():
    region = ieee14((9, 13, 14))
    rep = compare_rules(region, ["uniform"])[0]
    assert rep.method == "theory" and rep.exploitability < 1e-6
    assert np.allclose(rep.served, max_min_fair_point(region)) and np.isclose(rep.max_inflation, 1.0)
    assert rep.price_of_anarchy <= rep.poa_bound and rep.binding_limits


def test_proportional_inflation_on_one_limit():
    region = Region.single_limit([1.0, 2.0, 1.5], 6.0)
    rep = compare_rules(region, ["proportional"], fee=PerMWCommitment(0.4), beta=0.5)[0]
    assert rep.method == "theory"
    assert np.isclose(rep.max_inflation, proportional_inflation(3, 1.0, 0.4, 0.5))
    assert np.isclose(rep.unserved_requested_mw, (rep.max_inflation - 1) * sum(rep.served))


def test_paper_table_on_the_14_bus_triple():
    """Table 'Uniform versus proportional rule', buses 9, 13, 14 (V / kappa = 2.46, beta = 0)."""
    rep = {r.rule: r for r in compare_rules(ieee14((9, 13, 14)), ["uniform", "proportional"],
                                             fee=PerMWCommitment(1 / 2.46), beta=0.0)}
    assert np.allclose(rep["uniform"].served, [46.78, 25.40, 25.40], atol=0.01)
    assert rep["proportional"].method == "dynamics" and rep["proportional"].exploitability < 1e-6
    assert np.allclose(rep["proportional"].served, [54.89, 20.27, 29.73], atol=0.01)
    assert np.isclose(rep["proportional"].max_inflation, 1.686, atol=1e-3)
    assert np.isclose(rep["uniform"].price_of_anarchy, 1.24, atol=0.005)


def test_flat_fee_band_note():
    rep = compare_rules(ieee14((1, 2)), ["uniform"], fee=FlatFee(0.1))[0]
    assert rep.method == "theory" and "flat fee" in rep.note


def test_cli_writes_json_and_html(tmp_path, capsys):
    from icgym import save_region
    path = tmp_path / "region.json"
    save_region(ieee14((13, 14)), path)
    compare_main(["--region", str(path), "--rules", "uniform,proportional", "--json", str(tmp_path / "r.json"),
                  "--html", str(tmp_path / "r.html")])
    out = capsys.readouterr().out
    assert "uniform" in out and "proportional" in out
    data = json.loads((tmp_path / "r.json").read_text())
    assert [d["rule"] for d in data] == ["uniform", "proportional"]
    page = (tmp_path / "r.html").read_text()
    assert page.startswith("<!doctype html>") and "prefers-color-scheme" in page and "<svg" in page
    assert format_table([]).startswith("rule")
