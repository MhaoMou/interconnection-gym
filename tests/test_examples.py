"""Every example runs (with small budgets) and prints what it promises."""

import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def run(name, *args):
    old = sys.argv
    sys.argv = [name, *args]
    try:
        runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    finally:
        sys.argv = old


def test_quickstart(capsys):
    run("quickstart.py")
    assert "equilibrium requests: [1. 1. 2.]" in capsys.readouterr().out


def test_compare_rules(tmp_path, capsys):
    run("compare_rules.py", str(tmp_path / "r.html"))
    out = capsys.readouterr().out
    assert "proportional" in out and (tmp_path / "r.html").exists()


def test_location_game(capsys):
    run("location_game.py")
    assert "1617" in capsys.readouterr().out


def test_train_independent_pg(capsys):
    pytest.importorskip("pettingzoo")
    run("train_independent_pg.py", "40")
    assert "exploitability" in capsys.readouterr().out
