"""More project-page figures: the queue over time, the operator's landscape, efficiency, and generation regimes.

    python scripts/project_page/more_figures.py --out OUT_DIR [--only queue operator poa generation] [--workers 8]

Same style as make_figures.py (LaTeX text through matplotlib's usetex). Every
number is computed with the package; heavy intermediate results are cached in
OUT_DIR/cache_<figure>.json, so the plots can be redrawn quickly.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import make_figures as mf  # noqa: F401  (sets the LaTeX style)
import matplotlib.pyplot as plt
import numpy as np
from make_figures import DARK, GREY, LIGHT, RED, WIDTH, _clean
from matplotlib.patches import Patch

from icgym import (
    FlatFee,
    PerMWCommitment,
    activsg2000,
    ieee14,
    max_min_fair_point,
    poa_bound,
    price_of_anarchy,
    reference,
)
from icgym.rule_design import Objective, batch_metrics, equilibrium_response, make_rule
from icgym.sequential import ArrivalProcess, SequentialBatchGame

QUEUE_MENU = ("2", "4", "9", "13", "14")


def _cached(out: Path, name: str, compute):
    path = out / f"cache_{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = compute()
    path.write_text(json.dumps(data, default=float))
    return data


# ---------------------------------------------------------------------- 1. the queue over time
QUEUE_CONFIGS = [("uniform rule, per-MW commitment", "uniform", 0.4, "low", DARK, "-"),
                 ("uniform rule, flat fee", "uniform", 0.0, "high", RED, "-"),
                 ("proportional rule, per-MW commitment", "proportional", 0.4, "low", GREY, (0, (5, 2)))]


def _queue_run(job):
    label, rule_name, kappa, end, seed = job
    R = ieee14(QUEUE_MENU)
    arrivals = ArrivalProcess(menu=list(range(len(QUEUE_MENU))), rate=1.5, size=(15.0, 60.0), value=(1.0, 2.0))
    fee = PerMWCommitment(kappa) if kappa > 0 else FlatFee(0.0)
    game = SequentialBatchGame(R, arrivals, batches=10, rule=make_rule(rule_name, 0.5), fee=fee, max_tries=4)

    def policy(g):
        if not g.queue:
            return np.zeros(0)
        return equilibrium_response(g.batch_region(), g.rule, g.fee, [a.value for a in g.queue],
                                    need=[a.need for a in g.queue], flat_fee_end=end)
    res = game.simulate(policy, seed=seed)
    served = np.cumsum([h.served.sum() for h in res["history"]])
    wasted = np.cumsum([h.requests.sum() - h.served.sum() for h in res["history"]])
    return {"label": label, "seed": seed, "served": served.tolist(), "wasted": wasted.tolist()}


def queue_figure(out: Path, workers: int, seeds: int = 20):
    def compute():
        ieee14(QUEUE_MENU)    # export once before the workers start
        jobs = [(c[0], c[1], c[2], c[3], s) for c in QUEUE_CONFIGS for s in range(seeds)]
        with mp.get_context("spawn").Pool(workers) as pool:
            return pool.map(_queue_run, jobs)
    runs = _cached(out, "queue", compute)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    summary = {}
    for label, _, _, _, color, ls in QUEUE_CONFIGS:
        rows = [r for r in runs if r["label"] == label]
        for ax, key in zip(axes, ("served", "wasted")):
            Y = np.array([r[key] for r in rows])
            x = np.arange(1, Y.shape[1] + 1)
            ax.plot(x, Y.mean(axis=0), color=color, ls=ls, lw=1.9, label=label)
            ax.fill_between(x, np.percentile(Y, 25, axis=0), np.percentile(Y, 75, axis=0), color=color,
                            alpha=0.12, lw=0)
        summary[label] = {k: float(np.mean([r[k][-1] for r in rows])) for k in ("served", "wasted")}
    axes[0].set_ylabel(r"MW connected (cumulative)")
    axes[1].set_ylabel(r"MW requested, not served")
    for ax, t in zip(axes, (r"(a) capacity connected", r"(b) capacity paid for, not served")):
        ax.set_xlabel(r"batch")
        ax.set_title(t, loc="left")
        _clean(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.88))
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, fontsize=10.5, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2)
    fig.savefig(out / "queue.svg")
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------- 2. the operator's landscape
BETAS = np.linspace(0.0, 0.9, 10)
KAPPAS = np.linspace(0.0, 1.8, 10)


def _scenarios(k=12, seed=0):
    """The first ``k`` bundled price-of-anarchy batches, each with values drawn from U(1, 2)."""
    rng = np.random.default_rng(seed)
    names = [n for n in __import__("icgym").bundled("activsg2000") if n.startswith("poa/")]
    out = []
    for name in names[:k]:
        buses = name.split("buses_")[1].split("_")
        R = activsg2000(buses)
        out.append((buses, rng.uniform(1.0, 2.0, size=R.n).tolist()))
    return out


def _cell(job):
    rule_name, beta, kappa, scenarios = job
    objective = Objective(waste=0.5)
    vals = []
    for buses, values in scenarios:
        R = activsg2000(buses)
        rule = make_rule(rule_name, beta)
        fee = PerMWCommitment(kappa) if kappa > 1e-9 else FlatFee(0.0)
        r = equilibrium_response(R, rule, fee, values, flat_fee_end="high", grid=60, max_rounds=10)
        vals.append(objective(batch_metrics(R, r, rule.allocate(R, r))))
    return rule_name, beta, kappa, float(np.mean(vals))


def _cem(scenarios, iterations=15, population=24, elite=6, seed=0):
    """Cross-entropy search over the operator's action, each candidate scored on 3 random scenarios."""
    rng = np.random.default_rng(seed)
    mu, sd = np.array([0.5, 0.5, 0.5]), np.array([0.3, 0.3, 0.3])
    curve = []
    for _ in range(iterations):
        A = np.clip(mu + sd * rng.standard_normal((population, 3)), 0.0, 1.0)
        scores = []
        for a in A:
            name = ("uniform", "proportional")[min(int(a[0] * 2), 1)]
            picks = [scenarios[k] for k in rng.choice(len(scenarios), 3, replace=False)]
            scores.append(_cell((name, a[1] * 0.9, a[2] * 1.8, picks))[3])
        order = np.argsort(scores)[::-1][:elite]
        mu, sd = A[order].mean(axis=0), np.maximum(A[order].std(axis=0), 0.03)
        curve.append({"mean": float(np.mean(scores)), "best": float(np.max(scores)), "mu": mu.tolist()})
    return curve


def operator_figure(out: Path, workers: int):
    def compute():
        sc = _scenarios()
        jobs = [(r, float(b), float(k), sc) for r in ("uniform", "proportional") for b in BETAS for k in KAPPAS]
        with mp.get_context("spawn").Pool(workers) as pool:
            cells = pool.map(_cell, jobs)
        return {"cells": cells, "cem": _cem(sc)}
    data = _cached(out, "operator", compute)
    with plt.rc_context({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "legend.fontsize": 13}):
        return _operator_plot(out, data)


def _operator_plot(out: Path, data):
    fig = plt.figure(figsize=(WIDTH * 1.4, 3.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.32, 1.1], wspace=0.3)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 4])]
    cax = fig.add_subplot(gs[0, 2])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("p", ["#C0392B", "#F4D6D2", "#ffffff", LIGHT, DARK])
    vals = [c[3] for c in data["cells"]]
    lo, hi = min(vals), max(vals)
    for ax, name in zip(axes[:2], ("uniform", "proportional")):
        M = np.array([[next(c[3] for c in data["cells"] if c[0] == name and np.isclose(c[1], b) and np.isclose(c[2], k))
                       for k in KAPPAS] for b in BETAS])
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=lo, vmax=hi,
                       extent=[KAPPAS[0] - 0.1, KAPPAS[-1] + 0.1, BETAS[0] - 0.05, BETAS[-1] + 0.05])
        ax.set_title(rf"({'a' if name == 'uniform' else 'b'}) {name} rule", loc="left")
        ax.set_xlabel(r"commitment $\kappa$")
        ax.set_xticks([0.0, 0.5, 1.0, 1.5])
        ax.set_ylabel(r"minimum allocation $\beta$")
    axes[0].annotate(r"flat fee", (0.0, 0.45), xytext=(0.45, 0.5), color=GREY,
                     arrowprops={"arrowstyle": "->", "color": GREY, "lw": 0.8})
    axes[1].set_ylabel("")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"efficiency $-$ $0.5\,\times$ waste")
    curve = data["cem"]
    it = np.arange(1, len(curve) + 1)
    axes[2].plot(it, [c["mean"] for c in curve], color=DARK, lw=1.9, label=r"population mean")
    axes[2].plot(it, [c["best"] for c in curve], color=GREY, lw=1.2, ls=(0, (4, 2)), label=r"best candidate")
    axes[2].set_title(r"(c) an operator learning", loc="left")
    axes[2].set_xlabel(r"cross-entropy iterations")
    axes[2].set_ylabel(r"objective")
    axes[2].legend(frameon=False, loc="lower right")
    _clean(axes[2])
    fig.savefig(out / "operator.svg", bbox_inches="tight")
    plt.close(fig)
    last = curve[-1]["mu"]
    return {"range": [lo, hi], "cem_final": {"rule": "uniform" if last[0] < 0.5 else "proportional",
                                             "beta": last[1] * 0.9, "kappa": last[2] * 1.8,
                                             "objective": curve[-1]["mean"]}}


# ---------------------------------------------------------------------- 3. efficiency against spread
def poa_figure(out: Path):
    pts = []
    for name in __import__("icgym").bundled("activsg2000"):
        if name.startswith("poa/"):
            R = activsg2000(name.split("buses_")[1].split("_"))
            w = max_min_fair_point(R)
            h = R.hosting_capacities()
            pts.append((R.n, float(h.max() / h.min()), price_of_anarchy(R, w), poa_bound(R)))
    for q in reference("activsg2000", "poa_implicit")["poa_batches"]:
        n, b = len(q["sites"]), q["bound"]
        x = ((4 * b - 2) + np.sqrt((4 * b - 2) ** 2 - 4)) / 2          # x = nR solves (1 + x)^2 = 4 b x
        pts.append((n, x / n, q["poa"], b))
    fig, ax = plt.subplots(figsize=(WIDTH * 0.86, 3.3))
    Rg = np.logspace(0, np.log10(80), 200)
    for n, ls in ((2, "-"), (3, (0, (5, 2))), (5, (0, (1.5, 1.5)))):
        ax.plot(Rg, (1 + n * Rg) ** 2 / (4 * n * Rg), color=RED, lw=1.2, ls=ls, label=rf"bound, $n = {n}$")
    ax.plot(Rg, Rg, color=GREY, lw=0.9, ls=(0, (2, 2)), label=r"$\mathrm{PoA} = R$")
    for n, m in ((2, "o"), (3, "s"), (5, "^")):
        P = [p for p in pts if p[0] == n]
        ax.plot([p[1] for p in P], [p[2] for p in P], m, ms=5.5, mfc=LIGHT, mec=DARK, mew=1.0, ls="none",
                label=rf"batches of {n}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.9, 80)
    ax.set_ylim(0.9, 30)
    ax.set_xlabel(r"spread $R$ of hosting capacities")
    ax.set_ylabel(r"price of anarchy")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10.5, handlelength=2.4)
    _clean(ax)
    fig.savefig(out / "poa.svg")
    plt.close(fig)
    return {"batches": len(pts), "above_bound": sum(p[2] > p[3] * (1 + 1e-9) for p in pts),
            "closest_share": max(p[2] / p[3] for p in pts)}


# ---------------------------------------------------------------------- 4. generation regimes
KG = np.linspace(0.0, 12e6, 121)


def _gen_job(job):
    h, rule = job
    from icgym import NetworkGenerationGame, grid_equilibria
    from icgym.datasets import ieee14_network
    annuity = sum(1.07 ** -y for y in range(1, 11))
    ov = json.loads((Path(__import__("icgym").__file__).parent / "data" / "overlays" / "ieee14.json").read_text())
    W = np.repeat(ov["day_weights"], 24) * annuity
    V, KC, gamma = 1000.0 * W.sum(), 25e6, 0.5
    net = ieee14_network()
    game = NetworkGenerationGame(net, ["1", "2"], value=V, kappa=KC, gamma=gamma, run_hours=h, fuel_cost=150.0,
                                 period_weights=W, rule=rule)
    w = float(max_min_fair_point(game.base_region)[0])
    L = w
    for _ in range(30):
        Ln = max_min_fair_point(game._region([gamma * L] * 2))[0]
        if abs(Ln - L) < 1e-3:
            break
        L = Ln
    reqs = sorted(set(round(x, 2) for x in [w * f for f in (0.9, 1.0, 1.03)] +
                      [L * f for f in (0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1)]))
    strats = [(r, f * r) for r in reqs for f in (0.0, gamma)]
    _, table = grid_equilibria(game, [strats, strats])
    G = np.array([s[1] for s in strats])
    kinds = []
    for kg in KG:
        pay = {p: u - kg * G[list(p)] for p, u in table.items()}
        n = len(strats)
        eqs = [p for p in pay if all(pay[(k, p[1])][0] <= pay[p][0] + 1e3 for k in range(n)) and
               all(pay[(p[0], k)][1] <= pay[p][1] + 1e3 for k in range(n))]
        builds = [(G[p[0]] > 0, G[p[1]] > 0) for p in eqs]
        kinds.append("none" if not eqs else "all_cap" if all(b == (True, True) for b in builds) else
                     "no_build" if all(b == (False, False) for b in builds) else "mixed")
    return {"h": h, "rule": rule, "kinds": kinds}


def generation_figure(out: Path, workers: int):
    def compute():
        from icgym.datasets import fetch_ieee14
        fetch_ieee14(quiet=True)
        jobs = [(1.0, "gross"), (2.0, "gross"), (4.0, "gross"), (8.0, "gross"), (2.0, "credit")]
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            return pool.map(_gen_job, jobs)
    rows = _cached(out, "generation", compute)
    color = {"all_cap": LIGHT, "none": "#E8A79F", "mixed": "#d9d9d9", "no_build": "#ffffff"}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    labels = []
    for y, row in enumerate(rows):
        start = 0
        kinds = row["kinds"]
        for k in range(1, len(kinds) + 1):          # one block per run of the same regime (no seams)
            if k == len(kinds) or kinds[k] != kinds[start]:
                x0 = max(KG[start] / 1e6 - 0.05, -0.05)
                x1 = min(KG[k - 1] / 1e6 + 0.05, 12.05)
                ax.add_patch(plt.Rectangle((x0, y - 0.38), x1 - x0, 0.76, color=color[kinds[start]], lw=0))
                start = k
        labels.append(rf"$h = {row['h']:g}$ h/day" + (r", crediting" if row["rule"] == "credit" else ""))
        ax.add_patch(plt.Rectangle((-0.05, y - 0.38), 12.1, 0.76, fill=False, ec="#bbbbbb", lw=0.6))
    ax.set_xlim(-0.1, 12.1)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xlabel(r"generator capital cost $\kappa^G$ (\$M per MW)")
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(color=LIGHT, label=r"all build at the cap"),
                       Patch(color="#E8A79F", label=r"no pure equilibrium"),
                       Patch(color="#d9d9d9", label=r"one builds"),
                       Patch(facecolor="white", edgecolor="#bbbbbb", label=r"nobody builds")],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=4, fontsize=10.5, handlelength=1.4)
    fig.savefig(out / "generation.svg")
    plt.close(fig)
    return {f"{r['h']}_{r['rule']}": r["kinds"].count("none") for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=["queue", "operator", "poa", "generation"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = {}
    for name in args.only:
        fn = {"queue": lambda: queue_figure(out, args.workers), "operator": lambda: operator_figure(out, args.workers),
              "poa": lambda: poa_figure(out), "generation": lambda: generation_figure(out, args.workers)}[name]
        done[name] = fn()
        print(name, json.dumps(done[name], default=float)[:300], flush=True)
    (out / "more_figures.json").write_text(json.dumps(done, indent=1, default=float))


if __name__ == "__main__":
    main()
