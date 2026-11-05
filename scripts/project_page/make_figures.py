"""Figures of the project page (https://mhaomou.github.io/interconnection-gym/).

    python scripts/project_page/make_figures.py --out OUT_DIR \
        [--layout activsg2000_geo_layout.json --activsg path/to/activsg2000_scenario]

Writes SVG files with LaTeX-rendered text (matplotlib ``text.usetex``, so a
LaTeX installation is needed). Every number comes from the package: the
regions and rule comparison are recomputed, the learning curves are rerun, and
the benchmark table is read from ``benchmarks/results.json``. The map panel
needs bus coordinates (``--layout``: ``positions`` and ``background_paths``,
normalized, y pointing south) and the ACTIVSg2000 branch list (``--activsg``);
without them the teaser has two panels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.spatial import ConvexHull, HalfspaceIntersection  # noqa: E402

from icgym import PerMWCommitment, activsg2000_location, ieee14, max_min_fair_point  # noqa: E402
from icgym.benchmarks import TASKS, adapter  # noqa: E402
from icgym.compare import compare_rules  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DARK, LIGHT, GREY, RED, LIGHTRED = "#1F4E79", "#BDD7EE", "#595959", "#C0392B", "#F4D6D2"
WIDTH = 8.4   # inches; the page shows figures about 840 px wide, so 1 pt of text is about 1.4 px

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.size": 12.5, "axes.titlesize": 13.5,
    "axes.labelsize": 12.5, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.edgecolor": GREY, "axes.linewidth": 0.8, "xtick.color": GREY, "ytick.color": GREY,
    "axes.labelcolor": "#222222", "svg.fonttype": "path", "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "text.latex.preamble": r"\usepackage{amsmath}",
})


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ------------------------------------------------------------------ panels
def region_panel(ax):
    """Batch region of IEEE 14-bus buses 9 and 13, water-filling, and the planner's optimum."""
    R = ieee14((9, 13))
    G = np.vstack([R.A, -np.eye(2)])
    h = np.concatenate([R.b, [0, 0]])
    P = HalfspaceIntersection(np.hstack([G, -h[:, None]]), np.array([5.0, 5.0])).intersections
    V = P[ConvexHull(P).vertices]
    w = max_min_fair_point(R)
    total, opt = R.max_total()
    ax.fill(V[:, 0], V[:, 1], color=LIGHT, zorder=1)
    ax.plot(*np.vstack([V, V[:1]]).T, color=DARK, lw=1.6, zorder=2)
    t = np.linspace(0, 52, 2)
    ax.plot(t, t, color=RED, lw=1.4, ls=(0, (4, 3)), zorder=3)
    ax.plot(*w, "o", color=RED, ms=7.5, mec="white", mew=1.2, zorder=4)
    ax.plot(*opt, "o", color=DARK, ms=7, mec="white", mew=1.2, zorder=4)
    ax.annotate(r"equilibrium $w$", w, xytext=(10, 8), textcoords="offset points", color=RED)
    ax.annotate(r"planner", opt, xytext=(7, 7), textcoords="offset points", ha="left", color=DARK)
    ax.text(30, 8, r"$\{s \ge 0 : As \le b\}$", color=DARK, ha="center")
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 62)
    ax.set_xlabel(r"served at bus 9 (MW)")
    ax.set_ylabel(r"served at bus 13 (MW)")
    _clean(ax)
    return {"w": w.tolist(), "planner": opt.tolist(), "poa": float(total / w.sum())}


def map_panel(ax, layout, activsg):
    """ACTIVSg2000 grid with the location game's buses and its equilibrium placement."""
    lay = json.loads(Path(layout).read_text())
    pos = {b: (x, -y) for b, (x, y) in lay["positions"].items()}
    for path in lay.get("background_paths", []):
        p = np.array(path)
        ax.fill(p[:, 0], -p[:, 1], color="#f3f6fa", zorder=0)
        ax.plot(p[:, 0], -p[:, 1], color="#c9d3df", lw=0.6, zorder=0)
    segs = []
    with open(Path(activsg) / "branch_limits.csv", newline="") as f:
        for r in csv.DictReader(f):
            a, b = r["from_bus_id"], r["to_bus_id"]
            if r.get("status", "1") == "1" and a in pos and b in pos:
                segs.append([pos[a], pos[b]])
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors="#7f98b3", linewidths=0.45, zorder=1, rasterized=True))
    game = activsg2000_location()
    menu = game.menus[0]
    eq = game.pure_equilibria()[0]
    counts = {b: eq.count(b) for b in menu}
    for b in menu:
        x, y = pos[b]
        chosen = counts[b] > 0
        ax.plot(x, y, "o", ms=11 if chosen else 7, color=RED if chosen else "white",
                mec=RED if chosen else DARK, mew=1.4, zorder=3)
        if chosen:
            ax.annotate(rf"{counts[b]} applicant{'s' if counts[b] > 1 else ''}", (x, y), xytext=(0, 11),
                        textcoords="offset points", ha="center", color=RED, fontsize=11.5,
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})
    ax.set_aspect("equal")
    ax.axis("off")
    return {"menu": menu, "equilibrium": list(eq)}


def learning_panel(ax, iters=400, every=20, seed=0):
    """Independent policy gradient (rewards only): exploitability of the mean policy over training."""
    curves = {}
    for task, color, ls in [("uniform-ieee14", DARK, "-"), ("uniform-random8", DARK, (0, (5, 2))),
                            ("proportional-limit", GREY, "-"), ("generation-gross-cycle", RED, "-")]:
        ad = adapter(TASKS[task].make())
        xs, ys = _pg_curve(ad, iters, every, seed)
        curves[task] = {"iterations": xs, "exploitability": ys}
        ax.semilogy(xs, np.maximum(ys, 1e-5), color=color, ls=ls, lw=1.8)
    ax.text(iters, 0.045, r"no pure equilibrium", color=RED, ha="right", va="bottom")
    ax.text(iters, 4.5e-4, r"uniform rule (2 tasks)", color=DARK, ha="right", va="bottom")
    ax.text(iters, 1.6e-5, r"proportional rule", color=GREY, ha="right", va="bottom")
    ax.set_xlabel(r"policy-gradient iterations")
    ax.set_ylabel(r"exploitability")
    ax.set_ylim(1e-5, 1.5)
    _clean(ax)
    return curves


def _pg_curve(ad, iterations, every, seed, batch=128, lr=0.3):
    """The ``independent_pg`` baseline, scoring the mean policy every ``every`` iterations."""
    rng = np.random.default_rng(seed)
    n, d = ad.n, ad.dim
    mu, log_sigma = np.zeros((n, d)), np.zeros((n, d))
    xs, ys = [], []

    def score():
        a = 1.0 / (1.0 + np.exp(-mu))
        return ad.evaluate(ad.to_strategy(a[:, 0] if d == 1 else a))["exploitability"]

    for it in range(iterations + 1):
        if it % every == 0:
            xs.append(it)
            ys.append(score())
        if it == iterations:
            break
        step = lr * (1 - it / iterations) + 0.02
        sigma = np.exp(log_sigma)
        z = mu + sigma * rng.standard_normal((batch, n, d))
        A = 1.0 / (1.0 + np.exp(-z))
        R = ad.rewards(A[..., 0] if d == 1 else A)
        for i in range(n):
            adv = R[:, i] - R[:, i].mean()
            adv = adv / (adv.std() + 1e-8)
            eps = (z[:, i] - mu[i]) / sigma[i]
            mu[i] += step * sigma[i] * (adv[:, None] * eps).mean(axis=0)
            log_sigma[i] += 0.5 * step * (adv[:, None] * (eps ** 2 - 1)).mean(axis=0)
        log_sigma = np.clip(log_sigma, -6.0, 1.0)
    return xs, ys


# ------------------------------------------------------------------ figures
def teaser(out, layout=None, activsg=None):
    has_map = layout is not None and activsg is not None
    if has_map:
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH * 1.4, 3.5), gridspec_kw={"width_ratios": [1.25, 1, 1]})
        info = {"map": map_panel(axes[0], layout, activsg)}
        axes[0].set_title(r"(a) where applicants connect", loc="left")
        rest = axes[1:]
    else:
        fig, rest = plt.subplots(1, 2, figsize=(WIDTH, 3.3))
        info = {}
    info["region"] = region_panel(rest[0])
    rest[0].set_title(("(b) " if has_map else "(a) ") + r"what the rule serves", loc="left")
    info["learning"] = learning_panel(rest[1])
    rest[1].set_title(("(c) " if has_map else "(b) ") + r"adaptive applicants", loc="left")
    fig.tight_layout(w_pad=2.2)
    fig.savefig(out / "teaser.svg")
    plt.close(fig)
    return info


def rules_figure(out):
    reports = compare_rules(ieee14((9, 13, 14)), ["uniform", "proportional", "utilitarian"],
                            fee=PerMWCommitment(1 / 2.46), beta=0.0)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    width, buses = 0.26, ["bus 9", "bus 13", "bus 14"]
    for g, rep in enumerate(reports):
        for i, (s, r) in enumerate(zip(rep.served, rep.requests)):
            x = g + (i - 1) * (width + 0.03)
            ax.bar(x, s, width, color=DARK, zorder=2)
            if r > s + 1e-6:
                ax.bar(x, r - s, width, bottom=s, color="white", edgecolor=RED, hatch="////", lw=1.0, zorder=2)
            ax.text(x, -6.5, buses[i].split()[1], ha="center", va="top", color=GREY, fontsize=10.5)
        ax.text(g, 103, rf"efficiency {rep.efficiency:.3f}, inflation {rep.max_inflation:.2f}",
                ha="center", color=GREY, fontsize=11)
    ax.set_xticks(range(3), [rf"\textbf{{{r.rule}}}" for r in reports])
    ax.tick_params(axis="x", length=0, pad=22)
    ax.set_ylim(0, 112)
    ax.set_ylabel(r"MW")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=DARK, label=r"served"),
                       Patch(facecolor="white", edgecolor=RED, hatch="////", label=r"requested, not served")],
              frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    _clean(ax)
    fig.savefig(out / "rules.svg")
    plt.close(fig)
    return [{"rule": r.rule, "served": r.served, "requests": r.requests, "efficiency": r.efficiency,
             "inflation": r.max_inflation} for r in reports]


def benchmark_figure(out):
    res = json.loads((ROOT / "benchmarks" / "results.json").read_text())
    tasks = list(TASKS)
    bases = [("best_response", r"best responses"), ("hedge", r"Hedge"), ("independent_pg", r"policy gradient")]
    M = np.array([[np.mean([r["exploitability"] for r in res if r["task"] == t and r["baseline"] == b])
                   for b, _ in bases] for t in tasks])
    cmap = LinearSegmentedColormap.from_list("p", ["#ffffff", LIGHTRED, RED])
    fig, ax = plt.subplots(figsize=(WIDTH * 0.82, 3.9))
    ax.imshow(np.log10(np.maximum(M, 1e-4)), cmap=cmap, vmin=-4, vmax=0, aspect="auto")
    for i in range(len(tasks)):
        for j in range(len(bases)):
            v = M[i, j]
            ax.text(j, i, "0" if v < 5e-5 else f"{v:.4f}", ha="center", va="center",
                    color="white" if v > 0.1 else "#222222", fontsize=11.5)
    ax.set_xticks(range(len(bases)), [b for _, b in bases])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(tasks)), [r"\texttt{" + t.replace("-", r"-\allowbreak ") + "}" for t in tasks])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(bases)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tasks)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    fig.savefig(out / "benchmarks.svg")
    plt.close(fig)
    return {"tasks": tasks, "baselines": [b for b, _ in bases], "exploitability": M.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", help="bus coordinates JSON (positions, background_paths)")
    ap.add_argument("--activsg", help="ACTIVSg2000 scenario folder (for branch_limits.csv)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    numbers = {"teaser": teaser(out, args.layout, args.activsg), "rules": rules_figure(out),
               "benchmarks": benchmark_figure(out)}
    (out / "figures.json").write_text(json.dumps(numbers, indent=1, default=float))
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
