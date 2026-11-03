"""Statistics of an exported catalog: counterflow on pairs, and the batch
results recomputed on the exported (downward-closed) regions.

    python scripts/analyze_catalog.py --catalog CATALOG_DIR --reference REF_DIR --out stats.json

``REF_DIR`` holds the paper's ``pairs.json``, ``poa.json`` and ``location.json``
(for the adjacency flags and the values to compare against).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

from icgym import max_min_fair_point, poa_bound, price_of_anarchy
from icgym.scenarios import load_region_record


def _path(catalog: Path, group: str, buses) -> Path:
    return catalog / group / ("buses_" + "_".join(str(b) for b in buses) + ".json")


def _area(A, b):
    """Area of the polygon ``{x >= 0 : A x <= b}``."""
    G = np.vstack([A, -np.eye(2)])
    h = np.concatenate([b, [0.0, 0.0]])
    res = linprog([0, 0, -1], A_ub=np.hstack([G, np.linalg.norm(G, axis=1)[:, None]]), b_ub=h,
                  bounds=[(None, None)] * 2 + [(0, None)], method="highs")
    inner = res.x[:2]
    pts = HalfspaceIntersection(np.hstack([G, -h[:, None]]), inner).intersections
    return ConvexHull(pts).volume


def _cap(A, b, j, other):
    c = np.zeros(2)
    c[j] = -1.0
    bounds = [(0, None), (0, None)]
    bounds[1 - j] = (other, other)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return -res.fun if res.status == 0 else np.nan


def _max_total(A, b):
    res = linprog(-np.ones(A.shape[1]), A_ub=A, b_ub=b, bounds=[(0, None)] * A.shape[1], method="highs")
    return -res.fun


def pair_stats(rec) -> dict:
    A, b = rec["signed_A"], rec["signed_b"]
    Ap, bp = rec["region"].A, rec["region"].b
    rises = []
    for j in (0, 1):
        alone = _cap(A, b, j, 0.0)
        reach = _cap(A, b, 1 - j, 0.0)
        best = max(_cap(A, b, j, x) for x in np.linspace(0.0, reach, 201))
        # the maximum of a concave piecewise-linear function: refine on a vertex
        best = max(best, max(_cap(A, b, j, x) for x in _vertices_other(A, b, j, reach)))
        rises.append(float(best - alone))
    return {
        "counterflow": bool(np.any(A < -1e-9)),
        "rise_mw": rises,
        "area_loss_pct": 100 * (1 - _area(Ap, bp) / _area(A, b)),
        "total_loss_pct": 100 * (1 - _max_total(Ap, bp) / _max_total(A, b)),
    }


def _vertices_other(A, b, j, reach):
    """Values of the other coordinate at the region's vertices (where rises peak)."""
    G = np.vstack([A, -np.eye(2)])
    h = np.concatenate([b, [0.0, 0.0]])
    out = []
    for r, s in itertools.combinations(range(len(h)), 2):
        M = G[[r, s]]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        x = np.linalg.solve(M, h[[r, s]])
        if np.all(G @ x <= h + 1e-6) and 0 <= x[1 - j] <= reach + 1e-9:
            out.append(float(x[1 - j]))
    return out or [0.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cat, ref = Path(args.catalog), Path(args.reference)
    out = {}

    pairs = json.loads((ref / "pairs.json").read_text())
    rows = []
    for p in pairs:
        rec = load_region_record(_path(cat, "pairs", p["buses"]))
        s = pair_stats(rec)
        s.update(buses=p["buses"], adjacent=p["adjacent"], paper_rise_mw=p["max_rise_mw"],
                 paper_area_loss_pct=p["area_loss_pct"], paper_total_loss_pct=p["total_loss_pct"])
        rows.append(s)
    def summ(sel):
        sel = list(sel)
        return {"pairs": len(sel), "with_counterflow": sum(r["max_rise"] > 1e-3 for r in sel),
                "mean_area_loss_pct": float(np.mean([r["area_loss_pct"] for r in sel])),
                "max_area_loss_pct": float(np.max([r["area_loss_pct"] for r in sel])),
                "mean_total_loss_pct": float(np.mean([r["total_loss_pct"] for r in sel])),
                "max_total_loss_pct": float(np.max([r["total_loss_pct"] for r in sel])),
                "max_rise_mw": float(np.max([r["max_rise"] for r in sel]))}
    for r in rows:
        r["max_rise"] = max(r["rise_mw"])
    out["pairs"] = {"all": summ(rows), "adjacent": summ(r for r in rows if r["adjacent"]),
                    "random": summ(r for r in rows if not r["adjacent"]), "rows": rows}

    poa = json.loads((ref / "poa.json").read_text())
    prow = []
    for q in poa:
        region = load_region_record(_path(cat, "poa", q["sites"]))["region"]
        w = max_min_fair_point(region)
        prow.append({"sites": q["sites"], "w_mw": w.tolist(), "paper_w_mw": q["w_mw"],
                     "poa": price_of_anarchy(region, w), "paper_poa": q["poa"], "bound": poa_bound(region),
                     "max_w_diff_mw": float(np.max(np.abs(w - np.array(q["w_mw"]))))})
    out["poa"] = {"max_w_diff_mw": max(r["max_w_diff_mw"] for r in prow),
                  "max_poa_diff": max(abs(r["poa"] - r["paper_poa"]) for r in prow),
                  "violations_of_bound": sum(r["poa"] > r["bound"] * (1 + 1e-9) for r in prow), "rows": prow}

    loc = json.loads((ref / "location.json").read_text())
    menu = loc["menu"]
    w = {}
    for combo in itertools.combinations_with_replacement(menu, 3):
        ws = max_min_fair_point(load_region_record(_path(cat, "location", combo))["region"])
        for perm in set(itertools.permutations(range(3))):
            prof = tuple(combo[k] for k in perm)
            w[prof] = ws[list(perm)]
    pure, moves, viol = [], 0, 0
    for prof in w:
        eq = True
        for i in range(3):
            for bus in menu:
                if bus == prof[i]:
                    continue
                new = list(prof)
                new[i] = bus
                new = tuple(new)
                if w[new][i] > w[prof][i] + 1e-4:
                    eq, moves = False, moves + 1
                    d = np.sort(w[new]) - np.sort(w[prof])
                    nz = np.where(np.abs(d) > 1e-4)[0]
                    viol += int(len(nz) == 0 or d[nz[0]] < 0)
        if eq:
            pure.append(list(prof))
    out["location"] = {"profiles": len(w), "pure_equilibria": pure, "paper_pure_equilibria": loc["pure_equilibria"],
                       "improving_moves": moves, "paper_improving_moves": loc["improving_moves"],
                       "lexicographic_violations": viol}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"pairs": {k: v for k, v in out["pairs"].items() if k != "rows"},
                      "poa": {k: v for k, v in out["poa"].items() if k != "rows"},
                      "location": out["location"]}, indent=1))


if __name__ == "__main__":
    main()
