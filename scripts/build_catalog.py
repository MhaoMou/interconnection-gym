"""Export many batch regions of one network in a single worker pool.

Every (region, period) pair is one job, so a cluster stays busy even when the
regions are small. Usage:

    python scripts/build_catalog.py --data DIR --spec spec.json --out OUT --workers 80

``spec.json`` maps a group name to a list of bus lists, for example
``{"pairs": [["1001", "1064"], ...], "poa": [[...], ...]}``. Each region is
written to ``OUT/<group>/buses_<b1>_<b2>....json``; finished regions are
skipped on a rerun.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

from icgym import network
from icgym.network import DCNetwork, assemble_region
from icgym.scenarios import save_region


def _job(args):
    key, buses, t, tol = args
    try:
        return key, t, network.period_facets(network._NET.period_problem(t, buses), tol=tol)
    except Exception as exc:  # one bad LP must not stop the run
        return key, t, exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-7)
    args = ap.parse_args()
    net = DCNetwork.from_csv_dir(args.data)
    spec = json.loads(Path(args.spec).read_text())
    out = Path(args.out)
    todo = {}
    for group, sets in spec.items():
        (out / group).mkdir(parents=True, exist_ok=True)
        for buses in sets:
            buses = [str(b) for b in buses]
            path = out / group / ("buses_" + "_".join(buses) + ".json")
            if not path.exists():
                todo[(group, tuple(buses))] = path
    periods = list(range(net.periods))
    jobs = [(key, list(key[1]), t, args.tol) for key in todo for t in periods]
    print(f"{len(todo)} regions, {len(jobs)} period jobs, {args.workers} workers", flush=True)
    parts = defaultdict(dict)
    done, t0 = 0, time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=network._init_worker,
                                      initargs=(net.root, net.periods)) as pool:
        failed = set()
        for key, t, facets in pool.imap_unordered(_job, jobs, chunksize=4):
            if isinstance(facets, Exception):
                if key not in failed:
                    print(f"FAILED {key} period {t}: {facets}", flush=True)
                failed.add(key)
            parts[key][t] = facets
            if key in failed:
                if len(parts[key]) == len(periods):
                    del parts[key]
                continue
            if len(parts[key]) == len(periods):
                ex = assemble_region(net, key[1], periods, [parts[key][p] for p in periods], tol=args.tol)
                ex.metadata["has_counterflow"] = ex.has_counterflow
                save_region(ex.region, todo[key], signed=(ex.signed_A, ex.signed_b), **ex.metadata)
                del parts[key]
                done += 1
                if done % 10 == 0 or done == len(todo):
                    print(f"{done}/{len(todo)} regions ({time.time() - t0:.0f}s)", flush=True)
    print(f"finished: {done} written, {len(failed)} failed ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
