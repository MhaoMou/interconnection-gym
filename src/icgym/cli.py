"""Command line: export a batch region, or compare allocation rules on a batch.

    icgym-export --data data/teaching/ieee14_demo_representative --buses 9,13,14 --out r.json
    icgym-compare --region r.json --kappa 0.4 --beta 0.5 --html report.html
    icgym-compare --data path/to/scenario --buses 7106,5430,5276 --implicit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .network import DCNetwork, export_region
from .scenarios import save_region
from .waterfill import max_min_fair_point


def _periods(text: str | None):
    if not text:
        return None
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="scenario directory (bus.csv, branch_limits.csv, ...)")
    ap.add_argument("--buses", required=True, help="comma-separated bus ids, one applicant per bus")
    ap.add_argument("--periods", help="e.g. 0-23,48 (default: all)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    buses = [b.strip() for b in args.buses.split(",") if b.strip()]
    t0 = time.time()
    net = DCNetwork.from_csv_dir(args.data)
    ex = export_region(net, buses, periods=_periods(args.periods), workers=args.workers)
    ex.metadata["export_seconds"] = round(time.time() - t0, 1)
    ex.metadata["has_counterflow"] = ex.has_counterflow
    save_region(ex.region, args.out, signed=(ex.signed_A, ex.signed_b), **ex.metadata)
    w = max_min_fair_point(ex.region)
    print(f"{args.out}: {ex.region.m} limits, counterflow={ex.has_counterflow}, "
          f"hosting capacities {np.round(ex.region.hosting_capacities(), 2).tolist()} MW, "
          f"max-min fair point {np.round(w, 2).tolist()} MW ({ex.metadata['export_seconds']} s)")


def compare_main(argv=None) -> None:
    from .compare import RULES, compare_rules, format_table, report_html, to_json
    from .fees import FlatFee, PerMWCommitment
    from .scenarios import load_region

    ap = argparse.ArgumentParser(description="Compare allocation rules on one batch at their equilibria.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--region", help="region JSON (from icgym-export or save_region)")
    src.add_argument("--data", help="scenario directory; needs --buses")
    ap.add_argument("--buses", help="comma-separated bus ids, one applicant per bus")
    ap.add_argument("--implicit", action="store_true",
                    help="with --data: use DCOPFRegion (exact, any size) instead of exporting the region")
    ap.add_argument("--rules", default="uniform,proportional,utilitarian", help=f"comma-separated: {', '.join(RULES)}")
    ap.add_argument("--beta", type=float, default=0.5, help="minimum allocation (fraction of the request)")
    ap.add_argument("--value", type=float, default=1.0, help="value per MW served (same units as the fee)")
    fee = ap.add_mutually_exclusive_group()
    fee.add_argument("--kappa", type=float, help="commitment per MW requested (default 0.4)")
    fee.add_argument("--flat-fee", type=float, help="flat study fee per applicant instead of a per-MW commitment")
    ap.add_argument("--slow", action="store_true", help="search for equilibria on implicit regions too")
    ap.add_argument("--json", help="write the reports as JSON")
    ap.add_argument("--html", help="write a self-contained HTML report")
    args = ap.parse_args(argv)
    if args.region:
        region, label = load_region(args.region), args.region
    else:
        if not args.buses:
            ap.error("--data needs --buses")
        buses = [b.strip() for b in args.buses.split(",") if b.strip()]
        net = DCNetwork.from_csv_dir(args.data)
        if args.implicit:
            from .dcopf_region import DCOPFRegion
            region = DCOPFRegion(net, buses)
        else:
            region = export_region(net, buses).region
        label = f"{net.source}, buses {', '.join(buses)}"
    the_fee = FlatFee(args.flat_fee) if args.flat_fee is not None else PerMWCommitment(
        0.4 if args.kappa is None else args.kappa)
    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    unknown = [r for r in rules if r not in RULES]
    if unknown:
        ap.error(f"unknown rules {unknown}; choose from {list(RULES)}")
    reports = compare_rules(region, rules, fee=the_fee, value=args.value, beta=args.beta, slow=args.slow)
    names = list(region.applicant_names)
    print(format_table(reports, names))
    if args.json:
        Path(args.json).write_text(to_json(reports))
    if args.html:
        fee_text = (f"flat fee {args.flat_fee:g} per applicant" if args.flat_fee is not None
                    else f"commitment {the_fee.kappa:.3g} per MW requested")
        sub = f"{label}; value {args.value:g} per MW served, {fee_text}, minimum allocation {args.beta:g}"
        report_html(reports, args.html, subtitle=sub, applicant_names=names)


if __name__ == "__main__":
    main()
