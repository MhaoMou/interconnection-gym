"""Compare allocation rules and fees on one batch: what an operator would see.

:func:`compare_rules` finds each rule's equilibrium (the theory's answer where
one is known, checked numerically; otherwise best-response dynamics from the
max-min fair point) and reports what it serves, what it wastes, and how far it
is from the planner's optimum. :func:`report_html` writes a self-contained page.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .fees import Fee, PerMWCommitment
from .game import BatchGame
from .oracles import poa_bound, predicted_equilibrium
from .region import TOL
from .rules import AllocationRule, ProportionalRule, UniformRule, UtilitarianRule
from .waterfill import max_min_fair_point

RULES = {
    "uniform": lambda beta: UniformRule(beta=beta),
    "proportional": lambda beta: ProportionalRule(beta=beta),
    "utilitarian": lambda beta: UtilitarianRule(beta=beta),
}


@dataclass
class RuleReport:
    """One rule's equilibrium on the batch (MW and the fee's money units)."""

    rule: str
    method: str                    # "theory", "dynamics", "no convergence", or "skipped"
    requests: list = field(default_factory=list)
    served: list = field(default_factory=list)
    total_served: float = float("nan")
    planner_total: float = float("nan")
    efficiency: float = float("nan")          # total served / planner's optimum
    price_of_anarchy: float = float("nan")
    poa_bound: float | None = None            # only for the uniform rule
    max_inflation: float = float("nan")       # largest request / served
    unserved_requested_mw: float = float("nan")
    commitment_on_unserved: float = float("nan")
    payoffs: list = field(default_factory=list)
    exploitability: float = float("nan")      # largest gain from a unilateral deviation
    binding_limits: list = field(default_factory=list)
    rounds: int | None = None
    note: str = ""


def _binding(region, s, rel=1e-6) -> list:
    if getattr(region, "is_implicit", False):
        return []
    load = region.A @ s
    return [region.limit_names[c] for c in range(region.m) if load[c] >= region.b[c] * (1 - rel) - TOL
            and region.b[c] > TOL]


def _equilibrium(game: BatchGame, start, slow: bool, max_rounds: int, grid: int):
    """``(requests, method, rounds, note)``."""
    pred = predicted_equilibrium(game)
    implicit = getattr(game.region, "is_implicit", False)
    if pred.get("known"):
        if pred["kind"] == "band":
            return np.asarray(pred["low"], float), "theory", None, (
                "flat fee: every profile in [w, w/beta] is an equilibrium, all serving w")
        if "requests" in pred:
            return np.asarray(pred["requests"], float), "theory", None, ""
    if implicit and not slow:
        return None, "skipped", None, "no closed form here, and searching an implicit region is slow (slow=True)"
    r, _, rounds = game.best_response_dynamics(start, max_rounds=max_rounds, method="search", grid=grid,
                                                tol=1e-6 * max(1.0, float(np.max(start))))
    if rounds is None:
        return r, "no convergence", None, f"best responses did not settle within {max_rounds} rounds"
    return r, "dynamics", rounds, ""


def compare_rules(region, rules=("uniform", "proportional", "utilitarian"), fee: Fee | None = None,
                  value: float | np.ndarray = 1.0, beta: float = 0.5, slow: bool = False,
                  max_rounds: int = 25, check: bool = True, grid: int = 120) -> list[RuleReport]:
    """Equilibrium outcome of each rule on ``region``.

    ``rules`` are names from ``RULES`` or :class:`AllocationRule` objects.
    ``fee`` defaults to a per-MW commitment of 0.4 (in units of the value).
    ``check=True`` computes each equilibrium's exploitability by search (a
    theory answer should give zero); ``grid`` sets the best-response search
    (grid points plus local refinement). On an implicit region (``DCOPFRegion``)
    only closed-form equilibria are computed unless ``slow=True``.
    """
    fee = PerMWCommitment(0.4) if fee is None else fee
    implicit = getattr(region, "is_implicit", False)
    planner, _ = region.max_total()
    w = max_min_fair_point(region)
    out = []
    for spec in rules:
        rule: AllocationRule = RULES[spec](beta) if isinstance(spec, str) else spec
        name = spec if isinstance(spec, str) else type(spec).__name__
        game = BatchGame(region, rule, fee, value)
        r, method, rounds, note = _equilibrium(game, w.copy(), slow, max_rounds, grid)
        if r is None:
            out.append(RuleReport(name, method, planner_total=planner, note=note))
            continue
        s = game.allocate(r)
        total = float(s.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            infl = np.where(s > TOL, r / s, np.where(r > TOL, np.inf, 1.0))
        kappa = fee.marginal_request_cost()
        searched = check and (not implicit or slow)
        gain = float(game.exploitability(r, grid=max(grid, 200)).max()) if searched else float("nan")
        out.append(RuleReport(
            rule=name, method=method, requests=r.tolist(), served=s.tolist(), total_served=total,
            planner_total=planner, efficiency=total / planner if planner > 0 else float("nan"),
            price_of_anarchy=planner / total if total > 0 else float("inf"),
            poa_bound=poa_bound(region) if isinstance(rule, UniformRule) and rule.weights is None else None,
            max_inflation=float(np.max(infl)), unserved_requested_mw=float(np.sum(r - s)),
            commitment_on_unserved=float(kappa * np.sum(r - s)), payoffs=game.payoffs(r, s).tolist(),
            exploitability=gain, binding_limits=_binding(region, s), rounds=rounds, note=note))
    return out


# --------------------------------------------------------------------- output
def _f(x, d=2):
    if x is None:
        return "-"
    if isinstance(x, float) and not np.isfinite(x):
        return "inf" if x > 0 else "-"
    return f"{x:,.{d}f}"


def format_table(reports: list[RuleReport], applicant_names=None) -> str:
    """Plain-text table, one row per rule, then served MW per applicant."""
    head = ["rule", "method", "served MW", "planner MW", "efficiency", "PoA", "PoA bound", "inflation",
            "unserved req. MW", "exploitability"]
    rows = [[r.rule, r.method, _f(r.total_served), _f(r.planner_total), _f(r.efficiency, 3),
             _f(r.price_of_anarchy, 3), _f(r.poa_bound, 3), _f(r.max_inflation, 3), _f(r.unserved_requested_mw),
             _f(r.exploitability, 4)] for r in reports]
    widths = [max([len(h)] + [len(row[k]) for row in rows]) for k, h in enumerate(head)]
    def line(cells):
        return "  ".join(c.rjust(wd) if k > 1 else c.ljust(wd) for k, (c, wd) in enumerate(zip(cells, widths)))

    out = [line(head), line(["-" * wd for wd in widths])] + [line(row) for row in rows]
    n = max((len(r.served) for r in reports), default=0)
    if n:
        names = list(applicant_names or [f"applicant_{i}" for i in range(n)])
        out += ["", "served MW (requested MW)".ljust(24) + "".join(nm.rjust(22) for nm in names)]
        for r in reports:
            if r.served:
                cells = [f"{_f(s)} ({_f(q)})" for s, q in zip(r.served, r.requests)]
                out.append(r.rule.ljust(24) + "".join(c.rjust(22) for c in cells))
    notes = [f"{r.rule}: {r.note}" for r in reports if r.note]
    return "\n".join(out + ([""] + notes if notes else []))


def to_json(reports: list[RuleReport]) -> str:
    """The reports as JSON (non-finite numbers as null)."""
    def clean(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v
    return json.dumps([{k: clean(v) for k, v in asdict(r).items()} for r in reports], indent=1)


_CSS = (
    ":root{--bg:#fff;--fg:#1d1d1f;--muted:#595959;--line:#d9d9d9;--net:#1F4E79;--light:#BDD7EE;--bind:#C0392B}"
    "@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#ececec;--muted:#a6a6a6;--line:#333;"
    "--net:#7fb0e0;--light:#2c4a66;--bind:#e0705f}}"
    "body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,sans-serif;"
    "margin:0 auto;max-width:960px;padding:24px 16px}"
    "h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 20px}"
    ".wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}"
    "th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}"
    "th:first-child,td:first-child{text-align:left}"
    "th{color:var(--muted);font-weight:600}h2{font-size:17px;margin:28px 0 8px}.note{color:var(--muted);font-size:13px}"
    ".bar{fill:var(--net)}.req{fill:none;stroke:var(--muted);stroke-dasharray:3 2}"
    "svg text{fill:var(--muted);font-size:11px}"
)


def report_html(reports: list[RuleReport], path, title: str = "Allocation rule comparison", subtitle: str = "",
                applicant_names=None) -> None:
    """Write a self-contained HTML report (table plus served-versus-requested bars per rule)."""
    esc = html.escape
    cols = [("Rule", lambda r: r.rule), ("Method", lambda r: r.method), ("Served MW", lambda r: _f(r.total_served)),
            ("Planner MW", lambda r: _f(r.planner_total)), ("Efficiency", lambda r: _f(r.efficiency, 3)),
            ("PoA", lambda r: _f(r.price_of_anarchy, 3)), ("PoA bound", lambda r: _f(r.poa_bound, 3)),
            ("Max inflation", lambda r: _f(r.max_inflation, 3)),
            ("Unserved request MW", lambda r: _f(r.unserved_requested_mw)),
            ("Exploitability", lambda r: _f(r.exploitability, 4))]
    table = "<tr>" + "".join(f"<th>{esc(c)}</th>" for c, _ in cols) + "</tr>"
    for r in reports:
        table += "<tr>" + "".join(f"<td>{esc(str(fn(r)))}</td>" for _, fn in cols) + "</tr>"
    n = max((len(r.served) for r in reports), default=0)
    names = list(applicant_names or [f"applicant {i}" for i in range(n)])
    top = max([max(r.requests + r.served) for r in reports if r.served] + [1.0])
    charts = ""
    for r in reports:
        if not r.served:
            continue
        bw, gap, hgt = 36, 18, 140
        width = n * (bw + gap) + gap
        svg = [f'<svg viewBox="0 0 {width} {hgt + 34}" width="{width}" height="{hgt + 34}" role="img" '
               f'aria-label="served and requested MW, {esc(r.rule)}">']
        for i, (s, q) in enumerate(zip(r.served, r.requests)):
            x = gap + i * (bw + gap)
            hs, hq = hgt * s / top, hgt * min(q, top) / top
            svg.append(f'<rect class="bar" x="{x}" y="{hgt - hs:.1f}" width="{bw}" height="{hs:.1f}"/>')
            svg.append(f'<rect class="req" x="{x}" y="{hgt - hq:.1f}" width="{bw}" height="{hq:.1f}"/>')
            svg.append(f'<text x="{x + bw / 2}" y="{hgt + 14}" text-anchor="middle">{esc(str(names[i]))[:10]}</text>')
            svg.append(f'<text x="{x + bw / 2}" y="{hgt + 28}" text-anchor="middle">{_f(s, 1)}</text>')
        svg.append("</svg>")
        binding = ", ".join(r.binding_limits[:6]) + (" ..." if len(r.binding_limits) > 6 else "")
        charts += (f"<h2>{esc(r.rule)}</h2><div class='wrap'>{''.join(svg)}</div>"
                   f"<p class='note'>Bars: served MW; dashed outline: requested MW. "
                   f"{('Binding limits: ' + esc(binding) + '. ') if binding else ''}{esc(r.note)}</p>")
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{esc(title)}</title><style>{_CSS}</style></head>"
            f"<body><h1>{esc(title)}</h1><p class='sub'>{esc(subtitle)}</p><div class='wrap'><table>{table}</table>"
            f"</div><p class='note'>Efficiency is the total served over the planner's optimum (PoA is its inverse); "
            f"inflation is the largest request over its served load; exploitability is the largest gain any "
            f"applicant has from changing its request alone (zero at an equilibrium).</p>{charts}</body></html>")
    from pathlib import Path

    Path(path).write_text(page)


__all__ = ["RULES", "RuleReport", "compare_rules", "format_table", "report_html", "to_json"]
