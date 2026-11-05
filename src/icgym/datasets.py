"""Optional datasets that are built on the user's machine, not shipped.

The IEEE 14-bus scenario combines the public IEEE 14-bus case (topology,
reactances, base loads, generator sizes) with our own additions (thermal
limits, which the case lacks, generator costs, and hourly load multipliers of
five representative days), which ship with the package under the MIT license.
On first use, :func:`fetch_ieee14` downloads the case file from MATPOWER's
repository (a conversion of the University of Washington Power Systems Test
Case Archive file), checks its SHA-256 hash, and writes the scenario to a
local cache; :func:`ieee14` then exports and caches the batch region of any
set of buses (a few seconds each).

The case file is not redistributed by this package. Its archive describes its
material as provided for research and education, not for commercial gain;
check the terms of the sources before commercial use. A case file obtained
elsewhere can be supplied with ``case_file=`` or the ``ICGYM_IEEE14_CASE``
environment variable. The cache lives in ``ICGYM_DATA_DIR`` (default
``~/.cache/icgym``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from importlib import resources
from pathlib import Path

import numpy as np

from .region import Region

_OVERLAY = "ieee14.json"


def data_dir() -> Path:
    """Root of the local cache (``ICGYM_DATA_DIR``, default ``~/.cache/icgym``)."""
    return Path(os.environ.get("ICGYM_DATA_DIR", Path.home() / ".cache" / "icgym"))


def _overlay() -> dict:
    return json.loads((resources.files("icgym") / "data" / "overlays" / _OVERLAY).read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matrix(text: str, name: str) -> np.ndarray:
    body = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", text, re.S)
    if body is None:
        raise ValueError(f"mpc.{name} not found in the case file")
    rows = [line.split("%")[0].strip().rstrip(";") for line in body.group(1).strip().splitlines()]
    return np.array([[float(x) for x in r.split()] for r in rows if r])


def _case_file(case_file, ov, quiet: bool) -> Path:
    spec = ov["case_file"]
    given = case_file or os.environ.get("ICGYM_IEEE14_CASE")
    if given:
        path = Path(given)
    else:
        path = data_dir() / "ieee14" / spec["name"]
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            if not quiet:
                print(f"icgym: downloading the IEEE 14-bus case from {spec['url']} "
                      "(provided by its sources for research and education)", file=sys.stderr)
            try:
                with urllib.request.urlopen(spec["url"], timeout=30) as resp:
                    data = resp.read()
            except OSError as exc:
                raise RuntimeError(f"could not download {spec['url']} ({exc}); download it yourself and pass "
                                   "case_file= or set ICGYM_IEEE14_CASE") from exc
            tmp = path.with_name(f"{path.name}.{os.getpid()}.part")   # one temp file per process
            tmp.write_bytes(data)
            tmp.replace(path)                                          # atomic; the last writer wins
    digest = _sha256(path)
    if digest != spec["sha256"]:
        raise ValueError(f"{path} has SHA-256 {digest}, expected {spec['sha256']} ({spec['source']})")
    return path


def fetch_ieee14(case_file: str | Path | None = None, quiet: bool = False) -> Path:
    """Build the IEEE 14-bus representative-day scenario locally; returns its directory.

    The directory holds the CSV files that :meth:`icgym.network.DCNetwork.from_csv_dir`
    and ``icgym-export --data`` read.
    """
    ov = _overlay()
    out = data_dir() / "ieee14" / "scenario"
    stamp = out / "built_from.json"
    if stamp.is_file() and json.loads(stamp.read_text()).get("case_sha256") == ov["case_file"]["sha256"]:
        return out
    text = _case_file(case_file, ov, quiet).read_text()
    bus, gen, branch = _matrix(text, "bus"), _matrix(text, "gen"), _matrix(text, "branch")
    final = out
    out = out.with_name(f"scenario.{os.getpid()}.tmp")   # build privately, then move into place
    out.mkdir(parents=True, exist_ok=True)
    ids = [str(int(b[0])) for b in bus]
    (out / "bus.csv").write_text("bus_id,type\n" + "".join(f"{i},{int(b[1])}\n" for i, b in zip(ids, bus)))
    lines = ["branch_id,from_bus_id,to_bus_id,reactance_pu,thermal_limit_mva,status"]
    for k, br in enumerate(branch, start=1):
        f, t = int(br[0]), int(br[1])
        limit = ov["thermal_limit_mva"][f"{f}-{t}"]
        lines.append(f"br{k:02d},{f},{t},{float(br[3])!r},{float(limit)!r},{int(br[10])}")
    (out / "branch_limits.csv").write_text("\n".join(lines) + "\n")
    extra = {g["bus_id"]: g for g in ov["generators"]}
    lines = ["generator_id,bus_id,p_min_mw,p_max_mw,marginal_cost_c1_per_mwh,status,profile_type"]
    for g in gen:
        b = str(int(g[0]))
        e = extra[b]
        lines.append(f"{e['generator_id']},{b},{float(e['p_min_mw'])!r},{float(g[8])!r},"
                     f"{float(e['marginal_cost_per_mwh'])!r},"
                     f"{int(g[7])},thermal")
    (out / "generator_limits.csv").write_text("\n".join(lines) + "\n")
    base = bus[:, 2]
    d = ov["demand_decimals"]
    rows = ["period," + ",".join(ids)]
    for t, m in enumerate(ov["period_multipliers"]):
        rows.append(f"{t}," + ",".join(repr(round(float(p) * m, d)) for p in base))
    (out / "demand_bus.csv").write_text("\n".join(rows) + "\n")
    (out / "dcline.csv").write_text("status,from_bus_id,to_bus_id,Pt\n")
    (out / "built_from.json").write_text(json.dumps({"case_sha256": ov["case_file"]["sha256"],
                                                    "case_source": ov["case_file"]["source"],
                                                    "overlay": "icgym/data/overlays/" + _OVERLAY}, indent=1))
    try:
        out.rename(final)          # atomic when the target does not exist yet
    except OSError:                # another process built it first (or an old copy exists)
        if (final / "built_from.json").is_file():
            shutil.rmtree(out, ignore_errors=True)
        else:
            shutil.rmtree(final, ignore_errors=True)
            out.rename(final)
    return final


def ieee14_network(case_file: str | Path | None = None):
    """The IEEE 14-bus scenario as a :class:`icgym.network.DCNetwork` (built on first use)."""
    from .network import DCNetwork

    return DCNetwork.from_csv_dir(fetch_ieee14(case_file))


def ieee14_ac_network(case_file: str | Path | None = None):
    """The IEEE 14-bus scenario as an :class:`icgym.ac_region.ACNetwork`, for AC studies.

    The same case file and overlay as :func:`ieee14_network`: the overlay's
    thermal limits become MVA limits, its multipliers scale active and reactive
    base loads alike, and voltage limits, reactive limits, line resistances, and
    shunts come from the case file.
    """
    from .ac_region import ACNetwork

    ov = _overlay()
    path = _case_file(case_file, ov, quiet=False)
    return ACNetwork.from_matpower(path, rate=ov["thermal_limit_mva"], period_multipliers=ov["period_multipliers"],
                                   pmin={g["bus_id"]: g["p_min_mw"] for g in ov["generators"]}, source="ieee14")


def ieee14(buses, case_file: str | Path | None = None, workers: int = 1) -> Region:
    """IEEE 14-bus batch region for a bus set such as ``(9, 13, 14)``, columns in the given order.

    Exported on first use from the locally built scenario (five representative
    days, 120 hourly periods) and cached; applicants at the same bus are
    interchangeable, so one export serves every order.
    """
    from .network import export_region
    from .scenarios import _reorder, load_region, save_region

    want = [str(b) for b in buses]
    key = sorted(want, key=lambda b: (int(b) if b.isdigit() else 0, b))
    path = data_dir() / "ieee14" / "regions" / ("buses_" + "_".join(key) + ".json")
    if not path.is_file():
        net = ieee14_network(case_file)
        ex = export_region(net, key, workers=workers)
        path.parent.mkdir(parents=True, exist_ok=True)
        ex.metadata["has_counterflow"] = ex.has_counterflow
        tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
        save_region(ex.region, tmp, signed=(ex.signed_A, ex.signed_b), **ex.metadata)
        tmp.replace(path)
    return _reorder(load_region(path), key, want)
