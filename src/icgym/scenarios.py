"""Scenarios: regions to play on.

Synthetic regions for research and testing, small worked examples, and JSON
input/output so that regions computed from a network study (for example the
DCOPF hosting-capacity region of a set of buses) can be loaded.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

from .region import Region


def random_region(n: int, m: int, seed: int | None = None, density: float = 0.7,
                  skew: float = 1.0) -> Region:
    """Random downward-closed region with ``n`` applicants and ``m`` limits.

    ``density`` is the chance that an applicant loads a limit; ``skew > 1``
    spreads the loadings further apart.
    """
    rng = np.random.default_rng(seed)
    A = rng.random((m, n)) ** skew * (rng.random((m, n)) < density)
    for i in np.where(A.max(axis=0) <= 0)[0]:
        A[rng.integers(m), i] = rng.random() + 0.1
    b = rng.random(m) + 0.2
    return Region(A, b)


def flat_fee_example() -> Region:
    """Two applicants and one limit ``s_1 + 1.5 s_2 <= 3`` (paper, Fig. 5)."""
    return Region.single_limit([1.0, 1.5], 3.0, applicant_names=("site_1", "site_2"))


def poa_worst_case(n: int, R: float, k: int | None = None, eps: float = 1e-3) -> Region:
    """Regions that approach the price-of-anarchy bound ``(1+nR)^2/(4nR)``.

    ``k`` applicants with hosting capacity ``R`` barely load a shared limit
    that the other ``n - k``, with capacity 1, load fully.
    """
    if k is None:
        k = int(round((n * R - 1) / (2 * R)))
    k = min(max(k, 0), n - 1)
    rows = [np.concatenate([np.full(k, eps), np.ones(n - k)])]
    rhs = [1.0]
    for i in range(k):
        e = np.zeros(n)
        e[i] = 1.0
        rows.append(e)
        rhs.append(R)
    return Region(np.vstack(rows), rhs)


def save_region(region: Region, path: str | Path, signed=None, **metadata) -> None:
    """Write a region as JSON; ``signed=(A, b)`` stores the exact facets too."""
    from . import __version__

    data = {
        "format": "icgym.region.v1",
        "icgym_version": __version__,
        "A": region.A.tolist(),
        "b": region.b.tolist(),
        "limit_names": list(region.limit_names),
        "applicant_names": list(region.applicant_names),
        "metadata": metadata,
    }
    if signed is not None:
        data["signed_A"] = np.asarray(signed[0]).tolist()
        data["signed_b"] = np.asarray(signed[1]).tolist()
    Path(path).write_text(json.dumps(data, indent=1))


def _read(path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("format") != "icgym.region.v1":
        raise ValueError(f"{path} is not an icgym region file")
    return data


def load_region(path: str | Path) -> Region:
    data = _read(path)
    return Region(np.array(data["A"]), np.array(data["b"]),
                  limit_names=tuple(data["limit_names"]), applicant_names=tuple(data["applicant_names"]))


def load_region_record(path: str | Path) -> dict:
    """The full record: region, exact signed facets (if stored), and metadata."""
    data = _read(path)
    out = {"region": load_region(path), "metadata": data.get("metadata", {})}
    if "signed_A" in data:
        out["signed_A"], out["signed_b"] = np.array(data["signed_A"]), np.array(data["signed_b"])
    return out


def _data(system: str):
    return resources.files("icgym") / "data" / system


def _region_files(system: str):
    """Every bundled region file of ``system``, in the top folder and one level down."""
    root = _data(system)
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.is_dir() and entry.name != "ref":
            for f in sorted(entry.iterdir(), key=lambda p: p.name):
                if f.name.endswith(".json"):
                    yield f"{entry.name}/{f.name[:-5]}", f
        elif entry.name.endswith(".json"):
            yield entry.name[:-5], entry


def bundled(system: str = "activsg2000") -> list[str]:
    """Names of the regions shipped with the package for ``system`` (``'activsg2000'``).

    Names are ``group/buses_...`` for regions in a group folder (``pairs``,
    ``poa``, ``location``) and ``buses_...`` otherwise.
    """
    return [name for name, _ in _region_files(system)]


def _reorder(region: Region, have, want) -> Region:
    """Columns of ``region`` (applicants at buses ``have``) put in the order of ``want``."""
    free = list(range(len(have)))
    order = []
    for b in want:
        j = next(k for k in free if have[k] == b)
        free.remove(j)
        order.append(j)
    return Region(region.A[:, order], region.b, limit_names=region.limit_names,
                  applicant_names=tuple(np.array(region.applicant_names)[order]))


def bundled_region(system: str, buses) -> Region:
    """Bundled region of ``system`` for the bus multiset ``buses``, columns in the given order.

    Applicants at the same bus are interchangeable, so any bundled file with the
    same multiset of buses serves.
    """
    want = [str(b) for b in buses]
    for _, f in _region_files(system):
        stem = f.name[:-5]
        if not stem.startswith("buses_"):
            continue
        have = stem[len("buses_"):].split("_")
        if sorted(have) == sorted(want):
            return _reorder(load_region(f), have, want)
    raise KeyError(f"no bundled {system} region for buses {want}; see bundled({system!r})")


def ieee14(buses) -> Region:
    """IEEE 14-bus batch region for a bus set such as ``(9, 13, 14)``.

    Not shipped: the case file is downloaded from its public source on first
    use and the region is exported and cached locally (see :mod:`icgym.datasets`).
    """
    from .datasets import ieee14 as _ieee14

    return _ieee14(buses)


def activsg2000(buses) -> Region:
    """Bundled ACTIVSg2000 batch region (2,000 buses, five representative days).

    Exact exports for 150 bus pairs, 35 batches of two or three buses, and 8
    location multisets; see ``bundled('activsg2000')``. Larger batches need
    :class:`icgym.DCOPFRegion` on the network itself.
    """
    return bundled_region("activsg2000", buses)


def reference(system: str, name: str) -> dict:
    """Bundled reference results, e.g. ``reference('activsg2000', 'poa_implicit')``
    (25 batches of three and five buses evaluated with ``DCOPFRegion``) or
    ``'location_w'`` (max-min fair points of all 216 placements of the location game).
    """
    path = _data(system) / "ref" / f"{name}.json"
    if not path.is_file():
        raise KeyError(f"no reference {name!r} for {system}")
    return json.loads(path.read_text())
