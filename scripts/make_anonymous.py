"""Build an anonymized copy of the repository for double-blind review.

    python scripts/make_anonymous.py OUT_DIR [--name EXTRA ...]

Copies the files git tracks (or, outside git, every file not ignored below),
replaces the author names listed in ``pyproject.toml`` (and any ``--name``)
with "Anonymous Authors", drops the git history and this script, and then
searches the copy for every name, name part, and given extra string, case
insensitively. It exits with an error if anything is left.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "site", "dist", "build", "catalog", ".ruff_cache"}
SELF = Path(__file__).resolve().relative_to(ROOT)


def files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
        tracked = [Path(p) for p in out.splitlines() if p]
        if tracked:
            return tracked
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return [p.relative_to(ROOT) for p in ROOT.rglob("*")
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS) and not p.name.endswith(".egg-info")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--name", action="append", default=[], help="another string to remove (repeatable)")
    args = ap.parse_args()
    block = re.search(r"^authors\s*=\s*\[(.*?)\]", (ROOT / "pyproject.toml").read_text(), re.M | re.S)
    authors = re.findall(r'name\s*=\s*"([^"]+)"', block.group(1)) if block else []
    names = authors + args.name
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    pattern = re.compile("|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)), re.IGNORECASE)
    n = 0
    for rel in files():
        if rel == SELF:
            continue
        src, dst = ROOT / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = src.read_text()
        except UnicodeDecodeError:
            shutil.copy2(src, dst)
        else:
            if rel.name == "CITATION.cff":
                text = re.sub(r"authors:\n(  - .*\n(    .*\n)*)+", "authors:\n  - name: Anonymous Authors\n", text)
            dst.write_text(pattern.sub("Anonymous Authors", text))
        n += 1
    probes = sorted({p for name in names for p in [name, *name.split()] if len(p) >= 3}, key=str.lower)
    leaks = []
    for p in out.rglob("*"):
        if p.is_file():
            try:
                body = p.read_text()
            except UnicodeDecodeError:
                continue
            for probe in probes:
                for m in re.finditer(r"\b" + re.escape(probe) + r"\b", body, re.IGNORECASE):
                    leaks.append(f"{p.relative_to(out)}: {body[max(0, m.start() - 30):m.end() + 30]!r}")
    print(f"copied {n} files to {out}; searched for {len(probes)} strings")
    if leaks:
        print("\n".join(leaks[:20]))
        sys.exit(f"{len(leaks)} identifying strings left")
    print("no identifying strings found")


if __name__ == "__main__":
    main()
