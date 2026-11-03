"""Check that a built wheel contains every bundled data file and module.

    python scripts/check_wheel.py dist/interconnection_gym-*.whl
"""

import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "src"
want = {str(p.relative_to(root)) for p in (root / "icgym").rglob("*")
        if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts}
have = set(zipfile.ZipFile(sys.argv[1]).namelist())
missing = sorted(want - have)
if missing:
    sys.exit(f"{len(missing)} files missing from the wheel, e.g. {missing[:5]}")
print(f"wheel has all {len(want)} modules and data files")
