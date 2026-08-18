"""Vendored patch for the pandas-ta-openbb Python 3.12 import bug.

The upstream package (pandas-ta-openbb==0.4.24) has a lazy-import bug in
``pandas_ta/maps.py`` that breaks ``import pandas_ta`` on Python 3.12. This
script patches the installed copy in site-packages so that ANY direct import
(notebooks, user code) works, not just code routed through
``src/data/_pandas_ta_compat.py``.

- Idempotent: a marker comment makes re-runs a no-op.
- Verifies: runs ``python -c "import pandas_ta"`` after patching.
- Safe: backs up the original bytes in memory and restores on verification
  failure. The patch is a single added import line.

Usage:
    python scripts/apply_patches.py
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

MARKER = "# gstack/RL-trading compat patch (see scripts/apply_patches.py)"
OLD_LINE = "import importlib"
NEW_LINE = "import importlib.metadata  " + MARKER


def find_maps_py() -> Path:
    try:
        dist = importlib.metadata.distribution("pandas-ta-openbb")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "pandas-ta-openbb is not installed. Run `pip install -r "
            "requirements.txt` first, then re-run this script."
        ) from exc
    candidates = [f for f in dist.files if str(f).endswith("pandas_ta/maps.py")]
    if not candidates:
        raise SystemExit("could not locate pandas_ta/maps.py in the installed package")
    return Path(dist.locate_file(candidates[0]))


def verify_import() -> bool:
    return (
        subprocess.run(
            [sys.executable, "-c", "import pandas_ta; print('import OK')"],
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


def main() -> int:
    maps_py = find_maps_py()
    original = maps_py.read_text(encoding="utf-8")

    if MARKER in original:
        print(f"[apply_patches] {maps_py}: already patched (marker present)")
        return 0 if verify_import() else 1

    if OLD_LINE not in original:
        raise SystemExit(
            f"unexpected contents in {maps_py}: expected a line `{OLD_LINE}` "
            f"to patch. The package may have been updated; check for a newer "
            f"upstream fix before proceeding."
        )

    patched = original.replace(OLD_LINE, OLD_LINE + "\n" + NEW_LINE, 1)
    maps_py.write_text(patched, encoding="utf-8")
    print(f"[apply_patches] patched {maps_py}")

    if not verify_import():
        maps_py.write_text(original, encoding="utf-8")
        raise SystemExit(
            "verification failed after patching; original file restored. "
            "Report this — the patch may be stale."
        )

    print("[apply_patches] verification OK: `import pandas_ta` works")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())