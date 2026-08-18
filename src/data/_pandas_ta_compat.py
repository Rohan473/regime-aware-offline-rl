"""Reproducible compatibility layer for pandas-ta-openbb on Python 3.12.

Upstream bug (pandas-ta-openbb==0.4.24, ``pandas_ta/maps.py``): it does
``import importlib`` then accesses ``importlib.metadata`` without importing
the submodule. On Python 3.12 the ``metadata`` submodule is not bound by the
plain ``import importlib``, so importing ``pandas_ta`` raises
``AttributeError: module 'importlib' has no attribute 'metadata'``.

Fix: ``import importlib.metadata`` here BEFORE importing ``pandas_ta``. The
import machinery binds ``metadata`` as an attribute of the shared ``importlib``
module object, so ``pandas_ta/maps.py``'s later attribute access works. This is
a one-line, versioned, fully reproducible fix — no site-packages mutation, and
it stays correct even if the upstream package is ever fixed.

The probe below turns any residual failure (e.g. a broken install) into a
loud error with the remedy, instead of a cryptic AttributeError downstream.
"""

from __future__ import annotations

import importlib.metadata  # noqa: F401 - binds submodule; fixes the lazy-import bug

import pandas as pd
import pandas_ta as ta  # noqa: F401 - re-exported for callers

__all__ = ["ta"]


def _probe() -> None:
    try:
        ta.rsi(pd.Series([1.0, 2.0, 3.0, 4.0]), length=2)
    except Exception as exc:  # noqa: BLE001 - convert to actionable error
        raise RuntimeError(
            "pandas_ta (pandas-ta-openbb) is not functional in this "
            "environment. The known Python 3.12 import bug should already be "
            "handled by this module; if you see this, run:\n"
            "    python scripts/apply_patches.py\n"
            "or reinstall with: pip install -r requirements.txt"
        ) from exc


_probe()