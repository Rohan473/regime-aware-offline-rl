"""Windowed representation-lab data over an arbitrary causal feature subset.

The canonical rep-lab data is the 8-d z-scored DDR market path (see
``load_ddr_data``). For the feature-scaling experiment the encoder must see a
wider (or narrower) NESTED feature subset while keeping the SAME decision
dates, next-day returns, regimes and validity mask, so every run is
comparable. ``load_rep_data`` rebuilds only the windows from the 32-feature
bank (``src.data.feature_bank``); all other fields come from the canonical
DDR path unchanged.

``cfg.feature_cols is None`` -> canonical 8 (bit-identical to ``load_ddr_data``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from src.data.feature_bank import build_feature_bank, causal_zscore
from src.models.ddr.data import DDRData, load_ddr_data


@lru_cache(maxsize=1)
def _bank_z() -> pd.DataFrame:
    """The 32-feature bank, causally z-scored, built once per process."""
    return causal_zscore(build_feature_bank())


def load_rep_data(cfg) -> DDRData:
    """DDRData with windows over ``cfg.feature_cols`` (canonical 8 if None)."""
    base = load_ddr_data(cfg.window)
    cols = getattr(cfg, "feature_cols", None)
    if not cols:
        return base

    z = _bank_z()
    zcols = [f"z_{c}" for c in cols]  # bank columns are the causal z-scores
    missing = [c for c in zcols if c not in z.columns]
    if missing:
        raise KeyError(f"feature_cols not in bank: {missing}")
    pos = z.index.get_indexer(base.dates)
    if (pos < 0).any():
        raise ValueError("rep-lab dates missing from the feature bank calendar")

    mat = z[zcols].to_numpy(dtype="float64")
    W = np.stack([mat[p - cfg.window + 1: p + 1] for p in pos])
    if not np.isfinite(W).all():
        raise ValueError("feature-bank windows contain non-finite values")
    return DDRData(
        windows=torch.tensor(W, dtype=torch.float32),
        dates=base.dates,
        next_returns=base.next_returns,
        regimes=base.regimes,
        valid=base.valid,
    )
