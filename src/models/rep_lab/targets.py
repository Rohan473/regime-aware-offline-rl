"""Future-quantity targets for the predictive representation.

All targets are computed over the SAME DDR market path (close.pct_change),
at each decision date t:

  r1    = R_{t+1}                          next-day close-to-close return
  r5    = prod_{k=1..5}(1 + R_{t+k}) - 1   next-5-day cumulative return
  abs1  = |R_{t+1}|                        next-day magnitude
  vol5  = std(R_{t+1..t+5})                next-5-day realized volatility

The last four dates of a split lack full 5-day windows -> NaN rows that the
training loop masks out (the same isfinite() masking as the lab).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.ddr.data import DDRData, load_ddr_data


def predictive_targets(data: DDRData | None = None) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """(N, 4) targets + the dates they are aligned to."""
    if data is None:
        data = load_ddr_data(20)
    r = data.next_returns.numpy()
    N = len(r)
    out = np.full((N, 4), np.nan)
    out[:, 0] = r                                   # r1
    out[:, 2] = np.abs(r)                          # abs1
    for i in range(N - 4):
        out[i, 1] = float(np.prod(1.0 + r[i:i + 5]) - 1.0)   # r5
        out[i, 3] = float(np.std(r[i:i + 5]))              # vol5
    return out, data.dates


def split_targets(t_full: np.ndarray, full_dates: pd.DatetimeIndex,
                  split: DDRData) -> np.ndarray:
    """Row-align the full (N,4) targets to a split via its dates."""
    idx = np.searchsorted(np.asarray(full_dates), np.asarray(split.dates))
    return t_full[idx]