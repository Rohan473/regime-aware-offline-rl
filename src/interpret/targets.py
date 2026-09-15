"""Probe target construction for the representation laboratory.

All targets live on the shared SPY date grid (``features_regimes.parquet``,
clipped to SPLIT_TEST_END), strictly causal or forward-looking as labelled:

- ``fwd_ret_k`` / ``fwd_dir_k`` : forward k-day return / its sign (k=1,5,20)
- ``abs_ret_1``   : next-day |return| (magnitude probe)
- ``fwd_vol_k``   : realized vol over the NEXT k days (annualized, k=5,20)
- ``regime``      : Phase-1 regime label at t (bull/bear/crisis)
- ``z_*``         : current-day causally z-scored features at t (for the
                    descriptive/reconstruction probe)

``split`` is the shared time split (train <= 2018-12-31, val 2019-2020,
test 2021-2024) so probes always fit on train and eval on val/test — the
same protocol the models were evaluated with (no lookahead / leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loaders import REPO_ROOT
from src.models import SPLIT_TEST_END, SPLIT_TRAIN_END, SPLIT_VAL_END

Z_COLUMNS = [
    "z_ret_1d",
    "z_ret_5d",
    "z_ret_20d",
    "z_realized_vol_20d",
    "z_rsi_14",
    "z_macd_hist",
    "z_volume_zscore_20d",
    "z_bollinger_pos",
]


def _naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx


def build_targets(processed_dir: str | None = None) -> pd.DataFrame:
    """Target frame indexed by decision date (tz-aware, matching the loaders).

    Returns a DataFrame with one row per SPY trading day <= SPLIT_TEST_END.
    """
    processed_dir = processed_dir or (REPO_ROOT / "data" / "processed")
    feats = pd.read_parquet(f"{processed_dir}/features_regimes.parquet")
    keep = _naive(feats.index) <= pd.Timestamp(SPLIT_TEST_END)
    df = feats[keep].copy()
    close = df["close"]
    ret = close.pct_change()

    for k in (1, 5, 20):
        df[f"fwd_ret_{k}"] = close.shift(-k) / close - 1.0
        df[f"fwd_dir_{k}"] = np.sign(df[f"fwd_ret_{k}"])
        df.loc[df[f"fwd_dir_{k}"] == 0, f"fwd_dir_{k}"] = 1.0

    df["abs_ret_1"] = ret.shift(-1).abs()

    # forward realized vol over the next k days: reversed rolling std of the
    # next-day-return series (row t -> days t+1..t+k)
    for k in (5, 20):
        rev = ret.shift(-1).iloc[::-1].rolling(k).std().iloc[::-1]
        df[f"fwd_vol_{k}"] = rev * np.sqrt(252.0)

    naive = _naive(df.index)
    split = np.where(
        naive <= pd.Timestamp(SPLIT_TRAIN_END), "train",
        np.where(naive <= pd.Timestamp(SPLIT_VAL_END), "val", "test"),
    )
    df["split"] = split
    return df


def align(rep_dates: pd.DatetimeIndex, H: np.ndarray, targets: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Align a per-date representation matrix (N, D) onto the target grid.

    Returns (targets_subset, X): the target rows whose date appears in
    ``rep_dates``, and the representation rows in the same order. Both are
    indexed/ordered identically so probes can be fit on ``split == train``
    and evaluated on val/test without index bookkeeping.
    """
    rd = _naive(pd.DatetimeIndex(rep_dates))
    if len(rd) != H.shape[0]:
        raise ValueError(f"dates {len(rd)} != representations {H.shape[0]}")
    flat = targets.copy()
    flat.index = _naive(flat.index)
    flat = flat[~flat.index.duplicated(keep="first")]
    mask = flat.index.isin(pd.Index(rd))
    out = flat[mask].copy()
    hmap = {d: H[i] for i, d in enumerate(rd)}
    X = np.vstack([hmap[d] for d in out.index])
    return out, X