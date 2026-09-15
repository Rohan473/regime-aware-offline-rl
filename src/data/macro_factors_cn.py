"""CSI300 cross-asset / volatility features for the C+ sign model (China
proxies for the SPY macro set, PROJECT_NOTES 7.18).

The SPY version (``macro_factors.py``) uses 8 non-correlated drivers. For
CSI300, East Money is blocked and several US-style series have no clean China
analog, so this is the best-effort full set (see scripts/fetch_macro_cn.py):

  rs_500_1d    CSI300_ret - CSI500_ret    (large-cap vs small/cyclical  <- rs_iwm)
  rs_growth_1d CSI300_ret - ChiNext_ret   (blue-chip vs growth          <- rs_qqq)
  rs_ss50_1d   CSI300_ret - SSE50_ret     (broad market vs largest caps <- extra)
  qvix_chg_1d  1d change in 50ETF implied-vol level (vol regime)        <- vol_term)
  qvix_chg_5d  5d change in 50ETF implied-vol level
  qvix_level   50ETF implied-vol level (z-scored causally downstream)

All features are causal (only data up to t). Output is a DataFrame indexed by
the CSI300 daily calendar with raw columns; causal z-scoring is left to the
caller, matching the SPY feature protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.macro_factors import zscore_causal  # noqa: E402

MACRO_CN_DIR = Path(__file__).resolve().parents[2] / "data" / "macro_cn"

MACRO_FEATURES_CN = [
    "rs_500_1d",
    "rs_growth_1d",
    "rs_ss50_1d",
    "qvix_chg_1d",
    "qvix_chg_5d",
    "qvix_level",
]


def _ret(df: pd.DataFrame) -> pd.Series:
    close = df["adj_close"].astype(float)
    return close.pct_change()


def load_macro_cn(macro_dir: Path = MACRO_CN_DIR) -> dict[str, pd.DataFrame]:
    names = ["csi500", "growth", "ss50", "qvix"]
    out = {}
    for n in names:
        df = pd.read_parquet(macro_dir / f"{n}.parquet")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        out[n] = df
    return out


def macro_features_cn(csi_daily: pd.DataFrame, macro_dir: Path = MACRO_CN_DIR) -> pd.DataFrame:
    """Raw CSI300 macro feature columns indexed by the CSI300 daily calendar.

    ``csi_daily`` must carry a ``close`` column and a DatetimeIndex (naive).
    Rows where a series is missing (pre-inception, e.g. QVIX before 2015-02,
    ChiNext before 2010-06, CSI500 before 2005) stay NaN — the caller drops
    them, matching the SPY protocol.
    """
    macro = load_macro_cn(macro_dir)
    idx = pd.DatetimeIndex(csi_daily.index).tz_localize(None)
    csi_ret = pd.Series(csi_daily["close"].astype(float).to_numpy(), index=idx).pct_change()

    rs500_ret = _ret(macro["csi500"]).reindex(idx)
    rsgrw_ret = _ret(macro["growth"]).reindex(idx)
    rsss50_ret = _ret(macro["ss50"]).reindex(idx)
    qvix = macro["qvix"]["close"].astype(float).reindex(idx)

    out = pd.DataFrame(index=idx)
    out["rs_500_1d"] = csi_ret - rs500_ret
    out["rs_growth_1d"] = csi_ret - rsgrw_ret
    out["rs_ss50_1d"] = csi_ret - rsss50_ret
    out["qvix_chg_1d"] = qvix.diff()
    out["qvix_chg_5d"] = qvix.diff(5)
    out["qvix_level"] = qvix
    return out[MACRO_FEATURES_CN]


__all__ = [
    "MACRO_CN_DIR",
    "MACRO_FEATURES_CN",
    "load_macro_cn",
    "macro_features_cn",
    "zscore_causal",
    "np",
]
