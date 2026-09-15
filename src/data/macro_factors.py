"""Macro / cross-asset features for the SPY models (causal, date-aligned).

Adds NON-CORRELATED drivers the 8 SPY technicals cannot express — rates,
volatility term structure, dollar, credit stress, and cross-market momentum
— per the user's Strategy-1 spec (PROJECT_NOTES 7.18). All features use only
data up to and including day t (trailing windows / causal deltas / close
based), so they carry no lookahead.

Input: data/macro/*.parquet (daily closes, from scripts/fetch_macro.py) plus
the SPY daily frame (for SPY returns). Output: one DataFrame indexed by the
SPY daily calendar with one RAW column per feature (causal z-scoring is left
to the caller so the sign model can normalize like the 8 state features).

Feature set (8), all causal:
  risk_on_1d    SPY_ret - TLT_ret          (risk-on vs flight-to-safety)
  tnx_delta_1d  change in 10y yield (1d)    (discount-rate pressure)
  tnx_delta_5d  change in 10y yield (5d)
  vol_term      VIX3M - VIX                 (contango / inverted term structure)
  dxy_corr_20d  trailing 20d corr(SPY_ret, DXY_ret)
  credit_1d     HYG_ret - TLT_ret           (credit-stress proxy)
  rs_qqq_1d     SPY_ret - QQQ_ret           (large-cap vs tech)
  rs_iwm_1d     SPY_ret - IWM_ret           (large-cap vs small/cyclical)

Sentiment (17th feature for the C+ sign model):
  sentiment_skew  CBOE SKEW index level (options tail-risk / put-demand
                  sentiment; high skew = fear of downside tail). Causal
                  z-scoring by the caller.

DEVIATION FLAGGED: the credit feature is PRICE-based (HYG_ret - TLT_ret), a
proxy for the true HYG_yield - TLT_yield spread (yield series unavailable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

MACRO_DIR = Path(__file__).resolve().parents[2] / "data" / "macro"

MACRO_FEATURES = [
    "risk_on_1d",
    "tnx_delta_1d",
    "tnx_delta_5d",
    "vol_term",
    "dxy_corr_20d",
    "credit_1d",
    "rs_qqq_1d",
    "rs_iwm_1d",
]

SENTIMENT_FEATURES = ["sentiment_skew"]


def _adj_returns(df: pd.DataFrame) -> pd.Series:
    close = df["adj_close"].astype(float)
    return close.pct_change()


def load_macro(macro_dir: Path = MACRO_DIR) -> dict[str, pd.DataFrame]:
    names = ["tlt", "tnx", "vix", "vix3m", "dxy", "hyg", "qqq", "iwm", "skew"]
    out = {}
    for n in names:
        out[n] = pd.read_parquet(macro_dir / f"{n}.parquet")
    return out


def _rolling_corr_on_overlap(
    x: pd.Series, y: pd.Series, window: int = 20, min_periods: int = 20
) -> pd.Series:
    """Trailing rolling correlation of ``x`` and ``y`` computed only on dates
    where BOTH are finite, aligned per-date, then re-indexed back to ``x``'s
    original calendar.

    Fixes a calendar-misalignment artifact (PROJECT_NOTES 7.31): SPY and DXY
    trade on slightly different days, so ``x.rolling(window).corr(y)`` over the
    SPY calendar goes NaN for ~window rows around any single missing ``y`` day.
    Here we first drop rows where either series is missing (the moving window
    is then well-defined on the shared calendar), compute the rolling
    correlation on that clean series, then forward-fill the result back onto
    the full calendar from the most recent valid value — strictly causal (a
    value at t uses only returns up to t).
    """
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    corr = pair["x"].rolling(window, min_periods=min_periods).corr(pair["y"])
    out = corr.reindex(x.index)
    return out.ffill()


def macro_features(spy_daily: pd.DataFrame, macro_dir: Path = MACRO_DIR) -> pd.DataFrame:
    """Raw macro feature columns indexed by the SPY daily calendar.

    ``spy_daily`` must carry a ``close`` column and a DatetimeIndex. Rows
    where a macro series is missing (pre-inception, e.g. HYG before 2007-04)
    stay NaN — the caller drops them.
    """
    macro = load_macro(macro_dir)
    idx = pd.DatetimeIndex(spy_daily.index).tz_localize(None)  # naive SPY dates
    spy_ret = pd.Series(spy_daily["close"].astype(float).to_numpy(), index=idx).pct_change()

    tlt_ret = _adj_returns(macro["tlt"]).reindex(idx)
    hyg_ret = _adj_returns(macro["hyg"]).reindex(idx)
    qqq_ret = _adj_returns(macro["qqq"]).reindex(idx)
    iwm_ret = _adj_returns(macro["iwm"]).reindex(idx)
    dxy_ret = _adj_returns(macro["dxy"]).reindex(idx)

    tnx = macro["tnx"]["close"].astype(float).reindex(idx)
    vix = macro["vix"]["close"].astype(float).reindex(idx)
    vix3m = macro["vix3m"]["close"].astype(float).reindex(idx)

    out = pd.DataFrame(index=idx)
    out["risk_on_1d"] = spy_ret - tlt_ret
    out["tnx_delta_1d"] = tnx.diff()
    out["tnx_delta_5d"] = tnx.diff(5)
    out["vol_term"] = vix3m - vix
    out["dxy_corr_20d"] = _rolling_corr_on_overlap(spy_ret, dxy_ret, window=20)
    out["credit_1d"] = hyg_ret - tlt_ret
    out["rs_qqq_1d"] = spy_ret - qqq_ret
    out["rs_iwm_1d"] = spy_ret - iwm_ret
    out["sentiment_skew"] = macro["skew"]["close"].astype(float).reindex(idx)
    return out[MACRO_FEATURES + SENTIMENT_FEATURES]


def zscore_causal(
    series: pd.Series, min_periods: int = 60, mode: str = "expanding"
) -> pd.Series:
    """Causal z-score (only data up to t), matching the SPY feature protocol."""
    if mode == "expanding":
        mean = series.expanding(min_periods=min_periods).mean()
        std = series.expanding(min_periods=min_periods).std()
    else:
        raise ValueError(f"unknown mode {mode}")
    z = (series - mean) / std.replace(0, pd.NA)
    z.loc[std.isna() | (std == 0)] = 0.0
    return z