"""Technical feature computation (daily OHLCV -> feature frame).

Exactly 8 features are produced (cap is 10 per project scope; do not add more
without the user asking):

  ret_1d, ret_5d, ret_20d        log returns over 1/5/20 days
  realized_vol_20d               annualized std of daily returns (vol_window)
  rsi_14                         Wilder RSI
  macd_hist                      MACD(12,26) - signal(9)
  volume_zscore_20d              z-score of volume vs trailing window
  bollinger_pos                  (close - mid) / (2 * num_std * std) in [-0.5, 0.5]

``add_features`` returns the input frame plus one column per feature (raw
values). ``normalize_features`` then causally z-scores them for use as RL
state (expanding or rolling window; only data up to t is used, so no
lookahead). Raw features are kept in the processed frame; the offline dataset
consumes the normalized ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ._pandas_ta_compat import ta

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "realized_vol_20d",
    "rsi_14",
    "macd_hist",
    "volume_zscore_20d",
    "bollinger_pos",
]

LOG_RETURN_WINDOWS = {1: "ret_1d", 5: "ret_5d", 20: "ret_20d"}


def add_features(daily: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Append raw technical feature columns to the daily OHLCV frame."""
    if "close" not in daily.columns:
        raise ValueError("daily frame must contain a 'close' column")

    close = daily["close"]
    f = cfg.features
    out = daily.copy()

    # --- log returns -----------------------------------------------------
    log_close = np.log(pd.Series(close.to_numpy(dtype="float64"), index=close.index))
    log_ret = log_close.diff()
    for window, col in LOG_RETURN_WINDOWS.items():
        out[col] = log_ret.rolling(window).sum()

    # --- realized volatility (annualized) --------------------------------
    vol_window = int(f.realized_vol_window)
    daily_ret = close.pct_change()
    out["realized_vol_20d"] = daily_ret.rolling(vol_window).std() * (252**0.5)

    # --- RSI / MACD via pandas_ta ----------------------------------------
    out["rsi_14"] = ta.rsi(close, length=int(f.rsi_window))
    macd = ta.macd(
        close,
        fast=int(f.macd.fast),
        slow=int(f.macd.slow),
        signal=int(f.macd.signal),
    )
    hist_col = [c for c in macd.columns if "MACDh" in c][0]  # MACD - signal
    out["macd_hist"] = macd[hist_col]

    # --- volume z-score ----------------------------------------------------
    vol_window_z = int(f.volume_zscore_window)
    vol_mean = daily["volume"].rolling(vol_window_z).mean()
    vol_std = daily["volume"].rolling(vol_window_z).std()
    out["volume_zscore_20d"] = (daily["volume"] - vol_mean) / vol_std.replace(0, pd.NA)

    # --- Bollinger band position -------------------------------------------
    bb_window = int(f.bollinger.window)
    num_std = float(f.bollinger.num_std)
    mid = close.rolling(bb_window).mean()
    sd = close.rolling(bb_window).std()
    out["bollinger_pos"] = (close - mid) / (2.0 * num_std * sd.replace(0, pd.NA))

    # pandas_ta returns object dtypes in some versions; coerce to float
    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="raise")

    return out


def normalize_features(
    features: pd.DataFrame, cfg: DictConfig, *, min_periods: int | None = None
) -> pd.DataFrame:
    """Causally z-score the feature columns in place-style (new frame).

    Uses only data up to time t (expanding or trailing rolling window), so
    there is no lookahead. Constant features get z = 0 instead of NaN.
    """
    f = cfg.features
    if not f.normalize_state:
        return features.copy()

    min_periods = min_periods or int(f.normalize_min_periods)
    mode = str(f.normalize_mode)
    window = int(f.normalize_window)

    z = pd.DataFrame(index=features.index)
    for col in FEATURE_COLUMNS:
        if mode == "expanding":
            mean = features[col].expanding(min_periods=min_periods).mean()
            std = features[col].expanding(min_periods=min_periods).std()
        elif mode == "rolling":
            mean = features[col].rolling(window, min_periods=min_periods).mean()
            std = features[col].rolling(window, min_periods=min_periods).std()
        else:
            raise ValueError(f"unknown normalize_mode: {mode!r}")
        z[col] = (features[col] - mean) / std.replace(0, pd.NA)
        z.loc[std.isna() | (std == 0), col] = 0.0
    z.columns = [f"z_{c}" for c in FEATURE_COLUMNS]
    return z
