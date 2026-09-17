"""A 32-feature point-in-time-valid bank for the representation-scaling study.

The project's canonical state is 8 technical features (see
``technical_factors.FEATURE_COLUMNS``). To study how much independent
financial information a representation actually needs, this module builds a
larger bank and exposes NESTED subsets of size 4 / 8 / 12 / 16 / 24 / 32 so
the 8-point is exactly the canonical state.

TEMPORAL VALIDITY, NOT CAUSAL DISCOVERY: every column is constructed from
information available no later than the decision time t (trailing windows /
close-based quantities only), and is z-scored with an expanding window that
uses only data up to t. This establishes temporal availability / no
look-ahead; it does NOT establish causality. The nested sets are controlled
expansions of the observable state, not a causal feature-selection result.

Groups (nested, order matters):
  4   core momentum+vol   ret_1d, ret_5d, ret_20d, realized_vol_20d
  8   canonical           + rsi_14, macd_hist, volume_zscore_20d, bollinger_pos
  12  + 4 macro           + risk_on_1d, vol_term, credit_1d, dxy_corr_20d
  16  + 4 macro           + tnx_delta_1d, tnx_delta_5d, rs_qqq_1d, rs_iwm_1d
  24  + 8 multi-horizon   + ret_10d/60d, vol_5d/60d, rsi_6/28, macd_signal,
                            bollinger_width
  32  + 8 structure/macro + volume_z_5d/60d, ma_ratio_5_20/20_60,
                            price_vs_ma_20/60, sentiment_skew, drawdown_60d

The canonical 8 are read straight from the processed frame (so the 8-point is
bit-identical to the production state); the other 24 are recomputed here from
close/volume + the macro parquet with the same formulas. ``causal_zscore``
("causal" here means no look-ahead: it applies the project's expanding
z-score, min_periods=60, using only data up to t) is applied to ALL 32
uniformly, so every feature is on the same scale regardless of subset size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.loaders import REPO_ROOT

from ._pandas_ta_compat import ta
from .macro_factors import MACRO_FEATURES, SENTIMENT_FEATURES, macro_features
from .technical_factors import FEATURE_COLUMNS

CORE_4 = ["ret_1d", "ret_5d", "ret_20d", "realized_vol_20d"]

CANONICAL_8 = list(FEATURE_COLUMNS)

MACRO_4 = ["risk_on_1d", "vol_term", "credit_1d", "dxy_corr_20d"]
MACRO_8 = MACRO_4 + ["tnx_delta_1d", "tnx_delta_5d", "rs_qqq_1d", "rs_iwm_1d"]

MULTIHORIZON_8 = [
    "ret_10d", "ret_60d", "realized_vol_5d", "realized_vol_60d",
    "rsi_6", "rsi_28", "macd_signal", "bollinger_width",
]

STRUCTURE_8 = [
    "volume_zscore_5d", "volume_zscore_60d", "ma_ratio_5_20", "ma_ratio_20_60",
    "price_vs_ma_20", "price_vs_ma_60", "sentiment_skew", "drawdown_60d",
]

FEATURE_BANK = CORE_4 + CANONICAL_8[4:] + MACRO_8 + MULTIHORIZON_8 + STRUCTURE_8

FEATURE_SETS: dict[int, list[str]] = {
    4: CORE_4,
    8: CORE_4 + CANONICAL_8[4:],
    12: CORE_4 + CANONICAL_8[4:] + MACRO_4,
    16: CORE_4 + CANONICAL_8[4:] + MACRO_8,
    24: CORE_4 + CANONICAL_8[4:] + MACRO_8 + MULTIHORIZON_8,
    32: FEATURE_BANK,
}

assert len(FEATURE_BANK) == 32, len(FEATURE_BANK)
assert len(set(FEATURE_BANK)) == 32, "duplicate feature names"
for _n in (4, 8, 12, 16, 24):
    assert FEATURE_SETS[_n] == FEATURE_BANK[:_n], f"set {_n} not nested"


def _log_ret(close: pd.Series, window: int) -> pd.Series:
    log_close = np.log(pd.Series(close.to_numpy(dtype="float64"), index=close.index))
    return log_close.diff().rolling(window).sum()


def _vol(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change().rolling(window).std() * (252 ** 0.5)


def _volume_z(volume: pd.Series, window: int) -> pd.Series:
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std.replace(0, pd.NA)


def build_feature_bank(processed_dir=None, macro_dir=None, cfg=None) -> pd.DataFrame:
    """32 raw causal feature columns indexed by the processed SPY calendar."""
    processed_dir = processed_dir or (REPO_ROOT / "data" / "processed")
    cfg = cfg or OmegaConf.load(str(REPO_ROOT / "configs" / "data.yaml"))
    feats = pd.read_parquet(f"{processed_dir}/features_regimes.parquet")
    close = feats["close"].astype(float)
    volume = feats["volume"].astype(float)

    out = pd.DataFrame(index=feats.index)
    for col in CANONICAL_8:  # canonical state read verbatim
        out[col] = feats[col].astype(float)

    f = cfg.features
    out["ret_10d"] = _log_ret(close, 10)
    out["ret_60d"] = _log_ret(close, 60)
    out["realized_vol_5d"] = _vol(close, 5)
    out["realized_vol_60d"] = _vol(close, 60)
    out["rsi_6"] = ta.rsi(close, length=6)
    out["rsi_28"] = ta.rsi(close, length=28)

    macd = ta.macd(close, fast=int(f.macd.fast), slow=int(f.macd.slow),
                   signal=int(f.macd.signal))
    out["macd_signal"] = pd.to_numeric(
        macd[[c for c in macd.columns if "MACDs" in c][0]], errors="coerce")

    mid = close.rolling(int(f.bollinger.window)).mean()
    sd = close.rolling(int(f.bollinger.window)).std()
    num_std = float(f.bollinger.num_std)
    out["bollinger_width"] = (2.0 * num_std * sd) / mid.replace(0, pd.NA)

    out["volume_zscore_5d"] = _volume_z(volume, 5)
    out["volume_zscore_60d"] = _volume_z(volume, 60)

    ma5, ma20, ma60 = close.rolling(5).mean(), close.rolling(20).mean(), close.rolling(60).mean()
    out["ma_ratio_5_20"] = ma5 / ma20 - 1.0
    out["ma_ratio_20_60"] = ma20 / ma60 - 1.0
    out["price_vs_ma_20"] = close / ma20 - 1.0
    out["price_vs_ma_60"] = close / ma60 - 1.0
    out["drawdown_60d"] = close / close.rolling(60).max() - 1.0

    macro = macro_features(feats, macro_dir) if macro_dir is not None else macro_features(feats)
    for col in MACRO_FEATURES + SENTIMENT_FEATURES:
        out[col] = macro[col].reindex(out.index)

    return out[FEATURE_BANK].astype("float64")


def causal_zscore(bank: pd.DataFrame, min_periods: int = 60) -> pd.DataFrame:
    """Expanding causal z-score of every bank column (matches the pipeline)."""
    z = pd.DataFrame(index=bank.index)
    for col in bank.columns:
        s = bank[col].astype("float64")
        mean = s.expanding(min_periods=min_periods).mean()
        std = s.expanding(min_periods=min_periods).std()
        zz = ((s - mean) / std).astype("float64")
        zz[~np.isfinite(zz.to_numpy())] = 0.0  # warm-up / zero-variance -> 0
        z[col] = zz
    z.columns = [f"z_{c}" for c in bank.columns]
    return z
