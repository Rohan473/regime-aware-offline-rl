"""Shared pytest fixtures: synthetic daily OHLCV + default config.

Synthetic data is used ONLY in tests, so the suite is deterministic and
runs without the (incremental) real download. The production pipeline never
falls back to synthetic data — it fails loudly instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data.loaders import REPO_ROOT

CONFIGS = OmegaConf.load(str(REPO_ROOT / "configs" / "data.yaml"))
REGIMES = OmegaConf.load(str(REPO_ROOT / "configs" / "regimes.yaml"))

DEFAULT_CFG = OmegaConf.merge(CONFIGS, {"regimes": REGIMES.regimes})

# Engineered calendar so every regime appears with clean margins:
#   bull (low vol, up) -> bear (LOW vol, down — bear is a price regime,
#   not a vol regime, so it must stay below the crisis vol threshold) ->
#   crisis (bubble 20d up + crash 80d, vol ~0.05 daily) -> recovery.
BULL_DAYS, BEAR_DAYS, BUBBLE_DAYS, CRASH_DAYS, RECOVERY_DAYS = 300, 150, 20, 80, 250
CRISIS_DAYS = BUBBLE_DAYS + CRASH_DAYS


def synthetic_daily_ohlcv(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = BULL_DAYS + BEAR_DAYS + CRISIS_DAYS + RECOVERY_DAYS

    returns = np.zeros(n)
    vol = np.zeros(n)
    t = 0
    seg = (
        (BULL_DAYS, 0.0004, 0.005),
        (BEAR_DAYS, -0.0006, 0.005),
        (BUBBLE_DAYS, 0.0040, 0.050),
        (CRASH_DAYS, -0.0050, 0.055),
        (RECOVERY_DAYS, 0.0005, 0.005),
    )
    for days, drift, sigma in seg:
        returns[t : t + days] = rng.normal(drift, sigma, days)
        vol[t : t + days] = sigma
        t += days

    close = 100.0 * np.exp(np.cumsum(returns))
    index = pd.bdate_range("2006-01-02", periods=n)
    df = pd.DataFrame(index=index)
    df["close"] = close
    df["open"] = np.concatenate([[100.0], close[:-1]])
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.abs(rng.normal(0, 0.001, n)))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.abs(rng.normal(0, 0.001, n)))
    df["volume"] = 1e7 * (1 + rng.normal(0, 0.1, n)) * np.where(vol >= 0.04, 3.0, 1.0)
    df["date"] = df.index
    return df


@pytest.fixture
def cfg():
    return OmegaConf.merge(DEFAULT_CFG)


@pytest.fixture
def daily():
    return synthetic_daily_ohlcv()


@pytest.fixture
def regime_cfg():
    return REGIMES
