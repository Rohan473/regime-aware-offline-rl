"""Regime labeling tests: non-degeneracy, hysteresis (no single-day
flicker), crisis priority, entry confirmation, and exit band behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.regime_labeling import (
    BULL,
    BEAR,
    CRISIS,
    _apply_crisis_hysteresis,
    _runs_of,
    label_regimes,
)
from tests.conftest import (
    BULL_DAYS,
    BEAR_DAYS,
    CRISIS_DAYS,
    DEFAULT_CFG,
    synthetic_daily_ohlcv,
)

R = DEFAULT_CFG.regimes


def labels_of(daily: pd.DataFrame) -> pd.Series:
    return label_regimes(daily["close"], DEFAULT_CFG)


def test_all_three_regimes_appear():
    labels = labels_of(synthetic_daily_ohlcv()).dropna()
    assert {BULL, BEAR, CRISIS} <= set(labels)


def test_labels_valid_and_no_nan_in_labeled_range():
    labels = labels_of(synthetic_daily_ohlcv())
    labeled = labels.dropna()
    assert set(labeled.unique()) <= {BULL, BEAR, CRISIS}
    # labeled range must be contiguous from warm-up onward (no internal NaN)
    assert len(labeled) == labels.notna().sum()
    assert labeled.index.min() == labels.index[labels.notna()].min()


def test_no_single_day_flicker():
    """Every labeled run must be >= min_regime_duration_days."""
    labels = labels_of(synthetic_daily_ohlcv()).dropna()
    min_dur = int(R.hysteresis.min_regime_duration_days)
    for s, e, _label in _runs_of(labels):
        assert e - s + 1 >= min_dur, f"run {s}..{e} shorter than {min_dur}"


def test_crisis_overrides_bull():
    """Days with a positive 60d trailing return inside the crisis block must
    still be labeled crisis (crisis priority over bull/bear)."""
    daily = synthetic_daily_ohlcv()
    labels = labels_of(daily)
    close = daily["close"]
    trailing = np.log(close).diff(60)
    crisis_start = BULL_DAYS + BEAR_DAYS + 12  # after crisis entry confirmation (bubble phase)
    slice_end = BULL_DAYS + BEAR_DAYS + 30
    band = daily.index[crisis_start:slice_end]
    pos_ret = trailing.loc[band] > 0
    assert pos_ret.any(), "fixture must contain positive trailing returns in crisis"
    assert (labels.loc[band][pos_ret] == CRISIS).all()


def test_single_day_vol_spike_is_not_crisis():
    """A one-day vol spike must not produce a crisis label on the spike day
    itself (entry needs crisis_confirmation_days consecutive days)."""
    daily = synthetic_daily_ohlcv()
    close = daily["close"].copy()
    spike_idx = BULL_DAYS // 2  # deep inside the calm bull block
    close.iloc[spike_idx] *= 0.93  # -7% single-day shock
    labels = label_regimes(close, DEFAULT_CFG)
    assert labels.iloc[spike_idx] != CRISIS


def test_crisis_entry_needs_confirmation():
    vol = pd.Series([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    threshold = pd.Series([0.03] * 8)
    confirm = 2
    out = _apply_crisis_hysteresis(vol, threshold, confirm, 0.8, 0.0)
    # entry on day 1 (0-indexed): after 2 consecutive qualifying days
    assert out.tolist() == [False, True, True, True, True, True, True, True]


def test_crisis_exit_uses_frozen_entry_threshold():
    """Once in crisis, exit requires vol < exit_mult * threshold-frozen-at-entry.
    A threshold that creeps up mid-crisis must NOT cause a premature exit."""
    n = 30
    vol = pd.Series([0.05] * 10 + [0.30] * 20)  # calm then extreme crisis
    threshold = pd.Series([0.04] * n)
    out = _apply_crisis_hysteresis(vol, threshold, 2, 0.8, 0.0).tolist()
    # crisis persists while vol stays above 0.8 * 0.04 = 0.032
    assert out == [False, True] + [True] * (n - 2)


def test_crisis_exits_only_after_band_and_confirmation():
    n = 40
    vol = pd.Series([0.05] * 10 + [0.035] * 15 + [0.03] * 15)  # crisis -> borderline -> calm
    threshold = pd.Series([0.04] * n)
    out = _apply_crisis_hysteresis(vol, threshold, 2, 0.8, 0.0).tolist()
    # 0.035 is inside the band (>= 0.032): still crisis.
    # 0.03 < 0.032: exit after 2 consecutive days.
    assert out[10] and out[20]
    assert out[24:27] == [True, True, False]  # borderline days crisis, then exit
    assert out[-1] is False  # exited once calm


def test_crisis_persists_through_long_block_with_expanding_threshold():
    """With an EXPANDING-window threshold, the live crisis bar rises as
    crisis days accumulate (100 crisis days of 800 = 12.5% > 5%), yet the
    crisis must persist through the entire block because exit uses the
    FROZEN entry threshold, never the live one."""
    daily = synthetic_daily_ohlcv()
    labels = labels_of(daily)
    block_end = BULL_DAYS + BEAR_DAYS + CRISIS_DAYS - 1
    assert labels.iloc[block_end] == CRISIS


def test_hysteresis_output_is_stable_under_small_threshold_changes():
    """Tuning a threshold must not flip labels wholesale (sanity for the
    config-driven workflow)."""
    from omegaconf import OmegaConf

    base = labels_of(synthetic_daily_ohlcv()).dropna()
    tweaked = OmegaConf.merge(DEFAULT_CFG)
    tweaked.regimes.bull_threshold = 0.005
    alt = label_regimes(synthetic_daily_ohlcv()["close"], tweaked).dropna()
    shared = base.index.intersection(alt.index)
    agreement = (base.loc[shared] == alt.loc[shared]).mean()
    assert agreement > 0.85


def test_config_validation_guards_bad_percentile():
    from omegaconf import OmegaConf

    from src.data.pipeline import validate_config

    bad = OmegaConf.merge(DEFAULT_CFG)
    bad.regimes.crisis.percentile = 110.0
    with pytest.raises(ValueError):
        validate_config(bad)
