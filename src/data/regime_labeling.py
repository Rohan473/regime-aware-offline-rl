"""Regime labeling: {bull, bear, crisis} per trading day, with hysteresis.

Rules (all thresholds come from configs/regimes.yaml — nothing hardcoded):

1. Crisis candidate: trailing realized vol (``vol_window_days``, annualized)
   >= the ``percentile``-th percentile of ALL vol history up to that day
   (EXPANDING window, booting after ``percentile_min_periods`` days), and vol
   above an absolute floor. The expanding window keeps the crisis bar anchored
   to the long-run vol distribution: a long crisis (2008) cannot raise its own
   entry bar the way a trailing-window percentile would.

2. Bull/bear: trailing ``lookback_return_days`` log return vs
   ``bull_threshold`` (bull) and ``-bear_threshold`` (bear); in between,
   fall back to the *sign* of the trailing return (weak bull/bear).

3. Crisis overrides bull/bear whenever its raw condition holds.

Hysteresis (kills single-day flicker):
* Crisis entry requires ``crisis_confirmation_days`` consecutive raw-crisis
  days; on entry the threshold is FROZEN at its value when the entry streak
  started, and exit requires the same number of days below
  ``crisis_exit_mult * frozen_threshold`` (asymmetric band).
* After the state machine runs, any run of labels shorter than
  ``min_regime_duration_days`` is merged shortest-first into its longer
  neighbor (ties -> previous), until every run is at least that long.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

BULL, BEAR, CRISIS = "bull", "bear", "crisis"


def trailing_log_return(close: pd.Series, window: int) -> pd.Series:
    log_close = np.log(close.astype(float))
    return log_close.diff(window)


def realized_vol(close: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    vol = close.pct_change().rolling(window).std()
    return vol * (252**0.5) if annualize else vol


def _raw_labels(trailing_ret: pd.Series, cfg: DictConfig) -> pd.Series:
    """Bull/bear from trailing return sign + magnitude (no crisis yet)."""
    bull_thr = float(cfg.regimes.bull_threshold)
    bear_thr = float(cfg.regimes.bear_threshold)

    labels = pd.Series(index=trailing_ret.index, dtype="object")
    labels[trailing_ret > bull_thr] = BULL
    labels[trailing_ret < -bear_thr] = BEAR
    in_band = (trailing_ret <= bull_thr) & (trailing_ret >= -bear_thr)
    sign = pd.Series(
        np.where(trailing_ret[in_band].to_numpy() >= 0, BULL, BEAR),
        index=trailing_ret.index[in_band],
    )
    labels = labels.fillna(sign)
    return labels


def _blend_runs(labels: pd.Series, min_duration: int) -> pd.Series:
    """Merge every run shorter than min_duration into its longer neighbor
    (ties -> previous), iterating until all runs satisfy the minimum."""
    labels = labels.copy()
    while True:
        runs = _runs_of(labels)
        short_idx = [
            i for i, (s, e, _) in enumerate(runs) if (e - s + 1) < min_duration
        ]
        if not short_idx:
            break
        i = short_idx[0]
        s, e, _ = runs[i]
        len_prev = runs[i - 1][1] - runs[i - 1][0] + 1 if i > 0 else -1
        len_next = runs[i + 1][1] - runs[i + 1][0] + 1 if i < len(runs) - 1 else -1
        if len_prev >= len_next and i > 0:
            labels.iloc[s : e + 1] = labels.iloc[runs[i - 1][1]]
        elif i < len(runs) - 1:
            labels.iloc[s : e + 1] = labels.iloc[runs[i + 1][0]]
        else:
            labels.iloc[s : e + 1] = labels.iloc[runs[i - 1][1]]
    return labels


def _runs_of(labels: pd.Series) -> list[tuple[int, int, object]]:
    runs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels.iloc[i] != labels.iloc[start]:
            runs.append((start, i - 1, labels.iloc[start]))
            start = i
    return runs


def _apply_crisis_hysteresis(
    vol: pd.Series,
    threshold: pd.Series,
    confirm: int,
    exit_mult: float,
    min_vol: float,
) -> pd.Series:
    """Crisis state machine with asymmetric entry/exit band.

    Entry: `confirm` consecutive days with vol >= threshold and vol >= min_vol.
    On entry the threshold is *frozen* at its value when the entry streak
    started; exit requires `confirm` consecutive days with vol below
    `exit_mult * frozen_threshold`.

    Freezing the threshold matters: the *live* crisis threshold can rise as
    crisis days accumulate in the expanding window; exiting against the live
    threshold would end a long crisis early. The frozen entry threshold is
    the anchor for exit.

    Returns a boolean mask: True on days labeled crisis.
    """
    is_crisis_raw = (vol.to_numpy() >= threshold.to_numpy()) & (vol.to_numpy() >= min_vol)
    vol_raw = vol.to_numpy()
    thr_raw = threshold.to_numpy()

    in_crisis = False
    frozen_thr = np.nan
    entry_streak = 0
    streak_start_thr = np.nan
    exit_streak = 0
    out = np.zeros(len(vol), dtype=bool)
    for t in range(len(vol)):
        if not in_crisis:
            if is_crisis_raw[t]:
                if entry_streak == 0:
                    streak_start_thr = thr_raw[t]
                entry_streak += 1
                if entry_streak >= confirm and not np.isnan(streak_start_thr):
                    in_crisis = True
                    frozen_thr = streak_start_thr
                    entry_streak = 0
            else:
                entry_streak = 0
        else:
            exit_streak = (
                exit_streak + 1 if vol_raw[t] < exit_mult * frozen_thr else 0
            )
            if exit_streak >= confirm:
                in_crisis = False
                exit_streak = 0
        out[t] = in_crisis
    return pd.Series(out, index=vol.index)


def label_regimes(close: pd.Series, cfg: DictConfig) -> pd.Series:
    """Return a bull/bear/crisis label per day (NaN before the warm-up).

    The series index must be a daily DatetimeIndex (close prices).
    """
    if close.isnull().any():
        raise ValueError("close series contains NaN; label with clean data")

    r = cfg.regimes
    lookback = int(r.lookback_return_days)
    vol_win = int(r.crisis.vol_window_days)
    pctile = float(r.crisis.percentile)
    pctile_min_periods = int(r.crisis.percentile_min_periods)
    min_vol = float(r.crisis.min_vol_annualized)

    trailing_ret = trailing_log_return(close, lookback)
    vol = realized_vol(close, vol_win)
    # Expanding window: q95 of ALL vol history up to each day. The crisis bar
    # stays anchored to the long-run vol distribution.
    vol_threshold_series = vol.expanding(min_periods=pctile_min_periods).quantile(
        pctile / 100.0
    )
    vol_threshold_series = vol_threshold_series.clip(lower=min_vol)

    raw = _raw_labels(trailing_ret, cfg)

    # ---- hysteresis: crisis entry/exit state machine --------------------
    confirm = int(r.hysteresis.crisis_confirmation_days)
    exit_mult = float(r.hysteresis.crisis_exit_mult)
    hysteresis = _apply_crisis_hysteresis(
        vol, vol_threshold_series, confirm, exit_mult, min_vol
    )

    labels = pd.Series(np.where(hysteresis, CRISIS, raw.to_numpy()), index=raw.index,
                       dtype="object")

    # ---- hysteresis: minimum run duration --------------------------------
    labeled_idx = labels.index[labels.notna()]
    labels = _blend_runs(labels.loc[labeled_idx], int(r.hysteresis.min_regime_duration_days))
    return labels.reindex(close.index)