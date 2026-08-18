"""Offline data preparation for Model B (DDR).

The DDR model is trained on the SAME daily market path the Phase-1 behavior
policies rolled over. It sees the offline STATE sequence (trailing windows of
z-scored features) but generates its OWN actions, and its DSR reward is
computed from those own actions against the REALIZED next-day market returns
— it never learns from the logged actions or rewards of the behavior
policies. This is option (a) of the Phase-2 spec, stated explicitly.

Why the buy_and_hold + momentum filter of the spec is (exactly) the unique
market path: all four Phase-1 policies were rolled over the same daily
series, so their state sequences are IDENTICAL per date; only actions and
rewards differ. Filtering to buy_and_hold + momentum and deduplicating by
date therefore selects the full unique path, no more and no less. We
restrict to dates present in the offline dataset (the same warm-up exclusion
Phase 1 applied).

Input features are the 8 ``z_*`` columns of ``features_regimes.parquet`` —
the causal expanding z-scores computed in Phase 1, so the input is identical
to the offline states. The ``regime`` column is EXCLUDED from the input; it
is only joined by date for evaluation (never recomputed here).

Window semantics: the window for decision date t is the W days ending at t.
Features are causal (only data <= t), so a val/test window may contain
feature values from the train period without any label leakage — the label
(next-day return) is not part of the window.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.technical_factors import FEATURE_COLUMNS
from src.data.loaders import REPO_ROOT
from src.models import SPLIT_TEST_END, SPLIT_TRAIN_END, SPLIT_VAL_END

Z_COLUMNS = [f"z_{c}" for c in FEATURE_COLUMNS]


@dataclass
class DDRData:
    windows: torch.Tensor        # (N, W, F) float32
    dates: pd.DatetimeIndex      # tz-aware, one per window (decision date)
    next_returns: torch.Tensor   # (N,) realized next-day arithmetic returns
    regimes: pd.Series           # Phase-1 regime label per date (may contain NaN)
    valid: torch.Tensor | None = None  # (N,) bool; False where next_returns is not a
                                       # genuine 1-trading-day return (data hole boundary)


def compute_valid_mask(dates: pd.DatetimeIndex, features_index: pd.DatetimeIndex, max_gap_days: int = 6) -> np.ndarray:
    """True where the return realized FROM dates[i] is a genuine 1-trading-day
    return.

    The Phase-1 daily frame is built from an incremental downloader that has
    multi-month HOLES (e.g. 2022-04..07). ``close.pct_change()`` collapses a
    whole hole into one spurious "daily" return. A return from dates[i] is
    valid only if the next row in the features frame is at most
    ``max_gap_days`` calendar days later (legit holiday closures are 1-6 days;
    the holes are 33+ days). Invalid rows are excluded from training loss,
    eval metrics and baselines — they are not daily returns.
    """
    dates = dates.tz_localize(None) if getattr(dates, "tz", None) is not None else dates
    features_index = (
        features_index.tz_localize(None) if getattr(features_index, "tz", None) is not None else features_index
    )
    pos_next = features_index.searchsorted(dates, side="right")
    has_next = pos_next < len(features_index)
    gap_days = np.full(len(dates), np.nan)
    gap_days[has_next] = (features_index[pos_next[has_next]] - dates[has_next]).days
    return has_next & (gap_days <= max_gap_days)


def load_ddr_data(
    window_size: int,
    processed_dir: Path | None = None,
    behavior_policies: tuple[str, ...] = ("buy_and_hold", "momentum"),
) -> DDRData:
    """Load Phase-1 artifacts and build the windowed market path.

    ``behavior_policies`` selects which logged trajectories define the market
    path; as documented above, all policies share states so this selects the
    unique path. All four are kept only for faithfulness to the spec.
    """
    processed_dir = processed_dir or (REPO_ROOT / "data" / "processed")
    features = pd.read_parquet(processed_dir / "features_regimes.parquet")
    dataset = pd.read_parquet(processed_dir / "offline_dataset.parquet")

    dates = pd.DatetimeIndex(
        sorted(dataset[dataset["policy"].isin(behavior_policies)]["date"].unique())
    )
    pos = features.index.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError("offline dataset contains dates missing from features_regimes")
    if (pos < window_size - 1).any():
        raise ValueError(
            f"{int((pos < window_size - 1).sum())} dates lack a full {window_size}-day "
            "feature window (warm-up)"
        )

    windows = np.stack(
        [
            features.iloc[p - window_size + 1 : p + 1][Z_COLUMNS].to_numpy(dtype="float64")
            for p in pos
        ]
    )
    market_ret = features["close"].pct_change().shift(-1)
    next_returns = market_ret.loc[dates].to_numpy(dtype="float64")
    if np.isnan(next_returns).any():
        raise ValueError("next-day returns contain NaN (series does not extend past last date)")

    regimes = features["regime"].reindex(dates)
    valid = compute_valid_mask(dates, features.index)

    # The Phase-1 dataset may extend past the study horizon (SPLIT_TEST_END)
    # after a pipeline re-run; the splits are capped at test_end, so clip the
    # loaded path to the study window instead of failing downstream.
    keep = dates.tz_localize(None) <= pd.Timestamp(SPLIT_TEST_END)
    if not keep.all():
        print(f"[load_ddr_data] clipping {int((~keep).sum())} dates beyond {SPLIT_TEST_END} "
              f"(offline dataset extends to {dates.max().date()})", file=sys.stderr)
        dates = dates[keep]
        windows = windows[keep]
        next_returns = next_returns[keep]
        regimes = regimes[keep]
        valid = valid[keep]

    return DDRData(
        windows=torch.tensor(windows, dtype=torch.float32),
        dates=dates,
        next_returns=torch.tensor(next_returns, dtype=torch.float32),
        regimes=regimes,
        valid=torch.from_numpy(valid),
    )


def split_ddr_data(
    data: DDRData,
    train_end: str = SPLIT_TRAIN_END,
    val_end: str = SPLIT_VAL_END,
    test_end: str = SPLIT_TEST_END,
) -> dict[str, DDRData]:
    """Time-ordered split by decision date. Never shuffles across boundaries.

    train: dates <= train_end; val: (train_end, val_end]; test: > val_end.
    Returns keys "train", "val", "test" (test may be empty if the series is
    too short).
    """
    naive = data.dates.tz_localize(None)
    bounds = {
        "train": naive <= pd.Timestamp(train_end),
        "val": (naive > pd.Timestamp(train_end)) & (naive <= pd.Timestamp(val_end)),
        "test": naive > pd.Timestamp(val_end),
    }
    if naive.max() > pd.Timestamp(test_end):
        raise ValueError(f"dates beyond test_end={test_end}: {naive.max().date()}")

    def sel(mask: np.ndarray) -> DDRData:
        valid = data.valid if data.valid is not None else torch.ones(len(data.windows), dtype=torch.bool)
        return DDRData(
            windows=data.windows[mask],
            dates=data.dates[mask],
            next_returns=data.next_returns[mask],
            regimes=data.regimes[mask],
            valid=valid[mask],
        )

    return {name: sel(mask) for name, mask in bounds.items()}