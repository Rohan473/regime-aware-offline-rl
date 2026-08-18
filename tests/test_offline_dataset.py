"""Offline dataset tests: NaN-free features, action diversity, correct
transition structure, policy tags, reward formula, done flags, and the
minute->daily resampler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import _resample_daily
from src.data.offline_dataset import build_offline_dataset
from src.data.technical_factors import FEATURE_COLUMNS
from tests.conftest import BULL_DAYS, BEAR_DAYS, CRISIS_DAYS, synthetic_daily_ohlcv

POLICIES = ["momentum", "mean_reversion", "buy_and_hold", "random"]


@pytest.fixture
def dataset(cfg, daily):
    transitions, _frame = build_offline_dataset(cfg, daily=daily)
    return transitions


def test_no_nan_anywhere(dataset):
    assert dataset.notna().all().all()


def test_feature_count_within_cap():
    assert len(FEATURE_COLUMNS) <= 10


def test_feature_frame_has_regime_column(cfg, daily):
    _transitions, frame = build_offline_dataset(cfg, daily=daily)
    assert "regime" in frame.columns
    assert frame["regime"].dropna().isin(["bull", "bear", "crisis"]).all()


def test_all_policies_present_with_rows(dataset):
    for policy in POLICIES:
        assert len(dataset[dataset["policy"] == policy]) > 0
    assert set(dataset["policy"].unique()) == set(POLICIES)


def test_action_diversity(dataset):
    """No single policy may dominate the action distribution."""
    shares = dataset["policy"].value_counts(normalize=True)
    assert (shares <= 0.5).all(), f"policy shares: {shares.to_dict()}"
    # dataset-wide action std must be non-trivial (not all a=1.0)
    assert dataset["action"].std() > 0.05
    # random policy must explore the full position range, including shorts
    rnd = dataset[dataset["policy"] == "random"]["action"]
    assert rnd.std() > 0.2
    assert rnd.min() < -0.9 and rnd.max() > 0.9


def test_shorting_occurs(dataset):
    """With position limits [-1, 1], momentum and mean-reversion must both
    take negative (short) positions at some point, and the dataset action
    range must actually span negative territory."""
    assert dataset["action"].min() < 0
    for policy in ("momentum", "mean_reversion"):
        actions = dataset[dataset["policy"] == policy]["action"]
        assert (actions < 0).any(), f"{policy} never shorts"
        assert (actions > 0).any(), f"{policy} never goes long"


def test_buy_and_hold_reward_equals_next_day_return(cfg, daily):
    transitions, _ = build_offline_dataset(cfg, daily=daily)
    bh = transitions[transitions["policy"] == "buy_and_hold"].reset_index(drop=True)
    # reward_t = a_t * ret_{t+1} with a_t == 1.0 and zero cost, so reward must
    # equal the NEXT day's close-to-close return (causal, no lookahead).
    expected = daily["close"].pct_change().shift(-1).reindex(bh["date"])
    assert expected.notna().all(), "expected rewards must all be defined"
    assert np.allclose(bh["reward"].to_numpy(), expected.to_numpy(), atol=1e-9)
    assert np.allclose(bh["action"], 1.0)


def test_actions_within_position_limits(cfg, daily):
    transitions, _ = build_offline_dataset(cfg, daily=daily)
    assert transitions["action"].between(-1.0 - 1e-6, 1.0 + 1e-6).all()


def test_done_flag_only_on_last_transition_per_policy(dataset):
    for policy in POLICIES:
        grp = dataset[dataset["policy"] == policy]
        assert grp["done"].sum() == 1
        assert grp["done"].iloc[-1] == True  # noqa: E712
        assert not grp["done"].iloc[:-1].any()


def test_next_state_is_following_state(dataset):
    """Within a policy trajectory, next_state(t) must equal state(t+1)."""
    for policy in POLICIES:
        grp = dataset[dataset["policy"] == policy]
        dates = grp["date"].to_numpy()
        next_dates = np.roll(dates, -1)[:-1]
        assert (dates[1:] == next_dates).all()
        for col in ["z_ret_1d", "z_rsi_14", "z_volume_zscore_20d"]:
            assert np.allclose(
                grp[f"next_{col}"].to_numpy()[:-1],
                grp[col].to_numpy()[1:],
                atol=1e-6,
            )


def test_transition_count_matches_valid_days(dataset, cfg, daily):
    transitions, frame = build_offline_dataset(cfg, daily=daily)
    valid = frame[FEATURE_COLUMNS].notna().all(axis=1) & frame["regime"].notna()
    n_valid = int(valid.sum())
    assert len(transitions) == (n_valid - 1) * len(POLICIES)


def test_regime_distribution_non_degenerate(dataset):
    per_day = dataset.drop_duplicates("date")["regime"]
    counts = per_day.value_counts()
    assert set(counts.index) == {"bull", "bear", "crisis"}
    assert (counts > 0).all()


def test_no_state_features_are_constant(dataset):
    """Every state dimension must carry signal across the dataset."""
    for col in [c for c in dataset.columns if c.startswith("z_")]:
        assert dataset[col].std() > 1e-6, f"{col} is constant"


def test_states_are_causal_z_scores(dataset, daily):
    """In calm (pre-crisis) periods the expanding z-scores must behave like
    standard normals (|z| < 5 with 8 x ~200 values); the crisis block may
    legitimately produce large transient z's (that's the signal)."""
    z_cols = [c for c in dataset.columns if c.startswith("z_") and not c.startswith("next_")]
    calm = dataset[dataset["date"] < daily.index[BULL_DAYS + BEAR_DAYS]]
    assert len(calm) > 1000
    assert (calm[z_cols].abs() < 5).all().all()


def test_minute_to_daily_resample(cfg):
    """Resampler: open=first, high=max, low=min, close=last, volume=sum,
    grouped by America/New_York trading date."""
    idx = pd.date_range("2024-01-02 14:30", periods=120, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": 100 + np.arange(120) * 0.01,
            "high": 100 + np.arange(120) * 0.02,
            "low": 99.5 + np.arange(120) * 0.005,
            "close": 100 + np.arange(120) * 0.015,
            "volume": np.ones(120) * 100.0,
        },
        index=idx,
    )
    daily = _resample_daily(df, cfg)
    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["open"] == 100.0
    assert row["high"] == pytest.approx(100 + 119 * 0.02, rel=1e-9)
    assert row["low"] == pytest.approx(99.5, rel=1e-9)
    assert row["close"] == pytest.approx(100 + 119 * 0.015, rel=1e-9)
    assert row["volume"] == 120 * 100.0


def test_crisis_block_has_higher_volume(daily):
    """Fixture sanity: volume must spike in the crisis block so the volume
    z-score feature is non-degenerate."""
    crisis = daily.iloc[BULL_DAYS + BEAR_DAYS : BULL_DAYS + BEAR_DAYS + CRISIS_DAYS]
    calm = daily.iloc[:BULL_DAYS]
    assert crisis["volume"].mean() > 2 * calm["volume"].mean()
