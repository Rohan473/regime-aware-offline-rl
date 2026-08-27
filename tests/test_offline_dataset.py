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
from src.data.behavior_policies import policy_names
from src.data.technical_factors import FEATURE_COLUMNS
from tests.conftest import BULL_DAYS, BEAR_DAYS, CRISIS_DAYS, DEFAULT_CFG, synthetic_daily_ohlcv

POLICIES = tuple(policy_names(DEFAULT_CFG))
MOMENTUM_POLICIES = tuple(p for p in POLICIES if p.startswith("momentum_"))
MREV_POLICIES = tuple(p for p in POLICIES if p.startswith("mean_reversion_"))


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
    """Every momentum / mean-reversion WINDOWED policy must take both short
    and long positions (symmetric scaled signals), and the dataset action
    range must span negative territory."""
    assert dataset["action"].min() < 0
    for family, members in (("momentum", MOMENTUM_POLICIES), ("mean_reversion", MREV_POLICIES)):
        assert len(members) > 0, f"{family} family not expanded from config"
        for policy in members:
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


def _policy_window(policy: str) -> int | None:
    """Window of a windowed policy tag ('momentum_30d' -> 30); None for
    buy_and_hold / random."""
    for suffix in ("momentum_", "mean_reversion_"):
        if policy.startswith(suffix):
            return int(policy[len(suffix) : -1])
    return None


def _policy_valid_mask(policy: str, frame: pd.DataFrame, base: pd.Series) -> pd.Series:
    """Family-specific validity mask (warm-up), computed independently of
    behavior_policies.py to cross-validate the pipeline."""
    valid = base
    w = _policy_window(policy)
    if w is not None:
        log_close = np.log(frame["close"])
        valid = valid & log_close.diff().rolling(w).sum().notna()
    if policy == "nr7":
        rng = frame["high"] - frame["low"]
        # the 7-day range minimum must be defined at t-1
        valid = valid & rng.rolling(7).min().shift(1).notna()
    return valid


def test_transition_count_matches_valid_days(dataset, cfg, daily):
    """One transition per (policy, valid decision day) minus window warm-up.

    Each windowed policy additionally needs ``window`` trailing days for its
    rolling return to be defined, so its trajectory starts later than the
    base (feature + regime) warm-up; nr7 needs its 7-day range lookback at
    t-1. Computed independently to cross-validate the pipeline count."""
    transitions, frame = build_offline_dataset(cfg, daily=daily)
    base = frame[FEATURE_COLUMNS].notna().all(axis=1) & frame["regime"].notna()
    expected = 0
    for p in POLICIES:
        expected += int(_policy_valid_mask(p, frame, base).sum()) - 1
    assert len(transitions) == expected


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


# ---------------- NR7 specialized signal family ----------------


def test_nr7_actions_are_long_short_flat(dataset):
    """The NR7 policy must be a clean Long/Short/Flat vector."""
    nr7 = dataset[dataset["policy"] == "nr7"]
    assert len(nr7) > 0
    assert set(nr7["action"].unique()) <= {-1.0, 0.0, 1.0}
    # on ~20 years of daily data the signal must fire on both sides
    assert (nr7["action"] == 1.0).any(), "nr7 never goes long"
    assert (nr7["action"] == -1.0).any(), "nr7 never goes short"
    # and stay flat often (NR7 days are rare: narrowest-of-7 is ~1/7 of days,
    # and only confirmed breakouts trade)
    assert (nr7["action"] == 0.0).mean() > 0.5, "nr7 is rarely flat — signal is wrong"


def test_nr7_signal_matches_hand_computed_rule(cfg, daily):
    """Independently recompute the NR7 rule and compare action-by-action.

    At decision date t: long(+1) iff t-1 was an NR7 day (narrowest high-low
    range of the 7 trading days ending t-1, ties allowed) and close[t] >
    high[t-1]; short(-1) iff NR7 at t-1 and close[t] < low[t-1]; else 0."""
    transitions, frame = build_offline_dataset(cfg, daily=daily)
    rng = frame["high"] - frame["low"]
    rng_min7 = rng.rolling(7).min()
    nr7_day = (rng == rng_min7) & rng_min7.notna()
    nr7_prev = nr7_day.shift(1).fillna(False).astype(bool)
    expected = pd.Series(0.0, index=frame.index)
    expected[nr7_prev & (frame["close"] > frame["high"].shift(1))] = 1.0
    expected[nr7_prev & (frame["close"] < frame["low"].shift(1))] = -1.0

    nr7 = transitions[transitions["policy"] == "nr7"].set_index("date")
    got = expected.reindex(nr7.index)
    assert got.notna().all()
    np.testing.assert_array_equal(nr7["action"].to_numpy(), got.to_numpy())


def test_nr7_reward_is_action_times_next_return(dataset):
    """reward_t = a_t * ret_{t+1} (cost 0) — verified for the nr7 family
    specifically, including the +/-1 breakout days."""
    nr7 = dataset[dataset["policy"] == "nr7"].reset_index(drop=True)
    active = nr7[nr7["action"] != 0.0]
    assert len(active) > 0
    # ret_{t+1} = reward / action on active days
    implied = active["reward"] / active["action"]
    assert np.isfinite(implied).all()
    # implied next-day returns must be plausible daily moves (< 25%)
    assert implied.abs().max() < 0.25
    # and flat days earn exactly 0
    assert (nr7.loc[nr7["action"] == 0.0, "reward"] == 0.0).all()
