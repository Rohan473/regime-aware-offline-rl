"""Tests for Model B (DDR): exact DSR, time splits, regime eval, policy."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.eval import regime_eval
from src.models import SPLIT_TEST_END, SPLIT_TRAIN_END, SPLIT_VAL_END, WINDOW_DAYS
from src.models.ddr.config import DDRConfig
from src.models.ddr.data import DDRData, compute_valid_mask, load_ddr_data, split_ddr_data
from src.models.ddr.dsr import DSRState, VolTargetBuffer
from src.models.ddr.policy import DDRPolicy
from src.models.ddr.train import train_ddr

PROCESSED = None  # resolved lazily; real-data tests skip if artifacts absent


def _processed_dir():
    from src.data.loaders import REPO_ROOT

    return REPO_ROOT / "data" / "processed"


def _require_artifacts():
    p = _processed_dir()
    for f in ("features_regimes.parquet", "offline_dataset.parquet"):
        if not (p / f).exists():
            pytest.skip(f"Phase-1 artifact missing: {p / f}")


# --------------------------------------------------------------------------
# DSR exactness
# --------------------------------------------------------------------------

def test_dsr_matches_hand_computed_toy():
    """D_t against an independently written scalar recurrence (Moody-Saffell).

    Anchor: with eta=0.1, A0=0.01, B0=0.0002, r1=0.02 the numerator is
    2e-6 - 1e-6 = 1e-6 and the denominator is (1e-4)^1.5 = 1e-6, so
    D_1 = 1.0 exactly.
    """
    eta, a0, b0 = 0.1, 0.01, 0.0002
    returns = [0.02, -0.01, 0.015]

    A, B = a0, b0
    expected = []
    for r in returns:
        dA = r - A
        dB = r * r - B
        expected.append((B * dA - 0.5 * A * dB) / (B - A * A) ** 1.5)
        A = A + eta * dA
        B = B + eta * dB

    state = DSRState(eta=eta, warmup_steps=0, a0=a0, b0=b0, dtype=torch.float64)
    D = state.update(torch.tensor(returns, dtype=torch.float64))
    torch.testing.assert_close(D, torch.tensor(expected, dtype=torch.float64), rtol=1e-6, atol=1e-9)
    assert D[0].item() == pytest.approx(1.0, abs=1e-9)
    assert state.a.item() == pytest.approx(A, rel=1e-9)
    assert state.b.item() == pytest.approx(B, rel=1e-9)


def test_dsr_warmup_emits_nan_then_matches_after_warm_state():
    """Warm-up steps emit NaN; D after warm-up equals the recurrence computed
    from the warm EMA state."""
    eta, a0, b0 = 0.1, 0.01, 0.0002
    returns = [0.02, -0.01, 0.015, 0.008]
    warm = 2

    A, B = a0, b0
    for r in returns[:warm]:  # warm-up consumes steps without emitting D
        A = A + eta * (r - A)
        B = B + eta * (r * r - B)
    expected = []
    for r in returns[warm:]:
        dA = r - A
        dB = r * r - B
        expected.append((B * dA - 0.5 * A * dB) / (B - A * A) ** 1.5)
        A = A + eta * dA
        B = B + eta * dB

    state = DSRState(eta=eta, warmup_steps=warm, a0=a0, b0=b0, dtype=torch.float64)
    D = state.update(torch.tensor(returns, dtype=torch.float64))
    assert torch.isnan(D[:warm]).all()
    torch.testing.assert_close(D[warm:], torch.tensor(expected, dtype=torch.float64), rtol=1e-6, atol=1e-9)


def test_dsr_gradient_flows_to_action():
    """D_t must be differentiable w.r.t. the action (direct backprop path)."""
    eta, warm = 0.05, 3
    state = DSRState(eta=eta, warmup_steps=warm)
    a = torch.tensor([0.5], requires_grad=True)
    R = torch.tensor([0.01, -0.005, 0.02, 0.0, 0.01])
    D = state.update(a * R)
    assert D[3:].isfinite().all()
    D[3:].sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_dsr_ema_sharpe():
    state = DSRState(eta=0.1, warmup_steps=0, a0=0.01, b0=0.0002)
    state.update(torch.tensor([0.02, -0.01, 0.015]))
    s = state.sharpe_ema()
    expected = state.a / math.sqrt(state.b.item() - state.a.item() ** 2)
    assert s.item() == pytest.approx(expected.item(), rel=1e-9)


# --------------------------------------------------------------------------
# Policy network
# --------------------------------------------------------------------------

def test_policy_action_range():
    torch.manual_seed(0)
    model = DDRPolicy(input_dim=8, hidden_size=32, rnn_type="gru")
    x = torch.randn(10, WINDOW_DAYS, 8)
    actions = model(x)
    assert actions.shape == (10, 1)
    assert torch.abs(actions).max().item() <= 1.0


def test_policy_causal_within_window():
    """Toggling features before the end of the window must change the action,
    but the action at time t cannot depend on features after t (GRU reads
    left-to-right, so it is causal by construction; this guards the wiring)."""
    torch.manual_seed(1)
    model = DDRPolicy(input_dim=8, hidden_size=16, rnn_type="gru")
    x = torch.randn(1, WINDOW_DAYS, 8)
    a_full = model(x).item()
    x_blank = x.clone()
    x_blank[:, -1, :] = 0.0  # wipe the last day -> action must change
    assert model(x_blank).item() != pytest.approx(a_full, abs=1e-6)


# --------------------------------------------------------------------------
# Data: windowing, splits, Phase-1 equivalence
# --------------------------------------------------------------------------

def test_zstate_matches_offline_dataset():
    """DDR inputs (z_* columns) equal the Phase-1 offline states on shared
    dates — guards against preprocessing drift between phases."""
    _require_artifacts()
    p = _processed_dir()
    features = pd.read_parquet(p / "features_regimes.parquet")
    dataset = pd.read_parquet(p / "offline_dataset.parquet")
    ds = dataset.drop_duplicates("date").set_index("date").sort_index()
    shared = features.index.intersection(ds.index)
    for col in [c for c in features.columns if c.startswith("z_")]:
        # offline states are stored as float32 (Phase 1); frame is float64 —
        # same numbers, compare at float32 precision
        pd.testing.assert_series_equal(
            features.loc[shared, col].astype("float32"),
            ds.loc[shared, col],
            rtol=1e-6,
            atol=1e-8,
            check_dtype=False,
        )


def test_next_returns_match_buy_and_hold_reward():
    """The realized next-day returns used by DDR equal the Phase-1
    buy-and-hold reward (a=1.0, cost 0) — alignment of the market path."""
    _require_artifacts()
    data = load_ddr_data(WINDOW_DAYS)
    dataset = pd.read_parquet(_processed_dir() / "offline_dataset.parquet")
    bh = dataset[dataset["policy"] == "buy_and_hold"].set_index("date")["reward"]
    ours = pd.Series(data.next_returns.numpy(), index=data.dates)
    shared = ours.index.intersection(bh.index)
    pd.testing.assert_series_equal(
        ours.loc[shared].astype("float32").rename("reward"),
        bh.loc[shared],
        rtol=1e-6,
        atol=1e-9,
        check_names=True,
    )


def test_windows_are_causal_and_contiguous():
    """Window ending at date t must contain exactly W rows all <= t."""
    _require_artifacts()
    data = load_ddr_data(WINDOW_DAYS)
    features = pd.read_parquet(_processed_dir() / "features_regimes.parquet")
    assert data.windows.shape == (len(data.dates), WINDOW_DAYS, 8)
    pos = features.index.get_indexer(data.dates[:3])
    for i, p in enumerate(pos[:3]):
        window_dates = features.index[p - WINDOW_DAYS + 1 : p + 1]
        assert len(window_dates) == WINDOW_DAYS
        assert (window_dates <= data.dates[i]).all()
        assert window_dates.is_unique


def test_valid_mask_flags_hole_boundaries():
    """Days whose 'next-day' return spans a multi-month data hole must be
    flagged invalid; ordinary trading days stay valid."""
    features_index = pd.DatetimeIndex(
        list(pd.date_range("2022-01-03", "2022-01-14", freq="B"))
        + list(pd.date_range("2022-08-01", "2022-08-10", freq="B"))
    )
    dates = pd.DatetimeIndex(
        ["2022-01-03", "2022-01-05", "2022-01-06", "2022-01-07", "2022-01-14", "2022-08-01"]
    )
    valid = compute_valid_mask(dates, features_index)
    # Jan 14 is the last row before the Apr-Jul hole -> its "return" spans
    # ~199 calendar days -> invalid; 2022-08-01's next row is Aug 2 -> valid
    assert valid.tolist() == [True, True, True, True, False, True]


def test_valid_mask_keeps_long_holiday_weekends():
    """A 5-6 calendar-day closure (e.g. Christmas + New Year) is a genuine
    1-trading-day return and must stay valid."""
    features_index = pd.date_range("2008-01-01", "2008-01-15", freq="B")
    # Dec 24 2007 (Mon) -> Jan 2 2008 (Wed): 9 calendar days, market closed
    # Dec 25 + Jan 1. Single trading-day return across the closure.
    dates = pd.DatetimeIndex(["2007-12-24", "2008-01-02", "2008-01-03", "2008-01-15"])
    valid = compute_valid_mask(dates, features_index, max_gap_days=10)
    assert valid.tolist() == [True, True, True, False]  # last has no next row


def test_time_split_no_leakage():
    """Time split: disjoint, ordered, exact boundaries."""
    dates = pd.date_range("2005-01-03", periods=5000, freq="B")
    n = len(dates)
    fake = DDRData(
        windows=torch.zeros(n, WINDOW_DAYS, 8),
        dates=dates,
        next_returns=torch.zeros(n),
        regimes=pd.Series("bull", index=dates),
    )
    splits = split_ddr_data(fake)
    assert set(splits) == {"train", "val", "test"}
    assert not set(splits["train"].dates).intersection(splits["val"].dates)
    assert not set(splits["train"].dates).intersection(splits["test"].dates)
    assert not set(splits["val"].dates).intersection(splits["test"].dates)
    assert splits["train"].dates.max().tz_localize(None) <= pd.Timestamp(SPLIT_TRAIN_END)
    assert splits["val"].dates.min().tz_localize(None) > pd.Timestamp(SPLIT_TRAIN_END)
    assert splits["val"].dates.max().tz_localize(None) <= pd.Timestamp(SPLIT_VAL_END)
    assert splits["test"].dates.min().tz_localize(None) > pd.Timestamp(SPLIT_VAL_END)
    assert splits["test"].dates.max().tz_localize(None) <= pd.Timestamp(SPLIT_TEST_END)
    # chronological within each split
    for name, split in splits.items():
        assert split.dates.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Volatility-targeted DSR (reward fix)
# ---------------------------------------------------------------------------

def _vt_returns(actions, returns, **kw):
    buf = VolTargetBuffer(target_vol=0.15, window=20, max_leverage=2.0, **kw)
    _, out = buf(torch.tensor(actions, dtype=torch.float32),
                 torch.tensor(returns, dtype=torch.float32))
    return out


def test_voltarget_exact_scale_after_warmup():
    """After the 20-step vol warmup, the scale factor equals
    target_vol / (sample-std * sqrt(252)) of the trailing 20 strategy returns."""
    returns = torch.tensor([0.01 if i % 2 == 0 else -0.01 for i in range(60)], dtype=torch.float32)
    _, out = VolTargetBuffer(0.15, window=20, max_leverage=2.0)(
        torch.ones(60), returns
    )
    s20 = 0.01 * (20 / 19) ** 0.5 * (252 ** 0.5)   # annualized sample std
    scale = 0.15 / s20
    expected = scale * 0.01
    assert out[20] == pytest.approx(expected, rel=1e-6)   # even idx -> +0.01
    assert out[26] == pytest.approx(expected, rel=1e-6)


def test_voltarget_invariant_to_deleveraging():
    """THE fix: the DSR input series is identical whether the policy holds
    |a|=1 or |a|=0.5 (shrinking exposure no longer reduces the reward's
    variance), from the first fully-warmed step onward."""
    returns = torch.tensor([0.01 if i % 2 == 0 else -0.01 for i in range(60)], dtype=torch.float32)
    _, out_full = VolTargetBuffer(0.15, 20, 2.0)(torch.ones(60), returns)
    _, out_half = VolTargetBuffer(0.15, 20, 2.0)(0.5 * torch.ones(60), returns)
    assert torch.allclose(out_full[19:], out_half[19:], atol=1e-8)
    # ...but NOT during the vol warmup (scale=1 there), where halving the
    # action halves the reward input:
    assert out_full[:19].abs() == pytest.approx(0.01)
    assert out_half[:19].abs() == pytest.approx(0.005)


def test_voltarget_warmup_is_unscaled():
    """Steps before a full 20-return window get scale=1 (no targeting)."""
    returns = torch.full((25,), 0.01)
    _, out = VolTargetBuffer(0.15, 20, 2.0)(torch.ones(25), returns)
    assert out[:19] == pytest.approx(0.01)
    # constant returns -> zero vol -> scale stays 1 (never inf/NaN)
    assert out[19:] == pytest.approx(0.01)


def test_voltarget_clips_to_max_leverage():
    """Low-vol regime: scale would exceed max_leverage -> action clipped."""
    returns = torch.tensor([0.003 if i % 2 == 0 else -0.003 for i in range(60)], dtype=torch.float32)
    scaled, out = VolTargetBuffer(0.15, 20, 2.0)(torch.ones(60), returns)
    s20 = 0.003 * (20 / 19) ** 0.5 * (252 ** 0.5)
    assert (0.15 / s20) > 2.0          # the unclipped scale really is > 2
    assert torch.allclose(scaled[19:], torch.full((41,), 2.0), atol=1e-6)
    assert out[20] == pytest.approx(2.0 * 0.003)


def test_voltarget_window_is_causal_trailing():
    """The scale at step t uses ONLY the trailing 20 strategy returns:
    first 20 steps at low vol, then 20 at high vol -> scale drops exactly
    when the window fills with high-vol returns."""
    returns = torch.tensor(
        [0.01 if i % 2 == 0 else -0.01 for i in range(20)]
        + [0.04 if i % 2 == 0 else -0.04 for i in range(40)],
        dtype=torch.float32,
    )
    _, out = VolTargetBuffer(0.15, 20, 2.0)(torch.ones(60), returns)
    low = 0.15 / (0.01 * (20 / 19) ** 0.5 * (252 ** 0.5))
    high = 0.15 / (0.04 * (20 / 19) ** 0.5 * (252 ** 0.5))
    assert out[19] == pytest.approx(-low * 0.01, rel=1e-6)   # window still all low-vol
    assert out[40] == pytest.approx(high * 0.04, rel=1e-6)   # window all high-vol
    assert abs(out[26]) > abs(out[19])                       # transition began by then


def test_config_from_yaml():
    """configs/ddr.yaml is the shared config: vol-targeting ON, target 0.15,
    leverage 2.0, artifacts under checkpoints/vt."""
    cfg = DDRConfig.from_yaml()
    assert cfg.vol_targeting is True
    assert cfg.target_vol == pytest.approx(0.15)
    assert cfg.max_leverage == pytest.approx(2.0)
    assert cfg.vol_target_window == 20
    assert str(cfg.checkpoint_dir).replace("\\", "/").endswith("src/models/ddr/checkpoints/vt")
    assert cfg.epochs == 30
    # dataclass default stays naive (tests / old flows unaffected)
    assert DDRConfig().vol_targeting is False


def test_config_from_yaml_rejects_bad_target_vol():
    import tempfile, yaml as _yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        _yaml.safe_dump({"model": {"target_vol": 0.0, "vol_targeting": True}}, f)
        path = f.name
    try:
        import pytest as _pytest
        with _pytest.raises(ValueError):
            DDRConfig.from_yaml(path)
    finally:
        import os
        os.unlink(path)


# --------------------------------------------------------------------------
# Shared regime evaluation (src/eval/regime_eval.py)
# --------------------------------------------------------------------------

def test_regime_eval_synthetic_known_values():
    """Hand-computed regime breakdown on a synthetic labeled series.

    bull   : [+0.01, -0.005, +0.01, -0.005, +0.01]  mean 0.004, std ~0.0082158
              -> Sharpe_ann ~= 7.728; cum_ret ~= 0.0200147; MDD = -0.005
    bear   : [-0.01, -0.01, -0.01, +0.02]  mean -0.0025, std 0.015
              -> Sharpe_ann = -(1/6)*sqrt(252) ~= -2.64575
    crisis : [-0.02, +0.01, +0.01]  mean 0 -> Sharpe_ann == 0.0
    """
    rows = []
    for regime, rets in [
        ("bull", [0.01, -0.005, 0.01, -0.005, 0.01]),
        ("bear", [-0.01, -0.01, -0.01, 0.02]),
        ("crisis", [-0.02, 0.01, 0.01]),
    ]:
        for r in rets:
            rows.append({"date": pd.Timestamp(len(rows), unit="D"), "ret": r, "regime": regime})
    preds = pd.DataFrame(rows)
    table = regime_eval.evaluate_regime_breakdown(preds)

    bull = table.loc["bull"]
    assert bull["n_days"] == 5
    assert bull["sharpe_annualized"] == pytest.approx(0.004 / np.sqrt(6.75e-5) * np.sqrt(252), rel=1e-6)
    assert bull["cum_return"] == pytest.approx(1.01**3 * 0.995**2 - 1.0, rel=1e-9)
    assert bull["max_drawdown"] == pytest.approx(-0.005, rel=1e-9)
    assert table.loc["bear", "sharpe_annualized"] == pytest.approx(-(1 / 6) * np.sqrt(252), rel=1e-6)
    assert table.loc["crisis", "sharpe_annualized"] == pytest.approx(0.0, abs=1e-12)
    assert table.loc["all", "n_days"] == 12
    assert table.loc["all", "mean_daily_ret"] == pytest.approx(0.01 / 12, rel=1e-9)


def test_regime_eval_edge_cases():
    empty = pd.DataFrame(columns=["date", "ret", "regime"])
    with pytest.raises(ValueError):
        regime_eval.evaluate_regime_breakdown(empty)
    two_constant = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=2), "ret": [0.01, 0.01], "regime": ["bull", "bull"]}
    )
    table = regime_eval.evaluate_regime_breakdown(two_constant)
    assert np.isnan(table.loc["bull", "sharpe_annualized"])  # zero variance
    assert table.loc["bull", "max_drawdown"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# End-to-end training smoke test
# --------------------------------------------------------------------------

def test_train_smoke_on_synthetic_data(tmp_path):
    """Train 2 epochs on tiny synthetic data; wiring must not break and
    history must be logged."""
    torch.manual_seed(3)
    n = 120
    dates = pd.date_range("2005-03-01", periods=n, freq="B")  # all <= 2018 -> train only
    windows = torch.randn(n, 8, 8)
    next_returns = torch.tensor(np.where(np.arange(n) % 2 == 0, 0.01, -0.005), dtype=torch.float32)
    data = DDRData(
        windows=windows,
        dates=dates,
        next_returns=next_returns,
        regimes=pd.Series("bull", index=dates),
    )
    cfg = DDRConfig(
        window_size=8, hidden_size=8, eta=0.1, warmup_steps=8, lr=1e-2, epochs=2,
        checkpoint_dir=tmp_path,
    )
    model, history, checkpoint = train_ddr(cfg, data=data)
    assert len(history) == 2
    assert history["train_loss"].notna().all()
    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["epoch"] == 2
    assert set(payload["config"]) >= {"window_size", "hidden_size", "eta"}
    with torch.no_grad():
        actions = model(data.windows)
    assert torch.abs(actions).max().item() <= 1.0