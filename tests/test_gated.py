"""Tests for the regime-gated offline RL policy (src.models.rep_lab.gated)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.models.rep_lab.config import RepLabConfig
from src.models.rep_lab.gated import RegimeRisk, GatedPolicy, compose, train_gated
from src.models.rep_lab.offline_rl import OfflineRepData


def _synthetic(seed: int = 0, T: int = 800, hidden: int = 6, P: int = 4) -> OfflineRepData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=T).tz_localize("America/New_York")
    return OfflineRepData(
        H=torch.tensor(rng.normal(0, 1, (T, hidden)), dtype=torch.float32),
        actions=torch.tensor(rng.uniform(-0.5, 0.5, (P, T)), dtype=torch.float32),
        rewards=torch.tensor(rng.normal(0, 0.01, (P, T)), dtype=torch.float32),
        dones=torch.zeros(P, T, dtype=torch.bool),
        market_returns=torch.tensor(rng.normal(0, 0.01, T), dtype=torch.float32),
        valid=torch.ones(T, dtype=torch.bool),
        dates=dates,
        policies=tuple(f"p{i}" for i in range(P)),
    )


def test_regime_risk_shapes():
    d = _synthetic()
    tr = d.split("train")
    regimes = np.random.default_rng(1).choice(["bull", "bear", "crisis"], len(d.dates))
    risk = RegimeRisk().fit(d.H, regimes, tr)
    p = risk.proba(d.H)
    assert p.shape == (len(d.dates), 3)
    r = risk.risk(d.H)
    assert r.shape == (len(d.dates),)
    assert ((r >= 0) & (r <= 1)).all()


def test_gated_policy_compose():
    h = torch.randn(5, 6)
    head = GatedPolicy(6)
    lam, d = head.deterministic(h)
    assert lam.shape == (5,) and ((lam > 0) & (lam < 1)).all()
    assert (d.abs() <= 1).all()
    a = compose(torch.zeros(5), lam, d)
    assert (a.abs() <= 1).all()


def test_train_gated_runs():
    d = _synthetic()
    tr = d.split("train")
    regimes = np.random.default_rng(2).choice(["bull", "bear", "crisis"], len(d.dates))
    risk = RegimeRisk().fit(d.H, regimes, tr)
    cfg = RepLabConfig()
    cfg.seed = 1
    head, m = train_gated("raw", cfg, d, risk, rho=1.0, epochs=2, batch=64)
    assert np.isfinite(m["test_sharpe"])
    assert m["test_lambda"].shape == (int(d.split("test").sum()),)
    assert (m["test_lambda"] > 0).all() and (m["test_lambda"] < 1).all()