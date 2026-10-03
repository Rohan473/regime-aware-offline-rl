"""Tests for the exposure-only policy (src.models.rep_lab.exposure)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.models.rep_lab.config import RepLabConfig
from src.models.rep_lab.exposure import ExposurePolicy, train_exposure
from src.models.rep_lab.gated import RegimeRisk
from src.models.rep_lab.offline_rl import OfflineRepData


def _synthetic(seed: int = 0, T: int = 800, hidden: int = 6, P: int = 4) -> OfflineRepData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=T).tz_localize("America/New_York")
    return OfflineRepData(
        H=torch.tensor(rng.normal(0, 1, (T, hidden)), dtype=torch.float32),
        actions=torch.tensor(rng.uniform(-0.5, 0.5, (P, T)), dtype=torch.float32),
        rewards=torch.tensor(rng.normal(0, 0.01, (P, T)), dtype=torch.float32),
        dones=torch.zeros(P, T, dtype=torch.bool),
        market_returns=torch.tensor(rng.normal(0.0005, 0.01, T), dtype=torch.float32),
        valid=torch.ones(T, dtype=torch.bool),
        dates=dates,
        policies=tuple(f"p{i}" for i in range(P)),
    )


def test_exposure_policy_shape():
    h = torch.randn(8, 6)
    head = ExposurePolicy(6)
    e = head.e(h)
    assert e.shape == (8,)
    assert ((e > 0) & (e < 1)).all()


def test_train_exposure_runs_and_bounded():
    d = _synthetic()
    tr = d.split("train")
    regimes = np.random.default_rng(2).choice(["bull", "bear", "crisis"], len(d.dates))
    risk = RegimeRisk().fit(d.H, regimes, tr)
    cfg = RepLabConfig()
    cfg.seed = 1
    for rho in (0.0, 1.0):
        head, m = train_exposure("raw", cfg, d, risk=risk, rho=rho, epochs=3)
        assert np.isfinite(m["test_sharpe"])
        assert ((m["test_exposure"] > 0) & (m["test_exposure"] < 1)).all()