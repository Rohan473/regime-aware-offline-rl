"""Tests for the offline RL heads (A2C / IQL) on frozen representations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.models.rep_lab.config import RepLabConfig
from src.models.rep_lab.offline_rl import OfflineRepData, train_offline


def _synthetic_offline(seed: int = 0, T: int = 800, hidden: int = 6, P: int = 4) -> OfflineRepData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=T).tz_localize("America/New_York")
    return OfflineRepData(
        H=torch.tensor(rng.normal(0, 1, (T, hidden)), dtype=torch.float32),
        actions=torch.tensor(rng.uniform(-1, 1, (P, T)), dtype=torch.float32),
        rewards=torch.tensor(rng.normal(0, 0.01, (P, T)), dtype=torch.float32),
        dones=torch.zeros(P, T, dtype=torch.bool),
        market_returns=torch.tensor(rng.normal(0, 0.01, T), dtype=torch.float32),
        valid=torch.ones(T, dtype=torch.bool),
        dates=dates,
        policies=tuple(f"p{i}" for i in range(P)),
    )


def test_offline_split_boundaries():
    d = _synthetic_offline()
    assert d.split("train").sum() > 0
    assert d.split("val").sum() > 0
    assert d.split("test").sum() > 0
    assert (d.split("train") | d.split("val") | d.split("test")).all()


def test_train_offline_a2c_and_iql_run():
    d = _synthetic_offline()
    for algo in ("A2C", "IQL"):
        cfg = RepLabConfig()
        cfg.seed = 1
        _, m = train_offline("raw", algo, cfg, d, epochs=2, batch=64)
        assert m["algo"] == algo
        assert np.isfinite(m["test_sharpe"])
        assert np.isfinite(m["test_return"])
        assert m["test_turnover"] >= 0.0
