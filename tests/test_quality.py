"""Unit tests for the representation-quality scorecard (src.interpret.quality).

All tests are synthetic and deterministic — no checkpoints or real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.interpret.quality import (
    dimension_contribution, mean_pairwise_cka, perturbation_response,
    predictive_information, probe_classification_metrics, probe_regression_metrics,
    reconstruction_profile, trading_metrics,
)


def _split_sub(n_train=200, n_val=80, n_test=120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = n_train + n_val + n_test
    idx = pd.bdate_range("2000-01-03", periods=n)
    split = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return pd.DataFrame({
        "split": split,
        "fwd_dir_1": rng.choice([-1.0, 1.0], n),
        "fwd_dir_5": rng.choice([-1.0, 1.0], n),
        "fwd_dir_20": rng.choice([-1.0, 1.0], n),
        "abs_ret_1": np.abs(rng.normal(0, 0.01, n)),
        "fwd_vol_5": np.abs(rng.normal(0, 0.02, n)),
        "fwd_vol_20": np.abs(rng.normal(0, 0.02, n)),
        "fwd_dd_20": -np.abs(rng.normal(0, 0.05, n)),
        "regime": rng.choice(["bull", "bear", "crisis"], n),
    }, index=idx)


def test_classification_metrics_perfect_separation():
    rng = np.random.default_rng(1)
    n = 200
    y = np.array([0, 1] * (n // 2))
    X = np.column_stack([y * 3.0 + rng.normal(0, 0.1, n), rng.normal(0, 1, n)])
    m = probe_classification_metrics(X, y, X, y, X, y)
    assert m["test"]["accuracy"] == pytest.approx(1.0)
    assert m["test"]["auc"] == pytest.approx(1.0)
    assert m["test"]["balanced_accuracy"] == pytest.approx(1.0)
    assert m["test"]["log_loss"] < 0.1


def test_regression_metrics_exact_linear():
    rng = np.random.default_rng(2)
    n = 200
    X = rng.normal(0, 1, (n, 4))
    w = np.array([1.0, -2.0, 0.5, 0.0])
    y = X @ w
    m = probe_regression_metrics(X, y, X, y, X, y)
    assert m["test"]["r2"] == pytest.approx(1.0, abs=1e-6)
    assert m["test"]["mae"] < 1e-6
    assert m["test"]["spearman"] == pytest.approx(1.0, abs=1e-6)


def test_predictive_information_shapes():
    sub = _split_sub()
    X = np.random.default_rng(3).normal(0, 1, (len(sub), 6))
    info = predictive_information(X, sub)
    for name in ("direction_1", "magnitude_1", "vol_20", "regime", "drawdown_20"):
        assert name in info
        assert "test" in info[name]
    assert "auc" in info["direction_1"]["test"]
    assert "r2" in info["vol_20"]["test"]


def test_reconstruction_profile_keys():
    sub = _split_sub()
    z = ["z_a", "z_b", "z_c"]
    for c in z:
        sub[c] = np.random.default_rng(4).normal(0, 1, len(sub))
    X = sub[z].to_numpy()
    rec = reconstruction_profile(X, sub, z)
    assert set(rec["per_feature"]) == set(z)
    assert rec["mean"] > 0.9  # identity representation reconstructs itself


def test_dimension_contribution_rank_two_signal():
    sub = _split_sub(n_train=300, n_val=100, n_test=150)
    rng = np.random.default_rng(5)
    # two high-variance signal dims + low-variance noise: the leading PCA
    # components capture the signal, so 95% of the full metric is reached fast
    latent = 10.0 * rng.normal(0, 1, (len(sub), 2))
    sub["fwd_dir_1"] = np.where(latent[:, 0] + latent[:, 1] > 0, 1.0, -1.0)
    X = np.column_stack([latent, rng.normal(0, 0.01, (len(sub), 20))])
    dc = dimension_contribution(X, sub, "fwd_dir_1", "clf")
    assert dc["k_to_95"] is not None
    assert dc["k_to_95"] <= 4  # signal lives in the first two components


def test_trading_metrics_direction_and_costs():
    r = np.full(300, 0.001)
    a = np.ones(300)
    tm = trading_metrics(a, r)
    assert np.isfinite(tm["sharpe"])
    assert tm["total_return"] > 0
    assert tm["turnover"] == pytest.approx(1.0 / 300, abs=1e-9)
    # costs reduce the net Sharpe
    assert trading_metrics(a, r, cost_bps=5.0)["sharpe"] < tm["sharpe"]
    # flat policy -> undefined Sharpe
    assert np.isnan(trading_metrics(np.zeros(300), r)["sharpe"])


def test_perturbation_response():
    rng = np.random.default_rng(6)
    h = rng.normal(0, 1, (100, 8))
    assert perturbation_response(h, h)["dh"] == pytest.approx(0.0)
    assert perturbation_response(h, h + 1.0)["dh"] > 0
    a = np.linspace(-1, 1, 100)
    out = perturbation_response(h, h + 1.0, a, a + 0.5)
    assert out["da"] == pytest.approx(0.5)


def test_seed_stability_cka():
    from src.interpret.quality import linear_cka

    rng = np.random.default_rng(7)
    h = rng.normal(0, 1, (100, 8))
    assert linear_cka(h, h) == pytest.approx(1.0, abs=1e-9)
    # identical seeds -> mean pairwise CKA 1
    assert mean_pairwise_cka([h, h, h]) == pytest.approx(1.0, abs=1e-9)
    # a single seed has no stability estimate
    assert np.isnan(mean_pairwise_cka([h]))
    # an unrelated representation of the same samples is far less similar
    other = rng.normal(0, 1, (100, 8))
    assert mean_pairwise_cka([h, other]) < 0.9
