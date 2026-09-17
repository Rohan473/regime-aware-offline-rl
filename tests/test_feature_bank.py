"""Tests for the 32-feature causal bank (src.data.feature_bank)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.feature_bank import (
    CANONICAL_8, FEATURE_BANK, FEATURE_SETS, causal_zscore,
)
from src.data.technical_factors import FEATURE_COLUMNS


def test_bank_shape_and_nesting():
    assert len(FEATURE_BANK) == 32
    assert len(set(FEATURE_BANK)) == 32
    assert CANONICAL_8 == list(FEATURE_COLUMNS)
    for n in (4, 8, 12, 16, 24):
        assert len(FEATURE_SETS[n]) == n
        assert FEATURE_SETS[n] == FEATURE_BANK[:n]  # nested prefix
    assert FEATURE_SETS[32] == FEATURE_BANK


def test_causal_zscore_is_causal():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-03", periods=300)
    df = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300)}, index=idx)
    z1 = causal_zscore(df, min_periods=20)
    # perturb FUTURE values only; the past z-scores must be unchanged
    df2 = df.copy()
    df2.iloc[250:] += 100.0
    z2 = causal_zscore(df2, min_periods=20)
    assert np.allclose(z1.iloc[:250].to_numpy(), z2.iloc[:250].to_numpy(), equal_nan=True)
    assert not np.allclose(z1.iloc[250:].to_numpy(), z2.iloc[250:].to_numpy())
    assert list(z1.columns) == ["z_a", "z_b"]


def test_causal_zscore_finite_on_constant():
    idx = pd.bdate_range("2000-01-03", periods=100)
    df = pd.DataFrame({"a": np.ones(100)}, index=idx)
    z = causal_zscore(df, min_periods=20)
    assert np.isfinite(z.to_numpy()).all()
