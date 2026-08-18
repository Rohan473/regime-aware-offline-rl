"""Shared regime-conditioned policy evaluation (Models B/C/D).

Pure functions: they take a predictions DataFrame and return metrics; they
never load data and never recompute regime labels. Regime labels come from
Phase 1 (``features_regimes.parquet``) and are joined to predictions by date
by the caller (see ``src/models/ddr/eval.py``).

The per-regime breakdown is the PRIMARY result of the project's eval axis;
blended metrics are reported as secondary.

Design notes
- Sharpe is annualized with sqrt(periods_per_year) (daily default 252).
- ``max_drawdown`` is computed on the cumulative path WITHIN the selected
  days (for a regime row: the sub-path of that regime's days only, not the
  global equity curve). Documented so B/C/D interpret it identically.
- Undefined metrics (n < 2, zero variance, empty path) return NaN rather
  than raising or silently returning 0.
- Only regimes actually present in the predictions get rows; ``all`` is the
  blended row (secondary result).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_ORDER = ["bull", "bear", "crisis"]


def sharpe_ratio(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe of a daily return series.

    NaN when undefined (fewer than 2 observations, zero/negative variance).
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size < 2:
        return float("nan")
    mean = float(arr.mean())
    var = float(arr.var(ddof=1))
    if not np.isfinite(var) or var <= 0.0:
        return float("nan")
    return float(mean / np.sqrt(var) * np.sqrt(periods_per_year))


def max_drawdown(cum_returns: pd.Series | np.ndarray) -> float:
    """Maximum peak-to-trough decline (<= 0) on the cumulative-return path.

    NaN when the path is empty.
    """
    arr = np.asarray(cum_returns, dtype=float)
    if arr.size == 0:
        return float("nan")
    path = np.cumprod(1.0 + arr)
    return float((path / np.maximum.accumulate(path) - 1.0).min())


def evaluate_regime_breakdown(
    predictions: pd.DataFrame,
    date_col: str = "date",
    ret_col: str = "ret",
    regime_col: str = "regime",
    include_blended: bool = True,
) -> pd.DataFrame:
    """Per-regime + blended performance table for a predictions frame.

    predictions: DataFrame with at least date, ret (strategy daily returns)
        and regime (Phase-1 labels, joined by date).

    Returns a DataFrame indexed by regime (bull/bear/crisis in that order,
    plus ``all`` when include_blended) with columns:
        n_days, cum_return, sharpe_annualized, max_drawdown, mean_daily_ret
    """
    if predictions.empty:
        raise ValueError("empty predictions frame")
    missing = [c for c in (date_col, ret_col, regime_col) if c not in predictions.columns]
    if missing:
        raise ValueError(f"predictions missing required columns: {missing}")

    rets = predictions[ret_col].astype(float)
    labeled = predictions[regime_col].dropna()
    if labeled.empty:
        raise ValueError("no regime labels to group by")

    def metrics(sel: pd.Series) -> dict:
        r = rets[sel]
        arr = r.to_numpy(dtype=float)
        cum = float(np.prod(1.0 + arr) - 1.0) if arr.size else float("nan")
        return {
            "n_days": int(arr.size),
            "cum_return": cum,
            "sharpe_annualized": sharpe_ratio(arr, periods_per_year=252),
            "max_drawdown": max_drawdown(arr),
            "mean_daily_ret": float(arr.mean()) if arr.size else float("nan"),
        }

    rows, names = [], []
    for regime in REGIME_ORDER:
        idx = labeled[labeled == regime].index
        if not idx.empty:
            names.append(regime)
            rows.append(metrics(rets.index.isin(idx)))
    if include_blended:
        names.append("all")
        rows.append(metrics(pd.Series(True, index=rets.index)))
    return pd.DataFrame(rows, index=names)