"""Nonlinear probes (GBDT / MLP) alongside the linear probe battery.

The linear probes answer "is the information LINEARLY accessible from h_t?".
A reviewer can object that direction (or magnitude/volatility) may be encoded
NONLINEARLY, so this module adds two stronger probe families fit on the SAME
frozen representation with the SAME no-lookahead protocol (fit on TRAIN, eval
VAL/TEST, features standardized with train stats):

  linear   L2 logistic / ordinary least squares   (accessibility)
  gbdt     histogram gradient-boosted trees        (nonlinear, low-variance)
  mlp      small multi-layer perceptron            (nonlinear, smooth)

If direction AUC stays at chance under ALL families, the claim becomes "not
recoverable by the tested probes" rather than "not present".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

from .probes import drop_nan, standardize
from .quality import _clf_metrics, _reg_metrics

RNG = 42
FAMILIES = ("linear", "gbdt", "mlp")


def _make_model(family: str, kind: str):
    if family == "linear":
        return (LogisticRegression(penalty="l2", C=1.0, max_iter=5000, random_state=RNG)
                if kind == "clf" else LinearRegression())
    if family == "gbdt":
        return (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                               random_state=RNG)
                if kind == "clf" else
                HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                              random_state=RNG))
    if family == "mlp":
        return (MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=400,
                              early_stopping=True, random_state=RNG)
                if kind == "clf" else
                MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=400,
                             early_stopping=True, random_state=RNG))
    raise ValueError(f"unknown family {family!r}")


def probe_family_metrics(X: np.ndarray, sub: pd.DataFrame, col: str, kind: str,
                         family: str, min_rows: int = 20) -> dict | None:
    """Fit one probe family on TRAIN, evaluate VAL/TEST; None if too small."""
    y = sub[col].replace({"bull": 0, "bear": 1, "crisis": 2}).to_numpy(dtype="float64")
    tr = (sub["split"] == "train").to_numpy()
    va = (sub["split"] == "val").to_numpy()
    te = (sub["split"] == "test").to_numpy()
    Xt, yt = drop_nan(X[tr], X[tr], y[tr])
    Xv, yv = drop_nan(X[va], X[va], y[va])
    Xs, ys = drop_nan(X[te], X[te], y[te])
    if len(yt) < min_rows or len(yv) < min_rows or len(ys) < min_rows:
        return None
    if kind == "reg":  # standardize the target (scale-invariant R2; stabilizes MLP)
        mu, sd = float(yt.mean()), float(yt.std()) + 1e-12
        yt, yv, ys = (yt - mu) / sd, (yv - mu) / sd, (ys - mu) / sd
    Xtr, Xv_s = standardize(Xt, Xv)
    _, Xs_s = standardize(Xt, Xs)
    model = _make_model(family, kind).fit(Xtr, yt)
    metric = _clf_metrics if kind == "clf" else _reg_metrics
    return {"val": metric(model, Xv_s, yv), "test": metric(model, Xs_s, ys)}
