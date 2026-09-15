"""Linear-probe utilities for the representation laboratory.

Probes are deliberately simple (L2 logistic regression for classification,
linear regression for regression) per next_experiments.txt §3: if a linear
probe can read a property out of the frozen h_t, the representation
contains that information linearly. Every probe is fit on the TRAIN split
and evaluated on VAL and TEST — the same no-lookahead protocol as the
models. Inputs are standardized with scaler stats fit on train only.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

RNG = 42


def standardize(X_train: np.ndarray, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler().fit(X_train)
    return sc.transform(X_train), sc.transform(X_eval)


def probe_classification(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    C: float = 1.0,
) -> dict:
    """L2 logistic probe on train, accuracy on val+test, plus majority baseline."""
    Xtr, Xv = standardize(X_train, X_val)
    _, Xt = standardize(X_train, X_test)
    clf = LogisticRegression(penalty="l2", C=C, max_iter=5000, random_state=RNG)
    clf.fit(Xtr, y_train)
    acc_val = float((clf.predict(Xv) == y_val).mean())
    acc_test = float((clf.predict(Xt) == y_test).mean())
    baseline_test = float(max((y_test == c).mean() for c in np.unique(y_train)))
    return {
        "val": acc_val,
        "test": acc_test,
        "baseline_test": baseline_test,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def probe_regression(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> dict:
    """Linear regression probe; R2 on val+test (R2=0 = mean predictor)."""
    Xtr, Xv = standardize(X_train, X_val)
    _, Xt = standardize(X_train, X_test)
    reg = LinearRegression()
    reg.fit(Xtr, y_train)
    return {
        "val": float(reg.score(Xv, y_val)),
        "test": float(reg.score(Xt, y_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def drop_nan(rows: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows with any NaN in X or y (probe-time tail clipping)."""
    good = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[good], y[good]