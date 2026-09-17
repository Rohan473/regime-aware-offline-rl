"""Representation-quality scorecard for the representation laboratory.

"Representation quality" is NOT a single number. Following the project's
research design it is decomposed into the four axes below, each measured on
the frozen representation h_t with an identical no-lookahead protocol (fit on
TRAIN, evaluate on VAL + TEST):

  A. Information quality   - what does h_t contain?  linear probes against
                             forward direction / magnitude / volatility /
                             regime / drawdown (accuracy, balanced accuracy,
                             AUC, log loss; R2, MAE, Spearman).
  B. Compression           - how efficiently is it encoded?  effective rank,
                             spectral rank, reconstruction R2 (per feature),
                             and PCA dimension contribution (incremental
                             predictive information per principal direction).
  C. Stability             - is the encoding reproducible?  mean pairwise CKA
                             across independently trained seeds of the same
                             objective (NaN with a single seed).
  D. Decision utility      - can another model exploit it?  frozen-probe
                             downstream Sharpe and trading metrics
                             (return, max drawdown, turnover, cost robustness).

Plus two attribution axes (what information is actually USED):
  E. Temporal use          - masking a lag of the causal window -> ||dh||, |da|
  F. Feature use           - zeroing / permuting a canonical feature -> ||dh||, |da|

The metrics here are deliberately generic; :mod:`scripts.rep_quality_scorecard`
runs the battery over the representation zoo and writes a long-form scorecard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score,
)

from src.eval.regime_eval import max_drawdown, sharpe_ratio

from .probes import drop_nan, standardize

RNG = 42

# (name, kind, target column) — the predictive-information battery.
PREDICTIVE_TARGETS = [
    ("direction_1", "clf", "fwd_dir_1"),
    ("direction_5", "clf", "fwd_dir_5"),
    ("direction_20", "clf", "fwd_dir_20"),
    ("magnitude_1", "reg", "abs_ret_1"),
    ("vol_5", "reg", "fwd_vol_5"),
    ("vol_20", "reg", "fwd_vol_20"),
    ("regime", "clf", "regime"),
    ("drawdown_20", "reg", "fwd_dd_20"),
]

# Primary scalar metric per probe kind, used for dimension-contribution curves.
PRIMARY = {"clf": "auc", "reg": "r2"}


# --------------------------------------------------------------------------
# A. Information quality — probes with the full metric set
# --------------------------------------------------------------------------

def _clf_metrics(clf, X: np.ndarray, y: np.ndarray) -> dict:
    pred = clf.predict(X)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }
    classes = clf.classes_
    proba = clf.predict_proba(X)
    try:
        if len(classes) == 2:
            out["auc"] = float(roc_auc_score(y, proba[:, 1]))
        else:
            out["auc"] = float(roc_auc_score(y, proba, multi_class="ovr"))
    except ValueError:
        out["auc"] = float("nan")
    try:
        out["log_loss"] = float(log_loss(y, proba, labels=classes))
    except ValueError:
        out["log_loss"] = float("nan")
    return out


def probe_classification_metrics(X_train, y_train, X_val, y_val, X_test, y_test,
                                 C: float = 1.0) -> dict:
    """L2 logistic probe -> full classification metric set on val + test."""
    Xtr, Xv = standardize(X_train, X_val)
    _, Xt = standardize(X_train, X_test)
    clf = LogisticRegression(penalty="l2", C=C, max_iter=5000, random_state=RNG)
    clf.fit(Xtr, y_train)
    baseline = max(float((y_test == c).mean()) for c in np.unique(y_train))
    return {
        "val": _clf_metrics(clf, Xv, y_val),
        "test": _clf_metrics(clf, Xt, y_test),
        "baseline_accuracy_test": baseline,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def _reg_metrics(reg, X: np.ndarray, y: np.ndarray) -> dict:
    pred = reg.predict(X)
    rho = spearmanr(y, pred).correlation if len(y) > 2 else np.nan
    return {
        "r2": float(reg.score(X, y)),
        "mae": float(np.mean(np.abs(y - pred))),
        "spearman": float(rho) if np.isfinite(rho) else float("nan"),
    }


def probe_regression_metrics(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Linear regression probe -> R2 / MAE / Spearman on val + test."""
    Xtr, Xv = standardize(X_train, X_val)
    _, Xt = standardize(X_train, X_test)
    reg = LinearRegression().fit(Xtr, y_train)
    return {
        "val": _reg_metrics(reg, Xv, y_val),
        "test": _reg_metrics(reg, Xt, y_test),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def run_probe_metrics(X: np.ndarray, sub: pd.DataFrame, col: str, kind: str,
                      min_rows: int = 20) -> dict | None:
    """Fit a probe of ``kind`` on TRAIN, evaluate VAL + TEST; None if too small."""
    y = sub[col].replace({"bull": 0, "bear": 1, "crisis": 2}).to_numpy(dtype="float64")
    tr = (sub["split"] == "train").to_numpy()
    va = (sub["split"] == "val").to_numpy()
    te = (sub["split"] == "test").to_numpy()
    Xt, yt = drop_nan(X[tr], X[tr], y[tr])
    Xv, yv = drop_nan(X[va], X[va], y[va])
    Xs, ys = drop_nan(X[te], X[te], y[te])
    if len(yt) < min_rows or len(yv) < min_rows or len(ys) < min_rows:
        return None
    fn = probe_classification_metrics if kind == "clf" else probe_regression_metrics
    return fn(Xt, yt, Xv, yv, Xs, ys)


def predictive_information(X: np.ndarray, sub: pd.DataFrame) -> dict:
    """All predictive-information probes -> {target: {metric: {val, test}}}."""
    out = {}
    for name, kind, col in PREDICTIVE_TARGETS:
        r = run_probe_metrics(X, sub, col, kind)
        if r is not None:
            out[name] = {"kind": kind, "val": r["val"], "test": r["test"]}
    return out


# --------------------------------------------------------------------------
# B. Compression — reconstruction + effective rank + dimension contribution
# --------------------------------------------------------------------------

def reconstruction_profile(X: np.ndarray, sub: pd.DataFrame,
                           z_columns: list[str]) -> dict:
    """Per-feature and mean R2 of reconstructing the current-day z-features."""
    tr = (sub["split"] == "train").to_numpy()
    va = (sub["split"] == "val").to_numpy()
    te = (sub["split"] == "test").to_numpy()
    per = {}
    for col in z_columns:
        y = sub[col].to_numpy(dtype="float64")
        Xt, yt = drop_nan(X[tr], X[tr], y[tr])
        Xv, yv = drop_nan(X[va], X[va], y[va])
        Xs, ys = drop_nan(X[te], X[te], y[te])
        if len(yt) < 20 or len(yv) < 20 or len(ys) < 20:
            per[col] = float("nan")
            continue
        per[col] = probe_regression_metrics(Xt, yt, Xv, yv, Xs, ys)["test"]["r2"]
    vals = [v for v in per.values() if np.isfinite(v)]
    return {"mean": float(np.mean(vals)) if vals else float("nan"), "per_feature": per}


def dimension_contribution(X: np.ndarray, sub: pd.DataFrame, col: str, kind: str,
                           ks=(1, 2, 4, 8, 16, 32, 64, 128)) -> dict:
    """Incremental predictive information carried by the leading PCs of X.

    PCA + probe are fit on TRAIN only, evaluated on TEST. Returns the metric
    curve vs. k principal components and ``k_to_95`` (smallest k reaching 95%
    of the full-representation metric). This is the "representation quality
    vs. representation dimension" experiment.
    """
    y = sub[col].replace({"bull": 0, "bear": 1, "crisis": 2}).to_numpy(dtype="float64")
    tr = (sub["split"] == "train").to_numpy()
    te = (sub["split"] == "test").to_numpy()
    Xtr, ytr = drop_nan(X[tr], X[tr], y[tr])
    Xte, yte = drop_nan(X[te], X[te], y[te])
    if len(ytr) < 20 or len(yte) < 20:
        return {"ks": [], "metric": [], "full": float("nan"), "k_to_95": None}

    # PCA on the RAW covariance (consistent with eff_rank: the representation's
    # own variance-ordered axes), then standardize the PC scores for the probe.
    n_comp = min(Xtr.shape[0], Xtr.shape[1])
    pca = PCA(n_components=n_comp, random_state=RNG).fit(Xtr)
    Ptr, Pte = standardize(pca.transform(Xtr), pca.transform(Xte))

    metric_key = PRIMARY[kind]
    full = _fit_probe_metric(Ptr, ytr, Pte, yte, kind)[metric_key]
    curve = []
    for k in ks:
        kk = int(min(k, n_comp))
        if kk < 1 or (curve and curve[-1]["k"] == kk):
            continue
        m = _fit_probe_metric(Ptr[:, :kk], ytr, Pte[:, :kk], yte, kind)[metric_key]
        curve.append({"k": kk, "metric": float(m)})
    k95 = None
    if np.isfinite(full) and full > 0:
        for point in curve:
            if point["metric"] >= 0.95 * full:
                k95 = point["k"]
                break
    return {"ks": [c["k"] for c in curve],
            "metric": [c["metric"] for c in curve],
            "full": float(full), "k_to_95": k95}


def _fit_probe_metric(Xtr, ytr, Xte, yte, kind: str) -> dict:
    if kind == "clf":
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=5000,
                                 random_state=RNG).fit(Xtr, ytr)
        return _clf_metrics(clf, Xte, yte)
    reg = LinearRegression().fit(Xtr, ytr)
    return _reg_metrics(reg, Xte, yte)


# --------------------------------------------------------------------------
# C. Stability — cross-seed CKA (is the representation reproducible?)
# --------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representation matrices (same rows)."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    Kx, Ky = X @ X.T, Y @ Y.T
    Kxc = Kx - Kx.mean(0, keepdims=True) - Kx.mean(1, keepdims=True) + Kx.mean()
    Kyc = Ky - Ky.mean(0, keepdims=True) - Ky.mean(1, keepdims=True) + Ky.mean()
    h1 = float(np.sqrt(np.sum(Kxc * Kxc)))
    h2 = float(np.sqrt(np.sum(Kyc * Kyc)))
    if h1 == 0 or h2 == 0:
        return float("nan")
    return float(np.sum(Kxc * Kyc) / (h1 * h2))


def mean_pairwise_cka(H_list: list[np.ndarray]) -> float:
    """Mean linear CKA over all pairs of per-seed representation matrices.

    NaN when fewer than two seeds are available — the stability axis requires
    at least two independently trained encoders of the same objective."""
    if len(H_list) < 2:
        return float("nan")
    n = min(len(h) for h in H_list)
    vals = [linear_cka(H_list[i][:n], H_list[j][:n])
            for i in range(len(H_list)) for j in range(i + 1, len(H_list))]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------
# D. Decision utility — trading metrics for a policy's actions
# --------------------------------------------------------------------------

def trading_metrics(actions, r_next, cost_bps: float = 0.0) -> dict:
    """Sharpe / total return / max drawdown / turnover / exposure of a policy."""
    a = np.asarray(actions, dtype=float)
    r = np.asarray(r_next, dtype=float)
    good = np.isfinite(a) & np.isfinite(r)
    a, r = a[good], r[good]
    if a.size < 2:
        return {k: float("nan") for k in
                ("sharpe", "total_return", "max_drawdown", "turnover", "exposure")}
    gross = a * r
    turnover = np.abs(np.diff(a, prepend=0.0))
    net = gross - turnover * (cost_bps / 1e4)
    return {
        "sharpe": float(sharpe_ratio(net)),
        "total_return": float(np.prod(1.0 + net) - 1.0),
        "max_drawdown": float(max_drawdown(net)),
        "turnover": float(turnover.mean()),
        "exposure": float(np.abs(a).mean()),
    }


def cost_robustness(actions, r_next, cost_grid=(0.0, 1.0, 5.0)) -> dict:
    """Sharpe at several transaction-cost levels (economic fragility check)."""
    return {f"sharpe_cost{c:g}": trading_metrics(actions, r_next, c)["sharpe"]
            for c in cost_grid}


# --------------------------------------------------------------------------
# E/F. Information utilisation — perturbation responses
# --------------------------------------------------------------------------

def perturbation_response(h_base: np.ndarray, h_pert: np.ndarray,
                          a_base: np.ndarray | None = None,
                          a_pert: np.ndarray | None = None) -> dict:
    """||dh||, ||dh||/sigma_h and, when actions are given, mean |da|."""
    d = h_pert - h_base
    dh = float(np.linalg.norm(d, axis=1).mean())
    sigma = float(np.linalg.norm(h_base - h_base.mean(axis=0), axis=1).mean())
    out = {"dh": dh, "dh_rel": dh / sigma if sigma > 0 else float("nan")}
    if a_base is not None and a_pert is not None:
        out["da"] = float(np.mean(np.abs(a_pert - a_base)))
        out["da_sgn"] = float(np.mean(a_pert - a_base))
    else:
        out["da"] = float("nan")
        out["da_sgn"] = float("nan")
    return out


def _encode_np(encode, W: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return np.asarray(encode(W))


def temporal_feature_sensitivity(encode, windows: np.ndarray, rows: np.ndarray,
                                 in_dim: int, lags=(1, 2, 3, 5, 10, 15),
                                 action=None) -> list[dict]:
    """Mask each lag / perturb each canonical feature; report ||dh||, |da|.

    ``encode`` maps a (B, L, F) float32 tensor to (B, D) numpy; ``action``
    (optional) maps the same tensor to (B,) positions so |da| is reported.
    """
    W = torch.tensor(np.asarray(windows)[rows], dtype=torch.float32)
    L = W.shape[1]
    hb = _encode_np(encode, W)
    ab = None if action is None else action(W)
    out = []
    for lag in lags:
        if lag > L:
            continue
        Wp = W.clone()
        Wp[:, L - 1 - lag, :] = 0.0
        hp = _encode_np(encode, Wp)
        ap = None if action is None else action(Wp)
        out.append(dict(perturbation="mask_lag", unit=f"lag{lag}",
                        **perturbation_response(hb, hp, ab, ap)))
    rng = np.random.RandomState(0)
    for j in range(in_dim):
        Wp = W.clone()
        Wp[:, :, j] = 0.0
        hp = _encode_np(encode, Wp)
        ap = None if action is None else action(Wp)
        out.append(dict(perturbation="zero_feature", unit=f"f{j}",
                        **perturbation_response(hb, hp, ab, ap)))
        perm = torch.tensor(np.stack([rng.permutation(L) for _ in range(W.shape[0])]))
        Wp = W.clone()
        Wp[:, :, j] = Wp[:, :, j].gather(1, perm)
        hp = _encode_np(encode, Wp)
        ap = None if action is None else action(Wp)
        out.append(dict(perturbation="perm_feature", unit=f"f{j}",
                        **perturbation_response(hb, hp, ab, ap)))
    return out
