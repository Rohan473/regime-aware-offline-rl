"""Experiment: "What does the hidden state know?" (next_experiments.txt §28).

Extracts frozen representations h_t from the project's trained models
(DDR-GRU, TACR transformer, Model-D encoder, the C+ logistic's linear
scores, plus raw-feature baselines) and runs an identical battery of LINEAR
PROBES against future outcome / state targets on the shared SPY date grid:

  direction (1/5/20-day)  - sign of the forward cumulative return (acc)
  magnitude |R_{t+1}|     - next-day absolute return (R2)
  volatility (5/20-day)   - realized vol over the next k days (R2)
  regime                  - bull/bear/crisis label at t (acc)
  reconstruct             - current-day z-features from h_t (R2, descriptive)

Probe fit protocol matches the models: probes fit on the TRAIN split only
(scaler too) and evaluate on VAL + TEST. This answers "does the frozen
representation linearly contain directional / magnitude / volatility /
regime / descriptive information?" — the interpretability axis of the
paper, independent of Sharpe.

Outputs (data/interpret/): probe_results.csv (long form) + probe_table.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression

from src.interpret.extract import (
    D_CKPT, DDR_CKPT, SEED, TACR_CKPT, Extracted, extract_d, extract_ddr, extract_tacr,
)
from src.interpret.probes import drop_nan, probe_classification, probe_regression, standardize
from src.interpret.targets import Z_COLUMNS, _naive, align, build_targets
from src.models.d.config import DConfig
from src.models.ddr.config import DDRConfig
from src.models.tacr.config import TACRConfig

OUT_DIR = ROOT / "data" / "interpret"

PROBES = [
    ("direction_1", "clf", "fwd_dir_1"),
    ("direction_5", "clf", "fwd_dir_5"),
    ("direction_20", "clf", "fwd_dir_20"),
    ("magnitude_1", "reg", "abs_ret_1"),
    ("vol_5", "reg", "fwd_vol_5"),
    ("vol_20", "reg", "fwd_vol_20"),
    ("regime", "clf", "regime"),
]


def _z_matrix() -> np.ndarray:
    feats = pd.read_parquet(ROOT / "data" / "processed" / "features_regimes.parquet")
    return feats[Z_COLUMNS].to_numpy(dtype="float64")


def raw_window(sub: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """20x8 flattened raw z-window per target row (160-d) — the full linear
    information the windowed models see before encoding."""
    feats = pd.read_parquet(ROOT / "data" / "processed" / "features_regimes.parquet")
    z = feats[Z_COLUMNS].to_numpy(dtype="float64")
    pos = np.arange(len(feats))
    idx = pos[:, None] - 19 + np.arange(20)[None, :]
    valid = idx >= 0
    W = z[np.clip(idx, 0, None)] * valid[..., None].astype("float64")
    f_idx = _naive(feats.index)
    remap = {d: i for i, d in enumerate(f_idx)}
    rows = W[np.array([remap[d] for d in _naive(sub.index)])]
    return rows.reshape(rows.shape[0], -1), sub


def _feats_index():
    feats = pd.read_parquet(ROOT / "data" / "processed" / "features_regimes.parquet")
    return _naive(feats.index)


def fit_logistic_scores(sub: pd.DataFrame) -> tuple[np.ndarray, float, float, float]:
    """C+ logistic (L2 on the 8 z-features, C on val) -> decision scores per
    row, its (C, test_acc, baseline)."""
    z = _z_matrix()
    X0 = z[np.array([_feats_index().get_loc(d) for d in _naive(sub.index)])]
    y = np.sign(np.array(sub["close"].pct_change().shift(-1).to_numpy(dtype="float64"), copy=True))
    y[~np.isfinite(y)] = 0.0
    y[y == 0] = 1.0
    tr = (sub["split"] == "train").to_numpy()
    va = (sub["split"] == "val").to_numpy()
    te = (sub["split"] == "test").to_numpy()
    Xtr, Xv = standardize(X0[tr], X0[va])
    Xtr2, Xt = standardize(X0[tr], X0[te])
    best_C, best_acc, best = None, -1.0, None
    for C in (0.01, 0.1, 1.0, 10.0):
        m = LogisticRegression(penalty="l2", C=C, max_iter=5000).fit(Xtr, y[tr])
        acc = float((m.predict(Xv) == y[va]).mean())
        if acc > best_acc:
            best_C, best_acc, best = C, acc, m
    base = max(float((y[te] == 1).mean()), float((y[te] == -1).mean()))
    test_acc = float((best.predict(Xt) == y[te]).mean())
    scores = np.concatenate([best.decision_function(standardize(X0[tr], X0[tr])[1]),
                             best.decision_function(Xv),
                             best.decision_function(Xt)])
    order = np.concatenate([np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te)])
    out = np.full(len(sub), np.nan)
    out[order] = scores
    return out, best_C, test_acc, base


def run_probe(X, sub, probe, kind, col) -> dict:
    y = sub[col].replace({"bull": 0, "bear": 1, "crisis": 2}).to_numpy(dtype="float64")
    tr = (sub["split"] == "train").to_numpy()
    va = (sub["split"] == "val").to_numpy()
    te = (sub["split"] == "test").to_numpy()
    Xt, yt = drop_nan(X[tr], X[tr], y[tr])
    Xv, yv = drop_nan(X[va], X[va], y[va])
    Xs, ys = drop_nan(X[te], X[te], y[te])
    if len(yt) < 20 or len(yv) < 20 or len(ys) < 20:
        return {}
    r = probe_classification(Xt, yt, Xv, yv, Xs, ys) if kind == "clf" \
        else probe_regression(Xt, yt, Xv, yv, Xs, ys)
    return {"probe": probe, "val": r["val"], "test": r["test"],
            "baseline_test": r.get("baseline_test", 0.0),
            "n_train": r["n_train"], "n_test": r["n_test"]}


def reconstruction_rsq(X, sub) -> float:
    tr = (sub["split"] == "train").to_numpy()
    va = (sub["split"] == "val").to_numpy()
    te = (sub["split"] == "test").to_numpy()
    scores = []
    for col in Z_COLUMNS:
        y = sub[col].to_numpy(dtype="float64")
        Xt, yt = drop_nan(X[tr], X[tr], y[tr])
        Xv, yv = drop_nan(X[va], X[va], y[va])
        Xs, ys = drop_nan(X[te], X[te], y[te])
        if len(yt) < 20 or len(yv) < 20 or len(ys) < 20:
            continue
        scores.append(probe_regression(Xt, yt, Xv, yv, Xs, ys)["test"])
    return float(np.nanmean(scores)) if scores else float("nan")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = build_targets()
    sub = targets[~_naive(targets.index).duplicated(keep="first")].copy()
    sub.index = _naive(sub.index)

    ddr_cfg = DDRConfig.from_yaml()
    tacr_cfg = TACRConfig.from_yaml()
    d_cfg = DConfig.from_yaml()

    reps = [
        extract_ddr(ddr_cfg, DDR_CKPT / f"s{SEED}" / "ddr_best.pt"),
        extract_tacr(tacr_cfg, TACR_CKPT / f"s{SEED}" / "tacr_best.pt"),
        extract_d(d_cfg, D_CKPT / f"s{SEED}" / "d_best.pt"),
    ]
    print(f"representations ready: {[(r.name, r.H.shape, len(r.dates)) for r in reps]}")

    rows = []

    # ---- raw baselines (identity row 1: raw 8 features; row 2: raw window) ----
    z = _z_matrix()
    map_f = {d: i for i, d in enumerate(_feats_index())}
    Xraw = z[np.array([map_f[d] for d in sub.index])]
    Xwin, _ = raw_window(sub)
    for name, X in (("Raw 8 features (zd)", Xraw), ("Raw 20x8 window (160d)", Xwin)):
        for probe_name, kind, col in PROBES:
            r = run_probe(X, sub, probe_name, kind, col)
            if r:
                rows.append(dict(representation=name, **r))
        rows.append(dict(representation=name, probe="reconstruct", val=float("nan"),
                         test=reconstruction_rsq(X, sub), baseline_test=0.0,
                         n_train=int((sub["split"] == "train").sum()),
                         n_test=int((sub["split"] == "test").sum())))

    # ---- C+ logistic scores ----
    scores, logi_C, logi_acc, logi_base = fit_logistic_scores(sub)
    logi_rep = Extracted(name="C+ logistic score (1d)",
                         dates=pd.DatetimeIndex(sub.index), H=scores[:, None])

    for rep in [*reps, logi_rep]:
        rsub, X = align(rep.dates, rep.H, sub)
        for probe_name, kind, col in PROBES:
            r = run_probe(X, rsub, probe_name, kind, col)
            if r:
                rows.append(dict(representation=rep.name, **r))
        rows.append(dict(representation=rep.name, probe="reconstruct", val=float("nan"),
                         test=reconstruction_rsq(X, rsub), baseline_test=0.0,
                         n_train=int((rsub["split"] == "train").sum()),
                         n_test=int((rsub["split"] == "test").sum())))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "probe_results.csv", index=False)
    table = df.pivot_table(index="representation", columns="probe", values="test")
    table = table.reindex(columns=[p for p, _, _ in PROBES] + ["reconstruct"])
    table.to_csv(OUT_DIR / "probe_table.csv")

    print(f"\n[C+ logistic] C={logi_C}, test direction acc {logi_acc:.4f} "
          f"(majority baseline {logi_base:.4f}, raw-linear comparand below)")
    print("\n=== test-split linear-probe metrics (classification: accuracy | regression: R2) ===")
    print(table.round(3).to_string())
    print("\nprobe_results.csv / probe_table.csv written under", OUT_DIR)


if __name__ == "__main__":
    main()