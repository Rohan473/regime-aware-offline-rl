"""Representation-quality scorecard (multidimensional, not a single number).

Runs the :mod:`src.interpret.quality` battery over the representation zoo and
writes a long-form scorecard plus the quality-vs-dimension curve:

  information   linear probes: direction (1/5/20) / magnitude / vol (5/20) /
                regime / drawdown -> accuracy, balanced accuracy, AUC, log loss
                (classification) and R2, MAE, Spearman (regression)
  preservation  reconstruction R2 of the current-day z-features (mean + per feat)
  efficiency    effective rank / spectral rank / top1 / top5 / k95
  redundancy    PCA dimension contribution: incremental predictive info per PC,
                k_to_95 = how many dimensions carry 95% of the full signal
  stability     mean pairwise CKA across seeds of the same objective
  trading       supervised + frozen-RL policies: Sharpe, return, max drawdown,
                turnover, exposure, cost robustness (1/5 bps)
  temporal      masking lag L of the causal window -> ||dh||, |da|
  feature       zero/permute a canonical feature -> ||dh||, |da|

Representations: raw window, replab {auto, predictive, contrastive}; the
project models (DDR / TACR / D) can be added with ``--with-project-models``
(H-based axes only).

Outputs (data/interpret/):
  rep_quality_scorecard.csv    long form: representation, dim, axis, metric, value
  rep_quality_vs_dimension.csv representation x target x k -> probe metric
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.representation_rank import eff_rank
from src.interpret.quality import (
    PREDICTIVE_TARGETS, cost_robustness, dimension_contribution,
    mean_pairwise_cka, predictive_information, reconstruction_profile,
    run_probe_metrics, temporal_feature_sensitivity, trading_metrics,
)
from src.interpret.targets import Z_COLUMNS, _naive, align, build_targets
from src.models.ddr.data import load_ddr_data, split_ddr_data
from src.models.rep_lab.config import OBJECTIVES, RepLabConfig, tag_path
from src.models.rep_lab.model import DownstreamAgent, RawFeat
from src.models.rep_lab.train import BEST, extract_rep, load_rep, raw_window_rep

OUT_DIR = ROOT / "data" / "interpret"
CANONICAL = ["ret_1d", "ret_5d", "ret_20d", "rv20", "rsi", "macd", "volz", "boll"]


def _add(rows, rep, dim, axis, metric, value, split="test"):
    rows.append(dict(representation=rep, dim=int(dim), axis=axis,
                     metric=metric, value=float(value) if np.isfinite(value) else float("nan"),
                     split=split))


# --------------------------------------------------------------------------
# representation builders
# --------------------------------------------------------------------------

def rep_H(name: str, cfg: RepLabConfig):
    if name == "raw":
        return raw_window_rep(cfg)
    return extract_rep(name, cfg, tag_path(cfg, name) / BEST)


def rep_encoder(name: str, cfg: RepLabConfig):
    if name == "raw":
        return None
    return load_rep(name, cfg, tag_path(cfg, name) / BEST).encoder


def downstream_agent(name: str, cfg: RepLabConfig):
    ck = tag_path(cfg, name, subdir="downstream") / BEST
    if not ck.exists():
        return None
    feat = None if name != "raw" else RawFeat(cfg.window * cfg.in_dim, cfg.hidden)
    agent = DownstreamAgent(cfg.hidden, feat)
    agent.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"])
    agent.eval()
    return agent


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------

def supervised_actions(X: np.ndarray, rsub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = rsub["fwd_dir_1"].to_numpy()
    tr = (rsub["split"] == "train").to_numpy()
    te = (rsub["split"] == "test").to_numpy()
    good = np.isfinite(y) & np.isfinite(X).all(axis=1)
    Xtr, ytr = X[tr & good], y[tr & good]
    if len(ytr) < 10 or np.isclose((ytr > 0).mean(), 0) or np.isclose((ytr < 0).mean(), 0):
        return np.array([]), np.array([])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    yhat = clf.decision_function(sc.transform(X[te]))
    return np.sign(yhat), rsub["fwd_ret_1"].to_numpy()[te]


def rl_actions(name: str, cfg: RepLabConfig) -> tuple[np.ndarray, np.ndarray]:
    agent = downstream_agent(name, cfg)
    if agent is None:
        return np.array([]), np.array([])
    data = load_ddr_data(cfg.window)
    test = split_ddr_data(data)["test"]
    from src.models.objective_lab.train import _valid_mask

    mask = _valid_mask(test)
    if int(mask.sum()) < 2:
        return np.array([]), np.array([])
    encoder = rep_encoder(name, cfg)

    def feed(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return encoder.encode(x) if encoder is not None else x.reshape(x.shape[0], -1)

    with torch.no_grad():
        a = agent.mu(feed(test.windows[mask])).numpy()
    return a, test.next_returns[mask].numpy()


# --------------------------------------------------------------------------
# per-representation battery
# --------------------------------------------------------------------------

def score_representation(name: str, cfg: RepLabConfig, sub: pd.DataFrame,
                         rows: list, dim_rows: list, with_sensitivity: bool) -> None:
    rep = rep_H(name, cfg)
    rsub, X = align(rep.dates, rep.H, sub)
    dim = X.shape[1]
    te = (rsub["split"] == "test").to_numpy()

    # A. information quality
    info = predictive_information(X, rsub)
    for target, payload in info.items():
        for split in ("val", "test"):
            for metric, value in payload[split].items():
                _add(rows, rep.name, dim, "information", f"{target}.{metric}", value, split)
    for target, kind, col in PREDICTIVE_TARGETS:
        r = run_probe_metrics(X, rsub, col, kind)
        if r is not None and kind == "clf":
            _add(rows, rep.name, dim, "information", f"{target}.baseline_acc",
                 r["baseline_accuracy_test"], "test")

    # B. preservation + efficiency
    rec = reconstruction_profile(X, rsub, Z_COLUMNS)
    _add(rows, rep.name, dim, "preservation", "reconstruction_r2_mean", rec["mean"])
    for col, v in rec["per_feature"].items():
        _add(rows, rep.name, dim, "preservation", f"reconstruction_r2.{col}", v)
    for k, v in eff_rank(X[te]).items():
        _add(rows, rep.name, dim, "efficiency", k, v)

    # B'. redundancy / quality-vs-dimension
    for target, kind, col in (("direction_1", "clf", "fwd_dir_1"),
                              ("magnitude_1", "reg", "abs_ret_1")):
        dc = dimension_contribution(X, rsub, col, kind)
        for k, m in zip(dc["ks"], dc["metric"]):
            dim_rows.append(dict(representation=rep.name, dim=int(dim), target=target,
                                 k=int(k), metric=float(m)))
        _add(rows, rep.name, dim, "redundancy", f"{target}.full_metric", dc["full"])
        if dc["k_to_95"] is not None:
            _add(rows, rep.name, dim, "redundancy", f"{target}.k_to_95", dc["k_to_95"])

    # C. stability (cross-seed CKA over every trained seed of this objective)
    if name != "raw":
        seeds = sorted(tag_path(cfg, name).parent.glob("s*/best.pt"))
        Hs = [extract_rep(name, cfg, p).H for p in seeds]
        _add(rows, rep.name, dim, "stability", "seed_cka_mean", mean_pairwise_cka(Hs))
        _add(rows, rep.name, dim, "stability", "n_seeds", len(seeds))

    # D. trading utility
    for policy, (a, r) in (("Supervised", supervised_actions(X, rsub)),
                           ("RL-A2C", rl_actions(name, cfg))):
        if a.size < 2:
            _add(rows, rep.name, dim, "trading", f"{policy}.sharpe", float("nan"))
            continue
        tm = trading_metrics(a, r)
        for metric, value in tm.items():
            _add(rows, rep.name, dim, "trading", f"{policy}.{metric}", value)
        for metric, value in cost_robustness(a, r).items():
            _add(rows, rep.name, dim, "trading", f"{policy}.{metric}", value)

    # E/F. information utilisation (learned reps only)
    if not with_sensitivity:
        return
    data = load_ddr_data(cfg.window)
    test = split_ddr_data(data)["test"]
    from src.models.objective_lab.train import _valid_mask

    rows_pos = np.flatnonzero(_valid_mask(test))
    if len(rows_pos) > 300:
        rows_pos = np.unique(rows_pos[np.linspace(0, len(rows_pos) - 1, 300).astype(int)])
    agent = downstream_agent(name, cfg)
    encoder = rep_encoder(name, cfg)

    def encode(W: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return (encoder.encode(W) if encoder is not None
                    else W.reshape(W.shape[0], -1)).numpy()

    def action(W: torch.Tensor) -> np.ndarray | None:
        if agent is None:
            return None
        with torch.no_grad():
            z = encoder.encode(W) if encoder is not None else W.reshape(W.shape[0], -1)
            return agent.mu(z).numpy()

    sens = temporal_feature_sensitivity(
        encode, test.windows.numpy(), rows_pos, cfg.in_dim,
        action=action if agent is not None else None)
    for s in sens:
        axis = "temporal" if s["perturbation"] == "mask_lag" else "feature"
        tag = f"{s['perturbation']}:{s['unit']}"
        _add(rows, rep.name, dim, axis, f"{tag}.dh", s["dh"])
        _add(rows, rep.name, dim, axis, f"{tag}.dh_rel", s["dh_rel"])
        _add(rows, rep.name, dim, axis, f"{tag}.da", s["da"])


def score_project_models(sub: pd.DataFrame, rows: list, dim_rows: list) -> None:
    """DDR / TACR / D: H-based axes only (info, preservation, efficiency, dim)."""
    from src.interpret.extract import (
        D_CKPT, DDR_CKPT, SEED, TACR_CKPT, extract_d, extract_ddr, extract_tacr,
    )
    from src.models.d.config import DConfig
    from src.models.ddr.config import DDRConfig
    from src.models.tacr.config import TACRConfig

    ddr_cfg, tacr_cfg, d_cfg = DDRConfig.from_yaml(), TACRConfig.from_yaml(), DConfig.from_yaml()
    specs = [
        (DDR_CKPT, "ddr_best.pt", extract_ddr, ddr_cfg),
        (TACR_CKPT, "tacr_best.pt", extract_tacr, tacr_cfg),
        (D_CKPT, "d_best.pt", extract_d, d_cfg),
    ]
    reps = [extract(cfg, root / f"s{SEED}" / fname) for root, fname, extract, cfg in specs]
    for rep, (root, fname, extract, cfg) in zip(reps, specs):
        rsub, X = align(rep.dates, rep.H, sub)
        dim = X.shape[1]
        te = (rsub["split"] == "test").to_numpy()
        for target, payload in predictive_information(X, rsub).items():
            for metric, value in payload["test"].items():
                _add(rows, rep.name, dim, "information", f"{target}.{metric}", value)
        rec = reconstruction_profile(X, rsub, Z_COLUMNS)
        _add(rows, rep.name, dim, "preservation", "reconstruction_r2_mean", rec["mean"])
        for k, v in eff_rank(X[te]).items():
            _add(rows, rep.name, dim, "efficiency", k, v)
        for target, kind, col in (("direction_1", "clf", "fwd_dir_1"),
                                  ("magnitude_1", "reg", "abs_ret_1")):
            dc = dimension_contribution(X, rsub, col, kind)
            for k, m in zip(dc["ks"], dc["metric"]):
                dim_rows.append(dict(representation=rep.name, dim=int(dim), target=target,
                                     k=int(k), metric=float(m)))
            if dc["k_to_95"] is not None:
                _add(rows, rep.name, dim, "redundancy", f"{target}.k_to_95", dc["k_to_95"])
        seeds = sorted(root.glob(f"s*/{fname}"))
        Hs = [extract(cfg, p).H for p in seeds]
        _add(rows, rep.name, dim, "stability", "seed_cka_mean", mean_pairwise_cka(Hs))
        _add(rows, rep.name, dim, "stability", "n_seeds", len(seeds))


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_summary(sc: pd.DataFrame) -> None:
    pd.set_option("display.width", 240)
    info = sc[(sc["axis"] == "information") & (sc["split"] == "test")].copy()
    key = {"direction_1.auc": "dir1_AUC", "magnitude_1.r2": "mag_R2",
           "vol_20.r2": "vol20_R2", "regime.balanced_accuracy": "regime_bacc",
           "drawdown_20.r2": "dd20_R2"}
    info = info[info["metric"].isin(key)]
    if not info.empty:
        info["metric"] = info["metric"].map(key)
        print("\n=== information quality (test) ===")
        print(info.pivot_table(index="representation", columns="metric",
                               values="value").round(3).to_string())

    eff = sc[(sc["axis"] == "efficiency") &
             (sc["metric"].isin(["effective_rank", "spectral_rank", "k95"]))]
    if not eff.empty:
        print("\n=== efficiency (test) ===")
        print(eff.pivot_table(index="representation", columns="metric",
                              values="value").round(2).to_string())

    tr = sc[(sc["axis"] == "trading") &
            (sc["metric"].isin(["Supervised.sharpe", "RL-A2C.sharpe",
                                "Supervised.max_drawdown", "Supervised.turnover"]))]
    if not tr.empty:
        print("\n=== decision utility (test) ===")
        print(tr.pivot_table(index="representation", columns="metric",
                             values="value").round(3).to_string())

    st = sc[(sc["axis"] == "stability") & (sc["metric"] == "seed_cka_mean")]
    ns = sc[(sc["axis"] == "stability") & (sc["metric"] == "n_seeds")]
    if not st.empty:
        n_map = dict(zip(ns["representation"], ns["value"].astype(int)))
        st = st.assign(n_seeds=[n_map.get(r, 0) for r in st["representation"]])
        print("\n=== stability (cross-seed CKA) ===")
        print(st[["representation", "value", "n_seeds"]].round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-project-models", action="store_true",
                    help="also score DDR / TACR / D (H-based axes only)")
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="skip temporal/feature perturbation passes")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = RepLabConfig()
    sub = build_targets()
    sub = sub[~_naive(sub.index).duplicated(keep="first")].copy()
    sub.index = _naive(sub.index)

    rows: list = []
    dim_rows: list = []
    for name in ["raw", *OBJECTIVES]:
        if name != "raw" and not (tag_path(cfg, name) / BEST).exists():
            print(f"[skip] {name}: no checkpoint")
            continue
        print(f"[score] {name} ...")
        try:
            score_representation(name, cfg, sub, rows, dim_rows,
                                 with_sensitivity=not args.no_sensitivity)
        except Exception as exc:  # keep the scorecard running
            print(f"  [error] {name}: {type(exc).__name__}: {exc}")

    if args.with_project_models:
        print("[score] project models (DDR/TACR/D) ...")
        score_project_models(sub, rows, dim_rows)

    sc = pd.DataFrame(rows)
    sc.to_csv(OUT_DIR / "rep_quality_scorecard.csv", index=False)
    dim_df = pd.DataFrame(dim_rows)
    dim_df.to_csv(OUT_DIR / "rep_quality_vs_dimension.csv", index=False)

    print_summary(sc)
    print(f"\nwrote {len(sc)} scorecard rows and {len(dim_df)} dimension points to {OUT_DIR}")


if __name__ == "__main__":
    main()
