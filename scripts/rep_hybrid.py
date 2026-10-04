"""Hybrid exposure blend (H7): SPY / CSI300 / NIFTY.

Policies compared:
  buy_and_hold      a = 1
  adaptive          a = e_t (max-Sharpe exposure policy, frozen)
  constant_mean     a = mean(e)  (isolates timing from scaling)
  blend_50          a = e + 0.5*(1-e)  (simple constant blend)
  floor_50          a = max(e, 0.5)    (floor-exposure baseline)
  blend_risk        a = e + alpha_t*(1-e), alpha_t = sigma(W.[risk,ood,vol])
                    (proposed risk-aware blend; alpha uses only info at t)

Metrics: Sharpe, return, maxDD, exposure, upside capture, downside loss,
regime-conditioned exposure, and alpha stats for the proposed blend.

usage: python scripts/rep_hybrid.py [--markets spy,csi300,nifty] [--seeds 3]
Output: data/interpret/rep_hybrid.csv + printed summary.
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

from src.data.loaders import REPO_ROOT
from src.eval.regime_eval import max_drawdown, sharpe_ratio
from src.models.rep_lab.config import RepLabConfig
from src.models.rep_lab.exposure import train_exposure
from src.models.rep_lab.gated import RegimeRisk
from src.models.rep_lab.hybrid import compose, train_blend
from src.models.rep_lab.offline_rl import load_offline_rep

OUT_DIR = ROOT / "data" / "interpret"
SEEDS = [20260814, 111, 222]
REGIMES = ["bull", "bear", "crisis"]
MARKETS = {
    "spy": (None, ""),
    "csi300": (REPO_ROOT / "data" / "processed_csi300_backup", "csi"),
    "nifty": (REPO_ROOT / "data" / "processed_nifty_backup", "nifty"),
}


def _regime_of(dates, processed_dir):
    processed_dir = processed_dir or (REPO_ROOT / "data" / "processed")
    f = pd.read_parquet(processed_dir / "features_regimes.parquet")
    idx = f.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    reg = pd.Series(f["regime"].to_numpy(), index=idx)
    d = dates.tz_localize(None) if getattr(dates, "tz", None) is not None else dates
    return reg.reindex(d).to_numpy()


def _features(processed_dir, dates):
    processed_dir = processed_dir or (REPO_ROOT / "data" / "processed")
    f = pd.read_parquet(processed_dir / "features_regimes.parquet")
    idx = f.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    f.index = idx
    d = dates.tz_localize(None) if getattr(dates, "tz", None) is not None else dates
    return f["z_realized_vol_20d"].reindex(d).to_numpy()


def _ood(H_train, H_all):
    mu = H_train.mean(0)
    inv = np.linalg.pinv(np.cov(H_train, rowvar=False) + 1e-6 * np.eye(H_train.shape[1]))
    d = H_all - mu
    return np.sqrt((d @ inv * d).sum(1))


def _metrics(a, r, regimes):
    a, r = a[np.isfinite(a) & np.isfinite(r)], r[np.isfinite(a) & np.isfinite(r)]
    sr = a * r
    up, dn = r > 0, r < 0
    out = {"sharpe": float(sharpe_ratio(sr, 252)),
           "return": float(np.prod(1 + sr) - 1),
           "maxdd": float(max_drawdown(sr)),
           "exposure": float(np.abs(a).mean()),
           "upside_capture": float((a * r)[up].mean() / r[up].mean()) if up.sum() else np.nan,
           "downside_loss": float((a * r)[dn].mean() / r[dn].mean()) if dn.sum() else np.nan}
    for g in REGIMES:
        m = regimes == g
        out[f"exposure_{g}"] = float(a[m].mean()) if m.sum() else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(MARKETS))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    markets = [m for m in args.markets.split(",") if m]
    seeds = SEEDS[: args.seeds]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(market, policy, metric, value):
        rows.append({"market": market, "policy": policy, "metric": metric,
                     "value": float(value)})

    for market in markets:
        processed_dir, tag = MARKETS[market]
        cfg0 = RepLabConfig()
        cfg0.processed_dir, cfg0.tag = processed_dir, tag
        data = load_offline_rep("predictive", cfg0)
        tr, va, te = data.split("train"), data.split("val"), data.split("test")
        r = data.market_returns[te].numpy()
        regimes = _regime_of(data.dates, processed_dir)[te]

        # frozen adaptive exposure policy (mean over seeds), full-date exposure
        e_full_list = []
        for seed in seeds:
            cfg = RepLabConfig(); cfg.seed = seed
            head, _ = train_exposure("predictive", cfg, data, risk=None, rho=0.0)
            with torch.no_grad():
                e_full_list.append(head.e(data.H).numpy())
        e_full = np.mean(e_full_list, axis=0)
        e = e_full[te]
        # train-split exposures for alpha training
        risk_all = RegimeRisk().fit(data.H, _regime_of(data.dates, processed_dir), tr)
        risk_v = risk_all.risk(data.H)
        vol_all = _features(processed_dir, data.dates)
        ood_all = _ood(data.H[tr].numpy(), data.H.numpy())
        F = np.stack([risk_v, ood_all, vol_all], axis=1).astype("float64")
        e_train = torch.tensor(e_full[tr], dtype=torch.float32)
        e_val = torch.tensor(e_full[va], dtype=torch.float32)
        e_te = torch.tensor(e, dtype=torch.float32)  # e is already the test slice
        F_tr = torch.tensor(F[tr], dtype=torch.float32)
        F_va = torch.tensor(F[va], dtype=torch.float32)
        F_te = torch.tensor(F[te], dtype=torch.float32)
        r_tr = torch.tensor(data.market_returns[tr], dtype=torch.float32)
        r_va = torch.tensor(data.market_returns[va], dtype=torch.float32)

        head, _ = train_blend(e_train, F_tr, r_tr, e_val, F_va, r_va, seed=20260814)
        mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-8
        with torch.no_grad():
            alpha_te = head.alpha((F_te - mu) / sd).numpy()
            a_blend = compose(e_te, torch.tensor(alpha_te)).numpy()

        policies = {
            "buy_and_hold": np.ones_like(r),
            "adaptive": e,
            "constant_mean": np.full_like(r, e.mean()),
            "blend_50": e + 0.5 * (1 - e),
            "floor_50": np.maximum(e, 0.5),
            "blend_risk": a_blend,
        }
        for name, a in policies.items():
            for m, v in _metrics(a, r, regimes).items():
                add(market, name, m, v)
        # alpha stats for the proposed blend
        add(market, "blend_risk", "alpha_mean", alpha_te.mean())
        for g in REGIMES:
            mm = regimes == g
            add(market, "blend_risk", f"alpha_{g}", alpha_te[mm].mean() if mm.sum() else np.nan)
        add(market, "blend_risk", "corr_alpha_vol",
            float(np.corrcoef(alpha_te, vol_all[te])[0, 1]))
        print(f"[done] {market}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "rep_hybrid.csv"
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        old = old[~old["market"].isin(markets)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)

    pd.set_option("display.width", 240)
    for market in markets:
        d = df[df["market"] == market]
        p = d.pivot_table(index="policy", columns="metric", values="value")
        cols = [c for c in ["sharpe", "return", "maxdd", "exposure",
                            "upside_capture", "downside_loss", "alpha_mean",
                            "alpha_bull", "alpha_bear", "alpha_crisis"] if c in p.columns]
        print(f"\n=== {market} ===")
        print(p[cols].round(3).to_string())
    print("\nwrote rep_hybrid.csv under", OUT_DIR)


if __name__ == "__main__":
    main()