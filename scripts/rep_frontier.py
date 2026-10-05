"""Empirical risk-return trade-off map of the exposure-policy family.

NOT a Markowitz efficient-frontier claim. We plot annualized volatility vs
annualized return for a family of exposure policies and check (i) which lie
above the buy-and-hold ray (same Sharpe) and (ii) which are Pareto-non-dominated
in (vol, return).

Family (per market):
  constant-exposure sweep   a = c for c in {0,.1,...,1}   -> the BH ray
  constant-alpha blends      a = e + al*(1-e), al in {0,...,1} (curve)
  adaptive                   a = e_t (max-Sharpe exposure)
  regime-constrained         a = e_t trained with +rho*E[e*risk]
  floor                      a = max(e, .5)
  hybrid                     a = e_t + alpha_t*(1-e_t), alpha learned
  buy_and_hold               a = 1

5-seed mean daily series; metrics annualized (mean*252, std*sqrt(252)).

usage: python scripts/rep_frontier.py [--markets spy,nifty,csi300]
Output: data/interpret/rep_frontier.csv + paper/risk_return_frontier.png.
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
SEEDS = [20260814, 111, 222, 333, 444]
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


def metrics(daily: np.ndarray) -> dict:
    d = daily[np.isfinite(daily)]
    ann_ret = float(d.mean() * 252)
    ann_vol = float(d.std() * np.sqrt(252))
    return {"ann_return": ann_ret, "ann_vol": ann_vol,
            "return": float(np.prod(1 + d) - 1),
            "sharpe": float(sharpe_ratio(d, 252)),
            "maxdd": float(max_drawdown(d))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(MARKETS))
    args = ap.parse_args()
    markets = [m for m in args.markets.split(",") if m]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for market in markets:
        processed_dir, tag = MARKETS[market]
        cfg0 = RepLabConfig()
        cfg0.processed_dir, cfg0.tag = processed_dir, tag
        data = load_offline_rep("predictive", cfg0)
        tr, va, te = data.split("train"), data.split("val"), data.split("test")
        r = data.market_returns[te].numpy()

        e_list, e_reg_list = [], []
        for seed in SEEDS:
            cfg = RepLabConfig(); cfg.seed = seed
            h, _ = train_exposure("predictive", cfg, data, risk=None, rho=0.0)
            with torch.no_grad():
                e_list.append(h.e(data.H).numpy())
            risk = RegimeRisk().fit(data.H, _regime_of(data.dates, processed_dir), tr)
            h2, _ = train_exposure("predictive", cfg, data, risk=risk, rho=1.0)
            with torch.no_grad():
                e_reg_list.append(h2.e(data.H).numpy())
        e_full = np.mean(e_list, axis=0)
        e_reg_full = np.mean(e_reg_list, axis=0)
        e_te = e_full[te]
        e_reg_te = e_reg_full[te]

        # hybrid (learned alpha on risk/ood/vol), 3 seeds
        risk_all = RegimeRisk().fit(data.H, _regime_of(data.dates, processed_dir), tr)
        F = np.stack([risk_all.risk(data.H), _ood(data.H[tr].numpy(), data.H.numpy()),
                      _features(processed_dir, data.dates)], axis=1).astype("float64")
        a_hyb = []
        for seed in SEEDS[:3]:
            e_tr = torch.tensor(e_full[tr], dtype=torch.float32)
            e_va = torch.tensor(e_full[va], dtype=torch.float32)
            e_te_t = torch.tensor(e_te, dtype=torch.float32)
            F_tr = torch.tensor(F[tr], dtype=torch.float32)
            F_va = torch.tensor(F[va], dtype=torch.float32)
            F_te = torch.tensor(F[te], dtype=torch.float32)
            r_tr = torch.tensor(data.market_returns[tr], dtype=torch.float32)
            r_va = torch.tensor(data.market_returns[va], dtype=torch.float32)
            head, _ = train_blend(e_tr, F_tr, r_tr, e_va, F_va, r_va, seed=seed)
            mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-8
            with torch.no_grad():
                al = head.alpha((F_te - mu) / sd).numpy()
            a_hyb.append(compose(e_te_t, torch.tensor(al)).numpy())
        a_hyb = np.mean(a_hyb, axis=0)

        policies = {"buy_and_hold": np.ones_like(r), "adaptive": e_te,
                    "regime_constrained": e_reg_te, "floor_50": np.maximum(e_te, 0.5),
                    "hybrid": a_hyb}
        for c in np.arange(0.0, 1.01, 0.1):
            policies[f"const_{c:.1f}"] = np.full_like(r, c)
        for al in np.arange(0.0, 1.01, 0.2):
            policies[f"blend_{al:.1f}"] = e_te + al * (1 - e_te)
        for name, a in policies.items():
            m = metrics(a * r)
            rows.append({"market": market, "policy": name, **m})
        print(f"[done] {market}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rep_frontier.csv", index=False)

    # Pareto frontier per market on (ann_vol, ann_return)
    def pareto(g):
        pts = g[["ann_vol", "ann_return"]].to_numpy()
        keep = []
        for i in range(len(pts)):
            dominated = ((g["ann_return"].to_numpy() >= pts[i, 1]) &
                         (g["ann_vol"].to_numpy() <= pts[i, 0]) &
                         ((g["ann_return"].to_numpy() > pts[i, 1]) |
                          (g["ann_vol"].to_numpy() < pts[i, 0])))
            keep.append(not dominated.any())
        return keep

    pd.set_option("display.width", 240)
    for market in markets:
        g = df[df["market"] == market].reset_index(drop=True)
        g["pareto"] = pareto(g)
        print(f"\n=== {market}: risk-return coordinates ===")
        print(g[["policy", "ann_return", "ann_vol", "sharpe", "maxdd", "pareto"]]
              .round(3).to_string(index=False))

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(markets), figsize=(6 * len(markets), 5),
                                 squeeze=False)
        for ax, market in zip(axes[0], markets):
            g = df[df["market"] == market]
            ax.scatter(g["ann_vol"], g["ann_return"], s=24, c="#3b5b8c")
            for _, row in g.iterrows():
                ax.annotate(row["policy"], (row["ann_vol"], row["ann_return"]),
                            fontsize=6.5, xytext=(3, 3), textcoords="offset points")
            bh = g[g["policy"] == "buy_and_hold"].iloc[0]
            xs = np.linspace(0, g["ann_vol"].max() * 1.05, 20)
            ax.plot(xs, bh["sharpe"] * xs, "--", color="#8c3b3b", lw=1,
                    label=f"BH ray (Sharpe {bh['sharpe']:.2f})")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("annualized volatility")
            ax.set_ylabel("annualized return")
            ax.set_title(market)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ROOT / "paper" / "risk_return_frontier.png", dpi=150)
        print("\nwrote paper/risk_return_frontier.png")
    except Exception as exc:  # noqa: BLE001
        print(f"[figure skipped] {type(exc).__name__}: {exc}")
    print("wrote rep_frontier.csv under", OUT_DIR)


if __name__ == "__main__":
    main()