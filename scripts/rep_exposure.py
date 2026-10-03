"""Exposure-only policy test (H6): SPY / CSI300 / NIFTY.

Policies compared (long-only, no directional head):
  buy_and_hold         e = 1
  constant_learned     e = mean learned exposure (isolates scale from timing)
  exposure_sharpe      e_t = sigmoid(g(h_t)), max differentiable Sharpe
  exposure_regime      exposure_sharpe + rho*E[e_t*risk_t] (uncertainty-aware)
  A2C                  current RL policy (from rep_diagnose, for reference)

Evaluation: Sharpe, return, max drawdown, exposure, upside capture, downside
loss, regime-conditioned exposure. The question: can adaptive exposure improve
the risk/return trade-off WITHOUT directional skill?

usage: python scripts/rep_exposure.py [--markets spy,csi300,nifty] [--seeds 3]
Output: data/interpret/rep_exposure.csv + printed summary.
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
from src.models.rep_lab.offline_rl import load_offline_rep

OUT_DIR = ROOT / "data" / "interpret"
SEEDS = [20260814, 111, 222]
REGIMES = ["bull", "bear", "crisis"]
MARKETS = {
    "spy": (None, ""),
    "csi300": (REPO_ROOT / "data" / "processed_csi300_backup", "csi"),
    "nifty": (REPO_ROOT / "data" / "processed_nifty_backup", "nifty"),
}


def _regime_of(dates: pd.DatetimeIndex, processed_dir) -> np.ndarray:
    processed_dir = processed_dir or (REPO_ROOT / "data" / "processed")
    feats = pd.read_parquet(processed_dir / "features_regimes.parquet")
    idx = feats.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    reg = pd.Series(feats["regime"].to_numpy(), index=idx)
    d = dates.tz_localize(None) if getattr(dates, "tz", None) is not None else dates
    return reg.reindex(d).to_numpy()


def _metrics(a, r, regimes, cost_bps: float = 0.0) -> dict:
    a, r = a[np.isfinite(a) & np.isfinite(r)], r[np.isfinite(a) & np.isfinite(r)]
    sr = a * r - (cost_bps / 1e4) * np.abs(np.diff(a, prepend=0.0))
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
        te = data.split("test")
        r = data.market_returns[te].numpy()
        regimes = _regime_of(data.dates, processed_dir)[te]
        risk = RegimeRisk().fit(data.H, _regime_of(data.dates, processed_dir),
                                data.split("train"))

        for m, v in _metrics(np.ones_like(r), r, regimes).items():
            add(market, "buy_and_hold", m, v)

        exposure_rows = {variant: [] for variant in ("exposure_sharpe", "exposure_regime")}
        for seed in seeds:
            cfg = RepLabConfig(); cfg.seed = seed
            for variant, rho in (("exposure_sharpe", 0.0), ("exposure_regime", 1.0)):
                _, m = train_exposure("predictive", cfg, data, risk=risk, rho=rho)
                exposure_rows[variant].append(m)
        for variant, ms in exposure_rows.items():
            # average exposures across seeds (mean policy)
            a = np.mean([m["test_exposure"] for m in ms], axis=0)
            for met, val in _metrics(a, r, regimes).items():
                add(market, variant, met, val)
            add(market, variant, "sharpe_seed_mean",
                float(np.mean([m["test_sharpe"] for m in ms])))
        # constant at the mean learned exposure (scale effect only)
        mean_e = float(np.mean([m["test_exposure"] for m in exposure_rows["exposure_sharpe"]]))
        for met, val in _metrics(np.full_like(r, mean_e), r, regimes).items():
            add(market, "constant_learned", met, val)
        print(f"[done] {market}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "rep_exposure.csv"
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
                            "upside_capture", "downside_loss", "exposure_bull",
                            "exposure_bear", "exposure_crisis"] if c in p.columns]
        print(f"\n=== {market} ===")
        print(p[cols].round(3).to_string())
    print("\nwrote rep_exposure.csv under", OUT_DIR)


if __name__ == "__main__":
    main()