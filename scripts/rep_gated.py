"""Regime-aware, uncertainty-constrained offline RL - the H5 experiment.

Can an offline RL system LEARN when it is safe to deviate from observed
behaviour and when it should stay conservative?

Policies compared (all on the same frozen representation and transitions):
  behaviour_clone   a = a_beh                              (lambda = 0)
  A2C               a = full deviation                     (lambda = 1, unconstrained)
  gated_rho0        a = a_beh + lambda*d, no penalty       (adaptive, unconstrained)
  gated_rho1        a = a_beh + lambda*d, + rho*lambda*risk(proposed model)
  oracle            a = a_beh + (1 if bull else .1)*(a_A2C - a_beh)  (true test
                    regime labels = look-ahead upper bound)

Question answered by: (1) does gated_rho1 beat behaviour_clone AND A2C?
(2) is the learned lambda higher in bull and lower in bear/crisis?
(3) how close does it get to the oracle?

usage: python scripts/rep_gated.py [--reps predictive,contrastive] [--seeds 10]
Output: data/interpret/rep_gated.csv + printed summary.
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
from src.eval.regime_eval import sharpe_ratio
from src.models.rep_lab.config import RepLabConfig
from src.models.rep_lab.gated import RegimeRisk, compose, train_gated
from src.models.rep_lab.offline_rl import load_offline_rep, train_offline

OUT_DIR = ROOT / "data" / "interpret"
SEEDS = [20260814, 111, 222, 333, 444, 555, 666, 777, 888, 999]
REPS = ["predictive", "contrastive"]
REGIMES = ["bull", "bear", "crisis"]
MARKETS = {
    "spy": (None, ""),
    "csi300": (REPO_ROOT / "data" / "processed_csi300_backup", "csi"),
    "nifty": (REPO_ROOT / "data" / "processed_nifty_backup", "nifty"),
}


def _regime_of(dates: pd.DatetimeIndex, processed_dir=None) -> np.ndarray:
    processed_dir = processed_dir or (REPO_ROOT / "data" / "processed")
    feats = pd.read_parquet(processed_dir / "features_regimes.parquet")
    idx = feats.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    reg = pd.Series(feats["regime"].to_numpy(), index=idx)
    d = dates.tz_localize(None) if getattr(dates, "tz", None) is not None else dates
    return reg.reindex(d).to_numpy()


def _group_sharpe(a: np.ndarray, r: np.ndarray, regimes: np.ndarray) -> dict:
    sr = a * r
    out = {}
    for g in REGIMES:
        m = regimes == g
        if m.sum() >= 2:
            out[g] = float(sharpe_ratio(sr[m], 252))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default=",".join(REPS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--market", default="spy", choices=list(MARKETS))
    args = ap.parse_args()
    reps = [r for r in args.reps.split(",") if r]
    seeds = SEEDS[: args.seeds]
    processed_dir, tag = MARKETS[args.market]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(rep, variant, seed, metric, value, group="all"):
        rows.append({"market": args.market, "rep": rep, "variant": variant,
                     "seed": seed, "metric": metric, "group": group,
                     "value": float(value)})

    for rep in reps:
        cfg0 = RepLabConfig()
        cfg0.processed_dir, cfg0.tag = processed_dir, tag
        data = load_offline_rep(rep, cfg0)
        tr_mask, te_mask = data.split("train"), data.split("test")
        dates_te = data.dates[te_mask]
        regimes_te = _regime_of(dates_te, processed_dir)
        a_beh_te = data.actions[:, te_mask].numpy().mean(axis=0)
        r_te = data.market_returns[te_mask].numpy()

        # regime estimate from h_t (train-only) for the gated model
        regimes_all = _regime_of(data.dates, processed_dir)
        risk = RegimeRisk().fit(data.H, regimes_all, tr_mask)

        # behaviour clone baseline
        add(rep, "behaviour_clone", 0, "sharpe", sharpe_ratio(a_beh_te * r_te, 252))
        for g, v in _group_sharpe(a_beh_te, r_te, regimes_te).items():
            add(rep, "behaviour_clone", 0, "sharpe_regime", v, g)

        # A2C (unconstrained, full deviation) baseline
        a2c_actions = {}
        for seed in seeds:
            cfg = RepLabConfig(); cfg.seed = seed
            head, m = train_offline(rep, "A2C", cfg, data, epochs=cfg.epochs)
            with torch.no_grad():
                a = head.mu(data.H[te_mask]).numpy()
            a2c_actions[seed] = a
            add(rep, "A2C", seed, "sharpe", m["test_sharpe"])
            for g, v in _group_sharpe(a, r_te, regimes_te).items():
                add(rep, "A2C", seed, "sharpe_regime", v, g)

        # gated variants + oracle
        for seed in seeds:
            cfg = RepLabConfig(); cfg.seed = seed
            for rho in (0.0, 1.0):
                _, m = train_gated(rep, cfg, data, risk, rho=rho, epochs=cfg.epochs)
                lam, a = m["test_lambda"], m["test_actions"]
                add(rep, f"gated_rho{int(rho)}", seed, "sharpe", m["test_sharpe"])
                add(rep, f"gated_rho{int(rho)}", seed, "lambda_mean", lam.mean())
                for g in REGIMES:
                    mm = regimes_te == g
                    if mm.sum():
                        add(rep, f"gated_rho{int(rho)}", seed, "lambda_regime", lam[mm].mean(), g)
                        if (a[mm] * r_te[mm]).size >= 2:
                            add(rep, f"gated_rho{int(rho)}", seed, "sharpe_regime",
                                sharpe_ratio((a * r_te)[mm], 252), g)
            # oracle: lambda = 1 in bull else .1, applied to A2C's deviation
            d = a2c_actions[seed] - a_beh_te
            lam_or = np.where(regimes_te == "bull", 1.0, 0.1)
            a_or = np.clip(a_beh_te + lam_or * d, -1, 1)
            add(rep, "oracle", seed, "sharpe", sharpe_ratio(a_or * r_te, 252))
            for g, v in _group_sharpe(a_or, r_te, regimes_te).items():
                add(rep, "oracle", seed, "sharpe_regime", v, g)
        print(f"[done] {rep}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "rep_gated.csv"
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        keys = set(zip(df["market"], df["rep"], df["variant"], df["seed"],
                       df["metric"], df["group"]))
        old = old[~old.apply(lambda r: (r["market"], r["rep"], r["variant"],
                                        r["seed"], r["metric"], r["group"]) in keys,
                             axis=1)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)

    pd.set_option("display.width", 240)
    print("\n=== overall test Sharpe (mean over seeds) ===")
    print(df[df["metric"] == "sharpe"].pivot_table(index="rep", columns="variant",
                                                   values="value").round(3).to_string())
    print("\n=== lambda by regime (gated_rho1, mean over seeds) ===")
    lam = df[(df["metric"] == "lambda_regime") & (df["variant"] == "gated_rho1")]
    print(lam.pivot_table(index="rep", columns="group", values="value").round(3).to_string())
    print("\n=== regime Sharpe: predictive rep ===")
    pr = df[(df["metric"] == "sharpe_regime") & (df["rep"] == "predictive")]
    print(pr.pivot_table(index="group", columns="variant", values="value").round(3).to_string())
    print("\nwrote rep_gated.csv under", OUT_DIR)


if __name__ == "__main__":
    main()