"""Exposure attribution: WHAT information drives the exposure timing?

Exposure-only policies improve risk-adjusted utility (H6, 8.18) without a
directional head. This study identifies WHICH information inside h_t the
exposure decision responds to, separating risk-management skill from
directional skill.

For the test split of each market:
  drivers per date:
    vol20       z_realized_vol_20d        (realized risk)
    risk        P(bear)+P(crisis)          (regime risk, from a train-fit head)
    p_bull      P(bull)                    (regime probability)
    drawdown60  close/rollmax(60)-1        (drawdown state)
    ood         Mahalanobis dist to train h (OOD distance)
  checks:
    corr(e, r_next)        ~ 0 if no direction leakage
    corr(e, |r_next|)      < 0 if the policy de-risks before volatile days
    corr(e, fwd_vol_5)     < 0 if it anticipates volatility
  attribution:
    OLS e ~ vol20 + risk + p_bull + drawdown60 + ood (standardized) -> which
    drivers load on exposure
  perturbation:
    for each regime, Sharpe(e + delta) - Sharpe(e) shows where exposure timing
    has marginal value

usage: python scripts/rep_exposure_attribution.py [--markets spy,nifty]
Output: data/interpret/rep_exposure_attribution.csv + printed summary.
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

from scipy.stats import pearsonr
from src.data.loaders import REPO_ROOT
from src.eval.regime_eval import sharpe_ratio
from src.models.rep_lab.config import RepLabConfig
from src.models.rep_lab.exposure import train_exposure
from src.models.rep_lab.gated import RegimeRisk
from src.models.rep_lab.offline_rl import load_offline_rep

OUT_DIR = ROOT / "data" / "interpret"
SEEDS = [20260814, 111, 222]
MARKETS = {
    "spy": (None, ""),
    "nifty": (REPO_ROOT / "data" / "processed_nifty_backup", "nifty"),
}


def _features(processed_dir, dates: pd.DatetimeIndex) -> pd.DataFrame:
    processed_dir = processed_dir or (REPO_ROOT / "data" / "processed")
    f = pd.read_parquet(processed_dir / "features_regimes.parquet")
    idx = f.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    f.index = idx
    d = dates.tz_localize(None) if getattr(dates, "tz", None) is not None else dates
    out = pd.DataFrame(index=d)
    out["vol20"] = f["z_realized_vol_20d"].reindex(d).to_numpy()
    out["drawdown60"] = (f["close"] / f["close"].rolling(60).max() - 1.0).reindex(d).to_numpy()
    return out


def _ood_mahalanobis(H_train: np.ndarray, H_test: np.ndarray) -> np.ndarray:
    mu = H_train.mean(0)
    cov = np.cov(H_train, rowvar=False) + 1e-6 * np.eye(H_train.shape[1])
    inv = np.linalg.pinv(cov)
    d = H_test - mu
    return np.sqrt((d @ inv * d).sum(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(MARKETS))
    args = ap.parse_args()
    markets = [m for m in args.markets.split(",") if m]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(market, metric, value, driver=None, group="all"):
        rows.append({"market": market, "metric": metric, "driver": driver,
                     "group": group, "value": float(value)})

    for market in markets:
        processed_dir, tag = MARKETS[market]
        cfg0 = RepLabConfig()
        cfg0.processed_dir, cfg0.tag = processed_dir, tag
        data = load_offline_rep("predictive", cfg0)
        te = data.split("test")
        tr = data.split("train")
        r = data.market_returns[te].numpy()
        reg_all = pd.read_parquet((processed_dir or REPO_ROOT / "data" / "processed")
                                  / "features_regimes.parquet")["regime"]
        idx = reg_all.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        reg = pd.Series(reg_all.to_numpy(), index=idx)
        d = data.dates.tz_localize(None) if getattr(data.dates, "tz", None) is not None else data.dates
        regimes = reg.reindex(d).to_numpy()[te]

        regimes_all = reg.reindex(d).to_numpy()
        risk = RegimeRisk().fit(data.H, regimes_all, tr)
        probs = risk.proba(data.H)  # columns in class_ order
        names = list(risk.clf.classes_)
        p_bull = probs[:, names.index("bull")]
        risk_v = risk.risk(data.H)

        feats = _features(processed_dir, data.dates)
        vol20 = feats["vol20"].to_numpy()[te]
        dd60 = feats["drawdown60"].to_numpy()[te]
        ood = _ood_mahalanobis(data.H[tr].numpy(), data.H[te].numpy())
        r_next = np.abs(data.market_returns.numpy())
        fwd_vol = np.full(len(r_next), np.nan)
        for i in range(len(r_next) - 5):
            fwd_vol[i] = np.std(r_next[i + 1:i + 6])

        # mean exposure policy across seeds
        expos = []
        for seed in SEEDS:
            cfg = RepLabConfig(); cfg.seed = seed
            _, m = train_exposure("predictive", cfg, data, risk=risk, rho=0.0)
            expos.append(m["test_exposure"])
        e = np.mean(expos, axis=0)

        drivers = {"vol20": vol20, "risk": risk_v[te], "p_bull": p_bull[te],
                   "drawdown60": dd60, "ood": ood, "abs_r_next": r_next[te],
                   "fwd_vol_5": fwd_vol[te]}
        for name, x in drivers.items():
            ok = np.isfinite(x) & np.isfinite(e)
            if ok.sum() > 2:
                add(market, "corr_e_driver", pearsonr(e[ok], x[ok])[0], driver=name)
        ok = np.isfinite(e) & np.isfinite(data.market_returns[te].numpy())
        add(market, "corr_e_rnext", pearsonr(e[ok], data.market_returns[te].numpy()[ok])[0])

        # OLS attribution: standardized drivers -> e
        import statsmodels.api as sm
        X = pd.DataFrame({d_: drivers[d_][np.isfinite(e) & np.isfinite(drivers[d_])]
                          for d_ in ["vol20", "risk", "p_bull", "drawdown60", "ood"]})
        y = e[np.isfinite(e) & X.notna().all(1).to_numpy()]
        X = X[X.notna().all(1)]
        Xs = (X - X.mean()) / (X.std() + 1e-12)
        Xc = sm.add_constant(Xs)
        res = sm.OLS(y, Xc).fit(cov_type="HC1")
        for term, coef, pv in zip(res.params.index, res.params, res.pvalues):
            add(market, "ols_beta", coef, driver=term)
            add(market, "ols_p", pv, driver=term)

        # perturbation by regime: Sharpe(e +- delta) change
        base = sharpe_ratio(e * r, 252)
        for g in ["bull", "bear", "crisis"]:
            m_g = regimes == g
            if m_g.sum() < 2:
                continue
            for sign in (+1.0, -1.0):
                e2 = e.copy(); e2[m_g] = np.clip(e2[m_g] + sign * 0.1, 0, 1)
                add(market, f"sharpe_delta_{'+' if sign>0 else '-'}{0.1}", 
                    sharpe_ratio(e2 * r, 252) - base, driver=g)
        add(market, "sharpe_base", base)
        print(f"[done] {market}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rep_exposure_attribution.csv", index=False)

    pd.set_option("display.width", 240)
    for market in markets:
        d = df[df["market"] == market]
        print(f"\n=== {market}: corr(exposure, driver) ===")
        c = d[d["metric"] == "corr_e_driver"].pivot_table(index="driver", values="value")
        print(c.round(3).to_string())
        print(f"  corr(e, r_next) = {d[d.metric=='corr_e_rnext'].value.iloc[0]:+.3f}")
        print(f"\n=== {market}: OLS e ~ drivers (std betas) ===")
        b = d[d["metric"] == "ols_beta"].pivot_table(index="driver", values="value")
        p = d[d["metric"] == "ols_p"].pivot_table(index="driver", values="value")
        print(pd.concat([b.rename(columns={"value": "beta"}),
                         p.rename(columns={"value": "p"})], axis=1).round(3).to_string())
        print(f"\n=== {market}: Sharpe change on +-0.1 exposure by regime ===")
        pert = d[d["metric"].str.startswith("sharpe_delta_")]
        print(pert.pivot_table(index="driver", columns="metric", values="value").round(3).to_string())
    print("\nwrote rep_exposure_attribution.csv under", OUT_DIR)


if __name__ == "__main__":
    main()