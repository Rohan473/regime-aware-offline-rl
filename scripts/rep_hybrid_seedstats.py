"""Seed-level inference for the hybrid exposure policy (H7 validation).

Separates two kinds of uncertainty:
  (i)  stochastic TRAINING uncertainty  <- the 10 independent seeds;
  (ii) finite test-PATH uncertainty      <- the date block bootstrap (in
       rep_hybrid_validate.py).

Here we compute, per market, the 10 paired seed differences
    Delta_s = Sharpe_policy,s - Sharpe_BH
for the hybrid and the adaptive exposure, and report the mean, SD, a seed-level
bootstrap CI, a paired t CI, a Wilcoxon signed-rank test, and a sign test.
BH has a single test path, so Delta_s isolates training variability.

Output: data/interpret/rep_hybrid_seedstats.csv + printed summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as tdist, wilcoxon, binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders import REPO_ROOT
from src.eval.regime_eval import sharpe_ratio
from src.models.rep_lab.config import RepLabConfig
from src.models.rep_lab.exposure import train_exposure
from src.models.rep_lab.gated import RegimeRisk
from src.models.rep_lab.hybrid import compose, train_blend
from src.models.rep_lab.offline_rl import load_offline_rep

OUT_DIR = ROOT / "data" / "interpret"
SEEDS = [20260814, 111, 222, 333, 444, 555, 666, 777, 888, 999]
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


def _seed_stats(delta: np.ndarray, n=10000, seed=0) -> dict:
    d = delta[np.isfinite(delta)]
    n_s = len(d)
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(n, n_s), replace=True).mean(axis=1)
    se = d.std(ddof=1) / np.sqrt(n_s)
    tcrit = tdist.ppf(0.975, n_s - 1)
    try:
        w = wilcoxon(d).pvalue
    except Exception:
        w = np.nan
    k = int((d > 0).sum())
    sign_p = binomtest(k, n_s, 0.5, alternative="two-sided").pvalue
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "boot_lo": float(np.percentile(boot, 2.5)),
            "boot_hi": float(np.percentile(boot, 97.5)),
            "t_lo": float(d.mean() - tcrit * se), "t_hi": float(d.mean() + tcrit * se),
            "t_p": float(2 * tdist.sf(abs(d.mean() / se), n_s - 1)) if se > 0 else np.nan,
            "wilcoxon_p": float(w), "n_pos": k, "n": n_s, "sign_p": float(sign_p)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for market in MARKETS:
        processed_dir, tag = MARKETS[market]
        cfg0 = RepLabConfig()
        cfg0.processed_dir, cfg0.tag = processed_dir, tag
        data = load_offline_rep("predictive", cfg0)
        tr, va, te = data.split("train"), data.split("val"), data.split("test")
        r = data.market_returns[te].numpy()
        bh_sharpe = sharpe_ratio(r, 252)
        risk = RegimeRisk().fit(data.H, _regime_of(data.dates, processed_dir), tr)
        F = np.stack([risk.risk(data.H), _ood(data.H[tr].numpy(), data.H.numpy()),
                      _features(processed_dir, data.dates)], axis=1).astype("float64")

        d_hyb, d_ad, hyb_daily, ad_daily = [], [], [], []
        for seed in SEEDS:
            cfg = RepLabConfig(); cfg.seed = seed
            head_e, _ = train_exposure("predictive", cfg, data, risk=None, rho=0.0)
            with torch.no_grad():
                e_full = head_e.e(data.H).numpy()
            e_te = e_full[te]
            sr_ad = e_te * r
            ad_daily.append(sr_ad)
            d_ad.append(sharpe_ratio(sr_ad, 252) - bh_sharpe)
            e_tr = torch.tensor(e_full[tr], dtype=torch.float32)
            e_va = torch.tensor(e_full[va], dtype=torch.float32)
            e_te_t = torch.tensor(e_te, dtype=torch.float32)
            F_tr = torch.tensor(F[tr], dtype=torch.float32)
            F_va = torch.tensor(F[va], dtype=torch.float32)
            F_te = torch.tensor(F[te], dtype=torch.float32)
            r_tr = torch.tensor(data.market_returns[tr], dtype=torch.float32)
            r_va = torch.tensor(data.market_returns[va], dtype=torch.float32)
            ah, _ = train_blend(e_tr, F_tr, r_tr, e_va, F_va, r_va, seed=seed)
            mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-8
            with torch.no_grad():
                al = ah.alpha((F_te - mu) / sd).numpy()
            a_h = compose(e_te_t, torch.tensor(al)).numpy()
            sr_h = a_h * r
            hyb_daily.append(sr_h)
            d_hyb.append(sharpe_ratio(sr_h, 252) - bh_sharpe)
        d_hyb, d_ad = np.array(d_hyb), np.array(d_ad)
        mean_daily = np.stack(hyb_daily).mean(0)
        ad_mean_daily = np.stack(ad_daily).mean(0)
        rows.append({"market": market, "policy": "hybrid_vs_BH", "bh_sharpe": bh_sharpe,
                     "hybrid_mean_sharpe": bh_sharpe + d_hyb.mean(),
                     "ann_return": float(mean_daily.mean() * 252),
                     "ann_vol": float(mean_daily.std() * np.sqrt(252)),
                     **_seed_stats(d_hyb)})
        rows.append({"market": market, "policy": "adaptive_vs_BH", "bh_sharpe": bh_sharpe,
                     "hybrid_mean_sharpe": bh_sharpe + d_ad.mean(),
                     "ann_return": float(ad_mean_daily.mean() * 252),
                     "ann_vol": float(ad_mean_daily.std() * np.sqrt(252)),
                     **_seed_stats(d_ad)})
        print(f"[done] {market}: hybrid Delta = {d_hyb.mean():+.3f} "
              f"[{_seed_stats(d_hyb)['boot_lo']:+.3f}, {_seed_stats(d_hyb)['boot_hi']:+.3f}]")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rep_hybrid_seedstats.csv", index=False)
    pd.set_option("display.width", 240)
    print("\n=== SEED-LEVEL paired differences vs buy-and-hold (10 seeds) ===")
    print(df[["market", "policy", "bh_sharpe", "hybrid_mean_sharpe", "mean", "sd",
              "boot_lo", "boot_hi", "wilcoxon_p", "n_pos", "sign_p"]].round(3).to_string(index=False))
    print("\nwrote rep_hybrid_seedstats.csv under", OUT_DIR)


if __name__ == "__main__":
    main()