"""Validation of the hybrid exposure blend (H7) - 10 seeds, no new architecture.

  A. 10-seed replication: BH, adaptive, hybrid, floor, blend_50 (mean +/- SD)
     + 5,000-resample bootstrap CIs on the daily-return Sharpe and PAIRED
     hybrid-vs-BH Sharpe difference (identical test dates).
  B. Transaction-cost robustness (0/1/5/10 bp).
  C. Out-of-sample sub-periods (2021-2022 vs 2023-2024).
  D. Ablation of the three alpha inputs: {risk, ood, vol} subsets (7 configs);
     the {ood, vol} subset = the mechanism WITHOUT explicit regime labels.
  E. No-look-ahead permutation control: shuffle the alpha sequence over dates
     (destroying time alignment) - the benefit should vanish if it comes from
     genuine at-t timing.

usage: python scripts/rep_hybrid_validate.py [--markets spy,nifty,csi300]
       [--seeds 10]
Output: data/interpret/rep_hybrid_validate.csv + printed summary.
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
ABLATIONS = {
    "risk": [0], "ood": [1], "vol": [2],
    "risk_ood": [0, 1], "risk_vol": [0, 2], "ood_vol": [1, 2],
    "all": [0, 1, 2],
}
COSTS = [0.0, 1.0, 5.0, 10.0]


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


def _net(a, r, bps):
    return a * r - (bps / 1e4) * np.abs(np.diff(a, prepend=0.0))


def _boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    sh = rng.choice(x, size=(n, len(x)), replace=True)
    m = sh.mean(axis=1) / (sh.std(axis=1) + 1e-12)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(MARKETS))
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    markets = [m for m in args.markets.split(",") if m]
    seeds = SEEDS[: args.seeds]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(market, policy, metric, value, group="all", seed=0):
        rows.append({"market": market, "policy": policy, "seed": seed,
                     "metric": metric, "group": group, "value": float(value)})

    for market in markets:
        processed_dir, tag = MARKETS[market]
        cfg0 = RepLabConfig()
        cfg0.processed_dir, cfg0.tag = processed_dir, tag
        data = load_offline_rep("predictive", cfg0)
        tr, va, te = data.split("train"), data.split("val"), data.split("test")
        r_te = data.market_returns[te].numpy()
        years = data.dates[te].year
        regimes = _regime_of(data.dates, processed_dir)[te]

        risk_all = RegimeRisk().fit(data.H, _regime_of(data.dates, processed_dir), tr)
        risk_v = risk_all.risk(data.H)
        vol_all = _features(processed_dir, data.dates)
        ood_all = _ood(data.H[tr].numpy(), data.H.numpy())
        F = np.stack([risk_v, ood_all, vol_all], axis=1).astype("float64")

        # per-seed daily return arrays for the fixed baselines and the hybrid
        daily = {name: [] for name in ("BH", "adaptive", "floor", "blend50", "hybrid")}
        daily_perm = []
        for seed in seeds:
            cfg = RepLabConfig(); cfg.seed = seed
            head_e, _ = train_exposure("predictive", cfg, data, risk=None, rho=0.0)
            with torch.no_grad():
                e_full = head_e.e(data.H).numpy()
            e_te = e_full[te]

            daily["BH"].append(r_te.copy())
            daily["adaptive"].append(e_te * r_te)
            daily["floor"].append(np.maximum(e_te, 0.5) * r_te)
            daily["blend50"].append((e_te + 0.5 * (1 - e_te)) * r_te)

            # hybrid with the full {risk, ood, vol} alpha (per-seed alpha head)
            e_tr_t = torch.tensor(e_full[tr], dtype=torch.float32)
            e_va_t = torch.tensor(e_full[va], dtype=torch.float32)
            e_te_t = torch.tensor(e_te, dtype=torch.float32)
            F_tr = torch.tensor(F[tr], dtype=torch.float32)
            F_va = torch.tensor(F[va], dtype=torch.float32)
            F_te = torch.tensor(F[te], dtype=torch.float32)
            r_tr_t = torch.tensor(data.market_returns[tr], dtype=torch.float32)
            r_va_t = torch.tensor(data.market_returns[va], dtype=torch.float32)
            alpha_head, _ = train_blend(e_tr_t, F_tr, r_tr_t, e_va_t, F_va, r_va_t,
                                        seed=seed)
            mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-8
            with torch.no_grad():
                alpha_te = alpha_head.alpha((F_te - mu) / sd).numpy()
            a_h = compose(e_te_t, torch.tensor(alpha_te)).numpy()
            daily["hybrid"].append(a_h * r_te)
            # permutation control: shuffle alpha across dates
            shuf = np.random.default_rng(seed).permutation(alpha_te)
            a_s = compose(e_te_t, torch.tensor(shuf)).numpy()
            daily_perm.append(a_s * r_te)

            # ablations: alpha head per input subset (reuse same e)
            for aname, cols in ABLATIONS.items():
                Fs_tr = torch.tensor(F[tr][:, cols], dtype=torch.float32)
                Fs_va = torch.tensor(F[va][:, cols], dtype=torch.float32)
                Fs_te = torch.tensor(F[te][:, cols], dtype=torch.float32)
                ah, _ = train_blend(e_tr_t, Fs_tr, r_tr_t, e_va_t, Fs_va, r_va_t,
                                    seed=seed)
                mus, sds = Fs_tr.mean(0), Fs_tr.std(0) + 1e-8
                with torch.no_grad():
                    als = ah.alpha((Fs_te - mus) / sds).numpy()
                a_ab = compose(e_te_t, torch.tensor(als)).numpy()
                add(market, f"hybrid[{aname}]", "sharpe", sharpe_ratio(a_ab * r_te, 252),
                    seed=seed)

        # ---- A. means + bootstrap CIs + paired difference ----
        for name, arrs in daily.items():
            mats = np.stack(arrs)
            sh = np.array([sharpe_ratio(x, 252) for x in mats])
            lo, hi = _boot_ci(mats.mean(0) if False else mats.sum(0) / mats.shape[1], )
            # bootstrap over the seed-mean daily series
            mean_daily = mats.mean(0)
            blo, bhi = _boot_ci(mean_daily)
            add(market, name, "sharpe", sh.mean(), group="mean")
            add(market, name, "sharpe_sd", sh.std(ddof=1), group="mean")
            add(market, name, "sharpe_ci_lo", blo, group="mean")
            add(market, name, "sharpe_ci_hi", bhi, group="mean")
        # paired hybrid - BH difference bootstrap on the seed-mean daily series
        d_h = np.stack(daily["hybrid"]).mean(0)
        d_b = np.stack(daily["BH"]).mean(0)
        diff = d_h - d_b
        # bootstrap Sharpe difference on paired daily series
        rng = np.random.default_rng(0)
        ds = []
        for _ in range(5000):
            idx = rng.integers(0, len(diff), len(diff))
            ds.append(sharpe_ratio(d_h[idx], 252) - sharpe_ratio(d_b[idx], 252))
        add(market, "hybrid-BH", "paired_delta_sharpe", np.mean(ds), group="mean")
        add(market, "hybrid-BH", "paired_ci_lo", np.percentile(ds, 2.5), group="mean")
        add(market, "hybrid-BH", "paired_ci_hi", np.percentile(ds, 97.5), group="mean")

        # ---- B. cost robustness (recover the mean action from a*r) ----
        for name in ("BH", "adaptive", "hybrid"):
            mats = np.stack(daily[name])
            a_mean = mats.mean(0) / (r_te + 1e-30)  # invert a*r -> a
            for bps in COSTS:
                add(market, name, "sharpe_cost", sharpe_ratio(_net(a_mean, r_te, bps), 252),
                    group=f"cost{bps:g}")

        # ---- C. sub-periods ----
        for name in ("BH", "adaptive", "hybrid"):
            mats = np.stack(daily[name]).mean(0)
            for lo_y, hi_y, label in ((2021, 2022, "2021-22"), (2023, 2024, "2023-24")):
                m = (years >= lo_y) & (years <= hi_y)
                add(market, name, "sharpe_period", sharpe_ratio(mats[m], 252), group=label)

        # ---- E. permutation control ----
        perm_mats = np.stack(daily_perm)
        add(market, "hybrid_permuted", "sharpe",
            sharpe_ratio(perm_mats.mean(0), 252), group="mean")
        print(f"[done] {market}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rep_hybrid_validate.csv", index=False)

    pd.set_option("display.width", 240)
    for market in markets:
        d = df[df["market"] == market]
        print(f"\n=== {market}: 10-seed Sharpe (mean / SD / 95% CI) + paired hybrid-BH ===")
        s = d[(d["group"] == "mean") & d["metric"].isin(
            ["sharpe", "sharpe_sd", "sharpe_ci_lo", "sharpe_ci_hi",
             "paired_delta_sharpe", "paired_ci_lo", "paired_ci_hi"])]
        print(s.pivot_table(index="metric", columns="policy", values="value").round(3).to_string())
        print(f"\n--- {market}: cost robustness (Sharpe at bp) ---")
        c = d[d["metric"] == "sharpe_cost"]
        print(c.pivot_table(index="policy", columns="group", values="value").round(3).to_string())
        print(f"\n--- {market}: sub-periods ---")
        p = d[d["metric"] == "sharpe_period"]
        print(p.pivot_table(index="policy", columns="group", values="value").round(3).to_string())
        print(f"\n--- {market}: ablations (hybrid alpha inputs) + permutation ---")
        ab = d[(d["metric"] == "sharpe") & (d["group"] == "mean") &
               d["policy"].str.startswith("hybrid")]
        print(ab.pivot_table(index="policy", values="value").round(3).to_string())
    print("\nwrote rep_hybrid_validate.csv under", OUT_DIR)


if __name__ == "__main__":
    main()