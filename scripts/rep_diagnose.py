"""Why is learned utility below buy-and-hold? - diagnostic decomposition.

NO tuning, NO new model design. Runs the same battery for SPY, CSI300, NIFTY:

  A. Market vs policy: buy-and-hold, behavior(mean), constant-mean, random,
     A2C, BC, IQL, CQL (+ regime-gated and a regime oracle for reference).
     Delta_RL,BH = Sharpe_RL - Sharpe_BH, decomposed by regime.
  B. Where the return disappears: R_policy = direction + timing - turnover
     - cost, via directional accuracy, exposure, avg position (up/down),
     turnover, max drawdown, upside capture, downside loss, cost robustness.
  C. CSI300 oracle question: BH -> behavior -> oracle -> learned; the oracle
     gap Sharpe_oracle - Sharpe_RL tells whether the period is hostile or the
     decision learner is the bottleneck.

The learned policies are the PRE-SPECIFIED heads (predictive rep, seed 20260814
+ 2 more for spread); they are re-run only to recover their actions, not tuned.

Output: data/interpret/rep_diagnose.csv (long form) + printed summary.
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
from src.models.rep_lab.gated import RegimeRisk, compose, train_gated
from src.models.rep_lab.offline_rl import load_offline_rep, train_offline

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


def diagnostics(a: np.ndarray, r: np.ndarray, regimes: np.ndarray,
                cost_bps: float = 0.0) -> dict:
    a, r = a[np.isfinite(a) & np.isfinite(r)], r[np.isfinite(a) & np.isfinite(r)]
    sr = a * r - (cost_bps / 1e4) * np.abs(np.diff(a, prepend=0.0))
    up, dn = r > 0, r < 0
    out = {
        "sharpe": float(sharpe_ratio(sr, 252)),
        "return": float(np.prod(1 + sr) - 1) if sr.size else np.nan,
        "maxdd": float(max_drawdown(sr)),
        "turnover": float(np.abs(np.diff(a, prepend=0.0)).mean()),
        "exposure": float(np.abs(a).mean()),
        "avg_pos": float(a.mean()),
        "avg_pos_up": float(a[up].mean()) if up.sum() else np.nan,
        "avg_pos_down": float(a[dn].mean()) if dn.sum() else np.nan,
        "dir_acc": float((np.sign(a) == np.sign(r)).mean()),
        "upside_capture": float((a * r)[up].mean() / r[up].mean()) if up.sum() else np.nan,
        "downside_loss": float((a * r)[dn].mean() / r[dn].mean()) if dn.sum() else np.nan,
    }
    for g in REGIMES:
        m = regimes == g
        out[f"sharpe_{g}"] = float(sharpe_ratio((a * r)[m], 252)) if m.sum() >= 2 else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(MARKETS))
    ap.add_argument("--rep", default="predictive")
    args = ap.parse_args()
    markets = [m for m in args.markets.split(",") if m]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(market, policy, metric, value):
        rows.append({"market": market, "policy": policy, "metric": metric,
                     "value": float(value) if np.isfinite(value) else np.nan})

    for market in markets:
        processed_dir, tag = MARKETS[market]
        cfg0 = RepLabConfig()
        cfg0.processed_dir, cfg0.tag = processed_dir, tag
        data = load_offline_rep(args.rep, cfg0)
        te = data.split("test")
        r = data.market_returns[te].numpy()
        regimes = _regime_of(data.dates, processed_dir)[te]
        a_beh = data.actions[:, te].numpy().mean(axis=0)
        all_actions = data.actions.numpy().ravel()

        # ---- A. baselines (no training) ----
        bh = np.ones_like(r)
        rand = np.random.default_rng(0).uniform(-1, 1, len(r))
        policies_base = {
            "buy_and_hold": bh,
            "behavior_mean": a_beh,
            "constant_mean": np.full_like(r, all_actions.mean()),
            "random": rand,
        }
        for name, a in policies_base.items():
            for m, v in diagnostics(a, r, regimes).items():
                add(market, name, m, v)

        # ---- learned policies (pre-specified heads, re-run to recover actions)
        learned = {a: [] for a in ("A2C", "BC", "IQL", "CQL")}
        for seed in SEEDS:
            cfg = RepLabConfig(); cfg.seed = seed
            head, _ = train_offline(args.rep, "A2C", cfg, data, epochs=cfg.epochs)
            with torch.no_grad():
                learned["A2C"].append(head.mu(data.H[te]).numpy())
            for algo in ("BC", "IQL", "CQL"):
                head, _ = train_offline(args.rep, algo, cfg, data, epochs=cfg.epochs)
                with torch.no_grad():
                    learned[algo].append(head.mu(data.H[te]).numpy())
        # regime-gated (proposed model)
        regimes_all = _regime_of(data.dates, processed_dir)
        risk = RegimeRisk().fit(data.H, regimes_all, data.split("train"))
        gated = []
        for seed in SEEDS:
            cfg = RepLabConfig(); cfg.seed = seed
            _, m = train_gated(args.rep, cfg, data, risk, rho=1.0, epochs=cfg.epochs)
            gated.append(m["test_actions"])
        learned["gated"] = gated
        # oracle: regime-aware gate on A2C's deviation (look-ahead, upper bound)
        a2c_seed0 = learned["A2C"][0]
        lam_or = np.where(regimes == "bull", 1.0, 0.1)
        oracle = np.clip(a_beh + lam_or * (a2c_seed0 - a_beh), -1, 1)

        for name, acts in learned.items():
            mean_d = diagnostics(np.mean(np.stack(acts), axis=0), r, regimes)
            for m, v in mean_d.items():
                add(market, name, m, v)
        for m, v in diagnostics(oracle, r, regimes).items():
            add(market, "oracle", m, v)

        # ---- C. delta vs buy-and-hold (overall + per regime) ----
        bh_s = diagnostics(bh, r, regimes)
        bh_all, bh_reg = bh_s["sharpe"], {g: bh_s[f"sharpe_{g}"] for g in REGIMES}
        for name in [*policies_base, *learned, "oracle"]:
            d_all = diagnostics(policies_base[name] if name in policies_base
                                else (oracle if name == "oracle"
                                      else np.mean(np.stack(learned[name]), axis=0)),
                                r, regimes)
            add(market, name, "delta_BH", d_all["sharpe"] - bh_all)
            for g in REGIMES:
                add(market, name, f"delta_BH_{g}", d_all[f"sharpe_{g}"] - bh_reg[g])
        print(f"[done] {market}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "rep_diagnose.csv"
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        old = old[~old["market"].isin(markets)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)

    pd.set_option("display.width", 240)
    for market in markets:
        d = df[df["market"] == market]
        print(f"\n=== {market}: Sharpe / return / turnover / dir_acc / upside / downside / delta_BH ===")
        p = d.pivot_table(index="policy", columns="metric", values="value")
        cols = [c for c in ["sharpe", "return", "maxdd", "turnover", "exposure",
                            "dir_acc", "upside_capture", "downside_loss", "delta_BH"]
                if c in p.columns]
        print(p[cols].round(3).to_string())
        print(f"\n--- {market}: regime Sharpe and delta vs buy-and-hold ---")
        reg = d[d["metric"].isin([f"sharpe_{g}" for g in REGIMES] +
                                 [f"delta_BH_{g}" for g in REGIMES])]
        print(reg.pivot_table(index="policy", columns="metric", values="value").round(3).to_string())
    print("\nwrote rep_diagnose.csv under", OUT_DIR)


if __name__ == "__main__":
    main()