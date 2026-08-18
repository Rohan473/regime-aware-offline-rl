"""Paired-significance + residual hole-contamination audit for Model B.

Answers the reviewer's three questions with evidence:
  A. Paired test DDR vs buy-and-hold, per seed, per day, per regime (bull/bear).
  B. Residual hole contamination: how many test days have (i) a 20d momentum
     window, or (ii) a 60d regime-label window, spanning one of the six data
     holes -- and what happens to the numbers on the clean-label subset.
  C. Momentum-baseline coherence: sign consistency (by construction), where
     its crisis +1.47 comes from, and how many of its actions are computed
     on hole-spanning windows.

Outputs under src/models/ddr/checkpoints/robustness/:
  paired_ddr_vs_bh.csv / paired_summary.csv
  contamination_audit.csv, clean_label_subset.csv
  momentum_audit.csv
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import regime_eval  # noqa: E402
from src.models.ddr.config import DDRConfig  # noqa: E402
from src.models.ddr.data import load_ddr_data, split_ddr_data  # noqa: E402
from src.models.ddr.eval import roll_test_predictions  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
ROBUST_DIR = DDRConfig().checkpoint_dir / "robustness"


def paired_test(cfg: DDRConfig, data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DDR minus B&H on the SAME days, per seed, per regime."""
    daily = []
    for seed in SEEDS:
        c = replace(cfg, seed=seed, checkpoint_dir=ROBUST_DIR / f"s{seed}")
        preds = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "ddr_best.pt", data=data)
        p = preds[preds["valid"]].copy()
        p["seed"] = seed
        p["ddr_minus_bh"] = p["ret"] - p["market_ret"]
        daily.append(p[["seed", "date", "regime", "ret", "market_ret", "ddr_minus_bh"]])
    df = pd.concat(daily)

    rows = []
    for regime in ("bull", "bear"):
        sub = df[df["regime"] == regime]
        per_day = sub.groupby("date")["ddr_minus_bh"].mean()  # mean over seeds, per day
        t, p = stats.ttest_1samp(per_day.to_numpy(), 0.0)
        per_seed = sub.groupby("seed")["ddr_minus_bh"].mean()
        rows.append(
            {
                "regime": regime,
                "n_days": int(len(per_day)),
                "ddr_mean_daily": float(sub["ret"].mean()),
                "bh_mean_daily": float(sub["market_ret"].mean()),
                "mean_diff_daily": float(per_day.mean()),
                "t_stat": float(t),
                "p_value": float(p),
                "n_seeds_ddr_gt_bh": int((per_seed > 0).sum()),
                "per_seed_diffs": "; ".join(f"{s}:{v:+.5f}" for s, v in per_seed.items()),
            }
        )
    summary = pd.DataFrame(rows)
    per_seed_table = (
        df.groupby(["seed", "regime"])[["ret", "market_ret"]]
        .mean()
        .rename(columns={"ret": "ddr_mean_daily", "market_ret": "bh_mean_daily"})
        .round(6)
    )
    summary.to_csv(ROBUST_DIR / "paired_summary.csv", index=False)
    per_seed_table.to_csv(ROBUST_DIR / "paired_ddr_vs_bh.csv")
    print("=== A. paired test: DDR minus buy-and-hold, same days ===")
    print(summary.to_string(index=False))
    print("\nper-seed mean daily returns:")
    print(per_seed_table.to_string())
    return summary, per_seed_table


def hole_spanning_flags(features: pd.DataFrame, max_cal_days: int = 10) -> pd.DataFrame:
    """For each row: does the trailing 20d (momentum/features) or 60d (label)
    window contain a data hole? A window spans a hole if the calendar gap
    from the row to the window-start row is far larger than the number of
    trading days suggests.

    Legit closure: W trading days ~= 1.4*W calendar days, plus holiday-heavy
    stretches push that to ~1.5*W; holes add 33+ days over the trading-day
    count. Threshold: 1.6*W calendar days + 8 slack.
    """
    idx = features.index
    out = pd.DataFrame(index=idx)
    for back, name in ((19, "win20d_spans_hole"), (59, "win60d_spans_hole")):
        cal = (idx[back:] - idx[:-back]).days
        flag = pd.Series(False, index=idx)
        flag.iloc[back:] = cal.to_numpy() > 1.6 * (back + 1) + 8
        out[name] = flag
    return out


def contamination_audit(cfg: DDRConfig, data) -> pd.DataFrame:
    features = pd.read_parquet(ROOT / "data" / "processed" / "features_regimes.parquet")
    flags = hole_spanning_flags(features)
    test = split_ddr_data(data)["test"]
    valid_mask = test.valid.numpy()
    valid_dates = test.dates[valid_mask]

    audit = flags.loc[valid_dates].copy()
    audit["regime"] = test.regimes.to_numpy()[valid_mask]
    audit.to_csv(ROBUST_DIR / "contamination_audit.csv")

    print("\n=== B. residual hole contamination (TEST split, valid days) ===")
    tab = (
        audit.groupby("regime")[["win20d_spans_hole", "win60d_spans_hole"]]
        .sum()
        .reindex(["bull", "bear"])
    )
    tab["n_days"] = audit.groupby("regime").size().reindex(["bull", "bear"])
    print(tab.to_string())
    print("\ntest days whose 60d REGIME-LABEL window spans a hole:")
    flagged = audit[audit["win60d_spans_hole"]].copy()
    if len(flagged):
        flagged["date"] = flagged.index.date
        print(flagged[["date", "regime", "win20d_spans_hole"]].to_string(index=False))
    else:
        print("(none)")

    clean = audit[~audit["win60d_spans_hole"]]
    sub = pd.DataFrame(
        {
            "regime": ["bull", "bear", "all"],
            "n_days": [int((clean["regime"] == "bull").sum()), int((clean["regime"] == "bear").sum()), len(clean)],
        }
    )
    sub.to_csv(ROBUST_DIR / "clean_label_subset.csv", index=False)
    print("\nclean-label subset size (excludes days whose label window spans a hole):")
    print(sub.to_string(index=False))
    return audit


def momentum_audit(cfg: DDRConfig) -> pd.DataFrame:
    ds = pd.read_parquet(ROOT / "data" / "processed" / "offline_dataset.parquet")
    mom = ds[ds["policy"] == "momentum"].set_index("date").sort_index()
    features = pd.read_parquet(ROOT / "data" / "processed" / "features_regimes.parquet")
    test = split_ddr_data(load_ddr_data(cfg.window_size))["test"]
    valid_mask = test.valid.numpy()
    valid_dates = test.dates[valid_mask]
    mom = mom.loc[mom.index.intersection(valid_dates)]

    flags = hole_spanning_flags(features).loc[mom.index]
    ret20 = features["ret_20d"].reindex(mom.index)
    next_ret = features["close"].pct_change().shift(-1).reindex(mom.index)
    realized_ret = mom["action"].to_numpy() * next_ret.to_numpy()

    rows = []
    for regime in ("bull", "bear", "crisis", "all"):
        mask = np.ones(len(mom), bool) if regime == "all" else (mom["regime"] == regime).to_numpy()
        a = mom["action"].to_numpy()[mask]
        r = realized_ret[mask]
        rows.append(
            {
                "regime": regime,
                "n_days": int(mask.sum()),
                "sign_agree_pct": round(100 * (np.sign(a) == np.sign(ret20.to_numpy()[mask])).mean(), 1),
                "mean_daily_ret": round(float(r.mean()), 6),
                "cum_return": round(float(np.prod(1 + r) - 1), 4),
                "action_uses_hole_window_pct": round(100 * flags["win20d_spans_hole"].to_numpy()[mask].mean(), 1),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(ROBUST_DIR / "momentum_audit.csv", index=False)
    print("\n=== C. momentum baseline coherence ===")
    print(df.to_string(index=False))
    aug = mom[(mom.index >= "2022-08-01") & (mom.index <= "2022-08-31")]
    aug_ret = aug["action"].to_numpy() * next_ret.reindex(aug.index).to_numpy()
    print(f"\nmomentum in Aug-2022 (the fake crisis run): n={len(aug)}, "
          f"mean action={aug['action'].mean():.3f} (short = trend-down logic), "
          f"cum ret={np.prod(1 + aug_ret) - 1:.4f}")
    return df


def bh_drawdown(cfg: DDRConfig, data) -> None:
    test = split_ddr_data(data)["test"]
    valid = test.valid.numpy() if test.valid is not None else np.ones(len(test.dates), bool)
    rets = test.next_returns.numpy()[valid]
    dates = test.dates[valid]
    print("\n=== buy-and-hold reference on TEST (valid days) ===")
    print(f"max drawdown: {regime_eval.max_drawdown(rets):.4f}")
    print(f"cum return:   {np.prod(1 + rets) - 1:.4f}")
    print(f"sharpe ann:   {regime_eval.sharpe_ratio(rets):.4f}")
    for r in ("bull", "bear"):
        mask = test.regimes[valid] == r
        rr = rets[mask]
        print(f"  {r}: max dd {regime_eval.max_drawdown(rr):.4f}, cum {np.prod(1 + rr) - 1:.4f}, n {mask.sum()}")


def main() -> None:
    cfg = DDRConfig()
    data = load_ddr_data(cfg.window_size)
    paired_test(cfg, data)
    contamination_audit(cfg, data)
    momentum_audit(cfg)
    bh_drawdown(cfg, data)


if __name__ == "__main__":
    main()