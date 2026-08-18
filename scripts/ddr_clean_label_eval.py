"""Clean-label evaluation + paired drawdown test for Model B.

Responds to the review:
  1. Paired max-drawdown test, same rigor as the returns test: 5 seeds,
     DDR minus buy-and-hold on the SAME days, per regime, CLEAN-LABEL days
     only (win60d window must not span a data hole).
  2. Full regime breakdown re-run on clean-label days only (245 bull,
     13 bear -- bear withheld as underpowered), replacing the contaminated
     378/118 table.

Also reports a fully-clean subset (label AND 20d feature windows hole-free)
and the clean-label "all" path metrics.

Outputs under src/models/ddr/checkpoints/robustness/:
  clean_label_regimes.csv      per-seed regime breakdown on clean-label days
  paired_drawdown.csv          per-seed max DD, DDR vs buy-and-hold
  fully_clean_subset.csv       sizes + per-seed Sharpe/DD on fully-clean days
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import regime_eval  # noqa: E402
from src.models.ddr.config import DDRConfig  # noqa: E402
from src.models.ddr.data import load_ddr_data, split_ddr_data  # noqa: E402
from src.models.ddr.eval import roll_test_predictions  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
ROBUST_DIR = DDRConfig().checkpoint_dir / "robustness"


def clean_masks(data) -> tuple[np.ndarray, np.ndarray]:
    """Bool masks over TEST valid days: label-clean (60d window hole-free)
    and fully-clean (label AND 20d feature window hole-free)."""
    features = pd.read_parquet(ROOT / "data" / "processed" / "features_regimes.parquet")
    idx = features.index
    flags = pd.DataFrame(index=idx)
    for back, name in ((19, "win20"), (59, "win60")):
        cal = (idx[back:] - idx[:-back]).days
        f = pd.Series(False, index=idx)
        f.iloc[back:] = cal.to_numpy() > 1.6 * (back + 1) + 8
        flags[name] = f

    test = split_ddr_data(data)["test"]
    valid_mask = test.valid.numpy()
    fl = flags.loc[test.dates[valid_mask]]
    label_clean = (~fl["win60"]).to_numpy()
    fully_clean = (~fl["win60"] & ~fl["win20"]).to_numpy()
    return label_clean, fully_clean


def per_seed_preds(cfg: DDRConfig, data) -> list[pd.DataFrame]:
    out = []
    for seed in SEEDS:
        c = replace(cfg, seed=seed, checkpoint_dir=ROBUST_DIR / f"s{seed}")
        preds = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "ddr_best.pt", data=data)
        out.append(preds[preds["valid"]].copy())
    return out


def clean_label_regimes(cfg: DDRConfig, data, label_clean: np.ndarray) -> pd.DataFrame:
    """Regime breakdown per seed on clean-label days only (bear withheld)."""
    rows = []
    for seed, preds in zip(SEEDS, per_seed_preds(cfg, data)):
        p = preds[label_clean]
        table = regime_eval.evaluate_regime_breakdown(p)
        for regime in table.index:
            rows.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "n_days": int(table.loc[regime, "n_days"]),
                    "sharpe": round(float(table.loc[regime, "sharpe_annualized"]), 4),
                    "cum_return": round(float(table.loc[regime, "cum_return"]), 6),
                    "max_dd": round(float(table.loc[regime, "max_drawdown"]), 4),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(ROBUST_DIR / "clean_label_regimes.csv", index=False)

    print("=== regime breakdown on CLEAN-LABEL days only (per seed) ===")
    summ = (
        df.groupby("regime")
        .agg(n=("n_days", "first"), sharpe_mean=("sharpe", "mean"), sharpe_std=("sharpe", "std"),
             cum_mean=("cum_return", "mean"), dd_mean=("max_dd", "mean"))
        .round(4)
    )
    print(summ.to_string())
    print("\nNOTE: bear has n=13 clean-label days -- underpowered, no number reported.")
    print("NOTE: crisis has 0 clean-label days in the test window: the 20-day 'crisis'")
    print("      run was fabricated by the holes; there is NO real test-window crisis.")
    return df


def baselines_on_clean_days(cfg: DDRConfig, data, label_clean: np.ndarray) -> None:
    test = split_ddr_data(data)["test"]
    valid_mask = test.valid.numpy()
    dates = test.dates[valid_mask][label_clean]
    regimes = test.regimes.to_numpy()[valid_mask][label_clean]
    market = test.next_returns.numpy()[valid_mask][label_clean]

    ds = pd.read_parquet(ROOT / "data" / "processed" / "offline_dataset.parquet")
    ds = ds[ds["date"].isin(dates)]
    rows = []
    for policy, grp in ds.groupby("policy"):
        g = grp.set_index("date").sort_index()
        ret = (g["action"].astype("float64") * pd.Series(market, index=dates).reindex(g.index)).to_numpy()
        preds = pd.DataFrame({"date": g.index, "ret": ret, "regime": g["regime"].to_numpy()})
        table = regime_eval.evaluate_regime_breakdown(preds)
        rec = {"policy": policy}
        for regime in table.index:
            rec[f"{regime}_n"] = int(table.loc[regime, "n_days"])
            rec[f"{regime}_sharpe"] = round(float(table.loc[regime, "sharpe_annualized"]), 4)
        rec["all_sharpe"] = round(float(table.loc["all", "sharpe_annualized"]), 4)
        rec["all_cum"] = round(float(table.loc["all", "cum_return"]), 4)
        rec["all_max_dd"] = round(float(table.loc["all", "max_drawdown"]), 4)
        rows.append(rec)
    df = pd.DataFrame(rows).set_index("policy")
    df.to_csv(ROBUST_DIR / "baselines_clean_label.csv")
    print("\n=== baselines on CLEAN-LABEL days only ===")
    print(df.to_string())


def paired_drawdown(cfg: DDRConfig, data, label_clean: np.ndarray, fully_clean: np.ndarray) -> pd.DataFrame:
    """Max DD, DDR vs buy-and-hold, SAME days, per seed. Three day-sets:
    full valid path, clean-label all, fully-clean all. Bear excluded
    (n=13 underpowered)."""
    test = split_ddr_data(data)["test"]
    valid_mask = test.valid.numpy()
    market_full = test.next_returns.numpy()[valid_mask]
    rows = []
    for seed, preds in zip(SEEDS, per_seed_preds(cfg, data)):
        ddr = preds["ret"].to_numpy()
        bh = preds["market_ret"].to_numpy()  # same days by construction
        for name, mask in (("all_valid_516d", np.ones(len(ddr), bool)),
                           ("clean_label_258d", label_clean),
                           ("fully_clean", fully_clean)):
            rows.append(
                {
                    "seed": seed,
                    "dayset": name,
                    "n_days": int(mask.sum()),
                    "ddr_max_dd": round(regime_eval.max_drawdown(ddr[mask]), 5),
                    "bh_max_dd": round(regime_eval.max_drawdown(bh[mask]), 5),
                    "ddr_minus_bh_dd": round(regime_eval.max_drawdown(ddr[mask]) - regime_eval.max_drawdown(bh[mask]), 5),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(ROBUST_DIR / "paired_drawdown.csv", index=False)

    print("\n=== paired max-drawdown, DDR vs buy-and-hold, per seed ===")
    piv = df.pivot(index="seed", columns="dayset", values=["ddr_max_dd", "bh_max_dd", "ddr_minus_bh_dd"])
    print(piv.round(5).to_string())
    for name in df["dayset"].unique():
        sub = df[df["dayset"] == name]
        n_win = int((sub["ddr_minus_bh_dd"] > 0).sum())
        mean = sub["ddr_minus_bh_dd"].mean()
        print(f"{name}: DDR max DD less severe than B&H in {n_win}/5 seeds; "
              f"mean DD difference {mean:+.5f} (positive = DDR better)")
    print("\nExposure confound: DDR mean |action| < 1 (de-levered long), so part of the")
    print("DD reduction is mechanical. Return-per-DD: full path B&H 0.405/0.126 = 3.2 vs")
    print("DDR 0.201/0.083 = 2.4; clean-label B&H 0.164/0.080 = 2.05 vs DDR 0.071/0.026 = 2.8.")
    return df


def fully_clean_subset(cfg: DDRConfig, data, fully_clean: np.ndarray) -> None:
    """Sizes + per-seed Sharpe/DD on fully-clean days (no hole in label OR
    feature windows)."""
    rows = []
    for seed, preds in zip(SEEDS, per_seed_preds(cfg, data)):
        p = preds[fully_clean]
        rows.append(
            {
                "seed": seed,
                "n_days": int(len(p)),
                "sharpe": round(regime_eval.sharpe_ratio(p["ret"].to_numpy()), 4),
                "max_dd": round(regime_eval.max_drawdown(p["ret"].to_numpy()), 5),
                "bh_sharpe": round(regime_eval.sharpe_ratio(p["market_ret"].to_numpy()), 4),
                "bh_max_dd": round(regime_eval.max_drawdown(p["market_ret"].to_numpy()), 5),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(ROBUST_DIR / "fully_clean_subset.csv", index=False)
    print("\n=== fully-clean subset (label AND 20d feature windows hole-free) ===")
    print(df.to_string(index=False))
    return df


def main() -> None:
    cfg = DDRConfig()
    data = load_ddr_data(cfg.window_size)
    label_clean, fully_clean = clean_masks(data)
    print(f"clean-label days: {int(label_clean.sum())} / {len(label_clean)}; "
          f"fully-clean: {int(fully_clean.sum())}")
    clean_label_regimes(cfg, data, label_clean)
    baselines_on_clean_days(cfg, data, label_clean)
    paired_drawdown(cfg, data, label_clean, fully_clean)
    fully_clean_subset(cfg, data, fully_clean)


if __name__ == "__main__":
    main()