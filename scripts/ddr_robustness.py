"""Robustness evidence for Model B (DDR): seed sweep, hyperparameter grid,
behavior-policy baselines, bear-regime day-by-day P&L, and data-gap report.

Outputs under src/models/ddr/checkpoints/robustness/:
  gaps_report.csv          multi-day holes in the Phase-1 daily frame + invalid days
  seed_sweep.csv           per-seed per-regime test Sharpe / cum return / n_days
  seed_sweep_summary.csv   mean +- std across seeds (PRIMARY robustness table)
  grid.csv                 eta x lr sensitivity (best-val + test regime Sharpe)
  baselines_test.csv       four Phase-1 policies' regime breakdown on the test split
  bear_pnl.csv             test-set bear-regime day-by-day P&L (DDR)

All test metrics exclude data-hole boundary days (not daily returns).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import regime_eval  # noqa: E402
from src.models.ddr.config import DDRConfig  # noqa: E402
from src.models.ddr.data import load_ddr_data, split_ddr_data  # noqa: E402
from src.models.ddr.eval import baseline_policy_table, roll_test_predictions  # noqa: E402
from src.models.ddr.train import train_ddr  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
EPOCHS = 30
GRID = {"eta": [0.01, 0.05], "lr": [1e-3, 3e-4]}


def gaps_report(outdir: Path) -> pd.DataFrame:
    daily = pd.read_parquet(ROOT / "data" / "processed" / "daily_ohlcv.parquet")
    diff = daily.index.to_series().diff().dt.days
    gaps = diff[diff >= 4]
    df = pd.DataFrame(
        {
            "prev_date": gaps.index - pd.to_timedelta(gaps.values, unit="D"),
            "next_date": gaps.index,
            "gap_days": gaps.values,
        }
    )
    df["classification"] = np.where(
        (df["gap_days"] <= 6) & (df["gap_days"] >= 4),
        "holiday/weekend closure (or tiny hole — eyeball)",
        "HOLE: multi-day missing chunk",
    )
    data = load_ddr_data(DDRConfig().window_size)
    invalid_dates = data.dates[~data.valid.numpy()]
    df.to_csv(outdir / "gaps_report.csv", index=False)
    print("=== data gaps in Phase-1 daily frame ===")
    print(df.to_string(index=False))
    print(f"\ninvalid decision days (return spans a hole): {len(invalid_dates)}")
    print("  " + ", ".join(str(d.date()) for d in invalid_dates))
    return df


def run_sweep(cfg: DDRConfig, outdir: Path) -> pd.DataFrame:
    records = []
    data = load_ddr_data(cfg.window_size)
    for seed in SEEDS:
        c = replace(cfg, seed=seed, epochs=EPOCHS, checkpoint_dir=outdir / f"s{seed}")
        _, hist, ckpt = train_ddr(c, data=data)
        preds = roll_test_predictions(c, checkpoint=ckpt, data=data)
        table = regime_eval.evaluate_regime_breakdown(preds[preds["valid"]])
        best_epoch = int(hist["val_sharpe_ann"].idxmax() + 1)
        best_val = float(hist["val_sharpe_ann"].max())
        for regime in table.index:
            records.append(
                {
                    "seed": seed,
                    "best_val_sharpe": round(best_val, 4),
                    "best_val_epoch": best_epoch,
                    "regime": regime,
                    "n_days": int(table.loc[regime, "n_days"]),
                    "sharpe": round(float(table.loc[regime, "sharpe_annualized"]), 4),
                    "cum_return": round(float(table.loc[regime, "cum_return"]), 6),
                }
            )
    df = pd.DataFrame(records)
    df.to_csv(outdir / "seed_sweep.csv", index=False)

    summ = (
        df.groupby("regime")
        .agg(
            n_days=("n_days", "first"),
            sharpe_mean=("sharpe", "mean"),
            sharpe_std=("sharpe", "std"),
            sharpe_min=("sharpe", "min"),
            sharpe_max=("sharpe", "max"),
            cum_mean=("cum_return", "mean"),
            cum_std=("cum_return", "std"),
        )
        .reindex(["bull", "bear", "crisis", "all"])
        .round(4)
    )
    summ.to_csv(outdir / "seed_sweep_summary.csv")
    print("\n=== seed sweep: test Sharpe per regime, mean +- std over "
          f"{len(SEEDS)} seeds ===")
    print(summ.to_string())
    bv = df.groupby("seed")[["best_val_sharpe", "best_val_epoch"]].first()
    print("\n=== best-val selection per seed ===")
    print(bv.to_string())
    return df


def run_grid(cfg: DDRConfig, outdir: Path) -> pd.DataFrame:
    data = load_ddr_data(cfg.window_size)
    rows = []
    for eta, lr in product(GRID["eta"], GRID["lr"]):
        c = replace(
            cfg,
            eta=eta,
            lr=lr,
            epochs=EPOCHS,
            checkpoint_dir=outdir / f"grid_eta{eta}_lr{lr}",
        )
        _, hist, ckpt = train_ddr(c, data=data)
        preds = roll_test_predictions(c, checkpoint=ckpt, data=data)
        table = regime_eval.evaluate_regime_breakdown(preds[preds["valid"]])
        rows.append(
            {
                "eta": eta,
                "lr": lr,
                "best_val_sharpe": round(float(hist["val_sharpe_ann"].max()), 3),
                "best_val_epoch": int(hist["val_sharpe_ann"].idxmax() + 1),
                "test_all_sharpe": round(float(table.loc["all", "sharpe_annualized"]), 3),
                "test_bull": round(float(table.loc["bull", "sharpe_annualized"]), 3),
                "test_bear": round(float(table.loc["bear", "sharpe_annualized"]), 3),
                "test_crisis": round(float(table.loc["crisis", "sharpe_annualized"]), 3),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "grid.csv", index=False)
    print("\n=== eta x lr grid (default: eta=0.01, lr=1e-3, Moody & Saffell / Adam defaults) ===")
    print(df.to_string(index=False))
    return df


def run_baselines(cfg: DDRConfig, outdir: Path) -> pd.DataFrame:
    table = baseline_policy_table(cfg)
    table = table.round(4)
    table.to_csv(outdir / "baselines_test.csv")
    print("\n=== Phase-1 behavior-policy baselines, TEST split (valid days) ===")
    print(table.to_string())
    return table


def bear_dive(cfg: DDRConfig, outdir: Path) -> pd.DataFrame:
    data = load_ddr_data(cfg.window_size)
    preds = roll_test_predictions(cfg, data=data)
    bear = preds[preds["regime"] == "bear"].copy().sort_values("date")
    bear["subpath_cum"] = np.cumprod(1.0 + bear["ret"]) - 1.0
    bear.to_csv(outdir / "bear_pnl.csv", index=False)

    print("\n=== bear-regime day-by-day P&L (DDR, test) ===")
    show = bear[
        ["date", "action", "market_ret", "ret", "subpath_cum", "valid"]
    ].copy()
    show["date"] = show["date"].dt.date
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(show.to_string(index=False))

    valid_bear = bear[bear["valid"]]
    all_rets = valid_bear["ret"].to_numpy()
    worst = bear.loc[bear["ret"].idxmin()]
    stats = {
        "n_days": int(len(bear)),
        "n_valid_days": int(len(valid_bear)),
        "cum_return": float(valid_bear["ret"].sum() if False else np.prod(1 + all_rets) - 1),
        "sharpe_ann": regime_eval.sharpe_ratio(all_rets),
        "sharpe_ann_excl_worst_day": regime_eval.sharpe_ratio(
            np.delete(all_rets, np.argmin(all_rets))
        ),
        "mdd_within_bear": regime_eval.max_drawdown(all_rets),
        "worst_day": str(worst["date"].date()),
        "worst_day_action": round(float(worst["action"]), 4),
        "worst_day_market_ret": round(float(worst["market_ret"]), 4),
        "worst_day_ret": round(float(worst["ret"]), 4),
        "worst_day_valid": bool(worst["valid"]),
        "n_up": int((all_rets > 0).sum()),
        "n_down": int((all_rets < 0).sum()),
        "days_with_|ret|>5pct": int((np.abs(all_rets) > 0.05).sum()),
    }
    print("\n=== bear-regime stats ===")
    for k, v in stats.items():
        print(f"  {k:<28} {v}")
    return bear


def main() -> None:
    parser = argparse.ArgumentParser(description="DDR robustness evidence")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=DDRConfig().seed, help="seed used for grid")
    args = parser.parse_args()

    outdir = DDRConfig().checkpoint_dir / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)
    base = DDRConfig(epochs=args.epochs, seed=args.seed)

    gaps_report(outdir)
    run_sweep(base, outdir)
    run_grid(base, outdir)
    run_baselines(base, outdir)
    bear_dive(base, outdir)
    print(f"\nall artifacts under {outdir}")


if __name__ == "__main__":
    main()