"""Representation x {A2C, IQL}: does the decision layer matter?

For each frozen representation (raw window / replab auto / predictive /
contrastive) train two offline RL heads on the SAME Phase-1 transition
dataset and evaluate them identically (test-split Sharpe of the deterministic
policy on realized market returns). Multiple head seeds per cell give error
bars. This is the "representation quality != trading performance" stress test.

usage:
  python scripts/rep_offline_rl.py [--reps raw,auto,predictive,contrastive]
                                   [--algos A2C,IQL] [--seeds 10] [--epochs 30]

Outputs (data/interpret/):
  rep_offline_rl.csv          per (rep, algo, seed) val/test metrics
  rep_offline_rl_summary.csv  mean/std per (rep, algo)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.rep_lab.config import OBJECTIVES, RepLabConfig
from src.models.rep_lab.offline_rl import ALGOS, load_offline_rep, train_offline

OUT_DIR = ROOT / "data" / "interpret"
SEEDS = [20260814, 111, 222, 333, 444, 555, 666, 777, 888, 999]
REPS = ["raw", *OBJECTIVES]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default=",".join(REPS))
    ap.add_argument("--algos", default=",".join(ALGOS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reps = [r for r in args.reps.split(",") if r]
    algos = [a for a in args.algos.split(",") if a]
    seeds = SEEDS[: args.seeds]

    rows = []
    for rep in reps:
        cfg0 = RepLabConfig()
        data = load_offline_rep(rep, cfg0)
        print(f"[data] {rep}: H={tuple(data.H.shape)} actions={tuple(data.actions.shape)} "
              f"dates={len(data.dates)}")
        for algo in algos:
            for seed in seeds:
                cfg = RepLabConfig()
                cfg.seed = seed
                _, m = train_offline(rep, algo, cfg, data, epochs=args.epochs)
                m["rep"] = rep
                rows.append(m)
                print(f"  [{rep}/{algo}] seed {seed}: val {m['val_sharpe']:.3f} "
                      f"test {m['test_sharpe']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rep_offline_rl.csv", index=False)
    summary = (df.groupby(["rep", "algo"])[["val_sharpe", "test_sharpe", "test_return",
                                            "test_maxdd", "test_turnover"]]
               .agg(["mean", "std"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT_DIR / "rep_offline_rl_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== representation x algorithm (test Sharpe mean over seeds) ===")
    print(df.pivot_table(index="rep", columns="algo", values="test_sharpe",
                         aggfunc="mean").round(3).to_string())
    print("\nwrote rep_offline_rl.csv / rep_offline_rl_summary.csv under", OUT_DIR)


if __name__ == "__main__":
    main()
