"""Sensitivity of each decision algorithm to REPRESENTATION DIMENSION.

The scaling sweep measured the supervised direction policy. This sweep asks
whether the representation x algorithm interaction depends on latent
dimension: for each objective and hidden dim (4..128) we freeze that encoder
and train BC / A2C / IQL / CQL on the SAME offline transitions, with multiple
head seeds.

usage:
  python scripts/rep_rl_dim_sweep.py [--objectives predictive,contrastive]
                                     [--algos BC,A2C,IQL,CQL] [--seeds 3]

Output: data/interpret/rep_rl_dim_sweep.csv + a printed table.
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
DIM_SIZES = [4, 8, 16, 32, 64, 128]
HEAD_SEEDS = [20260814, 111, 222, 333, 444]


def dim_cfg(dim: int, seed: int = 20260814) -> RepLabConfig:
    cfg = RepLabConfig()
    cfg.seed = seed
    if dim == 128:
        cfg.tag = ""
    else:
        cfg.hidden, cfg.tag = dim, f"h{dim}"
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objectives", default="predictive,contrastive")
    ap.add_argument("--algos", default=",".join(ALGOS))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    objectives = [o for o in args.objectives.split(",") if o]
    algos = [a for a in args.algos.split(",") if a]
    seeds = HEAD_SEEDS[: args.seeds]

    rows = []
    for objective in objectives:
        for dim in DIM_SIZES:
            data = load_offline_rep(objective, dim_cfg(dim))
            for algo in algos:
                for seed in seeds:
                    cfg = RepLabConfig()
                    cfg.seed = seed
                    _, m = train_offline(objective, algo, cfg, data, epochs=cfg.epochs)
                    rows.append({"objective": objective, "dim": dim, "algo": algo,
                                 "seed": seed, "val_sharpe": m["val_sharpe"],
                                 "test_sharpe": m["test_sharpe"],
                                 "test_turnover": m["test_turnover"]})
            print(f"[done] {objective} dim={dim}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rep_rl_dim_sweep.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== test Sharpe mean over head seeds ===")
    for objective in objectives:
        o = df[df["objective"] == objective]
        print(f"\n--- {objective} ---")
        print(o.pivot_table(index="dim", columns="algo", values="test_sharpe",
                            aggfunc="mean").round(3).to_string())
    print("\nwrote rep_rl_dim_sweep.csv under", OUT_DIR)


if __name__ == "__main__":
    main()
