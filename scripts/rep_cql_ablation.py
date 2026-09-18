"""CQL implementation ablation: does the BC-regularized actor matter?

CQL(H) with a deterministic actor can saturate to a constant position; adding
a TD3+BC-style actor regularizer (bc_coef * MSE(pi(h), a_logged)) prevents it.
This ablation quantifies the effect so the main-table CQL is reproducible and
the "collapse" claim is explicit.

For each representation x seed x bc_coef in {0.0, 0.5}: train CQL on the frozen
representation and report behavior divergence, action std, the fraction of
seeds whose policy collapsed (action std < 0.01), and test Sharpe.

usage: python scripts/rep_cql_ablation.py [--seeds 10]
Output: data/interpret/rep_cql_ablation.csv + printed table.
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

from src.models.rep_lab.config import OBJECTIVES, RepLabConfig
from src.models.rep_lab.offline_rl import load_offline_rep, train_offline

OUT_DIR = ROOT / "data" / "interpret"
SEEDS = [20260814, 111, 222, 333, 444, 555, 666, 777, 888, 999]
REPS = ["raw", *OBJECTIVES]
BC_COEFS = [0.0, 0.5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    seeds = SEEDS[: args.seeds]

    rows = []
    for rep in REPS:
        data = load_offline_rep(rep, RepLabConfig())
        te = data.split("test")
        beh_mean = data.actions[:, te].numpy().mean(axis=0)
        for bc in BC_COEFS:
            for seed in seeds:
                cfg = RepLabConfig()
                cfg.seed = seed
                cfg.cql_bc_coef = bc
                head, m = train_offline(rep, "CQL", cfg, data, epochs=cfg.epochs)
                with torch.no_grad():
                    a = head.mu(data.H[te]).numpy()
                rows.append(dict(rep=rep, bc_coef=bc, seed=seed,
                                 action_std=float(a.std()),
                                 collapsed=bool(a.std() < 0.01),
                                 divergence=float(np.mean(np.abs(a - beh_mean))),
                                 test_sharpe=m["test_sharpe"]))
            print(f"[done] {rep} bc_coef={bc}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rep_cql_ablation.csv", index=False)

    summ = (df.groupby(["rep", "bc_coef"])
            .agg(divergence=("divergence", "mean"), action_std=("action_std", "mean"),
                 collapse_frac=("collapsed", "mean"), test_sharpe=("test_sharpe", "mean"))
            .round(3))
    pd.set_option("display.width", 200)
    print("\n=== CQL actor-regularization ablation ===")
    print(summ.to_string())
    print("\nwrote rep_cql_ablation.csv under", OUT_DIR)


if __name__ == "__main__":
    main()
