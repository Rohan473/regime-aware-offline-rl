"""Does representation quality predict trading utility? (the dissociation test)

Reads the scaling sweep (rep_scaling_summary.csv) and the rep x algorithm
matrix (rep_offline_rl_summary.csv) and asks the paper's central question:

    do information / compression / stability axes actually predict the
    downstream supervised or RL Sharpe?

Computes the cross-config correlation of each quality axis with the
supervised direction Sharpe (10-seed means) and writes it to
data/interpret/rep_quality_utility_corr.csv, then prints the headline tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "interpret"
AXES = ["dir1_auc_mean", "mag_r2_mean", "vol20_r2_mean", "regime_bacc_mean",
        "effective_rank_mean", "cross_seed_cka"]


def main() -> None:
    s = pd.read_csv(OUT_DIR / "rep_scaling_summary.csv")
    y = s["supervised_sharpe_mean"].to_numpy()

    rows = []
    for m in AXES:
        x = s[m].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        rows.append(dict(axis=m.replace("_mean", ""), n=int(ok.sum()),
                         pearson=float(pearsonr(x[ok], y[ok])[0]),
                         spearman=float(spearmanr(x[ok], y[ok])[0])))
    corr = pd.DataFrame(rows)
    corr.to_csv(OUT_DIR / "rep_quality_utility_corr.csv", index=False)

    pd.set_option("display.width", 240)
    print("=== quality axis vs supervised Sharpe (36 scaling cells, 10 seeds) ===")
    print(corr.round(3).to_string(index=False))

    print("\n=== feature scaling: supervised Sharpe (mean over 10 seeds) ===")
    f = s[s["axis"] == "feature"]
    print(f.pivot_table(index="size", columns="objective",
                        values="supervised_sharpe_mean").round(3).to_string())
    print("\n=== feature scaling: dir1 AUC (near-chance everywhere) ===")
    print(f.pivot_table(index="size", columns="objective",
                        values="dir1_auc_mean").round(3).to_string())

    print("\n=== latent-dim scaling: supervised Sharpe ===")
    d = s[s["axis"] == "dim"]
    print(d.pivot_table(index="size", columns="objective",
                        values="supervised_sharpe_mean").round(3).to_string())

    o = pd.read_csv(OUT_DIR / "rep_offline_rl_summary.csv")
    print("\n=== representation x algorithm: test Sharpe (mean over 10 seeds) ===")
    print(o.pivot_table(index="rep", columns="algo",
                        values="test_sharpe_mean").round(3).to_string())
    print("\n=== representation x algorithm: test Sharpe (std over 10 seeds) ===")
    print(o.pivot_table(index="rep", columns="algo",
                        values="test_sharpe_std").round(3).to_string())
    print("\nwrote rep_quality_utility_corr.csv under", OUT_DIR)


if __name__ == "__main__":
    main()
