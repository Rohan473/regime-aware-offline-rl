"""Basin-frequency probe for the naive DSR (Model B) on CURRENT data.

Seed 4 of the canonical naive_new 5-seed run converged to a qualitatively
different policy basin (late-spiking best-val epoch 30 vs 3-5 for the other
seeds; cross-seed test-prediction correlation 0.33 vs 0.93-0.98; test all
Sharpe 0.51 vs 1.06-1.15; loses to the exposure-matched control). Training
is deterministic per seed (torch.manual_seed + np.random.seed), so this
probe draws 5 FRESH seeds (5, 6, 7, 8, 9) into a separate directory
(checkpoints/naive_probe) to estimate how often the DSR+GRU lands in the
good basin vs the seed-4 basin, WITHOUT touching the canonical artifacts.

For each probe seed: best-val epoch/value, final-val (ep30), train DSR at
ep30, and test-roll Sharpe + cross-seed prediction correlation vs the
canonical pack. Output: checkpoints/naive_probe/basin_probe.csv.
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
from src.models.ddr.data import load_ddr_data  # noqa: E402
from src.models.ddr.eval import roll_test_predictions  # noqa: E402
from src.models.ddr.train import train_ddr  # noqa: E402

PROBE_SEEDS = [5, 6, 7, 8, 9]
PROBE_DIR = DDRConfig().checkpoint_dir / "naive_probe"
PACK_SEEDS = [20260814, 1, 2, 3]  # canonical seeds in the good basin
OUT = PROBE_DIR / "basin_probe.csv"


def main() -> None:
    data = load_ddr_data(DDRConfig().window_size)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== training probe seeds 5-9 (naive DSR, current data) ===")
    hist = {}
    for seed in PROBE_SEEDS:
        c = replace(DDRConfig(), seed=seed, checkpoint_dir=PROBE_DIR / f"s{seed}")
        model, h, path = train_ddr(c, data=data)
        hist[seed] = h
        best_ep = int(h["val_sharpe_ann"].idxmax()) + 1
        print(f"seed {seed}: best val {h['val_sharpe_ann'].max():.3f} @ epoch {best_ep}, "
              f"val@30 {h['val_sharpe_ann'].iloc[-1]:.3f}, "
              f"train DSR@30 {h['train_sharpe_ema'].iloc[-1]:.2f}")

    print("\n=== test rolls (probe seeds) ===")
    pack_actions = []
    for seed in PACK_SEEDS:
        c = replace(DDRConfig(), checkpoint_dir=DDRConfig().checkpoint_dir / "naive_new" / f"s{seed}")
        p = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "ddr_best.pt", data=data)
        pack_actions.append(p[p["valid"]]["action"].to_numpy())
    pack_mean = np.mean(pack_actions, axis=0)

    rows = []
    for seed in PROBE_SEEDS:
        c = replace(DDRConfig(), checkpoint_dir=PROBE_DIR / f"s{seed}")
        p = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "ddr_best.pt", data=data)
        p = p[p["valid"]]
        a = p["action"].to_numpy()
        r = p["ret"].to_numpy()
        h = hist[seed]
        best_ep = int(h["val_sharpe_ann"].idxmax()) + 1
        rec = {
            "seed": seed,
            "best_val_epoch": best_ep,
            "best_val": round(float(h["val_sharpe_ann"].max()), 4),
            "val_ep30": round(float(h["val_sharpe_ann"].iloc[-1]), 4),
            "train_dsr_ep30": round(float(h["train_sharpe_ema"].iloc[-1]), 4),
            "corr_vs_pack": round(float(np.corrcoef(a, pack_mean)[0, 1]), 4),
            "mean_abs_action": round(float(np.abs(a).mean()), 4),
            "short_frac": round(float((a < 0).mean()), 4),
            "sharpe_all": round(regime_eval.sharpe_ratio(r), 4),
            "sharpe_bull": round(regime_eval.sharpe_ratio(r[p["regime"].to_numpy() == "bull"]), 4),
        }
        rows.append(rec)
        basin = "GOOD" if (rec["corr_vs_pack"] > 0.8 and rec["best_val_epoch"] <= 10) else "SEED4-LIKE"
        print(f"seed {seed}: corr_vs_pack {rec['corr_vs_pack']:.3f}, best_ep {best_ep}, "
              f"sharpe_all {rec['sharpe_all']:.3f} -> {basin}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    n_bad = int((df["corr_vs_pack"] <= 0.8).sum())
    print(f"\nbasin probe: {n_bad}/5 seeds in the seed-4-like basin "
          f"(expected if the canonical 1/5 was representative: 1). "
          f"saved: {OUT}")


if __name__ == "__main__":
    main()
