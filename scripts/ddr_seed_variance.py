"""naive_new seed-variance gut-check (Model B baseline).

Canonical Model B is the naive DSR retrained on CURRENT hole-free data
(checkpoints/naive_new). Its 5-seed mean is 0.99 +- 0.24 Sharpe (all days),
with seed 4 an outlier that picked its best checkpoint at epoch 29 and lost
to the exposure-matched control. Before locking this in as the baseline for
the Model C/D comparison, this script asks whether seed 4 is a
qualitatively different training regime or plain seed noise:

  1. training dynamics per seed from training_log.csv (best-val epoch,
     best-val value, trajectory shape -- is the val curve monotone,
     plateaued, or late-spiking?),
  2. per-seed test roll (all/bull/bear Sharpe, exposure, max DD),
  3. cross-seed prediction correlation on the test set (a policy stuck in
     a different optimum shows LOW mean correlation with the others),
  4. verdict: if seed 4's trajectory SHAPE matches the winning seeds and
     its predictions correlate with them, it is noise -- note the variance
     and move on. If it is a different optimum (late-spiking val, low
     correlation), flag it as a recurrent failure mode for C and D.

Output: checkpoints/naive_new/seed_variance.csv + prints.
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

SEEDS = [20260814, 1, 2, 3, 4]
NAIVE_NEW_DIR = DDRConfig().checkpoint_dir / "naive_new"


def training_dynamics() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        log = pd.read_csv(NAIVE_NEW_DIR / f"s{seed}" / "training_log.csv")
        val = log["val_sharpe_ann"].to_numpy()
        best_ep = int(np.nanargmax(val)) + 1
        rows.append(
            {
                "seed": seed,
                "best_val_epoch": best_ep,
                "best_val": float(np.nanmax(val)),
                "val_ep10": float(val[9]),
                "val_ep20": float(val[19]),
                "val_ep30": float(val[29]),
                "val_rise_10to20": float(val[19] - val[9]),
                "val_rise_20to30": float(val[29] - val[19]),
                "train_dsr_ep30": float(log["train_sharpe_ema"].iloc[-1]),
                "train_dsr_rise_20to30": float(
                    log["train_sharpe_ema"].iloc[-1] - log["train_sharpe_ema"].iloc[19]
                ),
                "train_mean_abs_ep30": float(log["train_mean_abs_action"].iloc[-1]),
                "train_short_frac_ep30": float(log["train_short_frac"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def test_rolls() -> list[pd.DataFrame]:
    cfg = DDRConfig()
    out = []
    for seed in SEEDS:
        c = replace(cfg, checkpoint_dir=NAIVE_NEW_DIR / f"s{seed}")
        preds = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "ddr_best.pt")
        out.append(preds[preds["valid"]].copy())
    return out


def roll_summary(preds: pd.DataFrame) -> dict:
    a = preds["action"].to_numpy()
    r = preds["ret"].to_numpy()
    out = {
        "mean_abs_action": round(float(np.abs(a).mean()), 4),
        "short_frac": round(float((a < 0).mean()), 4),
        "max_abs_action": round(float(np.abs(a).max()), 4),
        "sharpe_all": round(regime_eval.sharpe_ratio(r), 4),
        "cum_all": round(float(np.prod(1 + r) - 1), 4),
        "max_dd": round(regime_eval.max_drawdown(r), 5),
    }
    for regime in ("bull", "bear", "crisis"):
        m = preds["regime"].to_numpy() == regime
        out[f"sharpe_{regime}"] = round(
            regime_eval.sharpe_ratio(r[m]) if m.sum() else float("nan"), 4
        )
        out[f"n_{regime}"] = int(m.sum())
    return out


def main() -> None:
    data = load_ddr_data(DDRConfig().window_size)
    dyn = training_dynamics()
    print("=== 1. training dynamics (naive_new, 30 epochs) ===")
    print(dyn.round(4).to_string(index=False))

    preds_all = test_rolls()
    rows = []
    for seed, p in zip(SEEDS, preds_all):
        rec = {"seed": seed}
        rec.update(roll_summary(p))
        rows.append(rec)
    summ = pd.DataFrame(rows)
    print("\n=== 2. test roll (valid days only, all/bull/bear) ===")
    print(summ.round(4).to_string(index=False))

    print("\n=== 3. cross-seed prediction correlation (test days) ===")
    A = np.vstack([p["action"].to_numpy() for p in preds_all])
    corr = np.corrcoef(A)
    corr_df = pd.DataFrame(corr, index=SEEDS, columns=SEEDS).round(4)
    print(corr_df.to_string())
    mean_corr = {
        s: float(np.nanmean([corr[i][j] for j in range(5) if j != i]))
        for i, s in enumerate(SEEDS)
    }
    print("mean off-diagonal corr per seed:", {s: round(v, 4) for s, v in mean_corr.items()})

    print("\n=== 4. verdict ===")
    s4 = dyn[dyn["seed"] == 4].iloc[0]
    others = dyn[dyn["seed"] != 4]
    late_spike = s4["val_rise_20to30"] > 0 and s4["val_rise_10to20"] <= 0
    low_corr = mean_corr[4] < np.nanmedian(list(mean_corr.values()))
    if late_spike and low_corr:
        print("seed 4 is a QUALITATIVELY different training regime (late-spiking val,")
        print("low cross-seed prediction correlation) -- flag as a recurrent failure")
        print("mode to watch in Model C/D, not plain noise.")
    else:
        print("seed 4's trajectory shape matches the winning seeds and its predictions")
        print("correlate with them -> the 0.99 +- 0.24 spread is seed noise.")
        if late_spike:
            print("  (note: late-spiking val -- checkpoint selection is picking the tail;")
            print("   verify this in C/D by checking best-val epoch distributions)")

    summ.to_csv(NAIVE_NEW_DIR / "seed_variance.csv", index=False)
    print(f"\nseed_variance saved: {NAIVE_NEW_DIR / 'seed_variance.csv'}")


if __name__ == "__main__":
    main()
