"""Leverage-matched baseline test for Model B (DDR).

Decomposes DDR's Sharpe edge over buy-and-hold into (i) position SIZING
(de-leveraging) vs (ii) directional SIGN timing (shorts/flips):

  B&H            ret = m_t                    full long, leverage 1.0
  CL (constant)  ret = mean(|a|) * m_t        constant leverage -- Sharpe is
                                              scale-INVARIANT, so CL Sharpe
                                              equals B&H Sharpe by identity
  EM (matched)   ret = |a_t| * m_t            DDR's day-by-day position size,
                                              always long (sizing, no timing)
  DDR            ret = a_t * m_t              sizing + sign timing

EM vs B&H isolates the effect of time-varying sizing; DDR vs EM isolates
the value of sign timing (shorting). Clean-label days only.

Outputs under src/models/ddr/checkpoints/robustness/leverage_test.csv
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


def clean_masks(data) -> np.ndarray:
    features = pd.read_parquet(ROOT / "data" / "processed" / "features_regimes.parquet")
    idx = features.index
    cal = (idx[59:] - idx[:-59]).days
    f = pd.Series(False, index=idx)
    f.iloc[59:] = cal.to_numpy() > 1.6 * 60 + 8
    test = split_ddr_data(data)["test"]
    valid = test.valid.numpy()
    return (~f.reindex(test.dates[valid])).to_numpy()


def per_seed_preds(cfg: DDRConfig, data) -> list[pd.DataFrame]:
    out = []
    for seed in SEEDS:
        c = replace(cfg, seed=seed, checkpoint_dir=ROBUST_DIR / f"s{seed}")
        preds = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "ddr_best.pt", data=data)
        out.append(preds[preds["valid"]].copy())
    return out


def main() -> None:
    cfg = DDRConfig()
    data = load_ddr_data(cfg.window_size)
    clean = clean_masks(data)

    rows = []
    daily_diffs = []
    for seed, preds in zip(SEEDS, per_seed_preds(cfg, data)):
        p = preds[clean].copy()
        a = p["action"].to_numpy()
        m = p["market_ret"].to_numpy()
        lam = np.abs(a).mean()
        p["ret_bh"] = m
        p["ret_cl"] = lam * m
        p["ret_em"] = np.abs(a) * m
        p["ret_ddr"] = p["ret"].to_numpy()
        for regime in ("bull", "all"):
            mask = p["regime"].to_numpy() == regime if regime != "all" else np.ones(len(p), bool)
            for name in ("ret_bh", "ret_cl", "ret_em", "ret_ddr"):
                r = p[name].to_numpy()[mask]
                rows.append(
                    {
                        "seed": seed,
                        "regime": regime,
                        "policy": name.replace("ret_", ""),
                        "n_days": int(mask.sum()),
                        "sharpe": round(regime_eval.sharpe_ratio(r), 4),
                        "cum": round(float(np.prod(1 + r) - 1), 6),
                        "max_dd": round(regime_eval.max_drawdown(r), 5),
                    }
                )
        d = p["ret_ddr"] - p["ret_em"]
        bull = p["regime"].to_numpy() == "bull"
        daily_diffs.append((seed, p["date"][bull].to_numpy(), d[bull]))

    df = pd.DataFrame(rows)
    df.to_csv(ROBUST_DIR / "leverage_test.csv", index=False)

    piv = df.pivot_table(index=["regime", "policy"], columns="seed", values="sharpe")
    piv["mean"] = piv.mean(axis=1)
    piv["std"] = piv.std(axis=1)
    print("=== Sharpe: DDR vs leverage-matched baselines (clean-label days) ===")
    print(piv.round(4).to_string())

    print("\n=== paired per-day: DDR minus exposure-matched (EM) = value of sign timing ===")
    for regime in ("bull",):
        diffs = np.concatenate([d for s, dt, d in daily_diffs])
        t, pval = stats.ttest_1samp(diffs, 0.0)
        print(f"bull: mean diff {diffs.mean():+.6f}/day, t={t:.3f}, p={pval:.4f}")

    cum = df.pivot_table(index=["regime", "policy"], columns="seed", values="cum")
    dd = df.pivot_table(index=["regime", "policy"], columns="seed", values="max_dd")
    print("\n=== cum returns (mean over seeds) ===")
    print(cum.mean(axis=1).round(5).to_string())
    print("\n=== max drawdown (mean over seeds) ===")
    print(dd.mean(axis=1).round(5).to_string())

    print("\nKey identity check: CL (constant leverage) Sharpe should equal B&H Sharpe")
    print("(Sharpe is scale-invariant). Deviation from 0 in the printout = float noise.")


if __name__ == "__main__":
    main()