"""Confidence-signal check for Model B (distinguishes the two mechanism
hypotheses recorded in PROJECT_NOTES 7.7).

H1 (confidence-encoding): the DSR's small positions encode low conviction.
  Predicts: in the UNCONSTRAINED naive_new policy, per-unit performance
  (ret/|a| = sign(a) * R, cost=0) RISES with position size, and/or small
  positions cluster in ambiguous regimes (bear/crisis) while large ones
  cluster in bull.
H2 (generic interference): the exposure penalty destabilizes optimization
  regardless of what naive_new's exposure means. Predicts: per-unit
  performance is FLAT across position size in naive_new.

Also reports the per-seed exp-pack breakdown (bimodality check): if 2
seeds flipped catastrophically and 3 degraded mildly, the "penalty
universally induces flipping" story is wrong — it is training instability
under a conflicting objective.

Outputs: prints + scripts/../src/models/ddr/checkpoints/naive_new/confidence_signal.csv
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.eval import regime_eval  # noqa: E402
from src.models.ddr.config import DDRConfig  # noqa: E402
from src.models.ddr.data import load_ddr_data, split_ddr_data  # noqa: E402
from src.models.ddr.eval import roll_test_predictions  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
NAIVE_NEW_DIR = DDRConfig().checkpoint_dir / "naive_new"
EXP_DIR = DDRConfig().checkpoint_dir / "exp"


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
        c = replace(cfg, seed=seed, checkpoint_dir=Path(cfg.checkpoint_dir) / f"s{seed}")
        preds = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "ddr_best.pt", data=data)
        out.append(preds[preds["valid"]].copy())
    return out


def size_bucket_analysis(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-seed tercile buckets of |a_t|: per-unit performance + regime mix."""
    rows = []
    for seed, p in zip(SEEDS, preds):
        p = p.copy()
        a = p["action"].to_numpy()
        buckets = pd.qcut(pd.Series(np.abs(a)), 3, labels=["small", "med", "large"])
        p["bucket"] = buckets
        for b in ("small", "med", "large"):
            g = p[p["bucket"] == b]
            rows.append(
                {
                    "seed": seed,
                    "bucket": b,
                    "n": len(g),
                    "mean_abs_action": float(np.abs(g["action"]).mean()),
                    "per_unit_ret": float((g["ret"] / np.abs(g["action"]).clip(1e-12)).mean()),
                    "mean_abs_market_ret": float(np.abs(g["market_ret"]).mean()),
                    "bull_frac": float((g["regime"] == "bull").mean()),
                    "bear_frac": float((g["regime"] == "bear").mean()),
                    "crisis_frac": float((g["regime"] == "crisis").mean()),
                }
            )
    return pd.DataFrame(rows)


def corr_checks(preds: list[pd.DataFrame]) -> pd.DataFrame:
    """corr(|a_t|, per-unit ret) and corr(|a_t|, |R|) per seed + pooled."""
    rows = []
    pooled_a, pooled_q, pooled_m = [], [], []
    for seed, p in zip(SEEDS, preds):
        a = np.abs(p["action"].to_numpy())
        q = (p["ret"].to_numpy() / np.clip(a, 1e-12))  # sign(a)*R
        m = np.abs(p["market_ret"].to_numpy())
        rows.append(
            {
                "seed": seed,
                "corr_abs_a_perunit": float(np.corrcoef(a, q)[0, 1]),
                "corr_abs_a_abs_mkt": float(np.corrcoef(a, m)[0, 1]),
            }
        )
        pooled_a.append(a); pooled_q.append(q); pooled_m.append(m)
    a = np.concatenate(pooled_a); q = np.concatenate(pooled_q); m = np.concatenate(pooled_m)
    rows.append(
        {
            "seed": "pooled",
            "corr_abs_a_perunit": float(np.corrcoef(a, q)[0, 1]),
            "corr_abs_a_abs_mkt": float(np.corrcoef(a, m)[0, 1]),
        }
    )
    return pd.DataFrame(rows)


def regime_exposure(preds: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for seed, p in zip(SEEDS, preds):
        for regime in ("bull", "bear", "crisis"):
            g = p[p["regime"] == regime]
            if len(g) < 5:
                continue
            rows.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "n": len(g),
                    "mean_abs_action": float(np.abs(g["action"]).mean()),
                    "sharpe_ann": regime_eval.sharpe_ratio(g["ret"].to_numpy()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    data = load_ddr_data(DDRConfig().window_size)
    clean = clean_masks(data)
    na = [p[clean] for p in per_seed_preds(replace(DDRConfig(), checkpoint_dir=NAIVE_NEW_DIR), data)]
    exp = [p[clean] for p in per_seed_preds(replace(DDRConfig(), checkpoint_dir=EXP_DIR), data)]

    print("=== 1. exp-pack per-seed breakdown (bimodality check) ===")
    rows = []
    for seed, p in zip(SEEDS, exp):
        a = p["action"].to_numpy()
        rows.append(
            {
                "seed": seed,
                "sharpe_all": regime_eval.sharpe_ratio(p["ret"].to_numpy()),
                "mean_abs_action": float(np.abs(a).mean()),
                "short_frac": float((a < 0).mean()),
                "turnover": float(np.abs(np.diff(a)).mean()),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 2. naive_new: per-unit performance by |a_t| tercile ===")
    buckets = size_bucket_analysis(na)
    print(buckets.groupby("bucket", sort=False)[
        ["per_unit_ret", "mean_abs_market_ret", "bull_frac", "bear_frac", "crisis_frac"]
    ].mean().round(4).to_string())

    print("\n=== 3. naive_new: corr(|a|, per-unit ret) and corr(|a|, |R|) ===")
    print(corr_checks(na).round(4).to_string(index=False))

    print("\n=== 4. naive_new: mean |a| and Sharpe by regime ===")
    print(regime_exposure(na).pivot_table(
        index="regime", columns="seed", values="mean_abs_action"
    ).round(3).to_string())

    out = size_bucket_analysis(na)
    out.to_csv(NAIVE_NEW_DIR / "confidence_signal.csv", index=False)
    print(f"\nwrote {NAIVE_NEW_DIR / 'confidence_signal.csv'}")


if __name__ == "__main__":
    main()