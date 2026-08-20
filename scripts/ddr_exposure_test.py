"""Exposure-regularized DDR (Model B variant) — 5-seed protocol test.

Tests the direct exposure-level regularization fix against the canonical
naive_new baseline and the vol-targeted (vt) variant, on the same SEEDS
and the same clean-day leverage reporting used by ddr_vt_retrain.py.

The DSR gradient is level-free (a uniform de-leverage cancels in the
Sharpe ratio), which is the diagnosed mechanism behind the drift to
mean|a| ~ 0.26. This variant adds lambda * |mean(|a_t|) - target| per
training block, forcing the average magnitude of positions toward the
target while leaving the DSR's direction/relative-confidence signal
untouched. Mean-variance reward (E[r] - kappa*Var[r]) is deliberately
NOT implemented here — it is logged in PROJECT_NOTES 7.8 as an explicit
separate variant (it replaces the paper's reward entirely and adds a
risk-aversion parameter).

Output: checkpoints/exp/s{seed}/ddr_best.pt + training_log.csv per seed,
and checkpoints/exp/exposure_test.csv with the comparison table.
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
from src.models.ddr.train import train_ddr  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
EXP_DIR = DDRConfig().checkpoint_dir / "exp"            # exposure-regularized (this test)
NAIVE_NEW_DIR = DDRConfig().checkpoint_dir / "naive_new"  # canonical B baseline
VT_DIR = DDRConfig().checkpoint_dir / "vt"              # vol-targeted variant

TARGET_EXPOSURE = 0.75
LAMBDA = 1.0


def train_seeds(cfg: DDRConfig, out_dir: Path) -> None:
    for seed in SEEDS:
        c = replace(cfg, seed=seed, checkpoint_dir=out_dir / f"s{seed}")
        ckpt = c.checkpoint_dir / "ddr_best.pt"
        if ckpt.exists():
            print(f"[{out_dir.name}] seed {seed}: reusing {ckpt}")
            continue
        model, history, checkpoint = train_ddr(c)
        print(f"[{out_dir.name}] seed {seed}: best val Sharpe {history['val_sharpe_ann'].max():.3f} "
              f"@ epoch {int(history['val_sharpe_ann'].idxmax())} | "
              f"mean|a| {history['train_mean_abs_action'].iloc[-1]:.3f}")


def per_seed_preds(cfg: DDRConfig, data) -> list[pd.DataFrame]:
    out = []
    for seed in SEEDS:
        c = replace(cfg, seed=seed, checkpoint_dir=Path(cfg.checkpoint_dir) / f"s{seed}")
        preds = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "ddr_best.pt", data=data)
        out.append(preds[preds["valid"]].copy())
    return out


def clean_masks(data) -> np.ndarray:
    features = pd.read_parquet(ROOT / "data" / "processed" / "features_regimes.parquet")
    idx = features.index
    cal = (idx[59:] - idx[:-59]).days
    f = pd.Series(False, index=idx)
    f.iloc[59:] = cal.to_numpy() > 1.6 * 60 + 8
    test = split_ddr_data(data)["test"]
    valid = test.valid.numpy()
    return (~f.reindex(test.dates[valid])).to_numpy()


def leverage_table(preds_by_seed, clean: np.ndarray) -> pd.DataFrame:
    rows = []
    for seed, preds in zip(SEEDS, preds_by_seed):
        p = preds[clean].copy()
        a = p["action"].to_numpy()
        for regime in ("all", "bull"):
            mask = p["regime"].to_numpy() == regime if regime != "all" else np.ones(len(p), bool)
            r = p["ret"].to_numpy()[mask]
            rows.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "sharpe_ann": regime_eval.sharpe_ratio(r),
                    "mean_abs_action": float(np.abs(a[mask]).mean()),
                    "short_frac": float((a[mask] < 0).mean()),
                    "turnover": float(np.abs(np.diff(a[mask])).mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    base = replace(DDRConfig(), vol_targeting=False)  # same base as canonical naive_new
    exp = replace(
        base,
        exposure_regularization=True,
        target_exposure=TARGET_EXPOSURE,
        exposure_lambda=LAMBDA,
    )
    train_seeds(exp, EXP_DIR)

    data = load_ddr_data(base.window_size)
    clean = clean_masks(data)
    exp_preds = per_seed_preds(replace(exp, checkpoint_dir=EXP_DIR), data)
    na_preds = per_seed_preds(replace(base, checkpoint_dir=NAIVE_NEW_DIR), data)
    vt_preds = per_seed_preds(replace(base, checkpoint_dir=VT_DIR), data)

    exp_t = leverage_table(exp_preds, clean)
    na_t = leverage_table(na_preds, clean)
    vt_t = leverage_table(vt_preds, clean)
    for t, name in ((exp_t, "exp"), (na_t, "naive_new"), (vt_t, "vt")):
        t["run"] = name
    table = pd.concat([na_t, vt_t, exp_t], ignore_index=True)

    print("\n=== exposure test (clean-label days, all test) ===")
    print(table[table["regime"] == "all"].to_string(index=False))
    print("\n=== bull ===")
    print(table[table["regime"] == "bull"].to_string(index=False))

    for regime in ("all", "bull"):
        for run in ("naive_new", "vt", "exp"):
            s = table[(table["run"] == run) & (table["regime"] == regime)]
            print(f"{run} {regime}: Sharpe {s['sharpe_ann'].mean():.3f} +- {s['sharpe_ann'].std():.3f} | "
                  f"mean|a| {s['mean_abs_action'].mean():.3f}")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(EXP_DIR / "exposure_test.csv", index=False)
    print(f"\nwrote {EXP_DIR / 'exposure_test.csv'}")


if __name__ == "__main__":
    main()