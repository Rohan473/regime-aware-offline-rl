"""Representation-scaling sweeps: feature count (4->32) and latent dim.

Trains the SAME encoder across a grid and measures the quality axes, so we can
ask whether more features / more latent dimensions actually buy information or
downstream utility -- and whether the two move together.

  --axis feature   feature subsets 4/8/12/16/24/32 (nested 32-feature bank),
                   hidden fixed at 128
  --axis dim       hidden 4/8/16/32/64/128, canonical 8 features

The (8-feature, 128-d) cell IS the headline rep-lab configuration, so it
reuses the existing untagged checkpoints; every other cell gets a tag
(``f16`` / ``h32``) under the same checkpoint root.

Per (axis, size, objective, seed) we record: dir1 AUC, magnitude R2, vol20 R2,
regime balanced accuracy, effective rank, and supervised direction Sharpe.
Per (axis, size, objective) we also record the cross-seed CKA (reproducibility)
and the mean/std across seeds.

Outputs (data/interpret/):
  rep_scaling.csv          per-seed metrics
  rep_scaling_summary.csv  mean/std + cross-seed CKA per config
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

from scripts.representation_rank import eff_rank
from src.data.feature_bank import FEATURE_SETS
from src.interpret.quality import mean_pairwise_cka, predictive_information
from src.interpret.targets import _naive, align, build_targets
from src.models.rep_lab.config import OBJECTIVES, RepLabConfig, tag_path
from src.models.rep_lab.train import (
    BEST, extract_rep, supervised_direction_sharpe, train_representation,
)

OUT_DIR = ROOT / "data" / "interpret"
SEEDS = [20260814, 111, 222, 333, 444, 555, 666, 777, 888, 999]
FEATURE_SIZES = [4, 8, 12, 16, 24, 32]
DIM_SIZES = [4, 8, 16, 32, 64, 128]


def make_cfg(axis: str, size: int, seed: int) -> RepLabConfig:
    cfg = RepLabConfig()
    cfg.seed = seed
    if axis == "feature":
        if size == 8:  # headline configuration -> reuse untagged checkpoints
            cfg.feature_cols, cfg.tag = None, ""
        else:
            cfg.feature_cols, cfg.tag = tuple(FEATURE_SETS[size]), f"f{size}"
    else:  # dim
        if size == 128:
            cfg.tag = ""
        else:
            cfg.hidden, cfg.tag = size, f"h{size}"
    return cfg


def rep_name(objective: str, cfg: RepLabConfig) -> str:
    return objective if not cfg.tag else f"{objective}_{cfg.tag}"


def train_cell(axis: str, size: int, objective: str, seeds: list[int]) -> None:
    for seed in seeds:
        cfg = make_cfg(axis, size, seed)
        ck = tag_path(cfg, objective) / BEST
        if ck.exists():
            continue
        print(f"[train] axis={axis} size={size} obj={objective} seed={seed}")
        train_representation(objective, cfg)


def measure_cell(axis: str, size: int, objective: str, seeds: list[int],
                 sub: pd.DataFrame) -> tuple[list, dict | None]:
    rows, Hs = [], []
    for seed in seeds:
        cfg = make_cfg(axis, size, seed)
        ck = tag_path(cfg, objective) / BEST
        if not ck.exists():
            continue
        rep = extract_rep(objective, cfg, ck)
        rsub, X = align(rep.dates, rep.H, sub)
        te = (rsub["split"] == "test").to_numpy()
        info = predictive_information(X, rsub)
        row = {"axis": axis, "size": size, "objective": objective, "seed": seed,
               "dir1_auc": info.get("direction_1", {}).get("test", {}).get("auc", np.nan),
               "mag_r2": info.get("magnitude_1", {}).get("test", {}).get("r2", np.nan),
               "vol20_r2": info.get("vol_20", {}).get("test", {}).get("r2", np.nan),
               "regime_bacc": info.get("regime", {}).get("test", {}).get("balanced_accuracy", np.nan),
               "effective_rank": eff_rank(X[te])["effective_rank"],
               "supervised_sharpe": supervised_direction_sharpe(
                   X, rsub, rsub["fwd_ret_1"].to_numpy())}
        rows.append(row)
        Hs.append(X[te])
    if not rows:
        return rows, None
    n = min(len(h) for h in Hs)
    summary = {"axis": axis, "size": size, "objective": objective,
               "n_seeds": len(rows),
               "cross_seed_cka": mean_pairwise_cka([h[:n] for h in Hs])}
    df = pd.DataFrame(rows)
    for col in ("dir1_auc", "mag_r2", "vol20_r2", "regime_bacc", "effective_rank",
                "supervised_sharpe"):
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["feature", "dim", "both"], default="both")
    ap.add_argument("--objectives", default=",".join(OBJECTIVES))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--no-measure", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    objectives = [o for o in args.objectives.split(",") if o]
    seeds = SEEDS[: args.seeds]
    axes = ["feature", "dim"] if args.axis == "both" else [args.axis]

    sub = build_targets()
    sub = sub[~_naive(sub.index).duplicated(keep="first")].copy()
    sub.index = _naive(sub.index)

    all_rows, all_summary = [], []
    for axis in axes:
        sizes = FEATURE_SIZES if axis == "feature" else DIM_SIZES
        for objective in objectives:
            for size in sizes:
                if not args.no_train:
                    train_cell(axis, size, objective, seeds)
                if not args.no_measure:
                    rows, summary = measure_cell(axis, size, objective, seeds, sub)
                    all_rows += rows
                    if summary:
                        all_summary.append(summary)
                    print(f"[measure] axis={axis} size={size} obj={objective} "
                          f"n={len(rows)} cka={summary['cross_seed_cka'] if summary else float('nan'):.3f}")

    if all_rows:
        pd.DataFrame(all_rows).to_csv(OUT_DIR / "rep_scaling.csv", index=False)
    if all_summary:
        s = pd.DataFrame(all_summary)
        s.to_csv(OUT_DIR / "rep_scaling_summary.csv", index=False)
        pd.set_option("display.width", 240)
        print("\n=== scaling summary ===")
        for axis in axes:
            a = s[s["axis"] == axis]
            if a.empty:
                continue
            print(f"\n--- {axis} ---")
            print(a.pivot_table(index="size", columns="objective",
                                values="supervised_sharpe_mean").round(3).to_string())
    print("\nwrote rep_scaling.csv / rep_scaling_summary.csv under", OUT_DIR)


if __name__ == "__main__":
    main()
