"""Cross-market representation transfer: SPY -> CSI300.

Separates REPRESENTATION transfer from POLICY transfer. A representation is
trained on one market, frozen, and evaluated on another market's decision
task with the SAME protocol (fit on train, eval on test; identical probes and
supervised-direction Sharpe). Compared sources on CSI300:

  raw                 CSI300 raw 20x8 window (no learning)
  csi_<obj>           encoder trained ON CSI300 (in-market reference)
  spy_<obj>           encoder trained on SPY, frozen, applied to CSI300
  random_<obj>        randomly initialized encoder (control)

If spy_<obj> approaches csi_<obj>, the representation transfers; if it
collapses to random_<obj>, financial representations are market-specific.

usage:
  python scripts/rep_cross_market.py [--seeds 3]

Output: data/interpret/rep_cross_market.csv + a printed table.
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

from scripts.representation_rank import eff_rank
from src.data.loaders import REPO_ROOT
from src.interpret.quality import predictive_information
from src.interpret.targets import _naive, align, build_targets
from src.models.rep_lab.config import OBJECTIVES, RepLabConfig, tag_path
from src.models.rep_lab.data import load_rep_data
from src.models.rep_lab.model import build_rep_objective
from src.models.rep_lab.train import (
    BEST, load_rep, supervised_direction_sharpe, train_representation,
)

OUT_DIR = ROOT / "data" / "interpret"
CSI_DIR = REPO_ROOT / "data" / "processed_csi300_backup"
SPY_DIR = REPO_ROOT / "data" / "processed"
SEEDS = [20260814, 111, 222]


def csi_cfg(seed: int, tag: str = "csi") -> RepLabConfig:
    cfg = RepLabConfig()
    cfg.seed, cfg.tag, cfg.processed_dir = seed, tag, CSI_DIR
    return cfg


def spy_cfg(seed: int) -> RepLabConfig:
    cfg = RepLabConfig()
    cfg.seed, cfg.tag = seed, ""
    return cfg


def encode_native(objective: str, cfg: RepLabConfig, windows: torch.Tensor) -> np.ndarray:
    rep = load_rep(objective, cfg, tag_path(cfg, objective) / BEST)
    with torch.no_grad():
        return rep.encoder.encode(windows).numpy()


def encode_random(objective: str, cfg: RepLabConfig, windows: torch.Tensor) -> np.ndarray:
    torch.manual_seed(cfg.seed)
    rep = build_rep_objective(objective, cfg)
    with torch.no_grad():
        return rep.encoder.encode(windows).numpy()


def measure(label: str, seed: int, dates, H: np.ndarray, sub: pd.DataFrame) -> dict:
    rsub, X = align(dates, H, sub)
    te = (rsub["split"] == "test").to_numpy()
    info = predictive_information(X, rsub)
    return {
        "source": label, "seed": seed, "dim": X.shape[1],
        "dir1_auc": info.get("direction_1", {}).get("test", {}).get("auc", np.nan),
        "mag_r2": info.get("magnitude_1", {}).get("test", {}).get("r2", np.nan),
        "vol20_r2": info.get("vol_20", {}).get("test", {}).get("r2", np.nan),
        "regime_bacc": info.get("regime", {}).get("test", {}).get("balanced_accuracy", np.nan),
        "effective_rank": eff_rank(X[te])["effective_rank"],
        "supervised_sharpe": supervised_direction_sharpe(X, rsub, rsub["fwd_ret_1"].to_numpy()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = SEEDS[: args.seeds]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub = build_targets(str(CSI_DIR))          # CSI300 targets
    sub = sub[~_naive(sub.index).duplicated(keep="first")].copy()
    sub.index = _naive(sub.index)

    # CSI300-native encoders (train once per objective/seed)
    for objective in OBJECTIVES:
        for seed in seeds:
            cfg = csi_cfg(seed)
            if not (tag_path(cfg, objective) / BEST).exists():
                print(f"[train csi] {objective} seed {seed}")
                train_representation(objective, cfg)

    csi_base = load_rep_data(csi_cfg(seeds[0]))
    windows, dates = csi_base.windows, csi_base.dates

    rows = []
    for seed in seeds:
        rows.append(measure("raw", seed, dates, windows.reshape(len(dates), -1).numpy(), sub))
        for objective in OBJECTIVES:
            rows.append(measure(f"csi_{objective}", seed, dates,
                                encode_native(objective, csi_cfg(seed), windows), sub))
            rows.append(measure(f"spy_{objective}", seed, dates,
                                encode_native(objective, spy_cfg(seed), windows), sub))
            rows.append(measure(f"random_{objective}", seed, dates,
                                encode_random(objective, csi_cfg(seed), windows), sub))
        print(f"[done] seed {seed}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rep_cross_market.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== CSI300 test metrics (mean over seeds) ===")
    print(df.groupby("source")[["dir1_auc", "vol20_r2", "regime_bacc",
                                "effective_rank", "supervised_sharpe"]]
          .mean().round(3).to_string())
    print("\nwrote rep_cross_market.csv under", OUT_DIR)


if __name__ == "__main__":
    main()
