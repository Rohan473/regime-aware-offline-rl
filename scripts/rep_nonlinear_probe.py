"""Linear vs GBDT vs MLP probes on the frozen representations.

Tests whether directional (and magnitude / volatility) information that is not
LINEARLY accessible from h_t is nonetheless recoverable NONLINEARLY. Same
frozen representations, same no-lookahead protocol, three probe families.

usage:
  python scripts/rep_nonlinear_probe.py [--seeds 10] [--targets direction_1,magnitude_1,vol_20]

Output: data/interpret/rep_nonlinear_probe.csv (long form) + a printed table.
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

from src.interpret.nonlinear import FAMILIES, probe_family_metrics
from src.interpret.targets import _naive, align, build_targets
from src.models.rep_lab.config import OBJECTIVES, RepLabConfig, tag_path
from src.models.rep_lab.train import BEST, extract_rep, raw_window_rep

OUT_DIR = ROOT / "data" / "interpret"
SEEDS = [20260814, 111, 222, 333, 444, 555, 666, 777, 888, 999]
TARGETS = {"direction_1": ("clf", "fwd_dir_1"),
           "magnitude_1": ("reg", "abs_ret_1"),
           "vol_20": ("reg", "fwd_vol_20")}
REPS = ["raw", *OBJECTIVES]


def rep_for(name: str, cfg: RepLabConfig):
    if name == "raw":
        return raw_window_rep(cfg)
    return extract_rep(name, cfg, tag_path(cfg, name) / BEST)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--targets", default=",".join(TARGETS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [t for t in args.targets.split(",") if t]
    seeds = SEEDS[: args.seeds]

    sub = build_targets()
    sub = sub[~_naive(sub.index).duplicated(keep="first")].copy()
    sub.index = _naive(sub.index)

    rows = []
    for rep_name in REPS:
        for seed in seeds:
            cfg = RepLabConfig()
            cfg.seed = seed
            rep = rep_for(rep_name, cfg)
            rsub, X = align(rep.dates, rep.H, sub)
            for target in targets:
                kind, col = TARGETS[target]
                for family in FAMILIES:
                    r = probe_family_metrics(X, rsub, col, kind, family)
                    if r is None:
                        continue
                    row = {"representation": rep.name, "rep": rep_name, "seed": seed,
                           "target": target, "family": family}
                    row.update({f"{k}_{m}": v for k, d in r.items() for m, v in d.items()})
                    rows.append(row)
        print(f"[done] {rep_name}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rep_nonlinear_probe.csv", index=False)

    pd.set_option("display.width", 220)
    for target in targets:
        t = df[df["target"] == target]
        key = "test_auc" if TARGETS[target][0] == "clf" else "test_r2"
        print(f"\n=== {target} ({key}) mean over seeds ===")
        print(t.pivot_table(index="rep", columns="family", values=key,
                            aggfunc="mean").round(3).to_string())
    print("\nwrote rep_nonlinear_probe.csv under", OUT_DIR)


if __name__ == "__main__":
    main()
