"""7.27.3 short-frac / final-epoch mechanism sweep (2026-08-30).

Deliberately upweights the mean_reversion family's SAMPLING weight while
holding total volume fixed (same pool, same batch 64 x 3000 budget), and
watches whether final-epoch failure returns as short_frac rises.

Design (single-axis contrast along the mean_reversion weight):
  w_m in [0.25, 0.40, 0.55, 0.70, 0.85]
  bh = momentum = random = (1 - w_m)/3 (each split uniformly within family)
  mean_reversion = w_m (split uniformly over its 20 policies); nr7 = 0
  -> w_m = 0.25 reproduces the comp-matched baseline (0% short, stable); the
     gradient drives the contrarian family share from 0.25 up past the 32p
     unweighted level (0.625) and beyond.

Readout per (point, variant, seed): all-days Sharpe (best-val), FINAL-epoch
Sharpe, short_frac, mean |a|  ->  final-epoch diff = best - final. If
failure (diff << 0) returns at the point where short_frac climbs off zero,
the instability is keyed to the short-side support; if it appears only far
from the old mix regardless of short_frac, it is composition-distance.

Run: python scripts/d_msw_sweep.py [--only wm:seed]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.d.config import DConfig  # noqa: E402
from src.models.d.data import load_d_data  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
VARIANTS = ["d_minus_fuzzy", "d"]
GRID = [0.25, 0.40, 0.55, 0.70, 0.85]
TAG_ROOT = "d_msw"


def family_of(policy: str) -> str:
    return policy.split("_")[0]


def weights_for(cfg_order: list[str], w_m: float) -> list[float]:
    counts = Counter(family_of(p) for p in cfg_order)
    w = []
    for p in cfg_order:
        fam = family_of(p)
        if fam == "mean":
            w.append(w_m / counts["mean"])
        elif fam == "nr7":
            w.append(0.0)
        else:
            w.append((1.0 - w_m) / 3.0 / counts[fam])
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="'0.40:1' -> single job for timing")
    args = ap.parse_args()

    cfg = DConfig.from_yaml()
    pols = list(load_d_data(cfg).policies)
    print(f"policy order ({len(pols)}): {pols}", flush=True)

    jobs = []
    for w_m in GRID:
        w = weights_for(pols, w_m)
        assert abs(sum(w) - 1.0) < 1e-6, (w_m, sum(w))
        fam = {f: round(sum(x for p, x in zip(pols, w) if family_of(p) == f), 3)
               for f in ("buy", "momentum", "mean", "random", "nr7")}
        print(f"w_m={w_m}: family weights {fam}", flush=True)
        wstr = ",".join(f"{x:.9f}" for x in w)
        for variant in VARIANTS:
            for seed in SEEDS:
                jobs.append((w_m, variant, seed, wstr))

    if args.only:
        w_m_s, seed_s = args.only.split(":")
        w_m = float(w_m_s)
        seed = int(seed_s)
        jobs = [(wm, v, s, ws) for (wm, v, s, ws) in jobs
                if wm == w_m and s == seed and v == "d_minus_fuzzy"]

    def run(job):
        w_m, variant, seed, wstr = job
        cmd = ["python", "-m", "src.models.d.train", "--variant", variant,
               "--seed", str(seed), "--tag", f"{TAG_ROOT}/{w_m:g}", "--policy-weights", wstr]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            print(f"!!! {variant} s{seed} w={w_m} rc={r.returncode}", flush=True)
            print((r.stderr or "")[-800:], flush=True)
            return
        last = [l for l in (r.stdout or "").strip().splitlines() if l][-1]
        print(f"ok {variant} s{seed} w={w_m:g}: {last}", flush=True)

    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(run, jobs))
    print("ALL SWEEP TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()