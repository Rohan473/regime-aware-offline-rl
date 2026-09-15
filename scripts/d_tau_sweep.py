"""τ sweep on the 32-policy unweighted DNF dataset.

TWO taus in Model D (src/models/d/config.py):
  - polyak tau   (cfg.tau, default 0.005) : target-network EMA rate
  - expectile tau(cfg.expectile, default 0.7): IQL value conservatism level

Goal: does tuning τ stabilize the 32p collapse that β could not (7.27.4)?

Pre-registered success criteria:
  (1) Does any τ (either family) achieve final-epoch PASS on 32p-unweighted?
  (2) If so, does the corresponding τ on comp-matched improve Sharpe > 0.837?
  (3) If no τ stabilizes 32p, τ (like β) is not the binding constraint.

Run: python scripts/d_tau_sweep.py
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEEDS = [20260814, 1, 2, 3, 4]
VARIANTS = ["d_minus_fuzzy", "d"]

# (arg_flag, values, tag_root, label)
SWEEPS = [
    ("--tau", [0.001, 0.0025, 0.005, 0.01, 0.02], "d_tau_polyak", "polyak"),
    ("--expectile", [0.5, 0.6, 0.7, 0.8, 0.9], "d_tau_exp", "expectile"),
]


def main() -> None:
    jobs = []
    for flag, values, tag_root, label in SWEEPS:
        for v in values:
            for variant in VARIANTS:
                for seed in SEEDS:
                    jobs.append((flag, v, variant, seed, tag_root, label))

    def run(job):
        flag, v, variant, seed, tag_root, label = job
        cmd = ["python", "-m", "src.models.d.train", "--variant", variant,
               "--seed", str(seed), "--tag", f"{tag_root}/{v:g}",
               flag, str(v)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            print(f"!!! {variant} s{seed} {label}={v:g} rc={r.returncode}", flush=True)
            print((r.stderr or "")[-800:], flush=True)
            return
        last = [l for l in (r.stdout or "").strip().splitlines() if l][-1]
        print(f"ok {variant} s{seed} {label}={v:g}: {last}", flush=True)

    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(run, jobs))
    print("ALL TAU SWEEP TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()