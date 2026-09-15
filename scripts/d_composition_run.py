"""7.27.2 volume-vs-composition discriminator for the Model D/IQL regression.

Holds TOTAL VOLUME FIXED and varies ONLY the family proportions at sampling
time, per the clean-confounding-control protocol:

  - pool         : the full 32-policy offline dataset (168,516 rows), the
                   SAME pool the unweighted d_32p run used
  - total volume : identical (batch 64 x 3000 steps, same screen budget)
  - treatment    : policy_weights re-map the per-family shares back to the
                   pre-7.11 4-policy proportions (buy_and_hold 0.25 /
                   momentum 0.25 / mean_reversion 0.25 / random 0.25; nr7
                   excluded — it was not in the old mix), split uniformly
                   within each family.

The three-point comparison: (a) 4-policy old -> D-minus-fuzzy 0.881;
(b) 32-policy UNWEIGHTED -> 0.714; (c) 32-policy COMPOSITION-MATCHED -> ?.
If (c) ~ (a): composition was the driver, volume is fine. If (c) ~ (b):
volume itself (or the expectile/AWR fit at 6.6x samples) is the driver.

Run: python scripts/d_composition_run.py   (writes checkpoints/d_cmp/)
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.d.config import DConfig  # noqa: E402
from src.models.d.data import load_d_data  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
TAG = "d_cmp"
# pre-7.11 per-family proportions (buy_and_hold / momentum_20d /
# mean_reversion_5d / random were each 1 of 4 policies sampled uniformly)
FAMILY_TARGET = {"buy": 0.25, "momentum": 0.25, "mean": 0.25, "random": 0.25, "nr7": 0.0}


def family_of(policy: str) -> str:
    return policy.split("_")[0]


def main() -> None:
    cfg = DConfig.from_yaml()
    data = load_d_data(cfg)  # policy ORDER only (same tuple for both variants)
    policies = data.policies
    counts = Counter(family_of(p) for p in policies)
    weights = [FAMILY_TARGET[family_of(p)] / counts[family_of(p)] for p in policies]
    assert abs(sum(weights) - 1.0) < 1e-6

    fam_grid = {fam: round(sum(w for p, w in zip(policies, weights)
                               if family_of(p) == fam), 3)
                for fam in FAMILY_TARGET}
    print(f"policies ({len(policies)}): {list(policies)}", flush=True)
    print(f"sampling family weights (7.27.2): {fam_grid}", flush=True)
    print(f"pool family counts: {dict(Counter(family_of(p) for p in policies))}", flush=True)
    wstr = ",".join(f"{w:.9f}" for w in weights)

    jobs = []
    for variant in ("d_minus_fuzzy", "d"):
        for seed in SEEDS:
            jobs.append(["python", "-m", "src.models.d.train", "--variant", variant,
                         "--seed", str(seed), "--tag", TAG, "--policy-weights", wstr])

    def run(cmd):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print(f"### {cmd[4]} s{cmd[6]} rc={r.returncode}", flush=True)
        if r.returncode:
            print((r.stderr or "")[-1500:], flush=True)
            return
        lines = [l for l in (r.stdout or "").strip().splitlines() if l]
        print(lines[-1] if lines else "(no output)", flush=True)

    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(run, jobs))
    print("ALL TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()