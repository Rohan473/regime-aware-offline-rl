"""Exp 1 (7.30.1): 16-dim macro TACR training — 5 seeds.

Matches the canonical fix_a_nr7 recipe (dq-MIN + uniform + no-balanced-
families) exactly, adding only --state-macro (widen state to 8 SPY z + 8
macro causal z, dates restricted to 2007+) and a distinct --tag macro16.

Checkpoints: src/models/tacr/checkpoints/tacr/macro16/s{seed}/tacr_best.pt

Run: python scripts/macro16_run.py
"""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20260814, 1, 2, 3, 4]
TAG = "macro16"

CMD = [
    "python", "-m", "src.models.tacr.train",
    "--use-double-q", "--double-q-mode", "min",
    "--no-balanced-families",
    "--state-macro",
    "--tag", TAG,
]

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(
            lambda s: (print(f"$ {' '.join(CMD + ['--seed', str(s)])}", flush=True),
                       subprocess.run(CMD + ["--seed", str(s)], cwd=ROOT, check=True)),
            SEEDS,
        ))
    print(f"=== suite complete: {TAG} ===")
