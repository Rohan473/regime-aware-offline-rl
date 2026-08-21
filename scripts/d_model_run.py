"""Model D (Phase 4) screen driver — the pre-registered protocol end to end.

1. Train BOTH variants (D, D-minus-fuzzy) x 5 seeds at the 3k-step screen
   budget (10 epochs x 300 steps; 2-parallel like Models B/C).
2. Run the shared eval (regime table, basin screening, vs B/C/EM, and the
   pre-registered screen / escalation / ablation verdicts).
3. IF the screen summary says "escalate" (either variant beat EM in >= 3/5
   seeds), train BOTH variants x 5 seeds at the 20k escalation budget
   (40 x 500) under tag d_20k and re-run eval there.

Usage:  python scripts/d_model_run.py [--parallel 2]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20260814, 1, 2, 3, 4]
VARIANTS = ("d", "d_minus_fuzzy")
ESCALATION_EPOCHS, ESCALATION_SPE = 40, 500


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_variants(parallel: int, tag: str | None = None,
                   epochs: int | None = None, spe: int | None = None) -> None:
    jobs = []
    for variant in VARIANTS:
        for seed in SEEDS:
            cmd = ["python", "-m", "src.models.d.train", "--variant", variant, "--seed", str(seed)]
            if tag:
                cmd += ["--tag", tag]
            if epochs is not None:
                cmd += ["--epochs", str(epochs)]
            if spe is not None:
                cmd += ["--steps-per-epoch", str(spe)]
            jobs.append(cmd)
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        list(ex.map(run, jobs))


def eval_run(tag: str | None = None) -> pd.DataFrame:
    cmd = ["python", "-m", "src.models.d.eval"]
    if tag:
        cmd += ["--tag", tag]
    run(cmd)
    out = ROOT / "src" / "models" / "d" / "checkpoints" / (tag or "") / "screen_summary.csv"
    if not out.exists():
        raise FileNotFoundError(f"screen_summary.csv not found at {out}")
    return pd.read_csv(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--escalate-only", action="store_true",
                        help="skip the 3k screen (already done) and run the 20k escalation directly")
    args = parser.parse_args()

    if not args.escalate_only:
        print("=== Phase 4 screen: 3k steps, 5 seeds, both variants ===")
        train_variants(args.parallel)
        screen = eval_run()
        print(screen.to_string(index=False))

        escalate = bool(screen.loc[screen["metric"] == "escalate_3k_to_20k",
                                    ["D", "D_minus_fuzzy"]].iloc[0].any())
        if not escalate:
            print("=== no escalation: neither variant cleared the screen at 3k ===")
            print("D remains closed at the 3k budget; no 20k run (pre-registered).")
            return
    else:
        print("=== escalate-only: 3k screen results already on disk ===")

    print(f"=== escalation triggered: {ESCALATION_EPOCHS}x{ESCALATION_SPE} = "
          f"{ESCALATION_EPOCHS * ESCALATION_SPE} steps, both variants, tag d_20k ===")
    train_variants(args.parallel, tag="d_20k", epochs=ESCALATION_EPOCHS, spe=ESCALATION_SPE)
    screen20 = eval_run(tag="d_20k")
    print(screen20.to_string(index=False))
    print("=== Phase 4 protocol complete ===")


if __name__ == "__main__":
    main()