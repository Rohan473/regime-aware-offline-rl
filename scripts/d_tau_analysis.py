"""τ sweep analysis (polyak tau + expectile tau) on 32p-unweighted.

Readout per (family, τ, variant, seed): BEST and FINAL all-days Sharpe,
final-epoch diff, short_frac (best/final), mean|a|. Compares against the
d_32p anchor.

Run: python scripts/d_tau_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.regime_eval import sharpe_ratio  # noqa: E402
from src.models.d.config import DConfig  # noqa: E402
from src.models.d.eval import VARIANT_DIRS, BEST_NAME, FINAL_NAME  # noqa: E402
from src.models.d.eval import roll_test_preds  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
VARIANTS = ["D", "D-minus-fuzzy"]
FAMILIES = {
    "polyak": ("d_tau_polyak", [0.001, 0.0025, 0.005, 0.01, 0.02]),
    "expectile": ("d_tau_exp", [0.5, 0.6, 0.7, 0.8, 0.9]),
}
ROWS = []


def add(tag: str, base: Path, variants: list[str], seeds: list[int]) -> None:
    for variant in variants:
        for s in seeds:
            for name, kind in ((BEST_NAME, "best"), (FINAL_NAME, "final")):
                cp = base / VARIANT_DIRS[variant] / f"s{s}" / name
                if not cp.exists():
                    continue
                c = DConfig.from_yaml()
                c.checkpoint_dir = base
                try:
                    p = roll_test_preds(c, variant, s, name)
                except FileNotFoundError:
                    continue
                a = p["action"].to_numpy()
                r = p["ret"].to_numpy()
                sh = sharpe_ratio(r)
                if kind == "best":
                    ROWS.append({"tag": tag, "variant": variant, "seed": s,
                                 "sh_best": sh, "sf_best": float(np.mean(a < 0)),
                                 "absa_best": float(np.mean(np.abs(a)))})
                else:
                    for row in ROWS:
                        if (row["tag"], row["variant"], row["seed"]) == (tag, variant, s):
                            row["sh_final"] = sh
                            row["sf_final"] = float(np.mean(a < 0))
                            row["absa_final"] = float(np.mean(np.abs(a)))


def main() -> None:
    ck = ROOT / "src/models/d/checkpoints"
    for family, (tag_root, vals) in FAMILIES.items():
        for v in vals:
            add(f"{family}/{v:g}", ck / tag_root / f"{v:g}", VARIANTS, SEEDS)
    add("d_32p", ck / "d_32p", VARIANTS, SEEDS)

    df = pd.DataFrame(ROWS)
    df["diff"] = df["sh_final"] - df["sh_best"]
    out = ck / "d_tau_polyak"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tau_analysis.csv", index=False)

    print("=== pack-level table (D-minus-fuzzy) ===", flush=True)
    sub = df[df.variant == "D-minus-fuzzy"]
    g = sub.groupby("tag").agg(
        sh_best=("sh_best", "mean"), sh_final=("sh_final", "mean"),
        sf_final=("sf_final", "mean"), diff=("diff", "mean"),
        absa_final=("absa_final", "mean"),
    ).round(4)
    print(g.to_string(), flush=True)

    print("\n=== mechanism: diff vs sf_final across every seed ===", flush=True)
    d = sub.dropna(subset=["diff", "sf_final"])
    print(f"n={len(d)}  corr(diff, sf_final)={np.corrcoef(d['sf_final'], d['diff'])[0,1]:+.3f}",
          flush=True)

    for family in FAMILIES:
        fam = df[(df.variant == "D-minus-fuzzy") & (df.tag.str.startswith(family))]
        fv = fam["tag"].str.replace(f"{family}/", "").astype(float).to_numpy()
        dv = fam["diff"].to_numpy(float)
        if len(dv) > 2:
            print(f"\ncorr(diff, {family})={np.corrcoef(fv, dv)[0,1]:+.3f}  (n={len(dv)})",
                  flush=True)


if __name__ == "__main__":
    main()