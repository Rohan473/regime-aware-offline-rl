"""β sweep analysis: does lowering β stabilize the 32-policy unweighted model?

Readout per (β, variant, seed): BEST and FINAL all-days Sharpe, final-epoch
diff, short_frac, mean |a|. Compares against d_32p anchor.

Run: python scripts/d_beta_analysis.py
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
GRID = [0.5, 1.0, 1.5, 2.0, 3.0]
ROWS = []


def add(pack_tag: str, beta: float, base: Path, variants: list[str], seeds: list[int]) -> None:
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
                m = p["market_ret"].to_numpy()
                r = p["ret"].to_numpy()
                sh = sharpe_ratio(r)
                if kind == "best":
                    ROWS.append({"tag": pack_tag, "beta": beta, "variant": variant,
                                 "seed": s, "sh_best": sh, "sf_best": float(np.mean(a < 0)),
                                 "absa_best": float(np.mean(np.abs(a)))})
                else:
                    for row in ROWS:
                        if (row["tag"], row["variant"], row["seed"]) == (pack_tag, variant, s):
                            row["sh_final"] = sh
                            row["sf_final"] = float(np.mean(a < 0))
                            row["absa_final"] = float(np.mean(np.abs(a)))


def main() -> None:
    ck = ROOT / "src/models/d/checkpoints"
    for beta in GRID:
        add(f"beta/{beta:g}", beta, ck / "d_beta" / f"{beta:g}", VARIANTS, SEEDS)
    add("d_32p", 3.0, ck / "d_32p", VARIANTS, SEEDS)

    df = pd.DataFrame(ROWS)
    df["diff"] = df["sh_final"] - df["sh_best"]
    df.to_csv(ck / "d_beta" / "beta_analysis.csv", index=False)

    print("=== pack-level table (D-minus-fuzzy) ===", flush=True)
    sub = df[df.variant == "D-minus-fuzzy"]
    g = sub.groupby("tag").agg(
        sh_best=("sh_best", "mean"), sh_final=("sh_final", "mean"),
        sf_best=("sf_best", "mean"), sf_final=("sf_final", "mean"),
        diff=("diff", "mean"), absa_best=("absa_best", "mean"),
        absa_final=("absa_final", "mean"),
    ).round(4)
    print(g.to_string(), flush=True)

    print("\n=== per-seed detail (D-minus-fuzzy) ===", flush=True)
    print(sub[["tag", "seed", "sh_best", "sh_final", "diff", "sf_best", "sf_final"]].round(4).to_string(index=False), flush=True)

    print("\n=== mechanism: diff vs short_frac (FINAL) across every seed ===", flush=True)
    d = df[df.variant == "D-minus-fuzzy"].dropna(subset=["diff", "sf_final"])
    sf = d["sf_final"].to_numpy(float)
    dv = d["diff"].to_numpy(float)
    beta_v = d["beta"].to_numpy(float)
    print(f"n={len(d)}  corr(diff, sf_final)={np.corrcoef(sf,dv)[0,1]:+.3f}  "
          f"corr(diff, beta)={np.corrcoef(beta_v,dv)[0,1]:+.3f}", flush=True)


if __name__ == "__main__":
    main()