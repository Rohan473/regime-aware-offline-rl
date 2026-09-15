"""7.27.2 review: pack-variance + bull-mechanism comparison across the three
D packs (old 4-policy / 32p unweighted / 32p composition-matched).

Answers two reviewer questions without a new experiment:
 1) Is "recovers ~95% of its old level" a false precision — is 0.837 vs 0.881
    inside the seed spread of either pack?
 2) Is the comp-matched Sharpe recovery the SAME long-bull participation
    mechanism as the original 4-policy run (7.9's reading), or did the 32p
    pool's residual influence change HOW long exposure is taken?
Run: python scripts/d_cmp_review.py
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
from src.models.d.eval import BEST_NAME, VARIANT_DIRS, roll_test_preds  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
VARIANTS = ["D", "D-minus-fuzzy"]
PACKS = {
    "old_4p": ROOT / "src/models/d/checkpoints",
    "d_32p": ROOT / "src/models/d/checkpoints/d_32p",
    "d_cmp": ROOT / "src/models/d/checkpoints/d_cmp",
}
N = 5


def se(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def mean_se(x: np.ndarray) -> str:
    v = float(np.mean(x))
    return f"{v:.4f} +- {float(np.std(x, ddof=1)):.4f} (se {se(x):.4f})"


def main() -> None:
    cfg = DConfig.from_yaml()
    out = {}
    for pack, base in PACKS.items():
        for variant in VARIANTS:
            rows = []
            for s in SEEDS:
                c = cfg.__class__.from_yaml()
                c.checkpoint_dir = base
                preds = roll_test_preds(c, variant, s, BEST_NAME, data_full=None)
                a = preds["action"].to_numpy()
                m = preds["market_ret"].to_numpy()
                r = preds["ret"].to_numpy()
                regime = preds["regime"].to_numpy()
                bull = regime == "bull"
                rows.append(
                    {
                        "seed": s,
                        "sh_all": sharpe_ratio(r),
                        "em_all": sharpe_ratio(np.abs(a) * m),
                        "sh_bull": sharpe_ratio(r[bull]),
                        "mean_a_bull": float(np.mean(a[bull])),
                        "long_frac_bull": float(np.mean(a[bull] > 0)),
                        "short_frac": float(np.mean(a < 0)),
                        "mean_abs_a": float(np.mean(np.abs(a))),
                        "corr_a_m": float(np.corrcoef(a, m)[0, 1]),
                    }
                )
            df = pd.DataFrame(rows)
            out[(pack, variant)] = df
            print(f"== {pack}/{variant} ==", flush=True)
            print(f"  all-days Sharpe : {mean_se(df['sh_all'].to_numpy())}")
            print(f"  EM (|a|*m)      : {mean_se(df['em_all'].to_numpy())}")
            print(f"  bull Sharpe     : {mean_se(df['sh_bull'].to_numpy())}")
            print(f"  mean a|bull     : {mean_se(df['mean_a_bull'].to_numpy())}")
            print(f"  long% |bulldays : {mean_se(df['long_frac_bull'].to_numpy())}")
            print(f"  short_frac      : {mean_se(df['short_frac'].to_numpy())}")
            print(f"  mean |a|        : {mean_se(df['mean_abs_a'].to_numpy())}")
            print(f"  corr(a,m)       : {mean_se(df['corr_a_m'].to_numpy())}")
            print(f"  per-seed all    : {[round(x,4) for x in df['sh_all'].tolist()]}")
            print(f"  per-seed margin : {[round(x,4) for x in (df['sh_all']-df['em_all']).tolist()]}")

    print("\n== pack-mean difference tests (all-days Sharpe, best-val) ==", flush=True)
    for variant in VARIANTS:
        old = out[("old_4p", variant)]["sh_all"].to_numpy()
        cmp_ = out[("d_cmp", variant)]["sh_all"].to_numpy()
        d_global = (float(old.mean() - cmp_.mean()))
        pooled = float(np.sqrt(np.std(old, ddof=1) ** 2 + np.std(cmp_, ddof=1) ** 2))
        print(f"{variant}: old {old.mean():.4f} vs cmp {cmp_.mean():.4f} | "
              f"gap {d_global:+.4f} vs pooled seed-std {pooled:.4f} "
              f"(i.e. gap / seed-std = {abs(d_global)/pooled if pooled else 0:.2f}x)")

    # EM-margin basis: is the inside-pack seed spread comparable to the gap?
    print("\n== inside-pack spread vs between-pack gap ==", flush=True)
    for pack, variant in ([("old_4p", "D-minus-fuzzy"), ("d_cmp", "D-minus-fuzzy"),
                           ("old_4p", "D"), ("d_cmp", "D")]):
        a = out[(pack, variant)]["sh_all"].to_numpy()
        print(f"{pack}/{variant}: mean {a.mean():.4f} std {float(np.std(a,ddof=1)):.4f} "
              f"min {a.min():.4f} max {a.max():.4f}")


if __name__ == "__main__":
    main()