"""7.27.3 sweep analysis: does final-epoch failure track short_frac, not w_m?

Rolls BEST and FINAL checkpoints of the d_msw sweep (plus the comp-matched
and unweighted-32p anchor packs) on the test grid and computes per seed:
  test all-days Sharpe (best / final)
  final-epoch diff = mean(final) - mean(best)   (pack-level in eval.py; here
  also per seed for the regression)
  short_frac (best / final), mean |a|

Then tests the mechanism: across all (point, seed) rows, does diff track
SHORT_FRAC (per the user's hypothesis) rather than only w_m / composition
distance to old-4p?

Run: python scripts/d_msw_analysis.py
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
from src.models.d.eval import VARIANT_DIRS  # noqa: E402
from src.models.d.eval import BEST_NAME, FINAL_NAME  # noqa: E402
from src.models.d.eval import roll_test_preds  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
VARIANTS = ["D", "D-minus-fuzzy"]
GRID = [0.25, 0.40, 0.55, 0.70, 0.85]
ROWS = []


def add(pack_tag: str, w_m: float, base: Path, variants: list[str] = VARIANTS,
        seeds: list[int] | None = None) -> None:
    seeds = seeds or SEEDS
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
                    ROWS.append({"tag": pack_tag, "w_m": w_m, "variant": variant,
                                 "seed": s, "sh_best": sh, "sf_best": float(np.mean(a < 0)),
                                 "absa_best": float(np.mean(np.abs(a)))})
                else:
                    for row in ROWS:
                        if (row["tag"], row["variant"], row["seed"]) == (pack_tag, variant, s):
                            row["sh_final"] = sh
                            row["sf_final"] = float(np.mean(a < 0))
                            row["absa_final"] = float(np.mean(np.abs(a)))


def variant_key(variant: str) -> str:
    return {"d": "D", "d_minus_fuzzy": "D-minus-fuzzy"}[variant]


def w_m_of(pack: str):
    base = ROOT / "src/models/d/checkpoints" / pack
    return base


def main() -> None:
    ck = ROOT / "src/models/d/checkpoints"
    for w_m in GRID:
        add(f"msw/{w_m:g}", float(w_m), ck / "d_msw" / f"{w_m:g}")
    add("d_cmp", 0.25, ck / "d_cmp")                      # comp-matched anchor
    add("d_32p", 0.625, ck / "d_32p")                     # unweighted anchor
    add("old_4p", 0.25, ck, variants=["D-minus-fuzzy"])   # pre-7.11 reference

    df = pd.DataFrame(ROWS)
    df["diff"] = df["sh_final"] - df["sh_best"]
    df.to_csv(ck / "d_msw" / "sweep_analysis.csv", index=False)

    print("=== pack-level table (mean over seeds) ===", flush=True)
    for variant in VARIANTS:
        sub = df[df.variant == variant]
        g = sub.groupby("tag").agg(
            sh_best=("sh_best", "mean"), sf_final=("sf_final", "mean"),
            sf_best=("sf_best", "mean"), diff=("diff", "mean"),
            absa_final=("absa_final", "mean"),
        ).round(4)
        print(f"\n- {variant}", flush=True)
        print(g.to_string(), flush=True)

    print("\n=== point-table: w_m / short-frac / final-epoch diff (per-seed) ===", flush=True)
    pv = df.pivot_table(index=["w_m", "variant"], values=["diff", "sf_final"], aggfunc="mean").round(4)
    print(pv.to_string(), flush=True)

    # --- mechanism regression across ALL (point, seed) rows (both variants) ---
    print("\n=== mechanism test: diff vs short_frac across every seed ===", flush=True)
    d = df[["diff", "sf_final", "w_m", "tag"]].dropna()
    r_sf = np.corrcoef(d["sf_final"], d["diff"])[0, 1]
    r_wm = np.corrcoef(d["w_m"], d["diff"])[0, 1]
    n = len(d)
    print(f"n={n}  corr(diff, short_frac) = {r_sf:+.3f}   corr(diff, w_m) = {r_wm:+.3f}", flush=True)
    # within-point residual: does short_frac explain diff ABOVE w_m alone?
    # regress diff on w_m, correlate residuals with short_frac.
    A = np.column_stack([np.ones(n), d["w_m"]])
    beta, *_ = np.linalg.lstsq(A, d["diff"], rcond=None)
    resid = d["diff"] - A @ beta
    r_resid_sf = np.corrcoef(resid, d["sf_final"])[0, 1]
    print(f"corr(diff _, w_m residual, short_frac) = {r_resid_sf:+.3f}  "
          "(short_frac predictive ABOVE/ORTHOGONAL to w_m)", flush=True)

    print("\n=== area table with per-seed short-frac bins ===", flush=True)
    sb = d.groupby(pd.cut(d["sf_final"], bins=[-0.001, 0.001, 0.03, 0.10, 0.25, 0.50],
                         labels=["0", "0-3%", "3-10%", "10-25%", ">25%"])).agg(
        n=("diff", "size"), mean_diff=("diff", "mean"))
    print(sb.to_string(), flush=True)


if __name__ == "__main__":
    main()