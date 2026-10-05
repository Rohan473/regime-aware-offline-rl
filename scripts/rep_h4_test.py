"""Pre-specified H4 follow-up test (frozen design, one run, no search).

Question: does policy divergence predict utility conditional on representation
and regime? The algorithm-level r=.935 (n=4) is descriptive; this is the
granular test using existing seed-level data only (NO new training).

PRE-REGISTERED SPEC (do not vary after this):
  Stage 2 (secondary): seed-level divergence -> within-representation utility,
      n = 48 (4 reps x 4 algos x 3 seeds),
      U = b0 + b1*D + C(rep) + C(algo) + e.  Primary coefficient b1.
      Variant: same model with JS divergence in place of D (pre-specified
      alternative divergence measure, not a search).
  Stage 1 (primary): regime-specific utility,
      n = 144 (48 units x 3 regimes),
      U_g = b0 + b1*D + b2*G + b3*(D x G) + C(rep) + C(algo) + e,
      clustered by (rep, algo, seed) unit.  Primary coefficient b3 (D x G):
      does the divergence-utility slope differ across bull/bear/crisis?

Output: data/interpret/rep_h4.csv (all coefficients), printed summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "interpret"
ALGOS = ["A2C", "BC", "IQL", "CQL"]
REPS = ["raw", "auto", "predictive", "contrastive"]


def _interaction_figure(long, model, path) -> None:
    """Utility vs divergence by regime with per-regime fitted lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = ["bull", "bear", "crisis"]
    marks = {"bull": "o", "bear": "s", "crisis": "^"}
    # detect the reference level (the one with no D:C interaction term)
    inter = [t for t in model.params.index if t.startswith("D:C")]
    levels = {t.split("[T.")[1].rstrip("]") for t in inter}
    ref = (set(regimes) - levels).pop()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    lo, hi = long["D"].min(), long["D"].max()
    xs = np.linspace(lo, hi, 20)
    base = long["D"].median()
    for g in regimes:
        sub = long[long["regime"] == g]
        ax.scatter(sub["D"], sub["U_g"], s=16, alpha=0.6, marker=marks[g], label=g)
        coef = 0.0 if g == ref else model.params.get(f"D:C(regime)[T.{g}]", 0.0)
        slope = model.params["D"] + coef  # absolute per-regime slope (ref = bear)
        intercept = 0.0 if g == ref else model.params.get(f"C(regime)[T.{g}]", 0.0)
        ax.plot(xs, intercept + slope * (xs - base), lw=1.6,
                label=f"{g} slope {slope:+.2f}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("policy divergence from behavior")
    ax.set_ylabel("regime-conditioned Sharpe (regime-mean adjusted)")
    ax.set_title(f"Divergence x regime -> utility (slopes, ref={ref})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


def main() -> None:
    diag = pd.read_csv(OUT_DIR / "rep_rl_diagnostics.csv")
    off = pd.read_csv(OUT_DIR / "rep_offline_rl.csv")
    off = off[off["rep"].isin(REPS) & off["algo"].isin(ALGOS)]

    div = (diag[diag["metric"] == "divergence"][["rep", "algo", "seed", "value"]]
           .rename(columns={"value": "D"}))
    js = (diag[diag["metric"] == "js_divergence"][["rep", "algo", "seed", "value"]]
          .rename(columns={"value": "JS"}))
    reg = (diag[diag["metric"] == "sharpe_regime"]
           .pivot_table(index=["rep", "algo", "seed"], columns="group", values="value")
           .reset_index())
    U = off[["rep", "algo", "seed", "test_sharpe"]]

    out = []

    # ---- Stage 2 (secondary): seed-level divergence -> utility, n=48 ----
    b = U.merge(div, on=["rep", "algo", "seed"]).dropna(subset=["D", "test_sharpe"])
    m = smf.ols("test_sharpe ~ D + C(rep) + C(algo)", data=b).fit(
        cov_type="HC3")
    out.append(dict(stage=2, term="D (seed-level divergence)", coef=m.params["D"],
                    se=m.bse["D"], t=m.tvalues["D"], p=m.pvalues["D"],
                    n=len(b), r2=m.rsquared, note="controlling rep + algo"))
    bj = U.merge(js, on=["rep", "algo", "seed"]).dropna(subset=["JS", "test_sharpe"])
    mj = smf.ols("test_sharpe ~ JS + C(rep) + C(algo)", data=bj).fit(cov_type="HC3")
    out.append(dict(stage=2, term="JS (action-distribution divergence)",
                    coef=mj.params["JS"], se=mj.bse["JS"], t=mj.tvalues["JS"],
                    p=mj.pvalues["JS"], n=len(bj), r2=mj.rsquared,
                    note="pre-specified alternate divergence measure"))

    # ---- Stage 1 (primary): divergence x regime -> utility, n=144 ----
    long = (reg.melt(id_vars=["rep", "algo", "seed"], var_name="regime",
                     value_name="U_g")
            .dropna(subset=["U_g"])
            .merge(div, on=["rep", "algo", "seed"]))
    long["unit"] = long["rep"] + "|" + long["algo"] + "|" + long["seed"].astype(str)
    m1 = smf.ols("U_g ~ D + C(regime) + D:C(regime) + C(rep) + C(algo)",
                 data=long).fit(cov_type="cluster", cov_kwds={"groups": long["unit"]})
    for term, coef in m1.params.items():
        if not term.startswith("D:C"):
            continue
        out.append(dict(stage=1, term=term, coef=coef, se=m1.bse[term],
                        t=m1.tvalues[term], p=m1.pvalues[term], n=len(long),
                        r2=m1.rsquared, note="cluster SE by (rep,algo,seed)"))
    # joint test of the whole D x regime interaction
    wt = m1.wald_test_terms(scalar=True).table
    row = wt.loc["D:C(regime)"]
    out.append(dict(stage=1, term="joint D:C(regime) (chi2 test)",
                    coef=float(row["statistic"]), se=np.nan, t=np.nan,
                    p=float(row["pvalue"]), n=len(long), r2=m1.rsquared,
                    note=f"df={int(row['df_constraint'])}, cluster SE"))

    df = pd.DataFrame(out)
    df.to_csv(OUT_DIR / "rep_h4.csv", index=False)

    _interaction_figure(long, m1, ROOT / "paper" / "rep_divergence_regime.png")

    pd.set_option("display.width", 220)
    print("=== PRE-SPECIFIED H4 TEST (one run, no specification search) ===")
    print(df.round(4).to_string(index=False))
    print("\nKey result Stage 2: D coef b1 (controlling rep+algo), n=48: "
          f"{m.params['D']:+.4f}, p={m.pvalues['D']:.4f}")
    print("Key result Stage 1: divergence x regime interaction (joint chi2): "
          f"chi2={float(row['statistic']):.1f} (df={int(row['df_constraint'])}), "
          f"p={float(row['pvalue']):.2e}")
    b1 = m1.params["D"]
    _inter = [t for t in m1.params.index if t.startswith("D:C")]
    _ref = ({"bull", "bear", "crisis"} - {t.split("[T.")[1].rstrip("]") for t in _inter}).pop()
    print(f"  reference regime = {_ref}; D main effect (ref slope) = {b1:+.4f}")
    for g in ("bull", "bear", "crisis"):
        if g == _ref:
            print(f"  absolute {g:6s} slope = {b1:+.4f}  (reference)")
        else:
            coef = m1.params.get(f"D:C(regime)[T.{g}]", 0.0)
            pv = m1.pvalues.get(f"D:C(regime)[T.{g}]", float("nan"))
            print(f"  absolute {g:6s} slope = {b1 + coef:+.4f}  "
                  f"(vs {_ref}: {coef:+.4f}, p={pv:.4f})")
    print("\nNote: these are OLS associations, not causal; the design is frozen.")
    print("wrote rep_h4.csv under", OUT_DIR)


if __name__ == "__main__":
    main()