"""Mechanism synthesis + statistical presentation.

Turns the experiment tables into the paper's central quantitative claims:

  A. bootstrap confidence intervals for the key comparisons (mean +/- SD + 95% CI)
  B. algorithm x representation interaction: per-algorithm Range and CV of
     utility across representations, plus a two-way variance decomposition
  C. behavior divergence <-> representation sensitivity association
  D. policy divergence x regime -> utility

The proposed chain:
  Representation -> policy divergence -> representation sensitivity ->
  regime-dependent utility

Inputs (data/interpret/): rep_offline_rl.csv, rep_rl_diagnostics.csv,
rep_scaling.csv.
Outputs: rep_bootstrap_ci.csv, rep_mechanism.csv, rep_mechanism.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "interpret"
REPS = ["raw", "auto", "predictive", "contrastive"]
ALGOS = ["A2C", "BC", "IQL", "CQL"]
N_BOOT = 10000


def boot_ci(x, n: int = N_BOOT, seed: int = 0) -> tuple[float, float, float, float]:
    """(mean, sd, 2.5th pct, 97.5th pct) of the bootstrap mean."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan,) * 4
    if x.size == 1:
        return float(x[0]), 0.0, float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(x.std(ddof=1)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def diff_ci(a, b, n: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI for mean(a) - mean(b) (unpaired)."""
    a = np.asarray(a, float)[np.isfinite(a)]
    b = np.asarray(b, float)[np.isfinite(b)]
    rng = np.random.default_rng(seed)
    d = (rng.choice(a, (n, a.size), replace=True).mean(axis=1)
         - rng.choice(b, (n, b.size), replace=True).mean(axis=1))
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    off = pd.read_csv(OUT_DIR / "rep_offline_rl.csv")
    diag = pd.read_csv(OUT_DIR / "rep_rl_diagnostics.csv")
    scal = pd.read_csv(OUT_DIR / "rep_scaling.csv")

    off = off[off["rep"].isin(REPS) & off["algo"].isin(ALGOS)]
    ci_rows = []

    def add_ci(family, condition, metric, x):
        m, sd, lo, hi = boot_ci(x)
        ci_rows.append(dict(family=family, condition=condition, metric=metric,
                            n=int(np.isfinite(x).sum()), mean=m, sd=sd, ci_lo=lo, ci_hi=hi))

    # ---- A. rep x algo matrix with bootstrap CI ----
    cell_mean = {}
    for rep in REPS:
        for algo in ALGOS:
            x = off[(off["rep"] == rep) & (off["algo"] == algo)]["test_sharpe"].to_numpy()
            add_ci("rep_x_algo", f"{rep}|{algo}", "test_sharpe", x)
            cell_mean[(rep, algo)] = float(np.nanmean(x))

    # key contrasts (unpaired, per representation, pooled over seeds)
    contrast_rows = []
    for rep in REPS:
        a2c = off[(off["rep"] == rep) & (off["algo"] == "A2C")]["test_sharpe"].to_numpy()
        for algo in ("BC", "IQL", "CQL"):
            x = off[(off["rep"] == rep) & (off["algo"] == algo)]["test_sharpe"].to_numpy()
            d, lo, hi = diff_ci(x, a2c)
            contrast_rows.append(dict(rep=rep, contrast=f"{algo}-A2C", diff=d, ci_lo=lo, ci_hi=hi))

    # feature/latent scaling key cells
    for axis in ("feature", "dim"):
        for obj in ("auto", "predictive", "contrastive"):
            s = scal[(scal["axis"] == axis) & (scal["objective"] == obj)]
            for size, g in s.groupby("size"):
                add_ci(f"{axis}_scaling", f"{obj}|{int(size)}", "supervised_sharpe",
                       g["supervised_sharpe"].to_numpy())

    pd.DataFrame(ci_rows).to_csv(OUT_DIR / "rep_bootstrap_ci.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(OUT_DIR / "rep_contrasts.csv", index=False)

    # ---- B. interaction / sensitivity ----
    mat = pd.DataFrame([[cell_mean[(r, a)] for a in ALGOS] for r in REPS],
                       index=REPS, columns=ALGOS)  # reps x algos
    per_algo = []
    for algo in ALGOS:
        vals = mat[algo].to_numpy()
        rng = vals.max() - vals.min()
        cv = vals.std(ddof=1) / abs(vals.mean()) if vals.mean() != 0 else np.nan
        per_algo.append(dict(algo=algo, mean=vals.mean(), sd=vals.std(ddof=1),
                             range=float(rng), cv=float(cv)))
    per_algo = pd.DataFrame(per_algo)

    # two-way variance decomposition of the 4x4 cell-mean matrix
    grand = mat.to_numpy().mean()
    ss_algo = mat.shape[0] * ((mat.mean(axis=0) - grand) ** 2).sum()
    ss_rep = mat.shape[1] * ((mat.mean(axis=1) - grand) ** 2).sum()
    ss_total = ((mat.to_numpy() - grand) ** 2).sum()
    ss_inter = ss_total - ss_algo - ss_rep
    decomp = dict(ss_total=float(ss_total), pct_algo=float(100 * ss_algo / ss_total),
                  pct_rep=float(100 * ss_rep / ss_total),
                  pct_interaction=float(100 * ss_inter / ss_total))

    # ---- C. divergence <-> representation sensitivity ----
    div = (diag[diag["metric"] == "divergence"]
           .groupby(["rep", "algo"])["value"].mean().reset_index())
    div_a = div.groupby("algo")["value"].mean()
    mech = per_algo.merge(div_a.rename("mean_divergence"), left_on="algo", right_index=True)
    from scipy.stats import pearsonr, spearmanr
    r_p = pearsonr(mech["mean_divergence"], mech["range"])
    r_s = spearmanr(mech["mean_divergence"], mech["range"])
    # divergence vs |utility - mean utility| at the (rep, algo) level
    div_cell = div.copy()
    div_cell["u"] = [cell_mean[(r, a)] for r, a in zip(div_cell["rep"], div_cell["algo"])]
    div_cell["abs_dev"] = div_cell.groupby("algo")["u"].transform(lambda s: (s - s.mean()).abs())
    r_cell = pearsonr(div_cell["value"], div_cell["abs_dev"])

    # ---- D. divergence x regime -> utility ----
    d = diag[diag["metric"] == "divergence"][["rep", "algo", "seed", "value"]].rename(
        columns={"value": "divergence"})
    reg = (diag[diag["metric"] == "sharpe_regime"]
           .pivot_table(index=["rep", "algo", "seed"], columns="group", values="value")
           .reset_index())
    merged = d.merge(reg, on=["rep", "algo", "seed"])
    merged.to_csv(OUT_DIR / "rep_divergence_regime.csv", index=False)
    regime_corr = {}
    for g in ("bull", "bear", "crisis"):
        if g in merged:
            v = merged[["divergence", g]].dropna()
            regime_corr[g] = float(pearsonr(v["divergence"], v[g])[0]) if len(v) > 2 else np.nan

    mech["regime_corr_bull"] = regime_corr.get("bull", np.nan)
    mech.to_csv(OUT_DIR / "rep_mechanism.csv", index=False)

    # ---- report ----
    pd.set_option("display.width", 240)
    print("=== A. representation x algorithm test Sharpe (mean +/- sd [95% CI]) ===")
    for rep in REPS:
        line = f"{rep:12s}"
        for algo in ALGOS:
            r = next(x for x in ci_rows if x["condition"] == f"{rep}|{algo}")
            line += f"  {algo} {r['mean']:.3f}+/-{r['sd']:.3f} [{r['ci_lo']:.2f},{r['ci_hi']:.2f}]"
        print(line)
    print("\n=== key contrasts vs A2C ===")
    print(pd.DataFrame(contrast_rows).round(3).to_string(index=False))

    print("\n=== B. representation sensitivity per algorithm ===")
    print(per_algo.round(3).to_string(index=False))
    print(f"two-way decomposition: algorithm {decomp['pct_algo']:.1f}% | "
          f"representation {decomp['pct_rep']:.1f}% | interaction {decomp['pct_interaction']:.1f}%")

    print("\n=== C. divergence vs representation sensitivity ===")
    print(mech.round(3).to_string(index=False))
    print(f"  across algorithms: pearson(divergence, range)={r_p[0]:+.3f} (p={r_p[1]:.3f}), "
          f"spearman={r_s[0]:+.3f}")
    print(f"  at (rep,algo) level: pearson(divergence, |U-meanU|)={r_cell[0]:+.3f} (p={r_cell[1]:.3f})")

    print("\n=== D. divergence vs regime-conditioned Sharpe ===")
    for g, c in regime_corr.items():
        print(f"  {g:6s}: pearson(divergence, Sharpe)={c:+.3f}")

    # ---- figure ----
    try:
        _figure(mat, mech, merged, OUT_DIR / "rep_mechanism.png")
        print("\nwrote rep_mechanism.png")
    except Exception as exc:  # noqa: BLE001
        print(f"[figure skipped] {type(exc).__name__}: {exc}")
    print("wrote rep_bootstrap_ci.csv, rep_contrasts.csv, rep_mechanism.csv, rep_divergence_regime.csv")


def _figure(mat, mech, merged, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    im = ax.imshow(mat[ALGOS].to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ALGOS)), ALGOS)
    ax.set_yticks(range(len(mat.index)), mat.index)
    for i in range(mat.shape[0]):
        for j in range(len(ALGOS)):
            ax.text(j, i, f"{mat[ALGOS].to_numpy()[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=9)
    ax.set_title("representation x algorithm\n(test Sharpe)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    ax.scatter(mech["mean_divergence"], mech["range"], s=60)
    for _, r in mech.iterrows():
        ax.annotate(r["algo"], (r["mean_divergence"], r["range"]),
                    textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("mean policy divergence from behavior")
    ax.set_ylabel("representation sensitivity (Sharpe range)")
    ax.set_title("C. divergence vs representation sensitivity")

    ax = axes[2]
    for g, mk in (("bull", "o"), ("bear", "s"), ("crisis", "^")):
        if g in merged:
            ax.scatter(merged["divergence"], merged[g], s=18, marker=mk, alpha=0.7, label=g)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("policy divergence from behavior")
    ax.set_ylabel("regime-conditioned Sharpe")
    ax.set_title("D. divergence x regime -> utility")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
