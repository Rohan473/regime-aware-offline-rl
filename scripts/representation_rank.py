"""Representation quality: effective rank / collapse (next_experiments.txt
information-efficiency + effective-rank items, frozen representations).

For each representation h_t on the TEST split we report how much of its
available linear dimensionality is actually used:

  effective_rank (participation ratio)  (sum lam)^2 / sum lam^2   of cov(h)
  rank_share      eff.rank / min(n, d)  - fraction of useable dims used
  top1 / top5     variance share of the largest 1 / 5 principal components
  k95             components needed to reach 95% of total variance
  scale           total variance (sum of eigenvalues; on standardized per-dim)
  slab            d (analyzed embedding dim) and n (test samples)

Comparison rows: raw 8 features, raw 20x8 window, and the learned hidden
states of DDR / TACR / D at seed 20260814.

Output: data/interpret/representation_rank.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interpret.extract import (
    D_CKPT, DDR_CKPT, SEED, TACR_CKPT, extract_d, extract_ddr, extract_tacr,
)
from src.interpret.targets import Z_COLUMNS, _naive, build_targets, align
from src.models.d.config import DConfig
from src.models.ddr.config import DDRConfig
from src.models.tacr.config import TACRConfig

OUT_DIR = ROOT / "data" / "interpret"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def eff_rank(x: np.ndarray) -> dict:
    xc = x - x.mean(axis=0)
    if xc.shape[0] < 2:
        raise ValueError("need >=2 samples")
    n = xc.shape[0]
    lam = np.linalg.eigvalsh(xc.T @ xc) / (n - 1)
    lam = np.maximum(lam, 0.0)
    lam.sort()
    lam = lam[::-1]
    tot = lam.sum()
    if tot <= 0:
        return dict(effective_rank=float("nan"), rank_share=float("nan"),
                    spectral_rank=float("nan"), top1=float("nan"), top5=float("nan"),
                    k95=float("nan"), scale=float(tot), n=n, d=x.shape[1])
    pr = tot**2 / np.square(lam).sum()
    p = lam / tot
    nz = p > 0
    sp = float(np.exp(-(p[nz] * np.log(p[nz])).sum()))  # idea 19: spectral entropy
    cum = np.cumsum(lam) / tot
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    return dict(
        effective_rank=float(pr),
        spectral_rank=sp,
        rank_share=float(pr / min(n, x.shape[1])),
        top1=float(lam[0] / tot),
        top5=float(lam[:5].sum() / tot if len(lam) >= 5 else 1.0),
        k95=int(k95),
        scale=float(tot),
        n=int(n),
        d=int(x.shape[1]),
    )


def main() -> None:
    targets = build_targets()
    sub = targets[~_naive(targets.index).duplicated(keep="first")].copy()
    sub.index = _naive(sub.index)

    feats = pd.read_parquet(ROOT / "data" / "processed" / "features_regimes.parquet")
    z = feats[Z_COLUMNS].to_numpy(dtype="float64")
    map_f = {d: i for i, d in enumerate(_naive(feats.index))}
    Xcoord = np.array([map_f[d] for d in sub.index])
    test_mask = (sub["split"] == "test").to_numpy()

    Xraw = z[Xcoord]
    pos = np.arange(len(feats))
    idx = pos[:, None] - 19 + np.arange(20)[None, :]
    valid = idx >= 0
    W = z[np.clip(idx, 0, None)] * valid[..., None].astype("float64")
    Xwin = W[Xcoord].reshape(len(sub), -1)

    ddr = extract_ddr(DDRConfig.from_yaml(), DDR_CKPT / f"s{SEED}" / "ddr_best.pt")
    tacr = extract_tacr(TACRConfig.from_yaml(), TACR_CKPT / f"s{SEED}" / "tacr_best.pt")
    d = extract_d(DConfig.from_yaml(), D_CKPT / f"s{SEED}" / "d_best.pt")

    reps = {"Raw 8 features (zd)": Xraw, "Raw 20x8 window (160d)": Xwin}
    for rep in (ddr, tacr, d):
        rsub, X = align(rep.dates, rep.H, sub)
        reps[rep.name] = X[(rsub["split"] == "test").to_numpy()]

    rows = []
    for name, Xt in reps.items():
        metric = {"representation": name}
        metric.update(eff_rank(Xt))
        rows.append(metric)
        print(f"{name:<26s} eff_rank={metric['effective_rank']:.2f} "
              f"spectral_rank={metric['spectral_rank']:.2f} "
              f"share={metric['rank_share']:.2f} top1={metric['top1']:.3f} "
              f"top5={metric['top5']:.3f} k95={metric['k95']} scale={metric['scale']:.1f} "
              f"(n={metric['n']}, d={metric['d']})")

    pd.DataFrame(rows).to_csv(OUT_DIR / "representation_rank.csv", index=False)
    print("\nrepresentation_rank.csv written to", OUT_DIR)


if __name__ == "__main__":
    main()