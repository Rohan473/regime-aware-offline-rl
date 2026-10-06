"""Diagnostic: is OOD distance just another volatility proxy?

The exposure ablation found representation-space OOD distance to be the most
robust at-t input. This computes the association of OOD distance with
volatility, |next-day return| and drawdown state, on train+val, so a reader can
judge whether OOD is merely repackaging volatility.

No training. Output: data/interpret/rep_ood_diagnostic.csv + printed table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders import REPO_ROOT
from src.models.rep_lab.config import RepLabConfig
from src.models.rep_lab.offline_rl import load_offline_rep

OUT_DIR = ROOT / "data" / "interpret"
MARKETS = {
    "spy": (None, ""),
    "csi300": (REPO_ROOT / "data" / "processed_csi300_backup", "csi"),
    "nifty": (REPO_ROOT / "data" / "processed_nifty_backup", "nifty"),
}


def _reindex(processed_dir, dates):
    processed_dir = processed_dir or (REPO_ROOT / "data" / "processed")
    f = pd.read_parquet(processed_dir / "features_regimes.parquet")
    idx = f.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    f.index = idx
    d = dates.tz_localize(None) if getattr(dates, "tz", None) is not None else dates
    return f, d


def _ood(H_train, H_all):
    mu = H_train.mean(0)
    inv = np.linalg.pinv(np.cov(H_train, rowvar=False) + 1e-6 * np.eye(H_train.shape[1]))
    d = H_all - mu
    return np.sqrt((d @ inv * d).sum(1))


def main() -> None:
    rows = []
    for market, (pd_dir, tag) in MARKETS.items():
        cfg = RepLabConfig(); cfg.processed_dir, cfg.tag = pd_dir, tag
        data = load_offline_rep("predictive", cfg)
        tr, va, te = data.split("train"), data.split("val"), data.split("test")
        f, d = _reindex(pd_dir, data.dates)
        f = f.reindex(d)
        vol = f["z_realized_vol_20d"].to_numpy()
        dd = (f["close"] / f["close"].rolling(60).max() - 1.0).to_numpy()
        r = data.market_returns.numpy()
        absr = np.abs(r)
        ood = _ood(data.H[tr].numpy(), data.H.numpy())
        mask = tr | va  # train+val (in-sample to the policy's development phase)
        for name, x in (("vol", vol), ("abs_return", absr), ("drawdown60", dd)):
            m = mask & np.isfinite(x) & np.isfinite(ood)
            rows.append({"market": market, "pair": f"OOD~{name}",
                         "pearson": float(pearsonr(ood[m], x[m])[0]),
                         "spearman": float(spearmanr(ood[m], x[m])[0])})
        # OOD vs forward vol (5d) - does OOD anticipate future volatility?
        fwd = np.full(len(r), np.nan)
        for i in range(len(r) - 5):
            fwd[i] = np.std(r[i + 1:i + 6])
        m = mask & np.isfinite(fwd) & np.isfinite(ood)
        rows.append({"market": market, "pair": "OOD~fwd_vol_5",
                     "pearson": float(pearsonr(ood[m], fwd[m])[0]),
                     "spearman": float(spearmanr(ood[m], fwd[m])[0])})
        # vol~|return| for reference (vol is already correlated with |return|)
        m = mask & np.isfinite(absr) & np.isfinite(vol)
        rows.append({"market": market, "pair": "vol~abs_return (ref)",
                     "pearson": float(pearsonr(vol[m], absr[m])[0]),
                     "spearman": float(spearmanr(vol[m], absr[m])[0])})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rep_ood_diagnostic.csv", index=False)
    pd.set_option("display.width", 200)
    print("=== OOD-distance associations (train+val) ===")
    print(df.pivot_table(index="pair", columns="market", values="pearson").round(3).to_string())
    print("\nwrote rep_ood_diagnostic.csv under", OUT_DIR)


if __name__ == "__main__":
    main()