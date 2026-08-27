"""Sentiment (SKEW) as the 17th feature of the C+ sign model (7.22).

Builds BOTH the 16-feature (reference C+) and 17-feature (+ sentiment_skew)
matrices in one protocol and compares on the test window:
  ref16 : 8 SPY z + 8 macro z
  s17   : ref16 + sentiment_skew (causal z of the CBOE SKEW level)
Sensitivity variant: s17c replaces the skew LEVEL with its 1d change.

Same discipline as 7.18: L2 LogisticRegression, C on val from
{0.01,0.1,1,10}, TEST untouched; fix_a_nr7 magnitude; margins vs the
hybrid's own EM. 5-seed pack.

Run: python scripts/hybrid_sentiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders import REPO_ROOT  # noqa: E402
from src.data.macro_factors import MACRO_FEATURES, macro_features, zscore_causal  # noqa: E402
from src.eval.regime_eval import sharpe_ratio  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data  # noqa: E402
from src.models.tacr.eval import load_checkpoint, roll_actions  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
TAG = "fix_a_nr7"
C_GRID = [0.01, 0.1, 1.0, 10.0]
CHECKPOINTS = ROOT / "src" / "models" / "tacr" / "checkpoints" / "tacr"
SPY_Z = ["z_ret_1d", "z_ret_5d", "z_ret_20d", "z_realized_vol_20d",
         "z_rsi_14", "z_macd_hist", "z_volume_zscore_20d", "z_bollinger_pos"]


def _signed(m):
    y = np.sign(m)
    y[y == 0] = 1
    return y


def fit(Xtr, ytr, Xv, yv):
    best = (None, -1.0, None)
    for C in C_GRID:
        m = LogisticRegression(penalty="l2", C=C, max_iter=2000).fit(Xtr, ytr)
        acc = float((m.predict(Xv) == yv).mean())
        if acc > best[1]:
            best = (m, acc, C)
    return best[0], best[2]


def main() -> None:
    cfg = TACRConfig.from_yaml()
    data = load_tacr_data(20, exclude_policies=cfg.exclude_policies)

    feats = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_regimes.parquet")
    spy_z = feats[SPY_Z].reindex(data.dates).to_numpy(float)
    raw = macro_features(feats)                      # 8 macro + sentiment_skew (raw)
    naive = data.dates.tz_localize(None)
    macro_z = np.column_stack([zscore_causal(raw[c]) for c in MACRO_FEATURES])
    macro_z = macro_z[np.searchsorted(raw.index, naive)]
    skew = zscore_causal(raw["sentiment_skew"])
    skew = skew.to_numpy()[np.searchsorted(raw.index, naive)]
    skew_chg = zscore_causal(raw["sentiment_skew"].diff())
    skew_chg = skew_chg.to_numpy()[np.searchsorted(raw.index, naive)]

    F16 = np.concatenate([spy_z, macro_z], 1)                     # (T, 16)
    F17 = np.concatenate([F16, skew[:, None]], 1)                 # (T, 17) level
    F17c = np.concatenate([F16, skew_chg[:, None]], 1)            # (T, 17) change
    ret = data.market_returns.numpy()
    y = _signed(ret)

    dates = naive
    tr = dates <= pd.Timestamp("2018-12-31")
    va = (dates > pd.Timestamp("2018-12-31")) & (dates <= pd.Timestamp("2020-12-31"))
    te = dates > pd.Timestamp("2020-12-31")

    def mags(mask):
        out = {}
        for seed in SEEDS:
            c = TACRConfig.from_yaml(); c.checkpoint_dir = CHECKPOINTS / TAG / f"s{seed}"
            m, am, _ = load_checkpoint(c.checkpoint_dir / "tacr_best.pt", c)
            dts = data.dates[mask]
            acts = roll_actions(m, c, data, dts, c.rtg_target, action_model=am, bcq_phi=c.bcq_phi)
            out[seed] = acts[data.dates.get_indexer(dts)]
        return out

    val_mags = mags(va)
    test_mags = mags(te)

    print(f"{'model':7s} {'C':>4s} {'acc':>6s} {'short':>6s} {'margin':>8s} {'wins':>5s} {'sharpe':>8s}")
    for name, X in [("ref16", F16), ("s17", F17), ("s17c", F17c)]:
        ok = np.isfinite(X).all(1)
        m, C = fit(X[tr & ok], y[tr & ok], X[va & ok], y[va & ok])
        margins, sharpes, shorts, accs = [], [], [], []
        for seed in SEEDS:
            acts = test_mags[seed][np.isin(np.where(te)[0], np.where(te & ok)[0])]
            Xt = X[te & ok]
            sig = m.predict(Xt)
            a = sig * np.abs(acts)
            s = sharpe_ratio(a * ret[te & ok])
            em = sharpe_ratio(np.abs(acts) * ret[te & ok])
            margins.append(s - em)
            sharpes.append(s)
            shorts.append(float((sig < 0).mean()))
            accs.append(float((sig == _signed(ret[te & ok])).mean()))
        print(f"{name:7s} {C:>4g} {np.mean(accs):>6.4f} {np.mean(shorts):>6.4f} "
              f"{np.mean(margins):>8.4f} {int(sum(np.array(margins)>0)):>5d} {np.mean(sharpes):>8.4f}")
    print("\nBAR for s17: mean Sharpe > 1.146 (ref16) AND wins >= 3/5 — applied from the table above.")


if __name__ == "__main__":
    main()