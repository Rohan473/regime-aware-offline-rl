"""Trend-day filter for Model C+ — "regime-first hybrid" (PROJECT_NOTES 7.20).

Composite: a_t = I(trend_pred=1) * sign_model(s_t) * |a_TACR(s_t)|, where
trend_pred is a second L2 logistic on the same 16 causal features predicting
Is_Trend = I(|r_{t+1}| > tau). On predicted-oscillation days the position is
FLAT (0.0).

Selection (VAL only, test untouched): (tau, C_trend) maximizing the mean
5-seed val composite Sharpe over
  tau in {0.003,0.004,0.005,0.0075,0.010}, C in {0.01,0.1,1,10}, cutoff 0.5.
The sign model is the C+ B model (16 features, C selected on val as in 7.18).

EM control is matched to the deployed exposure (I(trend)*|a_TACR|*m), so the
margin isolates the sign value on the days the filter keeps.

Run: python scripts/hybrid_trend_filter.py
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
CHECKPOINTS = ROOT / "src" / "models" / "tacr" / "checkpoints" / "tacr"
SPY_Z = ["z_ret_1d", "z_ret_5d", "z_ret_20d", "z_realized_vol_20d",
         "z_rsi_14", "z_macd_hist", "z_volume_zscore_20d", "z_bollinger_pos"]
C_GRID = [0.01, 0.1, 1.0, 10.0]
TAU_GRID = [0.003, 0.004, 0.005, 0.0075, 0.010]


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


def composite_sharpe(Xm, model, trend_model, acts, mm):
    """Sharpe of a = I(trend)*sign*|a| over a day set."""
    sig = model.predict(Xm)
    fire = trend_model.predict(Xm) == 1
    a = np.where(fire, sig * np.abs(acts), 0.0)
    return sharpe_ratio(a * mm), fire


def main() -> None:
    cfg = TACRConfig.from_yaml()
    data = load_tacr_data(20, exclude_policies=cfg.exclude_policies)

    feats = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_regimes.parquet")
    spy_z = feats[SPY_Z].reindex(data.dates).to_numpy(float)
    macro_raw = macro_features(feats)
    naive = data.dates.tz_localize(None)
    macro_z = np.column_stack([zscore_causal(macro_raw[c]) for c in MACRO_FEATURES])
    macro_z = macro_z[np.searchsorted(macro_raw.index, naive)]
    Fb = np.concatenate([spy_z, macro_z], 1)
    ret = data.market_returns.numpy()
    y = _signed(ret)
    ok = np.isfinite(Fb).all(1)

    dates = naive
    tr = dates <= pd.Timestamp("2018-12-31")
    va = (dates > pd.Timestamp("2018-12-31")) & (dates <= pd.Timestamp("2020-12-31"))
    te = dates > pd.Timestamp("2020-12-31")

    # C+ sign model (16 features), C on val
    sign_model, sign_C = fit(Fb[tr & ok], y[tr & ok], Fb[va & ok], y[va & ok])

    # preload per-seed magnitude arrays for val + test
    def mags(mask, name):
        out = {}
        for seed in SEEDS:
            c = TACRConfig.from_yaml(); c.checkpoint_dir = CHECKPOINTS / TAG / f"s{seed}"
            m, am, _ = load_checkpoint(c.checkpoint_dir / "tacr_best.pt", c)
            dts = data.dates[mask]
            acts = roll_actions(m, c, data, dts, c.rtg_target, action_model=am, bcq_phi=c.bcq_phi)
            out[seed] = acts[data.dates.get_indexer(dts)]
        return out
    val_mags = mags(va, "val")
    test_mags = mags(te, "test")
    Xv = Fb[va & ok]
    Xt = Fb[te & ok]
    ret_v = ret[va & ok]
    ret_t = ret[te & ok]
    # (magnitude arrays are per valid test/val day; features ok on those days)
    v_idx = np.where(va & ok)[0]
    t_idx = np.where(te & ok)[0]

    # --- select (tau, C_trend) on val ---
    best = (None, -1.0, None, None)
    for tau in TAU_GRID:
        yt = (np.abs(ret) > tau).astype(int)
        for C in C_GRID:
            tm = LogisticRegression(penalty="l2", C=C, max_iter=2000).fit(
                Fb[tr & ok], yt[tr & ok])
            shp = []
            for seed in SEEDS:
                acts = val_mags[seed][np.searchsorted(v_idx, v_idx)]  # aligned to Xv rows
                # val_mags aligned to val dates; Xv rows are va&ok subset
                acts_sub = val_mags[seed][np.isin(np.where(va)[0], v_idx)]
                s, _ = composite_sharpe(Xv, sign_model, tm, acts_sub, ret_v)
                shp.append(s)
            msh = float(np.mean(shp))
            if msh > best[1]:
                best = (tm, msh, tau, C)
    trend_model, val_sharpe, best_tau, best_C = best
    print(f"selected tau={best_tau} C_trend={best_C} | val composite Sharpe {val_sharpe:.4f}")

    # --- test evaluation ---
    margins, sharpes, trend_rates, wins = [], [], [], []
    for seed in SEEDS:
        acts_sub = test_mags[seed][np.isin(np.where(te)[0], t_idx)]
        s, fire = composite_sharpe(Xt, sign_model, trend_model, acts_sub, ret_t)
        em = sharpe_ratio(np.where(fire, np.abs(acts_sub), 0.0) * ret_t)
        margins.append(s - em)
        sharpes.append(s)
        trend_rates.append(float(fire.mean()))
    print(f"\ntrend-day rate on test: {np.mean(trend_rates):.3f}")
    print(f"filtered pack mean test Sharpe {np.mean(sharpes):.4f} +- {np.std(sharpes):.4f} "
          f"(C+ unfiltered 1.146) | margins {np.mean(margins):+.4f} | wins {int(sum(np.array(margins)>0))}/5")
    print(f"BAR (mean > 1.146 & wins >= 3/5): "
          f"{'PASS' if (np.mean(sharpes) > 1.146 and int(sum(np.array(margins)>0)) >= 3) else 'FAIL'}")
    for seed, s, mg in zip(SEEDS, sharpes, margins):
        print(f"  s{seed}: test {s:+.4f} margin {mg:+.4f}")


if __name__ == "__main__":
    main()