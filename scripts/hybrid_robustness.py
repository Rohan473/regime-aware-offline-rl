"""Robustness check for Model C+ — does the macro sign edge generalize to
shifted test periods? (PROJECT_NOTES 7.19)

Sign models A_match (8 SPY) vs B (8 SPY + 8 macro) are re-fitted on each
shifted split (C selected on that shift's val); the fix_a_nr7 magnitude is
held fixed. The B-vs-A_match margin delta on each period isolates the macro
sign contribution (identical magnitude), which is the claim under test.

Shifts:
  orig : train<=2018, val 2019-20, test 2021-24   (reference)
  back1: train<=2017, val 2018-19, test 2020-23
  fwd1 : train<=2019, val 2020-21, test 2022-24

Pre-registered bar: on BOTH shifted splits, B's pack-mean margin > A_match's
AND B's margin positive in >= 4/5 seeds.

Run: python scripts/hybrid_robustness.py
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

SHIFTS = {
    "orig": ("2018-12-31", "2020-12-31"),
    "back1": ("2017-12-31", "2019-12-31"),
    "fwd1": ("2019-12-31", "2021-12-31"),
}


def _signed(m: np.ndarray) -> np.ndarray:
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
    spy_z = feats[SPY_Z].reindex(data.dates).to_numpy(dtype="float64")   # (T, 8)
    macro_raw = macro_features(feats)
    naive = data.dates.tz_localize(None)
    macro_z = np.column_stack([zscore_causal(macro_raw[c]) for c in MACRO_FEATURES])
    macro_z = macro_z[np.searchsorted(macro_raw.index, naive)]
    Fb = np.concatenate([spy_z, macro_z], axis=1)                         # (T, 16)
    y = _signed(data.market_returns.numpy())
    ok = np.isfinite(Fb).all(1)

    print(f"{'shift':6s} {'model':8s} {'C':>4s} {'test_n':>7s} {'acc':>6s} {'short':>6s} "
          f"{'margin':>8s} {'wins':>5s} {'sharpe':>8s}  macro_delta")
    for shift, (tr_end, val_end) in SHIFTS.items():
        dates = naive
        tr = dates <= pd.Timestamp(tr_end)
        va = (dates > pd.Timestamp(tr_end)) & (dates <= pd.Timestamp(val_end))
        te = dates > pd.Timestamp(val_end)
        test_dates = data.dates[te]

        rows = {}
        for name, Xm in [("A_match", spy_z), ("B", Fb)]:
            m, C = fit(Xm[tr & ok], y[tr & ok], Xm[va & ok], y[va & ok])
            margins, sharpes, shorts, accs = [], [], [], []
            for seed in SEEDS:
                c = TACRConfig.from_yaml(); c.checkpoint_dir = CHECKPOINTS / TAG / f"s{seed}"
                model, am, _ = load_checkpoint(c.checkpoint_dir / "tacr_best.pt", c)
                acts = roll_actions(model, c, data, test_dates, c.rtg_target,
                                    action_model=am, bcq_phi=c.bcq_phi)
                acts = acts[data.dates.get_indexer(test_dates)]
                pos = data.dates.get_indexer(test_dates)
                Xt = Xm[pos]
                sig = m.predict(Xt)
                a = acts
                mm = data.market_returns.numpy()[pos]
                hyb = sig * np.abs(a)
                sh = sharpe_ratio(hyb * mm)
                em = sharpe_ratio(np.abs(a) * mm)
                margins.append(sh - em)
                sharpes.append(sh)
                shorts.append(float((sig < 0).mean()))
                accs.append(float((sig == _signed(mm)).mean()))
            rows[name] = (C, margins, sharpes, shorts, accs)
        mA = rows["A_match"]
        mB = rows["B"]
        delta = float(np.mean(mB[1]) - np.mean(mA[1]))
        print(f"{shift:6s} {'A_match':8s} {mA[0]:>4g} {int(te.sum()):>7d} {np.mean(mA[4]):>6.3f} "
              f"{np.mean(mA[3]):>6.3f} {np.mean(mA[1]):>8.4f} {int(sum(np.array(mA[1])>0)):>5d} {np.mean(mA[2]):>8.3f}")
        print(f"{shift:6s} {'B':8s} {mB[0]:>4g} {int(te.sum()):>7d} {np.mean(mB[4]):>6.3f} "
              f"{np.mean(mB[3]):>6.3f} {np.mean(mB[1]):>8.4f} {int(sum(np.array(mB[1])>0)):>5d} {np.mean(mB[2]):>8.3f}  {delta:+.4f}")


if __name__ == "__main__":
    main()