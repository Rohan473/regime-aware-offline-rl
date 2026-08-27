"""Strategy 1 test — does macro/cross-asset data improve the LINEAR SIGN
model (the hybrid's directional head)? (PROJECT_NOTES 7.18)

Compares three L2-logistic sign models on the SAME fix_a_nr7 magnitudes:
  A_full : 8 SPY z-features, trained on the full train split (the
           established hybrid — reference).
  A_match: 8 SPY z-features, trained ONLY on dates where the macro set is
           available (>= 2007-05) — isolates the date-matching effect.
  B      : 8 SPY z-features + 8 macro features (causal z-scores), same
           matched dates — isolates the macro-feature contribution.

Discipline identical to 7.17: L2 LogisticRegression, C selected on VAL from
{0.01,0.1,1,10}, TEST untouched. Margins vs the hybrid's own EM control
(|a_TACR|*m). 5-seed pack = fix_a_nr7's per-seed checkpoints.

Run: python scripts/hybrid_sign_macro.py
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
from src.models import SPLIT_TEST_END, SPLIT_TRAIN_END, SPLIT_VAL_END  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data, split_tacr_data  # noqa: E402
from src.models.tacr.eval import roll_test_predictions  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
TAG = "fix_a_nr7"
C_GRID = [0.01, 0.1, 1.0, 10.0]
CHECKPOINTS = ROOT / "src" / "models" / "tacr" / "checkpoints" / "tacr"
SPY_Z = ["z_ret_1d", "z_ret_5d", "z_ret_20d", "z_realized_vol_20d",
         "z_rsi_14", "z_macd_hist", "z_volume_zscore_20d", "z_bollinger_pos"]


def _signed(m: np.ndarray) -> np.ndarray:
    y = np.sign(m)
    y[y == 0] = 1
    return y


def fit(Xtr, ytr, Xv, yv) -> tuple[LogisticRegression, float]:
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
    splits = split_tacr_data(data)
    train, val, test = splits["train"], splits["val"], splits["test"]

    feats = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_regimes.parquet")
    spy_z = feats[SPY_Z].reindex(data.dates).to_numpy(dtype="float64")     # (T, 8)
    macro_raw = macro_features(feats)
    naive = data.dates.tz_localize(None)
    macro_z = np.column_stack([zscore_causal(macro_raw[c]) for c in MACRO_FEATURES])
    macro_z = macro_z[np.searchsorted(macro_raw.index, naive)]             # (T, 8)
    Fb = np.concatenate([spy_z, macro_z], axis=1)                          # (T, 16)

    dates = data.dates.tz_localize(None)
    tr_mask = dates <= pd.Timestamp(SPLIT_TRAIN_END)
    va_mask = (dates > pd.Timestamp(SPLIT_TRAIN_END)) & (dates <= pd.Timestamp(SPLIT_VAL_END))
    te_mask = (dates > pd.Timestamp(SPLIT_VAL_END)) & (dates <= pd.Timestamp(SPLIT_TEST_END))

    y = _signed(data.market_returns.numpy())
    ok_spy = np.isfinite(spy_z).all(1)
    ok_mac = np.isfinite(Fb).all(1)

    models = {
        "A_full": fit(spy_z[tr_mask & ok_spy], y[tr_mask & ok_spy], spy_z[va_mask & ok_spy], y[va_mask & ok_spy]),
        "A_match": fit(spy_z[tr_mask & ok_mac], y[tr_mask & ok_mac], spy_z[va_mask & ok_mac], y[va_mask & ok_mac]),
        "B": fit(Fb[tr_mask & ok_mac], y[tr_mask & ok_mac], Fb[va_mask & ok_mac], y[va_mask & ok_mac]),
    }

    print(f"{'model':9s} {'C':>4s} {'train_n':>8s} {'test_acc':>8s} {'always+1':>9s} "
          f"{'short_frac':>10s} {'margin_mean':>11s} {'wins':>5s} {'sharpe_mean':>11s}")
    for name, Xm in [("A_full", spy_z), ("A_match", spy_z), ("B", Fb)]:
        m, C = models[name]
        tr_n = int((tr_mask & (ok_mac if name != "A_full" else ok_spy)).sum())
        margins, sharpes, shorts, accs = [], [], [], []
        for seed in SEEDS:
            c = TACRConfig.from_yaml(); c.checkpoint_dir = CHECKPOINTS / TAG / f"s{seed}"
            p = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "tacr_best.pt", data=data)
            p = p[p["valid"]]
            pos = data.dates.get_indexer(pd.DatetimeIndex(p["date"]))
            Xt = Xm[pos]
            good = np.isfinite(Xt).all(1)
            sig = m.predict(Xt[good])
            a = p["action"].to_numpy()[good]
            mm = p["market_ret"].to_numpy()[good]
            hyb = sig * np.abs(a)
            sh = sharpe_ratio(hyb * mm)
            em = sharpe_ratio(np.abs(a) * mm)
            margins.append(sh - em)
            sharpes.append(sh)
            shorts.append(float((sig < 0).mean()))
            accs.append(float((sig == _signed(mm)).mean()))
        print(f"{name:9s} {C:>4g} {tr_n:>8d} {np.mean(accs):>8.4f} {max((y[te_mask]==1).mean(),(y[te_mask]==-1).mean()):>9.4f} "
              f"{np.mean(shorts):>10.4f} {np.mean(margins):>11.4f} {int(sum(np.array(margins)>0)):>5d} {np.mean(sharpes):>11.4f}")


if __name__ == "__main__":
    main()