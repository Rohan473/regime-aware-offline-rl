"""Paper-method test — intraday-range trend labels + RF/NN (7.24).

Follows the Azizi (JRFM 2026) framing directly: label a day TREND when its
session INTRADAY HIGH-LOW RANGE exceeds tau (0.5/0.75/1.0%), classify with
Random Forest / Neural Network on trend features (VIX, RSI, ATR + current
range for the PAPER variant; the 16 C+ features + ATR + range for FULL), and
use the prediction as a filter on the C+ hybrid:
    a_t = I(trend_pred=1) * sign_logistic(s_t) * |a_TACR(s_t)|

(tau, classifier, feature-set) selected on VAL by mean 5-seed val composite
Sharpe; TEST untouched. Deviation flagged: no macro-announcement calendar.

Run: python scripts/hybrid_trend_paper.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

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
TAU_GRID = [0.005, 0.0075, 0.010]


def _signed(m):
    y = np.sign(m)
    y[y == 0] = 1
    return y


def atr14(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()


def fit_sign(Xtr, ytr, Xv, yv):
    best = (None, -1.0, None)
    for C in C_GRID:
        m = LogisticRegression(penalty="l2", C=C, max_iter=2000).fit(Xtr, ytr)
        acc = float((m.predict(Xv) == yv).mean())
        if acc > best[1]:
            best = (m, acc, C)
    return best[0], best[2]


def trend_classifier(kind: str):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=8,
                                      min_samples_leaf=20, random_state=0)
    return MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=2000,
                         random_state=0, early_stopping=True)


def main() -> None:
    cfg = TACRConfig.from_yaml()
    data = load_tacr_data(20, exclude_policies=cfg.exclude_policies)

    feats = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_regimes.parquet")
    spy_z = feats[SPY_Z].reindex(data.dates).to_numpy(float)
    raw = macro_features(feats)
    naive = data.dates.tz_localize(None)
    mac = np.column_stack([zscore_causal(raw[c]) for c in MACRO_FEATURES])
    mac = mac[np.searchsorted(raw.index, naive)]
    F16 = np.concatenate([spy_z, mac], 1)

    # trend features (causal z) on data dates
    rng = (feats["high"] - feats["low"]) / feats["close"]          # current range
    rng = zscore_causal(rng).reindex(data.dates).to_numpy()
    atr = zscore_causal(atr14(feats)).reindex(data.dates).to_numpy()
    vix = zscore_causal(pd.read_parquet(ROOT / "data" / "macro" / "vix.parquet")["close"])
    vix = vix.reindex(data.dates.tz_localize(None)).to_numpy()
    rsi = feats["z_rsi_14"].reindex(data.dates).to_numpy()

    F_paper = np.column_stack([vix, rsi, atr, rng])                # paper's feature families
    F_full = np.column_stack([F16, atr, rng])                      # 16 + ATR + range

    # labels: next-day intraday-range trend
    raw_range = ((feats["high"] - feats["low"]) / feats["close"])
    raw_range = raw_range.reindex(data.dates).to_numpy()
    next_range = np.roll(raw_range, -1)
    next_range[-1] = np.nan  # no next day

    y = _signed(data.market_returns.numpy())
    dates = naive
    tr = dates <= pd.Timestamp("2018-12-31")
    va = (dates > pd.Timestamp("2018-12-31")) & (dates <= pd.Timestamp("2020-12-31"))
    te = dates > pd.Timestamp("2020-12-31")

    ok_p = np.isfinite(F_paper).all(1)
    ok_f = np.isfinite(F_full).all(1)

    sign_model, sign_C = fit_sign(F16[tr & ok_f], y[tr & ok_f], F16[va & ok_f], y[va & ok_f])

    # magnitude per seed for val + test
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

    v_idx = np.where(va)[0]
    t_idx = np.where(te)[0]
    ret_v = data.market_returns.numpy()[va]
    ret_t = data.market_returns.numpy()[te]

    # val/test mask: label defined AND features defined (paper/full)
    configs = []
    for fname, F, okF in [("PAPER", F_paper, ok_p), ("FULL", F_full, ok_f)]:
        for tau in TAU_GRID:
            for kind in ["rf", "mlp"]:
                configs.append((fname, tau, kind, F, okF))

    best = (None, -1.0)
    for fname, tau, kind, F, okF in configs:
        trc = tr & okF & ~np.isnan(next_range)
        vac = va & okF & ~np.isnan(next_range)
        yt = (next_range > tau).astype(int)
        clf = trend_classifier(kind)
        clf.fit(F[trc], yt[trc])
        vs = []
        for seed in SEEDS:
            fire = clf.predict(F[vac]) == 1
            mag = val_mags[seed][np.isin(v_idx, np.where(vac)[0])]
            sign = sign_model.predict(F16[vac])
            a = np.where(fire, sign * np.abs(mag), 0.0)
            vs.append(sharpe_ratio(a * ret_v))
        msh = float(np.mean(vs))
        if msh > best[1]:
            best = ((fname, tau, kind, F, okF, clf), msh)
    (fname, tau, kind, F, okF, clf), val_sh = best
    print(f"selected: featset={fname} tau={tau:.4f} clf={kind} | val composite Sharpe {val_sh:.4f}")

    # test evaluation
    tec = te & okF & ~np.isnan(next_range)
    fire = clf.predict(F[tec]) == 1
    base_rate = float((next_range[tec] > tau).mean())
    acc = float((clf.predict(F[tec]) == (next_range[tec] > tau).astype(int)).mean())
    margins, sharpes = [], []
    for seed in SEEDS:
        mag = test_mags[seed][np.isin(t_idx, np.where(tec)[0])]
        sign = sign_model.predict(F16[tec])
        a = np.where(fire, sign * np.abs(mag), 0.0)
        ret_tc = data.market_returns.numpy()[tec]
        s = sharpe_ratio(a * ret_tc)
        margins.append(s - sharpe_ratio(np.where(fire, np.abs(mag), 0.0) * ret_tc))
        sharpes.append(s)
    print(f"trend-day base rate {base_rate:.3f} | classifier acc {acc:.4f} | trade rate {fire.mean():.3f}")
    print(f"filtered pack mean test Sharpe {np.mean(sharpes):.4f} +- {np.std(sharpes):.4f} "
          f"(C+ 1.146) | margins {np.mean(margins):+.4f} | wins {int(sum(np.array(margins)>0))}/5")


if __name__ == "__main__":
    main()