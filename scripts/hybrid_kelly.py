"""Kelly-sized C+ variant (7.23) — replace the TACR magnitude with
fractional-Kelly sizing from the logistic's probability:

    kelly_const : a_t = clip(k * (2P(up|s_t) - 1), -1, 1)
    kelly_vol   : a_t = clip(k * (2P(up|s_t) - 1) / sigma_t^2, -1, 1)

sigma_t = causal trailing 20d std of SPY daily returns. Sign = the same
16-feature logistic as C+. k is selected on VAL (sets the exposure level;
Sharpe and the EM margin are scale-invariant to k in the unsaturated
regime). The Kelly strategy is self-contained (no TACR magnitude) and hence
DETERMINISTIC — a single point estimate, compared against the C+ 5-seed
pack mean (1.146, margins +0.30) and per-seed range.

Run: python scripts/hybrid_kelly.py
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
from src.eval.regime_eval import sharpe_ratio, max_drawdown  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
SPY_Z = ["z_ret_1d", "z_ret_5d", "z_ret_20d", "z_realized_vol_20d",
         "z_rsi_14", "z_macd_hist", "z_volume_zscore_20d", "z_bollinger_pos"]
C_GRID = [0.01, 0.1, 1.0, 10.0]
K_GRID = {"const": [0.5, 1.0, 2.0, 5.0, 10.0], "vol": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]}


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
    raw = macro_features(feats)
    naive = data.dates.tz_localize(None)
    mac = np.column_stack([zscore_causal(raw[c]) for c in MACRO_FEATURES])
    mac = mac[np.searchsorted(raw.index, naive)]
    Fb = np.concatenate([spy_z, mac], 1)
    ret = data.market_returns.numpy()
    y = _signed(ret)
    ok = np.isfinite(Fb).all(1)

    spy_close = feats["close"].reindex(data.dates)
    sig = spy_close.pct_change().rolling(20, min_periods=20).std().to_numpy()  # daily

    dates = naive
    tr = dates <= pd.Timestamp("2018-12-31")
    va = (dates > pd.Timestamp("2018-12-31")) & (dates <= pd.Timestamp("2020-12-31"))
    te = dates > pd.Timestamp("2020-12-31")

    m, sign_C = fit(Fb[tr & ok], y[tr & ok], Fb[va & ok], y[va & ok])
    P = np.full(len(Fb), np.nan)
    P[ok] = m.predict_proba(Fb[ok])[:, 1]
    edge = 2.0 * P - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        edge_vol = np.where(sig > 1e-12, edge / (sig**2), np.nan)

    variants = {
        "const": edge,
        "vol": edge_vol,
    }
    t_idx = np.where(te & ok)[0]
    v_idx = np.where(va & ok)[0]
    ret_v = ret[va & ok]
    ret_t = ret[te & ok]

    print(f"sign model C={sign_C} | test sign acc {float((m.predict(Fb[te&ok])==y[te&ok]).mean()):.4f} "
          f"| |2P-1| mean {np.abs(edge[te&ok]).mean():.4f}")
    print(f"\n{'variant':12s} {'k(val)':>8s} {'val Sh':>7s} {'test Sharpe':>11s} {'margin':>8s} "
          f"{'mean|a|':>8s} {'maxDD':>8s}  (C+ ref: Sharpe 1.146, margin +0.296)")

    for vname, edge_arr in variants.items():
        # select k on VAL (deterministic kelly: single val Sharpe per k)
        best_k, best_vs = None, -1e9
        for k in K_GRID[vname]:
            a_v = np.clip(k * edge_arr[va & ok], -1.0, 1.0)
            vs = sharpe_ratio(a_v * ret_v)
            if vs > best_vs:
                best_vs, best_k = vs, k
        a_t = np.clip(best_k * edge_arr[te & ok], -1.0, 1.0)
        s = sharpe_ratio(a_t * ret_t)
        em = sharpe_ratio(np.abs(a_t) * ret_t)
        dd = max_drawdown(a_t * ret_t)
        print(f"{vname:12s} {best_k:>8g} {best_vs:>7.3f} {s:>11.4f} {s - em:>8.4f} "
              f"{np.abs(a_t).mean():>8.3f} {dd:>8.4f}")

    # C+ reference (TACR |a| x sign) pack, for the table
    from src.models.tacr.eval import load_checkpoint, roll_actions
    CK = ROOT / "src" / "models" / "tacr" / "checkpoints" / "tacr"
    sharpes, margins = [], []
    for seed in SEEDS:
        c = TACRConfig.from_yaml(); c.checkpoint_dir = CK / "fix_a_nr7" / f"s{seed}"
        mm, am, _ = load_checkpoint(c.checkpoint_dir / "tacr_best.pt", c)
        dts = data.dates[te]
        acts = roll_actions(mm, c, data, dts, c.rtg_target, action_model=am, bcq_phi=c.bcq_phi)
        acts = np.abs(acts[data.dates.get_indexer(dts)])[np.isin(np.where(te)[0], t_idx)]
        sign_t = np.sign(edge[te & ok])
        a = sign_t * acts
        s = sharpe_ratio(a * ret_t)
        margins.append(s - sharpe_ratio(acts * ret_t))
        sharpes.append(s)
    print(f"\nC+ reference (TACR |a|): pack Sharpe {np.mean(sharpes):.4f} +- {np.std(sharpes):.4f} "
          f"| margins {np.mean(margins):+.4f}, {int(sum(np.array(margins)>0))}/5")


if __name__ == "__main__":
    main()