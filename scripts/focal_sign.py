"""Focal-loss logistic regression for the Model C+ sign head (7.21).

Standard cross-entropy weights every day equally, so the sign model spends
capacity on already-confident days. Focal loss (Lin et al. 2017) down-weights
"easy" examples by (1-p)^gamma so the fit concentrates on the hard,
ambiguous days near the decision boundary — where the C+ sign accuracy
(53.9%) is currently weakest.

Implemented as a numpy logistic with an analytic gradient (verified against
a numerical gradient below) minimized by scipy L-BFGS-B, so it is
deterministic and gamma=0 reproduces standard cross-entropy exactly.

Run: python scripts/focal_sign.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

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
GAMMA_GRID = [0.0, 0.5, 1.0, 2.0]
LAM_GRID = [1e-3, 1e-2, 1e-1, 1.0]


def _signed(m):
    y = np.sign(m)
    y[y == 0] = 1
    return y


def _focal_loss_and_grad(theta, X, y, gamma, lam):
    """Binary focal loss + L2 on weights. y in {+1,-1}. gamma=0 == CE."""
    w = theta[:-1]
    b = theta[-1]
    z = X @ w + b
    p = expit(np.clip(z, -30, 30))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    pos = y == 1

    fl = np.where(pos, -(1 - p) ** gamma * np.log(p),
                  -(p ** gamma) * np.log(1 - p))
    loss = fl.mean() + 0.5 * lam * float(w @ w)

    # dFL/dp
    gp = np.zeros_like(p)
    gp[pos] = (1 - p[pos]) ** (gamma - 1) * (
        gamma * np.log(p[pos]) - (1 - p[pos]) / p[pos]
    )
    gp[~pos] = p[~pos] ** (gamma - 1) * (
        p[~pos] / (1 - p[~pos]) - gamma * np.log(1 - p[~pos])
    )
    gz = gp * p * (1 - p)
    gw = X.T @ gz / len(y) + lam * w
    gb = gz.mean()
    return loss, np.concatenate([gw, [gb]])


def focal_fit(X, y, gamma, lam):
    n, d = X.shape
    res = minimize(
        fun=lambda t: _focal_loss_and_grad(t, X, y, gamma, lam)[0],
        jac=lambda t: _focal_loss_and_grad(t, X, y, gamma, lam)[1],
        x0=np.zeros(d + 1), method="L-BFGS-B",
        options={"maxiter": 2000},
    )
    return res.x[:-1], res.x[-1]


def predict(w, b, X):
    return np.where(expit(X @ w + b) >= 0.5, 1, -1)


def _check_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = np.where(rng.uniform(size=50) > 0.5, 1, -1)
    theta = rng.normal(size=5)
    loss, g = _focal_loss_and_grad(theta, X, y, gamma=1.0, lam=0.1)
    eps = 1e-6
    g_num = np.zeros(5)
    for i in range(5):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        g_num[i] = (_focal_loss_and_grad(tp, X, y, 1.0, 0.1)[0]
                    - _focal_loss_and_grad(tm, X, y, 1.0, 0.1)[0]) / (2 * eps)
    err = float(np.max(np.abs(g - g_num)))
    print(f"gradient check max|g_analytic - g_numeric| = {err:.2e}")
    assert err < 1e-4, "analytic focal gradient does not match numeric"


def main() -> None:
    _check_gradient()
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
    Xv = Fb[va & ok]
    Xt = Fb[te & ok]
    ret_v = ret[va & ok]
    ret_t = ret[te & ok]
    v_idx = np.where(va & ok)[0]
    t_idx = np.where(te & ok)[0]

    # --- select (gamma, lam) on val by mean 5-seed composite Sharpe ---
    best = (None, -1.0, None, None)
    for gamma in GAMMA_GRID:
        for lam in LAM_GRID:
            w, b = focal_fit(Fb[tr & ok], y[tr & ok], gamma, lam)
            shp = []
            for seed in SEEDS:
                acts = val_mags[seed][np.isin(np.where(va)[0], v_idx)]
                sig = predict(w, b, Xv)
                a = sig * np.abs(acts)
                shp.append(sharpe_ratio(a * ret_v))
            msh = float(np.mean(shp))
            if msh > best[1]:
                best = (w, b, gamma, lam)
    w, b, gamma, lam = best
    print(f"selected gamma={gamma} lam={lam} | val composite Sharpe {best[1]:.4f}")

    # --- test evaluation ---
    margins, sharpes, accs = [], [], []
    for seed in SEEDS:
        acts = test_mags[seed][np.isin(np.where(te)[0], t_idx)]
        sig = predict(w, b, Xt)
        a = sig * np.abs(acts)
        s = sharpe_ratio(a * ret_t)
        em = sharpe_ratio(np.abs(acts) * ret_t)
        margins.append(s - em)
        sharpes.append(s)
        accs.append(float((sig == _signed(ret_t)).mean()))
    print(f"test sign accuracy {np.mean(accs):.4f} (sklearn CE was 0.539)")
    print(f"focal pack mean test Sharpe {np.mean(sharpes):.4f} +- {np.std(sharpes):.4f} "
          f"(C+ CE 1.146) | margins {np.mean(margins):+.4f} | wins {int(sum(np.array(margins)>0))}/5")
    print(f"BAR (mean > 1.146 & wins >= 3/5): "
          f"{'PASS' if (np.mean(sharpes) > 1.146 and int(sum(np.array(margins)>0)) >= 3) else 'FAIL'}")
    for seed, s, mg in zip(SEEDS, sharpes, margins):
        print(f"  s{seed}: test {s:+.4f} margin {mg:+.4f}")


if __name__ == "__main__":
    main()