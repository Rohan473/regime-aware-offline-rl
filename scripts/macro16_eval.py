"""Exp 1 (7.30.1): 16-dim TACR eval vs corrected C+ baseline (2026-09-08).

Pre-registered bars (PROJECT_NOTES 7.30.1) + user protocol (1 bps primary
selection, net-of-cost, identical cost model for both arms):

  (a) Does the ACTOR alone (16-dim TACR action sign = direction) produce
      positive Sharpe AND beat its own |a| EM control?
  (b) C+16 = [TACR_16 |a|] x [16-feat corrected sign] vs corrected C+ =
      [TACR_8 |a|] x [16-feat corrected sign] — does widening the magnitude
      head's info set change sizing enough to matter net-of-cost?
  (c) Cost-robustness of any margin gains at 1 bps (primary) + {0,0.5,2}
      sensitivity.

Cost model identical to scripts/hybrid_sign_cost_analysis.py / cost_sweep.py:
  model_ret = a*m - (bps/1e4)*|da|, a_{-1}=0; EM on |d|a||.
The corrected C+ reference numbers are replay-projected here on the identical
roll so both arms are like-for-like (same sign model, same 5 seeds, same cost).

Run: python scripts/macro16_eval.py
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
from src.eval.regime_eval import max_drawdown, sharpe_ratio  # noqa: E402
from src.models import SPLIT_TRAIN_END, SPLIT_VAL_END  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data  # noqa: E402
from src.models.tacr.eval import roll_test_predictions  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
TAG16 = "macro16"
TAG8 = "fix_a_nr7"
C_GRID = [0.01, 0.1, 1.0, 10.0]
CHECKPOINTS = ROOT / "src" / "models" / "tacr" / "checkpoints" / "tacr"
SPY_Z = ["z_ret_1d", "z_ret_5d", "z_ret_20d", "z_realized_vol_20d",
         "z_rsi_14", "z_macd_hist", "z_volume_zscore_20d", "z_bollinger_pos"]
COSTS = [0.0, 0.5, 1.0, 2.0]


def _signed(m: np.ndarray) -> np.ndarray:
    y = np.sign(m)
    y[y == 0] = 1
    return y


def _fit_val_selected(Xtr, ytr, Xv, yv) -> LogisticRegression:
    best, best_acc = None, -1.0
    for C in C_GRID:
        m = LogisticRegression(penalty="l2", C=C, max_iter=2000).fit(Xtr, ytr)
        acc = float((m.predict(Xv) == yv).mean())
        if acc > best_acc:
            best, best_acc = m, acc
    return best


def _net(a: np.ndarray, m: np.ndarray, bps: float):
    da = np.abs(np.diff(a, prepend=0.0))
    absa = np.abs(a)
    mr = a * m - (bps / 1e4) * da
    er = absa * m - (bps / 1e4) * np.abs(np.diff(absa, prepend=0.0))
    return mr, er


def main() -> None:
    cfg = TACRConfig.from_yaml()
    # data8 = canonical frame (8-dim) for the sign model / Fb / date bookkeeping.
    data8 = load_tacr_data(20, exclude_policies=cfg.exclude_policies)
    dates = data8.dates.tz_localize(None)

    feats = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_regimes.parquet")
    spy_z = feats[SPY_Z].reindex(data8.dates).to_numpy(dtype="float64")
    mraw = macro_features(feats)
    macro_z = np.column_stack([zscore_causal(mraw[c]) for c in MACRO_FEATURES])
    macro_z = macro_z[np.searchsorted(mraw.index, dates)]
    Fb = np.concatenate([spy_z, macro_z], axis=1)
    y = _signed(data8.market_returns.numpy())
    tr = dates <= pd.Timestamp(SPLIT_TRAIN_END)
    va = (dates > pd.Timestamp(SPLIT_TRAIN_END)) & (dates <= pd.Timestamp(SPLIT_VAL_END))
    ok = np.isfinite(Fb).all(1)
    sign = _fit_val_selected(Fb[tr & ok], y[tr & ok], Fb[va & ok], y[va & ok])

    # data16 = 16-dim dataset, used ONLY to roll the 16-dim model (its network
    # expects 16-dim states). Test dates are identical in both, so preds are
    # mapped back onto the canonical `dates` index for the sign join.
    data16 = load_tacr_data(20, exclude_policies=cfg.exclude_policies, state_macro=True)

    def roll(tag: str, data: object, state_macro: bool) -> list[pd.DataFrame]:
        out = []
        for seed in SEEDS:
            c = TACRConfig.from_yaml()
            c.state_macro = state_macro
            c.checkpoint_dir = CHECKPOINTS / tag / f"s{seed}"
            p = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "tacr_best.pt", data=data)
            p = p[p["valid"]].copy()
            pos = dates.get_indexer(pd.DatetimeIndex(p["date"]).tz_localize(None))
            Xt = Fb[pos]
            good = np.isfinite(Xt).all(1)
            p = p[good].copy()
            p["sign"] = sign.predict(Xt[good])
            p["date_naive"] = pd.DatetimeIndex(p["date"]).tz_localize(None)
            out.append(p.reset_index(drop=True))
        return out

    p16 = roll(TAG16, data16, state_macro=True)
    p8 = roll(TAG8, data8, state_macro=False)

    def report(tag: str, preds: list[pd.DataFrame], label: str, out_rows: list[dict]) -> None:
        a_all = np.concatenate([p["action"].to_numpy() for p in preds])
        m_all = np.concatenate([p["market_ret"].to_numpy() for p in preds])
        # (a) actor-only direction
        s_dirs, s_em = [], []
        for p in preds:
            a = p["action"].to_numpy(); mm = p["market_ret"].to_numpy()
            da = np.abs(np.diff(a, prepend=0.0))
            s_dirs.append(sharpe_ratio(a * mm - (1 / 1e4) * da))
            s_em.append(sharpe_ratio(np.abs(a) * mm - (1 / 1e4) * np.abs(np.diff(np.abs(a), prepend=0.0))))
        # (b) C+ = sign*|a| at each cost
        for bps in COSTS:
            margins, msh, emsh, wins = [], [], [], 0
            for p in preds:
                a = p["sign"].to_numpy() * np.abs(p["action"].to_numpy())
                mm = p["market_ret"].to_numpy()
                mr, er = _net(a, mm, bps)
                mm_, ee = sharpe_ratio(mr), sharpe_ratio(er)
                margins.append(mm_ - ee); msh.append(mm_); emsh.append(ee)
                wins += int(mm_ > ee)
            out_rows.append({
                "arm": label, "bps": bps,
                "model_sharpe": round(float(np.mean(msh)), 3),
                "em_sharpe": round(float(np.mean(emsh)), 3),
                "net_margin": round(float(np.mean(margins)), 3),
                "wins": wins,
            })
        # turnover / dd / short_frac (raw = 0 bps C+)
        a0 = np.concatenate([p["sign"].to_numpy() * np.abs(p["action"].to_numpy()) for p in preds])
        da0 = np.abs(np.diff(a0, prepend=0.0))
        r0, _ = _net(a0, np.concatenate([p["market_ret"].to_numpy() for p in preds]), 0.0)
        s_dd = [max_drawdown(_net(p["sign"].to_numpy() * np.abs(p["action"].to_numpy()),
                                  p["market_ret"].to_numpy(), 0.0)[0]) for p in preds]
        print(f"{label}:"
              f"  actor-dir Sharpe(1bp)={np.mean(s_dirs):.3f} (vs EM {np.mean(s_em):.3f})"
              f"  | turnover={da0.mean():.3f} daily / {da0.mean()*252:.0f} ann"
              f"  | short_frac={(a0<0).mean():.3f} | maxDD(0bp)={np.mean(s_dd):.3f}")
        return None

    rows = []
    print("=" * 80)
    print("Exp 1 (16-dim TACR) vs corrected C+ — net-of-cost (1 bps = PRIMARY)")
    print("=" * 80)
    report(TAG16, p16, "TACR16 actor/C+16", rows)
    report(TAG8, p8, "TACR8  actor/C+8 ", rows)
    print()
    print(f"{'arm':>12} {'bps':>5} {'model_sh':>9} {'em_sh':>8} {'net_margin':>11} {'wins':>5}")
    for r in rows:
        print(f"{r['arm']:>12} {r['bps']:>5.1f} {r['model_sharpe']:>9.3f} {r['em_sharpe']:>8.3f} {r['net_margin']:>11.3f} {r['wins']:>5}")

    out_rows = [{"arm": r["arm"], "bps": r["bps"], "model_sharpe": r["model_sharpe"],
                 "em_sharpe": r["em_sharpe"], "net_margin": r["net_margin"], "wins": r["wins"]}
                for r in rows]
    out = ROOT / "data" / "macro16_eval.csv"
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
