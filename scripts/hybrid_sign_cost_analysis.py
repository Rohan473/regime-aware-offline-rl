"""Corrected-C+ cost re-pricing — PROJECT_NOTES 7.31 (user protocol, 2026-09-08).

After the dxy_corr_20d alignment fix, the canonical C+ sign model changed from
+0.296/1.15 (settled) to +2.08/1.76 raw but with sign-head turnover ~0.71. Per
the user's protocol, this script:

  1. FREEZES the corrected C+ definition (identical features / sign model /
     val-selected C / split / sizing as hybrid_sign_macro.py B).
  2. Re-prices it net-of-cost on the shared turnover convention
     (model_ret = a*m - (bps/1e4)*|da|, a_{-1}=0; EM on |d|a||) — matching
     scripts/cost_sweep.py net_returns + src/data/behavior_policies.py.
  3. Reports gross/net Sharpe, gross/net margin, turnover, annualized
     turnover, max DD, cost drag across bps in {0, 0.5, 1, 2, 5, 10, 20}.
       1 bps  = PRIMARY selection metric (user protocol)
       0/0.5/2 = sensitivity; raw 0 bps = diagnostic only.
  4. Decomposes the post-fix margin into OLD (pre-fix retained) vs RECOVERED
     dates (dxy_corr_20d was NaN under the old poisoned definition), showing
     where the new edge and the new turnover actually live.

Does NOT tune anything on the 1-bps test result. Read-only over checkpoints.

Run: python scripts/hybrid_sign_cost_analysis.py
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
from src.data.macro_factors import (  # noqa: E402
    MACRO_FEATURES,
    macro_features,
    zscore_causal,
)
from src.eval.regime_eval import max_drawdown, sharpe_ratio  # noqa: E402
from src.models import SPLIT_TEST_END, SPLIT_TRAIN_END, SPLIT_VAL_END  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data  # noqa: E402
from src.models.tacr.eval import roll_test_predictions  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
TAG = "fix_a_nr7"
C_GRID = [0.01, 0.1, 1.0, 10.0]
CHECKPOINTS = ROOT / "src" / "models" / "tacr" / "checkpoints" / "tacr"
SPY_Z = ["z_ret_1d", "z_ret_5d", "z_ret_20d", "z_realized_vol_20d",
         "z_rsi_14", "z_macd_hist", "z_volume_zscore_20d", "z_bollinger_pos"]
COST_GRID = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
ANNUAL = 252


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


def _old_poisoned_dxy(spy_ret: pd.Series, dxy_ret: pd.Series) -> pd.Series:
    """The PRE-fix dxy_corr_20d (calendar-misalignment-poisoned) definition,
    reconstructed for the old-vs-recovered dates decomposition only."""
    return spy_ret.rolling(20, min_periods=20).corr(dxy_ret)


def main() -> None:
    cfg = TACRConfig.from_yaml()
    data = load_tacr_data(20, exclude_policies=cfg.exclude_policies)
    naive_dates = data.dates.tz_localize(None)

    feats = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_regimes.parquet")
    spy_z = feats[SPY_Z].reindex(data.dates).to_numpy(dtype="float64")

    # Corrected macro set (current macro_features) — the FROZEN definition.
    mraw = macro_features(feats)
    macro_z = np.column_stack([zscore_causal(mraw[c]) for c in MACRO_FEATURES])
    macro_z = macro_z[np.searchsorted(mraw.index, naive_dates)]
    Fb = np.concatenate([spy_z, macro_z], axis=1)

    # Reconstruct the OLD poisoned mask (pre-fix dxy) for the decomposition.
    idx = pd.DatetimeIndex(feats.index).tz_localize(None)
    spy_ret = pd.Series(feats["close"].astype(float).to_numpy(), index=idx).pct_change()
    dxy_ret = pd.Series(pd.read_parquet(ROOT / "data" / "macro" / "dxy.parquet")["close"]
                        .astype(float).pct_change()).reindex(idx)
    old_dxy = _old_poisoned_dxy(spy_ret, dxy_ret)
    old_dxy = old_dxy.reindex(naive_dates).to_numpy(dtype="float64")
    old_raw = pd.DataFrame({
        "risk_on_1d": mraw["risk_on_1d"].reindex(naive_dates).to_numpy(),
        "tnx_delta_1d": mraw["tnx_delta_1d"].reindex(naive_dates).to_numpy(),
        "tnx_delta_5d": mraw["tnx_delta_5d"].reindex(naive_dates).to_numpy(),
        "vol_term": mraw["vol_term"].reindex(naive_dates).to_numpy(),
        "dxy_corr_20d": old_dxy,  # poisoned version
        "credit_1d": mraw["credit_1d"].reindex(naive_dates).to_numpy(),
        "rs_qqq_1d": mraw["rs_qqq_1d"].reindex(naive_dates).to_numpy(),
        "rs_iwm_1d": mraw["rs_iwm_1d"].reindex(naive_dates).to_numpy(),
    })
    old_ok = np.isfinite(old_raw.to_numpy()).all(1)
    new_ok = np.isfinite(Fb).all(1)
    recovered = new_ok & ~old_ok   # dates the fix RE-ENABLED
    retained = new_ok & old_ok     # dates valid under BOTH definitions

    # y (direction) and split masks.
    dates = naive_dates
    y = _signed(data.market_returns.numpy())
    tr = dates <= pd.Timestamp(SPLIT_TRAIN_END)
    va = (dates > pd.Timestamp(SPLIT_TRAIN_END)) & (dates <= pd.Timestamp(SPLIT_VAL_END))
    te = dates > pd.Timestamp(SPLIT_VAL_END)

    model = _fit_val_selected(
        Fb[tr & new_ok], y[tr & new_ok], Fb[va & new_ok], y[va & new_ok]
    )

    # Collect test-day preds for every seed with the corrected sign.
    seed_preds = []
    for seed in SEEDS:
        c = TACRConfig.from_yaml(); c.checkpoint_dir = CHECKPOINTS / TAG / f"s{seed}"
        p = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "tacr_best.pt", data=data)
        p = p[p["valid"]].copy()
        pos = dates.get_indexer(pd.DatetimeIndex(p["date"]).tz_localize(None))
        Xt = Fb[pos]
        good = np.isfinite(Xt).all(1)
        sg = model.predict(Xt[good])
        p = p[good].copy()
        p["sign"] = sg
        p["hyb"] = sg * np.abs(p["action"].to_numpy())
        p["ret"] = p["hyb"] * p["market_ret"]
        p["date_naive"] = pd.DatetimeIndex(p["date"]).tz_localize(None)
        p["is_recovered"] = recovered[dates.get_indexer(p["date_naive"])].astype(bool)
        p["is_retained"] = retained[dates.get_indexer(p["date_naive"])].astype(bool)
        seed_preds.append(p.reset_index(drop=True))

    # ------------------------------------------------------------------ cost
    def cost_row(p: pd.DataFrame, bps: float) -> dict:
        a = p["hyb"].to_numpy()          # the traded (signed x |a|) action
        da = np.abs(np.diff(a, prepend=0.0))
        absa = np.abs(a)
        em_da = np.abs(np.diff(absa, prepend=0.0))
        cost_fn = (bps / 1e4)
        m = p["market_ret"].to_numpy()
        model_ret = a * m - cost_fn * da
        em_ret = absa * m - cost_fn * em_da
        ms, es = sharpe_ratio(model_ret), sharpe_ratio(em_ret)
        cum = np.prod(1 + model_ret) - 1
        return {
            "bps": bps,
            "model_sharpe": round(float(ms), 4),
            "em_sharpe": round(float(es), 4),
            "net_margin": round(float(ms - es), 4),
            "max_dd": round(float(max_drawdown(model_ret)), 4),
            "cost_drag_bps_day": round(float(cost_fn * da.mean()) * 1e4, 4),
        }

    print("=" * 78)
    print("Corrected C+ net-of-cost (5-seed mean; 1 bps = PRIMARY selection)")
    print("=" * 78)
    tbl = []
    # gross (0 bps) full-report row with turnover
    all_a = np.concatenate([p["hyb"].to_numpy() for p in seed_preds])
    all_da = np.abs(np.diff(all_a, prepend=0.0))
    turnover_daily = float(all_da.mean())
    annual_turn = float(all_da.mean() * ANNUAL)
    print(f"turnover (mean |da|): {turnover_daily:.4f} daily / {annual_turn:.1f} annualized"
          f"  | short_frac: "
          f"{float(np.concatenate([(p['sign'].to_numpy()<0) for p in seed_preds]).mean()):.4f}")
    header = f"{'bps':>5} {'model_sh':>9} {'em_sh':>8} {'net_margin':>11} {'wins':>5} {'max_dd':>8} {'drag_bp/d':>9}"
    print(header)
    agg = []
    for bps in COST_GRID:
        margins, wins, dds, drags = [], 0, [], []
        for p in seed_preds:
            r = cost_row(p, bps)
            margins.append(r["net_margin"])
            wins += int(r["model_sharpe"] > r["em_sharpe"])
            dds.append(r["max_dd"])
            drags.append(r["cost_drag_bps_day"])
        ms = float(np.mean([cost_row(p, bps)["model_sharpe"] for p in seed_preds]))
        es = float(np.mean([cost_row(p, bps)["em_sharpe"] for p in seed_preds]))
        row = f"{bps:>5.1f} {ms:>9.3f} {es:>8.3f} {np.mean(margins):>11.3f} {wins:>5} {np.mean(dds):>8.4f} {np.mean(drags):>9.3f}"
        print(row)
        agg.append({"bps": bps, "model_sharpe": round(ms, 3), "em_sharpe": round(es, 3),
                    "net_margin": round(float(np.mean(margins)), 3), "wins": wins,
                    "max_dd": round(float(np.mean(dds)), 4),
                    "drag_bps_day": round(float(np.mean(drags)), 3)})

    # ------------------------------------------------------------------ where
    print("\n" + "=" * 78)
    print("Where the post-fix edge lives — recovered vs retained test days")
    print("=" * 78)
    # concatenate all test days across seeds
    allp = pd.concat(seed_preds, ignore_index=True)
    for grp, mask, label in [
        ("recovered", allp["is_recovered"], "RECOVERED (old dxy was NaN)"),
        ("retained", allp["is_retained"], "RETAINED (valid under both defs)"),
        ("all", np.ones(len(allp), bool), "ALL test days"),
    ]:
        sub = allp[mask]
        if len(sub) == 0:
            print(f"{label:<32}: 0 days")
            continue
        r = sub["ret"].to_numpy()
        a = sub["hyb"].to_numpy()
        m = sub["market_ret"].to_numpy()
        em = np.abs(a) * m
        print(f"{label:<32}: n={len(sub):>5}  gross simult-sharpe={sharpe_ratio(r):>7.3f} "
              f"em={sharpe_ratio(em):>7.3f}  margin={sharpe_ratio(r)-sharpe_ratio(em):>7.3f}  "
              f"|da|={float(np.abs(np.diff(a,prepend=0)).mean()):>5.3f}  "
              f"short_frac={float((a<0).mean()):>5.3f}")

    out = ROOT / "data" / "hybrid_sign_cost_analysis.csv"
    pd.DataFrame(agg).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
