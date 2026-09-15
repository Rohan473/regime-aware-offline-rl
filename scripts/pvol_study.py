"""Exp 3 — P/vol sizing study on TRUE SPY under the FROZEN CANONICAL protocol.
PROJECT_NOTES 7.36.

Question: after the sign-head ablation (7.35) showed the bootstrap ensemble's
+0.228 margin was a single-seed artifact (mean ~0 across 10 RNG seeds), the only
cost-robust feature left on SPY was the scale-invariant event-scaled sizing
|2P-1| (+0.118 @1bp in 7.34/7.35). This study checks that edge carefully:

  1. PRIMARY: P_up from the FROZEN canonical single-logistic sign head
     (C-selected on val, deterministic solver), NOT the bootstrap ensemble.
     Does the sizing edge survive the canonical head?
  2. SIZE RULES on the test roll, compared each to its OWN EM (|a|) so the
     margin is scale-invariant:
        sign-only (|a|=1)            -- does direction beat the SPY forward-return
                                        (buy-and-hold) baseline?
        |2P-1|                       -- supervised confidence event-scaled
        |2P-1| / vol20d              -- + volatility normalization (claimed 7.34)
        |2P-1| * sign(P>0.5)
     with controls:
        1/vol20d                     -- pure inverse-volatility sizing (no P)
        P itself                     -- soft confidence version of |2P-1|
  3. COST GRID: net Sharpe at 0/0.5/1/2/5/10 bps vs own EM; margin = Sharpe(posn)
     - Sharpe(EM). Report turnover + short fraction per rule.
  4. ROBUSTNESS: same 10-RNG-seed bootstrap sweep as the sign-head ablation,
     but for the SIZING margins, so we do not repeat the lucky-seed artifact.
     Includes the single-seed ENSEMBLE P result for cross-reference.
  5. SAVE per-day test table -> data/pvol_study.csv.

Runs offline deterministically (no RL checkpoints touched). P comes only from
the corrected 16-feature frame (8 SPY z + 8 macro) + forward next-day returns.

Run: python scripts/pvol_study.py
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
from src.models import SPLIT_TRAIN_END, SPLIT_VAL_END  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
C_GRID = [0.01, 0.1, 1.0, 10.0]
N_MEM = 10
RNG0 = 20260908
RNG_SWEEP = [20260908, 1, 2, 3, 4, 42, 123, 777, 2024, 2025]
SPY_Z = ["z_ret_1d", "z_ret_5d", "z_ret_20d", "z_realized_vol_20d",
         "z_rsi_14", "z_macd_hist", "z_volume_zscore_20d", "z_bollinger_pos"]
COSTS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]


def _signed(m: np.ndarray) -> np.ndarray:
    y = np.sign(m)
    y[y == 0] = 1
    return y


def _fit_selected(Xtr, ytr, Xv, yv) -> LogisticRegression:
    best, best_acc = None, -1.0
    for C in C_GRID:
        m = LogisticRegression(penalty="l2", C=C, max_iter=2000).fit(Xtr, ytr)
        acc = float((m.predict(Xv) == yv).mean())
        if acc > best_acc:
            best, best_acc = m, acc
    return best


def _proba(m: LogisticRegression, X) -> np.ndarray:
    return np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)


def _net(a: np.ndarray, m: np.ndarray, bps: float):
    da = np.abs(np.diff(a, prepend=0.0))
    absa = np.abs(a)
    return (a * m - (bps / 1e4) * da,
            absa * m - (bps / 1e4) * np.abs(np.diff(absa, prepend=0.0)))


def sharpe_n(a: np.ndarray, m: np.ndarray, bps: float) -> float:
    mr, er = _net(a, m, bps)
    return sharpe_ratio(mr), sharpe_ratio(er)


def turnover(a: np.ndarray) -> float:
    return float(np.abs(np.diff(a, prepend=0.0)).mean())


def short_frac(a: np.ndarray) -> float:
    return float((a < 0).mean())


def main() -> None:
    cfg = TACRConfig.from_yaml()
    data = load_tacr_data(20, exclude_policies=cfg.exclude_policies)
    dates = data.dates.tz_localize(None)
    mkt = data.market_returns.numpy()
    y = _signed(mkt)

    feats = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_regimes.parquet")
    spy_z = feats[SPY_Z].reindex(data.dates).to_numpy(dtype="float64")
    macro_raw = macro_features(feats)
    macro_z = np.column_stack([zscore_causal(macro_raw[c]) for c in MACRO_FEATURES])
    macro_z = macro_z[np.searchsorted(macro_raw.index, dates)]
    Fb = np.concatenate([spy_z, macro_z], axis=1)
    ok = np.isfinite(Fb).all(1)

    tr = dates <= pd.Timestamp(SPLIT_TRAIN_END)
    va = (dates > pd.Timestamp(SPLIT_TRAIN_END)) & (dates <= pd.Timestamp(SPLIT_VAL_END))
    te = dates > pd.Timestamp(SPLIT_VAL_END)
    tr_idx = np.where(tr & ok)[0]
    Xv, yv = Fb[va & ok], y[va & ok]
    Xte, yte = Fb[te & ok], y[te & ok]
    mkte = mkt[te & ok]
    vole = (pd.Series(mkt).rolling(20).std().to_numpy() * np.sqrt(252))[te & ok]

    print("=" * 78)
    print("EXP 3 - P/vol SIZING STUDY, frozen canonical protocol, TRUE SPY")
    print("=" * 78)
    print(f"test days {Xte.shape[0]} | up-rate {float((yte>0).mean()):.4f} "
          f"| mean forward ret {float(mkte.mean())*1e4:+.2f} bp/day")
    print(f"training {tr_idx.size} / val {(va&ok).sum()} / test {(te&ok).sum()}\n")

    # ---- rainbow of P models -----------------------------------------------
    # canonical: frozen single logistic, C-selected on val, deterministic.
    can = _fit_selected(Fb[tr_idx], y[tr_idx], Xv, yv)
    P_can = _proba(can, Xte)

    # ensemble (single seed, cross-reference to 7.34/7.35 diagnostics)
    g = np.random.default_rng(RNG0)
    mem = []
    for _ in range(N_MEM):
        b = g.choice(tr_idx, size=len(tr_idx), replace=True)
        mem.append(_fit_selected(Fb[b], y[b], Xv, yv))
    Pmem = np.column_stack([_proba(mm_, Xte) for mm_ in mem])
    P_ens = Pmem.mean(axis=1)

    # ---- per-day test table (canonical P) ----------------------------------
    tbl = pd.DataFrame({
        "date": dates[te & ok].values,
        "P_up_can": P_can,
        "P_up_ens": P_ens,
        "sign_can": np.where(P_can - 0.5 >= 0, 1.0, -1.0),
        "sign_ens": np.where(P_ens - 0.5 >= 0, 1.0, -1.0),
        "fP": np.abs(2 * P_can - 1),
        "fP_vol": np.nan_to_num(np.abs(2 * P_can - 1) / vole, nan=0.0),
        "vol20d": vole,
        "ret": mkte,
        "ret_sign": yte,
    })
    out = ROOT / "data" / "pvol_study.csv"
    tbl.to_csv(out, index=False)

    sig = tbl["sign_can"].to_numpy()
    fP = tbl["fP"].to_numpy()
    fP_vol = tbl["fP_vol"].to_numpy()
    mm = tbl["ret"].to_numpy()
    inv_vol = np.nan_to_num(1.0 / vole, nan=0.0)

    # ---- rules --------------------------------------------------------------
    rules = {
        "sign-only  (|a|=1)": sig * 1.0,
        "|2P-1|   (canonical P)": sig * fP,
        "|2P-1| / vol20d": sig * fP_vol,
        "1/vol20d  (no P)": sig * inv_vol,
        "ensemble |2P-1| (ref)": np.where(P_ens - 0.5 >= 0, 1.0, -1.0)
        * np.abs(2 * P_ens - 1),
    }

    print(f'{"sizing rule":<28} {"turnover":>8} {"short%":>6}  | ' + " ".join(
        f"{int(b) if b==int(b) else b:>4}bp" for b in COSTS) + "   margin(1bp)")
    print("-" * 104)
    for name, a in rules.items():
        row = []
        mar1 = None
        for bps in COSTS:
            sp, e = sharpe_n(a, mm, bps)
            if bps == 1.0:
                mar1 = sp - e
            row.append(f"{sp:.2f}")
        print(f"{name:<28} {turnover(a):8.3f} {short_frac(a)*100:6.1f}  | "
              + " ".join(row) + f"   {mar1:+.4f}")

    print()
    print("  EM anchor for each rule = its own |a| (scale-invariant margin); ")
    print("  sign-only EM = buy-and-hold SPY forward-return baseline.\n")

    # ---- robustness: can the |2P-1| sizing edge survive reseeding? ----------
    print("=" * 78)
    print("ROBUSTNESS: 10 independent RNG seeds, resampled-bootstrap P heads,")
    print("sizing margins evaluated on the SAME test window (canonical rule).")
    print("=" * 78)
    marg_r = {"|2P-1|": [], "|2P-1|/vol": []}
    for rs in RNG_SWEEP:
        g = np.random.default_rng(rs)
        P_ = np.mean([_proba(
            _fit_selected(Fb[b], y[b], Xv, yv), Xte)
            for b in (g.choice(tr_idx, len(tr_idx), replace=True) for _ in range(N_MEM))], axis=0)
        sg = np.where(P_ - 0.5 >= 0, 1.0, -1.0)
        a1 = sg * np.abs(2 * P_ - 1)
        a2 = sg * np.nan_to_num(np.abs(2 * P_ - 1) / vole, nan=0.0)
        marg_r["|2P-1|"].append(sharpe_n(a1, mm, 1.0)[0] - sharpe_n(a1, mm, 1.0)[1])
        marg_r["|2P-1|/vol"].append(sharpe_n(a2, mm, 1.0)[0] - sharpe_n(a2, mm, 1.0)[1])
    for k, v in marg_r.items():
        v = np.array(v)
        print(f"  {k:<12} margins {np.round(v,4)}  min={v.min():+.4f} "
              f"max={v.max():+.4f} mean={v.mean():+.4f} posfrac={float((v>0).mean()):.2f}")

    # canonical (deterministic, no resampling) single-model reference for the
    # same margin — is the edge a property of P itself, not of any ensemble?
    a_can = sig * fP
    sp, e = sharpe_n(a_can, mm, 1.0)
    print(f"  CANONICAL |2P-1| (frozen, deterministic): margin(1bp)={sp-e:+.4f}")

    print(f"\nsaved per-day table -> {out}")


if __name__ == "__main__":
    main()