"""Exp 3 - NIFTY TRANSFER of the frozen canonical C+ protocol. PROJECT_NOTES 7.39.

Mirror of scripts/csi300_transfer.py applied to NIFTY 50:
same 8+8 feature construction (8 price-based causal z + 8 macro causal z using
US cross-asset conditioning), same split bounds (train <=2018-12-31, val
2019-2020, test 2021-2024, OOS 2025-2026), same single-logistic P(up) head
C-selected on val, same sizing rules vs own |a| EM (scale-invariant margin),
same cost grid, same 10-RNG-seed resampled-P robustness sweep.

COVERAGE NOTE (data availability, NOT a protocol change): Yahoo ^NSEI volume
is zero before ~2013-01, so the same volume>0 drop rule as CSI300 yields a
NIFTY frame starting 2013-01-21. The test (2021-2024) and OOS (2025-2026)
windows are fully covered; training starts 2013 instead of 2005.

The OOD arm applies the SPY-trained canonical P to NIFTY z-features WITHOUT
retraining -- a genuine cross-market probe (diagnostic only).

Artifacts: data/nifty_transfer.csv (per-day test table) + console report.

Run: python scripts/nifty_transfer.py
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
from src.models import SPLIT_TRAIN_END, SPLIT_VAL_END, SPLIT_TEST_END  # noqa: E402

C_GRID = [0.01, 0.1, 1.0, 10.0]
N_MEM = 10
RNG0 = 20260908
RNG_SWEEP = [20260908, 1, 2, 3, 4, 42, 123, 777, 2024, 2025]
SPY_Z = ["z_ret_1d", "z_ret_5d", "z_ret_20d", "z_realized_vol_20d",
         "z_rsi_14", "z_macd_hist", "z_volume_zscore_20d", "z_bollinger_pos"]
COSTS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
NIFTY_DIR = ROOT / "data" / "processed_nifty_backup"
SPY_DIR = ROOT / "data" / "processed"


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


def sharpe_n(a: np.ndarray, m: np.ndarray, bps: float) -> tuple[float, float]:
    mr, er = _net(a, m, bps)
    return sharpe_ratio(mr), sharpe_ratio(er)


def build_frame(features_path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Build the canonical 16-dim frame + forward market returns from a Phase-1
    features_regimes.parquet. Returns (naive dates, Fb, mkt, y) with rows on
    which ALL 16 features are finite."""
    feats = pd.read_parquet(features_path)
    idx = pd.DatetimeIndex(feats.index).tz_localize(None)
    spy_z = feats[SPY_Z].to_numpy(dtype="float64")
    macro_raw = macro_features(feats)
    macro_z = np.column_stack([zscore_causal(macro_raw[c], min_periods=60)
                               for c in MACRO_FEATURES]).astype(np.float64)
    Fb = np.concatenate([spy_z, macro_z], axis=1)
    close = feats["close"].astype(float).to_numpy()
    mkt = pd.Series(close, index=idx).pct_change().shift(-1).to_numpy()
    ok = np.isfinite(Fb).all(1)
    return idx[ok], Fb[ok], mkt[ok], _signed(mkt[ok])


def build_vol(mkt: np.ndarray) -> np.ndarray:
    return (pd.Series(mkt).rolling(20).std().to_numpy() * np.sqrt(252))


def turnover(a: np.ndarray) -> float:
    return float(np.abs(np.diff(a, prepend=0.0)).mean())


def short_frac(a: np.ndarray) -> float:
    return float((a < 0).mean())


def main() -> None:
    print("=" * 78)
    print("EXP 3 - NIFTY TRANSFER of frozen canonical C+ protocol")
    print("=" * 78)

    dates, Fb, mkt, y = build_frame(NIFTY_DIR / "features_regimes.parquet")
    tr = (dates <= pd.Timestamp(SPLIT_TRAIN_END))
    va = ((dates > pd.Timestamp(SPLIT_TRAIN_END)) & (dates <= pd.Timestamp(SPLIT_VAL_END)))
    te = (dates > pd.Timestamp(SPLIT_VAL_END)) & (dates <= pd.Timestamp(SPLIT_TEST_END))
    tr_idx = np.where(tr & np.isfinite(Fb).all(1))[0]
    Xv, yv = Fb[va & np.isfinite(Fb).all(1)], y[va & np.isfinite(Fb).all(1)]
    Xte, yte = Fb[te], y[te]
    mkte, vole = mkt[te], build_vol(mkt)[te]
    validte = np.isfinite(mkte) & np.isfinite(vole) & np.isfinite(Fb[te]).all(1)
    Xte, yte, mkte, vole = Xte[validte], yte[validte], mkte[validte], vole[validte]

    print(f"\nNIFTY: dates {dates.min().date()}..{dates.max().date()} "
          f"(n={len(dates)}); test {te.sum()} rows after valid-filter -> "
          f"{len(Xte)} used")
    print(f"  test up-rate {float((yte>0).mean()):.4f} | "
          f"mean fwd ret {float(mkte.mean())*1e4:+.2f} bp/day")

    can = _fit_selected(Fb[tr_idx], y[tr_idx], Xv, yv)
    P = _proba(can, Xte)
    sig = np.where(P - 0.5 >= 0, 1.0, -1.0)
    fP = np.abs(2 * P - 1)
    fP_vol = np.nan_to_num(fP / vole, nan=0.0)
    inv_vol = np.nan_to_num(1.0 / vole, nan=0.0)

    acc = float((sig == yte).mean())
    print(f"\n  CANONICAL P head test acc = {acc:.4f} (up-rate {float((yte>0).mean()):.4f})")
    print(f"  corr(P[t], mkt[t])   = {float(np.corrcoef(P, mkte)[0,1]):+.4f} "
          f"(predictive if ~0)")
    cum = float((1 + mkte).prod())
    print(f"  test-window close-to-close drift (cumprod ret) = {cum:.3f}  "
          f"(<1 = net DOWNTREND in test window; up-rate {float((yte>0).mean()):.4f})")
    print(f"  corr(sig, mkt) = {float(np.corrcoef(sig, mkte)[0,1]):+.4f} "
          f"(learned sign vs realized next-day ret)")

    tbl = pd.DataFrame({
        "date": dates[te][validte].values,
        "P_up": P, "sign": sig, "fP": fP, "fP_vol": fP_vol,
        "vol20d": vole, "ret": mkte, "ret_sign": yte,
    })
    tbl.to_csv(ROOT / "data" / "nifty_transfer.csv", index=False)

    rules = {
        "sign-only  (|a|=1)": sig * 1.0,
        "|2P-1|   (canonical P)": sig * fP,
        "|2P-1| / vol20d": sig * fP_vol,
        "1/vol20d  (no P)": sig * inv_vol,
    }
    print(f'\n{"sizing rule":<28} {"turnover":>8} {"short%":>6}  | '
          + " ".join(f"{int(b) if b==int(b) else b:>4}bp" for b in COSTS)
          + "   margin(1bp)")
    print("-" * 104)
    for name, a in rules.items():
        row, mar1 = [], None
        for bps in COSTS:
            sp, e = sharpe_n(a, mkte, bps)
            if bps == 1.0:
                mar1 = sp - e
            row.append(f"{sp:.2f}")
        print(f"{name:<28} {turnover(a):8.3f} {short_frac(a)*100:6.1f}  | "
              + " ".join(row) + f"   {mar1:+.4f}")

    print("\n" + "=" * 78)
    print("ROBUSTNESS: 10 RNG seeds, resampled-bootstrap P heads (NIFTY)")
    print("=" * 78)
    marg_r = {"|2P-1|": [], "|2P-1|/vol": []}
    for rs in RNG_SWEEP:
        g = np.random.default_rng(rs)
        P_ = np.mean([_proba(_fit_selected(Fb[b], y[b], Xv, yv), Xte)
                      for b in (g.choice(tr_idx, len(tr_idx), replace=True)
                                for _ in range(N_MEM))], axis=0)
        sg = np.where(P_ - 0.5 >= 0, 1.0, -1.0)
        a1 = sg * np.abs(2 * P_ - 1)
        a2 = sg * np.nan_to_num(np.abs(2 * P_ - 1) / vole, nan=0.0)
        marg_r["|2P-1|"].append(sharpe_n(a1, mkte, 1.0)[0] - sharpe_n(a1, mkte, 1.0)[1])
        marg_r["|2P-1|/vol"].append(sharpe_n(a2, mkte, 1.0)[0] - sharpe_n(a2, mkte, 1.0)[1])
    for k, v in marg_r.items():
        v = np.array(v)
        print(f"  {k:<12} margins {np.round(v,4)}  min={v.min():+.4f} "
              f"max={v.max():+.4f} mean={v.mean():+.4f} posfrac={float((v>0).mean()):.2f}")
    sp, e = sharpe_n(sig * fP, mkte, 1.0)
    print(f"  CANONICAL |2P-1| (frozen, deterministic): margin(1bp)={sp-e:+.4f}")

    print("\n" + "=" * 78)
    print("OOS WINDOW CHECK: same NIFTY head, untouched 2025-2026 (no retrain)")
    print("=" * 78)
    for lo, hi, lab in [("2021-01-01", "2024-12-31", "2021-24 canon test"),
                        ("2025-01-01", "2026-12-31", "2025-26 OOS window")]:
        m = (dates > pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        Xw, yw, mw = Fb[m], y[m], mkt[m]
        okw = np.isfinite(mw) & np.isfinite(Xw).all(1)
        Xw, yw, mw = Xw[okw], yw[okw], mw[okw]
        Pw = _proba(can, Xw)
        sgw = np.where(Pw - 0.5 >= 0, 1.0, -1.0)
        fPw = np.abs(2 * Pw - 1)
        sp_, e_ = sharpe_n(sgw * 1.0, mw, 1.0)
        sp2, e2 = sharpe_n(sgw * fPw, mw, 1.0)
        print(f"  {lab}: n={len(Xw)} up={float((yw>0).mean()):.3f} "
              f"acc={float((sgw==yw).mean()):.3f} | sign-only margin(1bp)="
              f"{sp_-e_:+.3f} |2P-1|={sp2-e2:+.3f} "
              f"corr(P,R)={float(np.corrcoef(Pw, mw)[0,1]):+.3f}")

    print("\n" + "=" * 78)
    print("OOD DIAGNOSTIC: SPY-trained canonical P -> NIFTY z-features (no retrain)")
    print("=" * 78)
    sdates, sFb, smkt, sy = build_frame(SPY_DIR / "features_regimes.parquet")
    str_ = (sdates <= pd.Timestamp(SPLIT_TRAIN_END))
    sva = (sdates > pd.Timestamp(SPLIT_TRAIN_END)) & (sdates <= pd.Timestamp(SPLIT_VAL_END))
    s_tr = np.where(str_ & np.isfinite(sFb).all(1))[0]
    spy_can = _fit_selected(sFb[s_tr], sy[s_tr], sFb[sva & np.isfinite(sFb).all(1)],
                            sy[sva & np.isfinite(sFb).all(1)])
    P_ood = _proba(spy_can, Xte)
    sig_o = np.where(P_ood - 0.5 >= 0, 1.0, -1.0)
    acc_o = float(((P_ood >= 0.5) == (yte > 0)).mean())
    sp_, e_ = sharpe_n(sig_o * 1.0, mkte, 1.0)
    sp2, e2 = sharpe_n(sig_o * np.abs(2 * P_ood - 1), mkte, 1.0)
    print(f"  acc(SPY model on NIFTY test) = {acc_o:.4f} (up-rate {float((yte>0).mean()):.4f})")
    print(f"  sign-only margin(1bp) = {sp_ - e_:+.4f}   |2P-1| margin(1bp) = {sp2 - e2:+.4f}")
    print(f"  corr(P_ood, P_nifty) = {np.corrcoef(P_ood, P)[0,1]:+.4f} "
          f"(1.0 would mean identical predictions)")
    print(f"\nsaved per-day table -> data/nifty_transfer.csv")


if __name__ == "__main__":
    main()