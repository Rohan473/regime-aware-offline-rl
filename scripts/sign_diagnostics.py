"""Sign/magnitude/uncertainty diagnostics — PROJECT_NOTES 7.33 (Exp 1 follow-up,
user protocol 2026-09-08).

Builds the per-day TEST diagnostic table (one row per test date, 2021-2024):

    date | P_up (10-member ensemble mean) | U_var (ensemble variance) |
    sign | c8 (C+ posn, seed-mean TACR-8 |a|) | c16 (C+16 posn) |
    |a_8| seed-mean | |a_16| seed-mean | ret (realized next-day) |
    ret_sign | vol20d (annualized realized vol)

then answers the 5 gating questions BEFORE any further model work:

  #1 Does corrected-C+ direction still dominate (net-of-cost at 1 bps)?
  #2 Does TACR magnitude add incremental value over simple P/vol sizing?
       C+8, C+16, f(P)=|2P-1|, f(P)/vol20d  — all sign(P-0.5)*|a|, margin
       is scale-invariant so no exposure calibration needed.
  #3 Does ensemble uncertainty U_var predict sign errors?
  #4 Does uncertainty predict bad C+ P&L?
  #5 Is #3/#4 different for predicted-long vs predicted-short days?

Ensemble = 10-member bootstrap L2-logistic on the same corrected 16-feature
set as C+'s B sign head, each member C-selected on val, TEST untouched.
Matches 7.30.1 Exp 2 "10-member bootstrap-logistic ensemble, C-selected on
val as-is".

Run: python scripts/sign_diagnostics.py
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
from src.models.tacr.eval import roll_test_predictions  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
TAG8 = "fix_a_nr7"
TAG16 = "macro16"
C_GRID = [0.01, 0.1, 1.0, 10.0]
N_MEM = 10
RNG = np.random.default_rng(20260908)
CHECKPOINTS = ROOT / "src" / "models" / "tacr" / "checkpoints" / "tacr"
SPY_Z = ["z_ret_1d", "z_ret_5d", "z_ret_20d", "z_realized_vol_20d",
         "z_rsi_14", "z_macd_hist", "z_volume_zscore_20d", "z_bollinger_pos"]


def _signed(m: np.ndarray) -> np.ndarray:
    y = np.sign(m)
    y[y == 0] = 1
    return y


def _net(a: np.ndarray, m: np.ndarray, bps: float):
    da = np.abs(np.diff(a, prepend=0.0))
    absa = np.abs(a)
    return (a * m - (bps / 1e4) * da,
            absa * m - (bps / 1e4) * np.abs(np.diff(absa, prepend=0.0)))


def _fit_selected(Xtr, ytr, Xv, yv) -> LogisticRegression:
    best, best_acc = None, -1.0
    for C in C_GRID:
        m = LogisticRegression(penalty="l2", C=C, max_iter=2000).fit(Xtr, ytr)
        acc = float((m.predict(Xv) == yv).mean())
        if acc > best_acc:
            best, best_acc = m, acc
    return best


def main() -> None:
    cfg = TACRConfig.from_yaml()
    data8 = load_tacr_data(20, exclude_policies=cfg.exclude_policies)
    data16 = load_tacr_data(20, exclude_policies=cfg.exclude_policies, state_macro=True)
    dates = data8.dates.tz_localize(None)

    feats = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_regimes.parquet")
    spy_z = feats[SPY_Z].reindex(data8.dates).to_numpy(dtype="float64")
    macro_raw = macro_features(feats)
    macro_z = np.column_stack([zscore_causal(macro_raw[c]) for c in MACRO_FEATURES])
    macro_z = macro_z[np.searchsorted(macro_raw.index, dates)]
    Fb = np.concatenate([spy_z, macro_z], axis=1)
    y = _signed(data8.market_returns.numpy())
    mkt = data8.market_returns.numpy()
    vol20d = pd.Series(mkt).rolling(20).std().to_numpy() * np.sqrt(252)

    tr = dates <= pd.Timestamp(SPLIT_TRAIN_END)
    va = (dates > pd.Timestamp(SPLIT_TRAIN_END)) & (dates <= pd.Timestamp(SPLIT_VAL_END))
    ok = np.isfinite(Fb).all(1)

    # ---- 10-member bootstrap ensemble (C-selected on val per member) --------
    tr_idx = np.where(tr & ok)[0]
    members = []
    for k in range(N_MEM):
        b = RNG.choice(tr_idx, size=len(tr_idx), replace=True)
        m = _fit_selected(Fb[b], y[b], Fb[va & ok], y[va & ok])
        members.append(m)

    # ---- roll both TACR packs on the canonical frame ------------------------
    def roll_acts(tag: str, dat: object, sel: bool) -> np.ndarray:
        absas = []
        for seed in SEEDS:
            c = TACRConfig.from_yaml()
            c.state_macro = sel
            c.checkpoint_dir = CHECKPOINTS / tag / f"s{seed}"
            p = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "tacr_best.pt", data=dat)
            p = p[p["valid"]]
            pos = dates.get_indexer(pd.DatetimeIndex(p["date"]).tz_localize(None))
            good = np.isfinite(Fb[pos]).all(1)
            a = np.full(len(dates), np.nan)
            a[pos[good]] = np.abs(p["action"].to_numpy()[good])
            absas.append(a)
        return np.nanmean(np.vstack(absas), axis=0)

    abs8 = roll_acts(TAG8, data8, False)
    abs16 = roll_acts(TAG16, data16, True)

    # ---- per-day test table ---------------------------------------------------
    te = dates > pd.Timestamp(SPLIT_VAL_END)
    # predict P(up) only on finite-feature rows; test days are all within ok
    P = np.full((len(dates), N_MEM), np.nan)
    P[ok] = np.column_stack([np.clip(m.predict_proba(Fb[ok])[:, 1], 1e-9, 1 - 1e-9)
                             for m in members])
    Pmean = np.nanmean(P, axis=1)
    Pvar = np.nanvar(P, axis=1)
    sign = np.where(Pmean - 0.5 >= 0, 1.0, -1.0)

    tbl = pd.DataFrame({
        "date": dates[te].values,
        "P_up": Pmean[te],
        "U_var": Pvar[te],
        "sign": sign[te],
        "c8": sign[te] * abs8[te],
        "c16": sign[te] * abs16[te],
        "a8": abs8[te],
        "a16": abs16[te],
        "ret": mkt[te],
        "ret_sign": y[te],
        "vol20d": vol20d[te],
    }).dropna().reset_index(drop=True)

    out = ROOT / "data" / "sign_diagnostics.csv"
    tbl.to_csv(out, index=False)

    a8 = tbl["c8"].to_numpy()
    a16 = tbl["c16"].to_numpy()
    mm = tbl["ret"].to_numpy()
    sig = tbl["sign"].to_numpy()

    # ---- #1 corrected-C+ direction dominance (1 bps, like-for-like) ---------
    print("=" * 74)
    print("#1 corrected C+ direction — net fit = sign(P>0.5) * seed-mean |a|")
    print("=" * 74)
    for name, a in [("C+8 (=sign*mean|a8|)", a8), ("C+16 (=sign*mean|a16|)", a16)]:
        mr, er = _net(a, mm, 0.0)
        mr1, _ = _net(a, mm, 1.0)
        print(f"  {name:<28} Sharpe(0bp)={sharpe_ratio(mr):6.3f}  Sharpe(1bp)={sharpe_ratio(mr1):6.3f}  "
              f"EM(0bp)={sharpe_ratio(er):6.3f}  margin(1bp)={sharpe_ratio(mr1)-sharpe_ratio(er):6.3f}")

    # ---- #2 TACR magnitude vs simple P/vol sizing ----------------------------
    print("\n" + "=" * 74)
    print("#2 does TACR magnitude add value over probability/vol sizing?")
    print("    (margin scale-invariant; EM = own |a|)")
    print("=" * 74)
    fP = np.abs(2 * tbl["P_up"].to_numpy() - 1)
    fP_vol = np.nan_to_num(fP / tbl["vol20d"].to_numpy(), nan=0.0)
    for name, a in [("TACR-8 magnitude (C+8)", a8),
                    ("TACR-16 magnitude (C+16)", a16),
                    ("P-sizing  f(P)=|2P-1|", sig * fP),
                    ("P/vol      f(P)/vol20d", sig * fP_vol)]:
        mr, er = _net(a, mm, 1.0)
        print(f"  {name:<28} Sharpe(1bp)={sharpe_ratio(mr):6.3f}  EM={sharpe_ratio(er):6.3f}  "
              f"margin(1bp)={sharpe_ratio(mr)-sharpe_ratio(er):6.3f}")

    # ---- #3/#4/#5 uncertainty vs sign errors and C+ P&L ----------------------
    U = tbl["U_var"].to_numpy()
    err = (tbl["sign"].to_numpy() != tbl["ret_sign"].to_numpy()).astype(int)
    pnl = sig * np.abs(a8) * mm           # C+ P&L at 0 bps, seed-mean mag
    pnl1, _ = _net(a8, mm, 1.0)

    def by_quantile(x: np.ndarray, y: np.ndarray, label: str, n: int = 3) -> None:
        qs = np.nanquantile(x, np.linspace(0, 1, n + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        print(f"  {label} by U_var quantile: ")
        for i in range(n):
            mask = (x >= qs[i]) & (x <= qs[i + 1])
            if mask.sum() == 0:
                continue
            er = y[mask].mean()
            print(f"    U[{i}] n={mask.sum():>4}  mean={er:8.4f}")
        # monotonic trend
        means = [y[(x >= qs[i]) & (x <= qs[i + 1])].mean()
                 for i in range(n) if ((x >= qs[i]) & (x <= qs[i + 1])).sum() > 0]
        print(f"    trend across quantiles: {[round(m, 4) for m in means]}")

    print("\n" + "=" * 74)
    print("#3 does ensemble uncertainty U_var predict SIGN ERRORS?")
    print("=" * 74)
    by_quantile(U, err, "sign-error rate")
    corr3 = float(np.corrcoef(U, err)[0, 1])
    print(f"  corr(U_var, err)={corr3:.4f}")

    print("\n" + "=" * 74)
    print("#4 does uncertainty predict bad C+ P&L (1 bps)?")
    print("=" * 74)
    by_quantile(U, pnl1, "mean C+ daily P&L(1bp)")
    corr4 = float(np.corrcoef(U, pnl1)[0, 1])
    print(f"  corr(U_var, P&L)={corr4:.4f}")

    print("\n" + "=" * 74)
    print("#5 long vs short asymmetry (U vs errors / P&L)")
    print("=" * 74)
    for leg, mask in [("LONG  (sign=+1)", sig > 0), ("SHORT (sign=-1)", sig < 0)]:
        print(f"  {leg}: n={mask.sum():>4}  err_rate={err[mask].mean():.4f}  "
              f"corr(U,err)={np.corrcoef(U[mask], err[mask])[0,1]:+.4f}  "
              f"corr(U,P&L)={np.corrcoef(U[mask], pnl1[mask])[0,1]:+.4f}  "
              f"meanP&L={pnl1[mask].mean():+.5f}")
    print(f"\nsaved per-day table -> {out}")
    print("\nVERDICT (per user decision tree):")
    print("  #3/#4 positive  -> invest in uncertainty-aware sizing")
    print("  #3/#4 null      -> abandon uncertainty as the main contribution")


if __name__ == "__main__":
    main()