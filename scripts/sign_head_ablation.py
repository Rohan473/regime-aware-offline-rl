"""Sign-head ablation on the CORRECTED TRUE-SPY protocol — PROJECT_NOTES 7.35
(user protocol, 2026-09-08).

Questions (all on the same corrected SPY frame as hybrid_sign_cost_analysis):
  1. Is the single-logistic C+ (FROZEN canonical) really negative on true SPY?
  2. Is the bootstrap-ensemble C+ positive, and is the gain real aggregation
     (variance reduction) or one/two lucky members carrying the book?
  3. What does an IDENTICAL-DATA ensemble do (no bootstrap resampling)?
     With a deterministic solver all members coincide with the single model,
     so any ensemble gain vs single must come from bootstrap resampling.
  4. No-uncertainty sizing: P_ens = mean(P_1..P_10), sizing |2P_ens-1| and
     |2P_ens-1|/vol20d — margins vs own EM. Disagreement itself is never
     used in the sizing rule (Q3/Q4 are null by 7.34).

Protocol (identical features / split / cost / rolls as the frozen re-pricing,
scripts/hybrid_sign_cost_analysis.py and scripts/sign_diagnostics.py):
  16-feat Fb (8 SPY z + 8 macro causal z, tz-safe); train <=2018-12-31,
  val 2019-2020, test 2021-2024; C val-selected on [0.01, 0.1, 1, 10];
  model_ret = a*m - (bps/1e4)*|da|, a_{-1}=0; EM on |d|a||; 1 bps PRIMARY;
  magnitude = fix_a_nr7 (5 seeds), BOTH per-seed (frozen convention) and
  seed-mean (diagnostics convention) margins reported.

Run: python scripts/sign_head_ablation.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", message="Unknown solver options")

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
C_GRID = [0.01, 0.1, 1.0, 10.0]
N_MEM = 10
RNG = np.random.default_rng(20260908)
CHECKPOINTS = ROOT / "src" / "models" / "tacr" / "checkpoints" / "tacr"
SPY_Z = ["z_ret_1d", "z_ret_5d", "z_ret_20d", "z_realized_vol_20d",
         "z_rsi_14", "z_macd_hist", "z_volume_zscore_20d", "z_bollinger_pos"]
BP = 1.0  # cps = bps/1e4


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


def _sign(p: np.ndarray) -> np.ndarray:
    return np.where(p - 0.5 >= 0, 1.0, -1.0)


def _net(a: np.ndarray, m: np.ndarray, bps: float = BP):
    da = np.abs(np.diff(a, prepend=0.0))
    absa = np.abs(a)
    return (a * m - (bps / 1e4) * da,
            absa * m - (bps / 1e4) * np.abs(np.diff(absa, prepend=0.0)))


def main() -> None:
    cfg = TACRConfig.from_yaml()
    data = load_tacr_data(20, exclude_policies=cfg.exclude_policies)
    dates = data.dates.tz_localize(None)
    mkt = data.market_returns.numpy()
    y = _signed(mkt)
    vol20d = pd.Series(mkt).rolling(20).std().to_numpy() * np.sqrt(252)

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
    Xtr, ytr = Fb[tr & ok], y[tr & ok]
    Xv, yv = Fb[va & ok], y[va & ok]
    Xte, yte = Fb[te & ok], y[te & ok]

    # ---- heads ----------------------------------------------------------------
    single = _fit_selected(Xtr, ytr, Xv, yv)
    members = []
    for _ in range(N_MEM):
        b = RNG.choice(tr_idx, size=len(tr_idx), replace=True)
        members.append(_fit_selected(Fb[b], y[b], Xv, yv))
    # identical-data ensemble: same full train for every member (deterministic)
    ident = [_fit_selected(Xtr, ytr, Xv, yv) for _ in range(N_MEM)]

    def ens_proba(X) -> np.ndarray:
        return np.mean([_proba(mm, X) for mm in members], axis=0)

    def ident_proba(X) -> np.ndarray:
        return np.mean([_proba(mm, X) for mm in ident], axis=0)

    # ---- roll TACR-8 |a| (5 seeds), per-seed and seed-mean ---------------------
    absa_by_seed = []
    for seed in SEEDS:
        c = TACRConfig.from_yaml()
        c.checkpoint_dir = CHECKPOINTS / TAG8 / f"s{seed}"
        p = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "tacr_best.pt", data=data)
        p = p[p["valid"]].copy()
        pos = dates.get_indexer(pd.DatetimeIndex(p["date"]).tz_localize(None))
        p["pos"] = pos
        p["absa"] = np.abs(p["action"].to_numpy())
        absa_by_seed.append(p)
    absa_mat = np.full((len(SEEDS), len(dates)), np.nan)
    for i, a in enumerate(absa_by_seed):
        absa_mat[i, a["pos"].to_numpy()] = a["absa"].to_numpy()
    te_absa_mean = np.nanmean(absa_mat, axis=0)[te]  # seed-mean |a|, test rows

    # ---- per-head test evaluation ---------------------------------------------
    def evaluate(name: str, proba_te: np.ndarray, verbose: bool = True) -> dict:
        sg = _sign(proba_te)
        # accuracy / short frac on the test frame
        acc = float((sg == yte).mean())
        shortf = float((sg < 0).mean())
        p_up = proba_te
        # frozen convention: per-seed margins (mean over seeds)
        margins_ps, sh_ps, em_ps = [], [], []
        for p in absa_by_seed:
            pos_ok = p["pos"].to_numpy()
            sx = _sign(proba_te[np.searchsorted(np.where(te)[0], pos_ok)])
            a = sx * p["absa"].to_numpy()
            m = p["market_ret"].to_numpy()
            mr, er = _net(a, m)
            margins_ps.append(sharpe_ratio(mr) - sharpe_ratio(er))
            sh_ps.append(sharpe_ratio(mr))
            em_ps.append(sharpe_ratio(er))
        # diagnostics convention: seed-mean |a| margin (one test series)
        mr_sm, er_sm = _net(sg * te_absa_mean, mkt[te])
        margin_sm = sharpe_ratio(mr_sm) - sharpe_ratio(er_sm)
        mr_sm0, er_sm0 = _net(sg * te_absa_mean, mkt[te], 0.0)
        out = {
            "head": name, "acc": acc, "short_frac": shortf,
            "margin_1bp_perseed": float(np.mean(margins_ps)),
            "margin_1bp_seedmean": float(margin_sm),
            "sharpe_1bp_seedmean": float(sharpe_ratio(mr_sm)),
            "sharpe_0bp_seedmean": float(sharpe_ratio(mr_sm0)),
            "P_up_mean": float(p_up.mean()), "P_up_std": float(p_up.std()),
        }
        if verbose:
            print(f"  {name:<42} acc={acc:.4f} shf={shortf:.4f} "
                  f"marg1(ps)={out['margin_1bp_perseed']:+.4f} "
                  f"marg1(sm)={margin_sm:+.4f}")
        return out

    print("=" * 78)
    print("SIGN-HEAD ABLATION — corrected TRUE-SPY protocol (1 bps PRIMARY)")
    print("=" * 78)
    print(f"test rows: {Xte.shape[0]}  | P(up) on test: {yte.mean():.4f} "
          f"(=2*uprate-1, up-rate {float((yte>0).mean()):.4f})")
    print("\n-- heads on test --")
    res = []
    res.append(evaluate("single logistic (frozen canonical)", _proba(single, Xte)))
    res.append(evaluate("10-member bootstrap ensemble (mean P)", ens_proba(Xte)))
    res.append(evaluate("10-member IDENTICAL-data ensemble (mean P)", ident_proba(Xte)))
    for i, m in enumerate(members):
        res.append(evaluate(f"bootstrap member {i:02d}", _proba(m, Xte), verbose=False))

    print("\n-- member distribution vs ensemble (margin_1bp_sm) --")
    member_margins = np.array([r["margin_1bp_seedmean"] for r in res[3:]])
    member_acc = np.array([r["acc"] for r in res[3:]])
    ens = res[1]
    print(f"  member margins: {np.round(member_margins, 4)}")
    print(f"  member margins: min={member_margins.min():+.4f} "
          f"max={member_margins.max():+.4f} mean={member_margins.mean():+.4f}")
    print(f"  ORIGINAL single margin(sm): {res[0]['margin_1bp_seedmean']:+.4f}")
    print(f"  BEST member margin(sm)    : {member_margins.max():+.4f}  "
          f"(is the winner a single member?)")
    print(f"  ENSEMBLE margin(sm)       : {ens['margin_1bp_seedmean']:+.4f}")
    top3 = np.argsort(member_margins)[-3:][::-1]
    print(f"  top-3 members: {[round(v, 4) for v in member_margins[top3]]} "
          f"-> ensemble is {('>' if ens['margin_1bp_seedmean'] > member_margins.max() else '<=')} best member")
    print(f"  ensemble acc {ens['acc']:.4f} vs mean member acc {member_acc.mean():.4f} "
          f"/ max {member_acc.max():.4f}")
    print(f"  P_up distribution: ensemble mean={ens['P_up_mean']:.4f} std={ens['P_up_std']:.4f}; "
          f"member std range=[{min(r['P_up_std'] for r in res[3:]):.4f}, "
          f"{max(r['P_up_std'] for r in res[3:]):.4f}]")

    print("\n-- no-uncertainty sizing with the ensemble probability --")
    P_ens = ens_proba(Xte)
    fP = np.abs(2 * P_ens - 1)
    fP_vol = np.nan_to_num(fP / vol20d[te], nan=0.0)
    for name, size in [("|2P_ens-1|", fP), ("|2P_ens-1| / vol20d", fP_vol)]:
        mr_, er_ = _net(_sign(P_ens) * size, mkt[te])
        print(f"  {name:<22} Sharpe(1bp)={sharpe_ratio(mr_):6.3f}  "
              f"EM={sharpe_ratio(er_):6.3f}  margin={sharpe_ratio(mr_)-sharpe_ratio(er_):+.4f}")

    print("\n-- robustness: identical protocol, N independent RNG seeds --")
    RNG_SEEDS = [20260908, 1, 2, 3, 4, 42, 123, 777, 2024, 2025]
    ens_m = []
    for rs in RNG_SEEDS:
        g = np.random.default_rng(rs)
        mem = []
        for _ in range(N_MEM):
            b = g.choice(tr_idx, size=len(tr_idx), replace=True)
            mem.append(_fit_selected(Fb[b], y[b], Xv, yv))
        Pe = np.mean([_proba(mm, Xte) for mm in mem], axis=0)
        sg = _sign(Pe)
        mr_, er_ = _net(sg * te_absa_mean, mkt[te])
        ens_m.append(sharpe_ratio(mr_) - sharpe_ratio(er_))
    ens_m = np.array(ens_m)
    print(f"  checked seeds: {RNG_SEEDS}")
    print(f"  margins(1bp,sm): {np.round(ens_m, 4)}")
    print(f"  distribution: min={ens_m.min():+.4f} max={ens_m.max():+.4f} "
          f"mean={ens_m.mean():+.4f} positive-frac={float((ens_m>0).mean()):.2f}")

    print("\n-- interpretation --")
    single_sm = res[0]["margin_1bp_seedmean"]
    ens_sm = ens["margin_1bp_seedmean"]
    if single_sm < 0 and ens_sm > member_margins.max():
        print("  ensemble > BEST member AND single negative -> evidence of real")
        print("  variance-reduction / probability aggregation, not a lucky member.")
    elif ens_sm <= member_margins.max():
        print("  ensemble <= best member -> a single (or few) members carry the edge.")
    if res[2]["margin_1bp_seedmean"] == res[0]["margin_1bp_seedmean"]:
        print("  IDENTICAL-data ensemble == single logistic exactly -> any ensemble")
        print("  gain REQUIRES bootstrap resampling (bagging), not averaging per se.")

    out = ROOT / "data" / "sign_head_ablation.csv"
    pd.DataFrame(res).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()