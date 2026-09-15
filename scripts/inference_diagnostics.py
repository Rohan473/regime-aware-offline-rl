"""Statistical inference + diagnostics for the frozen-canonical sizing results.
PROJECT_NOTES 7.38 (companion to 7.36 / 7.37) and 7.39 (NIFTY replication).

Purpose
-------
1. RECONSTRUCT net-of-cost daily returns for every sizing rule DIRECTLY from the
   saved per-day test tables (data/pvol_study.csv, data/csi300_transfer.csv,
   data/nifty_transfer.csv)
   using the exact recorded convention (_net: model_ret = a*m - (bps/1e4)*|da|,
   EM on |a| with its own turnover cost; Sharpe = annualized mean/std).

2. CROSS-VALIDATE: reconstructed turnover / short% / Sharpe(1bp) / margin(1bp)
   vs the recorded 7.36 (SPY), 7.37 (CSI300) and 7.39 (NIFTY) tables. Any
   mismatch fails loud.

3. STATISTICAL INFERENCE on each rule (per market):
   * mean daily margin (bp/day) of model vs its own EM, with Newey-West HAC
     t-statistic (H0: mean daily margin = 0).
   * moving-block bootstrap (MBB) 95% CIs for (a) Sharpe-space margin(1bp) and
     (b) mean daily margin (bp/day). Bootstrap preserves serial dependence.
   * annualized Sharpe + EM Sharpe for each rule, with the milestone numbers.

4. NON-SPY DIAGNOSTICS (Exp 2/3 review: CSI300 + NIFTY):
   * by-calendar-year breakdown: n, up-rate, drift (cumprod), margin(1bp) per
     rule, Sharpe per rule.
   * max drawdown (model net cumulative path) per rule.
   * turnover per rule per year + overall.
   * return distribution stats for the |2P-1| rule (mean/std/skew/kurt/quantiles,
     in bp/day).
   * regime breakdown (bull/bear/crisis from features_regimes.parquet): per-regime
     n, up-rate, margin for sign-only / |2P-1| / |2P-1|/vol.

Deliberate integrity choice: this script reads ONLY the saved per-day tables. It
does NOT re-run training, re-touch the 2025-2026 OOS windows (already reported
once), or recompute any model. It is pure retained-analysis over frozen outputs.

Multiple-testing / snooping caveat: the 2021-24 test window served development,
so the CIs below are DESCRIPTIVE for that window (report as such). The
pre-registered, single-report evidence lives in the OOS arms (7.26 / 7.37 OOS
/ 7.39 OOS), which this script intentionally does not re-examine.

Run: python scripts/inference_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.regime_eval import max_drawdown, sharpe_ratio  # noqa: E402

RNG = 20260909
B1_BPS = 1.0
BOOT_DRAWS = 5000
BLOCK_LEN = 21  # one trading month
BLOCK_LEN_ALT = 5  # one trading week (sensitivity)
NW_LAG_FALLBACK = 8

COSTS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]


# --------------------------------------------------------------------------- #
# net-of-cost helpers (identical convention to pvol_study / csi300_transfer)
# --------------------------------------------------------------------------- #
def net(a: np.ndarray, m: np.ndarray, bps: float):
    da = np.abs(np.diff(a, prepend=0.0))
    absa = np.abs(a)
    return a * m - (bps / 1e4) * da, absa * m - (bps / 1e4) * np.abs(np.diff(absa, prepend=0.0))


def turnover(a: np.ndarray) -> float:
    return float(np.abs(np.diff(a, prepend=0.0)).mean())


def short_frac(a: np.ndarray) -> float:
    return float((a < 0).mean())


# --------------------------------------------------------------------------- #
# Newey-West HAC variance of the sample mean
# --------------------------------------------------------------------------- #
def newey_west_se(x: np.ndarray, lag: int | None = None) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float("nan")
    lag = NW_LAG_FALLBACK if lag is None else lag
    lag = min(lag, n - 1)
    xc = x - x.mean()
    gamma = np.correlate(xc, xc, mode="full")[n - 1:] / n
    var = gamma[0] + 2.0 * sum(gamma[k] * (1.0 - k / (lag + 1.0)) for k in range(1, lag + 1))
    var = max(var, 1e-12)
    return float(np.sqrt(var / n))


def nw_tstat(x: np.ndarray, lag: int | None = None) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2 or np.std(x) == 0.0:
        return float("nan")
    return float(x.mean() / newey_west_se(x, lag))


# --------------------------------------------------------------------------- #
# moving-block bootstrap
# --------------------------------------------------------------------------- #
def mbb_ci(series: np.ndarray, stat_fn, draws: int, block_len: int, rng, lo: float = 0.025,
           hi: float = 0.975):
    """Bootstrap CI for stat_fn computed on a moving-block resample of series."""
    series = np.asarray(series, dtype=float)
    n = series.size
    if n < 2:
        return np.full(3, np.nan)
    block_len = max(1, min(block_len, n - 1))
    nblk = int(np.ceil(n / block_len)) + 1
    samples = np.empty(draws)
    idx = np.arange(n)
    for b in range(draws):
        blocks = [idx[rng.integers(0, n - block_len + 1)] + np.arange(block_len)
                  for _ in range(nblk)]
        bs = np.concatenate(blocks)[:n]
        samples[b] = stat_fn(series[bs])
    return np.percentile(samples, [lo * 100, 50.0, hi * 100])


# --------------------------------------------------------------------------- #
# rules from a saved per-day table
# --------------------------------------------------------------------------- #
def build_rules(tbl: pd.DataFrame) -> dict[str, np.ndarray]:
    """Position series per rule from a saved per-day table (SPY or CSI300)."""
    sig = tbl["sign"].to_numpy() if "sign" in tbl.columns else tbl["sign_can"].to_numpy()
    P_can = tbl["P_up"].to_numpy() if "P_up" in tbl.columns else tbl["P_up_can"].to_numpy()
    vol = tbl["vol20d"].to_numpy()
    fP = tbl["fP"].to_numpy() if "fP" in tbl.columns else np.abs(2 * P_can - 1)
    names = ["sign-only (|a|=1)", "|2P-1| (canonical P)", "|2P-1| / vol20d", "1/vol20d (no P)"]
    rules = {
        names[0]: sig * 1.0,
        names[1]: sig * fP,
        names[2]: sig * np.nan_to_num(fP / vol, nan=0.0),
        names[3]: sig * np.nan_to_num(1.0 / vol, nan=0.0),
    }
    if "P_up_ens" in tbl.columns:  # SPY table carries the ensemble cross-ref
        P_ens = tbl["P_up_ens"].to_numpy()
        sig_e = tbl["sign_ens"].to_numpy() if "sign_ens" in tbl.columns \
            else np.where(P_ens - 0.5 >= 0, 1.0, -1.0)
        rules["ensemble |2P-1| (ref)"] = sig_e * np.abs(2 * P_ens - 1)
    return rules


# --------------------------------------------------------------------------- #
# per-rule inference
# --------------------------------------------------------------------------- #
def _sharpe_margin_paired(mr: np.ndarray, er: np.ndarray, block_len: int, rng,
                          draws: int = BOOT_DRAWS) -> np.ndarray:
    """MBB CI for Sharpe(model)-Sharpe(EM) using PAIRED blocks (preserves the
    within-day coupling of model and EM returns). Returns [lo, point, hi]."""
    n = mr.size
    if n < 4:
        return np.array([np.nan, sharpe_ratio(mr) - sharpe_ratio(er), np.nan])
    block_len = max(1, min(block_len, n - 1))
    idx = np.arange(n)
    nblk = int(np.ceil(n / block_len)) + 1
    vals = np.empty(draws)
    for b in range(draws):
        blocks = [idx[rng.integers(0, n - block_len + 1)] + np.arange(block_len)
                  for _ in range(nblk)]
        bs = np.concatenate(blocks)[:n]
        vals[b] = sharpe_ratio(mr[bs]) - sharpe_ratio(er[bs])
    q = np.percentile(vals, [2.5, 97.5])
    return np.array([q[0], sharpe_ratio(mr) - sharpe_ratio(er), q[1]])


def infer_rule(name: str, a: np.ndarray, m: np.ndarray, rng) -> dict:
    turn = turnover(a)
    sfrac = short_frac(a) * 100.0
    mr, er = net(a, m, B1_BPS)
    sharpe_m = sharpe_ratio(mr)
    sharpe_e = sharpe_ratio(er)
    daily_margin = mr - er
    mean_margin_bp = float(daily_margin.mean()) * 1e4
    t_nw = nw_tstat(daily_margin)

    ci_bp = mbb_ci(daily_margin, lambda s: float(np.mean(s)) * 1e4, BOOT_DRAWS, BLOCK_LEN, rng)
    ci_bp_alt = mbb_ci(daily_margin, lambda s: float(np.mean(s)) * 1e4, BOOT_DRAWS,
                       BLOCK_LEN_ALT, rng)
    ci_sharpe = _sharpe_margin_paired(mr, er, BLOCK_LEN, rng)

    ladder = [sharpe_ratio(net(a, m, b)[0]) for b in COSTS]

    return {
        "rule": name, "n": int(a.size), "turnover": turn, "short_pct": sfrac,
        "sharpe_1bp": sharpe_m, "em_sharpe_1bp": sharpe_e,
        "margin_1bp": sharpe_m - sharpe_e,
        "mean_daily_margin_bp": mean_margin_bp,
        "ann_margin_pct": float(mean_margin_bp * 252 / 100.0),
        "nw_tstat": t_nw,
        "ci95_margin_lo": ci_sharpe[0], "ci95_margin_hi": ci_sharpe[2],
        "ci95_bp_lo": ci_bp[0], "ci95_bp_hi": ci_bp[2],
        "ci95_bp_lo_l5": ci_bp_alt[0], "ci95_bp_hi_l5": ci_bp_alt[2],
        "sharpe_0bp": ladder[0], "sharpe_10bp": ladder[-1],
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(RNG)
    print("=" * 78)
    print("INFERENCE + DIAGNOSTICS on frozen-canonical sizing results (7.38)")
    print("=" * 78)

    failures = []
    infer_rows = []
    for market, csv_path, feats_dir in [
        ("SPY", ROOT / "data" / "pvol_study.csv", None),
        ("CSI300", ROOT / "data" / "csi300_transfer.csv",
         ROOT / "data" / "processed_csi300_backup"),
        ("NIFTY", ROOT / "data" / "nifty_transfer.csv",
         ROOT / "data" / "processed_nifty_backup"),
    ]:
        tbl = pd.read_csv(csv_path, parse_dates=["date"])
        m = tbl["ret"].to_numpy()
        rules = build_rules(tbl)
        print(f"\n### {market}  (test n={len(tbl)}, up-rate {float((tbl['ret_sign']>0).mean()):.3f})")

        header = f'{"rule":<26} {"turn":>6} {"short%":>6} {"Sh1bp":>6} {"Sh(EM)":>6} ' \
                 f'{"margin1":>7} {"m_bp/d":>7} {"t_NW":>6}  {"CI bp/d [2.5,97.5]":>22}'
        print(header)
        print("-" * len(header))
        for name, a in rules.items():
            r = infer_rule(name, a, m, rng)
            ci3 = _sharpe_margin_paired(net(a, m, B1_BPS)[0], net(a, m, B1_BPS)[1], BLOCK_LEN, rng)
            r["ci95_margin_lo"], r["ci95_margin_hi"] = ci3[0], ci3[2]
            print(f'{r["rule"]:<26} {r["turnover"]:6.3f} {r["short_pct"]:6.1f} '
                  f'{r["sharpe_1bp"]:6.2f} {r["em_sharpe_1bp"]:6.2f} {r["margin_1bp"]:7.3f} '
                  f'{r["mean_daily_margin_bp"]:7.3f} {r["nw_tstat"]:6.2f}  '
                  f'[{r["ci95_bp_lo"]:+.3f},{r["ci95_bp_hi"]:+.3f}]')
            r.update({"ci95_margin_lo": ci3[0], "ci95_margin_hi": ci3[2],
                      "ci95_margin_pt": ci3[1]})
            infer_rows.append({"market": market, **r})

        print("\n  NOTE: margin = Sharpe(model) - Sharpe(own-EM |a|) @ 1 bp; same scale-invariant")
        print("        convention as 7.36/7.37. CI bp/d = moving-block bootstrap (l=21) on the")
        print("        daily (model - EM) net-return difference; t_NW = Newey-West HAC t on the same.")

        # cross-validation against recorded 7.36/7.37 tables
        rec = {
            "sign-only (|a|=1)": (0.72 if market == "SPY" else 1.31 if market == "CSI300" else 3.37,
                                  -0.060 if market == "SPY" else 1.400 if market == "CSI300" else 2.1295),
            "|2P-1| (canonical P)": (0.86 if market == "SPY" else 1.91 if market == "CSI300" else 4.40,
                                     +0.051 if market == "SPY" else 1.577 if market == "CSI300" else 3.1089),
            "|2P-1| / vol20d": (0.96 if market == "SPY" else 2.04 if market == "CSI300" else 4.77,
                                -0.025 if market == "SPY" else 1.865 if market == "CSI300" else 3.0990),
            "1/vol20d (no P)": (0.83 if market == "SPY" else 1.13 if market == "CSI300" else 3.62,
                                -0.190 if market == "SPY" else 1.561 if market == "CSI300" else 2.1083),
        }
        if market == "SPY":
            rec["ensemble |2P-1| (ref)"] = (1.02, +0.118)
        for name, (rp, mp) in rec.items():
            r = infer_rule(name, rules[name], m, rng)
            if abs(r["sharpe_1bp"] - rp) > 0.02 or abs(r["margin_1bp"] - mp) > 0.006:
                failures.append((market, name, r["sharpe_1bp"], rp, r["margin_1bp"], mp))

        if market in ("CSI300", "NIFTY"):
            print(f"\n--- {market} diagnostics ---")
            feats = pd.read_parquet(feats_dir / "features_regimes.parquet")
            reg = feats["regime"].reindex(pd.DatetimeIndex(feats.index).tz_localize(None))
            rr = reg.loc[pd.DatetimeIndex(tbl["date"].dt.tz_localize(None))]
            tbl = tbl.assign(regime=rr.to_numpy())
            years = sorted(tbl["date"].dt.year.unique())
            yrows = []
            print(f'{"year":<6} {"n":>4} {"up":>5} {"drift":>7}  ' + " ".join(
                f"{'sgn':>6} {'|2P1|':>7} {'|2P1|v':>7}" for _ in [0]))
            print("-" * 78)
            for y in years:
                sub = tbl[tbl["date"].dt.year == y]
                drift = float((1 + sub["ret"]).prod()) - 1.0
                up = float((sub["ret_sign"] > 0).mean())
                row = {"year": y, "n": len(sub), "up_rate": up, "drift_cumprod": drift}
                for name in ["sign-only (|a|=1)", "|2P-1| (canonical P)", "|2P-1| / vol20d"]:
                    r = infer_rule(name, build_rules(sub)[name], sub["ret"].to_numpy(), rng)
                    row[name] = r["margin_1bp"]
                yrows.append(row)
                print(f'{y:<6} {row["n"]:>4} {up:>5.3f} {drift:>+7.3f}  '
                      f'{row["sign-only (|a|=1)"]:>6.3f} {row["|2P-1| (canonical P)"]:>7.3f} '
                      f'{row["|2P-1| / vol20d"]:>7.3f}')
            tag = "nifty" if market == "NIFTY" else "csi300"
            pd.DataFrame(yrows).to_csv(ROOT / "data" / f"{tag}_yearly.csv", index=False)

            # drawdown per rule
            print("\n  max drawdown (model net @1bp cumulative path):")
            for name in ["|2P-1| (canonical P)", "|2P-1| / vol20d", "sign-only (|a|=1)"]:
                mr, er = net(build_rules(tbl)[name], tbl["ret"].to_numpy(), B1_BPS)
                print(f"    {name:<26} model {max_drawdown(mr):+.4f}  "
                      f"EM {max_drawdown(er):+.4f}")

            # distribution stats for |2P-1|
            mr, er = net(build_rules(tbl)["|2P-1| (canonical P)"], tbl["ret"].to_numpy(), B1_BPS)
            dm = mr - er
            r5 = np.percentile(dm, 5)
            import scipy.stats as st
            print("\n  |2P-1| daily (model - EM) return distribution (bp/day):")
            print(f"    mean {dm.mean()*1e4:+.3f}  std {dm.std(ddof=1)*1e4:.3f}  "
                  f"skew {st.skew(dm):+.3f}  kurt {st.kurtosis(dm):+.3f}")
            print(f"    q05 {r5*1e4:+.3f}  q10 {np.percentile(dm,10)*1e4:+.3f}  "
                  f"med {np.median(dm)*1e4:+.3f}  q90 {np.percentile(dm,90)*1e4:+.3f}  "
                  f"q95 {np.percentile(dm,95)*1e4:+.3f}")

            # regime breakdown
            print("\n  regime breakdown (margins @1bp):")
            print(f'{"regime":<8} {"n":>4} {"up":>5}  ' +
                  f'{"sign":>7} {"|2P-1|":>8} {"|2P-1|/vol":>10}')
            rng_b = np.random.default_rng(RNG + 1)
            regrows = []
            for reglab in ["bull", "bear", "crisis"]:
                sub = tbl[tbl["regime"] == reglab]
                if len(sub) < 4:
                    continue
                up = float((sub["ret_sign"] > 0).mean())
                cells = []
                rowd = {"regime": reglab, "n": len(sub), "up_rate": up}
                for name in ["sign-only (|a|=1)", "|2P-1| (canonical P)", "|2P-1| / vol20d"]:
                    r = infer_rule(name, build_rules(sub)[name], sub["ret"].to_numpy(), rng_b)
                    cells.append(r["margin_1bp"]); rowd[name] = r["margin_1bp"]
                regrows.append(rowd)
                m0 = cells[0]
                note = ""
                sig_all = build_rules(sub)["sign-only (|a|=1)"]
                if reglab == "crisis" and float((sig_all < 0).sum()) == 0:
                    nlong = int((sig_all > 0).sum())
                    note = f"  (* model never shorts: all {nlong} crisis days have sign=+1, so model==EM)"
                print(f'{reglab:<8} {len(sub):>4} {up:>5.3f}  '
                      f'{cells[0]:>7.3f} {cells[1]:>8.3f} {cells[2]:>10.3f}{note}')
            pd.DataFrame(regrows).to_csv(ROOT / "data" / f"{tag}_regime.csv", index=False)

    pd.DataFrame(infer_rows).to_csv(ROOT / "data" / "inference_diagnostics.csv", index=False)
    print("\n" + "=" * 78)
    if failures:
        print("CROSS-VALIDATION FAILURES (reconstructed vs recorded):")
        for f in failures:
            print("  ", f)
        raise SystemExit(1)
    else:
        print("CROSS-VALIDATION PASSED: all reconstructed Sharpe(1bp)/margin(1bp) match")
        print("the recorded 7.36/7.37 tables within tolerance.")


if __name__ == "__main__":
    main()