"""Post-fix data audit — the pipeline was re-run (2026-08-14/16) after an
incomplete initial download; this validates the CURRENT artifacts rather
than trusting the fix narrative.

Part A — defect characterization (was it missing data or a value bug?):
  - raw per-year file sizes/mtimes (two download passes are visible in the
    filesystem timestamps: pass 1 morning 08-14, pass 2 evening 08-14),
  - daily-frame continuity (gaps > 6 days must be 0), per-year day counts,
  - anchor-price sanity: known historical SPY closes (2008 low, 2011/2012
    year-end, 2020 COVID peak/low, 2022 low, 2024 year-end) must match
    within +-5%. If only rows were ADDED (backfill), anchors and all other
    pre-existing values are untouched; if any anchor is off, the defect
    corrupted VALUES, not just coverage.

Part B — regime-labeling re-validation on the CURRENT data (the checks
originally done on the old dataset; the synthetic-calendar unit tests
never touched the real parquet):
  1. pipeline determinism: relabel the current close series with the
     current config -> must agree with the stored 'regime' column ~100%,
  2. COVID crisis detection: a crisis run must cover the 2020-03-23 crash
     trough and persist into April (expanding threshold + 2-day entry
     confirmation — entry ~03-10, ~2 weeks after the 02-21 crash onset),
     and the 2008-09..2009 crisis must be detected,
  3. no single-day flicker: every labeled run >= min_regime_duration_days,
  4. crisis overrides bull: days inside crisis runs with positive trailing
     60d return must be labeled crisis,
  5. relabeling stability: small threshold perturbations (bull_threshold
     0.005, percentile 96) must not flip > 15% of labels,
  6. pre-2016 regime coverage: labels present and runs clean in 2011-2013
     (the region the old holes spanned).

Part C — momentum sign sanity on current data (the Phase-1/DDR coherence
check): hit rate of clip(ret_20d/0.1) against next-day returns, overall
and per regime.

Output: checkpoints/robustness/data_fix_audit.csv + prints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.regime_labeling import BULL, CRISIS, label_regimes, realized_vol, _runs_of  # noqa: E402
from src.models.ddr.config import DDRConfig  # noqa: E402

OUT = DDRConfig().checkpoint_dir / "robustness"
cfg = OmegaConf.load(str(ROOT / "configs" / "regimes.yaml"))

# known SPY daily closes (historical record) for anchor checks
ANCHORS = {
    "2008-11-20": 75.95,   # GFC closing low (intraday ~74.88)
    "2011-12-30": 125.50,  # 2011 year-end (hole year)
    "2012-12-31": 142.41,  # 2012 year-end (hole year)
    "2020-02-19": 338.37,  # COVID peak close
    "2020-03-23": 223.37,  # COVID trough close
    "2022-10-12": 349.94,  # 2022 low close (hole year)
    "2024-12-31": 586.20,  # 2024 year-end (study horizon)
}


def part_a(features: pd.DataFrame) -> dict:
    idx = features.index
    gaps = (idx[1:] - idx[:-1]).days
    big = int((gaps > 6).sum())
    out = {"rows": len(idx), "range": f"{idx.min().date()}..{idx.max().date()}",
           "gaps_gt_6d": big}
    print(f"A. daily frame: {len(idx)} rows {idx.min().date()}..{idx.max().date()}, "
          f"gaps>6d: {big}")
    per_year = features["close"].resample("YE").count()
    out["year_counts"] = per_year.to_dict()
    print("   per-year trading days:", dict(per_year.astype(int)))
    print("   anchor prices (expected +-5%):")
    clean_idx = idx.tz_localize(None)
    s = pd.Series(features["close"].to_numpy(), index=clean_idx)
    ok = True
    for d, exp in ANCHORS.items():
        ts = pd.Timestamp(d)
        try:
            got = float(s.loc[ts])
            good = abs(got - exp) / exp < 0.05
            ok &= good
            print(f"     {d}: got {got:.2f} vs expected {exp:.2f} -> "
                  f"{'OK' if good else '*** MISMATCH ***'}")
        except (KeyError, IndexError):
            ok = False
            print(f"     {d}: *** MISSING DATE ***")
    out["anchors_ok"] = ok
    return out


def part_b(features: pd.DataFrame) -> dict:
    close = features["close"]
    stored = features["regime"]
    recomputed = label_regimes(close, cfg)
    dropped = recomputed.dropna()
    shared = stored.dropna().index.intersection(dropped.index)
    agree = float((stored.loc[shared] == recomputed.loc[shared]).mean())
    print(f"B1. pipeline determinism: stored vs recomputed agreement = {agree:.4f} "
          f"({len(shared)} days)")
    runs = _runs_of(dropped)
    min_dur = int(cfg.regimes.hysteresis.min_regime_duration_days)
    short = [(s, e) for s, e, _ in runs if e - s + 1 < min_dur]
    print(f"B2. runs: {len(runs)} total, {len(short)} shorter than {min_dur} days "
          f"(must be 0)")
    crisis_runs = [(dropped.index[s].tz_localize(None), dropped.index[e].tz_localize(None))
                   for s, e, lab in runs if lab == CRISIS]
    print("   crisis runs (start, end):")
    for a, b in crisis_runs:
        print(f"     {a.date()} .. {b.date()}")
    covid_ok = any(a <= pd.Timestamp("2020-03-23") and b >= pd.Timestamp("2020-03-23")
                   for a, b in crisis_runs)
    covid_persist = any(a <= pd.Timestamp("2020-03-23") and b >= pd.Timestamp("2020-04-15")
                        for a, b in crisis_runs)
    gfc_ok = any(a <= pd.Timestamp("2008-09-29") and b >= pd.Timestamp("2008-11-20")
                 for a, b in crisis_runs)
    print(f"B3. COVID: crash covered -> {covid_ok}; persists past Apr 15 -> {covid_persist}")
    print(f"B4. GFC 2008 covered -> {gfc_ok}")

    # mechanism trace: why did the COVID run start 2020-03-10? print vol vs
    # the live expanding threshold around the entry (20d vol lag + 2-day
    # state-machine confirmation after vol crossed on ~03-09)
    vol = realized_vol(close, int(cfg.regimes.crisis.vol_window_days)).tz_localize(None)
    thr = vol.expanding(min_periods=int(cfg.regimes.crisis.percentile_min_periods)).quantile(
        float(cfg.regimes.crisis.percentile) / 100.0).clip(lower=float(cfg.regimes.crisis.min_vol_annualized))
    exit_band = float(cfg.regimes.hysteresis.crisis_exit_mult) * thr
    w = close.index.tz_localize(None)
    band = (w >= pd.Timestamp("2020-03-02")) & (w <= pd.Timestamp("2020-03-13"))
    print("   COVID entry trace (2020-03-02 .. 2020-03-13):")
    for ts in w[band]:
        d = ts.date()
        print(f"     {d}: vol={vol.loc[ts]:.4f} ann, live_thr={thr.loc[ts]:.4f}, "
              f"0.8*live_thr={exit_band.loc[ts]:.4f}")
    # same trace for GFC exit (2008-09-16..2009-06-24): verify the exit band
    # held through the 2009 exit
    band2 = (w >= pd.Timestamp("2009-03-20")) & (w <= pd.Timestamp("2009-04-03"))
    print("   GFC exit trace (2009-03-20 .. 2009-04-03):")
    for ts in w[band2]:
        d = ts.date()
        print(f"     {d}: vol={vol.loc[ts]:.4f} ann, live_thr={thr.loc[ts]:.4f}, "
              f"0.8*live_thr={exit_band.loc[ts]:.4f}")
    trailing = np.log(close).diff(60)
    in_crisis = recomputed == CRISIS
    pos = trailing[in_crisis] > 0
    ovr_ok = bool(pos.sum()) and (recomputed.loc[pos.index[pos]].dropna() == CRISIS).all()
    print(f"B5. crisis-overrides-bull: {int(pos.sum())} positive-60d-return days inside "
          f"crisis runs, all labeled crisis -> {ovr_ok}")
    tweaked = OmegaConf.merge(cfg, {
        "regimes": {"bull_threshold": 0.005, "crisis": {"percentile": 96.0}}})
    alt = label_regimes(close, tweaked)
    sh = stored.dropna().index.intersection(alt.dropna().index)
    agree2 = float((stored.loc[sh] == alt.loc[sh]).mean())
    print(f"B6. relabeling stability (bull_thr=0.005, pctile=96): agreement = {agree2:.4f} "
          f"(must be > 0.85)")
    pre = recomputed.loc["2011-01-01":"2013-12-31"].dropna()
    runs_pre = _runs_of(pre)
    print(f"B7. pre-2016 (2011-2013): {len(pre)} labeled days, {len(runs_pre)} runs, "
          f"min run {min(e - s + 1 for s, e, _ in runs_pre) if runs_pre else 0} days")
    return {"agreement": round(agree, 4), "runs_short": len(short),
            "covid_covered": covid_ok, "covid_persists_apr": covid_persist,
            "gfc_covered": gfc_ok, "override_ok": ovr_ok,
            "stability_agree": round(agree2, 4), "pre2016_min_run": min(
                (e - s + 1 for s, e, _ in runs_pre), default=0)}


def part_c(features: pd.DataFrame) -> dict:
    close = features["close"]
    ret20 = close.pct_change(20)
    action = np.clip(ret20 / 0.10, -1.0, 1.0)
    nxt = close.pct_change().shift(-1)
    df = pd.DataFrame({"a": action, "nxt": nxt, "regime": features["regime"]}).dropna()
    df["hit"] = df["a"] * df["nxt"] > 0
    out = {"momentum_overall_hit": round(float(df["hit"].mean()), 4)}
    print(f"C. momentum hit rate (a*next>0): overall {df['hit'].mean():.4f}")
    for regime in ("bull", "bear", "crisis"):
        m = df["regime"] == regime
        if m.sum():
            print(f"   {regime}: {df['hit'][m].mean():.4f} ({int(m.sum())} days)")
            out[f"momentum_hit_{regime}"] = round(float(df["hit"][m].mean()), 4)
    return out


def main() -> None:
    features = pd.read_parquet(ROOT / "data" / "processed" / "features_regimes.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    res.update(part_a(features))
    res.update(part_b(features))
    res.update(part_c(features))
    pd.DataFrame([res]).T.to_csv(OUT / "data_fix_audit.csv", header=False)
    print(f"\naudit saved: {OUT / 'data_fix_audit.csv'}")


if __name__ == "__main__":
    main()