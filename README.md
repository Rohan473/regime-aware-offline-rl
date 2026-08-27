# Regime-Aware Uncertainty Modeling for Offline RL in Financial Markets

Research codebase comparing offline-RL trading models on identical SPY data
and a shared, regime-conditional evaluation (train <= 2018-12-31, val
2019-2020, test 2021-2024; per-seed EM margins reported alongside every
win count — the 7.9.2 protocol rule):

- **Model B** — DDR-style recurrent baseline: a GRU policy trained by direct
  backprop through the Differential Sharpe Ratio (Moody & Saffell 1998).
  Canonical = naive DSR retrained on the current hole-free data; pack mean
  test Sharpe 0.99 ± 0.24 (good-basin pack 1.10 ± 0.04), EM margins +0.19.
- **Model C** — TACR-style Transformer actor-critic (Lee & Moon, IEEE Access
  2023). As a standalone actor it is the project's structural-failure case
  (below EM, negative margins, val->test collapse), despite clipped double-Q
  and family-balanced samplers. The paper's deviations are documented in
  `configs/tacr.yaml` and PROJECT_NOTES 7.6.
- **Model C+** (the project's best) — a HYBRID that decouples the two skills
  TACR splits: the Transformer supplies **leverage timing** (|a|, from the
  seed-stable double-Q/nr7 pack), and a simple L2 logistic regression on
  **8 SPY + 8 macro/cross-asset features** supplies **directional sign**.
  Pack mean test Sharpe **1.15**, EM margins **5/5 (+0.30)** — the first
  configuration to clear the ≥3/5 margin bar and to beat Model B. Robust
  under a shifted-split check; the edge is strongest in crisis windows.
  PROJECT_NOTES 7.17-7.22.
- **Model D** — fuzzy-uncertainty ablation: a fixed interval type-2 fuzzy
  layer + causal transformer + IQL. Stable (no collapse, thin positive
  margins) but the fuzzy layer is null vs its ablation.

No real model beats buy-and-hold on raw cumulative returns alone (Model C+:
test-window cum +0.26..+0.69, mean ~+0.49, vs B&H +0.59 at 100% exposure)
— the project's edge claims rest on the exposure-matched margin (Sharpe of
the strategy minus Sharpe of its own long-only sizing control), which is
scale-invariant and isolates directional skill from position sizing.

## Environment

Python 3.12.5 (the only interpreter on this machine; project requires-python is
`>=3.11,<3.13`). Install with:

```
python -m pip install -r requirements.txt
pip install -e .
```

### pandas-ta-openbb on Python 3.12: reproducible compatibility fix

`pandas-ta-openbb==0.4.24` has an upstream bug on Python 3.12:
`pandas_ta/maps.py` uses `importlib.metadata` without importing the
submodule, so `import pandas_ta` raises
`AttributeError: module 'importlib' has no attribute 'metadata'`.

The fix is fully reproducible — no manual site-packages editing:

1. **Runtime fix (automatic, no install step):**
   `src/data/_pandas_ta_compat.py` does `import importlib.metadata` before
   importing `pandas_ta`, which binds the submodule on the shared `importlib`
   module object. The pipeline imports pandas_ta through this module, and a
   probe turns any residual failure into a loud error with the remedy.
2. **Patch script (one command, for direct imports anywhere):**
   `python scripts/apply_patches.py` patches the installed
   `pandas_ta/maps.py` (single added import line, marker-based idempotent,
   verifies with a fresh `import pandas_ta` afterwards, restores on failure).
   Run it once after `pip install -r requirements.txt` so notebooks and any
   direct `import pandas_ta` work too.
3. **Pin:** `pandas-ta-openbb==0.4.24` in `requirements.txt`/`pyproject.toml`.

No step depends on a human reading a README note before the breakage happens:
the pipeline path fixes itself at import time and fails loudly otherwise.

## Data

- Raw source: the downloader writes yearly `SPY_YYYY.parquet` files of 1-minute
  bars into the repo root (2005..2026, backfilled and hole-free; see the data
  integrity audit below).
- `src/data/loaders.py` reads all matching files, resamples minute bars to daily
  OHLCV (grouped by America/New_York trading date), and caches to
  `data/processed/daily_ohlcv.parquet` guarded by a manifest of input file
  name/size/mtime + resample config. The cache rebuilds automatically when raw
  files change (e.g. the downloader appends a year).
- Missing years inside the configured range only warn — re-running the pipeline
  after the download finishes extends the range.
- Missing/corrupt raw data raises loudly; the pipeline never substitutes
  synthetic data. Synthetic data is used ONLY in pytest.

## Features (capped at 10 by design; do not add without asking)

`ret_1d, ret_5d, ret_20d, realized_vol_20d, rsi_14, macd_hist,
volume_zscore_20d, bollinger_pos` (8 total). RL state = causal expanding
z-scores of these (no lookahead). Raw features stay in
`data/processed/features_regimes.parquet` for inspection.

## Macro / cross-asset features (for the Model C+ sign head)

`data/macro/*.parquet` holds daily series fetched from Yahoo's public chart
API (`scripts/fetch_macro.py`, keyless, manifested): TLT (20y+), ^TNX (10y
yield), ^VIX, ^VIX3M, DX-Y.NYB (dollar), HYG, QQQ, IWM, and ^SKEW (sentiment).
`src/data/macro_factors.py` turns them into 8 causal features aligned to the
SPY calendar (plus `sentiment_skew`): risk-on spread (SPY−TLT), 10y-yield
delta (1d/5d), vol term structure (VIX3M−VIX), 20d SPY−DXY correlation,
credit proxy (HYG−TLT, price-based), and SPY−QQQ / SPY−IWM relative strength.
These live ONLY in the C+ linear sign model — the four-model state stays 8-d
(adding them to the TACR/B/D state is a deferred, protocol-breaking decision).

## Regimes: {bull, bear, crisis}

- bull/bear: trailing 60d return vs thresholds (`regimes.yaml`); sign fallback
  inside the band.
- crisis: 20d realized vol (annualized) >= 95th percentile of **all vol
  history up to that day** (expanding window, boots after
  `percentile_min_periods` = 300 days), plus an absolute floor. Crisis
  overrides bull/bear.
- Hysteresis: crisis entry needs `crisis_confirmation_days` consecutive days;
  on entry the vol threshold is **frozen** (the *live* expanding-window
  percentile rises as crisis days accumulate — a long crisis like 2008 would
  otherwise self-extinguish against its own elevated bar); exit needs the same
  streak below `exit_mult * frozen_threshold`. Then any label run shorter than
  `min_regime_duration_days` is merged into its neighbor (shortest-first),
  eliminating single-day flicker.

## Behavior policies

`momentum`, `mean_reversion`, `buy_and_hold`, `random` — each rolled over the
same daily series, each row tagged with its generating policy (needed later to
check whether a learned policy just mimics the dominant behavior policy).
Action = scalar position in [-1, 1] (shorting enabled: -1 = fully short,
+1 = fully long, config-driven).

`momentum` and `mean_reversion` are FAMILIES, expanded into one policy per
configured lookback window (tagged `<family>_<N>d` in the offline dataset):
- momentum windows `{30, 40, 60, 80, 100, 120, 150, 200, 250}` days — longs
  on positive trailing returns, shorts on negative, scaled by signal strength;
- mean_reversion windows `{1..20}` days — the symmetric contrarian
  (inverse of momentum on short windows).

`nr7` is a SPECIALIZED SIGNAL FAMILY (one trajectory, not window-expanded):
the classic NR7 narrow-range breakout, daily-close approximation. A day is
NR7 when its high-low range is the narrowest of the last 7 trading days; at
decision date t, if YESTERDAY was an NR7 day, go long (+1) when close[t]
breaks above that day's high, short (-1) below its low, flat (0) otherwise
(inside the range or no signal). Clean Long/Short/Flat vector, ~10.5% of
days active.

The per-window scale follows `scale_N = base * sqrt(N / ref_window)` so the
position distribution is comparable across horizons (cumulative-return std
grows ~ sqrt(N)). The multi-horizon returns are computed directly from the
daily close and are NOT added to the state — the RL state stays the 8
Phase-1 features. Reward = `a_t * ret_{t+1} - cost * |a_t|` (cost in bps,
default 0). Config: `configs/data.yaml` -> `behavior_policies`.

This yields 32 policy trajectories (9 momentum + 20 mean_reversion +
buy_and_hold + random + nr7). Because windowed families start after their
lookback warm-up, models C/D loaders intersect dates across the selected
policies (only dates every trajectory covers are used).

**NR7 + TACR result (PROJECT_NOTES 7.15)** — tested with double-Q and the
original uniform sampler per the pre-registered protocol: double-Q **min**
+ uniform + nr7 gives the most seed-stable TACR pack yet (test Sharpe
0.82 ± 0.15) but with ~zero EM margins (long-only ties — the 7.9.2
knife-edge, stability without skill); double-Q **mean** + uniform + nr7 is
the worst variant measured (0/5 vs EM, one degenerate always-short seed).
The natural-diversification hypothesis is not supported in margin terms;
no TACR configuration on the multi-window data clears the ≥3/5 EM-margin
bar.

## Run the pipeline end to end

```
python -m src.data.pipeline
```

Writes (all under `data/processed/`, gitignored):
- `daily_ohlcv.parquet` + `daily_ohlcv.manifest.json`
- `features_regimes.parquet` — OHLCV + raw features + `regime` column
- `offline_dataset.parquet` — transitions `(state, action, reward, next_state,
  done)` + `policy` + `date` + `regime` tags, state dim 8
- `dataset_manifest.json` — provenance + summary stats
- prints a summary table to stdout

## Run the tests

```
pytest
```

Covers: regime non-degeneracy (all three regimes appear), no single-day
flicker, crisis priority, entry confirmation, frozen-threshold exit band
(including persistence through a long crisis under the expanding threshold),
NaN leakage, action diversity (no >50% policy dominance), shorting occurs
(negative actions from momentum/mean_reversion/random), reward formula, done
flags, next-state consistency, minute->daily resampling.

## Notebooks

- `notebooks/01_data_sanity_check.ipynb` — loads the processed artifacts and
  plots regime timeline + dataset stats.
- `notebooks/02_ddr_training.ipynb` — trains Model B (DDR), shows training
  curves, evaluates with the regime breakdown.

## Model B: DDR baseline (Phase 2)

DDR-style direct reinforcement: a single-layer GRU/LSTM policy reads the
trailing `WINDOW_DAYS` (= 20, shared constant in `src/models/__init__.py`)
window of the 8 z-scored Phase-1 features and outputs one action in [-1, 1]
via tanh. Training signal is the Differential Sharpe Ratio (Moody & Saffell
1998) — exact EMA recurrence, reward per step is D_t, optimized by direct
backprop through the DSR (no REINFORCE).

**Offline adaptation (explicit):** original DDR is online/on-policy. Here the
policy is trained on the OFFLINE state sequence (feature windows from
`features_regimes.parquet`) but generates its OWN actions; the DSR reward is
computed from those own actions against the REALIZED next-day market returns
in that state. The logged behavior-policy actions and rewards are never used
— DDR learns from the logged market path, not the logged actions. (All four
Phase-1 policies share one market path, so the buy_and_hold + momentum
selection is the full unique date sequence.)

Time split (never shuffled): train <= 2018-12-31, val 2019-2020, test 2021+.
Run:

```
python -m src.models.ddr.train     # trains, saves ddr_best.pt + training_log.csv
python -m src.models.ddr.eval      # test-set roll -> regime_eval.csv + test_predictions.csv
```

Outputs land in `src/models/ddr/checkpoints/` (gitignored). The regime
breakdown is the PRIMARY eval result (shared `src/eval/regime_eval.py`, also
used by Models C/D); blended Sharpe is secondary. Regime labels are joined
by date from Phase 1 -- never recomputed, never part of model input.

### Data integrity: holes in the daily frame (must-read)

**STATUS UPDATE (2026-08-17): the pipeline has been re-run since this
audit — the CURRENT data is hole-free.** `features_regimes.parquet` now has
0 gaps > 6 calendar days, spans 2005-01-03 -> 2026-03-31 (5,344 rows), and
`offline_dataset.parquet` has 168,516 rows to 2026-03-30 (32 window-expanded
behavior policies incl. nr7; see the Behavior policies section). The study horizon
is capped at 2024-12-31 (`SPLIT_TEST_END`); `load_ddr_data` clips the path
to it (310 dates beyond are dropped with a stderr warning). Consequences:

- The valid test set is now **1,005 days** (769 bull / 221 bear / 15
  crisis), all valid — no hole-boundary days exist anymore. The
  516-valid / 258-clean-label day-sets below describe the OLD
  contaminated data and are historical only.
- The crisis regime is now a REAL label (15 test days, 527 all-time);
  the "Aug-2022 crisis is 100% artifact" finding applied to the old
  data and is superseded.
- Everything in the audit chain below (clean-label numbers, the
  leverage decomposition, the "no directional signal" verdict) was
  measured on the old contaminated data. **The verdict does not carry
  to the current data** — see the vol-targeted DSR section: retrained
  on the hole-free data, the naive DSR beats the exposure-matched
  control 4/5 seeds. The old null result was a data artifact, not a
  signal problem.
- The post-fix data audit (`scripts/audit_data_fix.py`, results in
  `checkpoints/robustness/data_fix_audit.csv`) PASSES: defect = missing
  coverage backfilled (two download passes in raw-file mtimes), all
  anchor prices match history, stored regime labels re-validate 1.0000
  against relabeling, COVID/GFC crisis runs detected, relabeling stable.

The original audit (historical record):

The Phase-1 daily frame had 6 multi-month missing chunks (2011-04..2012-12,
2018-04..07, 2021-04..05, 2021-09..12, 2022-04..07, 2022-09..10, 2023-10).
`pct_change()` collapses each hole into ONE fake "daily" return (e.g. the
"2022-03-31 market -9.4%" day is really Mar 31 -> Aug 1 2022). Consequences:

- **The ENTIRE test-window crisis bucket is a hole artifact.** All 20 test
  crisis days are Aug 2-29 2022, labeled crisis only because the fake vol
  spike inflated 20-day realized vol. Recomputing labels with the fake
  returns replaced by typical returns yields **0** crisis days in Aug-Sep
  2022 -- and **0 crisis days exist in the test window at all** on
  clean labels. Any Aug-2022 crisis Sharpe is meaningless.
- The early "bear Sharpe -0.21" included two fake hole days; the later
  "bear +1.68" was computed on labels that are 89% hole-contaminated.
  Neither number is reported anywhere as evidence.
- DDR-side mitigation (in-scope, `src/models/ddr/`): every decision day
  whose next-day return spans a hole is flagged invalid
  (`compute_valid_mask`, gap threshold 6 calendar days -- genuine holiday
  closures are 1-5 days) and excluded from training loss, val/test Sharpe,
  the regime table, and baselines. 7 invalid test-adjacent days: 2011-03-31,
  2018-03-29, 2021-03-31, 2021-08-31, 2022-03-31, 2022-08-31, 2023-09-29
  (5 fall in the test split). Phase-1 labels/artifacts were NOT rewritten
  (out of scope); the downloader should backfill the holes.
- **Residual contamination beyond the mask:** the mask removes boundary
  DAYS only; regime LABELS themselves are computed on hole-spanning
  windows for 89% of bear days and 35% of bull days in the test split.
  The clean-label day-set (258 days) is the only evidence used in this
  document (see below); the 378/118-day regime table is deprecated.

### Robustness evidence (scripts/ddr_robustness.py, scripts/ddr_paired_audit.py, scripts/ddr_clean_label_eval.py)

All test metrics exclude hole-boundary days. Seeds 20260814, 1, 2, 3, 4;
epochs 30, best-val checkpoint selection.

**CLEAN-LABEL evaluation (primary; the only regime numbers reported).** On
the 258 clean-label test days (245 bull, 13 bear, **0 crisis** — the
"crisis" regime does not exist in the test window; the 20-day Aug-2022
crisis run was fabricated by the data holes):

| dayset | n | DDR Sharpe (mean +/- std) | DDR cum | DDR max DD | B&H Sharpe | B&H cum | B&H max DD |
|--------|---|---------------------------|---------|-----------|------------|---------|-----------|
| bull   | 245 | 1.70 +/- 0.16 | +7.4% | -2.1% | 1.29 | +13.0%* | -8.0% |
| bear   | 13 | not reported (underpowered) | | | | | |
| all    | 258 | 1.46 +/- 0.11 | +7.1% | -2.6% | 1.14 | +16.4% | -8.0% |

*B&H bull cum implied from all-cum minus bear-leg; per-regime B&H cum is in
`baselines_clean_label.csv`. The 378/118-day regime table in the earlier
sweep artifacts is CONTAMINATED (labels computed on hole-spanning windows:
89% of bear days, 35% of bull days) and must not be cited.

**Leverage-matched decomposition (scripts/ddr_leverage_test.csv).** DDR's
Sharpe gap over B&H is entirely position SIZING, not signal. Four controls
on the same clean-label days, per seed: B&H (long 1.0); CL (constant
leverage mean(|a|) ~ 0.40 -- Sharpe scale-invariant, equals B&H by
identity, verified); EM (exposure-matched: DDR's day-by-day position size
|a_t|, always long); DDR.

| policy | bull Sharpe (mean+/-std) | all Sharpe (mean+/-std) |
|--------|--------------------------|-------------------------|
| B&H / CL | 1.29 | 1.14 |
| EM (sizing only, no shorts) | 2.02 +/- 0.28 | 1.74 +/- 0.23 |
| DDR (sizing + shorts) | 1.70 +/- 0.16 | 1.46 +/- 0.11 |

EM beats DDR in 5/5 seeds, and beats B&H by ~0.7 Sharpe. The per-day paired
DDR-minus-EM difference is negative (-0.006%/day, t=-1.31, p=0.19): the
model's SIGN decisions (shorts) reduce Sharpe; its sizing pattern is the
entire source of the apparent advantage. Max drawdown of EM (-2.1% bull,
-2.6% all) equals DDR's to 4 decimals -- **the drawdown reduction is 100%
mechanical de-leveraging, not risk management.** A leverage-matched B&H
closes the gap and then some. Finding, stated flatly: **DDR is a
lower-variance, lower-return policy whose only "edge" over buy-and-hold is
the average position size it happens to take. There is no evidence of
directional signal.**

**Why the DSR converged there -- TWO separate findings, do not conflate**
(training_log.csv; per-seed deployed numbers from the best-val checkpoints):

1. **Exposure capping is already present at the DEPLOYED epoch.** At the
   best-val checkpoints (epochs 5-8, the models every number in this
   document came from), train mean |action| is 0.26-0.35 and short
   fraction 0.04-0.15: the policy has already de-levered to ~1/3 of the
   [-1,1] range before overfitting even begins. The null result is
   produced by this early, non-degenerate checkpoint -- the DSR reward's
   variance-reduction bias is active from the start. On retrain, DSR
   reward misalignment (exposure capping) remains hypothesis #1; a
   bigger GRU will not fix it.
2. **The short-flip pathology (short_frac 7% -> 37%, train DSR EMA Sharpe
   exploding to 8.5, val decaying 1.18 -> 0.46 over epochs 20-30) is a
   SEPARATE post-val overfit artifact that best-val selection correctly
   discards.** It explains the training dynamics, not the null result --
   no deployed checkpoint exhibits it (max short_frac at best-val is
   0.15). Do not cite it as the cause of the null result.

**Paired returns (DDR minus buy-and-hold, same days, per seed):** on the
full valid path, bull mean diff -0.047%/day (t=-1.32, p=0.19), bear
-0.052%/day (t=-0.73, p=0.47) -- 0 of 5 seeds beat B&H on returns in
either regime (all 10 seed x regime diffs negative). On clean-label days
DDR also trails on returns (+7.1% vs +16.4% cum). DDR does not beat
buy-and-hold on returns anywhere.

**Momentum baseline:** sign-consistent 100% by construction (action =
clip(ret_20d/scale)); its crisis +1.47 was a full-short position on the
fake Aug-2022 "crisis" labels, with 90% of its actions decided on
hole-spanning windows. On clean-label days momentum is weak (all Sharpe
0.08, bull 0.25, max DD -14.5%). Mechanically coherent, but contaminated
inputs near holes.

Hyperparameters: eta=0.01 and lr=1e-3 are Moody & Saffell / Adam DEFAULTS,
not tuned. A 2x2 grid (eta in {0.01, 0.05} x lr in {1e-3, 3e-4}) keeps test
blended Sharpe in 0.99-1.13 -- the default is not a knife-edge. Best val
epoch is consistently 5-8 across seeds; after ~epoch 11 val Sharpe decays
monotonically (train EMA Sharpe explodes to 8.5) -- best-val selection is
mandatory. Full artifacts: `checkpoints/robustness/` (seed_sweep.csv,
seed_sweep_summary.csv, grid.csv, baselines_test.csv, bear_pnl.csv,
gaps_report.csv, paired_summary.csv, paired_ddr_vs_bh.csv,
contamination_audit.csv, momentum_audit.csv, clean_label_regimes.csv,
paired_drawdown.csv, baselines_clean_label.csv, fully_clean_subset.csv).

### Volatility-targeted DSR (reward fix) — run on the CURRENT hole-free data

`configs/ddr.yaml` (shared config; new fields `vol_targeting`,
`target_vol: 0.15`, `vol_target_window: 20`, `max_leverage: 2.0`).
Mechanism (`VolTargetBuffer`, src/models/ddr/dsr.py): before each DSR
increment, the action is scaled so the strategy-return series runs at
target annualized vol — `a'_t = clip(a_t * target_vol/vol_t, -2, +2)`,
`vol_t` = causal trailing 20-day sample std of `a*ret` (detached; scale=1
during the 20-step warmup). Because shrinking |a| shrinks BOTH the returns
AND the measured vol, the scaled series is invariant to uniform
de-leveraging: the policy can no longer game the DSR reward by shrinking
exposure. Val selection and the test P&L stay on RAW actions (shaping is
training-only).

Retrained 5 seeds x 30 epochs on the current data (checkpoints/vt/s{seed}/;
old naive checkpoints untouched). For a fair side-by-side, the naive DSR
was ALSO retrained on the same current data (checkpoints/naive_new/s{seed}/;
`naive_old` = the historical per-seed runs trained on the contaminated
data). Leverage decomposition on valid test days (1,005; clean-label
exclusions are moot on hole-free data), `scripts/ddr_vt_retrain.py`:

| run | policy | all Sharpe (mean+/-std) | bull Sharpe (mean+/-std) |
|-----|--------|--------------------------|--------------------------|
| (any) | B&H / CL | 0.78 | 1.24 |
| naive_old | EM | 0.92 +/- 0.04 | 1.47 +/- 0.14 |
| naive_old | DDR | 1.05 +/- 0.06 | 1.64 +/- 0.11 |
| naive_new | EM | 0.80 +/- 0.11 | 1.34 +/- 0.24 |
| naive_new | DDR | 0.99 +/- 0.24 | 1.51 +/- 0.54 |
| voltarget | EM | 0.80 +/- 0.14 | 1.45 +/- 0.20 |
| voltarget | DDR | 0.81 +/- 0.12 | 1.32 +/- 0.18 |

Verdict (criterion: DDR must beat the exposure-matched control in >= 3/5
seeds): **vol-targeted DDR beats EM 4/5 on all days (1/5 on bull) — PASS,
but barely, and it is WORSE than the naive reward retrained on the same
data (voltarget minus naive_new = -0.18 Sharpe/seed, 1/5 positive).** The
naive DSR itself beats EM 4/5 (all) and 4/5 (bull) on the current data.
Three conclusions, in order of importance:

1. **The old "no directional signal" verdict was a data artifact.** On the
   hole-free data the GRU extracts real signal with EITHER reward shape —
   the de-leveraged null never reproduces.
2. **Vol-targeting does NOT restore full exposure.** vt policies run
   mean |a| 0.14-0.23 (vs naive 0.27-0.37) with short_frac 0.22-0.49 (vs
   0.02-0.11). The invariance argument removes the incentive to shrink,
   but the DSR gradient carries no exposure pressure either — the level
   drifts (down, empirically). Reward shaping cannot fix the exposure
   level by construction; only a level-aware objective (e.g. an explicit
   vol-targeted P&L loss) could.
3. **Vol-targeting alone is insufficient; explicit exposure regularization
   is untested.** The fix is not harmful (still beats EM 4/5, max DD -5.3%
   vs B&H -25.3%) but it is not the lever that matters on good data.
   Citable finding: risk-normalization changes the incentive landscape but
   adds no countervailing incentive toward using leverage, so the policy
   drifts to whatever level minimizes variance in a different feature
   space. If the signal were missing, the next lever is the feature set /
   window — further reward-shape iteration is on hold until an explicit
   level-aware objective is tried, not closed.

#### Canonical Model B (decision, 2026-08-17): naive_new, NOT voltarget

The final B-vs-C-vs-D comparison uses **naive_new** (naive DSR retrained
on the current hole-free data) as Model B:
- It is the stronger baseline — beats the exposure-matched control 4/5
  seeds (all) and 4/5 (bull), and beats voltarget by -0.18 Sharpe/seed
  (1/5 positive). Using voltarget as B would flatter D artificially.
- Voltarget stays as a documented side-experiment (reward fix tried and
  characterized; see above), not a contender for the B slot.
- Canonical artifact: `checkpoints/naive_new/s20260814/ddr_best.pt`
  (reproducible: `python -m src.models.ddr.train`).

**Seed variance note (must-read for C and D):** the naive_new 5-seed
spread (0.99 ± 0.24) is NOT Gaussian noise — it is dominated by one rare
alternative basin. Seeds 20260814/1/2/3 converge to one shared policy
(best-val @ epochs 3-5, pairwise test-prediction corr 0.93-0.98): pack
mean all 1.10 ± 0.04, bull 1.77 ± 0.08 (test). Seed 4 is a different,
worse-testing basin (test all 0.51, loses to EM; bull 0.44): its val
KEPT rising to epoch 30 (the pack's decays after ~5 — overfitting), its
policy correlates only 0.33 with the pack, and its exposure profile
differs (mean|a| 0.33, short_frac 0.30 vs 0.23-0.26 / 0.14-0.22). A
5-seed probe (seeds 5-9, `checkpoints/naive_probe/`) found 0 more such
runs — basin frequency 1/10, i.e. rare but real. Screening rule for
C/D runs: if best-val is selected at epoch > 10 (every good-basin run
peaks at epochs 3-5), treat the run as suspect and reseed. Best-val
transfer holds in the good basin (val 1.03-1.18 -> test 1.01-1.15) and
broke in the seed-4 basin (0.87 val -> 0.51 test).

### Post-fix data audit (scripts/audit_data_fix.py) — PASS

The pipeline was externally re-run (2026-08-14/16) after an incomplete
initial download; `audit_data_fix.py` validates the CURRENT artifacts:
- **Defect characterization:** two download passes visible in raw file
  mtimes (08-14 morning: 2005-2010, 2012-2017, 2019-2020; evening: 2011,
  2018, 2021-2026 — exactly the old hole years + forward extension). The
  defect was missing coverage (backfilled rows), not value corruption: all
  7 anchor closes match history within +-5% (incl. pre-2016 hole years
  2011-12-30 = 125.43, 2012-12-31 = 142.55; 2008-11-20 = 75.95 exact),
  per-year day counts all 250-253, zero gaps > 6 days.
- **Regime labels re-validated on the real parquet** (not on faith):
  relabeled series agrees with the stored `regime` column 1.0000 on 5,284
  days; 74 runs, none shorter than min duration; COVID crisis run
  2020-03-10..2020-05-28 covers the 03-23 trough and persists past Apr 15
  (the ~2-week lag from the 02-21 onset is the 20d vol smoothing + 2-day
  confirmation, not a regression); GFC 2008-09-16..2009-06-24; 109
  positive-return days inside crisis runs all labeled crisis;
  relabeling stability 0.9873 under threshold perturbation; pre-2016
  (2011-2013) clean (11 runs, min 13 days).
- Momentum sign sanity holds on current data (overall hit 0.5215).
- Caveat: the test window has only 15 crisis days — the crisis-regime
  Sharpe is directionally suggestive at best and is not a finding.

Artifacts: `checkpoints/vt/leverage_test.csv` (3-run table),
`checkpoints/vt/s{seed}/ddr_best.pt` + `training_log.csv`,
`checkpoints/naive_new/s{seed}/` (control), `checkpoints/robustness/
data_fix_audit.csv`. `train`/`eval` CLI now load
`configs/ddr.yaml` by default (vol-targeted artifacts under
`checkpoints/vt`); pass `--no-vol-targeting` / `--config` to reproduce
naive behavior.

## Config

- `configs/data.yaml` — raw data location, universe, resample, features, dataset
  cost, behavior policy windows/scales/seed (window-expanded families + nr7).
- `configs/regimes.yaml` — all regime thresholds + hysteresis parameters.
  Nothing regime-related is hardcoded.
- `configs/ddr.yaml` — Model B (DDR) shared config (vol-targeting + artifacts).
- `configs/tacr.yaml` — Model C (TACR) shared config; paper values with the
  flagged deviations from PROJECT_NOTES 7.6 plus the structural-fix /
  sampler / exclusion fields (`use_double_q`, `double_q_mode`,
  `balanced_families`, `exclude_policies`, `use_bcq`/`use_par`, ...).

## Model C: TACR baseline (Phase 3) + Model C+ hybrid (the project's best)

Reproduction of Lee & Moon, "Transformer Actor-Critic with Regularization:
Automated Stock Trading using Reinforcement Learning", IEEE Access 2023
(DOI 10.1109/ACCESS.2023.3324458; authors' code github.com/VarML/TACR), in
`src/models/tacr/`: a Decision-Transformer-style causal transformer over
interleaved (return-to-go, state, action) triples with an offline
actor-critic update (critic TD + BC-regularized actor, paper eq. 4). Trained
offline on the window-expanded behavior-policy trajectories
(`offline_dataset.parquet`, 32 policies incl. nr7) with the same time splits
and 5-seed protocol as Model B.

```bash
python -m src.models.tacr.train --seed 20260814   # one seed; checkpoint -> checkpoints/tacr/s{seed}/
python -m src.models.tacr.train                    # default seed (configs/tacr.yaml)
python -m src.models.tacr.eval                     # 5-seed rolls + basin screening + vs Model B/EM
python scripts/hybrid_sign_macro.py                # Model C+ (TACR |a| x logistic sign on 8 SPY + 8 macro)
```

The paper-faithful default is now `balanced_families: true` (family-balanced
batch sampler so the 20-window mean_reversion family cannot hijack the BC
anchor), `exclude_policies: []` (random re-included as a capped family) and
`double_q_mode: mean`; pass `--no-balanced-families` / `--no-exclude` /
`--double-q-mode min` to reproduce earlier runs.

- **Eval protocol matches Model B exactly** (same splits, same regime
  breakdown; per-regime Sharpe primary, blended secondary; crisis regime
  caveat: 15 test days — directionally suggestive only, NOT a finding).
- **No future information at roll time**: eval feeds a constant RTG (0.0 by
  default) with the model's own autoregressive actions.
- **Basin screening carried from Model B** (PROJECT_NOTES 7.5.5): flag runs
  whose best-val epoch is anomalously late (>35% of the schedule) or whose
  test actions correlate < 0.7 with the pack. See
  `checkpoints/tacr/basin_screening.csv`.
- Validated by `tests/test_tacr.py` + `tests/test_tacr_fixes.py`
  (causal-masking perturbation, BC gradient, Model B context alignment, RTG
  correctness, double-Q / BCQ / PAR helpers, family-balanced sampler) —
  suite 78/78.
- Deviations from the paper (n_layer 4 vs 5; Linear+tanh vs Linear+Softmax
  action head; true return-to-go vs the paper code's immediate-reward
  channel; train-split-only normalization; 3k-step CPU-scaled budget;
  separate critic MLP, not a shared trunk) are documented in the package
  docstrings, `configs/tacr.yaml`, and PROJECT_NOTES 7.6.

Artifacts under `src/models/tacr/checkpoints/tacr/`: `s{seed}/tacr_best.pt` +
`training_log.csv` per seed; `regime_eval.csv`, `vs_model_b.csv`,
`basin_screening.csv` from the eval CLI; the structural-fix / sampler / nr7
variants under their `ctl_*` / `fix_*` / `*_nr7` tags. Notebook:
`notebooks/03_tacr_training.ipynb`.

### Standalone TACR — structural failure, confirmed at 3k and 20k (historical)

Test all-days Sharpe 0.61 +- 0.20 vs EM 0.80 +- 0.11 and Model B 0.99 +- 0.24;
TACR beats EM 1/5 seeds. A 20k-step run made it worse (0.80 -> 0.02); no
checkpoint on any trajectory generalizes (final-epoch check). The mechanism
is Q-inflation collapse: the actor loss goes negative as the critic's Q rises
and overwhelms the BC term. Structural fixes were tested per a pre-registered
protocol (PROJECT_NOTES 7.10): clipped double-Q kills the collapse
signature and produced the old-data margin +0.105 (4/5), but on the new
multi-window data no sampler/Q-aggregation variant (uniform, balanced,
random in/out, min/mean Q) cleared the ≥3/5 margin bar (7.12-7.14). The
RTG-relaxation lever is small (positive target ≈ training-median RTG adds
~+0.05 Sharpe; the 90th percentile hurts). Trend-day filtering (7.20),
focal-loss sign retraining (7.21) and SKEW sentiment (7.22) are all closed
as nulls. NOTE on 7.20: the trend/oscillation test is an ADAPTATION of the
Azizi (JRFM 2026) framing, not a reproduction — it labels trend days on the
CLOSE-TO-CLOSE return |r_{t+1}| (the session's intraday high-low range is
available in the daily frame's high/low columns, so the paper's threshold
variable IS constructible — the close-to-close label was an implementation
choice, not a data limitation) and uses an L2 logistic on the sign model's
own 16 features (the paper's RF/NN + ATR/macro-announcement classifiers are
not implemented). The null is therefore scoped: a return-based,
same-features logistic proxy for that paper's trend/oscillation framing did
not transfer to the C+ sign model — the paper's actual method (intraday-range
label, RF/NN + ATR/announcement features) is untested here.

### Model C+ — TACR magnitude x linear sign + macro (THE result)

TACR's standalone failure is its DIRECTIONAL SIGN (margins ~0); its LEVERAGE
TIMING (|a|) matches/exceeds the exposure-matched control. The hybrid
decouples them:

    a_t = sign_logistic(s_t) * |a_TACR(s_t)|

where the logistic is L2-regularized on the 8 SPY + 8 macro causal z-features
(C selected on val, test untouched) and |a| comes from the seed-stable
double-Q-min/uniform/nr7 pack (Sharpe 0.82 ± 0.15, tightest TACR variance).

**Result — pack mean test Sharpe 1.15 (5 seeds), EM margins 5/5 (+0.30):
the first configuration to clear the ≥3/5 margin bar and to beat Model B.**
Mechanism (decomposed): the sign model shorts only ~6% of days, and on those
days TACR's own sign was long on ALL of them — the linear head fixes TACR's
worst sign errors (net-losing long days flipped to winning shorts). The
margin is tail-concentrated (return-weighted down days), C-robust, and
honest out-of-sample (train-only fit, C on val, test untouched).

The 8 macro features are the load-bearing addition: on identical train dates
the margin is +0.30 with macro vs +0.18 without. A shifted-split robustness
check (test 2020-23 and 2022-24) shows the macro edge is regime-contingent —
largest in the crisis-heavy window (+0.30 delta) where the price-only sign
degrades — never materially harmful. Sentiment (CBOE SKEW) as a 17th feature
was tested and is a null (degrades the fit; §7.22).

Run: `python scripts/hybrid_sign.py` (8-feature) / `scripts/hybrid_sign_macro.py`
(16-feature, canonical) / `scripts/hybrid_robustness.py` (shifted splits) /
`scripts/hybrid_trend_filter.py` (filter, null) / `scripts/focal_sign.py`
(focal loss, null) / `scripts/hybrid_sentiment.py` (SKEW, null). Details:
PROJECT_NOTES 7.17-7.22.

## Model D: fuzzy + transformer + IQL (Phase 4)

Tests the central hypothesis: making regime uncertainty EXPLICIT (a fixed
interval type-2 fuzzy layer over realized_vol_20d / ret_20d / rsi_14 ->
18 memberships appended to the raw 8-d state = 26-d input) before the
policy sees the state, with a causal transformer state encoder (Model C's
verified blocks; no return-to-go channel) and Implicit Q-Learning
(Kostrikov et al. 2022: expectile V tau 0.7, AWR beta 3.0, lr 3e-4 —
paper values, deviations flagged in `configs/model_d.yaml`). D-minus-fuzzy
(8-d input, identical net) is the ablation; the fuzzy layer is a config
toggle in the same package `src/models/d/`.

```powershell
python scripts/d_model_run.py              # 5 seeds x both variants @ 3k + eval + auto-escalation
python -m src.models.d.train --variant d --seed 1           # single run
python -m src.models.d.train --variant d_minus_fuzzy --seed 1
python -m src.models.d.eval               # regime table, basin screening, screen verdicts
```

Validated by `tests/test_model_d.py` (causal masking, fuzzy determinism,
MF-init reproducibility, expectile/AWR loss formulas, 26-vs-8 data
boundary + compute-match). Artifacts under `src/models/d/checkpoints/`
(`d/`, `d_minus_fuzzy/`, `d_20k/` escalation). Notebook:
`notebooks/04_model_d_training.ipynb`.

**Result — D clears the pre-registered screen; the fuzzy-uncertainty
hypothesis is NULL.** All-days Sharpe (5 seeds): D 0.93 +- 0.02 @ 3k /
0.87 +- 0.03 @ 20k; D-minus-fuzzy 0.88 +- 0.03 / 0.88 +- 0.04; both beat
EM (0.80 +- 0.11) in >= 3/5 seeds at both budgets and pass the
final-epoch check everywhere (no TACR-style val-overfit collapse). But
the ablation delta (0.05 @ 3k, 0.01 @ 20k) is an order of magnitude below
the pre-registered noise floor (0.24 = B's seed spread): the fixed IT2
encoding adds no information the input projection cannot learn from the
raw channels at this data scale. Input-sensitivity was verified directly
(perturbation test, `scripts/d_diagnostics.py`): shuffling the whole state
destroys the action series (corr ~ 0) while shuffling only the fuzzy
channels barely moves it (corr ~ 0.93, landing at the D-minus-fuzzy
baseline) — so the null is a real "no marginal fuzzy information," not an
input-insensitive policy. The 3k->20k win-rate churn is a long-only tie
artifact (short_frac = 0 makes D identical to EM), not degradation. D sits inside B's seed spread
(0.99 +- 0.24) with ~10x lower seed variance — a stability gain, not a
performance gain.

**EM-margin audit (project-wide, `scripts/em_margin_audit.py`)** — the
"beats EM X/5" count is a knife-edge criterion (a long-only policy is
exactly its EM control), so every phase verdict was re-derived on per-seed
margins (Sharpe(a*m) - Sharpe(|a|*m)): B naive_new **+0.19 mean** (robust
skill, short side on 13-31% of days), C **-0.18** (anti-skill — its
shorts lose vs passive same-exposure holding; 3/4 negative on the intact
3k seeds — the 5th was overwritten by the 7.6.2 20k rerun, whose margin
is also negative), D **+0.02-0.03** (thin, never negative: stability, not
skill comparable to B's), and **Model C+ +0.30** (the hybrid's 5/5 — the
largest margin in the project, §7.18). Canonical-B is strengthened (vt
posted the same 4/5 count on ~70x thinner margins); C's failure is
sharpened (not a tie artifact); D's screen pass is re-classified as
thin-margin; C+ is the new headline. Win counts are never reported without
margins in this project. Details: PROJECT_NOTES 7.9.2. Scope: fixed-MF IT2 over these 3 features on this
data; learned MFs / u=60 logged as future work. Details: PROJECT_NOTES
7.9.

## Open decisions (flagged, not decided here)

- Offline RL library: Models C/D were implemented from scratch (C per the
  TACR paper, D per the IQL paper) — the `d3rlpy` optional extra was never
  needed and remains unverified/unused.
- Your spec said Python 3.11; this machine has only 3.12.5. Everything is
  verified on 3.12.5.
- The downloader writes raw files into the repo root, not `data/raw/`; the
  loader's `raw_dir` is `"."` accordingly. If you move files later, change
  `configs/data.yaml`.
