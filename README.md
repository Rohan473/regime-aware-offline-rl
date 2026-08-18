# Regime-Aware Uncertainty Modeling for Offline RL in Financial Markets

Research codebase comparing four models (B: DDR-style recurrent, C: TACR-style
Transformer actor-critic, D: target model with fuzzy uncertainty layer + offline
RL (CQL/IQL) + BC regularization, D-ablation: D minus the fuzzy layer) on
identical data and identical regime-conditional evaluation.

**Current session scope (done):** repository scaffolding, data pipeline, regime
labeling, multi-behavior-policy offline dataset generation.
**Out of scope (do NOT build):** Model A baseline, Models B/C/D, `src/models/`
and `src/eval/` are empty by design.

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
  bars into the repo root (currently 2005..2015, **download still in progress**).
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
+1 = fully long, config-driven). `momentum` longs on trailing returns and
shorts on negative ones; `mean_reversion` is the symmetric contrarian
(inverse of momentum). Reward = `a_t * ret_{t+1} - cost * |a_t|` (cost in
bps, default 0).

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
`offline_dataset.parquet` has 21,132 rows to 2026-03-30. The study horizon
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
  cost, behavior policy windows/scales/seed.
- `configs/regimes.yaml` — all regime thresholds + hysteresis parameters.
  Nothing regime-related is hardcoded.
- `configs/ddr.yaml` — Model B (DDR) shared config (vol-targeting + artifacts).
- `configs/tacr.yaml` — Model C (TACR) shared config; paper values with the
  flagged deviations from PROJECT_NOTES 7.6.

## Model C: TACR baseline (Phase 3)

Reproduction of Lee & Moon, "Transformer Actor-Critic with Regularization:
Automated Stock Trading using Reinforcement Learning", IEEE Access 2023
(DOI 10.1109/ACCESS.2023.3324458; authors' code github.com/VarML/TACR), in
`src/models/tacr/`: a Decision-Transformer-style causal transformer over
interleaved (return-to-go, state, action) triples with an offline
actor-critic update (critic TD + BC-regularized actor, paper eq. 4). Trained
offline on the four behavior-policy trajectories (`offline_dataset.parquet`)
with the same time splits and 5-seed protocol as Model B.

```bash
python -m src.models.tacr.train --seed 20260814   # one seed; checkpoint -> checkpoints/tacr/s{seed}/
python -m src.models.tacr.train                    # default seed (configs/tacr.yaml)
python -m src.models.tacr.eval                     # 5-seed rolls + basin screening + vs Model B/EM
```

- **Eval protocol matches Model B exactly** (same splits, same regime
  breakdown; per-regime Sharpe primary, blended secondary; crisis regime
  caveat: 15 test days — directionally suggestive only, NOT a finding).
- **No future information at roll time**: eval feeds a constant RTG of 0.0
  (the paper's eval feeds zeros) with the model's own autoregressive actions.
- **Basin screening carried from Model B** (PROJECT_NOTES 7.5.5): flag runs
  whose best-val epoch is anomalously late (>35% of the schedule) or whose
  test actions correlate < 0.7 with the pack; flagged seeds are reseeded,
  never included. See `checkpoints/tacr/basin_screening.csv`.
- Validated by `tests/test_tacr.py` (causal-masking perturbation test, BC
  gradient test, Model B context-alignment test, RTG correctness, eval
  determinism) — full suite 54/54.
- Deviations from the paper (n_layer 4 vs 5; Linear+tanh vs Linear+Softmax
  action head; true return-to-go vs the paper code's immediate-reward
  channel; train-split-only normalization; 3k-step CPU-scaled budget;
  separate critic MLP, not a shared trunk) are documented in the package
  docstrings, `configs/tacr.yaml`, and PROJECT_NOTES 7.6.

Artifacts under `src/models/tacr/checkpoints/tacr/`: `s{seed}/tacr_best.pt` +
`training_log.csv` per seed; `regime_eval.csv` (regime breakdown vs Model B
and EM), `vs_model_b.csv`, `basin_screening.csv` from the eval CLI.
Notebook: `notebooks/03_tacr_training.ipynb`.

**Result — TACR-as-reproduced-here does NOT clear the bar on this data
(final; budget confound ruled out; scoped — SPY, B's feature set,
u=20, tanh-head actions, alpha 0.9).**
Test all-days Sharpe 0.61 +- 0.20 (seeds 0.26/0.80/0.80/0.60/0.61) vs EM
0.80 +- 0.11 and Model B 0.99 +- 0.24; TACR beats EM 1/5 seeds (criterion
>= 3/5 not met), takes more exposure (mean|a| 0.39 vs 0.26) for less
return. The 13x CPU budget cut vs the paper's 40k steps was re-checked
with a 20k-step run (paper-proportional warmup): seed 1 dropped from 0.80
to 0.02 test Sharpe — longer training hurts, it does not help, so the
verdict is structural, not an undertraining artifact. Mechanism: TACR's
val Sharpe does not transfer to test, and a final-epoch check (pre-
registered bar) confirms it is not a selection-rule problem — no
checkpoint on the 20k trajectory generalizes (best-val and final-epoch
test Sharpe are both ~ -0.05 to -0.10); a 100x faster critic makes
selection worse, not better.
Basin screening flags all 5 seeds mechanically (no cohesive pack:
cross-seed corr 0.17-0.78 vs B's 0.93-0.98; the low corr is partly an
undertraining artifact — seed 1 rises to 0.73-0.87 at 20k — but the pack
converges onto the same underperforming region); the verdict is
basin-independent. Details: PROJECT_NOTES 7.6.1-7.6.2.

## Open decisions (flagged, not decided here)

- Offline RL library for Models C/D: `d3rlpy` vs `CORL` (your call said "scope
  might expand"). Dataset is exported as plain parquet — library-agnostic.
  `pyproject.toml` lists `d3rlpy>=2.3,<3` as the optional `rl` extra,
  unverified against this Python 3.12 env; verify when Models C/D start:
  `pip install -e .[rl]`.
- Your spec said Python 3.11; this machine has only 3.12.5. Everything is
  verified on 3.12.5.
- The downloader writes raw files into the repo root, not `data/raw/`; the
  loader's `raw_dir` is `"."` accordingly. If you move files later, change
  `configs/data.yaml`.
