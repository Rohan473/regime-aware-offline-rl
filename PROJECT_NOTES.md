PROJECT NOTES — Regime-Aware Uncertainty Modeling for Offline RL in Financial Markets
=====================================================================================
Working directory: E:\New folder\RL_trading

This file is a complete record of everything done in this project, in order,
with the final verdict stated plainly at the end. Read it top to bottom if
you are new; the last two sections are the ones that matter.

**CURRENT STATE (2026-09-05) — the project's best result is Model C+ (section
7.17-7.22):** a hybrid that decouples TACR's two skills. The Transformer
(fix_a_nr7, double-Q-min/uniform/nr7) supplies LEVERAGE TIMING |a|; a simple
L2 logistic on 8 SPY + 8 macro/cross-asset causal features supplies
DIRECTIONAL SIGN. Pack mean test Sharpe 1.15, EM margins 5/5 (+0.30) — the
first configuration to clear the >= 3/5 margin bar and to beat Model B.
The 8 macro features (rates, vol term structure, dollar, credit proxy,
cross-market momentum — fetched into data/macro/) are the load-bearing
addition (margin +0.30 with vs +0.18 without). Regime-filtering, focal-loss
sign retraining and SKEW sentiment were tested and are nulls (7.20-7.22).

**REPRESENTATION LABORATORY (2026-09-15, section 8.x):** a new thread that
makes the learned hidden state the object of study. Linear probes on the
frozen models show NO 1-day directional info anywhere (dir1 0.538 = baseline)
while regime and (in raw) vol are readable; all learned reps linearly collapse
(effective spectral rank 3.2-7.7 << 8). Idea 16 (one encoder, 4 objectives)
shows ~79-95% of the linear subspace is shared across objectives (reward
shapes the tail: DSR keeps the most dims + the only behaviorally strong arm).
The advisor's rep x policy design: three frozen representation learners
(auto/predictive/contrastive) + downstream A2C -> the predictive rep raises
the SUPERVISED direction policy Sharpe 0.394 -> 1.036 but RL-A2C is negative
on every rep (RL cannot compensate for the representation).

**CSI300 / CHINA EXTENSION (2026-09-04, section 7.28):** the full pipeline and
all four models were re-run on the CSI300 index (sh000300, Sina). DDR
vol-targeted is the best CSI300 model (+0.41 Sharpe, 5/5 vs EM); naive DDR
(+0.26, 5/5) and the C+ sign (margin +0.22, 5/5) also beat their EM controls;
China macro does NOT improve the sign head; TACR and D fail on CSI300 exactly
as on SPY. See section 7.28 for the full table.

Phases, in order:

  Phase 1 (complete, verified): a reproducible data pipeline that builds a
  daily OHLCV frame, 8 technical features (z-scored, causal), regime labels
  {bull, bear, crisis}, behavior-policy families (buy_and_hold, momentum at
  9 windows, mean_reversion at 20 windows, random, nr7 — 32 policies total)
  with actions in [-1, 1] (shorting enabled), and an offline RL dataset of
  transitions (state, action, reward, next_state).

  Phase 2 (complete): Model B, a DDR (Differential Sharpe Ratio) baseline —
  a GRU policy trained by direct backprop through the DSR reward (Moody &
  Saffell 1998) on the offline state sequence, evaluated per regime on a
  strict time split.

  Phase 3 (complete): Model C, the TACR Transformer actor-critic (Lee &
  Moon 2023). As a standalone actor it is the project's structural-failure
  case (7.6, 7.10-7.16); its leverage timing is salvageable and is the
  magnitude source of Model C+.

  Phase 4 (complete): Model D, the fuzzy-uncertainty ablation — trains
  stably while the fitted policy stays long-dominant (short_frac <= ~10%):
  in the 4-policy mix, at 6.6x volume with matched composition (7.9,
  7.27.2), and any mean-reversion sampling share that keeps the fitted
  short-side mass low (7.27.3); it unravels only as contrarian sampling
  pushes the fitted short_frac high — the boundary is the short-side mass,
  not distance from the old mix, and cannot be rescued by any IQL optimizer
  hyperparameter (AWR β 7.27.4, polyak τ / expectile τ 7.27.5: all r ~ 0).
  The fuzzy layer is null (7.9-7.9.2).

The single most important outcome of the project is NOT the model. It is the
discovery and root-causing of TWO independent failure modes:

  (1) DATA CONTAMINATION: the daily frame has multi-month missing chunks;
      pct_change() collapsed each into one fake "daily" return. This
      fabricated the entire test-window "crisis" regime (a 2.80 Sharpe
      crisis bucket that did not exist) and a fake catastrophic loss day.
  (2) REWARD HACKING: the DSR reward's variance-reduction bias made the
      policy de-lever to ~1/3 exposure — its entire apparent Sharpe edge
      over buy-and-hold decomposes into position sizing, not signal.

  (Both were resolved in later phases: the holes were backfilled (7.5.4),
  and the C+ hybrid's directional edge is margin-verified, not sizing.)

--------------------------------------------------------------------------------
2. ENVIRONMENT
--------------------------------------------------------------------------------

- Python 3.12.5 only (spec asked for 3.11; not available on this machine).
- Pins: numpy==1.26.3 (required by pandas-ta), pandas==3.0.5,
  torch==2.12.0, pandas-ta-openbb==0.4.24, pyarrow, omegaconf, scipy.
- torch >= 2.6 defaults to weights_only=True on torch.load, so checkpoint
  payloads must be pickle-safe (Path -> str config serialization).
- No git repo in the working directory.

--------------------------------------------------------------------------------
3. PHASE 1 — DATA PIPELINE (complete and verified)
--------------------------------------------------------------------------------

3.1 pandas-ta reproducibility fix (first task, completed)
  - pandas-ta-openbb was not importable on Python 3.12 (C extension issue).
  - src/data/_pandas_ta_compat.py: runtime compatibility fix.
  - scripts/apply_patches.py: idempotent, marker-based patch of the
    installed site-packages, verifies by importing pandas_ta in a
    subprocess. Site-packages was reverted then re-patched through the
    script; direct import verified.
  - This was the reproducibility requirement: pinned versions + a scripted,
    verifiable patch.

3.2 Data pipeline (src/data/)
  - Daily OHLCV frame: 2005-01-03 .. 2024-02-29 (4,275 rows), built from an
    incremental downloader that writes raw files into the repo root
    (loader raw_dir = "."; configs/data.yaml).
  - Features (8 used, capped at 10 by design — do not add without asking):
    ret_1d, ret_5d, ret_20d, realized_vol_20d, rsi_14, macd_hist,
    volume_zscore_20d, bollinger_pos. All z-scored with CAUSAL expanding
    statistics (no lookahead).
  - Regimes (configs/regimes.yaml, nothing hardcoded):
      * bull/bear from trailing 60-day return sign, with hysteresis.
      * crisis from 20-day realized vol above the EXPANDING 95th percentile
        (min_periods=300, clipped at 0.02 floor), entry confirmation (2
        consecutive days), priority over bull/bear.
      * Full-sample composition: bull 70.6%, bear 17.8%, crisis 11.6%.
      * Expanding threshold (not rolling) was the user's Phase-1
        requirement; regimes.yaml has percentile_min_periods: 300 and no
        percentile_window_days.
  - Behavior policies (src/data/behavior_policies.py), actions in [-1,1]:
      * buy_and_hold: always +1.
      * momentum: clip(ret_20d / scale, -1, 1) — longs uptrends, SHORTS
        downtrends.
      * mean_reversion: clip(-ret_20d / scale, -1, 1).
      * random: uniform in [-1, 1], seeded.
    Shorting enabled was the user's Phase-1 requirement; scale/seed in
    configs/data.yaml.
  - Offline dataset: 16,856 transitions = 4 policies x 4,214 unique dates.
    All four policies share identical states per date, so
    buy_and_hold + momentum filtering selects the full unique market path.

3.3 Phase-1 verifications (all PASS)
  - COVID crisis labeling: 2020-03-10 .. 2020-05-28 labeled crisis, max
    drawdown -33.4% inside the window, vol 0.68-0.84 vs threshold 0.323.
  - Momentum/mean-reversion sign sanity: 20/20 sampled dates agree with
    ret_20d sign; full-sample agreement 1.0000 (skew is composition: 65.6%
    of days have ret_20d > 0, magnitudes symmetric).
  - Relabeling stability: labels identical on all 2,771 non-NaN dates when
    the series is truncated at 2016 vs full.
  - 27/27 tests (pytest) covering regime non-degeneracy, no single-day
    flicker, crisis priority, entry confirmation, exit-band persistence,
    NaN leakage, action diversity, shorting occurs, reward formula, done
    flags, next-state consistency, minute->daily resampling.

--------------------------------------------------------------------------------
4. PHASE 2 — MODEL B: DDR BASELINE (complete, NULL result)
--------------------------------------------------------------------------------

4.1 Scope constraint (user-specified)
  - Add ONLY: src/models/ddr/, tests/test_ddr.py,
    notebooks/02_ddr_training.ipynb, src/eval/regime_eval.py.
  - DO NOT touch src/data/ (Phase-1 artifacts frozen — this later became
    the reason hole contamination could only be mitigated, not fixed).

4.2 Design decisions (documented in code)
  - Policy: single-layer GRU (hidden 32, gru|lstm switchable), window 20
    (shared WINDOW_DAYS in src/models/__init__.py so B/C/D window alike),
    8 z-features in, tanh action in [-1, 1] out.
  - Training signal: DSR D_t (exact EMA recurrence, eta=0.01, warmup 20
    counted once per sequence), reward per step = D_t, optimized by direct
    backprop through the DSR (chosen over REINFORCE: more sample-efficient,
    original paper's "direct reinforcement").
  - Truncated BPTT: EMA state carried across the sequence but detached at
    block boundaries (blocks of 20 days), so gradients flow within blocks.
  - Offline adaptation, option (a): trained on the OFFLINE state sequence
    (feature windows) but generates its OWN actions; reward computed from
    those own actions against REALIZED next-day market returns. Logged
    behavior-policy actions/rewards never used.
  - Time split (never shuffled): train <= 2018-12-31, val 2019-2020, test
    2021-01-04 .. 2024-02-28 (521 days).
  - Defaults: eta=0.01 and lr=1e-3 are Moody & Saffell / Adam DEFAULTS
    (not tuned; disclosed; grid sensitivity run — see 5.3).
  - epochs default raised 10 -> 30 with best-val checkpoint selection.
  - Cost 0 (matches Phase-1 reward with cost_bps=0).

4.3 Modules
  - src/models/__init__.py        shared WINDOW_DAYS + SPLIT_*_END
  - src/models/ddr/config.py      DDRConfig (validated dataclass)
  - src/models/ddr/dsr.py         DSRState (exact EMA, warmup)
  - src/models/ddr/policy.py      DDRPolicy (GRU/LSTM, tanh)
  - src/models/ddr/data.py        DDRData, load/split, compute_valid_mask
  - src/models/ddr/train.py       train_ddr, roll_sharpe, best-val saving
  - src/models/ddr/eval.py        roll_test_predictions, evaluate,
                                  baseline_policy_table, clean-label warning
  - src/eval/regime_eval.py       shared regime-breakdown evaluator
  - tests/test_ddr.py             13+ tests: hand-computed DSR toy (D1=1.0
                                  exact anchor), warmup NaN, gradient-to-
                                  action, time-split no-leakage, regime-eval
                                  synthetic literals, action range,
                                  causality, z-state/next-return
                                  equivalence vs Phase 1, train smoke,
                                  validity-mask unit tests.
  - Total suite: 42/42 passing.

4.4 Training dynamics (overfitting, mandatory best-val selection)
  - Best val Sharpe 1.176 at epoch 8; val decays monotonically to 0.46 by
    epoch 30 while the train-path DSR EMA Sharpe explodes to 8.5. Best-val
    selection is not optional. Best-val epoch is consistently 5-8 across
    seeds.

--------------------------------------------------------------------------------
5. THE DATA-INTEGRITY FINDING (the most important result of the project)
--------------------------------------------------------------------------------

5.1 The holes
  The Phase-1 daily frame has multi-month missing chunks:
    2011-04..2012-12 (~278d), 2018-04..07 (125d), 2021-04..05 (62d),
    2021-09..12 (125d), 2022-04..07 (123d), 2022-09..10 (62d),
    2023-10 (33d).  (gaps_report.csv has the full list.)
  pct_change() collapses each hole into ONE fake "daily" return. Example:
  the "2022-03-31 market -9.4%" day is really the accumulated
  Mar 31 -> Aug 1 2022 return.

5.2 Consequences, root-caused
  - The ENTIRE test-window crisis bucket (20 days, Aug 2-29 2022, "Sharpe
    2.80") is a hole artifact: 20-day realized vol was inflated by the fake
    return, crossing the crisis threshold. Recomputing labels with the fake
    returns replaced by typical returns yields ZERO crisis days in Aug-Sep
    2022 — and zero crisis days in the test window at all. The "2.80
    crisis Sharpe" never existed.
  - The early "bear Sharpe -0.21" was two fake hole days dragging a real
    path down (and the later "bear +1.68" was computed on 89%
    hole-contaminated labels — neither is evidence).
  - 7 decision days have returns spanning a hole: 2011-03-31, 2018-03-29,
    2021-03-31, 2021-08-31, 2022-03-31, 2022-08-31, 2023-09-29 (5 in test).

5.3 Mitigation (in-scope, DDR layer only)
  - compute_valid_mask: a return from date t is valid only if the next row
    in the features frame is <= 6 calendar days later (genuine holiday
    closures are 1-5 days; holes are 33+). Invalid days excluded from
    training loss, val/test Sharpe, regime tables, baselines; rows kept in
    CSVs with a valid flag.
  - Phase-1 labels/artifacts were NOT rewritten (out of scope, frozen by
    constraint). The downloader must backfill the holes; until then, the
    regime structure of the test window is not trustworthy.

--------------------------------------------------------------------------------
6. THE AUDIT CHAIN (every headline number decomposed before belief)
--------------------------------------------------------------------------------

Each round was a reviewer-driven demand to decompose the previous
headline. The chain is the deliverable.

6.1 Seed sweep (5 seeds x 30 epochs, hole days masked)
  bull 0.82 +/- 0.12 | bear 1.68 +/- 0.15 | crisis 1.20 +/- 2.07 | all 1.07 +/- 0.11
  -> crisis std 2.07 >> mean: noise. bull stable. bear positive.
  Grid (eta x lr, 2x2): blended Sharpe 0.99-1.13 — defaults not a
  knife-edge. Baselines (same days): B&H 1.15, momentum -0.01,
  mean_reversion 0.34, random -1.11.

6.2 Paired returns test (DDR minus B&H, same days, per seed)
  bull -0.047%/day (t=-1.32, p=0.19), bear -0.052%/day (t=-0.73, p=0.47).
  0 of 5 seeds beat B&H on returns in EITHER regime (all 10 diffs
  negative). The "bear 1.68 vs 1.61" was Sharpe arithmetic: B&H earned
  MORE on bear days, just with higher vol.

6.3 Residual label contamination audit
  The mask removes boundary DAYS; the LABELS themselves are computed on
  hole-spanning windows: 89% of bear days, 35% of bull days, 100% of
  crisis days in the test split. Clean-label day-set: 258 days
  (245 bull, 13 bear, 0 crisis).

6.4 Clean-label evaluation (the only regime numbers reported)
  bull (245d): DDR 1.70 +/- 0.16 vs B&H 1.29 (Sharpe), cum +7.4% vs ~+13%.
  bear (13d): NOT reported (underpowered). crisis: does not exist.
  all (258d): DDR 1.46 +/- 0.11 vs B&H 1.14; cum +7.1% vs +16.4%.

6.5 Paired drawdown test
  DDR max DD less severe than B&H in 5/5 seeds (full path -8.3% vs -12.6%;
  clean-label -2.6% vs -8.0%). BUT (6.7) this turned out to be 100%
  mechanical.

6.6 Momentum coherence
  Momentum is sign-consistent 100% by construction (action =
  clip(ret_20d/scale)); its "crisis +1.47" was a full-short position on
  the fake Aug-2022 labels with 90% of actions decided on hole-spanning
  windows. Mechanically coherent, same contamination as everyone else.

6.7 Leverage-matched decomposition (the decisive test)
  Four controls on the same clean-label days: B&H (long 1.0); CL (constant
  leverage mean(|a|) ~ 0.40 — Sharpe is scale-invariant, so CL Sharpe
  equals B&H by identity, verified); EM (exposure-matched: DDR's day-by-day
  position size |a_t|, always long); DDR.
    B&H / CL : bull 1.29, all 1.14
    EM       : bull 2.02 +/- 0.28, all 1.74 +/- 0.23   <- beats DDR 5/5 seeds
    DDR      : bull 1.70 +/- 0.16, all 1.46 +/- 0.11
  EM beats DDR in 5/5 seeds. Paired DDR minus EM: -0.006%/day (t=-1.31,
  p=0.19) — the model's sign decisions (shorts) REDUCE Sharpe. EM max DD
  equals DDR's to 4 decimals — the drawdown reduction is 100% mechanical
  de-leveraging. A leverage-matched B&H closes the gap and then inverts it.

  Verdict: DDR is a lower-variance, lower-return policy whose only "edge"
  over buy-and-hold is the average position size it happens to take.
  There is no evidence of directional signal. Null result, nailed shut
  three ways (returns, DD, leverage).

6.8 Reward-hacking diagnosis (why the DSR converged there)
  TWO separate findings, do not conflate:
  1. Exposure capping is present at the DEPLOYED epoch: best-val checkpoints
     (epochs 5-8) already run mean|a| 0.26-0.35, short fraction 0.04-0.15.
     The de-leveraging precedes overfitting; the null result comes from an
     early, non-degenerate, already-de-levered policy. The DSR reward's
     variance-reduction bias is hypothesis #1 for any retrain — NOT
     architecture or epochs; a bigger GRU finds the same lazy optimum
     faster. Candidate fixes: capped-|a| or realized-Sharpe reward.
  2. The short-flip pathology (short_frac 7% -> 37%, train DSR EMA Sharpe
     exploding to 8.5, val decaying 1.18 -> 0.46 over epochs 20-30) is a
     SEPARATE post-val overfit artifact that best-val selection correctly
     discards. It explains training dynamics, not the null result.

6.9 Corrections made during the audits (self-discipline record)
  - "mean |action| ~ 0.8" was a misread of the bear-day table; actual is
    0.40. Caught and fixed.
  - A sign inversion in the paired-DD summary line (which would have
    flipped the verdict) was caught before reporting.
  - The 378/118-day regime table was retired and is marked deprecated
    everywhere; no contaminated bear number is cited in any context.

--------------------------------------------------------------------------------
7. FINAL STATUS AND VERDICT
--------------------------------------------------------------------------------

  Pipeline:  correct, tested (42/42), reproducible (patched pandas-ta,
             pinned env, seeded everything).
  Model B:   negative-to-null. Does not beat buy-and-hold on returns
             anywhere (0/10 seed-regime pairs), nor on a leverage-matched
             baseline (EM beats it 5/5), nor on return-per-risk
             consistently. Its apparent Sharpe edge and drawdown reduction
             are both mechanical de-leveraging.
  Regime evidence on the test window: invalid until the data holes are
             backfilled (89% of bear labels, 35% of bull, 100% of crisis
             are hole-computed). Clean-label numbers (258 days) are the
             only ones reported.
  Open work: backfill the holes in the downloader; then re-estimate the
             regime structure and re-run Model B with a reward that does
             not reward de-leveraging.

  Interview framing (reviewer-endorsed): "Found and root-caused TWO
  independent failure modes in one project — a fabricated '2.80 crisis
  Sharpe' and a fake catastrophic loss day from a data-pipeline bug, and a
  model gaming its own DSR reward function by de-leveraging. Retracted my
  own headline results twice (bear win, then Sharpe edge) once the audits
  disproved them." That is the deliverable, not the model.

  IMPORTANT (2026-08-17): the status block above describes the OLD
  contaminated data. The pipeline has since been re-run — see section 7.5.
  On the CURRENT data the "no directional signal" verdict does not hold.

--------------------------------------------------------------------------------
7.5 DATA REGENERATION + VOLATILITY-TARGETED DSR REWARD FIX (2026-08-17)
--------------------------------------------------------------------------------

  7.5.1 The data changed under us (externally re-run pipeline)
    - features_regimes.parquet: 5,344 rows, 2005-01-03 -> 2026-03-31,
      ZERO gaps > 6 days (the 6 multi-month holes are GONE — backfilled).
    - offline_dataset.parquet: 21,132 rows (was 16,856) to 2026-03-30.
    - Consequence: the entire audit chain's day-set (516 valid test days,
      258 clean-label days, "0 crisis days exist" finding) is historical.
      The current test set is 1,005 valid days: 769 bull / 221 bear /
      15 crisis (a REAL crisis label now — 527 crisis days all-time).
    - Fix in src/models/ddr/data.py: load_ddr_data clips the path to
      SPLIT_TEST_END (2024-12-31, 310 dates dropped, stderr warning) so
      the study horizon stays fixed and splits don't raise.

  7.5.2 Vol-targeted DSR (user-specified reward fix)
    - Spec: scale a_t by target_vol / realized_vol_t (causal trailing 20d
      std of the a*ret series, annualized, DETACHED), clip to +-2.0, DSR
      sees scaled returns. Invariance: uniform |a| shrinkage leaves the
      scaled series unchanged -> the policy can no longer game the reward
      by de-leveraging. target_vol/clip in configs/ddr.yaml (shared),
      NOT hardcoded. Old naive checkpoints/artifacts preserved; new
      artifacts under checkpoints/vt/.
    - Implementation: VolTargetBuffer in src/models/ddr/dsr.py (tail
      buffer persists across blocks, resets per epoch; scale=1 during
      20-step warmup; vol < 1e-6 ann treated as rounding noise).
      train.py feeds it the DSR; val/test rolls stay on RAW actions.
      train/eval CLI load ddr.yaml by default (--no-vol-targeting to
      opt out). 7 new unit tests (exact scale, de-leveraging invariance,
      warmup, clipping, causality, yaml load/reject) — 49/49 pass.

  7.5.3 Results (valid test days, 5 seeds x 30 epochs, scripts/ddr_vt_retrain.py)
    - naive_old (trained on contaminated data, rolled on new): all 1.05,
      bull 1.64; EM 0.92/1.47.
    - naive_new (naive DSR RETRAINED on current data): all 0.99 +- 0.24,
      bull 1.51 +- 0.54; beats EM 4/5 all, 4/5 bull.
    - voltarget: all 0.81 +- 0.12, bull 1.32 +- 0.18; beats EM 4/5 all,
      1/5 bull; max DD -5.3% (vs B&H -25.3%).
    - Exposure: vt mean|a| 0.14-0.23 with short_frac 0.22-0.49 vs naive
      0.27-0.37 / 0.02-0.11. The fix did NOT restore exposure — the DSR
      gradient carries no level pressure; the level drifts.
    - Verdict: criterion met (4/5) but hollow — naive ALSO beats EM 4/5
      on the same data, and vt is worse than naive (-0.18 Sharpe/seed,
      1/5 positive). Conclusions: (1) the old null was a data artifact;
      the GRU extracts signal with EITHER reward shape on hole-free data.
      (2) Vol-targeting is not the lever: reward-shaping cannot set the
      exposure level by construction (only a level-aware objective
      could). (3) Reward-shape iteration is NOT closed — but the next
      step is NOT another reward tweak: vol-targeting alone is
      INSUFFICIENT; explicit exposure regularization is UNTESTED.
      (4) Finding (citable): risk-normalization changes the incentive
      landscape but adds no countervailing incentive toward using
      leverage, so the policy just drifts to whatever level minimizes
      variance in a different feature space — the DSR gradient carries
      no level pressure.
    - Crisis regime caveat: the current test window has only 15 crisis
      days; the crisis-regime Sharpe from eval is directionally
      suggestive at best and is NOT reported as a finding.
    - Artifacts: checkpoints/vt/ (5 checkpoints, training_logs,
      leverage_test.csv 3-run table), checkpoints/naive_new/ (control).

  7.5.4 Data-fix audit (scripts/audit_data_fix.py) — ALL CHECKS PASS
    - A. defect characterization: two raw-download passes visible in
      SPY_*.parquet mtimes (08-14 morning: 2005-2010, 2012-2017,
      2019-2020; evening: 2011, 2018, 2021-2026 = exactly the old hole
      years + extension). Defect = INCOMPLETE DOWNLOAD -> backfilled
      rows; no evidence of value corruption: all 7 anchor prices match
      history within +-5% (incl. pre-2016 hole years 2011-12-30=125.43,
      2012-12-31=142.55, 2008-11-20=75.95 exact), per-year counts all
      250-253, 0 gaps > 6d. Old parquet files were overwritten, so a
      direct old-vs-new value diff is impossible; anchors + determinism
      are the evidence.
    - B. regime-label re-validation on CURRENT data: (1) stored vs
      recomputed labels agree 1.0000 on 5,284 days; (2) 74 runs, 0
      shorter than min duration; (3) COVID: crisis run 2020-03-10..
      2020-05-28 covers the 03-23 crash trough and persists past
      Apr 15 (entry lag ~2 weeks after the 02-21 onset is the 20d vol
      smoothing + 2-day confirmation, not a regression — 2020 rows were
      complete in pass 1); (4) GFC detected 2008-09-16..2009-06-24
      (covers 09-29..11-20); (5) crisis-overrides-bull holds (109
      positive-60d-return days inside crisis runs, all labeled crisis);
      (6) relabeling stability 0.9873 under bull_thr=0.005/pctile=96;
      (7) pre-2016 (2011-2013): 754 labeled days, 11 runs, min 13d.
    - C. momentum sign sanity on current data: overall hit 0.5215
      (bull 0.5279 / bear 0.5084 / crisis 0.5009 on 527 days).
    - NOTE: an earlier version of this script mis-mapped run positions
      (positions of the dropped series onto the full NaN-warmup index)
      and printed shifted crisis boundaries; the stored labels were
      correct all along (determinism 1.0, value probes consistent).
      Fixed by indexing the dropped series only.
    - Output: checkpoints/robustness/data_fix_audit.csv.

  7.5.5 CANONICAL MODEL B DECISION (2026-08-17) + SEED-VARIANCE NOTE
    - Decision: Model B in the final B-vs-C-vs-D comparison is naive_new
      (naive DSR retrained on CURRENT hole-free data) — the stronger
      baseline (beats EM 4/5 all, 4/5 bull; beats voltarget by -0.18
      Sharpe/seed, 1/5 positive). Voltarget is a documented
      side-experiment (scripts/ddr_vt_retrain.py + README), NOT a
      contender for the B slot; using it would flatter D.
    - Canonical artifact: checkpoints/naive_new/s20260814/ddr_best.pt
      (reproducible via `python -m src.models.ddr.train`; configs/ddr.yaml
      seed). Variance statement for the comparison table: pack mean
      (seeds 20260814/1/2/3) all 1.10 +- 0.04, bull 1.77 +- 0.08.
    - Seed variance is BIMODAL, not Gaussian (scripts/ddr_seed_variance.py
      + scripts/ddr_basin_probe.py, artifacts seed_variance.csv /
      naive_probe/basin_probe.csv): 9/10 seeds across the canonical 5 +
      a 5-seed probe land in ONE shared basin (best-val @ epochs 3-5,
      pairwise test-prediction corr 0.93-0.98, val->test transfer holds:
      val 1.03-1.18 vs test 1.01-1.15). Seed 4 is a different, worse
      basin: best-val @ epoch 30 (val kept rising while the pack's
      decays -> overfit tail), corr vs pack 0.33, test all 0.51 (loses
      to EM), bull 0.44, higher exposure (mean|a| 0.33, short_frac 0.30
      vs 0.23-0.26/0.14-0.22). Probe found 0 more such runs -> frequency
      ~1/10: rare but real.
    - C/D screening rule (from this): good-basin runs always peak at
      epochs 3-5; any run whose best-val epoch > 10 is a suspect basin
      draw -> reseed. Also check cross-seed corr when C/D train.
    - The 5-seed mean 0.99 +- 0.24 (all days) in 7.5.3 is dominated by
      the seed-4 outlier; the pack consensus 1.10 +- 0.04 is the
      representative baseline.
    - MARGIN-AUDIT ADDENDUM (2026-08-21, scripts/em_margin_audit.py;
      full tables in 7.9.2): "beats EM X/5" was later shown to be a
      knife-edge criterion (a long-only policy is EXACTLY its EM
      control), so the canonical decision was re-derived on per-seed
      EM margins (Sharpe(a*m) - Sharpe(|a|*m)):
        naive_new: +0.17 / +0.31 / +0.28 / +0.28 / -0.08
                   (seeds 20260814/1/2/3/4; short_frac 0.14-0.31)
        vt:        -0.18 / +0.03 / +0.01 / +0.03 / +0.12 (mean +0.003)
      naive_new's margins are WIDE — robust directional skill; the
      canonical verdict never rested on the tie structure (seed 4's
      loss is a genuinely negative margin, not a tie). vt posts the
      SAME 4/5 win count on margins an order of magnitude thinner —
      knife-edge. The margin view therefore STRENGTHENS the canonical
      decision: the count called naive_new and vt equivalent; the
      margins separate them by ~70x. naive_new stays canonical, now on
      margin evidence rather than count alone.

  7.6 MODEL C (TACR) — implementation notes (2026-08-18)
    - Reproduces Lee & Moon, "Transformer Actor-Critic with Regularization:
      Automated Stock Trading using Reinforcement Learning", IEEE Access
      2023 (DOI 10.1109/ACCESS.2023.3324458), from the authors' code
      (github.com/VarML/TACR): Decision-Transformer-style causal GPT-2
      transformer over interleaved (rtg_t, s_t, a_t) triples (3u tokens),
      separate per-modality Linear embeddings + learned timestep embedding
      (positional embeddings REMOVED, as in the paper), L pre-LN decoder
      blocks with a strict lower-triangular causal mask, action head
      Linear+tanh (see deviations). Offline actor-critic per paper
      Algorithm 1: critic TD update r + gamma*Q_target(s', pi_target(s'))
      with polyak tau=0.005 and critic_lr 1e-6; actor loss
      -lambda*Q + BC with lambda = alpha/|Q|.abs().mean().detach(),
      alpha=0.9, grad clip 0.25, AdamW 1e-4/wd 1e-4.
    - Deviations from the paper (all flagged in code + configs/tacr.yaml):
      (1) n_layer 4 vs 5 (user spec 3-4); (2) action head Linear+tanh vs
      Linear+Softmax (continuous [-1,1] actions; deterministic roll);
      (3) return channel = true return-to-go (cumulative future logged
      reward) vs the paper's code which embeds the immediate reward;
      (4) state normalization stats on TRAIN split only vs whole file;
      (5) budget 3k steps (epochs 10 x 300, batch 64) vs 40k + warmup
      1k vs 10k — CPU-scaling; 192k samples = 12 passes over our 15.9k
      transitions; (6) critic is a separate MLP [s,a]->512->256->1 with
      target networks (the paper's implementation), NOT a shared-trunk
      value head as the Phase-3 spec assumed; (7) eval-time RTG constant
      0.0 (the paper's evaluate_episodes feeds zeros) -> no future info.
    - Validation (tests/test_tacr.py, 5 tests, all pass; suite 54/54):
      causal masking is bitwise-causal under future-token perturbation;
      BC gradient nonzero/finite with the exact MSE formula; context
      alignment u == WINDOW_DAYS and TACR's day-t state == last row of
      Model B's W-day window at t; RTG == cumulative future logged reward;
      constant-RTG roll is deterministic and immune to the realized
      (future-dependent) return channel.
    - Training behavior observed across seeds: seeds 1/2/3 show the
      Q-overestimation signature (actor loss goes NEGATIVE late in the
      schedule as Q_bar rises and lambda-normalized Q-gradient overwhelms
      the BC term) — best-val selection picks the EARLY epochs (1-3),
      matching the Model B pattern where good checkpoints are early.
      Seed 20260814 keeps BC-dominated (actor loss ~0.93, Q_bar negative)
      with its best val at epoch 8. Reported per-seed best val Sharpe:
      20260814 +0.942 @ 8, 1 +0.539 @ 2, 2 +0.716 @ 3, 3 +0.377 @ 1.
    - Basin screening (protocol from 7.5.5, mechanical): flag if best-val
      epoch > 35% of schedule (10-epoch -> >3) or cross-seed test corr
      < 0.7. NOTE: on the 10-epoch schedule the epoch rule is noisier
      than on B's 30-epoch schedule (val = 2019-2020 single path);
      interpretation leans on the correlation rule, reported in
      checkpoints/tacr/basin_screening.csv.
    - Eval protocol identical to Model B: same time splits, regime
      breakdown PRIMARY, blended secondary, crisis regime caveat (15 test
      days), wins criterion >= 3/5 seeds vs EM; exposure + turnover logged.
    - Artifacts: src/models/tacr/checkpoints/tacr/s{seed}/tacr_best.pt +
      training_log.csv; regime_eval.csv / vs_model_b.csv /
      basin_screening.csv (from `python -m src.models.tacr.eval`).
      Notebook 03_tacr_training.ipynb (single-seed walkthrough).

  7.6.1 RESULTS — FINAL (2026-08-18; provisional withdrawn when the 20k
      budget re-check landed, 7.6.2)
    - VERDICT: TACR < EM on this data — CONFIRMED, no longer provisional.
      The budget confound is ruled out by the 20k diagnostic (seed 1 at
      20k steps: test all-days Sharpe 0.016 vs 0.800 at 3k — 6.7x the
      steps made it WORSE, not better). More compute does not rescue
      TACR; the failure is structural, and its mechanism is val->test
      transfer (below), not undertraining.
    - SCOPE OF THIS CLAIM (read before citing): "TACR-as-reproduced-here,
      on this data" — SPY only; Model B's feature set (the shared 8
      FEATURE_COLUMNS fields, verified against src/data/
      technical_factors.py); context u=20 days; tanh-head discretized actions (OUR
      implementation choice — the authors never specify an action head);
      alpha=0.9, the paper default; two critic-lr regimes (1e-6, 1e-4).
      Varied: seed (5), budget (3k/20k), critic lr. NOT varied: asset,
      feature set, context length, action head, horizon, reward
      definition, alpha. This is evidence that a naive TACR backbone is
      not a sound basis for Model D — it is NOT a general claim that the
      paper's design fails on other assets or configs.
    - 5-seed 3k numbers (still the reference table; seed 1 now also has
      a 20k successor annotated in 7.6.2): test all-days Sharpe per seed
      0.256 / 0.800 / 0.799 / 0.597 / 0.609 (seeds 20260814/1/2/3/4) ->
      mean 0.612 +- 0.199 vs EM 0.80 +- 0.11 and B 0.99 +- 0.24; TACR
      beats EM 1/5. Bull 0.41-1.19 vs B pack 1.77 +- 0.08; exposure
      mean|a| 0.386 vs B 0.264. Basin screening flags all 5 (corr
      0.17-0.78); the verdict is basin-independent. KEEP: the
      threshold-recalibration insight (0.7 corr doesn't transfer from B
      to C) — and now also: the low 3k cross-seed corr was itself partly
      an undertraining artifact (seed 1 rises to 0.73-0.87 at 20k), yet
      the pack converges onto the same underperforming region, so the
      verdict does not move.
    - MARGIN-AUDIT ADDENDUM (2026-08-21, scripts/em_margin_audit.py;
      full tables in 7.9.2): per-seed EM margins
      (Sharpe(a*m) - Sharpe(|a|*m)):
        TACR: -0.20 / -0.62* / -0.04 / -0.06 / +0.05
              (seeds 20260814/1/2/3/4; short_frac 0.05-0.32)
      NEGATIVE in 4/5. *CAVEAT: the seed-1 number is the 20k rerun's
      (7.6.2 Run A) — that rerun OVERWROTE the 3k original checkpoint,
      and Phase-3 eval never saved per-seed action series, so the 3k
      seed-1 margin is unrecoverable. Robustness without seed 1: the
      four intact 3k seeds give -0.20 / -0.04 / -0.06 / +0.05 -> mean
      -0.06, 3/4 negative. The verdict DIRECTION (negative-mean margins
      with an active short side) survives either way; only the
      magnitude (-0.18 vs -0.06) carries the substitution caveat.
      TACR is not failing for lack of shorts — it shorts plenty, and
      the shorts LOSE vs passively holding the same exposure long. The
      1/5 failure is therefore anti-skill, not the long-only tie
      artifact that later showed up in D's win counts (7.9.1 CHECK 2)
      — the count's fragility cannot rescue this verdict. The
      structural finding (val->test transfer failure, below) stands on
      independent evidence; the audit adds that TACR's directional
      signal is worse than same-exposure passive holding on 3-4 of 5
      seeds, i.e. the failure is not confined to checkpoint selection
      — the policy itself is bad.
    - MECHANISM (confirmed by the final-epoch check, 7.6.2): TACR's
      failure mode is val->test transfer. Best-val-epoch selection
      (inherited from B's protocol, where B transfers tightly:
      |val-test| ~0.1-0.2) picks late-epoch checkpoints that memorize
      the val path: seed 1 @ 20k best-val 0.575 @ epoch 17/20 -> test
      0.016; Run B (fast critic) best-val 1.185 @ epoch 17 -> test
      -0.049. At 3k the same seed's best-val 0.539 @ epoch 2/10 (early!)
      transferred (0.800) — early stopping accidentally landed on a good
      region. But the final-epoch test (-0.10 / -0.10) proves the val
      memorization is a symptom of trajectory-level collapse: NO
      checkpoint generalizes, so this is NOT a selection-rule artifact
      — the TACR objective itself never finds a generalizing policy on
      this data.
    - Takeaway for D (fuzzy+transformer+offline RL): TACR's BC+Q
      objective does not transfer to this market; the DT-style triple
      architecture overfits the val path as training lengthens. If D
      reuses a TACR-style backbone: do NOT ship alpha=0.9 with the
      paper's critic_lr 1e-6 (see 7.6.2 Run B — a 1e-4 critic does not
      fix test either), and do NOT trust val-best selection — use
      early-epoch or frozen-policy evaluation.
    - D PROTOCOL IMPLICATIONS (binding, from this phase):
      (1) Do not inherit TACR's actor-critic training dynamics wholesale.
          The Q-collapse (actor loss -> -0.89, Q_bar -> +0.07, val
          memorization, no generalizing checkpoint at any budget or
          critic speed) is a KNOWN failure mode of this objective on
          this data — design against it (BC-dominant schedules, alpha
          budgeted by evidence, or a non-critic objective), don't
          discover it again.
      (2) Final-epoch sanity check is a standing protocol step: every D
          training run rolls BOTH the best-val and the final checkpoint
          and reports both test Sharpes. Built into D's harness from the
          start, not retrofitted like this phase.

  7.6.2 VERIFICATION vs AUTHORS' RELEASED CODE + BUDGET RECHECK (2026-08-18)
    - All implementation claims verified against the released repository
      (github.com/VarML/TACR, branch main, fetched 2026-08-18) — NOT
      inferred from the paper text:
      * Separate critic MLP: tac/training/trainer.py, Trainer.__init__:
        `self.critic = Critic(state_dim, action_dim)` with
        `self.l1 = nn.Linear(state_dim + action_dim, 512);
        self.l2 = nn.Linear(512, 256); self.l3 = nn.Linear(256, 1)`
        + critic_target deepcopy. The Phase-3 spec's shared-trunk
        assumption is contradicted by the code — VERIFIED deviation.
      * Return channel = IMMEDIATE reward, not return-to-go: train.py
        get_batch: `r.append(traj['rewards'][si:si + max_len]...)`;
        seq_trainer.py train_step feeds it as the return input:
        `self.actor.forward(states, actions, rewards, timesteps)`.
        The authors' code never computes cumulative returns. Our
        true-RTG conditioning (per Phase-3 spec) is therefore a VERIFIED
        deviation — as is the paper's own choice of conditioning on the
        immediate realized reward. Both eval with zeros (their test.py
        and our constant rtg_target).
      * critic_lr = 1e-6 is deliberate, not a typo: train.py argparse
        `--critic_learning_rate default 1e-6` + comment
        `# 1e-4 (hightech), 1e-6 (others)` — the NDX/MDAX/CSI-family
        value. The 100x actor/critic gap is the authors' own.
      * Budget 10 x 4000 = 40k steps, warmup 10k, batch 64, clip 0.25,
        polyak tau 0.005 on actor AND critic, actor loss
        -lambda*Q + BC with lambda = alpha/|Q|.abs().mean().detach() —
        line-for-line in train.py / seq_trainer.py. (Divergences inside
        the authors' own repo: critic optimizer is plain Adam; their
        save() references self.actor_optimizer which seq_trainer never
        sets — a latent bug in their code, not ours.)
    - BUDGET RECHECK RUNS (launched 11:13, ~4h ETA at 20:00):
      Run A = seed 1 @ 20k steps (epochs 20 x 1000, warmup 5k = 25% of
      schedule like the paper's 10k/40k, critic_lr 1e-6 paper value) ->
      checkpoints/tacr/s1/ (replaces the 3k s1 checkpoint).
      Run B = seed 1 @ 20k steps with critic_lr 1e-4 (diagnostic: tests
      the starved-critic hypothesis) -> checkpoints/tacr/s1_20k_clr1e-4/s1/.
      Interim read at epoch 11/20: actor collapse identical in both
      (actor loss ~-0.88, Q_bar +0.05) — not a step-budget artifact;
      Run B's val Sharpe (+1.04 vs +0.32) hints the critic speed DOES
      matter for outcome quality. Verdict only after both complete:
      compare test Sharpe + val-test transfer + cross-seed corr of the
      20k seed 1 against the 3k pack, then finalize 7.6.1.

    - PRE-REGISTERED DECISION RULES (written 2026-08-18, BEFORE the runs
      completed — do not move these bars after the fact):
      * "Converged toward the pack" := corr(20k seed-1 test actions, seed s)
        > 0.6 for >= 2 of the original pack seeds {20260814, 2, 3, 4},
        AND val->test transfer holds: |test_all_sharpe - best_val_sharpe|
        < 0.30 (best-val from the 20k training log).
      * Verdict mapping:
        - converged AND test_all >= 0.80 (EM mean): the 3k runs were a
          budget-stub artifact -> KILL the headline; re-run all 5 seeds
          at 20k before any TACR verdict.
        - NOT converged AND test_all < 0.80: 3k verdict CONFIRMED ->
          finalize "TACR < EM on this data" (with the 20k evidence).
        - NOT converged but test_all >= 0.80: seed-instability verdict ->
          KILL the headline; needs a reseed protocol at full budget.
        - converged but test_all < 0.80: ambiguous -> report as such,
          no headline.
      * Collapse-signature comparison (Run A vs B): "same collapse" :=
        actor_loss < 0 sustained for >= 20% of steps AND Q_bar slope
        > 0 over the last 25% of the schedule, in both runs.
      * Run B (critic_lr 1e-4) is PROVISIONAL n=2, NOT a finding. It
        graduates to a candidate finding only if BOTH: B_test_all >=
        A_test_all + 0.15 AND B's transfer holds — and even then it
        needs a third run (e.g., seed 2 @ critic_lr 1e-4) before being
        written up. It then informs Model D's critic lr choice.

    - PRE-REGISTERED RULE OUTCOMES (runs completed 18:30, evaluated
      2026-08-18):
      * Run A (20k, critic_lr 1e-6, paper): best val 0.575 @ ep 17/20;
        TEST all-days Sharpe 0.016 (bull 0.017, bear 0.112, crisis
        -2.650); mean|a| 0.290, short 0.327, turnover 0.094. Both actor
        collapse signature (actor loss -0.894, Q_bar +0.068) and
        sustained-negative rule (>20% of steps) satisfied — collapse
        CONFIRMED as schedule-property, not budget artifact.
      * Run B (20k, critic_lr 1e-4): best val 1.185 @ ep 17/20; TEST
        all-days Sharpe -0.049 (bull -0.038, bear 0.050, crisis -3.017);
        mean|a| 0.303. Same collapse signature (actor -0.892, Q_bar
        +0.090). The interim "3x better val" read was VAL MEMORIZATION,
        not quality — the fast critic overfits the val path harder.
      * Pack convergence: corr(20k s1 test actions, 3k pack) = 0.865 /
        0.551 / 0.837 / 0.726 vs seeds 20260814/2/3/4 -> 3/4 > 0.6
        (rule PASSED). Transfer: |test - best_val| = 0.559 vs 0.30
        (rule FAILED). Conjunction "converged toward pack" = FALSE.
      * Verdict mapping fired: NOT converged AND test < EM mean ->
        FINALIZE "TACR < EM on this data". 3k verdict confirmed at 20k.
      * Run B elevation rule: B_test (-0.049) >= A_test + 0.15 (0.166)?
        NO. Stays a provisional n=2 observation; the starved-critic
        hypothesis is answered (a 100x faster critic does NOT fix test),
        but per the pre-registered rule it is NOT written up as a
        finding. Would need seed 2 @ critic_lr 1e-4 to generalize.
    - Net result: the headline comparison stands; the 13x-budget
      objection is closed with evidence (longer training hurts, does not
      help). The mechanism found at 20k — val-overfit checkpoint
      selection — is the actionable item for Model D (7.6.1 takeaway).

    - FINAL-EPOCH CHECK (reruns 19:16-22:40, OMP_NUM_THREADS=4; pre-
      registered bar written BEFORE they ran: final-epoch test >= 0.60
      -> selection-rule problem; < 0.30 -> structural failure):
      Run A best-val (ep 17, val 0.714): test -0.051; final (ep 20, val
      0.100): test -0.104. Run B best-val (ep 12, val 1.161): test
      -0.055; final (ep 20, val 0.775): test -0.097. Both finals < 0.30
      -> STRUCTURAL FAILURE CONFIRMED: no checkpoint on either trajectory
      (early/mid/late) generalizes; the val-overfit is a symptom of a
      trajectory-level collapse (actor -> -0.89, Q_bar -> +0.07), not a
      selection-rule artifact. The val-selection note remains relevant
      to B's protocol transfer, but the D takeaway is now directly
      confirmed: the TACR objective never finds a generalizing policy on
      this data.
      Reproduction note: the 19:16 reruns were NOT bit-identical to the
      11:13 originals (best val 0.714 vs 0.575 for A @ ep 17; 1.161@ep12
      vs 1.185@ep17 for B) — thread-count fp nondeterminism. Same
      collapse signature, same near-zero/negative test region; all
      verdicts invariant to the drift.

--------------------------------------------------------------------------------
8. KEY COMMANDS AND ARTIFACTS
--------------------------------------------------------------------------------

  python -m src.data.pipeline          # Phase-1 end-to-end
  python -m src.models.ddr.train       # train Model B (30 epochs, best-val)
  python -m src.models.ddr.eval        # test roll -> regime_eval.csv
  python scripts/ddr_robustness.py     # seed sweep + grid + baselines + gaps
  python scripts/ddr_paired_audit.py   # paired returns + contamination
  python scripts/ddr_clean_label_eval.py  # clean-label regimes + paired DD
  python scripts/ddr_leverage_test.py  # leverage-matched decomposition
  python scripts/ddr_vt_retrain.py     # vol-targeted retrain + 3-run decomposition
  python scripts/ddr_seed_variance.py  # naive_new seed-basin analysis (canonical B)
  python scripts/ddr_basin_probe.py    # 5 fresh seeds -> basin frequency (1/10)
  python scripts/audit_data_fix.py     # post-fix data audit (A/B/C checks) -> robustness/data_fix_audit.csv
  python -m src.models.tacr.train      # train Model C (TACR), seed via --seed S
  python -m src.models.tacr.eval       # 5-seed test rolls + basin screening + vs B/EM
  pytest                               # 54 tests

  Artifacts under data/processed/: daily_ohlcv.parquet (HOLES — now
  backfilled in the CURRENT data), features_regimes.parquet,
  offline_dataset.parquet (21,132 rows), dataset_manifest.json.
  Model artifacts under src/models/ddr/checkpoints/: ddr_best.pt,
  training_log.csv, test_predictions.csv, regime_eval.csv, and
  robustness/ (seed_sweep.csv, seed_sweep_summary.csv, grid.csv,
  baselines_test.csv, bear_pnl.csv, gaps_report.csv, paired_summary.csv,
  paired_ddr_vs_bh.csv, contamination_audit.csv, momentum_audit.csv,
  clean_label_regimes.csv, paired_drawdown.csv, baselines_clean_label.csv,
  fully_clean_subset.csv, leverage_test.csv, data_fix_audit.csv), plus vt/ and naive_new/
  (reward-fix + canonical B; seed_variance.csv inside), naive_probe/
  (basin-frequency probe: basin_probe.csv). All gitignored.
  configs/ddr.yaml — shared Model B config (vol-targeting + artifacts).
  Notebooks: 01_data_sanity_check.ipynb, 02_ddr_training.ipynb (executed,
  matches CLI at epochs=30), 03_tacr_training.ipynb (Model C walkthrough).
  Model C artifacts under src/models/tacr/checkpoints/tacr/: s{seed}/
  (tacr_best.pt + training_log.csv), regime_eval.csv, vs_model_b.csv,
  basin_screening.csv. configs/tacr.yaml — Model C config.

7.7 MODEL B — EXPOSURE-REGULARIZATION TEST (2026-08-19)
    - Motivation: B's drift to mean|a| ~ 0.26 was diagnosed as
      under-leveraging (the DSR gradient is level-free — a uniform
      de-leverage cancels in the Sharpe ratio). A variance floor was
      rejected (user): variance can be gamed by erratic flipping
      without sustained exposure; it constrains the wrong quantity.
    - Implemented (src/models/ddr): direct exposure-level penalty on the
      training loss, per block: lambda * |mean(|a_t|) - target_exposure|
      (config fields exposure_regularization / target_exposure /
      exposure_lambda; CLI --exposure-reg/--target-exposure/
      --exposure-lambda/--tag). Default target 0.75 (vol-implied:
      0.15 ann / ~0.18-0.20 SPY vol), lambda 1.0 (vs -D.mean() O(0.1-1)).
      scripts/ddr_exposure_test.py runs the 5-seed protocol (SEEDS
      20260814/1/2/3/4, base = naive DSR like canonical naive_new).
    - RESULT (clean-label days): the penalty hits its target — exp
      mean|a| 0.732 (target 0.75) vs naive_new 0.264 / vt 0.176 — but
      Sharpe COLLAPSES: all-days 0.095 +- 0.531 vs naive_new 0.986 +-
      0.267 / vt 0.806 +- 0.130; bull 0.016 vs 1.506 / 1.319. Short
      fraction rises on most seeds and turnover roughly doubles
      (0.11 -> 0.48 all days). OBSERVED, mechanism UNCONFIRMED.
    - DECISION (canonical): naive_new stays the canonical Model B
      baseline. The exposure-regularization variant is NOT folded in —
      its only verified effects are (1) mechanical: exposure lands at
      the target, (2) destructive: Sharpe collapses relative to both
      canonical baselines at every regime. That decision does not depend
      on why the penalty collapsed the strategy.
    - MECHANISM HYPOTHESES (both consistent with the observed outcome,
      neither yet confirmed):
      (H1) The DSR's level-free sizing is load-bearing: small positions
           encode low confidence, and forcing mean|a| up flattens that
           signal, producing erratic large-position flipping. Predicts
           large positions underperform per unit of exposure (or cluster
           in regimes where the direction call is weak).
      (H2) Generic multi-objective interference: the added penalty
           destabilizes optimization regardless of what naive_new's
           exposure levels mean. Predicts per-unit performance is flat
           across position size in the unconstrained policy.
    - DISCRIMINATOR RESULTS (ddr_exposure_analysis.py, clean days):
      (a) exp-pack is BIMODAL, not uniformly flipped: seed 4 IMPROVED to
          0.734 vs its own naive_new 0.513 (short_frac 0.018, near-zero
          flipping); seeds 1 (-0.648, short 0.90) and 2 (-0.199)
          collapsed; seeds 20260814 (0.309) and 3 (0.280) mildly
          degraded. The aggregate 0.095 +- 0.531 is a mean over
          qualitatively different outcomes -> the mechanism is closer to
          "training instability under a conflicting objective" than to a
          clean behavioral substitution, and forcing exposure up is NOT
          universally destructive (seed 4's counterexample).
      (b) naive_new confidence signal: per-unit performance (sign(a)*R)
          rises across |a| terciles (small 0.0001 -> med 0.0008 -> large
          0.0010), but the continuous correlation is weak (pooled r =
          0.034; per-seed 0.019-0.067). The effect is concentrated in
          the large-position tail, which is the BEAR regime: large |a|
          bucket is 0.40 bear (vs 0.11 for small), and naive_new takes
          its LARGEST positions in bear (mean|a| 0.35-0.43 vs 0.19-0.32
          bull, 0.17-0.34 crisis). So H1 is only PARTIALLY supported:
          size encodes call quality at the bear-short tail, not as a
          clean monotone gradient; H2 (interference) describes the
          majority of seeds.
      (c) corr(|a|, |R_t+1|) is moderate (pooled 0.16): position size
          also scales with market volatility/opportunity, consistent
          with size encoding risk-aware conviction rather than pure
          noise.
      (d) BASIN CONFOUND (from the existing ddr_basin_probe.py result,
          not new computation): the rescue/collapse split aligns
          EXACTLY with the naive_new basin structure. Canonical seed 4
          is the documented outlier basin (cross-seed corr 0.33 vs
          0.93-0.98 for seeds 20260814/1/2/3; late-spiking best-val
          epoch 30 vs 3-5; naive test 0.513 vs 0.85-1.15 clean-day).
          The ONLY exp seed that improved (4, 0.734) is precisely the
          outlier-basin seed; the four good-basin seeds all
          degraded/collapsed (0.31 / -0.65 / -0.20 / 0.28). So the exp
          outcome is confounded with the starting basin: "the penalty
          rescued seed 4" is NOT independent evidence for it — the
          penalty moved an outlier from a weak basin to a mid basin
          (0.734, still below the good-basin naive level) while
          damaging every good-basin seed. And it cuts the other way: a
          seed already in a good basin has the most to lose from level
          interference. Either way: the exposure test does NOT tell us
          whether exposure regularization helps conditional on basin —
          it is entangled with it.
    - NET: canonical-B decision (naive_new unchanged) stands and is
      independent of the mechanism. The exposure-regularization variant
      is a seed-dependent interference effect — worse on 4 seeds,
      better on the weak seed 4 — NOT a clean fix, NOT a clean failure.
      Do not cite "exposure level is load-bearing" or "penalty induces
      flipping" as confirmed claims; cite the bimodal breakdown + the
      weak-but-positive size->quality tail + the basin confound instead.
    - EXPLICIT NEXT STEP (if the exposure question is ever revisited):
      rerun with BASIN-CONTROLLED seeds. Screen multiple candidate seeds
      into the good basin first (the existing rule: best-val epoch 3-5,
      cross-seed corr > 0.6 vs the pack), then apply the penalty only to
      confirmed-good-basin seeds. Five uncontrolled seeds cannot say
      anything about exposure regularization; the confound must be
      designed out, not post-hoc argued away.
    - STRAY THREAD (flagged, not pursued): basin membership under
      naive_new (DSR objective) predicting behavior under a DIFFERENT
      objective (exposure-penalized DSR) suggests basin membership may
      be a property of the feature representation or GRU initialization
      that persists across loss functions — not just a quirk of the DSR
      training dynamics. If Model D shows basin heterogeneity, check
      whether it tracks B's basin structure before blaming D's objective.
    - MEAN-VARIANCE VARIANT (logged, NOT implemented — explicit
      separate variant per user): reward = E[r] - kappa * Var[r],
      replacing the DSR entirely (bigger, defensible departure from
      Moody & Saffell). Advantage: no "shrink variance to raise the
      ratio" escape hatch — variance is subtracted, not divided.
      Cost: a new free parameter kappa needing tuning, and it replaces
      the paper's reward formulation rather than patching it. Decision
      deferred; shares a caveat with the exposure test in that any
      level-imposing objective must respect whatever the confidence
      check reveals about naive_new's position sizing.

7.8 MODEL D SCOPING — COMMITTED DECISIONS (2026-08-18)
    - Fork decision (user, explicit): "uncertainty" in D = FUZZY STATE
      ENCODING (T2F-DT style) — uncertainty lives in the INPUT space; a
      type-2 fuzzy layer fuzzifies the market-state features and the
      transformer consumes fuzzy states. Epistemic trust gating
      (ensemble/evidential head shrinking actions) is NOT part of D.
      Consequence: the Q-collapse failure mode of C is not addressed by
      a trust gate; it must be designed out of the objective instead
      (see below).
    - RATIONALE (user, recorded): epistemic-uncertainty mechanisms are
      entangled with the RL objective itself (CQL/IQL/PBS use
      uncertainty to regularize or penalize Q-values), so layering one
      on top of TACR's already-collapsing actor-critic would build D's
      novel component on the part of the system with direct evidence of
      instability. Fuzzy encoding is a representation-layer
      intervention — it changes what enters the encoder, not how value
      is computed — so the committed ablation (fuzzy-D vs D-minus-
      fuzzy, same encoder/transformer/objective) is cleanly separable,
      and it does not inherit C's failure surface. It is also
      regime-interpretable by construction (membership over low/med/
      high vol, bull/bear/crisis-adjacent), matching the project title
      "regime-aware uncertainty modeling" natively, where an ensemble
      would need a post-hoc correlation step to speak about regimes.
    - Carried from C's phase (binding): (1) no wholesale inheritance of
      TACR's actor-critic dynamics — Q-collapse is a known failure mode
      on this data; (2) final-epoch sanity check is a standing protocol
      step in D's harness (roll best-val AND final checkpoints, report
      both); (3) comparison protocol unchanged: 5 seeds, same eval,
      crisis = 15 test days, wins criterion >= 3/5 vs EM.
    - Open questions the D spec must answer (not yet decided):
      (a) objective — pure BC return-conditioned (DT-style, critic-free)
          vs BC+Q (redesigned: BC-dominant alpha, budgeted) vs IQL;
          [RESOLVED 2026-08-19 (user): D trains with IQL — expectile
          regression on V + advantage-weighted policy extraction. The
          AWR policy is data-anchored by construction and the Q-update
          never samples OOD actions — both properties directly counter
          C's Q-collapse; no C-shaped actor-critic loop. D and
          D-minus-fuzzy share IQL, so the fuzzy layer stays the only
          variable. (CQL rejected: conservatism penalty still requires
          the actor-critic loop + OOD action sampling, structurally
          close to the failed C surface.)]
      (b) return channel for conditioning — true RTG (C's deviation,
          kept per Phase-3 spec) vs immediate reward (authors' channel);
          [RESOLVED 2026-08-19: MOOT under IQL — value-based objective
          conditions implicitly via Q/V; there is no return channel.]
      (c) fuzzy layer design — which features are fuzzified, IT2 vs
          general T2, number of MFs, learned vs fixed MF parameters,
          and whether the layer outputs fuzzy features or rule-
          activation strengths, and how those enter the token embedding;
      (d) context length u=20 (B/C protocol) vs longer;
      (e) budget/warmup policy — C showed longer training hurt the
          critic objective; BC-only may scale differently, so the spec
          should pre-register a budget re-check like C's (7.6.2).
    - RESOLVED FORKS (user, 2026-08-19; spec to be written by the user):
      (c) FUZZY LAYER: IT2 (interval type-2, UMF/LMF) over a 3-feature
          interpretable core — realized_vol_20d, ret_20d (momentum),
          rsi_14 — into low/medium/high; MF parameters FIXED (Gaussian
          footprints), so the ablation isolates fuzzification itself.
          Output: 3 feats x 3 sets x 2 bounds = 18 membership values per
          timestep, concatenated onto the raw 8-feature state.
          EMBEDDING CHECK (verified against C's code): C's per-token
          input dim is 8 (shared FEATURE_COLUMNS: ret_1d/5d/20d,
          realized_vol_20d, rsi_14, macd_hist, volume_zscore_20d,
          bollinger_pos) with embed_dim 128, n_layer 4, n_inner 512.
          D's token input becomes 8 + 18 = 26 -> the ONLY change is the
          input projection Linear(26 -> 128); embed/n_inner/n_layer
          stay identical to C. The fuzzified three are already among
          the raw 8, so no new information source is introduced — only
          a derived, regime-legible representation. No second silent
          deviation rides in with this choice.
      (d) CONTEXT: u=20, protocol-consistent (shared WINDOW_DAYS with
          B/C). u=60 is recorded as a possible FOLLOW-UP ablation if
          longer context ever shows merit — NOT a design decision now;
          it would contaminate D-vs-B/C comparisons with a second
          variable.
      (e) BUDGET: C-matching schedule (epochs 10 x 300 = 3k steps,
          warmup 1k) for direct D-vs-C comparability, PLUS the
          pre-registered 20k re-check (if D clears EM at 3k, rerun at
          20k before finalizing) — the C-phase lesson, built into the
          protocol from the start.

--------------------------------------------------------------------------------
7.9 MODEL D — IMPLEMENTATION AND RESULTS (2026-08-21)
--------------------------------------------------------------------------------
    - Implemented per the user's Phase-4 spec, verbatim scope: new files
      only (src/models/d/, tests/test_model_d.py, configs/model_d.yaml,
      notebooks/04_model_d_training.ipynb, scripts/d_model_run.py);
      ddr/tacr/data/eval modules untouched. D and D-minus-fuzzy share the
      package; the fuzzy layer is a config toggle.
    - ARCHITECTURE AS BUILT: fixed IT2 Gaussian fuzzy layer (18
      memberships: realized_vol_20d/ret_20d/rsi_14 x low/mid/high x
      UMF/LMF; centers at the 33rd/50th/66th TRAIN-split percentiles, UMF
      std = 0.5 x bin width, LMF std = 0.8 x UMF; buffers only, no
      learned params) concatenated onto the raw 8-d state -> 26-d input
      for D, 8-d for D-minus-fuzzy (no padding). Causal transformer state
      ENCODER (C's exact DecoderBlock/CausalSelfAttention imported from
      tacr.policy; embed 128 / n_layer 4 / n_inner 512) producing h_t per
      timestep; IQL heads (Kostrikov et al. 2022): expectile V (tau 0.7),
      Q on (h_t, a), deterministic tanh policy via AWR (beta 3.0). NO
      return-to-go channel — the encoder is a state encoder, not an
      autoregressive RTG-conditioned generator, so the C failure class
      (future-return leakage through the conditioning channel) is
      structurally absent. Optimizer: 3x AdamW lr 3e-4 (paper value), no
      weight decay, no grad clip, warmup 1k, polyak 0.005; batch 64
      (paper 256; CPU-scaled, equals C's batch). Tests (8 new, 62 total
      green): causal masking, fuzzy determinism/passthrough, MF-init
      reproducibility + spec values (percentiles, UMF/LMF widths),
      expectile/AWR loss formulas, 26-vs-8 data boundary + compute-match,
      window left-padding.
    - RESULTS (pre-registered rules applied without adjustment; all-days
      Sharpe, 5 seeds, test 2021+ clipped 2024-12-31):

        3k screen      mean    std    vs EM    final-epoch check
        D (fuzzy)      0.9310  0.0192  4/5     pass (best 0.931 / final 0.894)
        D-minus-fuzzy  0.8807  0.0250  4/5     pass (best 0.881 / final 0.882)
        B (naive_new)  0.9859  0.2390   —      —
        EM control     0.7961  0.1050   —      —
        C (TACR)       0.6121  0.1988  fail    fail (structural collapse)

        20k escalation (both variants trained per the pre-registered
        "escalate BOTH if EITHER clears" rule):
        D (fuzzy)      0.8725  0.0329  3/5     pass (best 0.873 / final 0.871)
        D-minus-fuzzy  0.8837  0.0380  4/5     pass (best 0.884 / final 0.863)

      Ablation verdict: NULL at both budgets (|delta| 0.050 at 3k, 0.011
      at 20k; bar = max(pooled SD, naive_new seed spread 0.240)).
    - READINGS (scoped):
      (1) SCREEN: D clears the pre-registered bar (>= 3/5 vs EM) at both
          budgets. IQL's design goals held: no val-overfit collapse
          anywhere (final-epoch check passes for every variant/budget,
          unlike C where every checkpoint failed), and 20k training
          mildly degrades (0.931 -> 0.873) instead of exploding —
          consistent with mild overfitting, not the trajectory-level
          Q-collapse C showed.
      (2) CENTRAL HYPOTHESIS: NOT SUPPORTED. The fuzzy uncertainty layer
          is indistinguishable from its ablation (delta an order of
          magnitude below the noise floor at 20k). The 18 memberships
          are a deterministic function of 3 of the 8 raw features, and
          the encoder can learn any monotone re-weighting of those
          features from the raw channels alone — at this data scale
          (15.9k transitions, ~3.9k train days) the fixed IT2 encoding
          adds no information the 2-layer input projection cannot
          express. Scope: fixed-MF IT2 over these 3 features, this
          architecture/budget; NOT a claim about learned MFs, richer
          rule bases, or T2F-DT wholesale.
      (3) vs B: D (0.87-0.93) sits inside B's seed spread (0.99 +- 0.24)
          with far lower seed variance (std ~0.02-0.04 vs 0.24) — a
          stability gain, not a performance gain. D's edge over EM comes
          from the same place as B's: long-bull participation +
          exposure cuts in bear (bear Sharpe ~ 0 for both variants;
          short_frac < 1%), not from shorting skill.
      (4) BASIN: cross-seed test-action correlation is high for both
          variants (0.78-0.92) — no B-style outlier basin. Some 20k
          seeds show late best-val epochs (25-37/40), but their
          per-seed outcomes do not separate from the pack, so the flag
          is inert here; noted, not acted on (the 0.7-corr / epoch-frac
          thresholds were calibrated on B and remain coarse).
      (5) Crisis (15 test days): D 0.31-1.05, D-minus-fuzzy 0.42-1.30
          per seed — never a headline.
    - DEVIATIONS (from IQL paper, all flagged in config docstrings):
      batch 256 -> 64 (CPU; equals C's batch); budget ~1M -> 3k/20k
      (CPU-scaled, C-comparable); grad-clip/weight-decay absent (paper
      does not use them; C's 0.25/1e-4 were TACR paper values); warmup
      1k added (C's scaled pattern). One numerical guard: AWR advantage
      clamped at 10 before exp (inactive at daily-return scale).
    - FUTURE WORK (logged, not built): u=60 context ablation; learned
      MF parameters; larger term counts; fuzzy-output (rule-activation)
      variants; mean-variance reward for B (logged in 7.7). None change
      the current verdict.
    - CONCLUSION: Model D is the first model in the project that
      trains stably offline and beats EM control under the
      pre-registered protocol — but the explicit-uncertainty hypothesis
      it was built to test is null on this data. The four-model
      comparison ends: B (naive_new) remains the best headline policy;
      D is the stable-optimization alternative; C failed structurally;
      the uncertainty interventions tested (fuzzy encoding here,
      exposure regularization in 7.7) have both been null.

--------------------------------------------------------------------------------
7.9.1 MODEL D — POST-HOC DIAGNOSTICS: IS THE NULL ABLATION INTERPRETABLE?
      (2026-08-21, run before accepting 7.9's conclusion)
--------------------------------------------------------------------------------
    - MOTIVATION (user, recorded): a null D-vs-D-minus-fuzzy ablation has
      two indistinguishable causes — (a) "fuzzy doesn't help" or (b) "IQL's
      objective barely uses its inputs, converging to a bland
      dataset-anchored policy regardless of representation." A Sharpe
      table alone cannot separate them. Also flagged: D's win rate vs EM
      fell 4/5 -> 3/5 from 3k to 20k while D-minus-fuzzy held 4/5
      (TACR-shaped?), and the spec's basin-screening + compute-matching
      deliverables were not reported per variant. All checked via
      scripts/d_diagnostics.py (checkpoints/diagnostics/*.csv).
    - CHECK 1 — INPUT-SENSITIVITY PERTURBATION TEST (decisive). On the
      test split, perturb ONLY parts of the input matrix (timestep
      embeddings unchanged; best-val checkpoints; corr of the deterministic
      action series vs baseline, mean |delta a|):
        all-shuffle  (whole state of another day):  corr ~ 0.00-0.06, |da| ~ 0.10
        raw-shuffle  (raw 8 of another day, fuzzy kept): corr 0.07-0.15, |da| ~ 0.09
        fuzzy-shuffle (fuzzy 18 of another day, raw kept): corr ~ 0.93, |da| ~ 0.026
        fuzzy-zero   (memberships zeroed, raw kept):      corr ~ 0.96, |da| ~ 0.021
      (consistent across 3k and 20k, all 5 seeds.) READOUT: the policy is
      STRONGLY input-sensitive — confound (b) is ruled out: shuffling
      state content destroys the action series (corr ~ 0) and moves test
      Sharpe (0.93 -> 1.01 at 3k). The raw 8 channels dominate; the fuzzy
      channels ARE read (corr drops to 0.93, |da| ~ 1/4 of action std) but
      contribute marginally, and their information is redundant with raw:
      fuzzy-shuffle Sharpe (0.895 @3k) lands exactly at the D-minus-fuzzy
      baseline (0.881 @3k). Coherent picture: destroying the fuzzy signal
      degrades D to its ablation's level; the ablation null is therefore
      INTERPRETABLE as "fuzzy adds no marginal information over raw at
      this scale," not as an input-insensitivity artifact.
    - CHECK 2 — WIN-RATE CHURN IS A TIE ARTIFACT — AND THE ARTIFACT IS
      PROJECT-WIDE (see CHECK 5). Every seed that "flipped" (won @3k,
      lost @20k) — D seeds 1 and 20260814, D-minus-fuzzy seed 20260814 —
      has short_frac = 0.0000: a long-only policy is IDENTICAL to its EM
      control (a*m == |a|*m), margin exactly 0, counted as a loss under
      the strict > rule. The churn is seeds crossing the long-only
      boundary, not TACR-shaped collapse: per-seed Sharpe moved mildly
      (D seed 1: 0.942 -> 0.871), the final-epoch check passes
      everywhere, and D-minus-fuzzy shows the same tie-churn (its 20k
      "held" 4/5 only because seed 3 crossed back). Every seed with ANY
      short exposure (short_frac >= 0.8%) beats EM at BOTH budgets and
      variants — D's entire EM edge lives in the small short side
      (margins 0.02-0.09), consistent with B.
    - CHECK 3 — BASIN SCREENING PER VARIANT (spec section 0 deliverable;
      thresholds NOT re-calibrated to D a priori, reported raw):
        3k:  D flags 4/5 (seeds 1,2: cross-seed corr 0.689/0.691 just
             under the 0.7 bar; seeds 3,4: best-val ep 8,6 of 10 vs the
             0.35-frac rule) — outcomes uniform (0.903-0.951).
             D-minus-fuzzy flags 1/5 (seed 4, ep 9/10).
        20k: D flags 3/5 (seeds 20260814/2/3, best-val ep 25/34/21 of
             40); D-minus-fuzzy flags the SAME 3 seeds (ep 25/28/37).
      Flagged vs unflagged seeds do not separate in outcome at either
      budget. The within-D sharing of flagged seeds across variants is
      EXPECTED (near-identical architectures + same seed init => same
      optimization-trajectory shape) and carries no cross-architecture
      information. IMPORTANT CORRECTION (user, 2026-08-21): an earlier
      draft read this as "basin structure tracks the seed across
      architectures, matching the 7.7 stray thread" — that claim is
      CONTRADICTED by the seed identities. Under DDR naive_new (7.5),
      the good basin was seeds 20260814/1/2/3 and seed 4 was the
      outlier; D's 20k flagged seeds (20260814/2/3) are drawn from
      DDR's GOOD basin, and DDR's outlier (seed 4) is unflagged at 20k.
      The flag sets are also budget-unstable (3k D flags 1,2,3,4; 20k
      flags 20260814,2,3). Correct reading: D's basin flags are noise
      (see the val-landscape check below), tracking neither outcome nor
      any persistent per-seed property; this is evidence AGAINST the
      7.7 stray thread (cross-objective basin persistence), which
      remains open and now has one data point against it.
    - CHECK 3b — VAL LANDSCAPE (direct check, replaces the inferred
      "flat landscape" read). Val Sharpe over epochs 10-20 of the 20k
      runs sits on a TRENDLESS plateau for every seed, flagged or not:
      mean 0.68-0.77, epoch-to-epoch std 0.07-0.19 (pooled ~0.13). The
      best-val "peaks" are 1.7-4.9 sigma noise spikes above the plateau
      (best_val 0.94-1.21), so the best-val EPOCH (anywhere from 2 to
      37 of 40) is the location of the largest noise draw, not a
      trend peak — uninformative about selection pathology. This is
      consistent with (and now directly explains) the passing
      final-epoch checks: any epoch's policy is a plateau sample.
    - CHECK 4 — COMPUTE-MATCH VERIFIED from the configs saved in each
      checkpoint (not the repo yaml): epochs/steps_per_epoch/batch_size/
      warmup/lr/expectile/temperature identical across D and
      D-minus-fuzzy at matched budgets (10x300 and 40x500, batch 64);
      samples-per-step identical; only the input Linear width (26 vs 8)
      differs, and the shared transformer blocks dominate compute.
    - CHECK 5 — THE "BEATS EM" CRITERION IS A KNIFE-EDGE, RETROACTIVELY.
      The tie structure exposed in CHECK 2 is not D-specific: a
      long-only policy is EXACTLY its EM control, and every model in
      this project is ~long-only (short_frac: B 17%, C/D < 2%), so each
      X/5-vs-EM win-count compresses a continuous, thin short-side
      margin (D: 0.02-0.09; C's margins were negative; B's are
      documented in 7.5) into a fragile binary: "won" means "took a
      short position AND it helped," "lost" usually means "took none."
      This reframes — but does not overturn — the verdicts that leaned
      on win counts: B's canonical selection (7.5), C's 1/5 failure
      (7.6.1), D's screen pass (7.9). C's failure survives (its
      margins are negative even where shorts exist, and the structural
      evidence is independent); D's screen pass survives (positive
      margins in every short-taking seed at both budgets); B's margins
      were the widest. PROTOCOL RULE going forward: report per-seed EM
      margins alongside every win count, never the count alone.
      (Retroactive re-derivation with full tables: 7.9.2.)
    - VERDICT ON 7.9's CONCLUSION: it STANDS, now with the sensitivity
      evidence attached — "central hypothesis not supported" is scoped to
      "the fixed IT2 encoding over these 3 features adds no marginal
      information over the raw channels for this objective at this data
      scale," verified against the input-insensitivity confound. The
      4/5 -> 3/5 win-rate change is a long-only tie artifact, not
      regression. CAVEAT attached to the verdict (CHECK 5): "beats EM"
      verdicts everywhere in this project rest on thin short-side
      margins; report margins alongside win counts in any writeup.
      Basin flags are plateau noise (CHECK 3/3b) and say nothing about
      seeds or architectures. All diagnostics reproducible:
      python scripts/d_diagnostics.py

--------------------------------------------------------------------------------
7.9.2 EM-MARGIN AUDIT — EXPLICIT RE-DERIVATION OF THE §7.5/§7.6 VERDICTS
      (2026-08-21, scripts/em_margin_audit.py, user-mandated)
--------------------------------------------------------------------------------
    - WHY: 7.9.1 CHECK 5 showed "beats EM X/5" is a knife-edge criterion
      (long-only policy == its EM control exactly; margins are thin
      wherever shorts are thin). Every phase verdict that leaned on a
      win count must therefore be re-derived with the margins shown,
      not asserted. This section is that pass; addenda were also
      inserted at the point of use (7.5.5 and 7.6.1) so each section
      stands alone. Reproduce: python scripts/em_margin_audit.py
      (also writes diagnostics/em_margin_audit.csv).
    - MARGIN DEFINITION: per seed, margin = Sharpe(a*m) - Sharpe(|a|*m)
      where EM is the policy's OWN exposure-matched control (same |a|,
      long-only). Positive margin = the directional signal (the short
      side + timing) adds value over passively holding the same
      exposure. Ties (margin exactly 0) arise iff short_frac = 0.
    - FULL TABLE (per-seed margins; seeds 20260814/1/2/3/4):

        B naive_new   +0.17  +0.31  +0.28  +0.28  -0.08   mean +0.190
        B vt          -0.18  +0.03  +0.01  +0.03  +0.12   mean +0.003
        C (TACR)      -0.20  -0.62* -0.04  -0.06  +0.05   mean -0.176
        D @3k         +0.01  +0.02  +0.00  +0.03  +0.09   mean +0.029
        D-minus @3k   +0.02  +0.02  +0.00  +0.00  +0.09   mean +0.027
        D @20k        +0.00  +0.00  +0.03  +0.04  +0.04   mean +0.021
        D-minus @20k  +0.00  +0.00  +0.03  +0.02  +0.03   mean +0.017

      (short_frac: B naive_new 0.14-0.31; B vt 0.22-0.49; C 0.05-0.32;
      D < 0.025 everywhere — D's zero-margins are exactly its zero-short
      seeds. B rows verified against the on-disk Phase-2 checkpoints:
      pack means 1.104 / 0.806 match 7.5.3's documented values.)
      *CAVEAT: C's seed-1 margin is the 20k rerun's — 7.6.2's Run A
      overwrote the 3k original (confirmed by reading the checkpoint's
      saved config: epochs 20x1000, best_val 0.714, epoch 17), and no
      per-seed action series was saved in Phase 3, so the 3k seed-1
      margin is unrecoverable. The audit script now AUTO-DETECTS the
      budget of every C checkpoint. Robustness: the four intact 3k C
      seeds give mean -0.06 with 3/4 negative — the anti-skill
      direction survives; only the magnitude claim (-0.18) carries
      the substitution caveat.
    - THE COUNT COMPRESSES THREE DISTINCT REGIMES into one binary:
        robust skill   : B naive_new — margins ~0.2, driven by a real
                         short side (13-31% of days), negative only in
                         the known seed-4 outlier basin.
        knife-edge     : B vt and D — margins ~0.01-0.03 (or 0 by
                         construction); "wins" mean "took a small short
                         position and it helped slightly."
        anti-skill     : C — margins negative in 4/5; shorts actively
                         lose vs passive same-exposure holding.
    - RE-DERIVATION 1 — canonical-B decision (7.5.5): SURVIVES,
      STRENGTHENED. The win count called naive_new and vt EQUIVALENT
      (both 4/5); the margins separate them by ~70x (mean +0.190 vs
      +0.003). vt's 4/5 is itself knife-edge (three margins < 0.035).
      The canonical choice of naive_new is now margin-backed, not
      count-backed. Confidence UP.
    - RE-DERIVATION 2 — C's structural failure (7.6.1): SURVIVES,
      SHARPENED. C's 1/5 is not a tie artifact — C shorts on 5-32% of
      days and the margin is negative in 4/5 (mean -0.18 with the
      20k-substituted seed 1; mean -0.06, 3/4 negative, on the intact
      3k seeds): the policy is worse than passive same-exposure
      holding, independent of checkpoint selection. The val->test
      transfer failure remains the structural mechanism; the audit
      shows the deficiency is not confined to selection — the learned
      directional signal itself is anti-informative. Confidence UP.
      (Scope unchanged: "TACR as reproduced here, on this data.")
    - RE-DERIVATION 3 — D's screen pass (7.9): SURVIVES, RE-CLASSIFIED.
      Every D margin is >= 0 at both budgets (never negative — IQL's
      AWR anchoring avoids C's anti-skill), so the "clears the bar"
      verdict stands. But the edge is an order of magnitude thinner
      than B's (+0.02..+0.03 vs +0.19) and tie-driven on every
      zero-short seed. Honest framing: D's screen pass demonstrates
      STABILITY (no anti-skill, no collapse), not skill comparable to
      B's. Any writeup comparing B and D must state this in margin
      terms, not win counts.
    - PROJECT-LEVEL CONCLUSION: the operative "beats EM" criterion was
      under-powered from Phase 2 onward — it could not distinguish
      robust skill from knife-edge luck from anti-skill, and it
      silently equated "took no short positions" with "lost." The
      project's substantive verdicts survive the re-derivation
      (canonical-B: strengthened; C failure: sharpened; D screen:
      re-classified as thin), but ALL headline tables in any writeup
      must carry per-seed margins, and the final-models summary should
      be read as: B = wide-margin skill, D = thin-margin stability,
      C = negative-margin failure.
    - OPEN QUESTION (recorded, not chased): whether margin SIZE (not
      just sign) tracks anything nameable across the three regimes —
      e.g. training objective (B's wide margins came from direct DSR
      backprop; C's negative ones from actor-critic; D's thin-positive
      ones from IQL/AWR anchoring — a tempting but n=1-per-objective
      pattern), or basin structure (no test available for C: its 3k
      pack is UNIFORMLY flagged by the screening, so there is no
      flagged/unflagged contrast to compare margins against). Three
      models are too few to attribute; any future variant should log
      margins per seed from the start so this question can accumulate
      evidence instead of being re-derived.

--------------------------------------------------------------------------------
7.10 MODEL C — STRUCTURAL-FIX TRIALS (PRE-REGISTRATION, 2026-08-24)
------------------------------------------------------------------------------
    - MOTIVATION: 7.6 established C's failure as a trajectory-level
      Q-collapse (actor loss -> -0.89, Q_bar -> +0.07, no generalizing
      checkpoint at 3k or 20k, faster critic makes it worse). The prior
      attempts were SOFT (bigger budget, faster critic); this phase tests
      three STRUCTURAL changes to the optimization geometry, per user:
      A. Clipped double Q (TD3-style): TD target = r + gamma*min(Q1,Q2)
         with two independent critics; actor's Q = min(Q1,Q2). Pessimistic
         bias caps overestimation-driven collapse.
      B. BCQ-style hard action constraint: a generative model
         (ConditionalGaussian, p(a|s), fit on the logged train actions of
         all four behavior policies, frozen) generates a "data-support"
         action; the actor's proposal is pinned to within +-bcq_phi (0.5)
         of it for BOTH the critic bootstrap target and the actor loss;
         the BC term is dropped (bcq_bc_coeff=0) — the constraint is the
         regularizer. OOD escape is mathematically closed regardless of Q.
      C. PAR (direction-aware BC target replacement): per element, if the
         Q-gradient direction opposes the BC direction (cos < thresh=0),
         replace the BC target a_logged with a_proj = a_logged + phi*sign(g_Q)
         (aligned with Q, within phi=0.5 of the logged action = inside the
         data support); otherwise BC unchanged. Aims to stop the
         BC-vs-Q tug-of-war on the actor.
    - IMPLEMENTATION: src/models/tacr/{fixes.py,action_model.py}, config
      fields use_double_q / use_bcq / use_par + params (configs/tacr.yaml
      + CLI flags), train.py restructured to n-critic + optional action
      model (saved in the checkpoint), eval.py threads the constraint into
      the roll (deployed policy == constrained policy; val selection and
      test rolls both apply it). Defaults keep the paper-faithful
      objective byte-for-byte (all flags off). 9 new unit/smoke tests,
      suite 72/72.
    - PRE-REGISTERED PROTOCOL (written BEFORE any run; bars fixed):
      * Diagnostic: seed 1, 3k budget (10x300, warmup 1k, paper
        hyperparameters untouched — critic_lr 1e-6), four runs:
        control (repro), fix_a (double Q), fix_b (BCQ phi 0.5),
        fix_c (PAR phi 0.5, thresh 0). Tags checkpoints/tacr/{ctl,fix_a,
        fix_b,fix_c}/s1.
      * A fix "suppresses the collapse" iff the 7.6.2 sustained-negative
        rule FAILS: NOT (actor_loss < 0 for >= 20% of logged steps) AND
        NOT (Q_bar slope > 0 over the last 25% of the schedule).
      * A fix "clears the diagnostic bar" iff BOTH: (1) collapse
        suppressed, (2) final-epoch check passes (final-epoch test Sharpe
        >= 0.30, the 7.6.2 structural-failure bar) AND (3) val->test
        transfer holds (|test_best - best_val| < 0.30).
      * Escalation: any fix clearing the diagnostic bar gets the full
        5-seed pack at 3k; pre-registered comparison = per-seed EM
        margins AND win counts (protocol rule 7.9.1 CHECK 5), wins bar
        >= 3/5 vs EM, final-epoch check per seed, crisis (15 test days)
        never a headline.
    - MECHANISM NOTE (pre-registered expectation, will be checked): A and
      B attack Q inflation directly (the collapse's root); C does not
      bound Q, so its lone lever (BC re-targeting) may be too weak against
      the ~20x lambda-scaled Q term — predicted ordering A,B > C if the
      mechanism diagnosis is right.
    - DIAGNOSTIC RESULTS (seed 1, 3k, paper hyperparameters; rolled
      best-val AND final checkpoints per the standing protocol; all
      per-seed, single draw):
        run    actor_loss end | Q_bar end | collapse?  best-val(test)  final(test)  EM margin  mean|a|  short
        ctl    -0.70 rising  | +0.018 up | YES       0.8003           -0.52        -0.003     0.795   0.004
        fix_a   +0.91 flat   | -0.176 down| NO       0.8449           -0.36        +0.218     0.393   0.063
        fix_b  -0.89 rising  | +0.037 up | YES       0.5080           +0.61        +0.026     0.307   0.112
        fix_c  -0.82 rising  | +0.059 up | YES      -0.1364           -0.14        -0.579     0.218   0.575
      (ctl reproduces the historical seed-1 3k result: best-val 0.539 @
      ep 2 -> test 0.800, final test strongly negative.)
    - PRE-REGISTERED BARS APPLIED WITHOUT ADJUSTMENT:
      * collapse suppressed: fix_a YES (actor loss positive the whole
        schedule, Q decreasing) — the ONLY one. fix_b NO (Q still
        inflates 0.016->0.037: the hard constraint fails because the
        4-policy behavior data covers ~the whole [-1,1] action box
        (random policy), so "within phi of a generated action" barely
        constrains the bootstrap and the actor's Q chase). fix_c NO
        (Q inflates 0.028->0.059: the BC re-target is weight 1.0 vs the
        ~20x lambda-scaled Q term, as pre-registered).
      * diagnostic bar (suppression AND final-epoch test >= 0.30 AND
        |test_best - best_val| < 0.30): NO variant clears. fix_a fails
        the final-epoch (-0.36) and transfer (|0.845-0.086|=0.76) bars;
        fix_b/fix_c fail suppression. Per the pre-registered protocol
        NO escalation to the 5-seed pack.
    - READING (mechanism vs selection): fix_a is the ONLY variant that
      kills the collapse signature, and its best-val checkpoint (the
      operative protocol selection) yields test 0.845 with EM margin
      +0.218 — the largest margin of any TACR run in the project (vs
      control ~0; C's were negative). Its final-epoch failure (-0.36)
      is NOT the C collapse (actor loss positive, Q falling throughout)
      but late-schedule drift on a noisy val; the standing final-epoch
      bar was calibrated on C's every-checkpoint-collapse and does not
      distinguish this. The transfer bar (|test-val|<0.30) is near-
      impossible for TACR's single-path noisy val (best-val = noise-spike
      epoch), unlike B's tight val. So fix_a is mechanism-level SUCCESS
      + selection-rule unresolved on a single seed; fix_b's constraint is
      structurally ineffective on this benchmark's full-box behavior
      data; fix_c is actively harmful (anti-skill shorts, margin -0.58).
    - DECISION (recorded 2026-08-24): per pre-registration the diagnostic
      did not clear, so NO automatic escalation. fix_a's mechanism win
      and margin are a single-seed draw; whether to run its 5-seed pack
      under a revised protocol (val-noise-aware final-epoch + margin
      bars) is left to the user. Runs are cheap (~4 min/run, 3-parallel
      at 4 threads).
    - PACK ESCALATION (user decision 2026-08-24: run 5-seed packs for
      BOTH fix_a and fix_b). Revised pre-registered protocol, written
      BEFORE the runs:
      * Seeds 20260814/1/2/3/4, 3k budget (10x300, warmup 1k, paper
        hyperparameters, critic_lr 1e-6), OMP_NUM_THREADS=4 for every
        run in both packs (thread-count consistency within a pack).
      * METRIC (protocol rule 7.9.1 CHECK 5): report per-seed EM margins
        alongside every win count. margin = Sharpe(a*m) - Sharpe(|a|*m).
      * BAR 1 (primary): margin > 0 in >= 3/5 seeds.
      * BAR 2 (secondary, the "does it lift Sharpe" question): pack mean
        best-val test Sharpe > 0.612 (the historical C 3k pack mean:
        0.256/0.800/0.799/0.597/0.609). No ctl 3k pack was run this
        phase, so the historical C pack is the reference.
      * final-epoch check: REPORTED per seed (best-val AND final test
        Sharpe), NOT a pass/fail bar — the diagnostic showed TACR's
        final-epoch failure here is val-noise drift, not the C collapse
        the 7.6.2 bar was calibrated for. A final-epoch value < 0.30
        with a positive best-val test is read as drift, not structural
        failure; a pack where BOTH are < 0.30 everywhere would re-open
        the structural-failure reading.
      * Basin screening: cross-seed test-action corr reported raw;
        thresholds were calibrated on B (7.5.5) and are noted inert.
      * Crisis (15 test days) never a headline.
      * Reproduce: python -m src.models.tacr.train --seed S --tag
        fix_a|fix_b --use-double-q|--use-bcq; eval via
        python -m src.models.tacr.eval on the tag dirs.
    - PACK RESULTS (5 seeds x 3k, per-seed EM margins + win counts per
      protocol rule 7.9.1; artifacts fix_a_pack.csv / fix_b_pack.csv):
        fix_a (double Q):  best-val TEST all per seed: 0.850 / 0.845 /
        1.039 / 0.580 / 0.706  ->  mean 0.804 +- 0.172
                           margins +0.124 / +0.218 / +0.074 / -0.073 /
                           +0.182  ->  4/5 positive, mean +0.105
                           bull 1.04-1.45; mean|a| 0.23-0.39, short
                           0.03-0.20; final-epoch test -0.23/-0.36/
                           -0.08/-0.41/+0.00 (ALL negative-ish -> drift).
        fix_b (BCQ):       best-val TEST all per seed: -0.097 / 0.508 /
        0.862 / 0.699 / -0.539  ->  mean 0.287 +- 0.587
                           margins -0.746 / +0.026 / +0.000 / -0.052 /
                           -1.072  ->  1/5 positive (one is a long-only
                           tie, margin exactly 0). final-epoch test mean
                           +0.500 (INVERTED: later checkpoints better).
    - PRE-REGISTERED BAR OUTCOMES:
      * fix_a: BAR 1 PASS (margin > 0 in 4/5), BAR 2 PASS (0.804 >
        0.612, the historical C 3k pack mean). FIRST TACR configuration
         in the project with positive pack margins (vs C's negative, the
         control's ~0) and a pack mean above the exposure-matched control
         (0.804 vs EM mean 0.699). Still below Model B's pack (0.99-1.10);
         the Q-collapse is genuinely gone (best-val checkpoints transfer:
         every seed's best-val test positive 0.58-1.04, vs C where no
         checkpoint generalized).
      * fix_b: BAR 1 FAIL (1/5), BAR 2 FAIL (0.287). Diagnostic confirmed
         at pack scale: the hard constraint does not rescue this objective
         on this benchmark (the behavior data's support is ~the full
         [-1,1] box, so the constraint is ineffective; seed 4 is
         catastrophic -1.07 margin at 99.6% short).
    - CAVEATS ON fix_a (report these with any writeup):
      (1) NO cohesive basin: cross-seed test-action corr 0.320 (vs B's
          0.93-0.98 good-basin). Each seed's best-val is its own noise
          draw; the 4/5 margin is a per-seed property, not a shared
          policy. The B-calibrated basin thresholds are inert here (raw
          corr reported, per pre-registration).
      (2) final-epoch drift: EVERY fix_a final checkpoint fails test
          (mean -0.21) while every best-val checkpoint works. This is
          NOT the C collapse (actor loss positive, Q falling throughout
          the schedule) but late-schedule train overfit. Consequence:
          best-val selection is MANDATORY — the paper's own final-model
          checkpointing would still fail. fix_a fixes the collapse, not
          the val-selection requirement.
      (3) Crisis (15 test days) never a headline. Hyperparameters are
          the paper's (critic_lr 1e-6); double-Q at a faster critic is
          untested (would be a follow-up, not this phase).
    - NET ANSWER TO THE PHASE QUESTION ("can structural fixes lift C's
      Sharpe?"): YES for the Q-collapse mechanism — clipped double Q
      (fix_a) turns TACR from structural failure (C: 0.61, negative
      margins, no generalizing checkpoint) into a margin-positive policy
      (0.80 pack mean, +0.105 margins, beats EM 4/5) — the first TACR
      variant to do so. It does NOT reach Model B's level and remains
      seed-noisy and selection-dependent. The other two structural fixes
      do not work on this data: BCQ-style hard constraints (fix_b) fail
      because the 4-policy behavior data covers the full action box, and
      PAR (fix_c, diagnostic only) is actively harmful (anti-skill
      shorts, margin -0.58).

------------------------------------------------------------------------------
7.11 MULTI-WINDOW BEHAVIOR POLICY FAMILIES (2026-08-25)
------------------------------------------------------------------------------
    - MOTIVATION (user): the demonstrators were single-horizon (momentum 20d,
      mean_reversion 5d). Asked to diversify across time scales: momentum
      {30,40,60,80,100,120,150,200,250} days, mean_reversion {1..20} days.
    - SCOPE: Phase-1 data layer only (configs/data.yaml, src/data/). The RL
      state is UNCHANGED (8 Phase-1 features); the multi-horizon returns are
      computed from the daily close purely to derive the behavior actions.
      Downstream models B/C/D consume the same 8-d state.
    - DESIGN:
      * momentum / mean_reversion are now FAMILIES, expanded into one policy
        per window tagged "<family>_<N>d" (e.g. momentum_250d). 31 policies
        total = 9 momentum + 20 mean_reversion + buy_and_hold + random.
      * Scale per window: scale_N = base * sqrt(N / ref_window) so the
        position distribution is comparable across horizons (cumulative-return
        std grows ~sqrt(N)); verified flat on the real data (mean|a| ~0.27
        across all 20 mean-reversion windows; momentum_200/250d slightly
        elevated due to regime skew).
      * roll_policy rewritten VECTORIZED (was a per-day Python loop): the
        full 31-policy build dropped 38s -> 0.36s.
      * C/D loaders now derive the policy set from the dataset (no hardcoded
        "4 policies") and INTERSECT dates across selected policies, because
        windowed families start after their lookback warm-up -> their
        trajectories have different date coverage than buy_and_hold/random.
      * DDR loader default behavior_policies changed to ("buy_and_hold",)
        (the old "momentum" tag no longer exists; any single policy selects
        the unique shared-state path).
    - DATA: offline_dataset.parquet 21,132 -> 163,233 rows (31 policies x
      up to 5,283 dates). State stays 8-d. Manifest updated.
    - TESTS: updated test_offline_dataset.py (policy set derived from config,
      per-window short/long checks, per-policy-warm-up transition count) and
      test_tacr.py (calendar subset + a latent index bug: the window-alignment
      test was indexing DDR windows with arange(len(shared)) which only
      coincided when shared == the full DDR calendar). Suite 72/72.
    - NOTE (reporting): an ad-hoc per-window baseline table is NOT a finding.
      A naive drop of the rolling warm-up rows at the test start injects NaN
      into the Sharpe, so any such table must filter warm-up (as the pipeline
      does). The purpose of this change is the offline DATA distribution
      (broader behavior support for BC/offline-RL), not a claim about which
      single horizon trades best.
    - EFFECT ON DOWNSTREAM (not yet re-trained): C/D now train on 31
      trajectories (7.75x more logged data), which changes the behavior
      support the BC term anchors to and what IQL/AWR extracts. The earlier
      structural-fix results (7.10) were trained on the OLD 4-policy dataset
      and are NOT comparable to any new run until re-run on this data. The
      next step is a retrain + re-eval pass on the new dataset.

------------------------------------------------------------------------------
7.12 TACR — RANDOM POLICY EXCLUDED + RETRAIN (2026-08-25)
------------------------------------------------------------------------------
    - MOTIVATION (user): drop the uniform `random` demonstrator from TACR's
      training trajectories (its noise dilutes the BC anchor and inflates the
      action support the critic sees) and retrain on the window-expanded
      policy set.
    - SCOPE: TACR only. The offline dataset keeps `random` (D and the
      baselines still use it); TACR excludes it at load time via a new config
      field ``exclude_policies`` (configs/tacr.yaml -> ("random",), CLI
      --exclude-policies / --no-exclude).
    - EFFECT ON TRAINING DATA: 31 -> 30 trajectories, same date coverage
      (random had full coverage, so the intersection is unchanged).
    - PRE-REGISTERED PROTOCOL (before training): retrain the baseline (C) and
      the double-Q variant (fix_a) on the new data with random excluded, 5
      seeds, 3k budget, paper hyperparameters (critic_lr 1e-6), OMP_NUM_THREADS
      = 4. Report per-seed EM margins + win counts (7.9.1 rule), final-epoch
      check, cross-seed corr. Comparison vs EM (0.70 mean) and B (0.99). The
      fix_a-vs-C margin contrast (was +0.105 vs ~0 on the OLD data) is the
      headline; the random-drop is a data-support intervention that should
      sharpen BCQ-style reasoning but double-Q is independent of the support.
    - RESULTS (5 seeds x 3k, random excluded, window-expanded policies;
      artifacts checkpoints/tacr/{ctl_nr,fix_a_nr}/s{seed}/):
        ctl_nr (C baseline):  TEST per seed 0.906/0.644/0.757/-0.104/0.748
          mean 0.590 +- 0.40 | margins +0.03/+0.17/-0.00/-0.83/0.00 -> 2/5
          positive, mean -0.13 | final mean +0.13 | cross-seed corr 0.32.
        fix_a_nr (double Q):  TEST 0.843/0.778/0.717/-0.469/0.608
          mean 0.495 +- 0.55 | margins +0.00/+0.11/-0.04/-0.97/-0.02 -> 1/5
          positive, mean -0.18 | final mean +0.37 | corr 0.15.
    - PRE-REGISTERED BAR OUTCOMES: ctl_nr FAILS the margin bar (2/5);
      fix_a_nr FAILS (1/5). vs the OLD 4-policy data: C unchanged-in-kind
      (0.590 vs 0.612; mean margin -0.13 vs -0.18), fix_a REGRESSED
      (0.495 vs 0.804; 1/5 vs 4/5 margins; corr 0.15 vs 0.32 — much more
      seed-fragile).
    - READING (negative/null result, stated plainly): dropping `random`
      and widening to 30 windowed policies did NOT lift TACR. The
      hypothesis "random's noise dilutes the BC anchor" is NOT supported
      by the outcome. DOMINANT CONFOUND (design-level, flagged): sample_batch
      samples a policy UNIFORMLY per batch element, so the mean_reversion
      family (20 of 30 policies = 67% of the BC signal) now dominates the
      anchor, vs momentum (30%) + B&H (3%). This is a different distribution
      than the old 25%-each 4-policy set, so "same data minus random" is not
      what was tested — the intervention bundled (a) random-drop with (b) a
      family re-weighting toward mean_reversion. A family-BALANCED sampler
      (weight each FAMILY equally, then window inside it) is the untested
      follow-up that isolates (a) from (b). The 7.10 fix_a finding (on the
      old 4-policy data) is NOT invalidated by this — different data.
    - ACTIONABLE: the retrain does not change the model verdicts; B remains
      the strongest baseline, C/fix variants remain below EM on this data
      with high seed variance.

------------------------------------------------------------------------------
7.13 FAMILY-BALANCED BATCH SAMPLER (2026-08-25)
------------------------------------------------------------------------------
    - MOTIVATION (user): 7.12 showed the uniform-over-policies sampler lets
      the 20-window mean_reversion family supply ~67% of the BC signal and
      hijack the anchor. Implement a FAMILY-BALANCED sampler: first pick a
      family with probability 1/N, then a policy/window uniformly inside it.
      Prediction (user): Double-Q should RECOVER ~0.80 Sharpe because the BC
      term stops forcing a stylistic bias, letting the Critic guide on value
      rather than data frequency.
    - IMPLEMENTATION: sample_batch(..., balanced_families=True) in
      src/models/tacr/data.py (+_family_of helper, policy_idx returned for
      accounting); TACRConfig.balanced_families (configs/tacr.yaml default
      true; CLI --balanced-families / --no-balanced-families). Family set is
      derived from the loaded policy tags: with the standing exclude of
      random there are 3 families (momentum 9, mean_reversion 20,
      buy_and_hold 1) -> each gets 1/3 of every batch. Unit test asserts the
      1/N family distribution (and that the uniform sampler lets the 3-member
      mrev family dominate). Suite 73/73.
    - PRE-REGISTERED PROTOCOL (before any run): two 5-seed packs on the same
      data as 7.12 (windowed policies, random excluded), balanced_families
      = true, 3k budget, paper hyperparameters, OMP_NUM_THREADS=4:
      ctl_bal (baseline C) and fix_a_bal (double Q).
      * PRIMARY BAR (the user's prediction): fix_a_bal pack mean test Sharpe
        >= 0.70 (between the 7.12 fix_a_nr 0.495 and the old-data 0.804) AND
        margins > 0 in >= 3/5 seeds.
      * SECONDARY: ctl_bal beats ctl_nr (0.590) and fix_a_bal beats fix_a_nr
        (0.495) — the balanced sampler must beat the uniform sampler on the
        SAME data to be a win (not just vs the old data).
      * Reported: per-seed EM margins + win counts, final-epoch check,
        cross-seed corr. NOTE (design, pre-flagged): with random excluded,
        buy_and_hold is a 1-policy family getting 1/3 of the BC signal —
        a possible long-only bias; the result will show it.
    - RESULTS (5 seeds x 3k, balanced sampler, random excluded; verified the
      checkpoint configs carry balanced_families=True; artifacts
      checkpoints/tacr/{ctl_bal,fix_a_bal}/s{seed}/):
        ctl_bal (C):     TEST 0.871/0.049/0.753/0.802/-0.164
          mean 0.462 +- 0.48 | margins 1/5 (mean -0.21) | final -0.27 |
          corr 0.06.
        fix_a_bal (2Q):  TEST 0.744/0.785/0.684/-0.247/-0.415
          mean 0.310 +- 0.59 | margins 1/5 (mean -0.27) | final -0.09 |
          corr 0.10.
    - PRE-REGISTERED BAR OUTCOMES: the USER PREDICTION IS FALSIFIED. fix_a_bal
      does NOT recover ~0.80 (0.310); PRIMARY BAR FAIL (test 0.31 < 0.70,
      margins 1/5 < 3/5). SECONDARY (beat the uniform sampler on same data)
      FAIL: ctl_bal 0.462 < ctl_nr 0.590; fix_a_bal 0.310 < fix_a_nr 0.495.
      The balanced sampler is strictly WORSE and more seed-fragile
      (cross-seed corr collapsed to 0.06-0.10, vs 0.32 uniform).
    - MECHANISM (hypothesis, consistent with the outcome): with random
      excluded there are THREE families, and buy_and_hold (a constant +1
      action, zero conditional information) becomes a 1-policy family
      receiving 1/3 of every batch — a degenerate, always-long anchor that
      conflicts with the Q-gradient's regime timing (witness the erratic
      high short_frac 0.71-0.90 in the bad seeds). Family-balancing also
      dilutes each WINDOW's share (an mrev window went from 1/30 to
      1/60 of the batch), making the per-window BC prior sparser. The design
      error: B&H is not a "behavior family" in the signal sense. If
      family-balancing is revisited, it should run over the TWO signal
      families (momentum / mean_reversion, 50/50) with buy_and_hold
      excluded or down-weighted — untested, not run (per protocol).
    - NET: three negative results in a row on the new data (7.12 uniform
      no-random, 7.13 balanced). TACR remains below EM/B on this data
      regardless of the BC-prior aggregation; the objective itself is the
      limiter, not the sampler.

------------------------------------------------------------------------------
7.14 DOUBLE-Q MEAN AGGREGATION + RANDOM FAMILY RE-INCLUDED (2026-08-25)
------------------------------------------------------------------------------
    - MOTIVATION (user): (1) re-include the `random` policy in TACR's family
      set — under the family-balanced sampler it is a 1-policy family capped
      at 1/N of the batch, so its noise is bounded while still covering the
      action space; (2) switch the double-Q aggregation from the pessimistic
      min to the MEAN of the two critics (ensemble Q, no downward bias).
    - IMPLEMENTATION: double_q_mean + double_q(q1,q2,mode) in fixes.py;
      TACRConfig.double_q_mode ("min"|"mean", yaml default mean, CLI
      --double-q-mode); configs/tacr.yaml exclude_policies: [] (random
      re-included -> 4 families: momentum 9, mean_reversion 20,
      buy_and_hold 1, random 1, each 1/4 of every batch). Tests 75/75.
    - PRE-REGISTERED PROTOCOL (before running): two 5-seed packs, 3k, paper
      hyperparameters, OMP_NUM_THREADS=4:
      ctl_mix   = balanced sampler + random included, single critic.
      fix_a_mean = balanced sampler + random included + double-Q MEAN.
      * RECOVERY BAR (fix_a_mean): pack mean test >= 0.70 AND margins > 0 in
        >= 3/5 (the 7.13 bar, for a direct comparison).
      * IMPROVEMENT BAR: fix_a_mean > fix_a_bal (0.310) AND > fix_a_nr
        (0.495) — mean-Q + random-inclusion must beat both prior no-random
        variants on the same data. ctl_mix vs ctl_bal/ctl_nr attributes the
        random-inclusion effect for the single-critic baseline.
      * Reported: per-seed EM margins + win counts, final-epoch, corr.
    - RESULTS (5 seeds x 3k; verified checkpoint configs: balanced=True,
      exclude=(), fix_a_mean has use_double_q=True + double_q_mode=mean;
      artifacts checkpoints/tacr/{ctl_mix,fix_a_mean}/s{seed}/):
        ctl_mix (single critic, bal+random):  TEST 1.101/0.608/0.721/0.177/0.941
          mean 0.709 +- 0.35 | margins +0.00/+0.33/+0.00/-0.16/+0.12 -> 2/5,
          mean +0.055 | final 0.36 | corr 0.10. Best new-data single-critic
          config; seed 20260814 test 1.10 is the best single-seed TACR run.
        fix_a_mean (double-Q MEAN, bal+random): TEST 0.965/0.309/0.697/-0.215/0.764
          mean 0.504 +- 0.47 | margins 2/5, mean -0.088 | final -0.02 |
          corr 0.24.
    - PRE-REGISTERED BAR OUTCOMES: RECOVERY FAIL for fix_a_mean (0.504 <
      0.70, margins 2/5 < 3/5) — mean-Q + random-inclusion does NOT recover
      the old-data 0.804. IMPROVEMENT PASS vs the prior no-random variants
      (0.504 > fix_a_bal 0.310 and > fix_a_nr 0.495) — the change helps but
      lands far short of recovery. ctl_mix (0.709) beats fix_a_mean (0.504):
      on the balanced+random data the SINGLE critic is stronger than
      double-Q-mean.
    - READING: four consecutive nulls on the new data (7.12, 7.13, 7.14).
      The best new-data TACR variant (ctl_mix 0.709, margins +0.055 mean)
      is still below B (0.99) and fails the >=3/5 EM-margin bar. Every
      sampler / double-Q aggregation variant lands in 0.31-0.71 test Sharpe
      with margins near zero — the TACR objective on this data is the
      limiter, not the BC-prior aggregation or the Q-aggregation mode.
      fix_a_mean's corr 0.24 (vs 0.10 for ctl_mix/ctl_bal) hints mean-Q
      gives slightly more coherent policies, but not enough to matter.

------------------------------------------------------------------------------
7.15 NR7 BEHAVIOR POLICY — SPECIALIZED SIGNAL FAMILY (2026-08-26)
------------------------------------------------------------------------------
    - CONTEXT: parallel-agent TACR trials (7.10-7.14) on the window-expanded
      dataset (31 policies). Per user instruction: add an NR7 policy as a
      SPECIALIZED SIGNAL FAMILY (not window-expanded) and test it with the
      double-Q fix and the ORIGINAL UNIFORM sampler (not family-balanced,
      which the user flagged as broken) — hypothesis: natural
      diversification via a 4th distinct strategy family stabilizes the BC
      prior better than artificial re-weighting.
    - NR7 POLICY (src/data/behavior_policies.py family "nr7"): classic
      narrow-range breakout, daily-close approximation. Day d is NR7 iff
      its high-low range is the narrowest of the 7 trading days ending at
      d (ties allowed). At decision date t: t-1 was NR7 and close[t] >
      high[t-1] -> a_t = +1; close[t] < low[t-1] -> -1; inside the range
      or no signal -> 0. Clean Long/Short/Flat vector; causal (only data
      <= t); warm-up 8 rows; tagged policy='nr7' (own family root).
      Config: configs/data.yaml policies.nr7. Dataset regenerated from
      the CACHED daily frame (scripts/regen_dataset_nr7.py): all 31
      existing policies verified BIT-IDENTICAL, nr7 appended (5,283
      transitions: 314 long / 242 short / 4,727 flat = 10.5% active).
      Pre-NR7 backup: E:\Temp\opencode\offline_dataset_pre_nr7.parquet.
      Dataset-version boundary: trainings launched after ~2026-08-25
      22:30 see 32 policies.
    - TESTS (tests/test_offline_dataset.py; suite 78 green): nr7 actions
      in {-1,0,+1}, both sides fire, mostly flat; signal matches an
      INDEPENDENTLY recomputed NR7 rule action-by-action; reward = a *
      ret_{t+1} with flat days exactly 0; transition count
      cross-validated incl. nr7's warm-up.
    - SUITE 1 — fix_a_mean_nr7 (dq MEAN + uniform, 32 policies;
      scripts/nr7_tacr_run.py, eval scripts/nr7_eval.py): FAILED — the
      worst TACR variant measured: 0/5 vs EM, mean margin -0.41, Sharpe
      +0.22 +- 0.55. Seed 1 DEGENERATE (short_frac 0.999, margin -1.45);
      seeds 2/4 long-only ties; final-epoch check collapses 3/5; basin
      flags all 5, cross-seed corr 0.12-0.29 (no pack).
    - SUITE 2 — fix_a_nr7 (dq MIN + uniform + nr7; the de-confounding
      arm, scripts/nr7_min_run.py): STABLE BUT MARGIN-NULL: Sharpe
      +0.82 +- 0.15 (the LOWEST seed variance of any TACR pack) yet
      margins ~ 0 (mean -0.03, 1/5; seeds 20260814/1 are long-only ties
      by the knife-edge, margin exactly 0); final-epoch 3/5 positive.
      NR7 + uniform + dq-min converges to a stable near-long-only policy
      — stability WITHOUT directional skill (the Model-D profile from
      7.9.2).
    - COMPARISON MATRIX (5-seed packs; canonical roll via
      scripts/tacr_tag_margins.py — matches the parallel agent's own pack
      numbers exactly on the new-data tags (ctl_mix 0.709, fix_a_mean
      0.504), validating the path. OLD-DATA tags (trained on the
      pre-7.11 4-policy dataset: ctl, fix_a, fix_b, fix_c) are hybrids
      when rolled on the new calendar — their genuine numbers are the
      parallel agent's packs (fix_a: 4/5, margins +0.105, mean 0.804 =
      "the old-data 0.804"). NEW-DATA rows are like-for-like:

        tag              objective  sampler  policies  wins  margin   Sharpe
        ctl_mix          plain      balanced 31        2/5   +0.055   0.71 +- 0.35
        fix_a_nr7        dq-MIN     uniform  32 +nr7   1/5   -0.030   0.82 +- 0.15
        fix_a_mean       dq-mean    balanced 31        2/5   -0.088   0.50 +- 0.47
        ctl_nr           plain      uniform  30 -rand  2/5   -0.127   0.59 +- 0.40
        fix_a_nr         dq-MIN     uniform  30 -rand  1/5   -0.183   0.50 +- 0.55
        ctl_bal          plain      balanced 31        1/5   -0.215   0.46 +- 0.48
        fix_a_bal        dq-MIN     balanced 31        1/5   -0.275   0.31 +- 0.59
        fix_a_mean_nr7   dq-mean    uniform  32 +nr7   0/5   -0.408   0.22 +- 0.55
        fix_b (old data) BCQ        -        31*       1/5   -0.455   0.23 +- 0.59

      READINGS: (1) The user's diversification hypothesis is NOT
      supported in margin terms: NR7 + uniform produces the flattest
      margins of the new-data arms (stability, not skill); the best
      new-data margins belong to the BALANCED single-critic ctl_mix
      (+0.055). (2) It IS supported in variance terms: fix_a_nr7's seed
      spread (0.15) is 2-4x tighter than every other new-data pack —
      but per 7.9.2 low variance + ~0 margins is the knife-edge
      signature, not evidence of a learned edge. (3) dq-mean + uniform +
      nr7 is a catastrophic combination (degenerate always-short seed);
      dq-min is the safer aggregation in every pairing. (4) No TACR
      configuration on the new data clears the >= 3/5 EM-margin bar —
      consistent with 7.14's conclusion that the objective, not the
      BC-prior aggregation, is the limiter. NR7 does not change that
      conclusion.
    - PROTOCOL NOTES: per-seed EM margins reported alongside every win
      count (7.9.2 rule). fix_a_nr7's two zero-margins are long-only
      ties (short_frac = 0 -> policy == own EM), the artifact documented
      in 7.9.1 CHECK 2. CRISIS (15 test days): suite-1 seed 20260814
      crisis Sharpe 3.97 — never a headline.
    - ARTIFACTS: scripts/regen_dataset_nr7.py (deterministic regen +
      bit-identity check), scripts/nr7_tacr_run.py, scripts/nr7_min_run.py,
      scripts/nr7_eval.py (full eval incl. margins), scripts/
      tacr_tag_margins.py (matrix); checkpoints/tacr/fix_a_mean_nr7/ and
      fix_a_nr7/ (margins.csv, regime_eval.csv, basin_screening.csv,
      final_epoch_check.csv where applicable).

------------------------------------------------------------------------------
7.16 RTG-RELAXATION TEST (2026-08-26) — user proposal "Relax the RTG constraint"
------------------------------------------------------------------------------
    - IDEA (user): the eval feeds a constant rtg_target of 0.0. Feed a
      POSITIVE target (e.g., the 90th percentile of training returns) so the
      DT-style actor conditions on a high-return context and "stitches"
      better trajectories instead of defaulting to mean behavior.
    - FEASIBILITY: no retraining needed — rtg_target is a roll-time constant;
      the model was trained on RTG values in [-0.66, 2.39], and the channel
      is read (test_eval_roll_immune confirms a different constant changes
      actions). Pooled TRAIN RTG percentiles on the CURRENT data (incl. the
      nr7 policy, which barely shifts them): p50 0.136, p75 0.356, p90 0.669.
    - PRE-REGISTERED PROTOCOL (bars before running): re-roll the best
      available checkpoints — ctl_mix (best new-data margin config), fix_a_mean,
      and the parallel agent's fix_a_nr7 (stable 0.82, the highest-Sharpe TACR
      pack) — with rtg_target in {0.0 (baseline), 0.136, 0.356, 0.669}, 5 seeds
      each, report per-seed test Sharpe + EM margins.
      STITCHING HYPOTHESIS SUPPORTED iff some positive target raises a pack's
      test-Sharpe mean above its 0.0 baseline AND raises (or matches) its
      margin count. Also record the best single (tag, target).
    - RESULTS (rolls of existing best checkpoints; no retraining):
        ctl_mix:     rtg 0.0 -> 0.709/2-5 (+0.055); 0.136 -> 0.734/2-5
                     (+0.082); 0.356 -> 0.661; 0.669 -> 0.561 (hurt).
        fix_a_mean:  rtg 0.0 -> 0.504/2-5 (-0.088); 0.136 -> 0.567/3-5;
                     0.356 -> 0.586; 0.669 -> 0.610.
        fix_a_nr7:   rtg 0.0 -> 0.820/1-5; 0.136 -> 0.841; 0.356 -> 0.867/2-5
                     (best test Sharpe of ANY TACR run, margins still ~0).
    - PRE-REGISTERED BAR OUTCOMES: STITCHING HYPOTHESIS MILDLY SUPPORTED.
      A moderately POSITIVE target (p50-p75 of train RTG, NOT p90) raises
      pack test Sharpe on ALL THREE configs (ctl_mix +0.024, fix_a_mean
      +0.063, fix_a_nr7 +0.047) while a very high target (p90=0.669) hurts
      (the model extrapolates to rare high-RTG contexts -> worse rolls).
      fix_a_mean @ rtg=0.136 reaches 3/5 EM-margin wins — the FIRST TACR
      config to clear 3/5 on ANY new-data variant — though its margin MEAN
      is still slightly negative (driven by 2 bad seeds).
    - NET: RTG relaxation is a real but SMALL lever: +0.02..+0.06 Sharpe
      across the board, best result fix_a_nr7 0.867 (margins ~0) and
      fix_a_mean 3/5 margin count. It does NOT break the ceiling in margin
      terms; the near-zero-margin profile persists where it matters. The
      optimal target is ~the training-median RTG, not the 90th percentile.

------------------------------------------------------------------------------
7.17 STRATEGY 3 — HYBRID: TACR MAGNITUDE x LINEAR SIGN (2026-08-26)
------------------------------------------------------------------------------
    - RATIONALE (user): TACR solves leverage timing (|a_t|, matches/exceeds
      EM) but fails directional sign (margins ~0). Replace the Transformer's
      sign with a simple linear sign model on the same 8 z-features, keeping
      TACR's magnitude. Uses fix_a_nr7 (the tightest-variance TACR pack) for
      |a|. Logistic regression is data-efficient and may capture weak linear
      directional signal the attention landscape ignores.
    - IMPLEMENTATION: scripts/hybrid_sign.py — L2 LogisticRegression
      (sklearn) on the 8 normalized states -> sign(r_{t+1}), trained on the
      TRAIN split only (<=2018-12-31); the L2 strength C is selected on the
      VAL split (2019-2020) from {0.01, 0.1, 1, 10}; TEST is never touched
      for selection. Composite: a_t = sign_linear(s_t) * |a_TACR(s_t)| from
      the fix_a_nr7 per-seed checkpoint. EM control = |a_TACR| * m (the
      hybrid's own exposure) -> margin = Sharpe(hybrid) - Sharpe(|a|*m)
      isolates the sign contribution.
    - PRE-REGISTERED SUCCESS BAR (user-specified, fixed before running):
      margin > 0 in >= 3/5 seeds AND pack mean test Sharpe > 0.99 (beats
      Model B's pack 0.99). Also report per-seed margins, val/test sign
      accuracy (floor: ~0.535 always-+1 baseline; the margin is the test),
      and the rtg_target=0.356 variant (the best RTG target from 7.16) as a
      secondary. 5-seed pack = fix_a_nr7's five seed checkpoints.
    - RESULTS (scripts/hybrid_sign.py; L2 logistic, C=0.1 chosen on val;
      artifacts checkpoints/tacr/hybrid_sign.csv):
        rtg 0.0   : margins 5/5 (mean +0.21) | pack mean test 1.0617 +- 0.16
        rtg 0.356 : margins 5/5 (mean +0.21) | pack mean test 1.0805 +- 0.19
        PRE-REGISTERED SUCCESS BAR: PASS on both (>=3/5 margins AND
        mean > 0.99). The FIRST configuration in the project to clear the
        margin bar 5/5, and it beats Model B's pack mean (0.99) on Sharpe.
    - ROBUSTNESS: C-robust (margin +0.196/+0.230/+0.194/+0.194 across
      C in {0.01,0.1,1,10}); sign model trained train-only, C on val, test
      untouched; test sign accuracy 0.538 vs 0.537 always-long baseline.
    - MECHANISM (decomposed, NOT a per-day-accuracy artifact): the model
      shorts 55/1005 days (5.5%). On those days TACR's OWN sign was LONG on
      ALL 55 (frac short 0.000) and longing them earned -0.038 cumulative
      (actual return sum -0.070: the shorts cluster on MAGNITUDE-weighted
      down days — worst returns -0.04/-0.025 vs best +0.028 — so day-count
      accuracy 50.9% is near coin-flip but the return-weighted accuracy is
      strongly down). Flipping those 55 days turns -0.038 into +0.038
      (swing +0.077), the entire +0.21 margin. mean|a_TACR| is comparable
      on long/short days (0.56/0.48), so it is NOT a leverage artifact.
    - NET: Strategy 3 WORKS and passes the bar. It is a COMPOSITE (TACR
      sizing + linear sign) — the transformer still cannot produce
      directional sign, but the hybrid extracts the value TACR's sign head
      leaves on the table: TACR was long on every day the linear model
      shorts. Caveats to carry: (1) the margin is TAIL-CONCENTRATED in
      5.5% of days (magnitude-weighted down days), not a broad daily edge;
      (2) the sign edge is ~54% day-accuracy, concentrated in return-weight;
      (3) this is not "TACR beats EM" — it is "TACR sizing + linear sign
      beats EM", the user's exact proposal. The 7.16 rtg=0.356 target
      stacks additively (1.08).

------------------------------------------------------------------------------
7.18 STRATEGY 1 — MACRO / CROSS-ASSET FEATURES (2026-08-26)
------------------------------------------------------------------------------
    - DATA (scripts/fetch_macro.py -> data/macro/*.parquet + manifest.json):
      Yahoo public chart API (keyless, period1/period2 daily), 8 series:
      tlt (TLT 20y+), tnx (^TNX 10y yield), vix (^VIX), vix3m (^VIX3M,
      2006-07+), dxy (DX-Y.NYB ICE dollar), hyg (HYG, 2007-04+), qqq, iwm.
      Full 2004/2006/2007 -> 2026-08 daily coverage, adjusted close kept.
    - FEATURES (src/data/macro_factors.py, 8 causal, aligned to the SPY
      calendar on naive dates): risk_on_1d (SPY-TLT ret), tnx_delta_1d/5d,
      vol_term (VIX3M-VIX), dxy_corr_20d, credit_1d (HYG-TLT ret; PRICE
      proxy for HYG_yield-TLT_yield — deviation flagged), rs_qqq_1d,
      rs_iwm_1d. All trailing/causal, z-scored causally like the 8 SPY
      features. Full set available from 2007-05 (HYG inception).
    - TEST (scripts/hybrid_sign_macro.py; L2 logistic, C on val from
      {0.01,0.1,1,10}, TEST untouched; same fix_a_nr7 magnitudes; margins
      vs the hybrid's own EM):
        model                  C  test_acc  short_frac  margin_mean  wins  Sharpe
        A_full (8 SPY, full tr) 0.1  0.5393   0.0557     +0.2259    5    1.076
        A_match (8 SPY, matched tr 2007+) 0.1 0.5393  0.0597   +0.1788  5  1.029
        B (8 SPY + 8 macro, matched)  1.0  0.5393   0.0637     +0.2962   5  1.146
      Per-seed B margins: +0.365/+0.313/+0.314/+0.376/+0.113 — improves on
      A_match on ALL 5 seeds (+0.07..+0.18 each); Sharpe 1.037..1.334.
    - READING: the macro set is genuinely active and HELPS the linear sign
      head (margin +0.1788 -> +0.2962, ~+66% relative, on identical train
      dates; C moved 0.1 -> 1.0, the richer input wants less shrinkage).
      Mechanism consistent with 7.17: test DAY-accuracy is unchanged (0.539)
      — the macro improves the RETURN-WEIGHTED tail selection (short_frac
      6.0% -> 6.4%), not broad daily direction. The user's Strategy-1
      hypothesis is SUPPORTED for the sign model.
    - CAVEATS / FORKS:
      (1) This augments the SIGN MODEL only. Adding macro to the TACR/B/D
          STATE (8 -> 16 dims) is a PROTOCOL-BREAKING change (invalidates
          the four-model comparison + every checkpoint); a separate decision
          the user must make explicitly before any state-dim expansion.
      (2) The credit feature is a price proxy (no yield series).
      (3) Data integrity: fetched live from Yahoo; stored parquet + manifest
          make the pipeline reproducible from disk (re-fetch only extends
          history). VIX3M ends 2026-07-17 (minor tail gap, outside the
          study horizon).
    - NET: Strategy 1 is no longer data-blocked; the hybrid is now
      [TACR magnitude] x [logistic on 16 causal features] = pack mean
      Sharpe 1.146, margins 5/5 (+0.296), the best configuration in the
      project.
    - CANONICAL MODEL C+ (user decision, 2026-08-26): the hybrid is the
      canonical Model C+ — TACR (fix_a_nr7, dq-min/uniform/nr7) provides
      |a|, a logistic on the 8 SPY + 8 macro causal z-features provides the
      sign. Protocol-compliant (state dim untouched; the four-model
      comparison intact); the macro features live ONLY in the linear head.
      Artifacts: data/macro/, src/data/macro_factors.py,
      scripts/{fetch_macro,hybrid_sign,hybrid_sign_macro}.py. The 16-dim
      TACR-state expansion is DEFERRED to a separate pre-registered
      experiment with its own protocol/checkpoints/bars.

------------------------------------------------------------------------------
7.19 ROBUSTNESS — SHIFTED SPLIT FOR MODEL C+ (2026-08-26)
------------------------------------------------------------------------------
    - QUESTION (user): is the macro edge specific to the 2021-2024 test
      regime? Shift the train/val/test split by 1 year and re-test.
    - DESIGN: sign models A_match (8 SPY) vs B (8 SPY + 8 macro) are
      re-fitted on each shifted split (C on that shift's val from
      {0.01,0.1,1,10}, test untouched); the fix_a_nr7 magnitude is held
      fixed (its |a| function is state-dependent, not split-dependent; the
      margin comparison B-vs-A_match uses the IDENTICAL magnitude on each
      period, so the macro delta isolates the sign contribution — which is
      the claim under test). Two shifts:
        back1: train <= 2017-12-31, val 2018-2019, test 2020-2023
               (incl. COVID crash + 2022 bear)
        fwd1 : train <= 2019-12-31, val 2020-2021, test 2022-2024
               (incl. 2022 bear, excludes 2021)
    - PRE-REGISTERED ROBUSTNESS BAR: the macro edge GENERALIZES iff on
      BOTH shifted splits B's pack-mean margin > A_match's AND B's
      margin stays positive (wins >= 4/5). The size of the macro delta
      (B - A_match margin) is reported per shift, not a bar.
    - RESULTS (scripts/hybrid_robustness.py; A_match vs B re-fitted per
      shift, C on that shift's val; fix_a_nr7 magnitude held fixed):
        shift  model    C    test_n  acc   short  margin   wins  Sharpe   macro_delta
        orig   A_match 0.1   1005  0.539 0.060  +0.179   5/5   1.029
        orig   B       1.0   1005  0.539 0.064  +0.296   5/5   1.146    +0.117
        back1  A_match 0.01  1258  0.540 0.065  +0.019   3/5   0.595
        back1  B       0.1   1258  0.548 0.087  +0.315   4/5   0.890    +0.295
        fwd1   A_match 0.01   753  0.529 0.033  +0.358   5/5   1.131
        fwd1   B       0.01   753  0.527 0.040  +0.316   4/5   1.089    -0.042
      (back1 = test 2020-2023, incl. COVID crash + 2022 bear; fwd1 = test
      2022-2024.)
    - PRE-REGISTERED BAR OUTCOME: NOT MET STRICTLY — the "B > A on BOTH
      shifted splits" leg fails on fwd1 (delta -0.042). But the picture is
      more informative than a pass/fail:
      (1) REGIME-CONTINGENT, NOT REGIME-SPECIFIC: the macro edge is LARGEST
          exactly where the SPY-only sign degrades — on back1 the SPY-only
          margin nearly collapses (+0.019, 0.595 Sharpe) while B holds
          (+0.315, 0.890) — a +0.295 macro delta in the crisis-heavy
          window. This directly supports the premise (macro = regime
          detection) even though the strict "helps on every shifted window"
          bar fails.
      (2) On fwd1 the SPY-only features already extract the edge (+0.358);
          macro adds nothing (-0.042) but stays strongly positive
          (+0.316, 4/5) — never materially harmful.
      (3) B's margin is positive in >= 4/5 seeds on BOTH shifts (the
          second leg of the bar, met).
    - NET: the macro edge GENERALIZES as a regime-contingent asset — it
      protects the crisis/crash windows where the price-only signal fails,
      at negligible cost in calm windows. Model C+ stands: [TACR |a|] x
      [logistic, 16 features] is robust and most valuable under regime
      stress. The strict bar failed only on the "monotone improvement in a
      window where SPY-only already wins" leg; the directional value is
      confirmed.
    - FRED OAS UPGRADE (user's optional next step): BAMLH0A0HYM2 etc. are
      NOT reachable from this environment (FRED fetches timed out earlier,
      while Yahoo's API is reachable) — the price-based credit proxy stands
      unless the data is provided offline.

------------------------------------------------------------------------------
7.20 TREND-DAY FILTER FOR MODEL C+ (2026-08-26) — "Regime-first hybrid"
------------------------------------------------------------------------------
    - IDEA (user, citing Azizi 2026, JRFM 19(4) 262 — "Distinguishing Market
      Trends from Oscillations in ETFs"): instead of forcing a sign bet on
      every day, FIRST classify each day as TREND (|r| > tau) vs OSCILLATION
      and force FLAT (a=0) on predicted-oscillation days. Composite:
          a_t = I(trend_pred=1) * sign_model(s_t) * |a_TACR(s_t)|
      Rationale: the C+ margin is tail-concentrated (~5% of days); filtering
      the coin-flip days should raise Sharpe by removing zero-mean noise
      while keeping the tail days.
    - IMPLEMENTATION (scripts/hybrid_trend_filter.py): a second L2 logistic
      on the SAME 16 features predicts Is_Trend = I(|r_{t+1}| > tau). The
      sign model is the C+ B model unchanged. The EM control is matched to
      the deployed exposure (I(trend)*|a_TACR|*m), so the margin isolates
      the sign value on the days the filter keeps.
    - PRE-REGISTERED PROTOCOL (before running): select (tau, C_trend) on the
      VAL split by mean 5-seed val composite Sharpe from tau in
      {0.003,0.004,0.005,0.0075,0.010} x C in {0.01,0.1,1,10}, filter prob
      cutoff 0.5; TEST untouched. Then evaluate on test, 5 seeds.
    - PRE-REGISTERED BAR (user's expected impact): filtered pack-mean test
      Sharpe > C+ unfiltered (1.146) AND margin > 0 in >= 3/5 seeds. Report
      also the trend-day rate on test and how much of the C+ margin is
      retained inside the filtered days.
    - RESULTS (scripts/hybrid_trend_filter.py; tau/C_trend selected on VAL,
      test untouched):
        selected tau=0.003, C_trend=1.0 | trend-day rate on test 0.911
        filtered pack mean test Sharpe 1.2806 +- 0.13 (vs unfiltered 1.146)
        | margins +0.3045, wins 5/5 -> PRE-REGISTERED BAR PASS (formally).
      SENSITIVITY (tau fixed, test; the bar rests on the mildest filter):
        tau     trade-rate  margin  Sharpe  wins
        0.0000  1.000       +0.296  1.146   5/5   (unfiltered C+)
        0.0030  0.911       +0.305  1.281   5/5   (val-selected)
        0.0050  0.459       +0.287  0.777   5/5   (user's suggested 0.5%)
        0.0075  0.224       -0.005  0.752   2/5
        0.0100  0.088       -0.011  0.632   0/5
    - SCOPE CAVEAT (added 2026-08-26, on closer reading of the cited paper —
      Azizi, JRFM 19(4) 262): this test is an ADAPTATION of the paper's
      framing, not a reproduction, on two axes that weaken any claim about
      "the paper's premise":
      (a) THRESHOLD VARIABLE: the paper defines oscillation/trend on the
          INTRADAY HIGH-LOW RANGE of a session; this test used the
          CLOSE-TO-CLOSE return |r_{t+1}|. NOTE (corrected 2026-08-26): the
          daily frame DOES carry high/low (features_regimes.parquet has
          open/high/low/close), so the session's intraday range
          (daily_high - daily_low) IS constructible — the raw 1-minute
          bars also carry per-minute high/low. The close-to-close proxy was
          an IMPLEMENTATION CHOICE, not a data limitation; the paper's
          threshold variable could be built and tested as a follow-up. A
          big intraday swing that closes flat is "trend" under the paper's
          definition but "oscillation" under the close-to-close proxy.
      (b) CLASSIFIER + FEATURES: the paper uses Random Forest / Neural
          Network classifiers with macro-announcement indicators and
          VIX/RSI/ATR as trend-detection features; this test used an L2
          logistic on the SIGN MODEL's own 16 features (8 SPY + 8 macro),
          with no ATR or announcement channel.
      So the correct claim is narrower than "the paper's premise is
      inverted": a RETURN-BASED, SAME-FEATURES LOGISTIC PROXY for the
      paper's trend/oscillation framing does not transfer to the C+ sign
      model on SPY.
    - READING (scoped to the proxy, not the paper's method): the trend
      filter only helps at the MILDEst setting (remove 9% of days). The
      user's suggested 0.5% threshold DESTROYS the margin (0.78 Sharpe,
      from 1.15). Mechanism check (tau=0.005): the ~54% predicted-
      "oscillation" days that the aggressive filter REMOVES have sign
      accuracy 0.544 (ABOVE the 0.539 overall) and NET-POSITIVE composite
      P&L (+0.165, seed 20260814) — i.e. the sign model's edge lives in
      the SMALL-move days, not the big-move days. Filtering to the big-move
      days removes the profitable days. (Small moves are predictably
      mean-reverting/trending; big moves are news-driven idiosyncratic
      noise — consistent with the proxy's failure, NOT evidence against
      the paper's intraday-range method, which was not tested.)
    - NET: the pre-registered bar passed formally (val-selection found the
      trivial 9% filter), but the trend/oscillation regime-first hypothesis
      is NOT supported by this proxy: aggressive filtering is strongly
      harmful, and the small +0.13 Sharpe at tau=0.003 is a mild removal of
      the smallest-move days, not a regime effect. Whether the paper's
      ACTUAL method (intraday-range label, RF/NN + ATR/announcement
      features) transfers is untested here and would need those inputs.
      The robust improvement to C+ remains the MACRO features (7.18), not
      the trend filter. A Focal-Loss retrain of the sign classifier (the
      user's secondary suggestion) is untested and could be a follow-up;
      the return-based filter direction is closed.

------------------------------------------------------------------------------
7.21 FOCAL-LOSS SIGN CLASSIFIER FOR MODEL C+ (2026-08-26)
------------------------------------------------------------------------------
    - IDEA (user): retrain the sign classifier with FOCAL LOSS (Lin et al.
      2017) — down-weight "easy" days where the model is already confident
      and force focus on the hard, ambiguous days near the decision
      boundary, where the current 53.9%-accuracy sign model likely fails.
    - IMPLEMENTATION: a numpy + scipy L-BFGS focal logistic (γ=0 collapses
      to standard cross-entropy — an in-grid sanity check). 16 causal
      features (8 SPY + 8 macro), L2 on the weights, analytic gradient
      verified against a numerical gradient before use.
    - PRE-REGISTERED PROTOCOL (before running): select (gamma, lam) on the
      VAL split by the mean 5-seed val composite Sharpe from
      gamma in {0, 0.5, 1, 2} x lam in {1e-3, 1e-2, 1e-1, 1}; TEST
      untouched. Same magnitude (fix_a_nr7) and margin convention as 7.18.
      BAR (same as C+): pack-mean test Sharpe > 1.146 AND margin > 0 in
>= 3/5 seeds. If gamma=0 is selected, the focal-loss hypothesis is
       not supported on this data (it reduces to standard CE).
    - RESULTS (scripts/focal_sign.py; analytic gradient verified vs numeric
      to 6e-11; gamma=0 focal reproduces the sklearn CE sign model 99.9%
      — implementation sanity confirmed):
        selected gamma=2.0 lam=1.0 on val (best focal val composite Sharpe
        was only 0.067 — every focal config underperformed CE on val)
        test sign accuracy 0.5373 (vs 0.539 CE)
        focal pack mean test Sharpe 0.850 +- 0.20 (vs C+ CE 1.146)
        margins EXACTLY 0.0000 on all 5 seeds (0/5): the gamma>0 model
        collapsed to an always-long classifier (no shorts) — margin 0 by
        the long-only tie construction (7.9.1 CHECK 2), Sharpe = the EM
        level (~0.85).
    - PRE-REGISTERED BAR: FAIL (0.85 < 1.146, 0/5 margins). The focal-loss
      hypothesis is NOT supported for the sign head.
    - MECHANISM: the C+ margin lives in ~5.5% HIGH-CONFIDENCE shorts (the
      tail). Focal loss down-weights confident examples and concentrates on
      the hard boundary days — which are exactly the 50/50-noise days where
      the linear sign model has NO edge. By de-emphasizing the confident
      (and value-carrying) tail, the fit regresses to the up-majority ->
      always-long -> the margin vanishes. The "easy examples" here ARE the
      signal; reweighting against them is destructive.
    - NET: both the regime-filter (7.20) and focal-loss sign retraining
      (7.21) are closed as nulls for Model C+. The robust improvements
      remain the macro features (7.18, margin +0.30, Sharpe 1.146) and the
      mild rtg=0.356 target (7.16). C+ is final: [TACR |a|] x [CE logistic
      on 16 causal features].

------------------------------------------------------------------------------
7.22 SENTIMENT AS THE 17TH C+ FEATURE (2026-08-26)
------------------------------------------------------------------------------
    - FEATURE (user): add a SENTIMENT signal as the 17th feature of the C+
      sign model. Source: CBOE SKEW index (^SKEW, options tail-risk /
      put-demand sentiment; full daily coverage 2004->2026, fetched into
      data/macro/skew.parquet). The 17th feature = causal z-score of the
      SKEW LEVEL (a second variant, the 1d change, is reported as
      sensitivity, not a bar). SKEW is distinct from the existing VIX-based
      vol_term (it is the skew of the vol surface, i.e. tail probability
      weighting, not the vol level).
    - IMPLEMENTATION: scripts/fetch_macro.py + src/data/macro_factors.py
      (SENTIMENT_FEATURES, column sentiment_skew); scripts/hybrid_sentiment.py
      builds BOTH the 16-feature (reference) and 17-feature matrices in one
      protocol (C on val from {0.01,0.1,1,10}, test untouched, same
      fix_a_nr7 magnitude and margin convention).
    - PRE-REGISTERED BAR (same as C+, directly comparable): 17-feature pack
      mean test Sharpe > 16-feature (1.146) AND margin > 0 in >= 3/5 seeds.
      Also report the 17-vs-16 margin delta and the skew-change sensitivity.
    - RESULTS (scripts/hybrid_sentiment.py; C on val, test untouched, same
      magnitude):
        model               C    acc    short  margin   wins  Sharpe
        ref16 (8+8)         1.0  0.5393 0.0637 +0.2962  5/5   1.1463
        s17   (+SKEW level) 0.01 0.5264 0.0767 +0.0069  3/5   0.8038
        s17c  (+SKEW 1d chg) 0.1 0.5264 0.0753 +0.0739  4/5   0.8273
      (ref16 reproduces the 7.18 C+ exactly — protocol sanity confirmed.)
    - PRE-REGISTERED BAR OUTCOME: FAIL. The 17-feature sign model is WORSE
      than the 16-feature C+ in every metric: Sharpe 0.80 vs 1.146, margin
      +0.007 vs +0.296, day-accuracy 0.526 vs 0.539 (BELOW the always-long
      0.537 baseline — the model is worse than predicting +1 every day).
      The SKEW-change variant (s17c) is less bad (4/5, +0.074) but still
      far below ref16.
    - MECHANISM (consistent): SKEW is a persistent, slowly mean-reverting
      sentiment level. Fed as an extra input to a ~3200-sample linear fit it
      adds capacity without a short-horizon directional signal, and it
      disrupts the tail-short selection that carries the C+ margin (the
      margin collapses 0.296 -> ~0.01 while shorts shift). C for s17 moved
      to 0.01 (max L2 in the grid) and STILL underperformed — not a
      regularization-range artifact.
    - NET: sentiment via CBOE SKEW (level or change) does NOT help Model
      C+; it degrades it. The only options-sentiment series available on
      the reachable source (Yahoo) is SKEW (put/call ratios are not
      exposed, CNN Fear&Greed/AAII are off-source). Model C+ stands at 16
      features: [TACR |a|] x [CE logistic, 8 SPY + 8 macro].

------------------------------------------------------------------------------
7.24 PAPER-METHOD TEST — INTRADAY-RANGE TREND LABELS + RF/NN (2026-08-26)
------------------------------------------------------------------------------
    - MOTIVATION (user): 7.20 tested a close-to-close proxy of the Azizi
      (JRFM 2026) trend/oscillation framing and it was null. The paper's
      ACTUAL threshold variable (session INTRADAY HIGH-LOW range, which the
      daily frame's high/low columns provide — confirmed present) plus its
      classifier families (Random Forest / Neural Network) and trend features
      (VIX, RSI, ATR) were not tested. This closes that gap.
    - SETUP (scripts/hybrid_trend_paper.py):
      * Label (next-day trend): y_trend_t = I(range_{t+1} > tau), range_t =
        (high_t - low_t) / close_t. tau in the paper's {0.5%, 0.75%, 1.0%}.
      * Classifiers: RandomForestClassifier (n_estimators=200, max_depth=8,
        min_samples_leaf=20) and MLPClassifier (32 hidden, early stopping) —
        ONE reasonable un-tuned config per family, selected together with tau
        on VAL by mean 5-seed val composite Sharpe (test untouched).
      * Features (causal, z-scored): PAPER = [VIX level, RSI, ATR(14),
        current-day range]; FULL = the 16 C+ features + ATR(14) + range.
      * DEVIATION FLAGGED: the paper's macro-ANNOUNCEMENT indicators are not
        available (no announcement calendar in this repo) — "the paper's
        method minus the announcement channel".
      * Composite: a_t = I(trend_pred=1) * sign_logistic(s_t) * |a_TACR(s_t)|,
        same fix_a_nr7 magnitude + margin convention as C+.
    - PRE-REGISTERED BAR (same as C+): filtered pack-mean test Sharpe > 1.146
      AND margin > 0 in >= 3/5 seeds. Also report trend-day rate and
      next-day-range prediction accuracy vs base rate (the key question: is
      next-day intraday-range trend even predictable here?).
    - RESULTS (scripts/hybrid_trend_paper.py; (featset, tau, clf) selected on
      VAL by mean 5-seed val composite Sharpe, test untouched):
      *** SUPERSEDED — the original run used daily high/low that a
      data-integrity audit found CORRUPT (bad ticks in the raw minute bars,
      2005-2013); those numbers are invalid. The data was cleaned and the
      test re-run in §7.25 (corrected result below). ***
      CORRECTED (clean data): selected PAPER featset (VIX/RSI/ATR/range),
      tau=0.005, MLP | next-day range base rate 0.945 (trend = majority at
      0.5%) | classifier acc 0.939 (~ base — no predictive power) | trade
      rate 0.994 (barely filters) | filtered pack Sharpe 1.146 +- 0.11 (==
      C+), margins +0.308, 5/5.
      [historical, corrupted data — DO NOT CITE]: selected FULL, tau=0.0075,
      MLP | base 0.789 | acc 0.796 | trade 0.897 | Sharpe 1.127 | +0.303.
    - PRE-REGISTERED BAR OUTCOME (corrected, clean data): FAIL on the Sharpe
      leg (filtered 1.146 == C+ 1.146, NOT >; margins 5/5 pass). The paper's
      ACTUAL framing does not beat C+ either.
    - MECHANISM (the reason, now direct, on clean data): next-day
      intraday-range trend-ness is UNPREDICTABLE from the available features
      (classifier acc 0.939 ~= the 0.945 always-trend base rate). At the
      paper's thresholds (0.5-1.0% of close) a trend day is the MAJORITY
      class on SPY (94.5% of days at 0.5% — SPY's daily range is ~1%), so
      the filter trades ~99% of days and degenerates to (near-)unfiltered
      C+. The paper's premise (trend vs oscillation is classifiable) does
      not transfer because (a) the label is a majority class on this asset
      and (b) the features carry no next-day-range signal.
    - DEVIATION (flagged): macro-announcement indicators are in the paper's
      feature set but unavailable here (no announcement calendar) — "the
      paper's method minus the announcement channel". RF and MLP were both
      in the selection grid; MLP won on val. VIX/RSI/ATR/range (PAPER
      featset) and the 16+ATR+range (FULL) were both in the grid; FULL won.
    - NET: the paper's intraday-range trend/oscillation method, tested
      faithfully (minus announcements), does NOT improve C+ as a filter —
      the label is a majority class and is not predictable. This closes the
      7.20 follow-up: the null was NOT a close-to-close proxy artifact; the
      actual method is also null. C+ remains canonical (1.146, +0.296, 5/5).

------------------------------------------------------------------------------
7.25 DATA-INTEGRITY FINDING — CORRUPT HIGH/LOW (2026-08-27)
------------------------------------------------------------------------------
    - DISCOVERY (user, during review of hybrid_trend_paper.py): the daily
      frame's HIGH/LOW columns contain corrupt values — 41 daily rows with a
      high > 1.5x close (e.g. 2009-06-25 high 100,000.00 vs close 91.95;
      2005-07-12 high 23,200 vs close 122.23; 2013-09-06 high 16,166) or a
      low < 0.5x close (e.g. 2007-09-11 low 1.86 vs close 147.36;
      2009-09-21 low 8.23). `close` is CLEAN (all prior audits — anchors,
      regimes, momentum — checked close, never high/low; this is why it
      went undetected).
    - ROOT CAUSE (traced to source): the corruption is in the RAW minute
      bars (SPY_YYYY.parquet), not the resampler — 68 corrupt minute ticks
      across 2005-2013 (23/15/15/5/5/1/2/2 by year): a single bar whose high
      or low (or open) is a glitch print (e.g. high 99,999.99 on 2009-06-25;
      open 58.74 / close 129.81 on 2006-02-27). The daily resample's max/min
      faithfully propagates each bad tick into the daily high/low. The
      minute-level structural check (high >= low, positive) passes because
      the bad values keep OHLC structure internally consistent.
    - FIX (src/data/loaders.py):
      * `_drop_corrupt_minutes`: a minute bar is dropped (loud stderr
        warning with count + dates) when its high/low is implausible vs its
        OWN open/close (>25% intra-minute excursion), or its open-to-close
        move >25% of the smaller. Removal of a demonstrably corrupt
        observation, not synthetic substitution. 68 bars dropped, 2005-2013.
      * `_validate_daily_ranges`: a FAIL-LOUD guard on DAILY frames (high >
        1.5x close OR low < 0.5x close -> raise), wired into the resample
        path AND the cache-load path, so surviving corruption is never
        silently used. The same philosophy as the Phase-1 hole guard.
      * `_CLEANING_VERSION` in the cache manifest digest: a rule change
        forces a daily-cache rebuild (the stale cached daily with corrupt
        high/low is invalidated).
    - IMPACT: features_regimes.parquet / offline_dataset.parquet regenerated
      on the clean daily frame (same row counts, 0 high/low issues). Only the
      §7.24 paper-method test used SPY high/low (range + ATR features); Model
      B/C/D, the macro features and the §7.20 close-to-close trend filter are
      UNAFFECTED. The 7.24 result was invalidated by the corruption and is
      re-run below. Suite 78/78.
    - CORRECTED §7.24 RESULT (clean data): selected PAPER featset (VIX/RSI/
      ATR/range), tau=0.005, MLP | next-day range base rate 0.945 (trend is
      the majority at 0.5%) | classifier acc 0.939 (~ base — no predictive
      power) | trade rate 0.994 (the filter barely filters) | filtered pack
      Sharpe 1.146 +- 0.11 (== C+), margins +0.308, 5/5. The bar still
      FAILS on the Sharpe leg. The 7.24 null STANDS on clean data: the
      paper's framing does not transfer because next-day intraday-range
      trend is a majority class and is unpredictable.

------------------------------------------------------------------------------
7.26 CONFIRMATORY RUN — C+ ON THE UNTOUCHED 2025-2026 WINDOW (2026-08-29)
------------------------------------------------------------------------------
    - PREREGISTERED PROTOCOL (single run, stop after one report): every
      7.10-7.25 experiment was evaluated on the SAME 2021-2024 test window,
      with each next experiment informed by those numbers (a garden-of-
      forking-paths hazard). This run evaluates the FROZEN canonical C+
      artifacts on 2025-01-01..2026-03-31 — a window no selection ever
      touched. Magnitude |a|: the fix_a_nr7 per-seed checkpoints (frozen
      .pt files). Sign: the ref16 CE logistic (8 SPY z + 8 macro causal z,
      C=1.0) refit DETERMINISTICALLY on the frozen train split
      (<=2018-12-31) — sklearn L2-LBFGS is deterministic on fixed data, so
      this reproduces the 7.18 weights exactly; nothing is fit on or touched
      by 2025-2026. `load_tacr_data` gained a backward-compatible
      `test_end` param (default SPLIT_TEST_END). Transformer timesteps past
      the last trained position (4782) are clamped to 4782 — the trajectory
      is frozen at its last trained step (never-seen embedding rows are not
      used). Script: scripts/hybrid_sign_confirmation.py.
    - REPRODUCIBILITY FINDING (surfaced by the user-demanded eyeball, before
      the confirmation run): the 7.18/7.22 B-row numbers are NOT byte-
      reproducible on the post-7.25 regenerated features frame. Current-state
      reproduction of the canonical 2021-2024 window with the frozen artifact:
      acc 0.5403 / short_frac 0.0627 / margin +0.3072 / 5/5 / Sharpe 1.1573,
      vs recorded 0.5393 / 0.0637 / +0.2962 / 1.146. The A rows reproduce
      EXACTLY (0.2259, 0.1788 to 4 decimals), so the 8 SPY z-features, the
      rolls and the 1005-day window are bit-identical; only the 16-dim macro
      head drifted, by ~one short day. Not pinned to a single row with
      surviving evidence (macro parquet + code unchanged; suspicion: a tiny
      macro-feed delta at train dates from the regeneration, or a library
      numerics path). Same qualitative verdict either way — recorded as-is,
      no retroactive edit of the 7.18 table.
    - CONFIRMATION WINDOW (2025-01-01..2026-03-31, 310 feasible days; regimes
      bull 229 / bear 60 / crisis 21), FROZEN artifact, reported ONCE:
        per-seed margin  +0.4389 / +0.3031 / +0.2672 / +0.3100 / +0.0551
        per-seed Sharpe   0.878 /  0.776 /  0.752 /  0.620 /  0.985
        pack mean margin +0.2749 | pack mean Sharpe 0.802 +- 0.14 |
        pack mean EM 0.527 | margin>0 5/5 | short_frac 0.0742 |
        sign_acc 0.5419 | artifact csv: src/models/tacr/checkpoints/tacr/
        hybrid_sign_confirmation.csv
    - VERDICT: CONFIRMATION HOLDS. C+ generalizes to the untouched window
      without retuning — pack margin +0.27 (5/5 positive) vs the frozen-
      artifact canonical +0.31. Absolute Sharpe is lower (0.80 vs 1.16) in a
      window that carries a bear rollout and crisis days, but the margin over
      the |a|*m exposure baseline is positive, seed-consistent, and in line
      with the in-sample level. Pre-registered stop honored: no iteration, no
      tuning on this window. The deferred 16-dim TACR-state expansion (macro
      features feeding the transformer state directly instead of only the
      sign head) is now UNLOCKED as a follow-up — not part of today's run.
    - suite: pytest 78/78 (5 files, post-loader-change).

------------------------------------------------------------------------------
7.27 MODEL D RETRAIN ON THE EXPANDED (32-POLICY) DATASET — THE DATA-SCALE
       EXCUSE, TESTED (2026-08-30)
------------------------------------------------------------------------------
    - MOTIVATION (user): 7.9.1 attributed the D fuzzy-null to DATA SCALE —
      "at this data scale (15.9k transitions, ~3.9k train days) the fixed
      IT2 encoding adds no information the 2-layer input projection cannot
      express" — a scale claim never tested. 7.11 expanded the offline
      dataset exactly for this, but Model D was trained (7.9, 08-21) BEFORE
      7.11 (08-25) and had never been retrained on the expanded data. User's
      premise to test: "IQL doesn't need matched behavior-policy semantics
      the way TACR's BC anchor does — it just wants more (s,a,r,s')
      transitions."
    - PROTOCOL: retrain BOTH D variants x 5 seeds at the pre-registered 3k
      screen budget (10x300, batch 64, seed set identical to 7.9) on the
      CURRENT offline dataset. Note: the dataset is now 32 policies /
      168,516 rows — 7.11's "31 policies / 163,233" plus the 7.15 nr7
      behavior policy (never excluded). Scale vs what 7.9 saw (4 policies /
      21,132 rows): 8.0x rows; 32 x 3272 valid train days = 104,704 train
      transitions = 6.6x the 15.9k. NO code, NO hyperparameter, NO eval
      change. Artifacts: checkpoints/d_32p/.
    - RESULTS (test 2021-12-31 clip, all-days Sharpe, 5 seeds, best-val ckpt):
        VARIANT            mean   std    vs EM   final-epoch check
        D (fuzzy) 32p      0.7106 0.0347 0/5     FAIL (best 0.711/final 0.592)
        D-minus-fuzzy 32p  0.7136 0.0213 1/5     FAIL (best 0.714/final 0.626)
        -- 7.9, 4-policy, for reference --
        D (fuzzy) 4p       0.9310 0.0192 4/5     pass (best 0.931/final 0.894)
        D-minus-fuzzy 4p   0.8807 0.0250 4/5     pass (best 0.881/final 0.882)
        NOTE: the 4p reference above is the 7.9-RECORDED table (pre-7.25
        loader). Same-pipeline recompute of the identical checkpoint files
        reads 0.9042/0.8837 (see 7.27.2 results + note) — the gap is the
        loader-clip reproducibility caveat, not a code change; all
        cross-pack comparisons in 7.27.2 run on the one current pipeline.
        ablation delta 32p 0.003 < bar 0.240 -> NULL (unchanged from 4p 0.050)
    - MECHANISM (why the premise failed): the IQL sampler draws the POLICY
      uniformly per batch. Of 110,276 train transitions (<=2018-12-31), the
      mean-reversion family is 69,260 = 62.8%, with mean_action -0.040 and
      short_frac 0.605 — 20 short-horizon CONTRARIAN policies dominate. The
      AWR-weighted policy re-targets from the 4-policy-era long-biased
      momentum follower (old D-minus-fuzzy short_frac < 2%, |a| ~ 0.29) into
      a contrarian low-|a| mush (32p D-minus-fuzzy short_frac 0.17-0.32, |a|
      0.15-0.20). Val collapses in lockstep (old best-val 1.21 for
      s20260814 -> 0.235). It is a SCALE change coupled to an IDENTITY
      change, not "more of the same data."
    - READINGS (the "data-scale" question, now closed, and the swap decision):
      (1) THE SCALE EXCUSE IS CLOSED for the fuzzy-vs-raw null: at ~6.6-8x
          more transitions the fixed-IT2 ablation is STILL null (delta
          0.003), so "more data would make the fixed IT2 encoding
          informative" is unsupported at this scale too. It died the wrong
          way (both variants regressed together) but it is closed — the
          fixed IT2 channel was not rescued by 8x data.
      (2) CONTRA THE USER PREMISE: IQL is NOT indifferent to the behavior
          mix. The window-expanded families are not interchangeable
          transitions; uniformly sampled, the 62.8% contrarian family
          re-weights the support and REGRESSES IQL sharply (0.88 -> 0.71,
          4/5 -> 1/5). A clean "does IQL improve with more data" test needs
          a mixture-WEIGHTED (or family-limited, e.g. momentum + bh + nr7)
          dataset — a NEW design, not run here (no speculative experiments).
      (3) C+ MAGNITUDE-SWAP DECISION (per the user's decision rule grounded
          on this rerun): REJECT — keep fix_a_nr7. The retrained
          D-minus-fuzzy is not a magnitude candidate: it loses to its EM
          control on 4/5 seeds, regressed structurally (0.88 -> 0.71), and
          now fails the final-epoch check. Even the best AVAILABLE IQL
          magnitude (the pre-7.11 20k D-minus-fuzzy: Sharpe 0.8837, stable,
          final-epoch pass) has |a| ~ 0.29 mean — vs fix_a_nr7's test-window
          |a| 0.56 pack mean (per-seed 0.25-0.84) — a materially smaller and
          flatter exposure shape. The "comparable to 0.82 +- 0.15" premise
          fails on level; nothing here overturns fix_a_nr7 as C+'s magnitude.
    - suite: pytest 78/78 (no code changed this run; retrain + eval only).

------------------------------------------------------------------------------
7.27.1 IQL "STABILITY" CLAIM — REVERSED AT 32-POLICY SCALE, THEN RESCOPED
      (2026-08-30; pulled out of the 7.27 ablation framing deliberately)
------------------------------------------------------------------------------
    - THE CLAIM THAT BROKE: 7.9's selling point for Model D was "the first
      model in the project that trains stably offline" — no collapse at any
      budget, final-epoch check passing everywhere "unlike C where every
      checkpoint failed" (7.9 READINGS (1), CONCLUSION; repeated verbatim in
      the executive summary line 39 as "Phase 4 ... stable"). At the
      32-POLICY, UNWEIGHTED scale (7.27) that claim is FALSE:
        D (fuzzy)      best-val mean 0.711 vs final 0.592  -> diff -0.119 (FAIL)
        D-minus-fuzzy  best-val mean 0.714 vs final 0.626  -> diff -0.087 (FAIL)
      Both variants now show exactly the TACR-shaped failure mode they were
      built to avoid: the best-val checkpoint does not generalize to the
      final optimizer state. This is NOT a footnote to the ablation null —
      the ablation staying null says nothing about stability, and the two
      are independent.
    - WHAT IT MEANS, CORRECTED: "D trains stably" was true at the 4-policy /
      ~16k-transition scale it was measured on and is a SCALE-AND-MIX-SCOPED
      property, not a model property. It correlates with what the behavior
      support looks like, established by 7.27.2 below: at matched
      composition even 6.6x more volume passes the final-epoch check cleanly
      (+0.045 / +0.033), so the stability is NOT broken by volume — it was
      broken by re-weighting the mix toward the contrarian families.
    - Exec summary (line 39) and FINAL VERDICT updated to carry this scope.

------------------------------------------------------------------------------
7.27.2 VOLUME vs COMPOSITION — THE DISCRIMINATOR (2026-08-30)
------------------------------------------------------------------------------
    - CONCERN (user): 7.27 watched TWO things change together (volume 6.6x
      AND the mix shifting to contrarian short-horizon families) and
      performance regress — a correlation, not a driver. The clean test:
      hold TOTAL VOLUME fixed, vary ONLY the family proportions.
    - PROTOCOL (scripts/d_composition_run.py): SAME pool (the 168,516-row /
      32-policy dataset), SAME screen budget (batch 64 x 3000 steps x 5
      seeds x both variants), SAME hyperparameters; ONLY the per-policy
      sampling weights change — re-mapped to the pre-7.11 family proportions
      (buy_and_hold 0.25 / momentum 0.25 / mean_reversion 0.25 / random
      0.25; nr7 excluded — it was not in the old mix; uniform within
      family). Implemented as a sampling-per-policy weight (config field
      DConfig.policy_weights + CLI --policy-weights), zero effect on the
      default uniform behavior.
    - RESULTS (test 2021-12-31, all-days Sharpe, best-val ckpts, n=5;
      reported mean +- seed std, SE = std/sqrt(5)):
        dataset            D (fuzzy)  D-minus-fuzzy  DNF final-epoch  DNF short_frac
        4-policy old (7.9) 0.9042+-0.053  0.8837+-0.042  pass              0-2.4%
        32p UNWEIGHTED     0.7106+-0.039  0.7136+-0.024  FAIL -0.087       17-32%
        32p COMP-MATCHED   0.8423+-0.060  0.8367+-0.038  pass +0.033       0.0% (BEST ckpts;
                                                                          FINAL ckpts take
                                                                          micro-shorts 0.6-1.1%,
                                                                          still stable — 7.27.3)
        ablation delta (comp-matched): 0.006 < bar 0.240 -> STILL NULL
      NOTE on the "4-policy old" row: the same-pipeline recompute (this
      script, current loaders) reads 0.9042/0.8837 off the 7.9 checkpoint
      files (verified: epoch 10x300, batch 64, temp 3.0, fuzzy flags set).
      The 7.9 header table recorded 0.9310/0.8807 under the pre-7.25 loader
      (no 310-date clip) — the D gap is the same reproducibility caveat as
      the A-row-exact / B-row-drift finding, NOT a second experiment. All
      three packs below are compared on the ONE current pipeline.
    - DIFFERENCE TESTS (two independent packs, n=5 each; pooled variance)
      D-minus-fuzzy: old vs comp   gap +0.047, se 0.025, t=1.86, p~0.10 (NS)
      D (fuzzy):     old vs comp   gap +0.062, se 0.036, t=1.73, p~0.12 (NS)
      D-minus-fuzzy: comp vs 32p   gap +0.123, se 0.020, t=6.2  (sig)
      D (fuzzy):     comp vs 32p   gap +0.132, se 0.032, t=4.1  (sig)
      The old-4p pack spans 0.839-0.934 (DNF) / 0.816-0.943 (D); the
      comp-matched pack spans 0.784-0.871 / 0.778-0.933 — the comp mean
      sits INSIDE the old pack's seed spread, so the 0.047-0.062 gap is
      inside the noise floor, not separable at n=5.
    - READOUT — COMPOSITION IS THE DRIVER; VOLUME ALONE IS FINE (WITHIN
      NOISE). Held at 6.6x volume with the old family proportions, IQL
      recovers to within noise of its old level (no significant gap, p~0.10)
      while sitting decisively above the unweighted 32p pack (t=4-6, p<0.005)
      AND passes the final-epoch check (which at unweighted-32p was the
      TACR-shaped FAIL). The regression, the reversal of the stability
      claim, and the confusion of the policy are explained by the mixture
      re-weighting, not by sample count: a bigger-but-mixed dataset breaks
      IQL because the sampler picks policies uniformly and the 20-policy
      contrarian family is 62.8% of the pool, not because IQL cannot use
      more transitions. ("Recovers ~95%" would be a false precision — the
      gap is not separable from seed noise and is stated as such.)
    - BULL-MECHANISM CHECK (Q2 — is the comp-matched recovery the SAME
      long-bull participation engine as the 4-policy run, or did the 32p
      pool change HOW long exposure is taken? same-pipeline read):
        metric                 old 4p DNF      comp-matched DNF
        mean a | bull days     0.305 +- 0.066   0.311 +- 0.096
        long%  | bull days     0.998            1.000
        bull-day Sharpe        1.334 +- 0.044   1.269 +- 0.033
        corr(a, m) all-days    0.0254           0.0291
        mean |a|               0.297 +- 0.068   0.306 +- 0.093
      SAME mechanism: the comp-matched recovery runs on the identical
      long-bull-participation engine (mean long exposure on bull days within
      2%, market correlation 0.025 vs 0.029 — both near zero and inside
      noise, bull Sharpe within noise, |a| actually slightly HIGHER). 7.9's
      own reading of D's edge — "long-bull participation, not from shorting
      skill" — applies verbatim to both.
    - THE 0/5 "TIE" IS NOW EVIDENCED, NOT ASSUMED: old-4p short_frac was
      0.6% (micro-shorts on 4 of 5 seeds) and the per-seed EM margins
      (+0.014/+0.018/+0.002/+0.000/+0.094) came EXCLUSIVELY from those <1%
      of short days; comp-matched went to exactly 0.0% short, so margins are
      exactly 0 in every seed (a > 0 everywhere => a == |a| => Sharpe(a*m) ==
      Sharpe(|a|*m)) and wins = 0/5 is the 7.9.1 CHECK-2 long-only knife
      edge, not a regression. The flip REINFORCES 7.9: the directional edge
      was never the short side (0.6% of days carried it); stripping it
      converts 4/5 margins of +0.01-+0.09 into exact ties at UNCHANGED long
      behavior. The one observed behavior delta (old -> comp) is the
      flattening of that micro-short fringe, at identical long participation.
    - RESIDUAL (honest, not papered over): within-noise gap to the old pack
      plus the flattened short fringe are observed; a small composition
      component (within-family policy variety — 20 mean/horizons vs the old
      single policy — or the shorter 32-policy common date grid) cannot be
      excluded and partly explains the REMAINING non-decisive gap. Claim
      made only at the family-proportion level; the across-family result is
      the decisive leg.
    - CONSEQUENCE for the C+ magnitude swap: still REJECT (unchanged from
      7.27). Comp-matched D-minus-fuzzy |a| = 0.306 mean, long-only — same
      conclusion as 7.27 (3): not comparable to fix_a_nr7's 0.56-pack
      test-window |a| (0.25-0.84/seed) and WITHOUT the short-side timing the
      EM margin needs. fix_a_nr7 stays.
    - suite: pytest 78/78 (3-file change is additive, backward-visible only).

------------------------------------------------------------------------------
7.27.3 SHORT-SIDE MASS IS THE INSTABILITY MECHANISM — THE w_m SWEEP
      (2026-08-30)
------------------------------------------------------------------------------
    - QUESTION (user): 7.27.2 proved composition (not volume) drives the
      IQL regression, but "composition matters" is underspecified. Deliberate
      single-axis sweep: upweight the mean_reversion family's SAMPLING
      weight w_m (while holding total volume fixed) and watch whether
      final-epoch failure returns as the fitted short_frac RISES —
      independent of whether the mix resembles old-4p or 32p-unweighted.
      If failure tracks short_frac, that is the mechanism; more useful than
      "composition matters."
    - PROTOCOL (scripts/d_msw_sweep.py): SAME pool (168,516-row / 32-
      policy), SAME budget (batch 64 x 3000 x 5 seeds x BOTH variants),
      SAME sampler mechanics; ONLY the family weights change along w_m in
      [0.25, 0.40, 0.55, 0.70, 0.85] with bh = momentum = random = (1-w_m)/3
      (nr7 excluded). w_m=0.25 reproduces the comp-matched baseline; 0.625
      is the 32p-unweighted share — the unweighted pack slots onto the
      curve. Readout per (point, variant, seed): test all-days Sharpe of the
      BEST and FINAL checkpoints, final-epoch diff = mean(final)-mean(best),
      short_frac of both, mean |a|. 50 sweep runs + 5-seed anchors.
    - RESULTS (pack means; test 2021-12-31 clip; diff = final - best):
        point (w_m, fam bh/mom/mean/rand)   short_frac(BEST)  short_frac(FINAL)  diff
        w=0.25 (0.25/0.25/0.25/0.25) =d_cmp  0.000            0.011              +0.033 (PASS)
        w=0.40 (0.20/0.20/0.40/0.20)         0.001            0.063              +0.029 (PASS)
        w=0.55 (0.15/0.15/0.55/0.15)         0.056            0.280              -0.062 (FAIL)
        w=0.625 unweighted (d_32p)           0.230            0.357              -0.087 (FAIL)
        w=0.70 (0.10/0.10/0.70/0.10)         0.182            0.536              -0.163 (FAIL)
        w=0.85 (0.05/0.05/0.85/0.05)         0.414            0.607              -0.183 (FAIL)
        (D-minus-fuzzy numbers; D moves in lockstep — flip between w=0.40
        and w=0.55 for both: D diff +0.013 -> -0.091)
      MECHANISM METRICS (n=65, d_cmp duplicate excluded):
        corr(final-epoch diff, short_frac) = -0.878  (short_frac explains
        ~77% of the diff variance)
        corr(final-epoch diff, w_m)         = -0.788
        partial corr(diff | w_m removed, short_frac) = -0.209 -> short_frac
        carries signal BEYOND the monotone w_m trend
        short-frac bins: 0% diff +0.024; 0-3% +0.052; 3-10% +0.021 (ALL
        PASS); 10-25% -0.042; >25% -0.101 (FAIL, worsening)
    - READOUT — THE INSTABILITY IS THE SHORT-SIDE MASS, NOT COMPOSITION
      DISTANCE. Two facts separate the candidates:
      (1) w=0.40 is the MOST composition-distance-far point among the
      PASSING set (bh/momentum/random all cut from 0.25 to 0.20, mean 0.40)
      yet still PASSES with diff +0.013/+0.029 — the pass/fail flip does
      NOT sit at a "distance from old-4p" boundary;
      (2) the flip sits exactly where the FINAL checkpoint's short-taking
      jumps through the 10-30% band: FINAL short_frac goes 0.6% (w=0.25)
      -> 6% (w=0.40, still pass) -> 28% (w=0.55, FAIL). Instability appears
      when the AWR target's fitted support turns strongly short-taking, and
      scales with how far short_frac climbs. old_4p (same family weights as
      comp-matched, short_frac 0.6-3%) is stable at +0.007, consistent with
      the low-short side of the curve.
      MECHANICAL STORY: as the contrarian family's sampling weight crosses
      ~0.5, the advantage-weighted mean action per day tips negative on
      non-bull days, the actor keeps drifting toward short mass at the
      FINAL epochs (BEST-ckpt short_frac stays low — best-val selection
      catches early low-short epochs — while FINAL short_frac jumps 0.006 ->
      0.28), and the optimizer is still moving where it should have
      settled; that is the TACR-shaped failure mode D was built to avoid.
      The w=0.40 point with a stable +0.029 diff proves the mechanism is
      NOT merely "any divergence from the old mix."
    - QUALIFYING NOTE on the comp-matched "0% short" claim (7.27.2): that
      short_frac was the BEST-ckpt read; the FINAL checkpoints of d_cmp /
      msw-0.25 carry micro-shorts (0.6-1.1% of days) and remain stable —
      consistent with this section: short grain below ~10% is harmless (it
      even reads slightly better on test, +0.045/+0.033 diffs).
    - SUITE: pytest 78/78 (analysis-only this section; the sweep reuses the
      7.27.2 DConfig.policy_weights plumbing, unchanged).
    - ACTIONABILITY (noted, NOT run — no speculative designs): the boundary
      being short-side mass suggests a short-suppression lever (e.g.
      censoring short actions from the support, or short-frac regularization)
      would rescue an IQL fit on large mixed datasets; a momentum-axis mirror
      sweep (upweight momentum with mean held low, keeping short_frac ~0 at
      high composition distance) is the clean orthogonal control to pin the
      causality against the residual partial corr. Both are follow-up
      candidates, out of scope tonight.

------------------------------------------------------------------------------
7.27.4 AWR TEMPERATURE β IS NOT THE BINDING CONSTRAINT (2026-08-31)
------------------------------------------------------------------------------
    - IDEA: β in AWR's exp(Q/β) controls the action distribution sharpness.
      The 32p-collapse may arise from a β that is too high (too diffuse)
      or too low (too peaked on a few high-advantage actions). Sweep β over
      {0.5, 1.0, 1.5, 2.0, 3.0} on the 32-policy unweighted dataset to
      test whether β is the binding constraint vs the data composition.
    - PROTOCOL: 5 seeds × 5 β × both variants = 50 runs. Tag d_beta/{β}.
      Default β=3.0 is the paper's AWR advantage temperature. β<3 →
      sharper action distribution; β>3 → more diffuse.
    - PRE-REGISTERED SUCCESS CRITERIA:
        (1) Does any β achieve final-epoch PASS on 32p-unweighted?
        (2) If so, does the corresponding β on comp-matched improve
            Sharpe > 0.837?
        (3) If no β stabilizes 32p, β is not the binding constraint.
    - RESULTS (scripts/d_beta_sweep.py; analysis: scripts/d_beta_analysis.py):

      ┌────────┬────────┬────────┬────────┬──────────┬──────────┐
      │   β    │ shbest │ shfinal│  diff  │ sf_final │ absa_fin │
      ├────────┼────────┼────────┼────────┼──────────┼──────────┤
      │  0.5   │ 0.698  │ 0.624  │ -0.074 │  0.358   │  0.141   │
      │  1.0   │ 0.701  │ 0.623  │ -0.078 │  0.363   │  0.140   │
      │  1.5   │ 0.711  │ 0.612  │ -0.099 │  0.377   │  0.141   │
      │  2.0   │ 0.709  │ 0.622  │ -0.087 │  0.364   │  0.141   │
      │  3.0   │ 0.714  │ 0.626  │ -0.087 │  0.357   │  0.141   │
      │d_32p   │ 0.714  │ 0.626  │ -0.087 │  0.357   │  0.141   │
      └────────┴────────┴────────┴────────┴──────────┴──────────┘
      (D-minus-fuzzy, pack means, 5 seeds. β=3.0 reproduces d_32p exactly.)

    - FINDING: NO β achieves final-epoch PASS. All five β values produce
      consistent negative diff (−0.074 to −0.099). The variation across β
      is small (range 0.025) and does not correlate with outcome:
        corr(diff, β) = -0.080 (essentially zero).
      Meanwhile corr(diff, sf_final) = -0.838 — the same short-side-mass
      mechanism from 7.27.3 remains the dominant driver. Changing β does not
      meaningfully alter sf_final (all cluster at 35-38%) because the problem
      is not the action distribution sharpness but the BEHAVIOR DATA MIX:
      63% mean-reversion rows force shorts regardless of β.
    - absa_final is flat at 0.140-0.141 across all β — the model's mean|a|
      is determined by the data, not by the temperature.
    - ANSWERS THE PRE-REGISTERED CRITERIA:
        (1) PASS/FAIL: FAIL — no β stabilizes 32p-unweighted.
        (2) N/A.
        (3) β is NOT the binding constraint; data composition is.
    - IMPLICATION: the 32p-collapse is irrecoverable by optimizer tuning.
      The only known lever that stabilizes D at32p is composition capping
      (w_m ≤ 0.40 keeps short_frac ≤ ~10%, per 7.27.3). For the 32-policy
      unweighted dataset, either (a) apply composition-aware sampling (7.27.2
      already proves this works) or (b) accept the 4-policy regime as
      D's operating envelope.
    - SUITE: pytest 78/78 (analysis-only; same policy_weights/plumbing as
      7.27.2-7.27.3).

------------------------------------------------------------------------------
7.27.5 TARGET-NETWORK τ AND EXPECTILE τ ARE ALSO NOT BINDING (2026-08-31)
------------------------------------------------------------------------------
    - IDEA: two remaining IQL hyperparameters, both named "tau" in Model D:
      the POLYAK target-network EMA rate (cfg.tau) and the EXPECTILE tau
      (cfg.expectile). Sweep both on the 32-policy unweighted dataset to
      test whether either stabilizes the collapse that β (7.27.4) could not.
    - ADDED --tau CLI arg to src/models/d/train.py (only --expectile existed).
    - GRID:
        polyak    tau ∈ {0.001, 0.0025, 0.005, 0.01, 0.02}
        expectile tau ∈ {0.5, 0.6, 0.7, 0.8, 0.9}
      each × 5 seeds × both variants = 50 + 50 = 100 runs. Defaults:
      polyak 0.005, expectile 0.7. Defaults reproduce d_32p exactly.
    - PRE-REGISTERED SUCCESS CRITERIA:
        (1) Does any τ (either family) achieve final-epoch PASS on
            32p-unweighted?
        (2) If so, does the corresponding τ on comp-matched improve
            Sharpe > 0.837?
        (3) If no τ stabilizes 32p, τ (like β) is not the binding
            constraint.
    - RESULTS (scripts/d_tau_sweep.py, d_tau_resume.py, d_tau_analysis.py;
      100 runs split across two threads; 28 resume jobs after an 8h timeout):

      ┌───────────────┬────────┬─────────┬────────┬────────┬─────────┐
      │      τ        │ sh_best│ sh_final│  diff  │sf_final│absa_final│
      ├───────────────┼────────┼─────────┼────────┼────────┼─────────┤
      │ polyak/0.001  │ 0.7135 │ 0.6171  │ -0.096 │ 0.380  │  0.140  │
      │ polyak/0.0025 │ 0.7141 │ 0.6126  │ -0.102 │ 0.375  │  0.141  │
      │ polyak/0.005  │ 0.7136 │ 0.6262  │ -0.087 │ 0.357  │  0.141  │ (default)
      │ polyak/0.01   │ 0.7318 │ 0.6321  │ -0.100 │ 0.366  │  0.139  │
      │ polyak/0.02   │ 0.7103 │ 0.6254  │ -0.085 │ 0.362  │  0.139  │
      │ expectile/0.5 │ 0.7089 │ 0.6159  │ -0.093 │ 0.374  │  0.142  │
      │ expectile/0.6 │ 0.7088 │ 0.6277  │ -0.081 │ 0.368  │  0.141  │
      │ expectile/0.7 │ 0.7136 │ 0.6262  │ -0.087 │ 0.357  │  0.141  │ (default)
      │ expectile/0.8 │ 0.7118 │ 0.6258  │ -0.086 │ 0.361  │  0.141  │
      │ expectile/0.9 │ 0.7114 │ 0.6097  │ -0.102 │ 0.380  │  0.139  │
      │     d_32p     │ 0.7136 │ 0.6262  │ -0.087 │ 0.357  │  0.141  │
      └───────────────┴────────┴─────────┴────────┴────────┴─────────┘
      (D-minus-fuzzy, pack means, 5 seeds.)

    - FINDING: NO τ (either family) achieves final-epoch PASS. All ten
      settings give consistent negative diff, tightly clustered (−0.081 to
      −0.102, range 0.021). Neither dimension correlates with outcome:
        corr(diff, polyak)    = +0.078   (n=25, ~zero)
        corr(diff, expectile) = -0.058   (n=25, ~zero)
      while corr(diff, sf_final) = -0.823 (n=55) — the 7.27.3 short-side-mass
      mechanism remains the sole structural driver. sf_final stays 36-38%
      (full contrarian failure) across every τ setting; absa_final is locked
      at 0.139-0.142 (data-driven exposure, not update-dynamics-driven).
    - ANSWERS THE PRE-REGISTERED CRITERIA:
        (1) FAIL — no τ stabilizes 32p-unweighted.
        (2) N/A.
        (3) τ (both families) is NOT the binding constraint; data
            composition is.
    - IMPLICATION: the 32p collapse is irrecoverable by ANY IQL update-
      dynamics hyperparameter (β 7.27.4, polyak τ, expectile τ). The IQL
      algorithm itself is not the failure; the behavior mix (63% mean-
      reversion forcing contrarian shorts) is. The levers that DO hold are
      structural: composition-aware sampling (7.27.2/7.27.3) or the 4-policy
      regime. D's optimizer hyperparameters are at the IQL paper defaults and
      remain so.
    - SUITE: pytest 78/78 (analysis-only; --tau arg added to train.py).

------------------------------------------------------------------------------
7.23 KELLY-SIZED C+ VARIANT (2026-08-26)
------------------------------------------------------------------------------
    - IDEA (user): the C+ composite uses TACR's learned |a| for sizing, but
      the logistic's CONFIDENCE (P(up|s)) is discarded. Replace the TACR
      magnitude with fractional-Kelly sizing from the probability:
          a_t = k * (2*P(up|s_t) - 1) / sigma_t^2    (vol-scaled)
          a_t = k * (2*P(up|s_t) - 1)                 (constant-variance)
      sigma_t = causal trailing 20d std of SPY daily returns. Sign = the
      same logistic. Kelly concentrates capital on high-conviction days and
      penalizes high-variance days — the principled answer to "what size?"
      that Sharpe/DSR objectives structurally cannot give (7.2 finding).
    - KEY PROPERTY (flagged pre-run): the composite Sharpe and the EM margin
      are SCALE-INVARIANT to k (Sharpe is scale-free; both a and |a| scale
      together), so k only sets the absolute exposure, not the headline
      metric. k is selected on VAL for the exposure level; the test metric
      is the SHAPE of the sizing vs TACR's.
    - PRE-REGISTERED PROTOCOL (before running): fit the same 16-feature
      logistic (C on val); select k per variant on val by mean 5-seed val
      composite Sharpe; TEST untouched. Compare:
        * kelly_vol   : a = clip(k(2P-1)/sigma^2, -1, 1)
        * kelly_const : a = clip(k(2P-1), -1, 1)
      vs C+ (TACR |a|): pack mean test Sharpe 1.146, margins 5/5 (+0.30).
      BAR (same as C+): pack-mean test Sharpe > 1.146 AND margin > 0 in
      >= 3/5 (margins vs each variant's OWN EM). Report exposure (mean|a|)
      and drawdown as the k-dependent secondary.
    - RESULTS (scripts/hybrid_kelly.py; k selected on VAL, test untouched;
      the kelly strategy is DETERMINISTIC — no TACR magnitude, so a single
      point estimate vs the C+ 5-seed pack):
        variant  k(val)  valSh  test Sharpe  margin  mean|a|  maxDD
        const    10      0.792  0.8251      +0.062  0.780   -0.20
        vol      0.01    1.186  1.0986      +0.284  0.969   -0.19
        C+ (TACR |a|)                1.1463 +- 0.11  +0.296  5/5
      (vol = a = k(2P-1)/sigma^2 with causal 20d SPY vol; const = k(2P-1).)
    - PRE-REGISTERED BAR OUTCOME: FAIL on the Sharpe leg for BOTH variants
      (vol 1.099 < 1.146; const 0.825). The margin leg PASSES (vol +0.284,
      const +0.062, both positive).
    - READING (parsimony result, not an improvement): the vol-scaled Kelly
      sizing a = k(2P-1)/sigma^2 — a PURE logistic probability + causal vol,
      with NO transformer — reproduces the C+ margin (+0.284 vs +0.296) and
      comes within the C+ pack's own seed std of its Sharpe (1.099 vs 1.146
      +- 0.11). So TACR's learned |a| is NOT needed to extract the hybrid
      edge; a probability/variance sizing matches it. But it does NOT beat
      it — the transformer's sizing is marginally better, and the const
      (probability-only) variant is clearly worse (0.825), so the 1/sigma^2
      variance penalty is load-bearing (consistent with the tail-concentrated
      margin). Kelly is a viable, more parsimonious substitute for the C+
      magnitude, not a strict improvement.
    - C+ remains canonical (1.146, +0.296, 5/5).

7.28 CSI300 / CHINA CROSS-ASSET EXTENSION (2026-09-04)
------------------------------------------------------------------------------
    - SCOPE (user): re-run the full pipeline and all four models on the CSI300
      index (sh000300) instead of SPY, and evaluate whether the project's SPY
      results transfer to a non-US market. Same protocol throughout: train <=
      2018-12-31, val 2019-2020, test 2021-2024 (capped at SPLIT_TEST_END
      2024-12-31), 5-seed pack, per-model exposure-matched EM control.
    - DATA: CSI300 daily OHLCV from Sina via akshare
      (scripts/download_csi300.py -> data/csi300_daily.csv, 2005-01-04 to
      2026-09-03). East Money endpoints are blocked in this environment; Sina
      works. Pre-2005 rows carry volume=0 (back-calculated) and are dropped.
      The CHN pipeline (scripts/csi300_pipeline.py) builds the same artifacts
      as SPY: 5,264 trading days, 165,956 transitions, 32 behavior policies,
      regimes bull 2,589 / bear 2,294 / crisis 320 days, state dim 8.
    - CHECKPOINT AUDIT (methodological note): the first CSI300 pass silently
      reused SPY-era seed checkpoints (s1-s4 for TACR/D dated 2026-08-18..20;
      DDR s{seed} dated 2026-08-17). All seed packs were DELETED and retrained
      fresh on CSI300 before any number below was computed.
    - CHINA MACRO SET (best-effort full set; user-approved): 6 features from
      data/macro_cn/ (scripts/fetch_macro_cn.py; Sina/akshare, East Money
      blocked): rs_500_1d (vs CSI500), rs_growth_1d (vs ChiNext), rs_ss50_1d
      (vs SSE50), qvix_chg_1d/5d, qvix_level (50ETF implied-vol analog).
      Skipped (NO reliable China proxy in this env): 10y CGB yield (spotty
      history), USDCNY, credit spread, options skew, vol-term (no 3m-variant).
      Module: src/data/macro_factors_cn.py.
    - RESULTS (test 2021-2024, 5 seeds; data/csi300_model_comparison.csv):
        model             Sharpe(model)  Sharpe(EM)   margin   wins vs EM
        B (DDR naive)     +0.2616        -0.0176      +0.2792  5/5
        B-vt (DDR vol)    +0.4129        +0.0402      +0.3727  5/5
        C (TACR)          -0.5840        -0.5850      +0.0009  1/5
        C+ (z-only)       -0.3637        -0.5850      +0.2212  5/5
        C+ (z+macro)      -0.3486        -0.5358      +0.1872  4/5
        D (fuzzy+IQL)     -0.2390        +0.0671      -0.3061  0/5
        D-minus-fuzzy     -0.2385        +0.0553      -0.2938  0/5
    - HEADLINE FINDINGS:
        (1) DDR transfers cleanly: vol-targeted DDR is the best CSI300 model
            (+0.41 Sharpe, 5/5 vs EM) with naive DDR close behind (+0.26, 5/5).
            The SPY robustness verdict (DDR beats exposure-matched control) is
            RECONFIRMED on a non-US market.
        (2) C+ sign adds value over EM on CSI300 (margin +0.22, 5/5, z-only)
            but the weak TACR magnitude base keeps the composite Sharpe below
            zero. The hybrid's SIGN skill transfers; the transformer MAGNITUDE
            does not.
        (3) China macro does NOT improve the sign head on CSI300 (margin
            +0.22 -> +0.19, wins 5/5 -> 4/5). The 7.18 SPY macro edge does not
            transfer to a China proxy set. (Scoped: 6-feature best-effort set,
            with tnx/usdcny/credit/skew unavailable.)
        (4) D and TACR fail on CSI300 exactly as on SPY: D is worse than its
            own EM (margin -0.31, 0/5; shorts lose), TACR has zero directional
            signal (margin ~0.001, 1/5). D-minus-fuzzy is indistinguishable
            (ablation delta 0.0005, far below noise floor) — the fuzzy layer
            is null on CHN too.
        (5) CRISIS CAVEAT carried over: only 15 test days — directionally
            suggestive only, NOT a finding.
    - ARTIFACTS: scripts/download_csi300.py, scripts/csi300_pipeline.py,
      scripts/fetch_macro_cn.py, src/data/macro_factors_cn.py,
      scripts/hybrid_sign_macro_cn.py, scripts/compile_csi300_comparison.py;
      data/csi300_model_comparison.csv, data/macro_cn/, data/csi300_daily.csv.
      Trained CSI300 checkpoints live in the normal per-seed checkpoint dirs.

7.29 TRANSACTION COSTS — FORMULA FIX + NET-OF-COST SWEEP (2026-09-05)
------------------------------------------------------------------------------
    - PROMPT (user): "add trading cost into our models." The reviewer's
      challenge: all EM margins / model selection used ZERO-cost Sharpe as
      the yardstick; turnover was logged (TACR eval mean_turnover) but never
      priced. Cost can change WHICH model wins, not just the size of the win;
      C+'s "shorts ~6% of days, crisis-concentrated" edge is exactly where
      spreads widen and slippage spikes.
    - CODE VERIFICATION of the prior diagnosis (all four models share one
      knob, uniformly 0.0; the formula was a holding tax):
        * reward = a_t*ret_{t+1} - cost*|a_t| at src/data/behavior_policies.py
          (drives the OFFLINE DATASET reward -> TACR and D critics at cost
          bps from configs/data.yaml, set 0.0);
        * DDR mirrors it in its own _strategy_returns (train.py) and
          VolTargetBuffer (train.py / dsr.py), knob configs/ddr.yaml = 0.0;
        * EM margins computed at zero cost on BOTH sides
          (scripts/compile_csi300_comparison.py).
    - FORMULA FIX (the diagnosis was right): |a_t| charges a per-day HOLDING
      TAX on absolute position (fully-long B&H pays every day forever); a
      real transaction cost is paid only when the position CHANGES:
          cost_t = (bps/1e4) * |a_t - a_{t-1}|       a_{-1} = 0 (entry once)
      Implemented everywhere (still 0 bps by default, so nothing moves):
        * src/data/behavior_policies.py  reward = a*ret - cost*|a_t - a_{t-1}|
        * src/models/ddr/train.py        _strategy_returns(..., prev_action)
                                          carries last block's action (detached,
                                          truncated-BPTT consistent)
        * src/models/ddr/dsr.py          VolTargetBuffer charges the turnover
          cost on the DEPLOYED (vol-scaled) actions |a'_t - a'_{t-1}|, carry
          across calls
        * config comments updated (data.yaml, ddr.yaml, ddr/config.py).
    - RETRAIN WITH COST is NOT done here; this note prices cost ON THE
      EXISTING CHECKPOINTS (the reviewer's "cheap this-week" step; the honest
      rerun-that-changes-selection is a follow-up pending user go-ahead).
    - NEW: scripts/cost_sweep.py -> data/cost_sweep.csv. Net-of-cost margins on
      the CSI300 test pack, cost charged to BOTH the model and its own EM
      control (EM rebalances only when |a| changes):
          model_t = a_t*m_t   - bps/1e4 * |a_t - a_{t-1}|
          em_t    = |a_t|*m_t - bps/1e4 * ||a_t| - |a_{t-1}||
      Grid: bps in {0, 0.5, 1, 2, 5, 10} + crisis-x5 sets at 1 and 5 bps.
    - RESULTS (CSI300 test, net margins; wins = model beats own net EM):
        model             @0bps   @1bps   @5bps   @10bps   at 10bps wins
        B (DDR-naive)     +0.279  +0.268  +0.222  +0.165   5/5
        B-vt (DDR-volt)   +0.373  +0.359  +0.303  +0.234   4/5
        C+ (z-only)       +0.221  +0.181  +0.018  -0.184   0/5
        C+ (z+macro)      +0.187  +0.115  -0.176  -0.538   0/5
        C (TACR)          +0.001  +0.001  +0.000  -0.000   0/5
        D (fuzzy)         -0.306  -0.309  -0.321  -0.336   0/5
        D-minus-fuzzy     -0.294  -0.297  -0.308  -0.322   0/5
    - FINDINGS:
        (1) The RANKING DOES NOT FLIP on CSI300: B-vt wins at EVERY cost
            level (margin +0.37 at 0bps -> +0.23 at 10bps, still 4/5) — it
            has the lowest turnover of the winners, so it is the most
            cost-robust. Model B naive holds 5/5 at 10bps.
        (2) C+'s claim is the vulnerable one, as the reviewer bet: its margin
            HALVES by 1 bps and is gone by 5 bps (z-only) / 2-5 bps
            (z+macro). The everyday turnover of the sign head — not the 15
            crisis days — is what cost erases (the crisis-x5 boost barely
            moves any number: cost cells have ~0 short-test weight).
        (3) TACR standalone and D are untouched conclusions: already
            negative-margin / failing, they only get worse with cost.
        (4) Scoping honesty: this reprice uses FROZEN checkpoints trained at
            zero cost. A policy trained WITH cost inside the objective would
            re-learn to trade less and could shift ordering — that is the
            methodologically honest follow-up (regenerate dataset reward with
            bps>0 + retrain B/C/D, then re-run the sweep). Pending user goahead.
    - Item: the sweep is read-only (never trains, never touches checkpoints).

7.29.1 VOL-SCALED SENSITIVITY BAND (2026-09-05, addendum to 7.29)
------------------------------------------------------------------------------
    - PROMPT (user): frame the continuous cost model correctly — a vol-scaled
      multiplier is NOT a prediction of what spreads were; it is a SENSITIVITY
      ASSUMPTION. Run it as a robustness band across plausible k and ask "does
      the ranking survive?", not "what did cost actually cost?".
    - FORM (in scripts/cost_sweep.py, added to the grid):
          cost_bps_t = base_bps * (1 + k * vol_z_t),   clamped to >= 0
      vol_z_t = the models' OWN causal z_realized_vol_20d state feature (the
      normalization they saw in training), joined by date; k in {0.5, 1.0,
      2.0} x base bps in {0.5, 1.0, 2.0, 5.0}. At vol_z=+2 the multiplier is
      2x-5x depending on k — the stress-widening band. k=0 collapses to the
      flat grid (already in the sweep). Reported as an assumption band, NOT
      a calibrated cost model.
    - RESULTS (CSI300 test, net margins; cost to model AND its own EM;
      margin range across all 12 vol-scaled cells / min wins):
        B (DDR-naive)          +0.237 .. +0.277   (5/5 everywhere)
        B-vt (DDR-voltarget)   +0.318 .. +0.369   (5/5 everywhere)
        C+ (hybrid z-only)     +0.090 .. +0.218   (5/5 everywhere)
        C+ (hybrid z+macro)    -0.057 .. +0.178   (0/5 at the steepest cells)
        C (TACR)               ~+0.001            (1/5, unchanged)
        D / D-minus-fuzzy      -0.318 .. -0.294   (0/5, unchanged)
    - FINDINGS:
        (1) The RANKING IS STABLE ACROSS THE WHOLE BAND: B-vt wins every one
            of the 12 vol-scaled cells (5/5), B-naive never worse than second.
            The flat-grid verdict (7.29 finding 1) holds under every plausible
            cost-widening-with-vol assumption.
        (2) k is NOT the dominant stress — BASE bps is. At a fixed base,
            higher k slightly IMPROVES the positive-margin models (e.g. C+
            z-only at 5bps: +0.090 at k=0.5 -> +0.185 at k=2.0), because the
            (1 + k*vol_z) scaling redistributes cost AWAY from the calm days
            where these strategies actually turn over and TOWARD high-vol
            days. The continuous band brackets the flat-grid result from
            both sides rather than widening it.
        (3) C+ z-only survives the band (positive margin, 5/5) but only just
            at the steep end (min +0.09 vs its +0.22 at zero cost); C+
            z+macro dips negative in the steepest cells. Same qualitative
            message as 7.29 finding 2: C+'s edge is the cost-sensitive slice.
        (4) Framing note (user's point, adopted): the numbers are an
            assumption band. To turn this into "what cost actually was" you'd
            need quote/TAQ data (absent for both SPY and CSI300 here) — the
            band is the honest ceiling on that claim.
    - ARTIFACTS: scripts/cost_sweep.py (grid + band), data/cost_sweep.csv
      (vol_k column added).
------------------------------------------------------------------------------
    - Model C+ is the project's best result: a hybrid that decouples TACR's
      two skills — Transformer LEVERAGE TIMING (|a| from fix_a_nr7) and a
      linear DIRECTIONAL SIGN on 8 SPY + 8 macro causal features. Pack mean
      test Sharpe 1.15, EM margins 5/5 (+0.30), the first configuration to
      clear the >= 3/5 margin bar and to beat Model B (0.99). The macro
      features are the load-bearing addition (7.18); the result is robust
      under shifted splits, strongest in crisis windows (7.19).
    - What failed (all pre-registered, all nulls): TACR standalone and every
      sampler/Q-aggregation/structural variant below the bar (7.10-7.14);
      RTG relaxation is a small lever (7.16); trend-day filtering (7.20),
      focal-loss sign retraining (7.21) and SKEW sentiment (7.22) are
      closed nulls. Model D's fuzzy layer is null (7.9). Model B remains a
      strong baseline but not the best model.
    - Model D's offline-stability claim is SCALED: it trains stably at a
      4-policy mix and at 6.6x volume when the family composition is held
      fixed, and collapses on the unweighted 32-policy pool (7.27.1). The
      7.27.2 discriminator isolates the driver as composition, not volume;
      7.27.3 then identifies the MECHANISM beneath it — the instability
      tracks the fitted policy's short-side mass (short_frac), not distance
      from the old mix: stable through ~10% short_frac, fails once
      short-taking drives past ~10-30%. 7.27.4 closes the remaining
      algorithmic lever: sweeping the AWR temperature β ∈ {0.5, 1.0, 1.5,
      2.0, 3.0} produces zero correlation with outcome (r = -0.08); no β
      stabilizes the 32p dataset. 7.27.5 closes the two remaining IQL
      update-dynamics hyperparameters (polyak target-EMA τ and expectile τ)
      — likewise ~zero correlation (r = +0.08 / -0.06), no τ stabilizes the
      32p dataset. IQL's stability is a property of the short/long structure
      of the behavior it must reproduce, not a property of the optimizer,
      the optimizer's sample count, the policy distribution's sharpness, or
      the target/value-update dynamics.
    - The deliverable, as ever, is the discipline: every headline number is
      decomposed to margins, every protocol bar is pre-registered, and the
      two Phase-2 failure modes (data contamination, reward hacking) were
      root-caused rather than papered over.
    - CSI300 extension (7.28): DDR vol-targeted is the best CSI300 model
      (+0.41 Sharpe, 5/5 vs EM) — the SPY DDR robustness verdict transfers;
      the C+ SIGN transfers (margin +0.22, 5/5) but its TACR magnitude does
      not; China macro does NOT help the sign head; TACR and D fail on CSI300
      exactly as on SPY.
    - Transaction costs (7.29): the reward formula was corrected from a
      per-day holding tax cost*|a_t| to a true turnover cost
      cost*|a_t - a_{t-1}| (all models share the knob, still 0 bps default;
      so this is a code fix, not a result change). Net-of-cost repricing of
      the frozen CSI300 checkpoints shows the ranking is STABLE — DDR
      vol-targeted wins at every cost level (margin +0.37 @ 0bps -> +0.23 @
      10bps, 4-5/5) and Model B naive holds 5/5 at 10bps — while C+'s sign
margin halves by 1 bps and vanishes by 5-10 bps, exactly the fragile
      slice the reviewer predicted (everyday sign-head turnover, not the 15
      crisis days). Training with cost inside the objective (retrain) is the
      still-open honest follow-up.

    - Transaction costs, honest follow-up (7.29.2): the open thread from
      7.29 is now closed �?" B, C, C+, and D were RETRAINED (all 5 seeds, all
      variants, 1 bps inside the objective) and re-evaluated net-of-cost on
      the full flat + vol-scaled band. Result: B-vt still wins every cost
      cell (net margin +0.26 @ 0bps -> +0.16 @ 10bps, 4-5/5) and B-naive
      takes the runner-up 5/5 across the whole band. C+'s SIGN head survives
      1 bps ({+0.218 @ 0, +0.175 @ 1, +0.003 @ 5} z-only; z+macro similar);
      the vol-scaled band confirms it holds vertically ({+0.079..+0.214,
      5/5} z-only; {-.012..+0.235, 2/5} z+macro) �?" but NOT TACR itself:
      TACR retrained under cost collapses its magnitude head to a
      constant-|a| (test margin flat 0.000, val_mean_abs_action ~0.4-0.5),
      and C+ is then a best-of-breed decollation spin, not a tradeable
      magnitude. D stays dead ({-0.30..-0.32}, 0/5). Takeaways: (i) cost-
      aware retraining does NOT reorder the pack �?" B-vt and B-naive lead at
      every cost; (ii) absent chance, C+ cost-margin comes entirely from its
      sign head, which is 1-thru-2 bps sustainable but collapses by 5 bps;
      (iii) the reviewer's fragility claim is upheld for the TACR-magnitude
      slice, while the sign slice and B are cost-robust. S/F fix: the
      next_returns==buy_and_hold test now reads the authoritative manifest
      cost and credits the first-day entry cost (a_-1=0), so the Phase-1
      equivalence invariant holds at any bps.

7.30 PROJECT REDIRECTION — RESEARCH AGENDA (2026-09-08, user)
-------------------------------------------------------------------------------
    - DIRECTION (user, explicit): stop "implement two papers"; the experiments
      have exposed the load-bearing fact — the signal is NOT a transformer
      learning the whole policy, but DECOUPLED sizing/timing vs DIRECTION.
      Reframe the paper around: "Can epistemic uncertainty improve risk-aware
      POSITION SIZING in an offline-RL trading system?" — reducing exposure
      on unreliable decisions while PRESERVING high-confidence tail
      opportunities (uncertainty != bad trade; the ~5.5% high-confidence
      short tail is the whole C+ edge, and focal-loss proved it collapses if
      you ship it in the wrong direction).
    - PRIORITY (user): 1) 16-dim TACR state expansion (macro into the
      transformer state) — test whether C failed from information poverty,
      not Transformer-in-RL incapacity; 2) uncertainty diagnostics on C+
      (does epistemic disagreement predict future error?); 3-5) conditional/
      asymmetric/continuous uncertainty sizing, then multi-market
      (SPY/CSI300/+1) frozen-method validation.
    - DESIGN DB guards (read before any of this): uncertainty != bad trade
      (7.21 focal-loss null); C+ asymmetry (edge is short-tail concentrated);
      cost-robustness is now a SELECTION criterion, not just a caveat — C+
      net margin collapses by 5 bps while B-vt survives to 10 (7.29/7.29.2);
      the "four-model comparison intact" constraint governs the CANONICAL
      pack — new experiments are SEPARATE pre-registered tracks with their
      own checkpoint dirs/bars, per user decision (Q1 2026-09-08).
    - DECISIONS (user, 2026-09-08): Exp 1 = NEW pre-registered experiment,
      canonical C+ untouched; Exp 2 = 10-member bootstrap-logistic sign
      ensemble, C-selected on val as-is; execution order = Exp 1 then Exp 2.

7.30.1 EXP 1 PRE-REGISTRATION — 16-DIM TACR STATE (2026-09-08)
-----------------------------------------------------------------------------
    - QUESTION: TACR_{8} -> TACR_{16} — does giving the transformer the same
      8 macro causal features the logistic sign head already receives (risk_on,
      tnx_delta_1d/5d, vol_term, dxy_corr_20d, credit_1d, rs_qqq_1d, rs_iwm_1d)
      recover C+'s directional edge inside the ACTOR itself (|a| and sign
      jointly), or was the sign-only head's directional skill a property of
      the linear decoupling, not of the information set?
    - PROTOCOL BAR (pre-registered; the SPLIT is untouched):
        * STATE: 16-dim = [8 SPY z] + [8 macro causal z] (zscore_causal,
          expanding, >=60d). Dates restricted at LOAD TIME to finite-macro
          rows (2007-05+; HYG inception) — mirrors the sign model's matched
          dates. Val/test fall entirely inside 2007+, so no val/test loss.
        * CHECKPOINT ROOT: src/models/tacr/checkpoints/tacr/macro16/s{seed}/
          (own dir/bars; canonical checkpoints/tacr/s{seed}/ untouched).
        * CONFIG FLAG: state_macro: true opt-in (default false keeps 8-dim);
          all other hyperparams IDENTICAL to canonical fix_a_nr7
          (u=20, embed 128, n_layer 4, alpha 0.9, critic_lr 1e-6, double_q
          mean, balanced_families, 3k steps, 5 seeds [20260814,1,2,3,4]).
        * BARS (one device per question, no p-hacking):
          (a) Does the ACTOR alone (16-dim TACR, its action sign = direction)
              produce positive Sharpe AND beat its own |a| EM control?
              -> if yes, transformer CAN learn direction given the info set.
          (b) Compare C+ [TACR_{8}|a|] x [16-feat sign] vs a hypothetical
              C+16 [TACR_{16}|a|] x [16-feat sign]: does giving the
              transformer's magnitude head the macro states change sizing
              enough to matter net-of-cost?
          (c) Cost-robustness of any margin gains (1 bps, matching 7.29.2).
        * CONTROL: the 8-dim canonical is the reference; identical seeds and
          eval roll. Interference with the four-model comparison is NONE by
          construction (canonical checkpoints untouched, separate dirs).
    - METHODS: config flag + load_tacr_data widening (state.shape[-1] feeds
      actor/critic/action_model/payload automatically — no other wiring
      change). Ensemble sign head for Exp 2 unchanged.

7.31 dxy_corr_20d NaN-POISONING ARTIFACT — FOUND AND FIXED (2026-09-08)
-----------------------------------------------------------------------------
    - STATUS: INVALID as evidence. REASON (7.34): this entry's corrected-C+
      re-pricing ran on CSI300 data substituted for SPY (data/processed was
      CSI300 from 05-09) plus a tz-aware index bug in macro_factors that
      zeroed 4 macro dims on true SPY. ALL numbers in this entry are
      CSI300-with-US-macro artifacts. PRESERVED for audit trail; superseded
      by 7.34 (TRUE SPY / CORRECTED PIPELINE).
    - DISCOVERED (implementing Exp 1): ``dxy_corr_20d`` (trailing 20d corr of
      SPY vs DXY) was NaN for 3177/5264 rows — not pre-inception, but a
      CALENDAR-MISALIGNMENT artifact. SPY and DXY trade on different days; a
      single missing DXY day inside the 20d window makes ``rolling(20).corr``
      NaN for ~20 consecutive outputs (one bad day slides the poisoned
      window). Net effect: this one feature silently dropped ~half the dates
      for EVERY model using the 8-macro set — including the CANONICAL C+
      sign model's ``ok_mac`` (its "matched 2007+" set was actually stripped
      to ~1094 train / ~504 test dates, not 2007+).
    - FIX (user approved, 2026-09-08): ``_rolling_corr_on_overlap`` computes
      the correlation on the INTERSECTING SPY∩DXY calendar (drop rows where
      either is missing, roll on the clean series, ffill back onto the full
      calendar — strictly causal). dxy_corr_20d NaN 3177 -> 20 (leading
      warm-up only). src/data/macro_factors.py.
    - CONSEQUENCE (this is a protocol change to REFERENCED C+ results, not
      just Exp 1): re-running the canonical C+ sign model (7.18 B) with the
      fixed features changes it materially:
          B sign head        test_acc  short_frac  margin_mean  wins  Sharpe
          pre-fix (7.18)     0.5393    0.0637      +0.2962      5     1.146
          post-fix           0.5588    0.4133      +2.0837      5     1.762
      Day-accuracy 0.539 -> 0.559 is robust across C in [0.1,10] (test acc
      0.5594-0.5606) — NOT a C-selection overfit (val-selection picks C=1/10
      at 0.592, no sharp spike).
    - BUT — pre-cost/internal-scrutiny, DO NOT yet trust the +2.08 as "C+ got
      7x better". Diagnostics (s1): hybrid Sharpe 1.925 vs its own
      always-long EM -0.247; margin>0 on only 208/859 test days (2022-23 bear
      leg carried by 41% short exposure). CRITICAL: sign-head TURNOVER jumped
      to mean|da|=0.711 (vs EM 0.018) — daily sign-flipping. Per the
      net-of-cost selection criterion (7.29/7.29.2), the revised C+ is likely
      FAR MORE cost-fragile than the settled 0.41/1.15 version. Must be
      re-priced net-of-cost before it becomes the Exp 1 baseline.
    - RESOLVED 2026-09-08 (user protocol, Option 1): corrected C+ FROZEN as
      the Exp 1 baseline. scripts/hybrid_sign_cost_analysis.py re-prices it
      net-of-cost on the shared turnover convention (model_ret =
      a*m - (bps/1e4)*|da|, a_{-1}=0; EM on |d|a||), 5 seeds, C val-selected.
          bps   model_sh  em_sh   net_margin  wins  max_dd   drag_bp/d
          0.0    1.762  -0.322     +2.084      5   -0.072      0.000
          0.5    1.704  -0.332     +2.037      5   -0.072      0.219
          1.0    1.647  -0.343     +1.989      5   -0.073      0.439   <- PRIMARY
          2.0    1.531  -0.364     +1.895      5   -0.075      0.878
          5.0    1.183  -0.428     +1.611      5   -0.089      2.194
         10.0    0.604  -0.534     +1.138      5   -0.134      4.388
         20.0   -0.546  -0.747     +0.201      4   -0.275      8.776
      VERDICT: corrected C+ is GENUINELY cost-robust at the 1-bps selection
      standard (net margin +1.99, 5/5 wins, Sharpe 1.65). It is NOT a
      high-turnover artifact: turnover ~0.44 daily / ~111x annualized; Sharpe
      only degrades meaningfully above ~10 bps (20 bps kills it, model
      Sharpe -0.55). Edge is BROAD, not recovered-date-concentrated: RECOVERED
      (2505 d) margin +1.60 vs RETAINED (1790 d) margin +2.56, turnover
      0.44/0.43, short_frac 0.43/0.39. The +0.30 -> +2.08 jump mostly reflects
      the old poisoned set evaluating on a tiny (~504 d) misaligned test window.
      KEEP old +0.296/1.15 as the pre-correction historical result.
    - NOTE (2026-09-08): the scripted hybrid/cost table initially DISPLAYED a
      flat model_sh (a display bug — the model_sh column was computed at zero
      cost instead of the costed series). Fixed; margins/max_dd were already
      costed correctly. Output: data/hybrid_sign_cost_analysis.csv.
    - NEXT (Exp 1): 16-dim TACR must clear the corrected C+ at the SAME 1-bps
      cost regime (net margin is the primary selection metric per user
      protocol); evaluate both gross + net + turnover + DD identically.

7.32 EXP 1 RESULT — 16-DIM TACR (macro state) (2026-09-08)
-----------------------------------------------------------------------------
    - STATUS: INVALID as evidence. REASON (7.34): macro16 training AND its
      eval ran on CSI300 data substituted for SPY (data/processed was CSI300
      from 05-09) with tz-broken macro features. Checkpoints macro16/s{seed}/
      and data/macro16_eval.csv have been REPLACED by the 7.34 SPY retrain and
      SPY eval. PRESERVED text below for audit trail only.
    - RUN: scripts/macro16_run.py = canonical fix_a_nr7 recipe (dq-min +
      uniform + no-balanced-families) + --state-macro + --tag macro16.
      5 seeds [20260814,1,2,3,4]; checkpoints
      src/models/tacr/checkpoints/tacr/macro16/s{seed}/tacr_best.pt.
      best-val Sharpe per seed: 1.096 / 1.223 / 1.626 / 1.273 / 0.938
      (mean ~1.23, vs fix_a_nr7 ~0.82 — val-like-strong, but val is NOT the
      selection metric). Suite completed cleanly (only benign runpy warning).
    - EVAL: scripts/macro16_eval.py — like-for-like with corrected C+
      (scripts/hybrid_sign_cost_analysis.py): same corrected 16-feat logistic
      sign head, same 5 seeds, SAME cost model (model_ret = a*m - (bps/1e4)*
      |da|, a_{-1}=0; EM on |d|a||), 16-dim roll uses the 16-dim dataset
      (data16) while Fb/sign stay on the canonical 8-dim frame.
    - BAR (a) ACTOR-ONLY DIRECTION (16-dim TACR action sign = direction,
      1 bps): actor-Sharpe -0.272 vs its |a| EM -0.266. -> FAIL (not
      positive, does not beat EM). Same as canonical 8-dim (-0.01/-0.34).
      Reading: the transformer actor STILL cannot learn direction even with
      8 macro features IN its state — directional skill remains a property
      of the LINEAR decoupled sign head, not the information set. Answers
      7.30.1's core question in the NEGATIVE.
    - BAR (b) C+16 [TACR_16|a|]x[sign] vs corrected C+ [TACR_8|a|]x[sign],
      net margin (1 bps = PRIMARY):
          arm   bps  model_sh  em_sh   net_margin  wins
          C+16  0.0   1.917  -0.262    +2.179      5
          C+16  1.0   1.801  -0.266    +2.068      5   <- PRIMARY
          C+16  2.0   1.685  -0.271    +1.956      5
          C+8   0.0   1.762  -0.322    +2.084      5
          C+8   1.0   1.647  -0.343    +1.989      5
          C+8   2.0   1.531  -0.364    +1.895      5
      C+16 WINS the 1-bps selection metric (+2.068 vs +1.989, margin gap
      survives 0..2 bps). Modest but cost-robust sizing gain from the macro
      magnitude head.
    - BAR (c) COST: C+16 turnover 0.744 daily / 188x annualized (vs C+8
      0.439 / 111x) — meaningfully higher, yet net margin still holds at 1-2
      bps (raw Sharpe 1.80 > 1.65 compensates). C+16 is the more
      cost-fragile at extreme bps; within the selection regime it holds.
    - HEADLINE: (1) actor-only direction FAILS for 16-dim (Exp 1 bar a) — no
      transformer-learned direction from the macro info set. (2) Widening the
      MAGNITUDE head does beat the corrected C+ slightly net-of-cost at 1 bps
      (+2.068 vs +1.989, 5/5), so Exp 1 bar b PASSES (modest). (3) Therefore
      canonical C+ is NOT displaced; C+16 is a marginal, cost-robust upgrade
      in sizing only, direction still supplied by the linear sign head.
    - ARTIFACTS: scripts/macro16_run.py, scripts/macro16_eval.py,
      data/macro16_eval.csv; checkpoints macro16/s{seed}/.

7.33 EXP 1 DIAGNOSTICS — WHERE DOES THE EDGE COME FROM? (2026-09-08)
-----------------------------------------------------------------------------
    - STATUS: INVALID as evidence. REASON (7.34): ran on CSI300 data
      substituted for SPY with tz-broken macro features; its per-day table
      data/sign_diagnostics.csv was 859 CSI300 test days. That CSV has been
      REPLACED by the 7.34 TRUE-SPY rerun (1005 test days). PRESERVED text
      below for audit trail only. (Its QUALITATIVE verdict — uncertainty is
      NULL, abandon uncertainty as main contribution — SURVIVES the data fix,
      see 7.34 #3/#4 on SPY.)
    - PROMPT (user): do NOT train another 5-seed model yet. Build a per-day
      test table (date, 16-D feats, P(up), sign, C+ posn, TACR-8/16 |a|,
      realized ret, vol) and answer five gating questions before any
      uncertainty investment. (User's decision tree: does uncertainty predict
      failure? No -> abandon uncertainty as the main contribution.)
    - METHOD: scripts/sign_diagnostics.py. 10-member bootstrap L2-logistic
      ensemble on the corrected 16-feat set (same as C+'s B sign head, each
      member C-selected on val, TEST untouched); seed-mean TACR magnitudes;
      net-of-cost via model_ret = a*m - (bps/1e4)|da|. Per-day table ->
      data/sign_diagnostics.csv (859 test days, 2021-01-05..2024-12-31,
      no NaNs).
    - Q1 corrected-C+ direction after DXY fix (seed-mean |a|, 1bp):
        C+8   Sharpe 1.308  margin 1.669
        C+16  Sharpe 1.390  margin 1.651
    - Q2 does TACR magnitude add value over simple P/vol sizing? (1bp;
        margin scale-invariant, EM = own |a|):
        sizing           Sharpe(1bp)  margin    turnover   ann
        TACR-8 |a|            1.31     1.677     0.38      96x
        TACR-16 |a|           1.39     1.654     0.66     165x
        P |2P-1|              1.94     1.712     0.13      33x
        P |2P-1|/vol          2.07     1.980     0.81     204x
        RESULT: simple supervised probability sizing BEATS the transformer
        magnitude head (plain |2P-1|: Sharpe 1.94 vs 1.31 TACR-8, with 3x
        LOWER turnover 0.13 vs 0.38; P/vol highest Sharpe 2.07). The learned
        magnitude adds nothing over P + vol. STRONG negative for the RL
        magnitude contribution.
    - Q3 does ensemble U_var predict sign errors? NO. err-rate by U tercile:
        0.530 / 0.460 / 0.383 (INVERTED — higher U => FEWER errors);
        corr(U, err) = -0.127.
    - Q4 does U predict bad C+ P&L? NO (inverted). mean C+ P&L(1bp) by U
        tercile: -0.0002 / +0.0004 / +0.0012; corr(U, P&L) = +0.089.
    - Q5 long vs short asymmetry: SHORT days have LOWER err-rate (0.426 vs
        0.476 long) and HIGHER mean P&L (+0.00081 vs +0.00027); corr(U,P&L)
        short +0.172 vs long +0.033. Short-side is the better-informed leg,
        consistent with the corrected C+'s recovered short exposure.
    - VERDICT (user decision tree): uncertainty is NULL (worse — inverted) =>
        DO NOT invest in uncertainty-aware sizing as the main contribution.
        The transformer magnitude head is also NOT the source of edge (simple
        P sizing beats it). Direction-of-edge lives in the SUPERVISED 16-feat
        signal; sizing adds little over |2P-1|; vol-scaling helps.
    - NEXT: (a) confirm this on CSI300 (Exp 2D macro_cn); (b) the decomposed
        "Learning What vs How Much" story is the paper thesis; uncertainty
        re-framed as a sizing/calibration variable only where Q3/Q4 positive
        (they are not here). P/vol remains an un-tuned probe worth a proper
        cost-aware study (turnover/DD) as the natural baseline TR vs C+.

7.34 SPY RESTORE — 7.31-7.33 WERE CSI300-WITH-US-MACRO ARTIFACTS (2026-09-08)
-----------------------------------------------------------------------------
    - DISCOVERED (re-deriving 7.31-7.33): ``data/processed`` has been CSI300,
      NOT SPY, since 05-09 (at-cost retrain window) — while all scripts call
      ``load_tacr_data(20, ...)`` with no ``processed_dir`` override (default =
      ``data/processed``). So the corrected-C+ freeze, macro16 training, its
      eval, and the sign diagnostics ALL ran on CSI300 data with US macro
      features (cross-market contamination).
      - Evidence: close levels = CSI300 (982 / 2858 / 5076 / 3579 at
        2005/2009/2015/2024); ``data/processed/offline_dataset.parquet`` =
        165,956 transitions ending 2026-09-02 (CSI300) vs SPY backup 168,516
        ending 2026-03-30 (SPY 2006-2026); 8-dim coverage 3152+487+969=4608 ==
        CSI300 load (SPY maps to 4783 dates). The manifest config claims
        ``universe SPY`` for BOTH because csi300_pipeline copies the SPY config
        object into the manifest while loading CSI300 daily data — parquet
        files, not the manifest, are the source of truth.
    - USER DECISION (2026-09-08): RESTORE SPY, re-run everything.
      - CSI300 preserved -> data/processed_csi300_backup/ (05-09 timestamps).
      - SPY restored into data/processed/ from data/processed_spy_backup/
        (03-09 timestamps). load_tacr_data now: 4783 dates, clipping 310,
        market mean 3.98 bp/day. INR LOGS 7.31-7.33 ARE SUPERSEDED.
    - SECOND BUG FOUND: ``macro_features`` NaN-poisoning on tz-aware SPY
      features. SPY features index is ``datetime64[ns, America/New_York]``
      (tz-aware); the old construction
      ``pd.Series(spy_daily["close"].astype(float).pct_change(), index=naive)``
      reindexes a tz-aware pct_change output by tz-naive labels -> ALL NaN.
      On CSI300 (naive index) it silently worked, which is why 7.31-7.33 saw
      "live" macro features. Net effect on TRUE SPY: risk_on_1d, dxy_corr_20d,
      rs_qqq_1d, rs_iwm_1d were all-NaN -> zscore_causal zeroed them ->
      the macro dims were dead in every SPY-era sign/magnitude analysis.
    - FIXES applied (all three files):
        src/data/macro_factors.py: spy_ret built from .to_numpy() onto the
          naive index, then pct_change() (label-independent).
        scripts/hybrid_sign_cost_analysis.py: same fix in the old-dxy
          reconstruction helper AND the test-day indexer (pd.DatetimeIndex(
          p["date"]).tz_localize(None)) — the roll's dates are tz-aware while
          the frame is naive; without it get_indexer returned -1 -> ALL sign
          preds came from row -1 (a spurious all-long collapse).
        src/data/macro_factors_cn.py: same defensive to_numpy fix.
      After fix on SPY: risk_on_1d 1 NaN, tnx_delta_1d 11, tnx_delta_5d 15,
      vol_term 386, dxy_corr_20d 20, credit_1d 571, rs_qqq_1d 1, rs_iwm_1d 1
      (5284/5344 finite).
    - RE-RUN ON TRUE SPY (5 seeds, 1 bps = PRIMARY selection metric):
      - CORRECTED C+ (FROZEN single-logistic sign head, hybrid_sign_macro B
        definition) net margins NEGATIVE at every cost level:
          bps:  0.0    0.5    1.0     2.0     5.0     10.0    20.0
          marg: -0.034  -0.046  -0.058  -0.082  -0.153  -0.273  -0.510
          wins: 1      1      1       0       0       0       0
          short_frac 0.0756, turnover 0.147 daily / 37x annual.
        On TRUE SPY the corrected C+ LOSES to EM at all costs — the +1.99 /
        +2.068 Exp-1 result was CSI300-with-macro contamination.
      - 16-feat logistic probe (SPY): test_acc 0.531-0.540 across C in
        [0.01,10], test short 0.076-0.099; up-rate 54.3%. Sign model is at
        base rate; no learned directional edge vs all-long.
      - macro16 RETRAINED on SPY (5 seeds, same recipe): best-val Sharpe
        (SPY) seed4 annealed early (0.77 @ e4; stale 15:02 log holds the
        CSI300-era 1.096/1.223/1.626/1.273/0.938 numbers — the SPY-era val
        numbers are NOT the ones in the log). Val is not the selection metric.
      - macro16_eval on SPY (both C+16 and C+8): net margin negative at all
        costs (C+16: -0.023/-0.036/-0.050/-0.077; C+8: -0.034/-0.046/-0.058/
        -0.082). C+16 does NOT beat corrected C+ on SPY.
      - sign_diagnostics on SPY (BOOTSTRAP-ENSEMBLE sign head, seed-mean |a|):
          #1  C+8  margin(1bp) +0.228 ; C+16 +0.244   (EM +0.846/+0.799)
          #2  P-sizing |2P-1| +0.118 ; P/vol +0.016
          #3  corr(U_var, err) +0.012 -> NULL (no inversion here, unlike 7.33)
          #4  corr(U_var, P&L)  +0.039 -> NULL
          #5  LONG n=905 err 0.458 ; SHORT n=100 err 0.490
    - RESOLUTION OF THE TWO CONTRADICTORY SPY RESULTS: the sign HEAD decides
      the outcome, not the magnitude.
          single logistic (frozen B)  margin(1bp) -0.052  (per-seed mean -0.058)
          10-member bootstrap ensemble  margin(1bp) +0.228
      Isolation: per-seed |a| with the single sign gives the SAME negative
      margin as seed-mean |a| (per-seed margins -0.056/-0.063/-0.049/+0.021/
      -0.144). The positive +0.23-0.24 comes from the ensemble-mean-proba sign,
      which reaches an extra ~2.4pp of shorts (0.100 vs 0.076). Both sign-head
      definitions are pre-registered in the repo (7.31 froze the single
      logistic; 7.33/7.30.1 Exp 2 use the bootstrap ensemble). On TRUE SPY the
      frozen single-logistic corrected C+ shows NO edge over EM at any cost.
    - VERDICT: uncertainty remains NULL on SPY (Q3/Q4 corr ~0) => the
      decision-tree conclusion (abandon uncertainty as the main contribution)
      SURVIVES the data fix. But the corrected-C+ "beats EM at 1 bps" baseline
      does NOT survive: the frozen sign definition loses to EM at all costs on
      true SPY, and its Exp-1 "winner" (C+16 +2.068) was a contamination
      artifact. The decomposed-policy thesis must be re-grounded on the TRUE
      SPY numbers (both sign heads, per-day table data/sign_diagnostics.csv
      now 1005 test days SPY).
    - NEXT: (a) decide the baseline sign-head definition (frozen single
      logistic vs bootstrap ensemble) on true SPY before any further
      experiment; (b) run the frozen-protocol CSI300 transfer decomposition
      (Exp 2) with the corrected data now in place (data/processed =
      SPY canonical, data/processed_csi300_backup = CSI300 preserved); (c)
      P/vol study (Exp 3) as the un-tuned strong baseline; (d) directional-
      information ablation (Exp 4). Sync changed files to E:\New folder(2).

7.35 SIGN-HEAD DECISION + ABLATION (user protocol, 2026-09-08)
-----------------------------------------------------------------------------
    - DECISION (user): do NOT pick between single-logistic and ensemble sign
      heads by which one preserves the original C+ story (outcome-driven).
        (1) FREEZE the single-logistic sign head as the CANONICAL C+ baseline
            (it was the frozen 7.31 definition; the ensemble was only
            introduced after inspecting the corrected SPY results).
        (2) PROMOTE the 10-member bootstrap ensemble to a FORMAL SECONDARY
            model; investigate WHY aggregation works (variance reduction?)
            before using it anywhere as the sign model.
        (3) The scientific framing is now: "The apparent C+ advantage was
            caused by data contamination and implementation bugs; after
            correcting the pipeline, the original single-model directional
            edge disappears on true SPY." That is itself the important result.
        (4) Uncertainty/disagreement stays OUT of sizing: P_ens = mean(P_1..
            P_10); sizing |2P_ens-1| and |2P_ens-1|/vol. This is coherent
            with Q3/Q4 being null (aggregation can improve the PROBABILITY
            ESTIMATE while disagreement has zero predictive value for errors).
        (5) PAUSE Exp 2/3/4 (no CSI300 yet). Only after the controlled
            ablation: if aggregation is robust -> use the ensemble as the
            sign model for CSI300 transfer, clearly labeled "ensemble
            extension", keeping single-logistic C+ as canonical baseline.
            If not robust -> abandon the C+ extension direction.
    - CONTROLLED ABLATION (scripts/sign_head_ablation.py), same corrected SPY
      protocol as the frozen cost re-pricing:
          head                                  purpose
          single logistic                       canonical C+ (frozen)
          10-member bootstrap logistic ens.     current positive result
          each individual bootstrap member      member quality / seed
                                                 sensitivity (isolate)
          ensemble on IDENTICAL training data   isolate ensembling: with a
                                                 deterministic solver these
                                                 members coincide with the
                                                 single model, so ANY ensemble
                                                 gain vs single is attributable
                                                 to bootstrap resampling.
      Per-member metrics on the test roll: Sharpe(0bp/1bp), margin(1bp),
      accuracy, short frequency, turnover, P(up) distribution — then compare
      the member distribution vs the ensemble average-probability curve.
      Final block: |2P_ens-1| and |2P_ens-1|/vol20d sizing (no uncertainty in
      the rule), margins vs EM.
    - RESULTS (corrected TRUE-SPY protocol; 1005 test days, up-rate 0.537):
      -- heads (1 bps net margin; ps = per-seed |a| frozen convention,
         sm = seed-mean |a| diagnostics convention) --
         single logistic (frozen canonical)    acc=0.5313  short=0.0756
             margin(1bp) ps=-0.0581  sm=-0.0520   -> NEGATIVE
         10-member bootstrap ensemble (mean P)  acc=0.5393  short=0.0995
             margin(1bp) ps=+0.1751  sm=+0.2279   -> POSITIVE (seed 20260908)
         10-member IDENTICAL-data ensemble (mean P)
             == single logistic EXACTLY (margins identical, -0.0581/-0.0520)
             => any ensemble gain is due to bootstrap RESAMPLING, not to
             probability averaging per se (deterministic solver).
      -- member-level isolation (single 10-member draw, sm margins):
         member margins: [-0.725 -0.244 -0.079 -0.125 -0.259 -0.399,
                          +0.438 +0.584 +0.009 +0.106] mean=-0.069
         The +0.228 ensemble is DRAGGED by TWO members (+0.584, +0.438); the
         other 8 are negative or ~zero. Ensemble <= best member.
         => NOT evidence of broad variance reduction; the mean-proba sign is
         a lucky-draw artifact of which bootstrap samples landed 2021-2024.
      -- robustness (identical protocol, 10 independent RNG seeds):
         margins(1bp,sm): +0.228 +0.068 -0.279 +0.021 -0.003 -0.239 -0.184
                          -0.015 +0.111 +0.084
         min -0.279 / max +0.228 / mean -0.021 / positive-frac 0.50
         => the ensemble's apparent edge is NOT robust to bootstrap seeding.
      -- no-uncertainty sizing with the ensemble probability (single seed):
         |2P_ens-1|        Sharpe(1bp)=1.020  EM=0.902  margin=+0.118
         |2P_ens-1|/vol20d Sharpe(1bp)=1.044  EM=1.028  margin=+0.016
         (the +0.118 for |2P-1| is exactly the 7.34 diagnostics value — the
         sizing edge is scale-invariant and does NOT depend on the sign-head
         du jour; it is a property of the P(up) calibration itself.)
    - DETERMINISTIC DATA AUDIT: the ablation re-verified data alignment.
      market_returns[t] IS the forward next-day return (corr=1.0 with
      close_pc[t+1]) — the correct reward target a*R_{t+1}. The confusing
      "P(up)=0.0746" is 2*uprate-1 of the +-1 sign series (up-rate 0.537),
      NOT a misalignment. No data bug here.
    - VERDICT (resolves the 7.34 single-vs-ensemble conflict): the
      single-logistic C+ is NEGATIVE on true SPY at all costs (frozen
      canonical, -0.05 to -0.06 @1bp). The bootstrap ensemble does NOT
      robustly restore a directional edge: its +0.228 was a single-seed
      artifact; across 10 seeds the ensemble margin is ~0 (mean -0.02,
      positive only half the time), driven by 1-2 lucky members. Therefore
      on TRUE SPY the ORIGINAL single-model directional C+ edge does NOT
      exist after the pipeline corrections. The only cost-robust feature
      on SPY is the scale-invariant supervised sizing |2P-1| (+0.118 @1bp),
      independent of which sign head produced P.
    - IMPLICATION for the thesis: "RL magnitude adds no value; direction is
      a supervised-sign property" is REINFORCED but in the weaker sense — on
      true SPY even the sign does not reliably beat EM after costs, and the
      ensemble can't be leaned on (unstable). The honest claim for the paper
      is the NULL/REDUX: the apparent C+ edge was pipeline contamination;
      after correction it does not reproduce on SPY. |2P-1|/vol-style sizing
      remains the only robust candidate edge and needs the full P/vol study
      (Exp 3) before any positive claim.
    - NEXT (per user decision tree): ensemble does NOT survive the controlled
      ablation => do NOT use the ensemble as the sign head for CSI300
      transfer as a substitute for C+. Instead: (a) proceed to CSI300 (Exp 2)
      ONLY under the FROZEN single-logistic C+ protocol to test transfer of
      the (SPY-null) canonical sign, reporting honestly; (b) prioritize the
      P/vol study (Exp 3) where the real candidate edge lives; (c) keep the
      ensemble ablation as a methodological appendix (post-hoc sign-head
      comparison, seed-sensitivity documented).

### 7.36 Exp 3 - P/vol SIZING study on TRUE SPY (frozen canonical protocol)

- SCOPE: after the sign-head ablation (7.35) demonstrated the directional sign
      is exhausted on TRUE SPY (single C+ negative at all costs; bootstrap
      ensemble positive only for a lucky seed), the surviving cost-robust
      feature was the scale-invariant event-scaled sizing |2P-1| (+0.118 @1bp
      in 7.34/7.35). This Exp rigorously tests that edge:
        * PRIMARY: P(up) from the FROZEN CANONICAL single-logistic sign head
          (C-selected on val, deterministic) - NOT the bootstrap ensemble.
        * rules (each vs OWN EM = |a|, so margin is scale-invariant):
              sign-only (|a|=1)          - does direction beat SPY forward ret?
              |2P-1|                      - canonical confidence sizing
              |2P-1| / vol20d             - + volatility normalization (run ?)
              1/vol20d (no P)             - pure inverse-vol control
              ensemble |2P-1| (ref)       - the 7.34/7.35 single-seed number
        * cost grid 0/0.5/1/2/5/10 bps, net Sharpe + margin(1bp).
        * ROBUSTNESS: exact same 10-RNG-seed bootstrap sweep as 7.35, but for
          the SIZING margins (so we do NOT repeat the lucky-seed mistake).
- ARTIFACTS: scripts/pvol_study.py (runnable), data/pvol_study.csv (1005-day
      per-day test table: P_up_can, P_up_ens, sign_can, sign_ens, fP=|2P-1|,
      fP_vol, vol20d, ret, ret_sign).
- RESULTS (test = 2021-2024, n=1005, up-rate 0.537, mean fwd ret +5.13 bp/d):
      sizing rule            turnover short%  | 1bp Sharpe  margin(1bp)
      sign-only (|a|=1)         0.216    7.6  | 0.72        -0.060
      |2P-1| (canonical)        0.048    7.6  | 0.86        +0.051
      |2P-1| / vol20d           0.343    7.6  | 0.96        -0.025
      1/vol20d (no P)           1.442    7.6  | 0.83        -0.190
      ensemble |2P-1| (ref)     0.055   10.0  | 1.02        +0.118
      (EM = own |a|; sign-only EM = buy-and-hold SPY fwd-return baseline.)
      ROBUSTNESS sweep margins(1bp, canonical rule, 10 resampled P heads):
        |2P-1|    [.118 .052 .022 .024 .123 .058 .062 .002 .035 .093]
                   min=+0.002 max=+0.123 mean=+0.059 posfrac=1.00
        |2P-1|/vol[.016 -.000 -.033 -.066 .020 -.024 -.025 -.075 -.036 .002]
                   min=-0.075 max=+0.020 mean=-0.022 posfrac=0.30
- FINDINGS:
  * |2P-1| SIZING IS THE ONLY SUB-EDGE THAT IS ROBUST TO RESEEDING: positive
    margin(1bp) in 10/10 bootstrap draws (mean +0.059, spread +0.002..+0.123).
    ~+0.05 Sharpe pts net at 1bp with the frozen canonical P - modest but
    sign-consistent, and NOT dependent on the ensemble or its RNG.
  * The /vol variant is NOT robust (posfrac 0.30, mean -0.022) and the pure
    inverse-volatility control is clearly negative (-0.190) - the 7.34/7.35
    "|2P-1|/vol +0.016" does NOT survive: volatility normalization is at best
    value-neutral, at worst harmful, in the cross-sectional daily setting. The
    claim "volatility-aware sizing" is insufficient; only the CALIBRATED
    confidence scaling |2P-1| adds value.
  * Mechanistic attribution (canonical P): the +0.051 margin of signed |2P-1|
    vs its own long-only EM comes ENTIRELY from the short leg (76 short days):
    zeroing the short days gives margin +0.000; and it concentrates in the
    top confidence tercile (fP>0.11: margin +0.150) while low-confidence
    terciles are negative (-0.033, -0.144). corr(|2P-1|, |R|)=+0.027 (P does
    NOT predict return magnitude); corr(P_up, |R|)=+0.004. So the sizing gain
    is NOT vol timing; it is that shorting is rare (7.6% of days) and only
    fires at high confidence where the conditional directional Sharpe is
    positive (SHORT leg n=76 SR 0.82 vs LONG n=929 SR 0.95).
  * Direction itself is confirmed exhausted on TRUE SPY: sign-only (|a|=1)
    gives margin -0.060 vs buy-and-hold, consistent with 7.34/7.35 (single
    logistic C+ negative at all costs).

### 7.37 Exp 2 - CSI300 TRANSFER of the frozen canonical C+ protocol

- SCOPE: transfer the FROZEN CANONICAL protocol (7.33/7.36) to CSI300 with the
      SAME feature construction (8 price causal z + 8 macro causal z with the
      CSI300 close as "instrument side" of the US cross-asset spreads), SAME
      splits (train <=2018-12-31, val 2019-2020, test 2021-2024), SAME
      single-logistic P(up) head C-selected on val, SAME sizing-vs-own-EM
      margin, SAME cost grid, SAME 10-RNG-seed resampling robustness. Run on
      the CSI300 Phase-1 artifacts (data/processed_csi300_backup/). Plus two
      additional arms:
        * OOS WINDOW CHECK - the SAME frozen head rolled onto the untouched
          2025-2026 window (no retrain, up-rate 0.547) to test for
          window-specific luck vs. a persistent edge.
        * OOD DIAGNOSTIC - the SPY-TRAINED canonical P applied directly to the
          CSI300 16-dim features WITHOUT retraining (a genuine cross-market
          out-of-distribution probe; interpret P~0.5 only, z-statistics are
          market-local).
- PIPELINE NOTE: the canonical pvol/sign path reads data/processed (SPY) and
      state_macro hardcodes the SPY features path, so the transfer runs on a
      NEW self-contained script (scripts/csi300_transfer.py) that builds the
      16-dim frame + forward market returns from a given processed dir's
      features_regimes.parquet. CSI300 frame is naive-indexed (2005-01-04..
      2026-07-17 after finite-feature filter, n=4700).
- ARTIFACTS: scripts/csi300_transfer.py (runnable), data/csi300_transfer.csv
      (867-row canonical-test-day table: P_up, sign, fP, fP_vol, vol20d, ret),
      PROJECT_NOTES 7.37.
- RESULTS (2021-2024 canonical test, n=867, up-rate 0.489, test-window drift
      cumprod 0.887 = net DOWNTREND; 1 bps):
      sizing rule          turnover  short% | 1bp Sharpe  margin(1bp)
      sign-only (|a|=1)       0.758   37.9  | 1.31        +1.400
      |2P-1| (canonical P)    0.113   37.9  | 1.91        +1.577
      |2P-1| / vol20d         0.707   37.9  | 2.04        +1.865
      1/vol20d (no P)         5.174   37.9  | 1.13        +1.561
      Canonical P head test acc 0.541 (up-rate 0.489); corr(P[t], R[t+1])=
      +0.157; corr(sign, R)=+0.093.
      ROBUSTNESS (10 RNG seeds, resampled P heads):
        |2P-1|    margins 1.58..1.94, mean +1.773, posfrac=1.00
        |2P-1|/vol         1.92..2.23, mean +2.069, posfrac=1.00
      OOS WINDOW (2025-2026, SAME head, n=329, up-rate 0.547, drift 1.213):
        acc 0.581 | sign-only margin +1.571 | |2P-1| +2.059 | corr(P,R)
        +0.231  => the CSI300 edge PERSISTS and even strengthens out-of-window
        (well above the canonical-test numbers).
      OOD DIAGNOSTIC (SPY head -> CSI300, no retrain):
        acc 0.489 (= up-rate, pure chance) | sign-only margin -0.197 |
        |2P-1| -0.398 | corr(P_ood, P_csi300)=+0.022  => zero-weight transfer
        of the learned SPY head FAILS (predictions essentially uncorrelated);
        but the PROTOCOL re-estimated on CSI300 finds a strong edge.
- FINDINGS / INTERPRETATION:
  * CSI300 is the OPPOSITE of SPY on every axis that mattered in 7.34-7.36:
    -- direction itself is strongly positive (sign-only +1.40 @1bp, vs SPY -0.06);
    -- |2P-1| sizing is strong AND robust (mean +1.77, 10/10 seeds, vs SPY +0.059);
    -- |2P-1|/vol beats plain |2P-1| HERE (+1.865 vs +1.577; and +2.07 vs +1.77
       across seeds), the exact opposite of SPY where vol-normalization DESTROYED
       the margin (posfrac 0.30). The 7.36 claim "vol normalization is harmful"
       does NOT transfer: on CSI300 conditional-vol scaling adds real value.
  * Neither the strong CSI300 result nor the null SPY result is a window
    artifact: SPY has two independent null windows, CSI300 has two independent
    positive windows (2021-24 + 2025-26 with the same frozen head).
  * The head C-selected on val has no test-touch; the robustness sweep resampled
    the TRAINING bootstrap per seed with the SAME test window - the CSI300 result
    is seed-stable under that protocol (posfrac 1.00 in both rules).
  * OOD transfer is null because the SPY face is null (7.34-7.36): a head with
    no directional signal on its home market cannot transfer one to a new market.
    This is consistent, not a leak: the mechanism claim is "the PROTOCOL
    transfers (feature build + C-selection + logistic) and finds a valid edge on
    CSI300", NOT "SPY learned weights transfer".
  * CAVEAT for the paper: the CSI300 margins (~1.4-2.1 Sharpe pts) are unusually
    large for a daily logistic on index returns and sit in a high-volatility
    Chinese-index regime (2021-24 downtrend, mean fwd ret -0.66 bp/day; 2025-26
    rebound). Report honestly with the OOS/OOD arms and do NOT over-claim a
    tradable edge: the point is that the supervised-sign + event-scaled sizing
    protocol REPRODUCES (strongly) on a second market, whereas the RL magnitude
    and uncertainty channels showed nothing on either market.

-------------------------------------------------------------------------------
### 7.38 INFERENCE + DIAGNOSTICS on the frozen-canonical sizing results (2026-09-09)
------------------------------------------------------------------------------
- SCOPE: statistical inference + CSI300 diagnostics for the paper (submission
      prep: "statistical significance, not only Sharpe"). Reconstructs net
      daily returns for every sizing rule DIRECTLY from the saved per-day test
      tables (data/pvol_study.csv, data/csi300_transfer.csv) using the EXACT
      recorded _net convention (model_ret = a*m - (bps/1e4)*|da|, a_-1=0; EM =
      |a| with its own turnover cost; Sharpe annualized mean/std*sqrt(252) via
      src/eval/regime_eval). Cross-validates every reconstructed Sharpe(1bp) /
      margin(1bp) against the recorded 7.36/7.37 tables (PASSED, all within
      tolerance). Adds Newey-West HAC t on the daily (model-EM) margin and
      moving-block-bootstrap (l=21, also l=5 sensitivity) 95% CIs. CSI300:
      by-year, max-drawdown, return distribution, regime breakdown.
- INTEGRITY: reads ONLY the saved per-day tables; does NOT re-run training,
      does NOT re-touch the already-reported-once 2025-2026 OOS windows.
- ARTIFACTS: scripts/inference_diagnostics.py (runnable), data/
      inference_diagnostics.csv (per rule: sharpe_1bp, em_sharpe_1bp,
      margin_1bp, mean_daily_margin_bp, ann_margin_pct, nw_tstat, ci95_bp_lo/hi
      [block bootstrap l=21], ci95_margin_lo/hi [paired-block Sharpe
      difference]), data/csi300_yearly.csv, data/csi300_regime.csv.
- SPY (test n=1005, up-rate 0.537), 1 bps:
      rule                 Sh1bp Sh(EM) margin1 m_bp/d  t_NW  CI95 bp/d (l=21)
      sign-only (|a|=1)     0.72   0.78  -0.060  -0.392  -0.20 [-3.93,+3.83]
      |2P-1| (canonical)    0.86   0.81  +0.051  +0.035  +0.41 [-0.080,+0.239]
      |2P-1| / vol20d       0.96   0.99  -0.025  -0.117  -0.30 [-0.53,+0.78]
      1/vol20d (no P)       0.83   1.02  -0.190  -7.611  -0.86 [-23.4,+11.6]
      ensemble |2P-1| (ref) 1.02   0.90  +0.118  +0.077  +0.76 [-0.075,+0.319]
      IMPORTANT NUANCE: the |2P-1| +0.051 margin on SPY is NOT statistically
      significant (t_NW 0.41; block-bootstrap CI on the daily margin [-0.08,
      +0.24] bp/d includes 0). Same for ensemble +0.118 (t 0.76, CI includes
      0). So on SPY the surviving claim is DIRECTION- and SEED-consistency
      (10/10 reseeded margins positive, 7.36), NOT significance. Report as
      "positive sign, consistent across reseeding, not individually
      significant." sign-only, /vol, 1/vol: null/negative, no significance.
- CSI300 (test n=867, up-rate 0.489), 1 bps:
      rule                 Sh1bp Sh(EM) margin1 m_bp/d  t_NW  CI95 bp/d (l=21)
      sign-only (|a|=1)     1.31  -0.09  +1.400  +10.555  +2.07 [-0.79,+22.7]
      |2P-1| (canonical)    1.91   0.34  +1.577   +1.850  +2.80 [+0.509,+3.58]
      |2P-1| / vol20d       2.04   0.17  +1.865  +10.171  +2.84 [+2.44,+19.4]
      1/vol20d (no P)       1.13  -0.43  +1.561  +62.562  +2.09 [-3.41,+132]
      SIGNIFICANCE: |2P-1| (t_NW 2.80) and |2P-1|/vol (2.84) have block-
      bootstrap 95% CIs that EXCLUDE 0 (daily margin) - statistically
      distinguishable from their own EM net of 1bp costs. sign-only (2.07) and
      1/vol (2.09) have t_NW >2 but block-bootstrap CIs include 0 (heavy right
      tail; high turnover). All four are robust in sign across the 10-seed
      sweep (7.37). The primary |2P-1| CSI300 claim IS significant.
- CSI300 BY-YEAR (margin@1bp; up-rate; drift):
      2021 n=221 up 0.534 drift -3.7%  sign +1.04 |2P-1| +1.73 /vol +1.33
      2022 n=216 up 0.472 drift -13.4% sign +4.42 |2P-1| +3.97 /vol +4.99
      2023 n=215 up 0.465 drift -4.0%  sign -0.12 |2P-1| +0.69 /vol +0.62
      2024 n=215 up 0.484 drift +10.8% sign +0.04 |2P-1| -0.03 /vol +0.11
      => edge is NOT uniform across years: it concentrates in the 2021-2022
      downtrend legs and is approx zero/negative in the 2024 rebound. This is
      consistent with the SPY mechanism (short-leg events carry the margin)
      and MUST be reported with the paper's CSI300 numbers or a reviewer will
      find it. 2025-2026 OOS (reported once in 7.37) was a rebound window yet
      still positive - so "downtrend only" is too strong; say "concentrated in
      high-confidence-event years, not uniform, persists in untouched window."
- CSI300 REGIME (margin@1bp; Phase-1 labels):
      bull  n=318 up 0.503  sign +1.35 |2P-1| +1.07 /vol +0.69
      bear  n=530 up 0.479  sign +1.50 |2P-1| +2.00 /vol +2.74
      crisis n=19 up 0.526  sign  0.00 |2P-1|  0.00 /vol  0.00
        (crisis: model NEVER shorts - all 19 days sign=+1 (mean P 0.607), so
         model==EM exactly; margin 0 by construction. Honest, not a bug.)
      => magnitude concentrates in bear votes; bull still positive; crisis has
      no short activity at all.
- CSI300 |2P-1| daily (model - EM) distribution: mean +1.85 bp/d, std 17.1,
      skew +2.52, kurt +29.2, q05 -10.9 / q10 -3.3 / med 0.0 / q90 +10.0 /
      q95 +27.0 => heavily right-tailed; median daily margin 0 (edge from rare
      large short-side wins) - matches the "event-driven short leg" mechanism.
- CSI300 max drawdown (model net @1bp cumulative path):
      |2P-1| model -0.018 vs EM -0.060; /vol model -0.108 vs EM -0.206;
      sign-only model -0.178 vs EM -0.328 => event-scaled sizing cuts DD
      roughly 3x vs its own long-only EM on CSI300.
- CROSS-VALIDATION: all 10 reconstructed Sharpe(1bp)/margin(1bp) rows match
      the recorded 7.36/7.37 tables within tolerance (fail-loud exit code 1).
      This independently confirms the recorded numbers are reproducible from
      the retained per-day tables.
- PAPER-SPOKEN SUMMARY for the reviewer-preparation work (User's checklist):
  * SPY: statistical CIs do NOT establish significance for any sizing rule;
    the |2P-1| claim is seed-robustness only. State that plainly.
  * CSI300: |2P-1| and |2P-1|/vol ARE significant vs own EM @1bp (t_NW ~2.8,
    bootstrapped daily-margin CI excludes 0).
  * CSI300 edge is year-nonuniform (2021/22 strong, 2023 weak, 2024 ~0) and
    regime-concentrated in bear; report the by-year table in the paper §4.x.
  * No conventional-ML baseline was added (user decision: logistic P head
    already is the conventional baseline).

-----------------------------------------------------------------------------------------------
### 7.39 Exp 4 - NIFTY TRANSFER of the frozen canonical C+ protocol (2026-09-10)
-----------------------------------------------------------------------------------------------
- SCOPE: third-market replication of 7.37 (Exp 2) on NIFTY 50, same frozen
      canonical protocol: same 8+8 feature build (8 price causal z + 8 macro
      causal z, market close as "instrument side" of the US cross-asset
      spreads), same splits (train <=2018-12-31, val 2019-2020, test
      2021-2024), same single-logistic P(up) C-selected on val, same
      sizing-vs-own-EM margin @ 1bp primary, same cost grid, same 10-RNG-seed
      resampled-P robustness, same OOS (2025-2026, untouched, no retrain),
      same OOD arm (SPY-trained P -> NIFTY z-features, no retrain). Then the
      SAME 7.38 inference gates (Newey-West HAC t, MBB l=21/l=5 CIs,
      cross-validation vs the recorded table) extended to NIFTY.
- DATA / COVERAGE (honest caveats, NOT protocol changes):
  * Downloaded from Yahoo ^NSEI (scripts/download_nifty.py, mirror of the
    keyless chart-API fetcher used for the macro series). NIFTY 50 is a price
    index (no dividends), consistent with the naive-index convention used for
    CSI300/SPY frames.
  * Yahoo carries zero volume before ~2013-01-21, so the SAME volume>0 drop
    rule as CSI300 yields NIFTY raw coverage 2013-01-21 .. 2026-09-09. The
    test (2021-2024, n=881) and OOS (2025-2026) windows are FULLY covered;
    the TRAINING window starts 2013 instead of 2005/2007 -> the NIFTY run has
    ~6y of training (2013-2018) vs SPY/CSI300 ~11-13y. This is a real,
    reportable asymmetry (the paper must state it), not an error.
  * Feature frame ends 2026-07-17 (not 2026-09-09) because the VIX3M macro
    series ends on that date -> the "2025-26 OOS window" arm is 2025-01-01 ..
    2026-07-17 (~1.5y, n=332), the same clipping mechanism the CSI300 OOS
    arm effectively had. Reported here once (no re-touch afterwards).
- ARTIFACTS: scripts/download_nifty.py, scripts/nifty_pipeline.py,
      scripts/nifty_transfer.py (all runnable); data/nifty_daily.csv,
      data/processed_nifty_backup/ (features_regimes.parquet,
      offline_dataset.parquet, dataset_manifest.json),
      data/nifty_transfer.csv (881-row per-day test table),
      data/nifty_yearly.csv, data/nifty_regime.csv. inference_diagnostics.py
      extended to iterate all three markets (+ NIFTY cross-validation).
- RESULTS (2021-2024 canonical test, n=881, up-rate 0.547, test-window drift
      cumprod 1.801 = strong UPTREND, mean fwd ret +7.10 bp/day; 1 bps):
      sizing rule            turnover short%  | 1bp Sharpe  margin(1bp)
      sign-only (|a|=1)         0.873   37.0  | 3.37        +2.130
      |2P-1| (canonical P)      0.217   37.0  | 4.40        +3.109
      |2P-1| / vol20d           1.700   37.0  | 4.77        +3.099
      1/vol20d (no P)           7.335   37.0  | 3.62        +2.108
      Canonical P head test acc 0.608 (up-rate 0.547); corr(P[t], R[t+1])=
      +0.313; corr(sign, R)=+0.204.
      ROBUSTNESS (10 RNG seeds, resampled P heads):
        |2P-1|    margins 3.01..3.25, mean +3.179, posfrac=1.00
        |2P-1|/vol         3.05..3.30, mean +3.197, posfrac=1.00
      OOS WINDOW (2025-01-01..2026-07-17, SAME head, n=332, up-rate 0.524):
        acc 0.566 | sign-only margin +2.300 | |2P-1| +3.234 | corr(P,R)
        +0.231  => the NIFTY edge PERSISTS out-of-window, above the
        canonical-test numbers (same pattern as CSI300 7.37).
      OOD DIAGNOSTIC (SPY head -> NIFTY, no retrain):
        acc 0.541 (= up-rate, near chance) | sign-only margin -0.524 |
        |2P-1| -0.002 | corr(P_ood, P_nifty)=+0.017  => weights-level transfer
        of the SPY head FAILS again; only the PROTOCOL re-estimation carries
        the edge (identical to the CSI300 OOD result: -0.197/-0.398, corr
        +0.022).
- STATISTICAL INFERENCE (same 7.38 gates; SPY/CSI300 rows unchanged for the
      record):
      NIFTY rule              Sh1bp Sh(EM) margin1 m_bp/d  t_NW  CI95 bp/d (l=21)
      sign-only (|a|=1)       3.37   1.24  +2.130 +11.838  +2.93 [+3.42,+21.0]
      |2P-1| (canonical)      4.40   1.29  +3.109  +4.402  +3.87 [+2.18,+6.94]
      |2P-1| / vol20d         4.77   1.67  +3.099 +28.431  +4.53 [+15.7,+42.4]
      1/vol20d (no P)         3.62   1.51  +2.108 +81.217  +2.98 [+24.1,+139]
      SIGNIFICANCE: |2P-1| (t_NW 3.87), |2P-1|/vol (4.53) AND sign-only
      (2.93) all have block-bootstrap 95% CIs EXCLUDING 0 - significant vs
      own EM @1bp. 1/vol (2.98) hits t>2 but the CI includes 0 (heavy tail).
      ON NIFTY the direction + confidence sizing are BOTH significant, unlike
      CSI300 where only |2P-1| and |2P-1|/vol were significant.
- NIFTY BY-YEAR (margin@1bp; up-rate; drift):
      2021 n=220 up 0.545 drift +24.1% sign -0.03 |2P-1| +1.03 /vol +0.74
      2022 n=223 up 0.511 drift  +5.9% sign +5.00 |2P-1| +5.31 /vol +4.80
      2023 n=219 up 0.571 drift +17.7% sign +2.69 |2P-1| +3.28 /vol +3.40
      2024 n=219 up 0.562 drift +16.4% sign +0.89 |2P-1| +2.90 /vol +2.90
      => UNLIKE CSI300, the NIFTY edge is present in ALL FOUR years
      (2021-2024), strongest in 2022. This is a major qualitative contrast to
      report: on NIFTY the claim is NOT "concentrated in downtrend legs" - the
      edge survives in up years too. NIFTY test window was a +80% uptrend, the
      opposite regime to CSI300's 2021-24 downtrend, and the edge is LARGER.
- NIFTY REGIME (margin@1bp; Phase-1 labels):
      bull   n=618 up 0.557  sign +1.36 |2P-1| +2.39 /vol +2.37
      bear   n=228 up 0.500  sign +4.68 |2P-1| +5.62 /vol +5.34
      crisis n= 35 up 0.686  sign -2.48 |2P-1| -0.83 /vol -0.85
      => bear is strongest again (same as CSI300), bull positive, BUT the
      NIFTY crisis/surge regime is NEGATIVE - the model shorts and LOSES in
      the 35 high-up-rate crisis days (up 0.686). This is the OPPOSITE of the
      CSI300 crisis arm (where the model never shorted, margin exactly 0).
      Must be reported with the regime table: "event-scaled sizing is
      negative in NIFTY crisis/surge labels" - honest, and it prevents
      over-claiming downside protection.
- NIFTY |2P-1| daily (model - EM) distribution: mean +4.40 bp/d, std 30.2,
      skew +5.67, kurt +77, q05 -12.7 / q10 -2.9 / med 0.0 / q90 +16.6 /
      q95 +40.3 => same right-tailed event-driven shape as CSI300, but
      fatter tail and bigger mean.
- NIFTY max drawdown (model net @1bp cumulative path):
      |2P-1| model -0.014 vs EM -0.069; /vol model -0.085 vs EM -0.321;
      sign-only model -0.086 vs EM -0.138 => event-scaled sizing cuts DD 3-5x
      vs its own long-only EM on NIFTY too.
- CROSS-VALIDATION: all 10 reconstructed Sharpe(1bp)/margin(1bp) rows for all
      THREE markets (incl. new NIFTY row) match the recorded 7.36/7.37/7.39
      tables within tolerance (fail-loud exit code 1). PASSED.
- FINDINGS / WHERE THIS LEAVES THE CLAIMS (paper implications):
  * The |2P-1| confidence-scaled sizing claim now REPLICATES, and is
    statistically significant, on 2 of 3 markets (CSI300, NIFTY); on SPY it
    is seed-stable in sign but not significant (7.38). This STRENGTHENS the
    paper's third result: "statistically significant exposure-scaled margin
    on two of three markets; seed-robust on all three."
  * NEW qualitative fact: NIFTY direction (sign-only) is ALSO significant
    (t 2.93, CI excl 0), so the "direction is exhausted" claim from SPY does
    NOT generalize - it was SPY-specific. The paper must say this: the
    protocol's directional value is market-dependent (SPY null, CSI300
    borderline, NIFTY significant).
  * NIFTY edge is near-uniform across years and NOT driven by the downtrend
    (test window was +80%); CSI300's year-nonuniformity does not repeat.
    Together they say: "market-dependent in LEVEL, not in existence" - the
    margin exists on both non-US markets, larger on NIFTY.
  * Crisis/surge regime negative on NIFTY (unlike CSI300) - a real negative
    finding that prevents over-claiming "downside protection." Report as-is;
    the dominant driver is bear/vote concentration on both markets.
  * Training-history asymmetry (NIFTY trains from 2013, not 2005) is a real
    limitation that must appear in the paper (shorter history, same protocol,
    LARGER edge - if anything strengthens the protocol-transfer claim but the
    paper must not hide the difference).
  * OOD (SPY->NIFTY) null again, consistent with CSI300: "learned weights do
    not transfer; the PROTOCOL does."
- TESTS: still 78/78 (no source-module changes; new scripts are standalone
      mirrors). Mirror E:\New folder(2) synced (notes + data/scripts).

### 7.40 PAPER UPDATE - NIFTY folded into paper/results.md (2026-09-10)
-----------------------------------------------------------------------------------------------
- SCOPE: integrate the 7.39 NIFTY record into the manuscript without changing
      any measured numbers and without renaming/reordering existing section
      anchors (discussion.md and related_work.md cite §4.4/4.6/4.7/4.8/4.9/4.10/
      4.11).
- CHANGES (paper/results.md, mirror synced):
  * §4.7 retitled "Cross-market validation on CSI300 and NIFTY"; the 3-market
      contrast table now carries NIFTY margin + t_NW columns
      (sign-only +2.13/t2.93; |2P-1| +3.11/t3.87; /vol +3.10/t4.53).
  * New NIFTY sub-block in §4.7 (after the CSI300 importance paragraph):
      coverage caveat (train starts 2013, ~6y vs ~11-13y; frame ends
      2026-07-17 via VIX3M), full 4-rule table with EM Sharpe + 1bp margin +
      t_NW + MBB CI (l=21), 10/10 reseed robustness (mean +3.18/+3.20),
      sign-only ALSO significant (t 2.93, CI excl 0) => "direction is
      exhausted" is SPY-specific, by-year table (all four years positive,
      largest 2022), by-regime (bear +5.62 strongest; crisis/surge NEGATIVE
      -0.83, n=35, up 0.686 - genuine negative finding vs CSI300 crisis
      degeneracy), daily-margin distribution (mean +4.40, std 30.2, skew +5.7,
      kurt +77), max-DD improvement (|2P-1| -0.014 vs EM -0.069 ≈ 5x,
      /vol -0.085 vs EM -0.321), verdict: significant, uniform across years,
      uptrend, crisis-negative.
  * §4.8 (OOW) extended: NIFTY arm reported once with CSI300 (acc 0.566,
      sign-only +2.30, |2P-1| +3.23, corr +0.231; n=332, up 0.524; same
      VIX3M end-date clip as CSI300's OOS arm).
  * §4.9 (OOD) extended: SPY head -> NIFTY null is recorded (acc 0.541 ≈
      up-rate 0.547; sign-only -0.52, |2P-1| -0.002, corr ≈ 0.02); blockquote
      now says "second or third market".
  * §4.10 summary table: added NIFTY significance rows; unified persistence /
      uniformity / weight-transfer rows to both markets; central claim
      rewritten to three-market statement (SPY seed-stable-not-significant,
      CSI300 significant-and-regime-dependent, NIFTY significant-and-
      near-uniform-but-crisis-negative; direction market-dependent: null/borderline/
      significant).
  * §4.5 distribution sentence: NIFTY same right-tailed shape at higher
      amplitude, cross-referenced to §4.7.
  * §4.11.3: input tables now include data/nifty_transfer.csv.
- NO numbers changed for SPY (7.36) or CSI300 (7.37); all NIFTY figures match
      7.39 / data/nifty_transfer.csv / nifty_yearly.csv / nifty_regime.csv /
      inference_diagnostics.csv.
- NEXT OPEN: §4.8 final sentence still refers to "the two reported-once
      out-of-window arms" - still accurate (CSI300 + NIFTY). Packaging (c)
      remains deferred per prior order; discussion.md/related_work.md still say
      "two non-US markets" implicitly via §4.7/4.8/4.9 references - no text
      needed beyond the results changes already made (re-check at packaging).
- TESTS: 78/78 (results.md/prose only). Mirror synced.

### 7.41 PAPER UPDATE - discussion.md + related_work.md aligned to 3 markets (2026-09-10)
-----------------------------------------------------------------------------------------------
- SCOPE: after folding NIFTY into results.md (7.40), update the two drafts that
      referenced the old two-market framing; no numerical changes, no new
      claims.
- CHANGES (paper/discussion.md, paper/related_work.md, mirror synced):
  * discussion.md 5.1: "six results" -> "seven results"; new item 5 (NIFTY
      replication, significant, all four years, crisis-negative); renumbered
      uncertainty (6) and protocol-transfer (7, now stripping "CSI300 features"
      -> "CSI300 or NIFTY features").
  * discussion.md 5.3: "The two markets differ in magnitude, not in kind" ->
      "three markets differ in magnitude, significance, and regime profile..."
      with a NIFTY bullet (near-uniform, crisis-negative) and an updated
      closing statement ("second and third market; CSI300 regime-concentrated,
      NIFTY near-uniform in level, crisis-negative").
  * discussion.md 5.5: rewritten to wed CSI300 concentration with the NIFTY
      counterpoint (uniform across years, uptrend, largest) -> explicit
      rejection of portable year-by-year edge; market-dependent in level, not
      in existence; both OOS arms reported (n=329 CSI300, n=332 NIFTY).
  * discussion.md 5.6: "Universal volatility-normalized sizing" bullet now
      says positive on CSI300 AND NIFTY but still not universal; notes
      sign-only IS significant on NIFTY (market-dependence of directional
      value); weight-transfer bullet -> SPY->CSI300/NIFTY.
  * discussion.md 5.7 limitations: "Two markets only" -> "Three markets, but
      one macro feature source" (SPY/CSI300/NIFTY share US-macro features);
      "Unusually large CSI300 effect" -> "Unusually large non-US effects"
      (+1.4..+1.9 CSI300, +3.1 NIFTY); "Regime concentration" -> "Regime
      dependence" (CSI300 non-uniform across years; NIFTY uniform but
      crisis/surge negative).
  * discussion.md 5.8: cleaned the garbled "distinguishable... in neither
      direction nor magnitude" sentence; now states magnitude adds nothing on
      any of the three markets while directional value is null on SPY and
      significant only on re-estimated (CSI300 regime-concentrated / NIFTY
      near-uniform, crisis-negative).
  * related_work.md 2.2: "(iv) validated on a two-second market" (typo) ->
      "validated on two further markets (CSI300, NIFTY) and an untouched
      out-of-window period."
  * related_work.md 2.5: "(harmful on SPY, beneficial but regime-concentrated
      on CSI300)" -> "(harmful on SPY, beneficial on CSI300 and NIFTY,
      § 4.4/4.7)".
- VERIFY: grep for remaining "second market" / "two markets" / "both markets"
      -> results.md:322 (CSI300-specific phrase, correct), results.md:424/429
      ("both markets" = CSI300+NIFTY in the OOW context, correct).
- TESTS: 78/78 (prose-only changes). Mirror synced.

### 8.0 REPRESENTATION LABORATORY - WHAT HIDDEN STATES KNOW (2026-09-15)
-------------------------------------------------------------------------------
- SCOPE: a new research thread that makes the LEARNED market-state
      representation the explicit object of study instead of an invisible
      step inside end-to-end RL. Four studies built on the frozen models
      (DDR/B/TACR/C/D, seed 20260814) plus two new training labs (Idea 16
      "objective lab", and the advisor's "representation x policy" design).
      All probes are linear (L2 logistic / linear regression) fit on TRAIN
      only and reported on TEST - the same no-lookahead protocol as the
      models. All outputs under data/interpret/.
- WHAT WAS BUILT (all reproducible):
  * src/interpret/{targets,extract,probes}.py - shared probe targets
      (direction_1/5/20, magnitude_1, vol_5/20, regime, reconstruction),
      frozen-model hidden-state extraction (DDR rnn out[:,-1], TACR
      state-token embeddings, D encode/h_last, C+ logistic scores, raw
      windows), linear-probe + standardize/drop_nan utilities.
  * scripts/probe_representations.py -> probe_results.csv / probe_table.csv.
  * scripts/representation_rank.py -> representation_rank.csv (effective
      participation-ratio + spectral-entropy rank, both NaN-safe).
  * scripts/rep_sensitivity.py -> sensitivity_scores.csv, cka_similarity.csv
      (temporal masks, feature zero/perm, +/-1sigma counterfactuals, CKA).
  * src/models/objective_lab/ + scripts/objective_lab.py,
      scripts/objective_lab_compare.py - IDEA 16 lab (below), outputs
      objective_{probe_table,rank,cka,outcomes}.csv.
  * scripts/temporal_receptive_field.py - block-mask receptive field + the
      feature x time importance matrix, outputs
      temporal_receptive_field.csv / temporal_feature_x_time.csv.
  * src/models/rep_lab/ + scripts/rep_lab.py, rep_lab_measure.py - the
      representation x policy lab, outputs rep_{measure,matrix,cka}.csv.
- TESTS: 78/78 (1 new pytest run, nothing below touches existing tests).

### 8.1 LINEAR-PROBE BATTERY + RANK ON THE FROZEN MODELS (SPY, test split)
-------------------------------------------------------------------------------
- Probe table (acc for clf / R2 for reg; baseline dir1 = 0.538):
      rep        dir1   dir5   dir20  |R|1    vol5   vol20  regime  recon
      raw 8      0.538  0.592  0.662  0.105  0.343  0.382  0.852   1.000
      raw 20x8   0.529  0.553  0.654  0.010  0.212  0.377  0.883   1.000
      DDR(GRU)   0.531  0.591  0.651  0.074  0.207  0.085  0.857   0.893
      TACR       0.501  0.579  0.643  0.013  0.209  0.242  0.886   0.982
      D          0.512  0.567  0.618 -0.052  0.134  0.149  0.909   0.945
      C+ logit   0.538  0.586  0.661 -0.007 -0.048 -0.073  0.768   0.162
  * Direction_1 is at/below majority chance everywhere (C+ == baseline
      0.5383 exactly): the 8 canonical features contain NO linear 1-day
      direction signal. Direction_5/20 are above baseline but <= raw.
  * D is the best regime carrier (0.909); C+ (decision scores) predicts
      almost nothing but regime (0.768) - its info content is the sign.
  * RL representations (DDR/TACR/D) linearly DROP magnitude/vol precision
      preserved by raw (vol20 0.382 -> 0.085-0.242).
- Rank (effective participation ratio / spectral entropy; spectral = exp H(p)):
      raw 8: 3.08 / 4.42 ; raw window: 7.60 / 17.24
      DDR: 1.79 / 3.17 (top1 74%) ; TACR: 3.55 / 7.71 ; D: 2.62 / 5.29
  * Reference guessed GRU~8, TACR~6; actual: GRU even more collapsed
      (~3), TACR ~7.7. All < 8 effective dims - severe linear collapse in
      every learned state (paper ref: spectral entropy of covariance eigs).

### 8.2 SENSITIVITY + CKA (masking perturbation on frozen test rows, n=300)
-------------------------------------------------------------------------------
- Temporal masks (zero the state row L days back -> |da| action movement):
      DDR: lag1 0.093, lag3 0.025, lag10 0.006 (dh_rel 0.27 -> 0.02): GRU
           receptive field ~ last 3 days.
      TACR: flat - masking ANY single day moves da ~0.0005, dh ~0.009 sigma.
      D: flat (~0.005 da).
- Feature importance (|da| whole-window zero): DDR volz 0.11 > boll 0.09 >
      macd 0.088 > ret_1d 0.075 ; TACR macd/boll ~0.024 > rsi 0.021 >
      volz 0.0065 > rv20 0.002.
- +/-1sigma counterfactual (last-day): DDR asymmetric - rsi -1s 0.084 vs
      +1s 0.047; volz +1s 0.129 vs -1s 0.088. TACR small and symmetric.
- Linear CKA (test): DDR-TACR 0.80, DDR-D 0.76, TACR-D 0.83.

### 8.3 IDEA 16 - OBJECTIVE LAB (trading representation vs reward representation)
-------------------------------------------------------------------------------
- DESIGN: four objectives share ONE 128-d single-layer GRU encoder over the
      20x8 z-window (identical data/seed/splits/blocks): A predictive (MSE
      to standardized next-day return), B DSR (DDR reward, vol-targeted),
      C A2C (ac-sigma 0.15, ent 0.003), D masked self-supervised
      reconstruction of today's features. Only the objective differs.
- TRAIN (val): A -1.608 (mean-predictor level), B Sharpe 1.461, C -0.276,
      D recon -0.363. TEST outcomes: A RMSE 104 bps corr +0.057; B Sharpe
      +0.901; C -0.784; D recon 0.488 z.
- CROSS-OBJECTIVE probes (test): all four ~ baseline on direction_1
      (0.51-0.55). Differences in the per-arm probe table
      (objective_probe_table.csv): actorcritic is the only arm with +|R|1
      (0.072) and the best vol5 0.291/vol20 0.210; dsr drops magnitude/vol
      (|R|1 -0.004, vol20 0.074) while keeping regime 0.860.
- RANK (test): dsr is the LEAST collapsed (spectral 9.23 / eff 5.03);
      predictive 3.64/2.35; actorcritic 3.57/2.31 (but scale 0.143 ->
      near-constant h!); masked 7.72/3.91.
- CKA: predictive-masked 0.952, predictive-AC 0.931, dsr-AC 0.792 (lowest),
      dsr-predictive 0.862, dsr-masked 0.856, AC-masked 0.885.
- VERDICT: at fixed architecture/capacity the reward function shapes the
      TAIL, not the core - ~79-95% of the linear subspace is common across
      objectives (direction + regime = the market-state core). The DSR
      reward arm is most distinct, keeps the most linear dimensions, is the
      only behaviorally strong arm (Sharpe 0.90), and is the one that drops
      linear magnitude/vol. So: "reward representation" episodes are real
      but bounded; the shared trading-state content dominates.
- CAVEAT: C is a generic A2C proxy (not literally TACR); architecture is
      shared by design so the objective is the clean comparison variable.

### 8.4 TEMPORAL RECEPTIVE FIELD + FEATURE x TIME importance
-------------------------------------------------------------------------------
- Block-mask protocol from next_experiments.txt (I_k = E[(a - a^mask)^2],
      freeze models, n=300 test days):
      I_k:          last5d   prior5d   oldest10d   all20d
      DDR           0.0355   0.0031    0.0008      0.0361
      TACR          0.0023   0.0000    0.0001      0.0023
      D             0.0787   0.0008    0.0037      0.0844
  * The claim "80% of action variance from the last 3 days" is reproduced
      and STRONGER: 93-98% of each model's masked response is in the last
      5 days. But TACR's TOTAL I_k is 0.0023 vs DDR's 0.036 - TACR barely
      moves when ANY context is destroyed, so it is not "last-3-days
      driven", it is near-context-invariant. dh_rel keeps the long block
      for D (0.30) but not DDR (0.08) / TACR (0.10).
- feature x time matrix (single-cell masks, |da|): both models are lag-1
      dominated: DDR lag1 boll 0.080 > volz 0.074 > macd 0.050 > rsi 0.046
      > ret_1d 0.035, decaying ~15x by lag5, ~0.001 beyond; TACR lag1
      boll 0.023 > macd 0.021 > rsi 0.016 > ret_5d 0.014, falling to
      ~0.0003 by lag5.
- NOT done (needs retraining): context scan u=10/20/40/60; Idea 5
      temporal-leakage shuffles; Idea 13 full matrix is done, Idea 6 null
      features not built.

### 8.5 REPRESENTATION x POLICY LAB (advisor design: learn rep, then RL as
       downstream test of representation usefulness)
-------------------------------------------------------------------------------
- DESIGN (Stage 1): three representation learners SHARE the same 128-d GRU
      encoder, only the objective differs - auto (decode the whole 20x8
      window from h_t), predictive (multi-task MSE to standardized
      {R1, R5, |R1|, vol5} targets), contrastive (InfoNCE, feature-mask +
      time-mask + z-noise views). Frozen encoders are measured, then
      consumed by (Stage 2) one generic A2C head (same form as lab arm C)
      and by a "Supervised" logistic direction policy on h_t.
- TRAIN (val): auto -0.730, predictive -1.356 (best @ ep26), contrastive
      -2.600 (InfoNCE); downstream A2C val Sharpe raw -0.224, auto +0.852,
      predictive -0.454, contrastive +0.258.
- MEASURE BEFORE TRADING (test), rep_measure.csv:
      rep          dir1  |R|1   vol5   vol20  regime  recon   eff/spec rank
      raw 160d     0.530 0.010  0.213  0.380  0.884   1.000   6.3/13.7
      auto         0.516 0.029  0.195  0.061  0.872   0.959   4.8/7.1
      predictive   0.537 -0.018 0.161  0.075  0.856   0.989   2.8/4.5
      contrastive  0.518 0.050  0.050 -0.054  0.861   0.941   5.3/8.5
  * No learned rep carries linear 1-day direction (all ~0.53 baseline) -
      the features lack it, not the learner. All encoders COLLAPSE
      long-horizon vol (vol20 0.38 -> <=0.08) while keeping regime.
- REPRESENTATION x POLICY test Sharpe (rep_matrix.csv):
      rep          Supervised  RL-A2C
      raw 160d        0.394    -0.221
      auto            0.635    +0.139
      predictive      1.036    -0.323
      contrastive     0.317    +0.076
  * RL does NOT compensate for representation: the same A2C head is
      negative/weak on EVERY frozen rep, while the supervised direction
      policy is 2.6x better on the predictive rep than raw (0.394 -> 1.036).
      Representation learning helps supervised decision-making, not RL.
- CKA (test): raw-auto 0.946, raw-predictive 0.689 (most transform), raw-
      contrastive 0.862; auto-contrastive 0.933, predictive-contrastive
      0.755, auto-predictive 0.782.
- CAVEATS: single untuned A2C head; Supervised sign-only, zero-cost, no
      buy-and-hold baseline printed (add next); TCN/Transformer + the 4/8/
      16-feature saturation axis + context scan all deferred.

### 8.6 OPEN NEXT STEPS
-------------------------------------------------------------------------------
- Buy-and-hold + |2P-1| baselines in rep_matrix; per-feature-dimension
      saturation run (4/8/16 features) on the predictive encoder - the
      "financial representation saturation" result.
- Idea 5 temporal-leakage shuffles (retrain same model on regime-shuffled /
      globally-shuffled sequences) to test temporal exploitation directly.
- Idea 6 null-feature substitution (same-distribution shuffled / autocorrelated
      / factor-matched nulls) for the temporal-information split.
- Context scan u=10/20/40/60 (retraining) for TACR/D.
- Tune the downstream RL column (annealed lr, entropy schedule, IQL bolt-on)
      before drawing stronger "RL cannot compensate" claims.

------------------------------------------------------------------------------
END OF NOTES
--------------------------------------------------------------------------------