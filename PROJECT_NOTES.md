PROJECT NOTES — Regime-Aware Uncertainty Modeling for Offline RL in Financial Markets
=====================================================================================
Working directory: E:\New folder\RL_trading

This file is a complete record of everything done in this project, in order,
with the final verdict stated plainly at the end. Read it top to bottom if
you are new; the last two sections are the ones that matter.

--------------------------------------------------------------------------------
1. PROJECT OVERVIEW
--------------------------------------------------------------------------------

Two phases, specified by the user:

  Phase 1 (complete, verified): a reproducible data pipeline that builds a
  daily OHLCV frame, 8 technical features (z-scored, causal), regime labels
  {bull, bear, crisis}, 4 behavior policies (buy_and_hold, momentum,
  mean_reversion, random) with actions in [-1, 1] (shorting enabled), and an
  offline RL dataset of transitions (state, action, reward, next_state).

  Phase 2 (complete, null result): Model B, a DDR (Differential Sharpe
  Ratio) baseline — a GRU policy trained by direct backprop through the DSR
  reward (Moody & Saffell 1998) on the offline state sequence, evaluated
  per regime on a strict time split.

The single most important outcome of the project is NOT the model. It is the
discovery and root-causing of TWO independent failure modes:

  (1) DATA CONTAMINATION: the daily frame has multi-month missing chunks;
      pct_change() collapsed each into one fake "daily" return. This
      fabricated the entire test-window "crisis" regime (a 2.80 Sharpe
      crisis bucket that did not exist) and a fake catastrophic loss day.
  (2) REWARD HACKING: the DSR reward's variance-reduction bias made the
      policy de-lever to ~1/3 exposure — its entire apparent Sharpe edge
      over buy-and-hold decomposes into position sizing, not signal.

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
      on this data" — SPY only; Model B's feature set (13 baseline
      fields); context u=20 days; tanh-head discretized actions (OUR
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

--------------------------------------------------------------------------------
END OF NOTES
--------------------------------------------------------------------------------