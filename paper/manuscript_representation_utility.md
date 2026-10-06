# What determines risk-adjusted utility in offline financial reinforcement learning? Representation, decision algorithm, and exposure

> Drafting note: citations use well-known landmarks; author/year/venue details
> should be verified against the target journal's style during packaging. Every
> empirical statement below is traceable to PROJECT_NOTES §8 and the CSV
> artifacts listed under "Reproducibility". No claim is stronger than the
> frozen evidence hierarchy in §5.1.

---

## Abstract

Offline reinforcement learning (RL) for trading is commonly presented as a
representation-learning problem: better representations are expected to yield
better policies. We separate three questions that are often conflated—what
information is *available* in the input state, what information is *encoded* by
the learned representation, and what information the downstream decision
algorithm actually *exploits*—and study them with a controlled
representation × decision-algorithm experiment on daily S&P 500 (SPY) data.

Three findings emerge. (i) Increasing the observable feature set from 4 to 32
point-in-time-valid features, and the latent dimension from 4 to 128, does not
uncover recoverable short-horizon directional information: direction AUC
remains at chance under linear, gradient-boosted-tree, and MLP probes, and
utility is non-monotone in both axes. (ii) Representation diagnostics—effective
rank, cross-seed CKA stability, reconstruction and probe performance—do not
provide a reliable one-dimensional proxy for downstream trading utility; CKA is
even negatively correlated with utility across the scaling grid. (iii) In a
formal two-way ANOVA over 4 representations × 4 decision algorithms × 10 seeds,
**decision-algorithm identity is the only robust inferential effect**
(F(3,144) = 9.05, p < .0001, η² = .142); representation identity is marginal
(p = .067) and the representation × algorithm interaction is not statistically
established (p = .206). A pre-specified follow-up test shows the utility of
behavior-policy divergence is strongly regime-dependent (divergence × regime
interaction χ²(2) = 80.9, p = 2.7e-18: the divergence slope is near zero in
bull (+0.04) and increasingly negative in bear (−1.04) and crisis (−4.05)),
while divergence alone does not predict utility once algorithm identity is
controlled (p = .446).

**Decision algorithm is a stronger determinant of downstream utility than
representation identity in the tested setting.** The original representation ×
decision-algorithm policies do not exceed a buy-and-hold baseline; subsequent
exposure-only and hybrid policies improve the tested risk–return trade-off. The
hybrid's advantage over buy-and-hold is directionally consistent and seed-robust
on SPY, but is **not statistically established when both training and
finite-test-path uncertainty are considered**: the SPY seed-level interval
excludes zero while the finite-path interval does not, and NIFTY/CSI300 are not
seed-level significant.

**Summary.** Offline financial RL does not need reliable directional prediction
to extract decision-relevant information from learned representations. In the
tested setting, that information is more useful for adaptive exposure and risk
control than for directional trading, while downstream algorithm choice remains
a stronger determinant of utility than representation identity. The
contribution is an understanding of the mechanics and evaluation of offline
financial RL, not a profitable trading strategy.

---

## 1. Introduction

Reinforcement learning is an appealing formalism for sequential portfolio
decisions, and a large literature applies it directly to trading (Moody &
Saffell, 2001; Deng et al., 2017; Jiang et al., 2017). Offline RL (Levine et
al., 2020) is particularly natural for finance, where online exploration is
expensive and risky, and conservative or value-based offline algorithms
(Kumar et al., 2020; Kostrikov et al., 2022) are now standard.

A common implicit assumption in this literature is a *representation
hierarchy*: richer inputs, more capacity, or more stable learned features
should produce better decisions. In practice, three distinct questions are
conflated:

1. **Availability.** Does the input state contain the relevant information?
2. **Encoding.** Does the learned representation preserve it?
3. **Exploitation.** Does the decision algorithm actually use it?

This paper studies the three questions separately and asks what actually
determines observed trading utility. Our setting is deliberately small and
controlled: a shared recurrent encoder over trailing windows of point-in-time
financial features, four representation-learning objectives, and four decision
algorithms trained on the same logged transitions.

**Two stages.** The study has two stages. First, we establish what determines
utility in a controlled representation × decision-algorithm experiment.
Second, motivated by the finding that directional information is not
recoverable while risk-state information remains accessible, we test whether
that information can be used for exposure rather than directional control. The
second stage is treated as a **constructive extension** of the first, not as
evidence that the original representation × algorithm hypothesis was confirmed.

**Contributions.**

1. **Controlled representation–decision analysis.** We disentangle information
   availability, representation encoding, and downstream decision optimization
   in a 4 × 4 × 10 representation × decision-algorithm experiment with fixed
   transitions and evaluation, plus feature- and latent-dimension sweeps.
2. **Representation-information characterization.** We combine linear,
   nonlinear, reconstruction, effective-rank, CKA, and perturbation diagnostics
   to separate information accessibility from downstream trading utility.
3. **Statistical evidence on decision optimization.** We show that
   decision-algorithm identity is the strongest robust determinant of utility
   in the tested setting (F(3,144) = 9.05, p < .0001, η² = .142), while
   representation identity is marginal (p = .067) and the interaction is not
   statistically established (p = .206).
4. **Regime-dependent divergence and exposure mechanism.** We show that the
   association between behavior-policy divergence and utility changes sharply
   across market regimes, and that representation-space risk information can
   instead be exploited through adaptive exposure control.
5. **Risk-aware constructive extension.** We evaluate exposure-only and
   continuous hybrid policies across SPY, CSI300 and NIFTY, with 10-seed
   validation, input ablations, transaction-cost robustness, and temporal
   permutation controls.

---

## 2. Related work

**Offline RL algorithms.** Behavior cloning learns a policy from logged
state–action demonstrations (Bain & Sammut, 1995); the earliest end-to-end
demonstration was Pomerleau (1991). Value-based offline RL addresses distributional shift with
conservatism or in-sample value estimation: BCQ (Fujimoto et al., 2019), CQL
(Kumar et al., 2020), IQL (Kostrikov et al., 2022), and TD3+BC (Fujimoto & Gu,
2021). We use BC, A2C (Mnih et al., 2016), IQL, and CQL on a common offline
transition dataset.

**Representations in RL.** Learned representations are central to deep RL, and
their geometry, collapse, and stability are actively studied. We follow the
probing tradition from representation learning (Alain & Bengio, 2017) and use
CKA (Kornblith et al., 2019) and effective rank (Roy & Vetterli, 2007) as
descriptive diagnostics, and the SimCLR/InfoNCE recipe (Chen et al., 2020;
van den Oord et al., 2018) as one representation objective.

**Financial RL.** Direct RL for trading (Moody & Saffell, 2001; Deng et al.,
2017; Jiang et al., 2017) and sequence-model approaches such as the Decision
Transformer (Chen et al., 2021) and Trajectory Transformer (Janner et al.,
2021) are widely applied to financial state sequences. Our contribution is not
a new algorithm but an analysis of what determines utility in this setting.

**Statistical rigor in RL.** Deep RL results are sensitive to seed noise
(Henderson et al., 2018), motivating careful reporting (Agarwal et al., 2021).
We report bootstrap confidence intervals, a two-way ANOVA with an error term,
and explicitly distinguish descriptive from inferential claims.

---

## 3. Methodology

### 3.1 Data and splits

Daily S&P 500 (SPY) OHLCV, 2005–2024. The canonical state is eight causally
z-scored technical features (1/5/20-day log returns, 20-day realized
volatility, RSI(14), MACD histogram, 20-day volume z-score, Bollinger
position). For scaling we build a **32-feature point-in-time-valid bank** with
nested subsets of size 4/8/12/16/24/32, where the 8-point is exactly the
canonical state. Each feature is constructed only from information available at
or before time t and z-scored with an expanding (no-look-ahead) window; this
establishes temporal validity, **not** causal discovery.

Time splits are fixed: train ≤ 2018, validation 2019–2020, test 2021–2024. All
probes and heads are fit on train, selected on validation, and evaluated on
test; the test set is never used for model selection.

### 3.2 Representations

All representation learners share one encoder (a 128-d GRU over the trailing
20-day window) and differ only in objective:

- **raw** — the flattened 20 × F window (no learning);
- **auto** — reconstruction of the full window from h_t;
- **predictive** — multi-task regression to standardized future targets
  (1-day return, 5-day cumulative return, |1-day return|, 5-day realized vol);
- **contrastive** — InfoNCE between two augmented views of the same window.

Encoders are frozen before any downstream evaluation. Latent dimension is
swept over 4/8/16/32/64/128.

### 3.3 Decision algorithms

Given a frozen representation h_t, we train four decision algorithms on the
**same** offline transition dataset (32 logged behavior policies) with
identical splits, evaluation code, and transaction-cost assumptions:

- **BC** — deterministic policy regressed on logged actions;
- **A2C** — Gaussian actor–critic with bootstrapped value on logged rewards;
- **IQL** — expectile value, target Q, advantage-weighted regression
  (Kostrikov et al., 2022);
- **CQL** — CQL(H) with K = 10 sampled actions and a TD3+BC-style actor
  regularizer (Fujimoto & Gu, 2021). The regularizer is required: unregularized
  CQL(H) collapses in 30–60% of seeds (Appendix A).

Each cell is run with 10 head seeds; the representation seed is held fixed
within a cell.

### 3.4 Probes and diagnostics

Frozen h_t is probed with an identical no-lookahead protocol using three
families: L2 logistic / linear regression (**linear**), histogram
gradient-boosted trees (**GBDT**), and a small MLP. Targets: 1/5/20-day
direction, next-day magnitude, 5/20-day realized volatility, forward 20-day
drawdown, and regime. Diagnostics: reconstruction R², effective and spectral
rank, and linear CKA between representations and across seeds.

### 3.5 Evaluation

Downstream utility is the annualized Sharpe ratio of the deterministic policy
applied to realized market returns on the test split. We also report trivial
baselines: buy-and-hold, constant-mean action, per-date behavior-mean action,
and random actions.

### 3.6 Statistical analysis

- **Formal inference**: two-way ANOVA
  U_{ra} = μ + α_r + β_a + (αβ)_{ra} + ε_{ra} over the 4 × 4 × 10
  observations, with the seed-within-cell residual as the error term; η² effect
  sizes.
- **Descriptive**: decomposition of the 4 × 4 cell-mean matrix (reported as
  "descriptive variance decomposition of cell-mean utility", not variance
  explained).
- **Bootstrap**: 10,000-resample percentile CIs for cell means and for key
  contrasts.
- **Mechanism (pre-specified, one run)**: a primary test of the divergence ×
  regime interaction on regime-specific utility (n = 144, clustered SEs,
  representation and algorithm as controls) and a secondary seed-level test of
  divergence on utility (n = 48, same controls), using behavior-policy
  divergence (mean |a_policy − a_behavior|) and action-distribution JS
  divergence.

---

## 4. Results

### 4.1 Does more information produce more useful representations?

Across the feature sweep (10 seeds), the predictive representation's
supervised Sharpe is non-monotone and **declines past 16 features**
(.826, .751, .770, .856, .657, .481 for 4/8/12/16/24/32); contrastive rises
(.928, .487, .818, .969, 1.005, 1.008); auto is flat (.751–.674–.730).
Across the latent sweep, the predictive Sharpe peaks at 8 dimensions (.969)
and is flat-to-worse at 128 (.751); contrastive declines monotonically to .487.

Critically, **direction AUC remains at chance at every size and dimension**
(.50–.53) and under every probe family. Increasing the observable feature set
does not uncover short-horizon directional information or monotonically improve
utility.

### 4.2 What does a representation encode?

| target | linear | GBDT | MLP |
|---|---|---|---|
| direction_1 (AUC) | .506–.521 | .503–.524 | .499–.509 |
| magnitude_1 (R²) | −.009–.018 | .023–.076 | .022–.079 |
| vol_5 (R²) | .034–.213 | .199–.293 | .046–.226 |
| vol_20 (R²) | −.007–.380 | .247–.288 | −.114–.098 |
| drawdown_20 (R²) | −.34–−.03 | −.19–−.16 | −.57–−.26 |

Direction is at chance under all families. Volatility is **nonlinearly**
encoded: GBDT recovers ≈.25–.29 from learned representations where linear
probes see ≈.03–.05. Forward drawdown is not recoverable by any probe.
Effective rank and cross-seed CKA do not track utility; CKA is negatively
correlated with utility across the scaling grid (r = −.23).

Learned representations differ substantially in what they encode, and no single
diagnostic is a one-dimensional proxy for utility.

### 4.3 Does representation quality translate into trading utility?

Test Sharpe (10 seeds), same frozen representations and transitions:

| rep | A2C | BC | IQL | CQL |
|---|---|---|---|---|
| raw | .362 | .682 | .559 | .574 |
| auto | .464 | .642 | .624 | .768 |
| predictive | .617 | .631 | .628 | .676 |
| contrastive | .527 | .641 | .621 | .709 |

Baselines: **buy-and-hold .784**, constant_mean .784, behavior_mean .725,
random −.456. **No policy in this representation × algorithm matrix exceeds
buy-and-hold** (later exposure-only and hybrid policies improve the
risk–return trade-off; §4.7–4.9).

Two-way ANOVA:

| source | df | F | p | η² |
|---|---|---|---|---|
| representation | 3 | 2.44 | .067 | .038 |
| **algorithm** | 3 | **9.05** | **<.0001** | **.142** |
| rep × algo | 9 | 1.37 | .206 | .065 |
| residual (seed) | 144 | – | – | .755 |

Algorithm identity is the only robust inferential effect; representation is
marginal; the interaction is not established. Bootstrap contrasts confirm the
algorithm gap is significant only on raw/auto (e.g. raw BC−A2C +.319
[.160, .472]); on the predictive representation all algorithms are
statistically equivalent.

Representation diagnostics do not provide a reliable one-dimensional proxy for
downstream utility.

### 4.4 Why does algorithm choice change representation sensitivity?

| algorithm | Sharpe range | CV | mean divergence from behavior |
|---|---|---|---|
| BC | .051 | .035 | .067 |
| IQL | .069 | .054 | .068 |
| CQL | .194 | .120 | .732 |
| A2C | .255 | .218 | .642 |

BC and IQL stay within ≈0.07 of the logged behavior policy and are nearly
representation-insensitive; A2C and CQL depart by .64–.73 and are 3–5× more
sensitive. Across the four algorithms the association is strong (Pearson
r = .935, Spearman .80), but with n = 4 this is **descriptive only**. At the
seed level the association is weak and inconsistent (pooled r = .13, p = .38).

Raw regime correlations (n = 48; not controlling for representation or
algorithm) show the divergence–utility relation changes sign: bull +.47,
bear −.60, crisis −.58. (The pre-specified regression below controls for
representation and algorithm and yields absolute slopes bull +0.04, bear
−1.04, crisis −4.05; the descriptive correlations and the controlled slopes
agree on the regime-dependence but not on the bull sign, which is why the
controlled test is the one reported.) Regime-conditioned Sharpe makes the
same point: BC/IQL win defensively (crisis IQL 3.18, BC 2.87 vs A2C .46);
A2C/CQL win in bull (CQL .81, A2C .77 vs BC/IQL ≈.36). Per-year, all
algorithms succeed in 2021/2023/2024 and fail together in 2022 (the bear year),
i.e. degradation is regime-driven, not representation-driven.

**Pre-specified follow-up test (single run, frozen design).** To test whether
this mechanism survives beyond the n = 4 algorithm-level observation, we fit,
on the existing seed-level data (no new training), two pre-registered models:

- Stage 1 (primary): regime-specific utility on divergence, regime, and their
  interaction, with representation and algorithm as controls and standard
  errors clustered by (rep, algo, seed) — n = 144;
- Stage 2 (secondary): seed-level utility on divergence, controlling for
  representation and algorithm — n = 48.

Stage 1 is **significant** (joint χ²(2) = 80.9, p = 2.7e-18). We find evidence
that the association between behavior-policy divergence and utility is
strongly regime-dependent: divergence is approximately neutral in bull regimes
(absolute slope **+0.04**), negatively associated with utility in bear regimes
(**−1.04**), and most strongly negative in crisis regimes (**−4.05**). Stage 2
is **null**: divergence alone does not predict utility once representation and
algorithm identity are controlled (β = +.135, p = .446; JS-divergence variant
+.264, p = .567), so this result should not be interpreted as a causal effect of
divergence.

Behaviour-policy divergence and regime conditioning provide a plausible
explanation for why some algorithms appear more representation-sensitive. The
granular test confirms the regime-dependence (evidence is consistent with
divergence being a regime-dependent moderator of representation utility) while
showing that divergence alone does not predict utility once algorithm identity
is controlled.

### 4.5 Can the mechanism be exploited? A regime-gated policy

If policy divergence becomes increasingly costly as market conditions
deteriorate, an offline RL policy should learn when to deviate and when to
remain conservative. We test this constructively with a regime-gated policy

a_t = clip(a_beh_t + λ_t · d_t),

where a_beh_t is the behavior-mean action (the conservative anchor), d_t is a
learned deviation, λ_t = σ(g(h_t)) ∈ (0,1) is a learned safety gate, and a
logistic regime head on h_t (train-only) supplies risk_t = P(bear) + P(crisis).
Training is the same offline actor–critic as the matrix, plus an uncertainty
constraint ρ·E[λ_t · risk_t] that pushes the gate down in risky states. We
compare against behavior cloning (λ = 0), unconstrained A2C (full deviation),
an unconstrained adaptive gate (ρ = 0), and a look-ahead oracle gate (λ = 1 in
bull, .1 otherwise), all on the same frozen representations and transitions.

| policy (predictive / contrastive) | test Sharpe, mean (std), 10 seeds |
|---|---|
| behavior clone (λ = 0) | .725 |
| A2C (full deviation) | .617 (.09) / .527 (.39) |
| gated, ρ = 0 | .782 (.23) / .784 (.19) |
| **gated, ρ = 1 (proposed)** | **.757 (.06) / .759 (.06)** |
| oracle (look-ahead) | .758 (.12) / .781 (.55) |

On SPY, the learned gate is regime-dependent, as H4 predicts: mean λ is highest
in bull (.19–.20) and lowest in bear (.09–.13). The uncertainty constraint
protects the risky regimes (crisis Sharpe 3.59 vs A2C 1.56; bear 1.38 vs A2C
.83) at a small cost in bull. The proposed policy beats both fixed extremes
with the lowest seed variance (std ≈ .06 vs .09–.39) and performs comparably to
the look-ahead oracle on the predictive representation (.757 vs .758), while
remaining below it on the contrastive representation (.759 vs .781). (As §4.5's
cross-market note below shows, this regime prior does not transfer unchanged to
other markets.)

**Finding (constructive support for the mechanism).** An offline RL system can
learn when it is safe to deviate from observed behavior and when to stay
conservative: adaptive gating dominates both the conservative (BC) and the
fully-deviating (A2C) endpoints and approaches the oracle. Gains over behavior
cloning are modest and at/below the buy-and-hold baseline.

### 4.6 Why is learned utility below buy-and-hold? A decomposition

Decompose policy return as R_policy = R_direction + R_timing − R_turnover −
R_cost, with the diagnostics in Table below (test 2021–2024, predictive
representation, 3 seeds; all markets).

| policy | Sharpe | exposure | dir_acc | upside capture | downside loss | Δ vs BH |
|---|---|---|---|---|---|---|
| buy-and-hold (SPY/CSI300/NIFTY) | .784 / −.352 / 1.006 | 1.00 | .54 / .48 / .54 | 1.00 | 1.00 | 0 |
| A2C | .692 / −.015 / .718 | .15 / .31 / .40 | .53 / .48 / .51 | .17 / .34 / .44 | .17 / .32 / .44 | −.09 / +.34 / −.29 |
| BC | .631 / −.119 / .571 | .15 / .16 / .12 | .52 / .48 / .47 | .10 / .07 / .10 | .09 / .07 / .10 | −.15 / +.23 / −.44 |
| IQL | .629 / −.099 / .463 | .16 / .17 / .15 | .52 / .50 / .47 | .10 / .07 / .09 | .09 / .07 / .09 | −.16 / +.25 / −.54 |
| CQL | .696 / −.316 / .630 | .77 / .40 / .26 | .53 / .48 / .50 | .78 / .22 / .24 | .78 / .24 / .24 | −.09 / +.04 / −.38 |
| gated (proposed) | .741 / −.381 / .518 | .16 / .17 / .16 | .52 / .51 / .48 | .10 / .05 / .12 | .08 / .06 / .12 | −.04 / −.03 / −.49 |
| oracle | .671 / −.339 / .659 | .31 / .23 / .38 | .53 / .50 / .52 | .29 / .16 / .40 | .27 / .17 / .40 | −.11 / +.01 / −.35 |

Three conclusions:

1. **Direction is at chance in every market and every policy** (dir_acc ≈ the
   market's up-day frequency). There is no predictive skill to exploit.
2. **The loss is a positioning problem, not a prediction or cost problem.**
   Turnover and costs are negligible (0.03–0.07); directional accuracy is
   identical to chance. Every learned policy *underexposes* (0.12–0.77 vs 1.00)
   and varies its position, and with no directional skill time-varying exposure
   adds variance without mean, which lowers Sharpe below buy-and-hold (constant
   exposure is Sharpe-scale-invariant; time-varying exposure is not).
3. **Regime coloring:** the policies' underexposure buys crisis/bear protection
   where buy-and-hold crashes (SPY crisis −1.64 vs learned +3–4) but sacrifices
   bull participation (SPY bull .96 vs learned .35–.84; NIFTY crisis 5.15 vs
   learned ≤3.5). On CSI300 the buy-and-hold itself is negative (−.352), and a
   look-ahead oracle is *also* negative (−.339) — the evaluation period is
   intrinsically hostile, and the decision learner is not the primary bottleneck
   (learned policies actually beat buy-and-hold there by underexposing).

The below-buy-and-hold result is therefore a positioning (timing) loss from
directionless underexposure, not a prediction or transaction-cost failure.

### 4.7 Exposure without direction: can adaptive market participation improve risk-adjusted utility?

If the binding problem is directionless underexposure, the natural control is
an **exposure-only policy**: a_t = e_t (long-only), e_t = σ(g(h_t)) ∈ (0,1),
with the directional component fixed at +1 and no directional prediction made.
We train two variants by maximizing the differentiable Sharpe of e_t·r_t over
the training path, with and without an uncertainty constraint
(ρ·E[e_t·risk_t], risk = P(bear)+P(crisis)); comparators are buy-and-hold and
a constant exposure at the learned mean.

| policy (SPY / CSI300 / NIFTY) | Sharpe | return | maxDD | exposure |
|---|---|---|---|---|
| buy-and-hold | .784 / −.352 / 1.006 | .586 / −.275 / .694 | −.253 / −.456 / −.172 | 1.00 |
| constant at learned mean | .784 / −.352 / 1.006 | .262 / −.133 / .281 | −.125 / −.252 / −.079 | .48 |
| **exposure, max Sharpe** | **.909 / −.355 / 1.033** | .309 / −.135 / .271 | −.114 / −.253 / −.081 | .48 |
| exposure, regime-constrained | .819 / −.357 / .971 | .236 / −.134 / .133 | −.109 / −.251 / −.046 | .43 |

Three findings:

1. **Holding at the learned mean exposure reproduces buy-and-hold Sharpe
   exactly** (Sharpe is scale-invariant under constant exposure). Any Sharpe
   gain must therefore come from *timing*, not from being underexposed.
2. **Under the tested protocol, adaptive exposure improves risk-adjusted
   utility without measurable directional skill.** The policy has no
   directional head, and the exposure is uncorrelated with next-day returns
   (corr(e_t, r_{t+1}) ≈ 0 on both SPY and NIFTY). It beats buy-and-hold on
   Sharpe on both positive-drift markets (SPY .909 vs .784; NIFTY 1.033 vs
   1.006) while roughly halving max drawdown.
3. **The regime-constrained variant is a risk-control knob**: it reduces
   downside exposure at the expense of some risk-adjusted efficiency (SPY .819
   vs .909, maxDD −.109 vs −.114; NIFTY .971 vs 1.033, maxDD −.046 vs −.081) —
   it is not presented as superior, but as the explicit trade-off lever.

**Finding.** An uncertainty- and risk-state-aware exposure policy can improve
the risk/return trade-off without requiring directional skill, beating
buy-and-hold on Sharpe while reducing drawdown on positive-drift markets. This is the
constructive complement to the §4.6 diagnosis: the tested exploitable margin is
in exposure/risk control rather than direction prediction.

**Boundary condition (CSI300).** The exposure policy reproduces buy-and-hold
there (−.355 ≈ −.352): exposure timing can improve the risk/return trade-off
when exploitable positive-drift/risk-timing structure exists, but it cannot
overcome an intrinsically unfavorable evaluation period.

### 4.8 What information drives exposure timing? An attribution

We ask which information inside h_t the exposure decision responds to,
separating risk-management skill from directional skill (SPY, NIFTY; mean
exposure policy over 3 seeds):

| driver | SPY corr(e,·) | SPY OLS β | NIFTY corr(e,·) | NIFTY OLS β |
|---|---|---|---|---|
| next-day return | **+.05** | – | **−.01** | – |
| realized vol (z) | **−.29** | −.040 | **−.34** | −.014 |
| |next-day return| | −.10 | – | −.26 | – |
| forward 5d vol | −.03 | – | −.21 | – |
| OOD distance (Mahalanobis) | −.25 | −.022 | −.33 | −.005 |
| drawdown state | +.03 | −.038 | **+.66** | +.099 |
| regime P(bull) | +.05 | −.005 | +.01 | −.028 |

Two clear patterns:

1. **No directional component.** The exposure is uncorrelated with next-day
   returns (SPY +.05, NIFTY −.01), confirming the policy does not use
   directional information.
2. **Risk timing, not regime labels.** Exposure is reduced ahead of
   high-volatility/risky periods — negative loadings on realized vol,
   |next-day return|, forward 5-day volatility, and OOD distance in both
   markets — while the regime-probability inputs contribute little. Drawdown
   state is used market-specifically (NIFTY increases exposure into drawdowns
   that recover; SPY slightly reduces it).

**Finding (identified pathway).** The exposure improvement comes from a
specific decision-relevant representation pathway — latent risk/volatility
(and drawdown-state) information is useful for exposure control even when
directional information is unavailable. This is the difference between
predictive skill and risk-management skill made operational.

### 4.9 Hybrid exposure: retaining buy-and-hold's upside while keeping risk timing

Since the exposure-only policy beats buy-and-hold on Sharpe but sacrifices
much of its participation (return .31 vs .59 on SPY), we test a continuous
blend a_t = e_t + α_t·(1 − e_t) with α_t ∈ [0,1] (α=0 → adaptive exposure,
α=1 → buy-and-hold), where α_t = σ(W·[risk, ood, vol]) is a three-parameter
head on explicit at-t features only (regime risk, Mahalanobis OOD distance,
realized volatility) — no future information, no threshold, no direction.

| policy (SPY / CSI300 / NIFTY) | Sharpe | return | maxDD | exposure |
|---|---|---|---|---|
| buy-and-hold | .784 / −.352 / 1.006 | .586 / −.275 / .694 | −.253 / −.456 / −.172 | 1.00 |
| adaptive exposure | .909 / −.355 / 1.033 | .309 / −.135 / .271 | −.114 / −.253 / −.081 | .48 |
| constant 50/50 blend | .827 / −.353 / 1.024 | .447 / −.204 / .473 | −.184 / −.360 / −.127 | .74 |
| floor max(e, .5) | .812 / −.350 / 1.053 | .300 / −.135 / .350 | −.134 / −.254 / −.087 | .51 |
| **risk-aware blend (proposed)** | **.890 / −.281 / 1.203** | .439 / −.199 / .661 | −.166 / −.383 / −.105 | .71 |

The proposed blend retains most of buy-and-hold's upside while keeping the
risk-timing benefit: on NIFTY it reaches Sharpe 1.203 with return .661 (vs .694
for buy-and-hold) and better drawdown; on SPY Sharpe .890 with return .439
(vs .586) and better drawdown; on CSI300 it is the best policy (−.281 vs −.352)
by recovering exposure. The mechanism is readable from the three-parameter head:
on SPY α correlates −.93 with volatility — the blend approaches buy-and-hold
when risk is low and reverts toward adaptive exposure when risk is high (α by
regime: bull .53 > bear .23 > crisis .13); on NIFTY α rises in the crisis
rally (.73). A *constant* 50/50 blend does not reproduce this (SPY .827, below
both adaptive and the risk-aware blend), confirming that the at-t risk/OOD/vol
control — not blending per se — is the differentiator.

**Finding.** The risk-aware hybrid improves the risk-adjusted outcome relative
to buy-and-hold across all three tested markets, with the largest improvement
in NIFTY; in SPY it retains a Sharpe advantage while sacrificing some return,
and in CSI300 it reduces losses during an intrinsically unfavorable evaluation
period. The data, not a prior, decides the upper anchor: in calm positive-drift
periods the blend leans toward buy-and-hold, and in high-risk states toward
adaptive exposure.

### 4.10 Validation of the hybrid result

The §4.9 numbers are a 3–5-seed development analysis; we therefore treat the
following 10-seed experiment as the **primary validation** of the hybrid rather
than relying on the earlier point estimate (the exploratory headline NIFTY
value of 1.203 is superseded by the validated 1.074). Validation covers 10
seeds, **two distinct sources of uncertainty** (stochastic training and the
finite test path), transaction-cost robustness, out-of-sample sub-periods, an
input ablation, an OOD-vs-volatility diagnostic, and a temporal permutation
control.

**10-seed Sharpe (mean) and finite-path paired hybrid − buy-and-hold
difference (5,000 date-block resamples over identical test dates):**

| market | BH | adaptive | hybrid | paired Δ (date-block 95% CI) |
|---|---|---|---|---|
| SPY | .784 | .941 | .905 | +.121 [−.059, .294] |
| CSI300 | −.352 | −.350 | −.337 | +.016 [−.020, .053] |
| NIFTY | 1.006 | 1.028 | 1.074 | +.096 [−.044, .234] |

**Training vs test-path uncertainty.** We separate (i) *stochastic training*
uncertainty (the 10 independent seeds, with buy-and-hold a fixed path) from
(ii) *finite test-path* uncertainty (the date-block bootstrap above). Seed-level
paired differences Δ_s = Sharpe_hybrid,s − Sharpe_BH, with a seed bootstrap
(10,000 resamples of the 10 Δ_s), a Wilcoxon signed-rank test, and a sign test:

| market | mean Δ | SD | seed-bootstrap 95% CI | Wilcoxon p | sign (positive) |
|---|---|---|---|---|---|
| SPY | +.121 | .039 | [+.096, +.141] | .002 | 10/10 |
| CSI300 | +.015 | .077 | [−.030, +.062] | .695 | 5/10 |
| NIFTY | +.068 | .138 | [−.013, +.151] | .193 | 6/10 |

On SPY the hybrid's advantage over buy-and-hold is **seed-robust** (all 10
seeds positive; Wilcoxon p = .002), yet the finite-path date-block interval is
wider and includes zero ([−.059, .294]) — the gain is consistent across training
runs but small relative to the noise of a single four-year path. On NIFTY and
CSI300 the seed-level difference is not significant (6/10 and 5/10 positive).
We therefore report the hybrid's advantage as **directionally consistent and
nontrivial in magnitude, but not statistically established on this horizon**;
seed-level intervals quantify training uncertainty, date-block intervals
quantify path uncertainty. Stated precisely: the SPY advantage is robust across
training seeds, but the finite test-path interval includes zero; NIFTY and
CSI300 are not seed-level significant. Thus the hybrid improvement is **not
statistically established when both training and finite-test-path uncertainty
are considered**.

**Input ablation (10 seeds):** OOD distance is the most valuable and robust
at-t input — best on all three markets, and dramatically so on CSI300
(−.124 vs −.337 for the full blend), where including volatility actively hurts
(−.434). Volatility helps on the positive-drift markets (SPY .882, NIFTY 1.021);
regime-risk alone is the weakest input (SPY .848, NIFTY .978, CSI300 −.350).
This refines the §4.8 pathway: representation-distance (OOD) information is the
key exposure-control signal, with volatility timing valuable only where
positive drift exists.

**Is OOD just a volatility proxy?** OOD distance correlates strongly with
contemporaneous volatility (Pearson r = .72 SPY, .84 NIFTY, .78 CSI300 on
train+val), and also with forward 5-day volatility (.55/.67/.52) and drawdown
state (−.67/−.71/−.38); it is **not** orthogonal to volatility. However, the
ablation shows the two inputs behave differently — on CSI300 OOD is the best
input (−.124) while volatility is the worst (−.434) — so OOD appears to carry
decision-relevant information not captured by contemporaneous volatility alone
under the tested ablation, concentrated in large distribution-shift states. We
state this plainly rather than claiming OOD is a distinct signal.

**Permutation control (temporal-alignment check):** shuffling the learned α
across dates (destroying its time alignment, keeping its distribution) reduces
the hybrid Sharpe on SPY (.905 → .848) and NIFTY (1.074 → 1.011) but does not
eliminate it. The at-t timing contributes a modest real increment (~.06 Sharpe);
the remainder comes from the blend structure recovering exposure. This supports
a genuine contribution from temporal alignment; combined with the point-in-time
feature construction, expanding normalization, frozen encoders, and strict
train/validation/test separation, it provides a check against accidental timing
artifacts (it does not by itself prove the absence of look-ahead).

### 4.11 The exposure family on the risk–return plane

We place the full exposure-policy family on the annualized (volatility, return)
plane (5-seed mean; SPY, NIFTY, CSI300). We do **not** claim an efficient
frontier — this is a risk–return trade-off map.

A textbook anchor falls out exactly. Constant exposure c gives R = c·R_BH, so
Sharpe(c·R_BH) = Sharpe(R_BH); the entire constant-exposure family lies on the
buy-and-hold ray (verified: constant 0.1–1.0 all have Sharpe equal to BH). Any
Sharpe change must therefore come from **time-varying** exposure.

| policy | SPY (ret/vol/Sharpe) | NIFTY (ret/vol/Sharpe) |
|---|---|---|
| buy-and-hold (ray) | .129 / .165 / **.784** | .147 / .146 / **1.006** |
| adaptive | .071 / .077 / **.927** | .064 / .062 / **1.039** |
| regime-constrained | .055 / .066 / .822 | .034 / .035 / .981 |
| floor max(e,.5) | .071 / .086 / .819 | .081 / .077 / 1.054 |
| **hybrid** | .096 / .106 / **.900** | .124 / .108 / **1.146** |
| constant sweep | on the BH ray (.784) | on the BH ray (1.006) |

On SPY and NIFTY the adaptive and hybrid points lie **above the buy-and-hold
ray** (higher return per unit risk) and the hybrid is Pareto-non-dominated in
the (volatility, return) plane — on NIFTY it attains the highest Sharpe with a
return close to buy-and-hold and much lower volatility. On CSI300 the
**evaluated risk–return set is degenerate over this test period**: the flat
(zero-exposure) policy Pareto-dominates everything because no policy creates
positive return, so the improvement there reduces to loss reduction.
`paper/risk_return_frontier.png` shows the map.

**Finding.** The adaptive and hybrid policies shift the realized risk–return
trade-off relative to buy-and-hold, with the hybrid providing a favorable
compromise between market participation and exposure reduction. This is a
dynamic risk–return improvement, established on positive-drift markets; it is
not a claim of mean–variance efficiency against the feasible set (an
efficient-frontier statement would require that stronger test).

**Cross-market (CSI300, NIFTY).** The gate consistently learns a
bull-vs-bear/crisis asymmetry in every market (λ higher in bull), but whether
that prior is *correct* is market-specific. On SPY it is right (gating beats
both extremes and performs comparably to the oracle on the predictive
representation); on CSI300 the test period is
unsolvable (all variants negative, a crisis-driven rally); on NIFTY the
bear/crisis penalty is wrong because crisis deviation pays (A2C crisis 2.86 vs
behavior −0.71), so the gate loses to unconstrained A2C (.622 vs .705).
"When it is safe to deviate" is therefore not a universal rule: the gate
reflects the training market's regime structure and the risk prior does not
transfer unchanged.

---

## 5. Discussion

### 5.1 Evidential architecture

We keep three levels of evidence strictly separate throughout.

**Level 1 — established statistically.**
- Algorithm main effect: F(3,144) = 9.05, p < .0001, η² = .142.
- No evidence sufficient to establish the representation × algorithm
  interaction (p = .206).
- Large seed-level variability (residual η² = .755).

**Level 2 — descriptive.**
- 58.0 / 15.6 / 26.4 cell-mean decomposition (algorithm / representation /
  interaction; no error term).
- Representation-sensitivity differences; feature and latent scaling patterns;
  probe results; cross-market observations.

**Level 3 — mechanism hypothesis, partially tested.**
- Behaviour-policy divergence ↔ representation sensitivity (r = .935, n = 4;
  descriptive).
- Divergence ↔ regime-dependent utility: the pre-specified interaction test is
  **significant** (χ²(2) = 80.9, p = 2.7e-18; slope near zero in bull +0.04,
  increasingly negative in bear −1.04 and crisis −4.05), while the seed-level
  divergence main effect, controlling for algorithm identity, is **not**
  (p = .446).
- The proposed mechanism in Figure 1.

**Headline.** Decision algorithm is a stronger determinant of downstream
utility than representation identity in the tested setting. In the two-way
ANOVA, algorithm identity was the only robust inferential effect,
F(3,144) = 9.05, p < .0001, η² = .142; representation identity was marginal
(p = .067), and the representation × algorithm interaction was not
statistically established (p = .206).

**Interpretation.** The results therefore do not support a simple hierarchy in
which increasingly informative or stable representations necessarily produce
better trading policies. Instead, they indicate that downstream decision
optimization is an important determinant of observed utility, while the
apparent dependence of utility on representation varies substantially across
algorithms and remains a hypothesis requiring further investigation. The
regime-gated policy of §4.5 provides constructive evidence for the mechanism:
an offline policy can learn to scale its divergence by a regime estimate,
beating both the conservative (BC) and fully-deviating (A2C) endpoints and
approaching the look-ahead oracle on the predictive representation while
remaining below it on the contrastive representation — while leaving the
utility gain modest and at/below the buy-and-hold baseline.

### 5.2 Seed variability and statistical power

The residual seed term accounts for η² = .755 of the variance, which explains
why the interaction does not reach significance despite a non-trivial
descriptive cell-mean interaction component. In offline financial RL,
random-seed variability can be sufficiently large that apparently meaningful
representation × algorithm differences are difficult to establish with
conventional inferential tests. Ten seeds per cell give repeated estimates of
stochastic variability but may still be insufficient to detect moderate
interaction effects. We do not attempt to rescue the interaction statistically
(no seed selection, no representation cherry-picking, no specification search).

### 5.3 Figure 1 — decision-relevant risk/exposure pathway

```
                    OBSERVED ASSOCIATIONS
                            |
        +-------------------+-------------------+
        v                   v                   v
 Representation      Decision algorithm    Market regime
        |                   |                   |
        |                   v                   |
        |           Policy divergence           |
        |                   |                   |
        +---------+---------+                   |
                  v                             v
          Representation               Regime-dependent
             sensitivity                    utility
                  |                             |
                  +-------------+---------------+
                                v
               Decision-relevant risk/exposure pathway
```

Solid arrows are observed associations; dashed arrows (in the manuscript
figure) denote the mechanism tested constructively in §4.5–4.9: it is
consistent across the tested markets, but the paired Sharpe advantage is not
statistically significant at 5% on the single four-year test period.

> **Tested mechanism (with caveat).** Observed behavior-policy divergence is
> associated at the algorithm level with representation sensitivity; the
> divergence–utility association is strongly regime-dependent; and a
> representation-space risk pathway (OOD/volatility) supports adaptive exposure
> control. These relationships are descriptive/associational (n = 4 for the
> divergence–sensitivity link) and do not establish mediation or causality.

---

## 6. Limitations

1. **Single primary financial environment** and limited cross-market transfer
   evidence (SPY → CSI300 transfer is weak and not separable from noise).
2. **Ten seeds** still leave substantial stochastic variance; power for
   interaction effects remains limited.
3. The representation × algorithm **interaction was not statistically
   established**.
4. Divergence analyses are **associational**, not mediation or causal evidence.
   The regime-interaction survives a granular clustered test, but divergence as
   a seed-level main effect does not (p = .446); the divergence–sensitivity
   association is n = 4.
5. **The original representation × decision-algorithm matrix does not exceed
   buy-and-hold** (Sharpe .784) and 2022 is negative for all algorithms. The
   later exposure-only and hybrid policies improve the tested risk–return
   trade-off, but **their paired Sharpe advantages are not statistically
   significant** on the single four-year test period (all paired bootstrap CIs
   include zero).
6. **Cross-market CSI300 results** were close to unsolvable under the tested
   protocol, limiting conclusions about representation transfer; its exposure
   evaluated risk–return set is degenerate over the test period (flat dominates).
7. **CQL behavior depends materially on implementation/regularization
   choices**; the main-table CQL is the regularized variant.
8. **The regime-gated mechanism is market-specific**: the gate learns a
   bull-vs-bear asymmetry in every tested market, but the sign of divergence's
   effect differs (deviation pays in bull on SPY, in crisis on NIFTY), so the
   "when to deviate" prior does not transfer unchanged.
9. The representation-quality framework evaluates **selected measurable
   properties** (probe recoverability, rank, CKA, perturbation sensitivity); it
   is **not a complete measure of the "information contained"** in a
   representation.
10. **Uncertainty is used in two distinct senses.** Ensemble predictive
    disagreement did not provide a useful *directional* signal; in contrast,
    representation-space **OOD distance** emerged as the strongest at-t input
    in the exposure ablation. These are different constructs and should not be
    conflated.

---

## 7. Conclusion

We separated availability, encoding, and exploitation in offline financial RL
and found that, in the tested setting, decision-algorithm identity is the
robust determinant of downstream utility, while representation identity is
marginal and the representation × algorithm interaction is not statistically
established. Representation diagnostics—probe performance, effective rank,
cross-seed stability—do not provide a reliable one-dimensional proxy for
trading utility. Behaviour-policy divergence and regime conditioning offer a
plausible mechanism for why some algorithms are more representation-sensitive.
A pre-specified granular test supports the regime-dependent component of this
mechanism (divergence × regime interaction, p = 2.7e-18) but not divergence as
a main effect once algorithm identity is controlled (p = .446); the
divergence–sensitivity association itself remains descriptive at the algorithm
level. Turning from prediction to participation, we then showed that when
directional skill is unavailable the exploitable margin is exposure/risk
control: an exposure-only policy improved risk-adjusted utility on
positive-drift markets, and a continuous risk-aware hybrid recovered most of
buy-and-hold's participation while retaining risk timing. In the 10-seed
validation, the hybrid achieved a mean Sharpe of 1.074 on NIFTY versus 1.006
for buy-and-hold, although the finite-test-path interval for the paired
difference included zero. The advantage is directionally consistent across
markets and seeds and survives cost robustness; it is not an efficient-frontier
claim. The strongest
final message is: **offline financial RL does not need reliable directional
prediction to extract decision-relevant information from learned
representations — in the tested setting that information is more useful for
adaptive exposure and risk control than for directional trading, while
downstream algorithm choice remains a stronger determinant of utility than
representation identity.** The findings describe the tested offline environment
and do not demonstrate market-beating performance or a causal effect of policy
divergence.

---

## Appendix A — CQL implementation and collapse

| rep | bc=0 collapse | bc=0 Sharpe | bc=.5 collapse | bc=.5 Sharpe |
|---|---|---|---|---|
| raw | .60 | .568 | .00 | .574 |
| auto | .30 | .256 | .00 | .768 |
| predictive | .40 | .232 | .00 | .676 |
| contrastive | .50 | .291 | .00 | .709 |

CQL(H) with a deterministic actor experienced action collapse (action standard
deviation < .01) in 30–60% of seeds; the TD3+BC actor regularizer eliminated
observed collapse and materially improved performance. This demonstrates
substantial sensitivity of CQL to actor regularization under our implementation
and dataset; it is not a general claim that CQL is unstable.

## Appendix B — development vs validation

To make the development/validation separation explicit (and guard against the
appearance of tuning on the final test), the following decisions were fixed
before the 10-seed validation run:

| component | development (exploratory) | validation (primary, reported) |
|---|---|---|
| representation | predictive / contrastive, 3 seeds | frozen, 1 seed encoder |
| feature & latent sweeps | 10 seeds (separate axis) | not revisited |
| hybrid functional form | a = e + α(1 − e) selected | frozen |
| α inputs | {risk, OOD, volatility} selected | frozen (ablation is diagnostic only) |
| α architecture | 3-parameter linear+sigmoid | frozen |
| seeds for hybrid | 3 (development), 5 (frontier) | **10 new/seeded runs** |
| test period | 2021–2024 | 2021–2024 (unchanged) |
| cost assumption | 0 bp for the map | 0/1/5/10 bp robustness |
| endpoint (oracle) | look-ahead, upper bound only | reported as a bound |

The headline hybrid numbers quoted in the abstract and conclusion are the
10-seed validation values, not the exploratory 3–5-seed values.

## Appendix C — artifact index

All tables are generated by scripts under `scripts/` and written to
`data/interpret/`:

- `rep_scaling.csv`, `rep_scaling_summary.csv` — feature/latent sweeps.
- `rep_offline_rl.csv`, `rep_offline_rl_summary.csv` — representation ×
  algorithm matrix and baselines.
- `rep_rl_dim_sweep.csv` — algorithm × latent-dimension.
- `rep_rl_diagnostics.csv`, `rep_divergence_regime.csv` — divergence, action
  distribution, regime and per-year utility.
- `rep_nonlinear_probe.csv` — linear/GBDT/MLP probes.
- `rep_cross_market.csv` — SPY → CSI300 transfer.
- `rep_bootstrap_ci.csv`, `rep_contrasts.csv`, `rep_anova.csv` — statistics.
- `rep_cql_ablation.csv` — CQL regularization ablation.
- `rep_hybrid_validate.csv`, `rep_hybrid_seedstats.csv` — hybrid 10-seed
  validation and seed-level paired inference.
- `rep_ood_diagnostic.csv` — OOD-vs-volatility/return/drawdown associations.
- `paper/rep_mechanism_schematic.png` — Figure 1.

## References

Reinforcement learning and representation learning:

- Agarwal, R., Schwarzer, M., Castro, P. S., & Courville, A. (2021). Deep
  reinforcement learning at the edge of the statistical precipice. Advances in
  Neural Information Processing Systems, 34.
- Alain, G., & Bengio, Y. (2017). Understanding intermediate layers using
  linear classifier probes. ICLR Workshop.
- Bain, M., & Sammut, C. (1995). A framework for behavioural cloning. Machine
  Intelligence, 15, 103–129.
- Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel,
  P., Srinivas, A., & Mordatch, I. (2021). Decision Transformer. Advances in
  Neural Information Processing Systems, 34.
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework
  for contrastive learning of visual representations. ICML, PMLR 119,
  1597–1607.
- Deng, Y., Bao, F., Kong, Y., Ren, Z., & Dai, Q. (2017). Deep direct
  reinforcement learning for financial signal representation and trading. IEEE
  Transactions on Neural Networks and Learning Systems, 28(3), 653–664.
- Fujimoto, S., & Gu, S. S. (2021). A minimalist approach to offline
  reinforcement learning. Advances in Neural Information Processing Systems, 34.
- Fujimoto, S., Meger, D., & Precup, D. (2019). Off-policy deep reinforcement
  learning without exploration. ICML, PMLR 97, 2052–2062.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D.
  (2018). Deep reinforcement learning that matters. AAAI.
- Janner, M., Li, Q., & Levine, S. (2021). Offline reinforcement learning as
  one big sequence modeling problem. Advances in Neural Information Processing
  Systems, 34.
- Jiang, Z., Xu, D., & Liang, J. (2017). A deep reinforcement learning
  framework for the financial portfolio management problem. arXiv:1706.10059.
- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of
  neural network representations revisited. ICML, PMLR 97, 3519–3529.
- Kostrikov, I., Nair, A., & Levine, S. (2022). Offline reinforcement learning
  with implicit Q-learning. ICLR.
- Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-learning
  for offline reinforcement learning. Advances in Neural Information Processing
  Systems, 33.
- Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline reinforcement
  learning: tutorial, review, and perspectives on open problems.
  arXiv:2005.01643.
- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T.,
  Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep
  reinforcement learning. ICML.
- Moody, J., & Saffell, M. (2001). Learning to trade via direct reinforcement.
  IEEE Transactions on Neural Networks, 12(4), 875–889.
- Pomerleau, D. A. (1991). Efficient training of artificial neural networks for
  autonomous navigation. Neural Computation, 3(1), 88–97.
- Roy, O., & Vetterli, M. (2007). The effective rank: a measure of effective
  dimensionality. EUSIPCO.
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with
  contrastive predictive coding. arXiv:1807.03748.

Quantitative finance, statistics, and regime references:

- Ang, A., & Bekaert, G. (2002). International asset allocation with regime
  shifts. The Review of Financial Studies, 15(4), 1137–1187.
- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio:
  correcting for selection bias, backtest overfitting, and non-normality. The
  Journal of Portfolio Management, 40(5), 94–107.
- Bollen, N. P. B., & Whaley, R. E. (2004). Does net buying pressure affect the
  shape of implied volatility functions? The Journal of Finance, 59(2), 711–753.
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on
  stocks and bonds. Journal of Financial Economics, 33(1), 3–56.
- Hamilton, J. D. (1989). A new approach to the economic analysis of
  nonstationary time series and the business cycle. Econometrica, 57(2),
  357–384.
- Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based
  out-of-distribution detection. Advances in Neural Information Processing
  Systems, 33, 21464–21475.
- Lo, A. W. (2002). The statistics of Sharpe ratios. Financial Analysts
  Journal, 58(4), 36–52.
- Markowitz, H. (1952). Portfolio selection. The Journal of Finance, 7(1),
  77–91.
- Zhang, Z., Zohren, S., & Roberts, S. (2020). Deep reinforcement learning for
  trading. The Journal of Financial Data Science, 2(2), 25–40. (overview;
  positioning)
