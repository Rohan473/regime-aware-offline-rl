# What determines trading utility in offline financial reinforcement learning? Representation, decision algorithm, and behaviour divergence

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
established (p = .206). Behaviour-policy divergence and regime conditioning
provide a plausible, but descriptive, explanation for why some algorithms
appear more representation-sensitive.

**Decision algorithm is a stronger determinant of downstream utility than
representation identity in the tested setting.** None of the learned policies
exceeds a buy-and-hold baseline (test Sharpe .784); the contribution is an
understanding of the mechanics and evaluation of offline financial RL, not a
profitable trading strategy.

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

**Contributions.**

- A controlled **representation × decision-algorithm** evaluation on daily SPY
  data with feature- and latent-dimension sweeps (10 seeds per cell).
- A **multi-probe** characterisation of what representations encode, including
  nonlinear probes that distinguish linear accessibility from nonlinear
  recoverability.
- A **formal statistical analysis** that separates inferential effects
  (two-way ANOVA) from descriptive decompositions and hypothesis-generating
  correlations.
- Evidence that, in this setting, **algorithm identity dominates
  representation identity**, with behaviour-policy divergence and market regime
  offering a plausible mechanism that we do not claim to have established.

---

## 2. Related work

**Offline RL algorithms.** Behaviour cloning regresses actions on states
(Pomerleau, 1991). Value-based offline RL addresses distributional shift with
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
**same** offline transition dataset (32 logged behaviour policies) with
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
baselines: buy-and-hold, constant-mean action, per-date behaviour-mean action,
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
- **Mechanism (hypothesis-generating)**: behaviour-policy divergence
  (mean |a_policy − a_behaviour|), action-distribution JS divergence,
  divergence–sensitivity association, and regime-conditioned utility.

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

Baselines: **buy-and-hold .784**, constant_mean .784, behaviour_mean .725,
random −.456. **No learned policy exceeds buy-and-hold.**

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

| algorithm | Sharpe range | CV | mean divergence from behaviour |
|---|---|---|---|
| BC | .051 | .035 | .067 |
| IQL | .069 | .054 | .068 |
| CQL | .194 | .120 | .732 |
| A2C | .255 | .218 | .642 |

BC and IQL stay within ≈0.07 of the logged behaviour policy and are nearly
representation-insensitive; A2C and CQL depart by .64–.73 and are 3–5× more
sensitive. Across the four algorithms the association is strong (Pearson
r = .935, Spearman .80), but with n = 4 this is **descriptive only**. At the
seed level the association is weak and inconsistent (pooled r = .13, p = .38).

Regime conditioning (n = 48) shows the divergence–utility relation changes
sign: bull +.47, bear −.60, crisis −.58. Regime-conditioned Sharpe makes the
same point: BC/IQL win defensively (crisis IQL 3.18, BC 2.87 vs A2C .46);
A2C/CQL win in bull (CQL .81, A2C .77 vs BC/IQL ≈.36). Per-year, all
algorithms succeed in 2021/2023/2024 and fail together in 2022 (the bear year),
i.e. degradation is regime-driven, not representation-driven.

Behaviour-policy divergence and regime conditioning provide a plausible
explanation for why some algorithms appear more representation-sensitive, but
this remains a hypothesis.

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

**Level 3 — hypothesis-generating.**
- Behaviour-policy divergence ↔ representation sensitivity (r = .935, n = 4).
- Divergence ↔ regime-dependent utility (bull +.47, bear −.60, crisis −.58).
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
algorithms and remains a hypothesis requiring further investigation.

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

### 5.3 Figure 1 — empirical mechanism hypothesis

```
                    EMPIRICAL OBSERVATIONS
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
                      Research hypothesis
```

Solid arrows are observed associations; dashed arrows (in the manuscript
figure) denote the proposed, untested mechanism.

> **Empirical mechanism hypothesis.** Observed behaviour-policy divergence is
> associated at the algorithm level with representation sensitivity, while
> regime-conditioned analyses suggest that the utility of divergence may depend
> on market regime. These relationships are descriptive and do not establish
> mediation or causality.

---

## 6. Limitations

1. **Single primary financial environment** and limited cross-market transfer
   evidence (SPY → CSI300 transfer is weak and not separable from noise).
2. **Ten seeds** still leave substantial stochastic variance; power for
   interaction effects remains limited.
3. The representation × algorithm **interaction was not statistically
   established**.
4. Divergence analyses are **associational**, not mediation or causal evidence.
5. **No learned policy exceeded buy-and-hold Sharpe = .784**; 2022 is negative
   for all algorithms.
6. **Cross-market CSI300 results** were close to unsolvable under the tested
   protocol, limiting conclusions about representation transfer.
7. **CQL behaviour depends materially on implementation/regularization
   choices**; the main-table CQL is the regularized variant.
8. The representation-quality framework evaluates **selected measurable
   properties** (probe recoverability, rank, CKA, perturbation sensitivity); it
   is **not a complete measure of the "information contained"** in a
   representation.

---

## 7. Conclusion

We separated availability, encoding, and exploitation in offline financial RL
and found that, in the tested setting, decision-algorithm identity is the
robust determinant of downstream utility, while representation identity is
marginal and the representation × algorithm interaction is not statistically
established. Representation diagnostics—probe performance, effective rank,
cross-seed stability—do not provide a reliable one-dimensional proxy for
trading utility. Behaviour-policy divergence and regime conditioning offer a
plausible mechanism for why some algorithms are more representation-sensitive,
but this remains a hypothesis. The findings describe the tested offline
environment and do not demonstrate market-beating performance or a causal
effect of policy divergence.

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

## Appendix B — artifact index

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
- `paper/rep_mechanism_schematic.png` — Figure 1.

## References

- Agarwal, R., Schwarzer, M., Castro, P. S., & Courville, A. (2021). Deep
  reinforcement learning at the edge of the statistical precipice. NeurIPS.
- Alain, G., & Bengio, Y. (2017). Understanding intermediate layers using
  linear classifier probes. ICLR Workshop.
- Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel,
  P., Srinivas, A., & Mordatch, I. (2021). Decision Transformer. NeurIPS.
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework
  for contrastive learning of visual representations. ICML.
- Deng, Y., Bao, F., Kong, Y., Ren, Z., & Dai, Q. (2017). Deep direct
  reinforcement learning for financial signal representation and trading. IEEE
  TNNLS.
- Fujimoto, S., & Gu, S. S. (2021). A minimalist approach to offline
  reinforcement learning. NeurIPS.
- Fujimoto, S., Meger, D., & Precup, D. (2019). Off-policy deep reinforcement
  learning without exploration. ICML.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D.
  (2018). Deep reinforcement learning that matters. AAAI.
- Janner, M., Li, Q., & Levine, S. (2021). Offline reinforcement learning as
  one big sequence modeling problem. NeurIPS.
- Jiang, Z., Xu, D., & Liang, J. (2017). A deep reinforcement learning
  framework for the financial portfolio management problem. arXiv:1706.10059.
- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of
  neural network representations revisited. ICML.
- Kostrikov, I., Nair, A., & Levine, S. (2022). Offline reinforcement learning
  with implicit Q-learning. ICLR.
- Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-learning
  for offline reinforcement learning. NeurIPS.
- Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline reinforcement
  learning: tutorial, review, and perspectives on open problems.
  arXiv:2005.01643.
- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T.,
  Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep
  reinforcement learning. ICML.
- Moody, J., & Saffell, M. (2001). Learning to trade via direct reinforcement.
  IEEE Transactions on Neural Networks.
- Pomerleau, D. A. (1991). Efficient training of artificial neural networks for
  autonomous navigation. Neural Computation.
- Roy, O., & Vetterli, M. (2007). The effective rank: a measure of effective
  dimensionality. EUSIPCO.
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with
  contrastive predictive coding. arXiv:1807.03748.
