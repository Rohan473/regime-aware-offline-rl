# Representation, decision algorithm, and trading utility in offline financial RL

Draft synthesis of the representation-laboratory thread (PROJECT_NOTES §8).
Written to be lifted into a Results + Discussion section. All numbers on the
SPY daily path, train <= 2018, val 2019-2020, test 2021-2024; 10 seeds per cell
for the representation x algorithm matrix unless noted.

---

## Headline claim (frozen)

> **Decision algorithm is a stronger determinant of downstream utility than
> representation identity in the tested setting.**

Immediately qualified:

> The formal two-way ANOVA identifies a significant algorithm main effect,
> whereas representation identity and the representation x algorithm
> interaction do not reach the conventional 5% significance threshold.

---

## What the experiment establishes vs. what it merely suggests

**Confirmatory result.** Algorithm identity significantly affects downstream
utility, F(3,144) = 9.05, p < .0001, eta^2 = .142.

**Inferential hierarchy.**

| effect | df | F | p | eta^2 | reading |
|---|---|---|---|---|---|
| representation | 3 | 2.44 | .067 | .038 | marginal |
| **algorithm** | 3 | **9.05** | **<.0001** | **.142** | **robust** |
| rep x algo | 9 | 1.37 | .206 | .065 | not established |
| residual (seed) | 144 | - | - | .755 | large |

> The only robust inferential effect in the two-way ANOVA was decision-algorithm
> identity. Representation identity showed a marginal effect, while the
> representation x algorithm interaction was not statistically significant.
> Consequently, we treat the observed relationship between policy divergence,
> representation sensitivity, and regime-dependent utility as an **empirical
> mechanism hypothesis** rather than an established causal or interaction
> effect.

**Separation of evidence types (kept throughout).**

- two-way ANOVA = **formal inference**;
- 4x4 cell-mean decomposition = **descriptive** (58.0% / 15.6% / 26.4% for
  algorithm / representation / interaction, no error term);
- divergence and regime correlations = **mechanism hypothesis**.

---

## Hypothesis status

| Hypothesis | Evidence | Status |
|---|---|---|
| H1: More input information improves utility | feature + latent scaling | **Not supported** |
| H2: Representations encode different information | linear/nonlinear probes, rank, CKA | **Supported descriptively** |
| H3: Representation properties predict utility | 36-cell correlations + diagnostics | **Not supported as a general mapping** |
| H4: Policy divergence mediates representation sensitivity | divergence/sensitivity + regime | **Hypothesis-generating, not established** |
| (confirmatory) Algorithm identity affects utility | two-way ANOVA | **Supported** |

---

## 4.1 Does more information produce more useful representations?

Nested, point-in-time-valid feature bank (4/8/12/16/24/32; the 8-point is the
production state) and latent dimension (4/8/16/32/64/128), 10 seeds.

- No monotonic improvement: the predictive representation's supervised Sharpe
  declines past 16 features (.856 -> .481 at 32) and is flat-to-worse at 128
  latent dims; contrastive declines monotonically with latent dim.
- Directional information is never recoverable: direction AUC ~.50-.53 at
  every feature count and latent dimension, and under every probe family.

**Finding.** Increasing the observable feature set does not uncover
short-horizon directional information or monotonically improve utility.

---

## 4.2 What does a representation encode?

| target | linear | GBDT | MLP |
|---|---|---|---|
| direction_1 (AUC) | .506-.521 | .503-.524 | .499-.509 |
| magnitude_1 (R2) | -.009..+.018 | .023-.076 | .022-.079 |
| vol_5 (R2) | .034-.213 | .199-.293 | .046-.226 |
| vol_20 (R2) | -.007..+.380 | .247-.288 | -.114..+.098 |
| drawdown_20 (R2) | -.34..-.03 | -.19..-.16 | -.57..-.26 |

- Direction is at chance under all probe families (not recoverable by the
  tested probes).
- Volatility is nonlinearly encoded: GBDT recovers ~.25-.29 from learned
  representations where linear probes see ~.03-.05.
- Forward 20-day drawdown is not recoverable by any probe.
- Effective rank and cross-seed CKA do not track downstream Sharpe (CKA
  correlates negatively, r = -.23, across the 36 scaling cells).

**Finding.** Learned representations differ substantially in what they encode,
and no single diagnostic is a one-dimensional proxy for utility.

---

## 4.3 Does representation quality translate into trading utility?

| rep | A2C | BC | IQL | CQL |
|---|---|---|---|---|
| raw | .362 | .682 | .559 | .574 |
| auto | .464 | .642 | .624 | .768 |
| predictive | .617 | .631 | .628 | .676 |
| contrastive | .527 | .641 | .621 | .709 |

Trivial baselines: **buy-and-hold .784**, constant_mean .784, behavior_mean
.725, random -.456. **None of the learned decision policies exceeds the
buy-and-hold baseline (.784) in aggregate test Sharpe.**

Bootstrap contrasts show the algorithm gap is significant only on the raw and
auto representations (e.g. raw BC-A2C +.319 [.160, .472]); on the predictive
representation all algorithms are statistically equivalent. This is consistent
with the representation mediating the size of the algorithm gap, but the
interaction itself is not significant.

**Finding.** Representation diagnostics such as effective rank, CKA, and
predictive probe performance do not provide a reliable one-dimensional proxy
for downstream trading utility.

---

## 4.4 Why does algorithm choice change representation sensitivity?

| algorithm | Sharpe range | CV | mean divergence from behavior |
|---|---|---|---|
| BC | .051 | .035 | .067 |
| IQL | .069 | .054 | .068 |
| CQL | .194 | .120 | .732 |
| A2C | .255 | .218 | .642 |

BC and IQL stay within ~0.07 of the logged behavior policy and are nearly
representation-insensitive; A2C and CQL depart by .64-.73 and are 3-5x more
sensitive. Across the four algorithms the association is strong (Pearson
r = .935, Spearman .80); with n = 4 this is **descriptive only**.

Regime conditioning (n = 48) shows the divergence-utility relation changes
sign: bull +.47, bear -.60, crisis -.58. Regime-conditioned Sharpe makes the
same point: BC/IQL win defensively (crisis IQL 3.18, BC 2.87 vs A2C .46),
A2C/CQL win in bull (CQL .81, A2C .77 vs BC/IQL ~.36).

At the seed level the divergence-utility association is weak and inconsistent
in sign (pooled r = .13, p = .38; per-algorithm CQL +.96, IQL -.76), so the
mechanism is not established as causal.

**Finding (hypothesis).** Behavior-policy divergence and regime conditioning
provide a plausible explanation for why some algorithms appear more
representation-sensitive, but this remains a hypothesis requiring larger and
more controlled experiments.

---

## Figure 1 - empirical observations and the proposed mechanism

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

Solid arrows denote observed associations; dashed arrows in the manuscript
denote the proposed (untested) mechanism.

> **Empirical mechanism hypothesis.** Observed behavior-policy divergence is
> associated at the algorithm level with representation sensitivity, while
> regime-conditioned analyses suggest that the utility of divergence may
> depend on market regime. These relationships are descriptive and do not
> establish mediation or causality.

---

## Discussion - seed variability and statistical power

The residual (seed-within-cell) term accounts for eta^2 = .755 of the variance.
This explains why the interaction does not reach significance despite a
non-trivial descriptive cell-mean interaction component, and yields a
methodological observation:

> In offline financial RL, random-seed variability can be sufficiently large
> that apparently meaningful representation x algorithm differences are
> difficult to establish with conventional inferential tests.

Ten seeds per cell provide repeated estimates of stochastic variability but may
still be insufficient to detect moderate interaction effects given the high
between-seed variance. We do not attempt to rescue the interaction
statistically (no seed selection, no representation cherry-picking, no
alternative specifications).

---

## Limitations (front and centre)

- **No market-beating result.** Buy-and-hold (.784) exceeds every learned
  policy; 2022 is negative for all algorithms. The contribution is
  understanding the mechanics and evaluation of offline financial RL, not a
  profitable trading strategy.
- **No causality.** Divergence -> sensitivity is n = 4 (algorithms); the
  seed-level association is weak and inconsistent (r = .13, p = .38). Regime
  correlations (n = 48) are descriptive.
- **Interaction not established** (p = .21); seed noise dominates (eta^2 =
  .755).
- **Scope.** One encoder family (GRU), SPY + US macro features, one offline
  transition dataset; cross-market transfer (SPY -> CSI300) is weak and not
  separable from noise.

---

## Appendix - CQL implementation and collapse

CQL(H) with a deterministic actor experienced action collapse (action
standard deviation < .01) in 30-60% of seeds; adding the TD3+BC-style actor
term `bc_coef * MSE(pi(h), a_logged)` (bc_coef = .5) eliminated observed
collapse and materially improved performance, most strikingly for the auto
representation (.256 -> .768).

| rep | bc=0 collapse | bc=0 Sharpe | bc=.5 collapse | bc=.5 Sharpe |
|---|---|---|---|---|
| raw | .60 | .568 | .00 | .574 |
| auto | .30 | .256 | .00 | .768 |
| predictive | .40 | .232 | .00 | .676 |
| contrastive | .50 | .291 | .00 | .709 |

This demonstrates substantial sensitivity of CQL to actor regularization under
our implementation and dataset; it is not a general claim that CQL is unstable.
The main-table CQL uses bc_coef = .5.
