# Representation, decision algorithm, and trading utility in offline financial RL

Draft synthesis of the representation-laboratory thread (PROJECT_NOTES §8).
This document is written to be lifted into a Results + Discussion section; all
numbers are on the SPY daily path, train <= 2018, val 2019-2020, test 2021-2024,
seed 20260814 unless noted.

## Contribution

> In offline financial RL, representation properties and trading utility are
> **not interchangeable**: downstream decision algorithms account for the
> largest share of observed utility variation, and algorithms that depart more
> strongly from logged behavior become substantially more sensitive to the
> learned representation, with the utility of that divergence varying across
> market regimes.

The paper separates three questions that are commonly conflated:

1. **What information is available** in the input state?
2. **What information is encoded** by the learned representation?
3. **What information does the decision algorithm actually exploit?**

The contribution is explanatory. It is **not** a claim of market-beating
performance, and none of the learned policies beats buy-and-hold.

---

## 4.1 Does more information produce more useful representations?

We sweep a nested, point-in-time-valid feature bank (4/8/12/16/24/32 features;
the 8-point is exactly the production state) and the latent dimension
(4/8/16/32/64/128), 10 seeds each, and measure the supervised direction policy.

- **No monotonic improvement.** The predictive representation's supervised
  Sharpe *declines* past 16 features (.856 -> .481 at 32) and is flat-to-worse
  at 128 latent dims; contrastive declines monotonically with latent dim.
- **Directional information is never recoverable.** Direction AUC is ~.50-.53
  at every feature count and every latent dimension, and under linear, GBDT,
  and MLP probes (below).

**Finding.** More information does not buy directional edge; extra capacity can
hurt. Information is not the bottleneck.

---

## 4.2 What does a representation actually encode?

Frozen `h_t` probed with an identical no-lookahead protocol.

| target | linear | GBDT | MLP |
|---|---|---|---|
| direction_1 (AUC) | .506-.521 | .503-.524 | .499-.509 |
| magnitude_1 (R2) | -.009..+.018 | .023-.076 | .022-.079 |
| vol_5 (R2) | .034-.213 | .199-.293 | .046-.226 |
| vol_20 (R2) | -.007..+.380 | .247-.288 | -.114..+.098 |
| drawdown_20 (R2) | -.34..-.03 | -.19..-.16 | -.57..-.26 |

- Direction is at chance under **all** probe families: not recoverable by the
  tested probes (weaker claim than "not present").
- Volatility is **nonlinearly** encoded: GBDT recovers ~.25-.29 from learned
  representations where linear probes see ~.03-.05. The earlier "vol collapse"
  reading was partly a linear-accessibility artifact.
- Forward 20-day drawdown is **not recoverable by any probe** (a clean null).
- Representation diagnostics dissociate: effective rank and cross-seed CKA do
  not track downstream Sharpe (cross-seed CKA even correlates negatively,
  r = -.23, across the 36 scaling cells).

**Finding.** Representation properties dissociate; no single diagnostic is a
proxy for utility.

---

## 4.3 Does representation quality translate into trading utility?

Frozen representations x four decision algorithms (BC, A2C, IQL, CQL), 10 head
seeds, on the same Phase-1 transitions and evaluation.

| rep | A2C | BC | IQL | CQL |
|---|---|---|---|---|
| raw | .362 | .682 | .559 | .574 |
| auto | .464 | .642 | .624 | .768 |
| predictive | .617 | .631 | .628 | .676 |
| contrastive | .527 | .641 | .621 | .709 |

Trivial baselines: **buy-and-hold .784**, constant_mean .784, behavior_mean
.725, random -.456. **No learned policy exceeds buy-and-hold.**

Two-way ANOVA on the full 4x4x10 observations (typ=2):

| source | df | F | p | eta^2 |
|---|---|---|---|---|
| representation | 3 | 2.44 | .067 | .038 |
| **algorithm** | 3 | **9.05** | **<.0001** | **.142** |
| rep x algo | 9 | 1.37 | .206 | .065 |
| residual (seed) | 144 | - | - | .755 |

The **algorithm main effect is the only robust inferential result**; the
representation effect is marginal; the interaction is descriptive (the
cell-mean decomposition attributes 58.0% / 15.6% / 26.4% to
algorithm / representation / interaction, but this carries no error term).
Bootstrap contrasts confirm the algorithm gap is significant only on the raw
and auto representations (e.g. raw BC-A2C +.319 [.160,.472]); on the predictive
representation all algorithms are statistically equivalent.

**Finding.** There is no simple representation -> utility mapping. Algorithm
identity explains the largest share; the representation x algorithm
interaction is suggestive but not established.

---

## 4.4 Why does algorithm choice change representation sensitivity?

Per-algorithm representation sensitivity and behavior divergence:

| algorithm | Sharpe range | CV | mean divergence from behavior |
|---|---|---|---|
| BC | .051 | .035 | .067 |
| IQL | .069 | .054 | .068 |
| CQL | .194 | .120 | .732 |
| A2C | .255 | .218 | .642 |

**Behavior divergence stratifies the algorithms into two regimes.** BC and IQL
stay within ~0.07 of the logged behavior policy and are nearly
representation-insensitive (range .05-.07); A2C and CQL depart by .64-.73 and
are 3-5x more sensitive (range .19-.26). Across the four algorithms the
association is strong (Pearson r = .935, Spearman .80); with n = 4 this is
**descriptive only**.

Regime conditioning shows the divergence-utility relation changes sign
(n = 48): bull **+.47**, bear **-.60**, crisis **-.58**. Regime-conditioned
Sharpe makes the same point: BC/IQL win defensively (crisis IQL 3.18, BC 2.87
vs A2C .46), A2C/CQL win in bull (CQL .81, A2C .77 vs BC/IQL ~.36).

**Finding (hypothesis).** The degree of departure from behavior is associated
with representation sensitivity, and the value of that departure is
regime-dependent.

---

## Central figure - empirical mechanism hypothesis

```
Representation
     |  changes what information is available
     v
Decision algorithm
     |  determines degree of departure from behavior
     v
Policy divergence
     |  associated with
     v
Representation sensitivity
     |  varies across
     v
Market regime
     v
Observed trading utility
```

Arrows are associations, not identified causal effects.

---

## Limitations (front and centre)

- **No market-beating result.** Buy-and-hold (.784) exceeds every learned
  policy; 2022 is negative for all algorithms. The paper explains behaviour, it
  does not propose a profitable strategy.
- **No causality.** The divergence -> sensitivity association is n = 4
  (algorithms); the seed-level divergence-utility association is weak and
  inconsistent in sign (pooled r = .13, p = .38). The regime correlations
  (n = 48) are the more robust but still descriptive evidence.
- **Interaction is not significant** (p = .21); seed noise dominates (eta^2 =
  .755). Treat "algorithm mediation" as a significant main effect and the
  interaction as a hypothesis.
- **Scope.** One encoder family (GRU), SPY + US macro features, one offline
  transition dataset; cross-market transfer (SPY -> CSI300) is weak and not
  separable from noise.

---

## Appendix - CQL implementation and collapse

CQL(H) with a deterministic actor can saturate to a constant position. With no
actor regularization, 30-60% of seeds collapse (action std < .01); adding the
TD3+BC-style actor term `bc_coef * MSE(pi(h), a_logged)` (bc_coef = .5) removes
collapse entirely and lifts test Sharpe.

| rep | bc=0 collapse | bc=0 Sharpe | bc=.5 collapse | bc=.5 Sharpe |
|---|---|---|---|---|
| raw | .60 | .568 | .00 | .574 |
| auto | .30 | .256 | .00 | .768 |
| predictive | .40 | .232 | .00 | .676 |
| contrastive | .50 | .291 | .00 | .709 |

The main-table CQL uses bc_coef = .5; CQL's behaviour is therefore not evidence
about conservative Q-learning per se but about a regularized variant.
