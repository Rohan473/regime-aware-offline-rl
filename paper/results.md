# 4. Results

> Framing title (recommended): *An empirical investigation of
> direction–magnitude decomposition in offline financial reinforcement
> learning.*
>
> The paper deliberately does NOT claim "decomposing direction from magnitude
> is universally superior." Its contribution is a systematic falsification /
> decomposition study: which of the two decision components (directional
> probability vs. learned magnitude) actually survives net-of-cost
> evaluation on real data.

## 4.1 Experimental protocol and baseline

We evaluate whether an offline Transformer reinforcement-learning policy can
simultaneously learn the **direction** and **magnitude** of daily market
exposure. The experiments progressively decompose the trading decision into
these two components and test whether the learned magnitude policy provides
incremental value over a supervised directional probability and simple
deterministic sizing rules.

All primary comparisons use the corrected data pipeline and a **1 bp
transaction-cost assumption**. Results at alternative transaction-cost levels
(0.5–10 bp) are used as sensitivity checks. Model selection is performed
without access to the final evaluation period (2019–2020 validation; the
2021–2024 window and the 2025–2026 out-of-window period are never used for
selection or early stopping).

Statistical inference. For each sizing rule we reconstruct the net-of-cost
daily return series directly from the day-level test tables using the exact
recording convention (model return $a_t m_t - c\,|a_t - a_{t-1}|$; equal-
magnitude EM $|a_t| m_t$ with its own turnover cost; annualized Sharpe
$\bar r/\sigma_r\sqrt{252}$). Reconstructed Sharpe/margin values
cross-validate against the tables below. For every rule we report:

- the **Newey–West heteroskedasticity-and-autocorrelation-consistent (HAC)
  $t$-statistic** on the mean of the daily (model $-$ EM) margin series
  (null hypothesis: zero mean margin), with lag $=8$;
- a **moving-block-bootstrap 95% confidence interval** (block length 21
  trading days, 5,000 resamples; sensitivity at block length 5 reported in
  the reproducibility appendix) for the same mean daily margin, in basis
  points/day;
- the paired-block bootstrap interval for the Sharpe-space margin.

Because each rule's margin is measured against its own equal-magnitude
long-only EM, margins are scale-invariant across rules. The 2021–2024 window
was used throughout development, so the intervals on that window are
descriptive (reported as such); the pre-registered, single-report confirmatory
evidence lives in the two out-of-window arms (§ 4.8), which are never
re-examined here.

An important implementation correction was made before the final experiments.
The initial experimental pipeline inadvertently operated on CSI300 data while
the SPY experiments were being interpreted as SPY experiments, and two
timezone / index-alignment bugs affected the macro features and the
directional predictions. These issues were corrected, the SPY data were
restored, and all affected models were retrained and reevaluated. Contaminated
runs are excluded from the analysis below; full details of the corruption and
the correction are recorded in the Experimental-Integrity appendix
(§ 4.11) and in PROJECT_NOTES entries 7.31–7.35.

## 4.2 End-to-end offline RL does not recover the directional component

The first experiment asks whether the failure of the Transformer actor to
learn useful direction is simply an information-set limitation.

We therefore widen the TACR state from the original 8-dimensional
representation to a 16-dimensional state incorporating the macro information
available to the supervised directional model. The two 8-feature blocks are
causally z-scored price technicals (returns at 1/5/20 days, realized vol, RSI,
MACD, volume z-score, Bollinger position) and causal cross-asset features
(risk-on spread vs. TLT, 10y-yield deltas, vol term structure, SPY–DXY return
correlation, credit spread proxy, relative strength vs. QQQ and IWM). Splits,
regularization, and selection are identical to the canonical 8-dim run.

The result is negative. At 1 bp, the 16-dimensional model attains a net margin
of −0.050 (Sharpe 0.746 vs. its EM control 0.796), essentially unchanged from
the 8-dimensional model's −0.058 (0.772 vs. 0.830) — the expansion does not
recover a positive directional edge. Thus, adding the macro information
available to the directional model does not cause the offline RL objective to
recover the missing directional signal. This provides evidence against the
hypothesis that the earlier directional failure was primarily caused by an
insufficient information set.

The small improvement in net margin (−0.058 → −0.050) is confined to the
magnitude/exposure channel; it does not close the gap to the supervised
directional baseline and, as shown in § 4.4, does not represent an
incremental RL-sizing advantage.

## 4.3 Decomposing direction from magnitude

We next separate the trading decision into

1. a supervised model estimating the probability of an upward next-day return,
   $P_t$, and
2. a magnitude / sizing component.

The canonical C+ construction uses the supervised directional probability
(for the sign) together with the magnitude generated by the offline RL model
($|a_t^{\text{RL}}|$). We evaluate this construction against deterministic
transformations of $P_t$ alone, isolating the marginal contribution of each
component.

The corrected SPY experiments substantially change the interpretation of this
construction. The frozen single-logistic directional model does **not** itself
generate a positive aggregate trading margin on SPY: sign-only exposure
($a_t=\operatorname{sign}(P_t-0.5)$) is approximately null to negative net of
costs (margin(1 bp) ≈ −0.06; the fully-costed canonical C+ is negative at
every cost level from 0 to 20 bp). This indicates that the useful component
of the strategy cannot be described as a persistent unconditional directional
advantage.

A bootstrap ensemble of the same logistic sign models *appears* to restore a
positive margin (+0.228 at 1 bp) and even to beat the best single member.
This impression does not survive scrutiny: across ten independently seeded
resamplings of the same protocol the ensemble margin has mean ≈ −0.02 and is
positive in only 50% of draws — the apparent edge is carried by one or two
lucky bootstrap members and disappears under reseeding. The ensemble therefore
does not rescue the directional channel. (Full member-level breakdown:
member margins range −0.73 to +0.58, mean −0.07; the ensemble never beats the
best member.) The sign-head ablation (§ 4.11.2) is retained as a methodological
appendix item.

Instead, the subsequent sizing analysis isolates the contribution of
**confidence-weighted exposure** (§ 4.4–4.5).

## 4.4 Simple probability-based sizing outperforms learned RL magnitude

We compare the TACR magnitude policy against deterministic transformations of
the supervised directional probability. The principal probability-based
sizing rule is

$$
a_t = \operatorname{sign}(2P_t-1)\,|2P_t-1|,
$$

with the corresponding volatility-normalized variant $|2P_t-1|/\mathrm{vol}_{t}$
evaluated separately.

On corrected SPY data, the learned TACR magnitude policies do **not** provide
incremental value over the simple probability-based rule. Net-of-cost margins
at 1 bp, each compared against its own EM control (so the margin is
scale-invariant):

| Sizing rule                     | Sharpe(1 bp) | EM(1 bp) | Margin(1 bp) | Turnover |
| ------------------------------- | -----------: | -------: | -----------: | -------: |
| TACR-8 magnitude (C+8, RL $|a|$)| 0.772        | 0.830    | −0.058       | 0.147    |
| TACR-16 magnitude (C+16, RL)    | 0.746        | 0.796    | −0.050       | —        |
| $\|2P-1\|$ (frozen canonical P) | 0.86         | 0.81     | **+0.051**   | 0.048    |
| $\|2P-1\|/\mathrm{vol}_{20d}$   | 0.96         | 0.98     | −0.025       | 0.343    |
| $1/\mathrm{vol}_{20d}$ (no P, control) | 0.83  | —        | −0.190       | 1.442    |

Two features stand out. First, both RL-magnitude constructions are negative
net of costs, while the confidence-scaled rule is positive. Second, the
".../vol" and inverse-volatility rows are **negative**: volatility
normalization at the index level does not add value on SPY.

The surviving claim is deliberately narrow, because the robustness analysis
(identical protocol, ten independent re-seeded probability heads) is
decisive:

- **$\|2P-1\|$ is the only sizing component that remains robust under
  reseeding.** Its 1-bp margin is positive in **10/10** bootstrap draws, with
  mean ≈ **+0.059** and range ≈ **+0.002 to +0.123**. The frozen canonical
  (deterministic) probability produces a margin of ≈ **+0.051**.
- **Volatility normalization does not survive.** The
  $\|2P-1\|/\mathrm{vol}$ rule is positive in only **30%** of draws, with mean
  margin ≈ **−0.022**. Pure inverse-volatility sizing is also negative.

It is essential to state what this does and does not establish. The
$\|2P-1\|$ margin on SPY is **positive in sign and consistent across
reseedings, but it is not individually statistically significant**: the
Newey–West HAC $t$-statistic on the daily model-vs-EM margin is $t=0.41$, and
the block-bootstrap 95% interval on the daily margin is
$[-0.08,+0.24]$ bp/day, containing zero. The magnitude is small
($+0.035$ bp/day, about 0.09% annualized against the EM), so the aggregate
SPY effect is economically and statistically indistinguishable from zero over
2021–2024. The same is true of the ensemble reference (§ 4.3), whose margin
$+0.118$ also carries $t=0.76$ with a bootstrap interval
$[-0.075,+0.319]$ bp/day that includes zero. Consequently, on SPY the edge
must be characterized as **seed-stable in sign but statistically
inconclusive**; it is not evidence of a positive aggregate Sharpe on SPY.

Consequently, the results do **not** support a claim that volatility-aware
sizing is responsible for the observed edge. The seed-stable component is the
event-scaled confidence $\|2P-1\|$ itself.

## 4.5 The surviving SPY mechanism is confidence scaling, not return-magnitude prediction

Further diagnostics clarify what the probability signal is actually doing.

The correlation between confidence, $|2P_t-1|$, and absolute next-period
return is only

$$
\operatorname{corr}\!\big(|2P_t-1|,\ |R_{t+1}|\big) \approx 0.027,
$$

and $\operatorname{corr}(P_t, |R_{t+1}|) \approx 0.004$. The probability model
is therefore **not** predicting the magnitude of the next return.

The incremental performance is instead concentrated in a small number of
high-confidence short positions. The short side represents ≈ 7.6% of the test
observations, and the entire positive contribution of the ‖2P−1‖ rule
disappears when the short days are excluded (margin → 0.00). Within the long
side, the edge is concentrated in the highest-confidence tercile
($|2P-1| > 0.11$), whose margin is approximately **+0.150**, while the two
lower-confidence terciles are negative (−0.033, −0.144).

This suggests that the economic role of $P_t$ is better interpreted as
**directional confidence** — the model says which (rare) days it is willing
to bet against the index at — rather than as a forecast of return magnitude.
This distinction matters for interpreting the failure of the RL magnitude
head: the magnitude model is attempting to learn a quantity for which the
supervised probability signal itself provides little evidence of predictable
return magnitude.

The CSI300 daily margin series displays the same signature at higher
amplitude: the $\|2P-1\|$ daily model-vs-EM margin has mean $+1.85$ bp/day,
median $0$, standard deviation $17.1$ bp/day, skewness $+2.5$ and excess
kurtosis $+29$ — a heavily right-tailed, zero-median distribution whose
aggregate is carried by rare large wins, consistent with an event-driven
short-leg mechanism rather than a smooth daily edge. The NIFTY series shows the
same shape at still higher amplitude (mean $+4.40$ bp/day, standard deviation
$30.2$, skewness $+5.7$, excess kurtosis $+77$; § 4.7), reinforcing the
event-driven interpretation across markets.

## 4.6 Uncertainty does not provide a useful risk signal

We investigate whether ensemble disagreement can identify observations on
which the directional model is likely to fail.

On the corrected SPY data it cannot. The relationships are not statistically
or economically meaningful: the correlation between ensemble disagreement
(variance of member probabilities, $U_t$) and the sign error is
$\operatorname{corr}(U,\text{err}) \approx +0.012$, and the correlation
between disagreement and the C+ daily P&L is
$\operatorname{corr}(U,\text{P&L}) \approx +0.039$ — both approximately zero
(note: an earlier *apparent* inversion, in which higher disagreement coincided
with *fewer* errors, was specific to the contaminated pre-correction data and
does not recur on the corrected pipeline; see § 4.11).

Higher disagreement therefore does not identify adverse outcomes, whether
measured as sign errors or as realized C+ P&L. The experiments do not support
using ensemble disagreement as

- a trading filter,
- an error detector, or
- a risk-sizing variable.

We consequently exclude uncertainty-aware filtering and uncertainty-based
sizing from the final trading mechanism.

## 4.7 Cross-market validation on CSI300 and NIFTY

The most important validation experiments evaluate whether the mechanism
observed on SPY is specific to that market. The frozen protocol is
re-estimated independently on two further markets, CSI300 and NIFTY 50.

The canonical directional protocol is **independently re-estimated** on CSI300
using the same frozen methodology: identical feature construction (with the
CSI300 daily return as the "instrument" side of the cross-asset spreads),
identical 16-feature causal z-score build, identical split bounds, and the
same single-logistic P head C-selected on validation. No SPY parameters are
carried over (direct weight transfer is tested separately in § 4.9).

At 1 bp, the contrast across the three markets is sharp (full NIFTY detail
follows below):

| Sizing rule                | SPY margin | CSI300 margin | CSI300 t_NW | NIFTY margin | NIFTY t_NW |
| -------------------------: | ---------: | ------------: | -----------: | -----------: | ---------: |
| Sign-only $|a|=1$          | −0.06      | **+1.40**     | 2.07         | **+2.13**    | **2.93**   |
| $\|2P-1\|$ sizing          | +0.05      | **+1.58**     | **2.80**     | **+3.11**    | **3.87**   |
| $\|2P-1\|/\mathrm{vol}$    | −0.02      | **+1.87**     | **2.84**     | **+3.10**    | **4.53**   |

The confidence-scaled strategy is robust across bootstrap draws on CSI300:
10/10 reseeded probability heads give a positive margin, mean ≈ **+1.77**
(range ≈ +1.58 to +1.94). Unlike SPY, the volatility-normalized rule is *also*
positive and robust here (mean ≈ +2.07; 10/10).

Critically, and in contrast to SPY, the aggregate CSI300 effect is
**statistically significant for the probability-driven rules**. For
$\|2P-1\|$ the Newey–West HAC $t$ is $2.80$ and the block-bootstrap 95%
interval on the daily model-vs-EM margin is $[+0.51,+3.58]$ bp/day,
excluding zero; for $\|2P-1\|/\mathrm{vol}$ the corresponding figures are
$t=2.84$ and $[+2.44,+19.4]$ bp/day. Sign-only exposure ($t=2.07$) and
$1/\mathrm{vol}$ ($t=2.09$) have HAC $t$-values above two but bootstrap
intervals containing zero, driven by heavier right tails and higher turnover.
Thus CSI300 provides statistically significant evidence for the strategy over
the aggregate evaluation period — unlike SPY.

This significance, however, is **not uniform across time or regime**, and the
nonuniformity is a property of the result, not a noise artifact:

| Year | $n$ | Up-rate | Drift (cum. ret.) | Sign-only | $\|2P-1\|$ | $\|2P-1\|/\mathrm{vol}$ |
| ---: | --: | ------: | ----------------: | --------: | ---------: | ----------------------: |
| 2021 | 221 | 0.534   | −3.7%             | +1.04     | **+1.73**  | +1.33                   |
| 2022 | 216 | 0.472   | −13.4%            | +4.42     | **+3.97**  | +4.99                   |
| 2023 | 215 | 0.465   | −4.0%             | −0.12     | **+0.69**  | +0.62                   |
| 2024 | 215 | 0.484   | +10.8%            | +0.04     | **−0.03**  | +0.11                   |

The margin is concentrated in the 2021–2022 (downturn) legs and is near zero —
even slightly negative — in the rising 2024 year. The by-regime breakdown
(Phase-1 regime labels) shows the same concentration: bull $n=318$,
$\|2P-1\|$ margin $+1.07$; bear $n=530$, $+2.00$;
$\|2P-1\|/\mathrm{vol}$ $+0.69$ vs. $+2.74$ respectively. In the crisis
regime ($n=19$, Oct–Nov 2024) the head **never shorts**: all 19 days carry
$P>0.5$ (mean $0.607$), so the model and its EM coincide exactly and the
margin is identically zero. This degeneracy is reported as a property of the
frozen head rather than a strategy decision.

We therefore report the CSI300 result as: **statistically significant over
the aggregate 2021–2024 window, but strongly time- and regime-dependent, and
not interpretable as a stable annual Sharpe.** The $2025$–$2026$ untouched
window (evaluated once, § 4.8) is a rebound period yet remains positive, so a
strict "downturn-only" reading is also unsupported; the correct statement is
that the effect concentrates in high-confidence-event years and is not
uniform.

This result is important for two reasons. First, it demonstrates that the
supervised directional component is capable of producing a substantial
directional edge in a second market, and that the *protocol* (feature build +
supervised sign + confidence scaling) transfers as a method even though SPY
itself showed only a small sign. Second, the sizing results are **market
dependent**: volatility normalization is harmful on corrected SPY but
beneficial on CSI300. The experiments therefore do not justify a universal
claim that one particular sizing transformation is optimal across markets.
The correct conclusion is that "is the RL magnitude useful?" and "does the
market contain a transferable directional signal?" are separate questions and
must be evaluated independently.

---

**NIFTY (independent re-estimation on a third market).** The same frozen
protocol — identical feature construction (with the NIFTY daily return on the
"instrument" side of the cross-asset spreads), identical split bounds, the
same single-logistic $P$ head C-selected on validation — is re-estimated on
NIFTY 50. A coverage caveat applies and is reported as a property of the data,
not a protocol change: the Yahoo-derived NIFTY series carries zero volume
before approximately 2013, so the same volume>0 drop rule yields coverage
2013-01-21 .. 2026-09-09. The 2021–2024 test window ($n=881$) and the
2025–2026 out-of-window arm are fully covered, while the **training window
starts in 2013** (~6 years of history vs. ~11–13 years for SPY/CSI300) — a
real, disclosed asymmetry. NIFTY also differs in the character of the test
window: it is a strong uptrend (up-rate 0.547, cumulative drift ×1.801 over
2021–2024), the opposite regime to CSI300's 2021–2024 downtrend, and the size
and significance of the effect are the largest of the three markets (all
margins vs. the rule's own equal-magnitude long-only EM, net of costs at
1 bp):

| Sizing rule                     | Sharpe(1 bp) | EM(1 bp) | Margin(1 bp) | t_NW | 95% CI (bp/day, l=21) |
| ------------------------------- | -----------: | -------: | -----------: | ---: | --------------------- |
| Sign-only $|a|=1$               | 3.37         | 1.24     | **+2.13**    | 2.93 | [+3.4, +21.0]         |
| $\|2P-1\|$ (canonical P)        | 4.40         | 1.29     | **+3.11**    | **3.87** | [+2.2, +6.9]      |
| $\|2P-1\|/\mathrm{vol}_{20d}$   | 4.77         | 1.67     | **+3.10**    | **4.53** | [+15.7, +42.4]    |
| $1/\mathrm{vol}_{20d}$ (no P)   | 3.62         | 1.51     | +2.11        | 2.98 | [+24.1, +139]          |

The confidence-scaled rules are robust across reseeds again (10/10 positive
resampled probability heads; mean margin ≈ **+3.18** for $\|2P-1\|$ and ≈
**+3.20** for $\|2P-1\|/\mathrm{vol}$). Crucially — and unlike CSI300, where
sign-only was borderline — on NIFTY **the direction is itself significant**
(sign-only $t=2.93$, bootstrap interval excluding zero). The "direction is
exhausted" reading of the SPY result (§ 4.3) therefore does **not**
generalize: the protocol's directional value is market-dependent (null on SPY,
borderline on CSI300, significant on NIFTY).

Unlike CSI300, the NIFTY margin is near-uniform across the four test years:

| Year | $n$ | Up-rate | Drift (cum. ret.) | Sign-only | $\|2P-1\|$ | $\|2P-1\|/\mathrm{vol}$ |
| ---: | --: | ------: | ----------------: | --------: | ---------: | ----------------------: |
| 2021 | 220 | 0.545   | +24.1%            | −0.03     | **+1.03**  | +0.74                   |
| 2022 | 223 | 0.511   | +5.9%             | +5.00     | **+5.31**  | +4.80                   |
| 2023 | 219 | 0.571   | +17.7%            | +2.69     | **+3.28**  | +3.40                   |
| 2024 | 219 | 0.562   | +16.4%            | +0.89     | **+2.90**  | +2.90                   |

The edge is present in **every** year and largest in 2022, so the CSI300
nonuniformity does not repeat; the cross-market pattern is "market-dependent
in level, not in existence." By-regime, the bear leg is again the strongest
(bear $n=228$, $\|2P-1\|$ margin $+5.62$; bull $n=618$, $+2.39$), but the
crisis/surge regime is **negative** ($n=35$, up-rate 0.686, $\|2P-1\|$ margin
$−0.83$; sign-only $−2.48$): the model shorts and loses during the
highest-up-rate surge days — the *opposite* of CSI300's crisis degeneracy and
a genuine negative finding that prevents any claim of downside protection.

The daily $\|2P-1\|$ model-vs-EM margin shows the same right-tailed
event-driven shape as CSI300 at higher amplitude: mean $+4.40$ bp/day, median
$0$, standard deviation $30.2$ bp/day, skewness $+5.7$, excess kurtosis $+77$
(CSI300: $+1.85$, $17.1$, $+2.5$, $+29$). Maximum drawdown of the net
cumulative path is improved by event-scaled sizing here too:
$\|2P-1\|$ $−0.014$ vs. its own long-only EM $−0.069$ (≈ 5×), and
$\|2P-1\|/\mathrm{vol}$ $−0.085$ vs. $−0.321$.

We therefore report the NIFTY result as: **statistically significant over the
aggregate window, present in all four test years and in a strong uptrend, but
negative in the crisis/surge regime.** The earlier training start (2013, not
2005 via the volume>0 rule) means a shorter history with an identical protocol
and a *larger* edge; this strengthens the protocol-level transfer reading
while remaining a real limitation that must be disclosed.

## 4.8 Out-of-window validation

To test whether the cross-market results are driven by the particular
evaluation window, each directional head estimated on the 2021–2024 training
window is rolled forward **without retraining or modification** into the
2025–2026 period.

For CSI300 the frozen model achieves

* accuracy: **0.581** (CSI300 up-rate in that window: 0.547),
* sign-only margin(1 bp): **+1.57**,
* $\|2P-1\|$ margin(1 bp): **+2.06**,
* $\operatorname{corr}(P, R) = +0.231$.

The NIFTY head — held to the same 2025-01-01..2026-07-17 arm, once, where the
frame ends because the VIX3M macro series ends on that date ($n=332$, up-rate
0.524) — persists **more strongly** out-of-window:

* accuracy: **0.566**,
* sign-only margin(1 bp): **+2.30**,
* $\|2P-1\|$ margin(1 bp): **+3.23**,
* $\operatorname{corr}(P, R) = +0.231$.

The persistence of the directional relationship *outside* the original
training/evaluation window, on both markets and under different market
conditions (the CSI300 out-of-window rebound from a prior downtrend and the
NIFTY continuation of its prior uptrend), provides evidence against the
interpretation that either result is an artifact of a particular
trend period. The magnitude of these results is nevertheless unusually large
for a daily logistic directional model on both markets; we therefore treat
them as evidence requiring further independent validation, not as a
demonstrated, exploitable Sharpe ratio.

## 4.9 Out-of-distribution weight transfer

We additionally test whether the SPY-trained directional weights can simply be
transferred to CSI300 and NIFTY.

They cannot. Zero-weight-transfer performance on CSI300 is approximately
equivalent to the unconditional CSI300 up-rate (accuracy 0.489 vs. up-rate
0.489), margins close to zero (sign-only −0.20, ‖2P−1‖ −0.40), and

$$
\operatorname{corr}\!\big(P_{\mathrm{OOD}},\ P_{\mathrm{CSI300}}\big)
\approx 0.02:
$$

the SPY model's predictions are essentially uncorrelated with the model
estimated on CSI300.

The same negative result repeats when the SPY head is applied to NIFTY
features without retraining: accuracy **0.541**, essentially equal to the NIFTY
up-rate (0.547), margins near zero (sign-only −0.52, ‖2P−1‖ −0.002), and

$$
\operatorname{corr}\!\big(P_{\mathrm{OOD}},\ P_{\mathrm{NIFTY}}\big)
\approx 0.02.
$$

This negative result is informative about the nature of the cross-market
result. It does **not** demonstrate that a directional model trained on one
market can be directly deployed on another. Instead, it supports a weaker and
more defensible statement:

> **The same learning protocol can recover a useful directional relationship
> when independently estimated on a second or third market.**

Thus, the cross-market result represents **protocol-level transfer**, not
parameter-level transfer.

## 4.10 Summary of findings

| Hypothesis                                            | Result                  |
| ------------------------------------------------------ | ----------------------- |
| Expanded macro state allows TACR to recover direction  | **Rejected**            |
| TACR magnitude provides incremental value (SPY)        | **Rejected**            |
| Sign-only trading is sufficient (SPY)                  | **Rejected**            |
| Volatility normalization universally improves sizing   | **Rejected**            |
| Ensemble disagreement identifies sign errors           | **Rejected**            |
| SPY confidence sizing $\|2P-1\|$ is seed-stable in sign | **Supported**           |
| SPY confidence sizing is statistically significant      | **Not established** (t=0.41; CI contains zero) |
| $\|2P-1\|$/$-\mathrm{vol}$ significant vs own EM (CSI300) | **Supported** (t≈2.8) |
| $\|2P-1\|$/$-\mathrm{vol}$ significant vs own EM (NIFTY)  | **Supported** (t≈3.9 / 4.5) |
| Direction significant as well (NIFTY sign-only)          | **Supported** (t≈2.9; unlike CSI300) |
| Directional model transfers under re-estimation          | **Supported** (CSI300, NIFTY) |
| CSI300 / NIFTY result persists into 2025–2026            | **Supported**           |
| Edge is uniform across years / regimes                   | **Rejected**            |
| SPY weights transfer directly to CSI300 / NIFTY          | **Rejected**            |

### Central empirical conclusion

The experiments favor a **decomposed view** of the trading decision:

$$
\boxed{
\text{directional probability }
\quad\longrightarrow\quad
\text{confidence-based exposure}
}
$$

rather than the assumption that an offline RL agent must learn direction and
magnitude jointly. Concretely, there is little evidence that the TACR
magnitude policy adds value beyond a simple probability transformation. The
defensible central claim is:

> **Confidence-scaled exposure $\|2P-1\|$ is the only sizing mechanism that
> remains positive across all SPY reseedings, although its aggregate effect
> on SPY is not statistically distinguishable from zero (t = 0.41). On the
> two other markets tested, the same frozen protocol produces a statistically
> significant margin: on CSI300 (t ≈ 2.8) strongly time- and regime-dependent,
> concentrating in high-confidence-event years rather than a stable annual
> edge, and on NIFTY (t ≈ 3.9), near-uniform across all four test years and
> present in a strong uptrend, though negative in the crisis/surge regime.
> The directional value is itself market-dependent: null on SPY, borderline
> on CSI300, significant on NIFTY.**

The results also provide negative evidence against two initially plausible
extensions — **uncertainty-based risk control** and **generic
volatility-normalized sizing** — neither of which survives the corrected
evaluation.

## 4.11 Appendix: experimental integrity and the sign-head ablation

### 4.11.1 Data-correction note (contaminated runs excluded)

The results above use corrected data and the restored SPY chronological
series. The following earlier experiment entries were invalidated and are
excluded from all tables: running the SPY experiments on CSI300 data
substituted for SPY in `data/processed` (starting 05-09, prior to restoration
at 7.34), together with two implementation bugs — a timezone-aware index
misalignment in the macro-feature handler, and a daily-index alignment error
in the cost re-pricing script that spuriously collapsed the short side.
These affected entries (PROJECT_NOTES 7.31, 7.32, 7.33) remain in the
project record for auditability but carry no evidential weight: every model
they reference was retrained and reevaluated on corrected TRUE-SPY data, and
the corrected results are the ones reported here. One pre-correction finding
that *qualitatively* survived the correction was the absence of a useful
uncertainty signal; on corrected data this appears as a null relationship
rather than an inversion (see § 4.6).

### 4.11.2 Sign-head ablation

Because the corrected SPY directional edge is small, we ran a controlled
four-arm ablation over the sign model (single logistic vs. 10-member
bootstrap ensemble, member isolation, and an identical-data ensemble) with
the same training/validation/test protocol. The deterministic solver makes
the identical-data ensemble exactly equal to the single model, so any ensemble
gain must be attributed to bootstrap resampling. Results: member margins range
from −0.73 to +0.58 (mean −0.07) and the ensemble margin (+0.23 in its
original seed) is positive in only 50% of reseeded draws, never beating the
best single member. The single-logistic head with canonical seed-mean
magnitude is used as the reference model everywhere in the main text, and the
bootstrap ensemble is retained only as a documented secondary model with its
documented seed-sensitivity.

### 4.11.3 Statistical-inference details

All significance figures are computed by `scripts/inference_diagnostics.py`
from the day-level test tables (`data/pvol_study.csv`,
`data/csi300_transfer.csv`, `data/nifty_transfer.csv`); no model is re-run. The
Newey–West HAC $t$-statistic uses lag 8 on the daily (model $-$ EM) net margin
series; moving-block bootstrap uses block length 21 with 5,000 resamples (block
length 5 gives materially wider intervals on SPY and does not change any
qualitative verdict). Reconstructed Sharpe/margin values reproduce the tables
above to within tolerance, cross-validating the recorded results. The
aggregate-window intervals are descriptive (the window served development);
the confirmatory evidence is the two reported-once out-of-window arms (§ 4.8),
which are never re-touched.