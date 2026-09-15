# 5. Discussion

This section interprets the frozen empirical record in § 4. It introduces no
new empirical claims; every statement below is either directly supported by a
§ 4 result or is explicitly labeled as an interpretation (a plausible
mechanism we do not claim to have established).

## 5.1 Main findings

Across the corrected pipeline, the experiments establish seven results.

1. **The offline RL actor does not reliably recover the directional
   component.** On true SPY data, the Transformer actor's exposure is not a
   positive directional signal: sign-only exposure is approximately null to
   negative net of costs, and the fully costed canonical C+ construction is
   negative at every cost level from 0 to 20 bp. Expanding the state from 8 to
   16 dimensions (adding the macro information available to the supervised
   head) does not recover direction (§ 4.2). This is a conditional negative:
   it does not say no information set could fix it, only that the information
   set that helps the supervised head does not help the RL head.

2. **The RL magnitude component provides no demonstrated incremental
   advantage.** Both the 8- and 16-dimensional TACR magnitude policies are
   negative net of costs and worse than a simple deterministic transformation
   of the supervised probability (§ 4.4). The paper therefore does not claim
   that offline-RL magnitude is worthless in general; it claims that in this
   protocol, on this data, it adds no measured value over a probability-derived
   exposure rule.

3. **SPY confidence sizing is seed-stable in sign but statistically
   inconclusive.** The $\|2P-1\|$ rule is positive in all ten reseeded
   probability heads (mean margin $+0.059$, range $+0.002$ to $+0.123$),
   making it the only sizing mechanism whose sign is robust to reseeding on
   SPY. Its aggregate margin ($+0.051$ @ 1 bp) is nevertheless not
   distinguishable from zero (Newey–West $t = 0.41$; block-bootstrap interval
   containing zero) (§ 4.4).

4. **CSI300 is statistically significant but strongly time- and
   regime-dependent.** The same protocol re-estimated on CSI300 yields
   $\|2P-1\|$ and $\|2P-1\|/\mathrm{vol}$ margins ($+1.58$, $+1.87$ @ 1 bp)
   that are significant against their own EMs (HAC $t \approx 2.8$ each;
   bootstrap intervals excluding zero), but the margin concentrates in the
   2021–2022 downturn legs and is near zero or negative in the rising 2024
   year (§ 4.7). This is not a stable annual Sharpe.

5. **NIFTY replicates and strengthens the non-US result.** Re-estimating the
   same protocol on NIFTY — a strong *uptrend* window, the opposite index
   regime to CSI300's 2021–2024 — yields $\|2P-1\|$ and
   $\|2P-1\|/\mathrm{vol}$ margins ($+3.11$, $+3.10$ @ 1 bp) significant
   against their own EMs (HAC $t \approx 3.9$ and $4.5$; bootstrap intervals
   excluding zero), present in all four test years and with the direction
   itself significant (sign-only $t \approx 2.9$). The crisis/surge window is
   negative ($-0.83$, $n=35$): the head shorts and loses in the highest
   up-rate surge days — a genuine negative finding that prevents a
   downside-protection reading (§ 4.7).

6. **Uncertainty is null.** Ensemble member disagreement does not identify
   sign errors or adverse outcomes on corrected SPY data (correlations
   $\approx +0.01$, $+0.04$), so uncertainty-based filtering, error detection,
   and risk sizing are all rejected (§ 4.6).

7. **Direct parameter transfer fails; protocol transfer is the informative
   finding.** SPY-trained weights applied directly to CSI300 or NIFTY features
   produce chance-level accuracy and zero margin, with predictions essentially
   uncorrelated with the re-estimated head. What transfers is the learning
   protocol itself (§ 4.9).

## 5.2 Direction–magnitude decomposition

The conceptual contribution of the paper is to treat the trading decision as
two separable learning problems — *what* direction to trade and *how much* to
expose — and to measure the value of the RL magnitude component only in the
presence of a separately supervised direction.

The evidence is consistent with the view that **joint optimization is not
presumptively superior to decomposition**: in these experiments the jointly
trained magnitude adds nothing measurable over the decomposed construction.
We are careful not to claim that decomposing direction and magnitude
*universally* improves trading. The favorable reading of a decomposition is
conditional: it isolates the components that can be learned reliably (here,
directional confidence) from those that could not be (here, magnitude and
uncertainty), and it makes the failure modes of each component visible — which
is what allowed the study to attribute nearly all of the SPY edge to a rare,
high-confidence short leg rather than to a smooth directional forecast
(§ 4.5).

What the study rules out is narrower but sharper: **for these tasks, the
magnitude problem did not exhibit negative-results-because-unmodeled
behavior.** The probability head carried whatever information existed, and
the RL head failed to add to it. This reframes the usual engineering question
from "how do we make the RL module stronger?" to "is the RL module's target
even learnable from this data?" — a question the decomposition makes
answerable.

## 5.3 Interpretation of confidence scaling

The $|2P_t-1|$ signal is best read as **confidence-weighted exposure, not
magnitude prediction**. The probability model does not forecast the size of
the next return: $\operatorname{corr}(|2P_t-1|, |R_{t+1}|) \approx 0.03$ and
$\operatorname{corr}(P_t, |R_{t+1}|) \approx 0.004$ on SPY (§ 4.5). Scaling
exposure by confidence therefore does *not* implement "more position when the
move is bigger"; it implements "more position when the model is more willing
to commit to one side." These are operationally different sizing policies, and
the poor showing of volatility-normalized sizing on SPY (§ 4.4) reinforces
that the mechanism is not volatility timing.

The three markets differ in magnitude, significance, and regime profile,
while sharing the same event-driven, confidence-scaled mechanism:

- On **SPY**, the effect is small and statistically inconclusive but
  sign-stable across reseeding, and concentrated in a small set of
  high-confidence short positions (§ 4.4–4.5). The defensible statement is
  seed-stability of sign, not statistical significance.
- On **CSI300**, the same signal is significant against its own EM
  (HAC $t \approx 2.8$) but the aggregate conceals strong heterogeneity:
  the margin is carried by the 2021–2022 bear legs, is near zero in the 2024
  rebound, and is zero by construction in the 19-day crisis window where the
  head never shorts (§ 4.7).
- On **NIFTY**, the same signal is significant (HAC $t \approx 3.9$) and
  near-uniform across all four test years, in a strong uptrend; the
  crisis/surge window is *negative* ($-0.83$; the head shorts and loses in
  the highest up-rate surge days), a genuine negative finding that reinforces
  the market-dependence of the effect (§ 4.7).

The paper does not generalize "confidence scaling works." It establishes that
confidence-weighted exposure is *the only mechanism whose sign survives all
SPY reseedings* and that *the same protocol on a second and third market
produces significant but market-dependent effects* (CSI300: regime-
concentrated; NIFTY: near-uniform in level, crisis-negative).

## 5.4 Why offline RL may not add magnitude value

We treat the following as plausible interpretations, not established causal
mechanisms:

- **Reward / objective mismatch.** The squared-error behavior policy
  objective of the offline RL actor is not the sign-weighted profitability
  objective that actually matters for the strategy; a magnitude that
  maximizes the former need not improve the latter. (Suggested by the
  protocol design; not isolated experimentally.)
- **Noisy daily rewards dominate the magnitude target.** Even if a
  magnitude relationship exists, daily index returns have a low signal-to-noise
  ratio at the single-day horizon, so a magnitude policy fitted on such rewards
  may estimate mostly noise. Consistent with the null uncertainty result
  (§ 4.6), which shows that even a direct ensemble disagreement signal cannot
  distinguish the model's good from its bad days.
- **Joint sign/exposure optimization is a harder optimization than sign
  alone.** The actor must solve the sign problem before it can size
  conditionally; if the sign is weakly learnable, the joint problem inherits
  that difficulty and adds a second, noisier target.
- **Offline action-distribution constraints.** Offline RL value learning is
  only reliable inside the behavioral policy's support, and support-limited
  learning can produce conservative magnitude estimates that converge to a
  near-constant exposure — behavior consistent with the observed results
  (though we did not directly measure policy conservatism).

These are hypotheses organized by the experimental record, not claims the data
proves. Distinguishing among them (reward reweighting, tethered magnitude
heads, deliberate support expansion) is left to future work.

## 5.5 Market and regime dependence

The 2021–2024 and by-regime results (§ 4.7) force an explicit rejection of a
universal-Sharpe interpretation of the cross-market findings. On CSI300 the
effect is time-concentrated (2021–2022 strong, 2024 near zero) and
regime-concentrated (bear stronger than bull; crisis degenerate because the
head never shorts), so CSI300 alone supports a "strong in some regimes, weakly
present in others" reading. NIFTY changes that reading: the effect there is
significant, present in all four test years, and largest in a strong uptrend,
so the cross-market conclusion is not a portable year-by-year edge. The
defensible statement is **market-dependent in level, not in existence**: SPY is
near zero, while both non-US markets carry significant margins with regime
profiles that differ by market (CSI300 concentrated in downturn legs and
crisis-degenerate; NIFTY near-uniform across years but negative in the
crisis/surge window). The out-of-window arms (§ 4.8) are important precisely
because they remained positive under different market conditions — the CSI300
rebound from a prior downtrend and the NIFTY continuation of a prior uptrend —
which rules out the strongest version of the "downturn-only" story, but their
sizes ($n=329$ and $n=332$, one report each) do not support a stable-Sharpe
claim either.

## 5.6 Negative findings

The following were plausible hypotheses going in and are rejected by the
corrected data:

- **Uncertainty-based risk control**: ensemble disagreement does not
  predict sign errors or adverse P&L (§ 4.6). There is no evidence for using
  it as a filter, error detector, or sizing variable.
- **Universal volatility-normalized sizing**: $\|2P-1\|/\mathrm{vol}$ is
  worse than $\|2P-1\|$ on SPY, and although it is positive on both CSI300 and
  NIFTY it remains a market-dependent, not a universal, improvement
  (§ 4.4, § 4.7).
- **Sign-only trading on SPY**: direction alone is null to negative
  (§ 4.3). (On NIFTY, direction alone *is* significant, which further shows
  the market-dependence of the directional value.)
- **Direct SPY→CSI300/NIFTY weight transfer**: chance-level; protocol
  transfer is what works (§ 4.9).

Recording these nulls is a contribution in itself: each was a candidate
mechanism for "how the model could be right," and each failed on the corrected
data.

## 5.7 Limitations and future work

- **Three markets, but one macro feature source.** SPY, CSI300, and NIFTY
  cover three distinct markets and index regimes, but all share the same
  US-macro conditioning features and the same protocol. Confirming on further,
  structurally different markets is the first priority.
- **Unusually large non-US effects.** Margins of $+1.4$ to $+1.9$ (CSI300)
  and $+3.1$ (NIFTY) Sharpe points for a daily logistic are atypically large;
  we report them honestly with the by-year and regime breakdowns and the OOS
  arms, and treat them as requiring independent validation rather than as a
  demonstrated, exploitable edge.
- **Regime dependence.** The CSI300 effect is not uniform across years, and
  the NIFTY effect — though uniform across years — is negative in the
  crisis/surge window; any application must confront each market's year- and
  regime-dependence.
- **Limited statistical power on SPY.** The SPY claim is seed-consistency,
  not significance; the sample is too noisy for the small effect to be
  resolved.
- **Exploratory development history and the contamination event.** All
  results on the 2021–2024 window are descriptive (development-informed); the
  integrity appendix (§ 4.11.1) documents the data-corruption event and the
  correction. Genuinely untouched future data is required to turn the
  confirmatory story into a stronger claim.
- **No causal mechanism established for RL magnitude failure** (§ 5.4);
  the four hypotheses are offered for future experiments to separate.

Future work: additional markets; a pre-registered frozen evaluation on a
never-observed window; deliberate experiments to separate the § 5.4
hypotheses; and magnitude-head variants (reward reweighting, tethered
magnitude) tested against the freed decomposition baseline.

## 5.8 Implications for offline financial RL

The study's most general implication is methodological: **the financial
trading decision should be decomposed and its components tested individually
before resources are committed to joint offline-RL optimization.** In this
protocol, the marginal value of the learned magnitude component was not
distinguishable from a probability-derived exposure rule in direction or
magnitude on any of the three markets; the *directional* value, by contrast,
was null on SPY and statistically significant only when the probability head
was re-estimated independently on CSI300 (regime-concentrated) or NIFTY
(near-uniform across years, crisis-negative). Until a magnitude-learning
mechanism demonstrates incremental value against a probability-scaled EM
baseline, joint RL optimization of direction and magnitude should not be
presumed to add value over decomposition. This is a falsifiable, and here
partially falsified, presumption — which is what makes it useful.