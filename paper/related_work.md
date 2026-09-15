# 2. Related Work

> Drafting note: citations below use well-known landmarks; author/year/venue
> details should be verified against the target journal's citation style
> during packaging. No claim here is supported by new experiments — the
> empirical record is § 4.

## 2.1 Direct reinforcement learning for trading

The dominant line of work applies RL directly to the trading decision,
typically with an actor learning positions or portfolio weights from raw
features: early gradient-based direct-reward approaches (Moody & Saffell,
2001) framed trading as a recurrent RL problem, and deep variants (e.g., Deng
et al., 2017; Liu et al., 2019) demonstrated that deep direct RL can
outperform heuristic baselines on intraday and daily signals. Jiang, Xu, and
Liang (2017) extended the framework to portfolio management with
deterministic-policy-gradient learners and a dedicated cost model.

Within this line, **direction and magnitude are almost always optimized
jointly** by a single policy. The present study does not propose a new direct
RL algorithm; it evaluates the *marginal value* of the joint learner's
magnitude output against a supervised directional signal used for the sign and
a deterministic confidence rule for the size. This is a different question
than "can RL trade well?".

## 2.2 Offline reinforcement learning for financial trading

Offline RL (Levine, Kumar, Tucker, & Fu, 2020; Kumar, Zhou, Tucker, &
Levine, 2020, CQL) learns from a fixed dataset without environment
interaction, which suits financial settings where exploration is expensive and
deployment is risky (Fu et al., 2020; for finance-specific surveys see
relevant recent reviews). Financial applications commonly inherit the
off-policy value function, add risk constraints or conservative regularization,
and report improved returns versus behavioral cloning or supervised
baselines.

A recurring difficulty in this literature is **distinguishing genuine
signal from over-fitting and from reward exploitation under offline
constraints.** The present study inherits that difficulty and addresses it
directly: every claim is (i) measured net of transaction costs, (ii) compared
against an equal-magnitude long-only EM so that margins are scale-invariant,
(iii) re-estimated across reseeded probability heads, and (iv) validated on
two further markets (CSI300, NIFTY) and an untouched out-of-window period.

## 2.3 Transformer-based offline RL

Transformer sequence models have been applied to offline RL (Decision
Transformer, Chen et al., 2021; Trajectory Transformer, Janner, Li, Levine,
Chelsea, & Fu, 2021), treating return-conditioned or trajectory-level
prediction as a sequence modeling task. TACR (Li, Du, et al.) applies a
Transformer to daily financial trajectories and was the concrete architecture
studied here — we reuse its exposure policy as the magnitude generator in the
C+ construction.

The novelty of our study is not architectural. We hold the architecture fixed
and ask whether its magnitude output survives a careful decomposition against
simpler baselines. This is orthogonal to most Transformer-RL papers, which
introduce the architecture and demonstrate it works; we instead document where
a well-behaved example of the class fails to add value and where a simpler
alternative dominates.

## 2.4 Supervised directional prediction and probabilistic signals

Supervised return/direction prediction is mature: linear models (Sharpe,
1964-style market models; Fama & French factor models), and empirical asset
pricing with machine learning (Gu, Kelly, & Xiu, 2020) show that ML can
extract modest out-of-sample predictive power for equity returns, though
predictable components are economically small. Directional forecast
evaluation, particularly the tendency of daily returns to be dominated by
noise relative to signal, is well documented (e.g., Welch & Goyal, 2008, for
predictability bounds).

Our paper uses a supervised probability of an upward next-day return primarily
as a **crisp decomposition tool**, not as a new predictor. The directional
head is deliberately simple (a logistic model C-selected on validation) and is
kept identical across markets so that the comparison between markets isolates
the strategy mechanism rather than a specific model's tuning.

## 2.5 Position sizing and risk-aware trading

Position sizing in finance is well developed along several axes that our work
touches:

- volatility targeting (e.g., Moreira & Muir, 2017) and risk-based portfolio
  construction (inverse-volatility weighting as a benchmark; see e.g. De
  Miguel, Garlappi, & Uppal, 2009, for naive vs. optimized diversification
  comparisons);
- Kelly-style sizing from a forecast edge (Kelly, 1956; MacLean, Thorp, &
  Ziemba, 2011), where the position is a deterministic function of the
  estimated advantage;
- confidence-conditioned trading filters (trade only when the model is
  confident).

We contribute to this line a specific, simple family: exposure proportional to
$|2P-1|$, the model's directional confidence, evaluated against its own
equal-magnitude long-only EM. This is a falsifiable sizing rule whose marginal
value we measure directly — and, importantly, we find that naive volatility
normalization added to this rule does **not** transfer universally (harmful on
SPY, beneficial on CSI300 and NIFTY, § 4.4/4.7).

## 2.6 Uncertainty and OOD control in offline RL

Uncertainty-aware offline RL uses ensemble disagreement,
uncertainty-quantified value functions, or conservative estimates to detect
out-of-distribution actions and improve robustness (e.g., uncertainty-penalized
RL in MOPO-era methods and finetuned value-based approaches). Such mechanisms
are often motivated in finance as "risk awareness" or "OOD control."

Our evidence does not support extending that narrative to the daily index
direction problem as implemented here: ensemble disagreement did not predict
sign errors or adverse P&L on corrected SPY data (§ 4.6). We interpret this as
a caution: uncertainty-aware RL procedures are not automatically useful for
financial risk control at this horizon, and should be validated against the
same null-hypothesis discipline as any other component. (The empirical record
is negative; the discussion in § 5.6 frames the interpretation.)

## 2.7 Positioning of the present study

The closest-contributing literature develops **stronger RL policies, risk
constraints, or uncertainty-aware decision procedures** for trading. Each
assumes the moral target of the RL module is well specified and worth learning.

> In contrast, this study empirically decomposes the financial trading
> decision into directional prediction and exposure magnitude and tests
> whether the offline-RL magnitude component provides incremental value over a
> simple probability-derived exposure rule.

The specific claims to distinguish the study from the related work above:

- **Against 2.1/2.2 (direct RL):** the study measures the marginal value of
  the RL magnitude output, not the output itself, holding direction fixed by a
  supervised sign.
- **Against 2.3 (Transformer RL):** the architecture is a vehicle, not the
  contribution; the question is decomposability of the decision, not sequence
  modeling power.
- **Against 2.4 (supervised prediction):** the directional head is a retained
  instrument, not a new predictor; the contribution is the interaction with the
  magnitude problem.
- **Against 2.5 (sizing):** it contributes a falsifiable, seed-robust
  confidence-sizing rule and a market-dependent finding about volatility
  normalization, rather than a new risk model.
- **Against 2.6 (uncertainty):** it reports a decisive null for
  disagreement-based risk control at this horizon, which the uncertainty-aware
  RL literature does not prominently report as a failed hypothesis.

The claimed novel contribution is therefore the **decomposition + falsifiable
component-valued baseline** framework applied to offline financial RL, with
its honest negative results and cross-market reproducibility, not a new
learning algorithm.