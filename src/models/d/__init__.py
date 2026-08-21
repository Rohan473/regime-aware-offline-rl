"""Model D — fuzzy uncertainty encoding + causal transformer + IQL.

Phase-4 model: tests the central hypothesis that making the agent's
uncertainty about market regime EXPLICIT (interval type-2 fuzzy memberships
over vol/momentum/RSI features) before the policy network sees the state
improves offline-RL trading vs exposure-matched control (EM) and the
Phase-2/3 baselines (B: naive_new DDR, C: TACR).

Components:
- ``fuzzy``     fixed IT2 Gaussian membership layer (train-split init only)
- ``backbone``  causal transformer state encoder (Model C's verified blocks)
- ``iql``       Implicit Q-Learning agent (expectile V, Q, AWR policy)
- ``data``      per-policy transition loader (8-d raw or 8+18=26-d input)
- ``train``     offline IQL training with best-val + final checkpoints
- ``eval``      regime table, basin screening, pre-registered screen rules

Variants: D (fuzzy=True, 26-d) and D-minus-fuzzy (fuzzy=False, 8-d) share
this package; the fuzzy layer is the only structural difference. See
config.py and PROJECT_NOTES 7.9.
"""