"""Representation laboratory (interpretability layer).

Post-hoc analysis of the frozen B/C/D checkpoints and their learned
representations h_t. Central question (next_experiments.txt): what does an
RL trading model actually learn from a small financial state — what is
encoded in h_t, what is discarded, and what drives the action?

Modules:
- ``targets``    : shared probe target construction (forward direction,
                   magnitude, volatility, regime, feature reconstruction).
- ``extract``    : hidden-state extraction for DDR (GRU out[:, -1]),
                   TACR (state-token embedding) and Model D (encoder h_last),
                   aligned to the shared SPY date grid.
- ``probes``     : linear-probe fit/eval helpers (classification accuracy +
                   baseline, regression R2).
"""