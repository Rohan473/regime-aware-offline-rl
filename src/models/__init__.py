"""Shared model configuration for the four-model comparison (B: DDR, C: TACR,
D: fuzzy + transformer + offline RL).

All models consume the SAME feature windowing and the SAME time splits, so
their results are comparable on the project's eval axis. These constants are
the single source of truth: models must import them, never redefine them.

- ``WINDOW_DAYS``: trailing feature window (in trading days) fed to the
  policy at every decision step. 20 by default; B/C/D must use this value.
- ``SPLIT_*_END``: time split boundaries for train / validation / test.
  Train: <= SPLIT_TRAIN_END, val: (TRAIN_END, VAL_END], test: > VAL_END.
"""

WINDOW_DAYS = 20

SPLIT_TRAIN_END = "2018-12-31"
SPLIT_VAL_END = "2020-12-31"
SPLIT_TEST_END = "2024-12-31"