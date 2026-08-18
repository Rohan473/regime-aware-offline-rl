"""DDR policy network: single-layer GRU/LSTM encoding the feature window.

Input per decision step: the trailing ``window_size`` days of z-scored
features (shape (B, W, F), batch first). Output: one scalar action in
[-1, 1] via tanh, matching the Phase-1 action space (position limits
[-1, 1] with shorting).

The recurrence is INSIDE the window: the network has no cross-day memory
between decisions. All cross-day memory lives in the DSR EMA (see dsr.py)
and in the window itself. This is intentionally the simplest policy in the
four-model comparison.
"""

from __future__ import annotations

import torch
from torch import nn


class DDRPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 32, rnn_type: str = "gru") -> None:
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(input_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W, F) -> actions (B, 1) in [-1, 1].

        Only the final hidden state (end of the window) produces the action.
        """
        out, _ = self.rnn(x)
        return torch.tanh(self.head(out[:, -1, :]))