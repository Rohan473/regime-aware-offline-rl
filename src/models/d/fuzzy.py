"""Interval type-2 (IT2) fuzzy state-encoding layer for Model D.

The layer makes the agent's uncertainty about regime-sensitive features
EXPLICIT as extra input channels: each of ``n_terms`` Gaussian membership
functions per fuzzy field produces an UPPER (UMF) and a LOWER (LMF)
membership, so a value is represented by an INTERVAL of memberships rather
than a point. This is the standard type-2 "footprint of uncertainty" (FOU)
decomposition.

Per the Phase-4 spec:
- Membership parameters are FIXED (registered buffers, no gradients), so
  the fuzzification treatment is isolated from any learned interaction —
  the layer is a deterministic, input-only transformation.
- Parameters are initialized from the TRAIN split ONLY (never val/test):
  centers at the 33rd / 50th / 66th percentiles of the train values;
  UMF std = ``mf_width_scale`` (0.5) x bin width; LMF std = UMF std x
  ``fou_scale`` (0.8). Bin width of the interior term = half the distance
  between its two neighbors; edge terms use the distance to their nearest
  neighbor (documented choice — deterministic and invertible in the sense
  that a wider bin -> wider membership -> more coverage).

Output order (appended AFTER the raw state, per field then per term):
for field f, term k: [UMF_{f,k}, LMF_{f,k}]  =>  18 channels for 3x3.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

FUZZY_TERM_NAMES = ("low", "mid", "high")


def compute_fuzzy_params(
    states_np: np.ndarray,
    field_indices: list[int],
    n_terms: int,
    mf_width_scale: float,
    fou_scale: float,
) -> dict[str, np.ndarray]:
    """IT2 Gaussian membership params from TRAIN-split normalized states.

    states_np: (T, F) normalized state matrix (train split only).
    Returns {centers, std_umf, std_lmf} each of shape (n_fields, n_terms).
    """
    if states_np.ndim != 2:
        raise ValueError("states_np must be (T, F)")
    sub = states_np[:, field_indices]  # (T, n_fields)
    n_fields = sub.shape[1]

    centers = np.empty((n_fields, n_terms))
    if n_terms == 3:
        pcts = np.array([33.0, 50.0, 66.0])  # spec: 33rd / 50th / 66th percentile
    else:
        pcts = np.linspace(
            100.0 / (n_terms + 1), 100.0 * n_terms / (n_terms + 1), n_terms
        )
    for f in range(n_fields):
        centers[f] = np.percentile(sub[:, f], pcts)  # sorted ascending

    bin_width = np.empty((n_fields, n_terms))
    for f in range(n_fields):
        for k in range(n_terms):
            if k == 0:
                bin_width[f, k] = centers[f, 1] - centers[f, 0]
            elif k == n_terms - 1:
                bin_width[f, k] = centers[f, -1] - centers[f, -2]
            else:
                bin_width[f, k] = (centers[f, k + 1] - centers[f, k - 1]) / 2.0

    std_umf = np.maximum(mf_width_scale * bin_width, 1e-6)
    std_lmf = np.maximum(fou_scale * std_umf, 1e-6)
    return {"centers": centers, "std_umf": std_umf, "std_lmf": std_lmf}


class FuzzyLayer(nn.Module):
    """Fixed IT2 Gaussian membership layer (buffers only, no learned params).

    forward(state): (..., F) -> (..., F + n_fields * n_terms * 2) with the
    fuzzy memberships concatenated AFTER the raw state. Deterministic in
    eval and train mode.
    """

    def __init__(
        self,
        state_dim: int,
        field_indices: list[int],
        n_terms: int,
        mf_width_scale: float,
        fou_scale: float,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.field_indices = list(field_indices)
        self.n_terms = n_terms
        self.mf_width_scale = mf_width_scale
        self.fou_scale = fou_scale
        n_fields = len(field_indices)
        self.register_buffer("centers", torch.zeros(n_fields, n_terms))
        self.register_buffer("std_umf", torch.ones(n_fields, n_terms))
        self.register_buffer("std_lmf", torch.ones(n_fields, n_terms))

    @property
    def n_fuzzy_features(self) -> int:
        return len(self.field_indices) * self.n_terms * 2

    def set_params_from(self, states_train_np: np.ndarray) -> "FuzzyLayer":
        """Initialize the fixed buffers from TRAIN-split normalized states."""
        p = compute_fuzzy_params(
            states_train_np,
            self.field_indices,
            self.n_terms,
            self.mf_width_scale,
            self.fou_scale,
        )
        self.centers.copy_(torch.tensor(p["centers"], dtype=torch.float32))
        self.std_umf.copy_(torch.tensor(p["std_umf"], dtype=torch.float32))
        self.std_lmf.copy_(torch.tensor(p["std_lmf"], dtype=torch.float32))
        return self

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (..., F); returns (..., F + 18) [raw, then fuzzy]."""
        sub = state[..., self.field_indices]  # (..., n_fields)
        x = sub.unsqueeze(-1)                 # (..., n_fields, 1)
        c = self.centers.unsqueeze(0)         # (1, n_fields, n_terms)
        umf = torch.exp(-0.5 * ((x - c) / self.std_umf.unsqueeze(0)) ** 2)
        lmf = torch.exp(-0.5 * ((x - c) / self.std_lmf.unsqueeze(0)) ** 2)
        # (..., n_fields, n_terms, 2) -> (..., n_fields * n_terms * 2)
        fuzzy = torch.stack((umf, lmf), dim=-1)
        fuzzy = fuzzy.reshape(state.shape[:-1] + (self.n_fuzzy_features,))
        return torch.cat((state, fuzzy), dim=-1)