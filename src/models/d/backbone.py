"""Model D backbone: causal transformer STATE ENCODER.

Unlike Model C's decision-transformer (which interleaves (rtg, state,
action) triples and is conditionally generated), Model D's transformer is
an ENCODER: position t receives only the (fuzzy-augmented) state vector of
day t plus a learned absolute timestep embedding, and STRICT causal
self-attention makes the output at position t a representation of the
window ending at t — h_t = f(s_{t-u+1}, ..., s_t). The IQL value / Q /
policy heads consume h_t at every decision step (spec section 3); there is
no return-to-go channel, so the encoder can never leak realized future
returns (the failure mode diagnosed for Model C in PROJECT_NOTES 7.6.1).

The attention and decoder blocks are Model C's EXACT modules (imported from
``src.models.tacr.policy``, which is byte-identical to the paper's GPT-2
setup minus positional embeddings), so the shared block hyperparameters
(embed_dim 128, n_layer 4, n_inner 512) give D a verified-correct causal
implementation. The causal-masking test in tests/test_model_d.py pins the
same guarantee as test_tacr.py: perturbing future tokens must not change
earlier outputs.

Block sizes are IDENTICAL across both D variants; the only difference is
the width of ``embed_state`` (26 vs 8), which is the tested treatment.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.tacr.policy import DecoderBlock  # C's verified blocks


class StateEncoder(nn.Module):
    """Causal encoder over per-day state vectors -> per-timestep h_t."""

    def __init__(
        self,
        state_dim: int,
        u: int,
        embed_dim: int = 128,
        n_layer: int = 4,
        n_head: int = 1,
        n_inner: int = 512,
        dropout: float = 0.1,
        max_ep_len: int = 5283,
    ) -> None:
        super().__init__()
        self.u = u
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)
        self.blocks = nn.Sequential(
            *[DecoderBlock(embed_dim, n_head, n_inner, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.embed_timestep.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embed_state.weight, mean=0.0, std=0.02)

    def forward(self, states: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """states (B, u, F); timesteps (B, u) long (absolute positions,
        left-pad positions use 0). Returns h (B, u, embed_dim)."""
        B, u = states.shape[:2]
        x = self.embed_state(states) + self.embed_timestep(timesteps)
        x = self.embed_ln(x)
        x = self.blocks(x)
        return self.ln_f(x)

    def encode_last(self, states: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Convenience: h_t for the LAST position of each window (B, E)."""
        return self.forward(states, timesteps)[:, -1]