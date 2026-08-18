"""Model C — TACR (Lee & Moon, IEEE Access 2023, DOI 10.1109/ACCESS.2023.3324458).

Reproduction of "Transformer Actor-Critic with Regularization" on the
project's continuous-action market, per the Phase-3 spec:

- DECISION-TRANSFORMER ACTOR: a GPT-2-style causal transformer over
  interleaved (return-to-go, state, action) MDP-element triples — NOT a
  flat feature window (that is Model B's GRU input). Each position attends
  only to earlier tokens (strict causal mask).
- BC-REGULARIZED ACTOR LOSS (the paper's contribution): the actor is
  trained offline on the logged behavior-policy trajectories with
  loss = -lambda * Q(s, pi(s)) + BC(pi, a_logged),
  lambda = alpha / |Q|.mean().detach() — the behavior-cloning term keeps the
  actor from drifting toward actions the critic overestimates but the
  dataset does not support (the paper's fix for Decision Transformer's
  inability to exceed the best behavior in the dataset).
- SEPARATE CRITIC MLP with target networks (the paper's code uses a
  standalone Critic([s,a] -> 512 -> 256 -> 1), NOT a shared-trunk head —
  followed the paper; see PROJECT_NOTES 7.6 for the deviation from the
  Phase-3 spec's "shared trunk" assumption).
- OFFLINE TRAINING: all objectives use the logged (s, a, r, s') transitions
  of the four Phase-1 behavior policies; no environment interaction, no
  online bootstrapping.

Adaptations for our action space (documented deviations, not simplifications
to hide):
  1. continuous actions [-1, 1] -> Linear+tanh action head (the paper used
     Linear+Softmax for discrete portfolio weights),
  2. the return channel is the true return-to-go (cumulative future logged
     reward); the paper's code embeds the immediate reward,
  3. at val/test roll time the return-to-go is a CONSTANT target
     (``rtg_target``, default 0.0 — the paper's eval feeds zeros), so no
     realized-future information enters evaluation,
  4. L = 4 decoder blocks (user-specified default range 3-4; paper used 5).
"""

from .config import TACRConfig
from .critic import TACRCritic
from .data import TACRData, load_tacr_data, split_tacr_data
from .policy import TACRPolicy
from .train import train_tacr

__all__ = [
    "TACRConfig",
    "TACRCritic",
    "TACRData",
    "load_tacr_data",
    "split_tacr_data",
    "TACRPolicy",
    "train_tacr",
]