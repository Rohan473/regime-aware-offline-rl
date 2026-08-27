"""Pure helpers for the structural-fix variants (A / B / C) — unit-testable.

These target the documented Q-collapse mechanism of the TACR objective on
this data (PROJECT_NOTES 7.6): the actor loss ``-lambda * Q + BC`` goes
negative as the critic's Q(s, pi(s)) inflates and overwhelms the BC term.
The fixes change the optimization GEOMETRY, not hyperparameters:

A (double Q, TD3-style): the critic TD target uses the MINIMUM of two
   independent critics, so a hallucinated overestimation in one critic is
   capped by the other. Used for both the bootstrap target and the actor's
   Q. Standard fix for Q overestimation collapse.

B (BCQ-style hard constraint): the actor's proposed action is pinned to
   within ``+-phi`` of a sample from a generative model of the
   behavior-policy actions (``ConditionalGaussian``, action_model.py).
   Gradient flows only while the proposal is inside the band, so the
   OOD escape hatch is closed regardless of how inflated Q becomes.

C (PAR, direction-aware BC target replacement): when the Q-gradient
   direction opposes the BC (toward-logged-action) direction on an
   element, the BC target is replaced with a projection that moves ``phi``
   in the Q-preferred direction FROM the logged action — aligned with Q,
   still within the data support (``phi`` of the logged action). Prevents
   the tug-of-war where the actor is torn between two conflicting losses.
"""

from __future__ import annotations

import torch


def double_q_min(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Clipped double Q: pessimistic min of two independent estimates."""
    return torch.minimum(q1, q2)


def double_q_mean(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Ensemble double Q: average of two independent estimates (less
    pessimistic than min; no downward bias on the actor's signal)."""
    return (q1 + q2) / 2.0


def double_q(q1: torch.Tensor, q2: torch.Tensor, mode: str = "min") -> torch.Tensor:
    """Aggregate two independent critics by ``mode`` in {"min", "mean"}."""
    if mode == "min":
        return double_q_min(q1, q2)
    if mode == "mean":
        return double_q_mean(q1, q2)
    raise ValueError(f"unknown double_q mode: {mode!r}")


def bcq_constrain(pi: torch.Tensor, a_gen: torch.Tensor, phi: float) -> torch.Tensor:
    """BCQ-style hard constraint: a_gen + clamp(pi - a_gen, -phi, +phi).

    Shapes broadcast (pi, a_gen). The output stays within ``phi`` of the
    generated action; gradients flow through ``pi`` only while unsaturated.
    """
    return a_gen + torch.clamp(pi - a_gen, -phi, phi)


def par_bc_target(
    pi: torch.Tensor,
    a_logged: torch.Tensor,
    g_q: torch.Tensor,
    phi: float,
    cos_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PAR BC-target replacement. Actions are scalars, so per-element
    "direction" is the sign.

    Returns ``(bc_target, cos, frac_replaced)``:
    - ``cos`` = normalized dot of the Q-gradient direction and the BC
      direction (a_logged - pi), per element, in {-1, 0, 1}.
    - elements with ``cos < cos_thresh`` get their BC target replaced by
      ``a_logged + phi * sign(g_q)`` (projected action: aligned with the
      Q-gradient, within ``phi`` of the logged action = inside the data
      support).
    """
    eps = 1e-8
    g = g_q / (g_q.abs() + eps)
    d = a_logged - pi
    d = d / (d.abs() + eps)
    cos = g * d
    replace = cos < cos_thresh
    proj = a_logged + phi * g
    target = torch.where(replace, proj, a_logged)
    return target, cos, replace.float().mean()