"""Differential Sharpe Ratio (DSR) reward, exactly as in Moody & Saffell (1998).

The DSR measures the incremental contribution of the latest return to the
running Sharpe ratio:

    A_t = A_{t-1} + eta * (r_t - A_{t-1})        EMA of returns
    B_t = B_{t-1} + eta * (r_t^2 - B_{t-1})      EMA of squared returns
    D_t = (B_{t-1} * dA_t - 0.5 * A_{t-1} * dB_t) / (B_{t-1} - A_{t-1}^2)^(3/2)

with dA_t = r_t - A_{t-1} and dB_t = r_t^2 - B_{t-1}. D_t is the derivative
of the running Sharpe ratio w.r.t. the adaptation rate eta evaluated at
eta -> 0. The denominator is the running variance of the strategy returns.

``DSRState`` is a sequential, autograd-friendly EMA accumulator: D_t is
differentiable w.r.t. r_t (and therefore w.r.t. the policy action a_t, since
r_t = a_t * R_{t+1} - cost * |a_t|), which enables direct backprop training
instead of a score-function gradient.

Numerical/definitional details:
- The DSR is undefined while the running variance is non-positive or before
  ``warmup_steps`` EMA steps exist. Such steps emit NaN; callers mask them
  out of the loss. This is not an approximation of D_t — it is a seeding
  rule, identical to Moody & Saffell starting the EMA on an initial window.
- A floor of 1e-12 is applied to the denominator so the quotient never
  divides by zero; it only binds where the step is NaN anyway.
- The EMA state is detached on every ``update`` call (truncated BPTT):
  gradients flow within the passed block of returns, not across blocks.
"""

from __future__ import annotations

import torch


class DSRState:
    def __init__(
        self,
        eta: float,
        warmup_steps: int = 0,
        a0: float = 0.0,
        b0: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.eta = eta
        self.warmup_steps = int(warmup_steps)
        self._a = torch.tensor(a0, dtype=dtype)
        self._b = torch.tensor(b0, dtype=dtype)
        self._steps = 0

    def reset(self, a0: float = 0.0, b0: float = 0.0) -> None:
        self._a = torch.tensor(a0, dtype=self._a.dtype)
        self._b = torch.tensor(b0, dtype=self._b.dtype)
        self._steps = 0

    @property
    def a(self) -> torch.Tensor:
        return self._a

    @property
    def b(self) -> torch.Tensor:
        return self._b

    def update(self, returns: torch.Tensor) -> torch.Tensor:
        """Feed one block of strategy returns; returns D_t per step.

        returns: 1-D tensor of length T. State is carried across calls
        (detached at the end of each call -> truncated BPTT).
        """
        returns = returns.to(self._a.dtype)
        T = returns.shape[0]
        out = torch.full((T,), float("nan"), dtype=returns.dtype, device=returns.device)
        A = self._a
        B = self._b
        for t in range(T):
            r = returns[t]
            dA = r - A
            dB = r * r - B
            var = B - A * A
            denom = var.clamp_min(1e-12).pow(1.5)
            D = (B * dA - 0.5 * A * dB) / denom
            valid = (var > 0.0) & (self._steps >= self.warmup_steps)
            out[t] = torch.where(valid, D, torch.tensor(float("nan"), dtype=D.dtype))
            A = A + self.eta * dA
            B = B + self.eta * dB
            self._steps += 1
        self._a = A.detach()
        self._b = B.detach()
        return out

    def sharpe_ema(self) -> torch.Tensor:
        """Running Sharpe implied by the EMA state (A / sqrt(B - A^2)).

        NaN when the running variance is non-positive.
        """
        var = self._b - self._a * self._a
        if bool((var > 0).logical_not()):
            return torch.tensor(float("nan"), dtype=self._a.dtype)
        return self._a / var.clamp_min(1e-12).sqrt()


class VolTargetBuffer:
    """Volatility-targeted DSR reward input (the reward fix).

    Before the DSR increment is computed, each action is scaled so the
    strategy-return series (a_t * ret_{t+1}) runs at a target annualized
    volatility:

        scale_t = target_vol / realized_vol_t        (annualized)
        a'_t     = clip(a_t * scale_t, -max_leverage, +max_leverage)
        r'_t     = a'_t * ret_{t+1} - cost * |a'_t|   (DSR input)

    ``realized_vol_t`` is the causal trailing standard deviation of the
    strategy-return series over ``window`` steps (detached: no gradient
    flows through the vol denominator, so blocks stay decoupled under
    truncated BPTT). Because shrinking |a_t| shrinks the strategy returns
    AND the measured vol, the scaled series is (forward) invariant to a
    uniform de-leveraging — the policy cannot game the DSR by shrinking
    exposure; it can only choose direction and relative confidence.

    Warm-up/undefined vol (fewer than ``window`` returns, or zero vol):
    scale = 1.0 (no targeting). The leverage clip always applies.
    """

    def __init__(
        self,
        target_vol: float,
        window: int = 20,
        max_leverage: float = 2.0,
        cost_bps: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.target_vol = float(target_vol)
        self.window = int(window)
        self.max_leverage = float(max_leverage)
        self.cost = float(cost_bps) / 1e4
        self._tail = torch.zeros(self.window - 1, dtype=dtype)
        self._tail_len = 0

    def __call__(
        self, actions: torch.Tensor, next_returns: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale one block of actions; returns (scaled_actions, DSR-input
        returns). The trailing strategy-return series carries across blocks
        (detached), so vol is causal over the whole training sequence."""
        returns = next_returns.to(actions.dtype)
        strategy = actions * returns
        full = torch.cat([self._tail, strategy])          # (tail + block,)
        T = strategy.shape[0]
        W = self.window
        scale = torch.ones(T, dtype=actions.dtype)
        start = (W - 1) - self._tail_len          # first valid entry in `full`
        for j in range(T):
            usable = full[start : (W - 1) + j + 1]
            if usable.shape[0] >= W:
                # vol < 1e-6 ann is float32 rounding noise on a constant
                # series, not a real regime -> leave scale at 1
                vol = usable[-W:].detach().std() * (252**0.5)
                if bool(vol > 1e-6):
                    scale[j] = self.target_vol / vol
        scaled = torch.clamp(actions * scale, -self.max_leverage, self.max_leverage)
        out = scaled * returns - self.cost * scaled.abs()
        self._tail = full[-(W - 1) :].detach()
        self._tail_len = min(self._tail_len + T, W - 1)
        return scaled, out