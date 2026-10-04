"""Tests for the hybrid exposure blend (src.models.rep_lab.hybrid)."""

from __future__ import annotations

import numpy as np
import torch

from src.models.rep_lab.hybrid import BlendPolicy, compose, train_blend


def test_compose_bounds():
    e = torch.tensor([0.2, 0.5, 0.9])
    a = compose(e, torch.zeros(3))
    assert torch.allclose(a, e)
    a = compose(e, torch.ones(3))
    assert torch.allclose(a, torch.ones(3))
    a = compose(e, torch.tensor([0.0, 0.5, 1.0]))
    assert (a >= 0).all() and (a <= 1).all()


def test_blend_policy_alpha():
    head = BlendPolicy(3)
    F = torch.randn(8, 3)
    al = head.alpha(F)
    assert al.shape == (8,)
    assert ((al > 0) & (al < 1)).all()


def test_train_blend_runs():
    rng = np.random.default_rng(0)
    T = 400
    e = torch.tensor(rng.uniform(0.3, 0.8, T), dtype=torch.float32)
    F = torch.tensor(rng.normal(0, 1, (T, 3)), dtype=torch.float32)
    r = torch.tensor(rng.normal(0.0005, 0.01, T), dtype=torch.float32)
    head, v = train_blend(e[:300], F[:300], r[:300], e[300:], F[300:], r[300:], seed=1,
                          epochs=20)
    assert np.isfinite(v)