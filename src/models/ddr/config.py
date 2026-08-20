"""Model B (DDR) hyperparameters.

``window_size`` defaults to the SHARED ``WINDOW_DAYS`` from
``src/models/__init__.py`` so B/C/D use identical windowing. Do not override
it in CLI invocations; tests may construct small configs with their own
values for speed.

The vol-targeted DSR reward fix (``vol_targeting``) is documented in
``configs/ddr.yaml`` — the shared config loaded by ``DDRConfig.from_yaml``
in the train/eval entry points. Dataclass defaults keep tests and the
naive-DSR behavior intact; the yaml flips on vol-targeting and writes to
``checkpoints/vt`` so old artifacts are never overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

from src.models import WINDOW_DAYS

CONFIG_YAML = Path(__file__).resolve().parents[3] / "configs" / "ddr.yaml"


@dataclass
class DDRConfig:
    window_size: int = WINDOW_DAYS      # shared across B/C/D
    hidden_size: int = 32               # GRU/LSTM hidden units (32-64 range)
    rnn_type: str = "gru"               # "gru" | "lstm"
    eta: float = 0.01                   # DSR EMA adaptation rate (Moody-Saffell range 0.01-0.1)
    warmup_steps: int = 20              # DSR steps before D_t is emitted (EMA seeding)
    lr: float = 1e-3
    epochs: int = 30                 # best-val checkpoint selection; overfit sets in after ~epoch 8-11
    seed: int = 20260814
    transaction_cost_bps: float = 0.0   # cost applied per trade: a*R - cost*|a|
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent / "checkpoints")
    device: str = "cpu"

    # --- volatility-targeted DSR reward fix ---
    vol_targeting: bool = False         # False = naive DSR on raw a_t * ret_{t+1}
    target_vol: float = 0.15            # annualized target vol for the scaled series
    vol_target_window: int = 20         # trailing (causal) window for realized vol
    max_leverage: float = 2.0           # clip bounds for the vol-scaled action

    # --- direct exposure-level regularization (the under-leverage fix) ---
    # Penalizes |mean(|a_t|) - target_exposure| over each training block, so
    # the DSR's level-free gradient cannot silently de-leverage the policy.
    # Unlike a variance floor, this targets the exact drifted quantity
    # (mean absolute position) and cannot be gamed by erratic flipping
    # without moving the very quantity being penalized.
    exposure_regularization: bool = False
    target_exposure: float = 0.75       # vol-implied: 0.15 target / ~0.18-0.20 SPY ann vol
    exposure_lambda: float = 1.0        # penalty weight vs -D.mean() (O(0.1-1))

    @classmethod
    def from_yaml(cls, path: Path | str = CONFIG_YAML) -> "DDRConfig":
        """Build a DDRConfig from configs/ddr.yaml (unknown keys ignored,
        so tests and older callers keep dataclass defaults)."""
        from omegaconf import OmegaConf

        raw = OmegaConf.load(str(path))
        d = asdict(cls())
        d.update({k: v for k, v in raw.model.items() if k in d})
        cfg = cls(**d)
        cfg.checkpoint_dir = Path(cfg.checkpoint_dir)
        if not cfg.checkpoint_dir.is_absolute():
            # anchor relative yaml paths (e.g. "checkpoints/vt") to the
            # module's checkpoints dir, NOT the CWD
            cfg.checkpoint_dir = Path(__file__).parent / cfg.checkpoint_dir
        cfg.validate()
        return cfg

    def validate(self) -> None:
        problems = []
        if self.window_size < 2:
            problems.append("window_size must be >= 2")
        if self.hidden_size < 1:
            problems.append("hidden_size must be >= 1")
        if self.rnn_type not in ("gru", "lstm"):
            problems.append("rnn_type must be 'gru' or 'lstm'")
        if not 0.0 < self.eta <= 0.5:
            problems.append("eta must be in (0, 0.5]")
        if self.warmup_steps < 0:
            problems.append("warmup_steps must be >= 0")
        if self.epochs < 1:
            problems.append("epochs must be >= 1")
        if self.lr <= 0:
            problems.append("lr must be > 0")
        if self.transaction_cost_bps < 0:
            problems.append("transaction_cost_bps must be >= 0")
        if self.vol_targeting:
            if self.target_vol <= 0:
                problems.append("target_vol must be > 0 when vol_targeting")
            if self.vol_target_window < 2:
                problems.append("vol_target_window must be >= 2 when vol_targeting")
            if self.max_leverage <= 0:
                problems.append("max_leverage must be > 0 when vol_targeting")
        if self.exposure_regularization:
            if self.target_exposure <= 0:
                problems.append("target_exposure must be > 0 when exposure_regularization")
            if self.exposure_lambda <= 0:
                problems.append("exposure_lambda must be > 0 when exposure_regularization")
        if problems:
            raise ValueError("invalid DDRConfig: " + "; ".join(problems))