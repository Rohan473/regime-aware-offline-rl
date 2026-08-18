"""Model C (TACR) configuration.

Reproduces Lee & Moon, "Transformer Actor-Critic with Regularization:
Automated Stock Trading using Reinforcement Learning" (IEEE Access 2023,
DOI 10.1109/ACCESS.2023.3324458; authors' code github.com/VarML/TACR) on
the project's continuous-action market.

Values mirror the authors' single-index (NDX/MDAX/CSI) setup unless the
project spec overrides them (see ``configs/tacr.yaml`` and the deviations
list in the package docstring):
  - u=20 (the paper's context length for single-index daily datasets) ==
    ``WINDOW_DAYS`` so Model C sees the same information horizon as B.
  - L decoder blocks default 4 (user-specified range 3-4; the paper used 5).
  - alpha=0.9 BC-regularization weight (paper's NDX/MDAX/CSI value).
  - critic learning rate 1e-6, gamma 0.99, tau 0.005, AdamW 1e-4 + wd 1e-4,
    gradient clip 0.25, batch 64 (all from the paper). Warmup and total step
    budget are scaled for CPU (10k -> 1k warmup; 40k -> 3k steps); see
    ``configs/tacr.yaml`` and the deviations list in the package docstring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.models import WINDOW_DAYS

CONFIG_YAML = Path(__file__).resolve().parents[3] / "configs" / "tacr.yaml"


@dataclass
class TACRConfig:
    # --- architecture (paper values; L per user spec) ---
    u: int = WINDOW_DAYS                  # context length in MDP triples
    embed_dim: int = 128                  # paper
    n_layer: int = 4                      # paper 5; user spec default 3-4 (deviation)
    n_head: int = 1                       # paper
    n_inner: int = 512                    # paper: 4 * embed_dim
    dropout: float = 0.1                  # paper
    max_ep_len: int = 5283                # longest trajectory (timestep embedding)
    action_head: str = "tanh"             # paper: linear+softmax (discrete); ours: linear+tanh (continuous)

    # --- training (paper values; budget/warmup scaled for CPU, see yaml) ---
    epochs: int = 10                      # val-eval cadence (3k steps total; paper 40k)
    steps_per_epoch: int = 300
    batch_size: int = 64                  # paper
    lr: float = 1e-4                      # paper
    weight_decay: float = 1e-4            # paper
    warmup_steps: int = 1000              # paper 10k (40k-step schedule); scaled to 3k steps
    grad_clip: float = 0.25               # paper
    alpha: float = 0.9                    # paper's BC-regularization weight (NDX/MDAX/CSI)
    gamma: float = 0.99                   # paper
    tau: float = 0.005                    # paper
    critic_lr: float = 1e-6               # paper default for NDX/MDAX/CSI
    seed: int = 20260814

    # --- eval / artifacts ---
    rtg_target: float = 0.0               # constant return-to-go at roll time (paper feeds zeros)
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent / "checkpoints")
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: Path | str = CONFIG_YAML) -> "TACRConfig":
        """Build from configs/tacr.yaml (unknown keys ignored)."""
        from omegaconf import OmegaConf

        raw = OmegaConf.load(str(path))
        d = asdict(cls())
        d.update({k: v for k, v in raw.model.items() if k in d})
        d.update({k: v for k, v in raw.training.items() if k in d})
        d.update({k: v for k, v in raw.eval.items() if k in d})
        cfg = cls(**d)
        cfg.checkpoint_dir = Path(cfg.checkpoint_dir)
        if not cfg.checkpoint_dir.is_absolute():
            cfg.checkpoint_dir = Path(__file__).parent / cfg.checkpoint_dir
        cfg.validate()
        return cfg

    def validate(self) -> None:
        problems = []
        if self.u < 2:
            problems.append("u (context length) must be >= 2")
        if self.u != WINDOW_DAYS:
            problems.append(
                f"u={self.u} deviates from WINDOW_DAYS={WINDOW_DAYS} — the B/C "
                "comparison requires identical information horizons"
            )
        if self.embed_dim < 8:
            problems.append("embed_dim must be >= 8")
        if self.n_layer < 1:
            problems.append("n_layer must be >= 1")
        if self.n_head < 1:
            problems.append("n_head must be >= 1")
        if self.action_head not in ("tanh", "gaussian"):
            problems.append("action_head must be 'tanh' or 'gaussian'")
        if self.epochs < 1 or self.steps_per_epoch < 1:
            problems.append("epochs and steps_per_epoch must be >= 1")
        if self.batch_size < 1:
            problems.append("batch_size must be >= 1")
        if self.lr <= 0 or self.critic_lr <= 0:
            problems.append("lr and critic_lr must be > 0")
        if not 0.0 <= self.alpha <= 10.0:
            problems.append("alpha must be in [0, 10]")
        if not 0.0 < self.gamma < 1.0:
            problems.append("gamma must be in (0, 1)")
        if not 0.0 < self.tau <= 1.0:
            problems.append("tau must be in (0, 1]")
        if self.max_ep_len < self.u:
            problems.append("max_ep_len must be >= u")
        if problems:
            raise ValueError("invalid TACRConfig: " + "; ".join(problems))
