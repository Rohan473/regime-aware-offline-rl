"""Model D (fuzzy + transformer + IQL) configuration.

Model D is the Phase-4 test of the project's central hypothesis: an agent
that makes its uncertainty about market regime EXPLICIT before the policy
network sees the state should beat exposure-matched control (EM) and the
Phase-2/3 baselines (B: naive_new DDR, C: TACR). Uncertainty is encoded by
a FIXED (non-learned) interval type-2 fuzzy layer over three volatility /
momentum / mean-reversion-sensitive features; the raw 8-d state is passed
through unchanged alongside. The policy is extracted by Implicit Q-Learning
(IQL, Kostrikov et al., ICLR 2022) over the four Phase-1 behavior-policy
trajectories, using a causal transformer (same block hyperparameters as
Model C) as the shared state encoder.

Two variants share this config (the fuzzy layer is a toggle, per the
Phase-4 spec — a separate fuzzy layer, not a separate package):

- ``fuzzy: true``   -> D (full): state input = 8 raw + 18 fuzzy = 26-d.
- ``fuzzy: false``  -> D-minus-fuzzy (ablation): state input = 8 raw.
  The transformer backbone, IQL machinery and ALL hyperparameters are
  identical; the input projection width is the only structural difference.
  No zero-padding is added to the 8-d variant to equalize input widths —
  that would change what the model sees, which is precisely the tested
  treatment (deliberate, flagged in the Phase-4 spec).

IQL hyperparameters are sourced from the IQL paper (Kostrikov et al. 2022,
Offline Reinforcement Learning with Implicit Q-Learning, ICLR; code
github.com/ikostrikov/implicit_q_learning):
  - expectile tau = 0.7        (paper's reported best across tasks)
  - temperature   = 3.0        (paper's AWR advantage temperature, beta)
  - lr            = 3e-4       (paper's default for all three networks)
  - batch_size    = 256 (paper) -> 64 here (CPU-scaling deviation, same
                                   batch size Model C uses; the 2x context
                                   forwards for (s, s') already double the
                                   per-step compute)
  - gamma         = 0.99       (paper)
  - tau (polyak)  = 0.005      (paper target network EMA)
  - weight_decay  = 0.0        (paper does not use weight decay)
  - grad clip     = none       (paper does not use gradient clipping;
                                Model C's 0.25 was the TACR paper's value)

Deviations / documented choices (also in PROJECT_NOTES 7.9):
  - training budget: paper ~1M steps on GPU; ours 3k (10 x 300) for the
    5-seed screen, 20k (40 x 500) for the pre-registered escalation —
    CPU-scaled like Models B/C, samples-per-step and batch size identical
    across both D variants so the ablation is compute-matched.
  - warmup_steps 1000 (linear warmup to lr; scaled to the 3k budget like
    Model C's 10k -> 1k).
  - transformer: state ENCODER at each timestep (causal window ending at
    t), NOT an RTG-conditioned autoregressive generator — IQL's value/Q/
    policy heads consume the per-timestep representation h_t (see spec
    section 3; the RTG channel of Model C is intentionally absent).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from src.data.technical_factors import FEATURE_COLUMNS
from src.models import WINDOW_DAYS

CONFIG_YAML = Path(__file__).resolve().parents[3] / "configs" / "model_d.yaml"

# Three fuzzy features, three Gaussian terms (low/mid/high), upper + lower
# memberships (interval type-2) => 3 * 3 * 2 = 18 fuzzy features.
FUZZY_FIELDS = ("realized_vol_20d", "ret_20d", "rsi_14")
N_TERMS = 3
MF_WIDTH_SCALE = 0.5
FOU_SCALE = 0.8
RAW_STATE_DIM = len(FEATURE_COLUMNS)  # 8
FUZZY_N_FEATURES = len(FUZZY_FIELDS) * N_TERMS * 2  # 18


@dataclass
class DConfig:
    # --- architecture (transformer block sizes = Model C's values) ---
    u: int = WINDOW_DAYS               # context length in days; == WINDOW_DAYS
    embed_dim: int = 128               # Model C value
    n_layer: int = 4                   # Model C value
    n_head: int = 1                    # Model C value
    n_inner: int = 512                 # Model C value
    dropout: float = 0.1               # Model C value
    max_ep_len: int = 5283             # longest trajectory (timestep embedding)
    fuzzy: bool = True                 # True = D; False = D-minus-fuzzy (8-d)

    # --- fuzzy layer (FIXED interval type-2 Gaussian membership functions) ---
    fuzzy_fields: tuple[str, ...] = FUZZY_FIELDS
    n_terms: int = N_TERMS
    mf_width_scale: float = MF_WIDTH_SCALE  # UMF std = scale * bin width
    fou_scale: float = FOU_SCALE            # LMF std = UMF std * fou_scale

    # --- IQL (Kostrikov et al. 2022 — see module docstring for sourcing) ---
    expectile: float = 0.7
    temperature: float = 3.0
    lr: float = 3e-4
    weight_decay: float = 0.0
    gamma: float = 0.99
    tau: float = 0.005

    # --- training ---
    epochs: int = 10                   # val-eval cadence (3k steps total)
    steps_per_epoch: int = 300
    batch_size: int = 64               # paper 256; CPU-scaled (documented)
    warmup_steps: int = 1000           # linear warmup, scaled to the 3k budget
    seed: int = 20260814

    # --- eval / artifacts ---
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent / "checkpoints")
    device: str = "cpu"

    @property
    def raw_state_dim(self) -> int:
        return RAW_STATE_DIM

    @property
    def fuzzy_n_features(self) -> int:
        return len(self.fuzzy_fields) * self.n_terms * 2

    @property
    def state_in_dim(self) -> int:
        """Per-spec: D = 8 + 18 = 26; D-minus-fuzzy = 8, NO padding."""
        return RAW_STATE_DIM + (self.fuzzy_n_features if self.fuzzy else 0)

    @classmethod
    def from_yaml(cls, path: Path | str = CONFIG_YAML) -> "DConfig":
        """Build from configs/model_d.yaml (unknown keys ignored)."""
        raw = OmegaConf.load(str(path))
        d = asdict(cls())
        for section in ("model", "fuzzy", "iql", "training", "eval"):
            if section in raw:
                d.update({k: v for k, v in raw[section].items() if k in d})
        cfg = cls(**d)
        cfg.checkpoint_dir = Path(cfg.checkpoint_dir)
        if not cfg.checkpoint_dir.is_absolute():
            cfg.checkpoint_dir = Path(__file__).parent / cfg.checkpoint_dir
        cfg.validate()
        return cfg

    def validate(self) -> None:
        problems = []
        if self.u != WINDOW_DAYS:
            problems.append(
                f"u={self.u} deviates from WINDOW_DAYS={WINDOW_DAYS} — the B/C/D "
                "comparison requires identical information horizons"
            )
        if self.u < 2:
            problems.append("u must be >= 2")
        if self.embed_dim < self.state_in_dim:
            problems.append("embed_dim must be >= state_in_dim")
        if self.n_layer < 1:
            problems.append("n_layer must be >= 1")
        if self.n_head < 1:
            problems.append("n_head must be >= 1")
        if not 0.5 <= self.mf_width_scale <= 2.0:
            problems.append("mf_width_scale must be in [0.5, 2.0]")
        if not 0.0 < self.fou_scale <= 1.0:
            problems.append("fou_scale must be in (0, 1] (LMF std <= UMF std)")
        if len(set(self.fuzzy_fields)) != len(self.fuzzy_fields):
            problems.append("fuzzy_fields must be unique")
        if not set(self.fuzzy_fields).issubset(set(FEATURE_COLUMNS)):
            problems.append(f"fuzzy_fields must be a subset of FEATURE_COLUMNS: {FEATURE_COLUMNS}")
        if self.n_terms < 2:
            problems.append("n_terms must be >= 2")
        if not 0.0 < self.expectile < 1.0:
            problems.append("expectile must be in (0, 1)")
        if self.temperature <= 0:
            problems.append("temperature must be > 0")
        if self.lr <= 0:
            problems.append("lr must be > 0")
        if not 0.0 < self.gamma < 1.0:
            problems.append("gamma must be in (0, 1)")
        if not 0.0 < self.tau <= 1.0:
            problems.append("tau must be in (0, 1]")
        if self.epochs < 1 or self.steps_per_epoch < 1:
            problems.append("epochs and steps_per_epoch must be >= 1")
        if self.batch_size < 1:
            problems.append("batch_size must be >= 1")
        if self.max_ep_len < self.u:
            problems.append("max_ep_len must be >= u")
        if problems:
            raise ValueError("invalid DConfig: " + "; ".join(problems))