"""Conceptual figure: information -> exposure pathway (hybrid result).

Solid boxes = established findings; the exposure -> hybrid gate -> utility chain
is the mechanism supported by the hybrid experiment. Output:
paper/hybrid_mechanism.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, text, w=2.6, h=0.9, fc="#eef3fb", ec="#3b5b8c"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5)


def arrow(ax, p, q, dashed=False, color="#333333"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12,
                                 lw=1.2, color=color,
                                 linestyle="--" if dashed else "-",
                                 shrinkA=2, shrinkB=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box(ax, 5, 9.3, "32 point-in-time features", fc="#dfe9f7")
    box(ax, 5, 7.9, "Learned representation h_t")
    box(ax, 2.3, 6.3, "Directional information\n~ chance", fc="#f6e3e3", ec="#8c3b3b")
    box(ax, 7.7, 6.3, "Risk-state information\nvol / OOD / drawdown", fc="#e6f3e6", ec="#3b8c4f")
    box(ax, 2.3, 5.0, "No reliable sign", fc="#f6e3e3", ec="#8c3b3b")
    box(ax, 7.7, 5.0, "Adaptive exposure e_t", fc="#e6f3e6", ec="#3b8c4f")
    box(ax, 5, 3.9, "Risk-aware hybrid gate\na = e + alpha(1-e)")
    box(ax, 2.5, 2.6, "Low risk\n-> toward buy-and-hold")
    box(ax, 7.5, 2.6, "High risk\n-> toward adaptive exposure")
    box(ax, 5, 1.2, "Improved risk-adjusted utility", w=3.2, fc="#eef7ee", ec="#3b8c4f")

    arrow(ax, (5, 8.85), (5, 8.35))
    for x in (2.3, 7.7):
        arrow(ax, (5, 7.45), (x, 6.75))
    arrow(ax, (2.3, 5.85), (2.3, 5.45))
    arrow(ax, (7.7, 5.85), (7.7, 5.45))
    arrow(ax, (2.3, 4.55), (4.2, 4.35), dashed=True)
    arrow(ax, (7.7, 4.55), (5.8, 4.35))
    arrow(ax, (5, 3.45), (2.9, 3.05))
    arrow(ax, (5, 3.45), (7.1, 3.05))
    arrow(ax, (2.5, 2.15), (4.4, 1.6))
    arrow(ax, (7.5, 2.15), (5.6, 1.6))

    ax.text(0.2, 0.3, "dashed arrow = proposed mechanism (hybrid experiment); solid = established",
            fontsize=7.5, color="#555555")
    fig.tight_layout()
    out = ROOT / "paper" / "hybrid_mechanism.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()