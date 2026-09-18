"""Conceptual mechanism figure (reviewer-proof framing).

Draws the empirical observations and the PROPOSED (untested) mechanism
separately: solid arrows are observed associations from the experiments,
dashed arrows are the research hypothesis. Output:
paper/rep_mechanism_schematic.png.
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


def box(ax, x, y, text, w=2.2, h=0.9, fc="#eef3fb", ec="#3b5b8c", style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=style, fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=9, wrap=True)


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

    box(ax, 5, 9.2, "EMPIRICAL OBSERVATIONS", w=4.0, h=0.8, fc="#dfe9f7")
    box(ax, 1.6, 7.4, "Representation")
    box(ax, 5.0, 7.4, "Decision\nalgorithm", fc="#f7e9df", ec="#8c5b3b")
    box(ax, 8.4, 7.4, "Market\nregime")
    box(ax, 5.0, 5.5, "Policy divergence\n(from behavior)", fc="#f7e9df", ec="#8c5b3b")
    box(ax, 2.6, 3.4, "Representation\nsensitivity")
    box(ax, 7.4, 3.4, "Regime-dependent\nutility")
    box(ax, 5.0, 1.2, "Research hypothesis", w=3.0, h=0.8, fc="#eef7ee", ec="#3b8c4f")

    for x in (1.6, 5.0, 8.4):
        arrow(ax, (5, 8.8), (x, 7.85))
    arrow(ax, (5.0, 6.95), (5.0, 5.95))          # algorithm -> divergence
    arrow(ax, (1.6, 6.95), (2.4, 3.85))          # representation -> sensitivity
    arrow(ax, (5.0, 5.05), (3.0, 3.85), dashed=True)   # divergence -> sensitivity (hypothesis)
    arrow(ax, (8.4, 6.95), (7.6, 3.85))          # regime -> utility
    arrow(ax, (2.6, 2.95), (4.4, 1.6), dashed=True)
    arrow(ax, (7.4, 2.95), (5.6, 1.6), dashed=True)

    ax.text(3.1, 4.45, "associated with\n(descriptive, n=4)", fontsize=7.5,
            color="#666666", ha="center")
    ax.text(0.2, 0.35, "solid = observed association   |   dashed = proposed mechanism",
            fontsize=8, color="#555555")

    fig.tight_layout()
    out = ROOT / "paper" / "rep_mechanism_schematic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
