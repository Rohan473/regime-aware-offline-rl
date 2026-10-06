"""Render the representation-utility manuscript to a Word .docx.

markdown -> HTML (python-markdown) -> BeautifulSoup -> python-docx. Headings,
paragraphs with inline bold/italic/code, bullet lists, tables, fenced code
blocks, and the four figures inserted in-text at the same anchors as the PDF.

usage: python scripts/make_paper_docx.py
Output: paper/manuscript_representation_utility.docx
"""
from __future__ import annotations

from pathlib import Path

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SRC = PAPER / "manuscript_representation_utility.md"
OUT = PAPER / "manuscript_representation_utility.docx"

BODY_FONT = "Calibri"
MONO_FONT = "Consolas"
FIGS = [
    ("4.6 Why", "rep_mechanism_schematic.png",
     "Figure 1. Decision-relevant pathway: representation -> directional "
     "information (terminates, ~chance) / risk-state information -> adaptive "
     "exposure -> hybrid. Consistent across tested markets; paired Sharpe "
     "advantage not significant at 5%."),
    ("4.7 Exposure", "hybrid_mechanism.png",
     "Figure 2. Hybrid exposure pathway: a_t = e_t + alpha_t(1 - e_t)."),
    ("5. Discussion", "risk_return_frontier.png",
     "Figure 3. Risk-return map: the constant-exposure family lies on the "
     "buy-and-hold ray; adaptive/hybrid policies sit above it on positive-drift "
     "markets."),
    ("Appendix A", "rep_divergence_regime.png",
     "Figure A1. Divergence x regime: absolute per-regime slopes "
     "(bull +0.04, bear -1.04, crisis -4.05)."),
]


def add_runs(par, node, bold=False, italic=False, mono=False):
    for child in node.children:
        if isinstance(child, NavigableString):
            text = str(child)
            if text:
                run = par.add_run(text)
                run.bold = bold
                run.italic = italic
                if mono:
                    run.font.name = MONO_FONT
                    run.font.size = Pt(9.5)
        elif isinstance(child, Tag):
            if child.name in ("strong", "b"):
                add_runs(par, child, True, italic, mono)
            elif child.name in ("em", "i"):
                add_runs(par, child, bold, True, mono)
            elif child.name == "code":
                add_runs(par, child, bold, italic, True)
            elif child.name == "a":
                add_runs(par, child, bold, italic, mono)
            else:
                add_runs(par, child, bold, italic, mono)


def add_image(doc, fname, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(PAPER / fname), width=Inches(5.8))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    soup = BeautifulSoup(html, "html.parser")

    doc = Document()
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(1)
        s.left_margin = s.right_margin = Inches(1)
    style = doc.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = Pt(11)

    for el in soup.children:
        if not isinstance(el, Tag):
            continue
        name = el.name
        if name in ("h1", "h2", "h3"):
            htext = el.get_text(" ", strip=True)
            # insert any pending figure whose anchor matches this heading
            if FIGS and FIGS[0][0] in htext:
                _, fname, cap = FIGS.pop(0)
                add_image(doc, fname, cap)
            level = {"h1": 0, "h2": 1, "h3": 2}[name]
            if level == 0:
                doc.add_heading(htext, level=0)
            else:
                doc.add_heading(htext, level=level)
        elif name == "p":
            par = doc.add_paragraph()
            add_runs(par, el)
        elif name in ("ul", "ol"):
            for li in el.find_all("li", recursive=False):
                par = doc.add_paragraph(style="List Bullet" if name == "ul" else "List Number")
                add_runs(par, li)
        elif name == "table":
            rows = el.find_all("tr")
            if not rows:
                continue
            ncol = len(rows[0].find_all(["th", "td"]))
            tbl = doc.add_table(rows=0, cols=ncol)
            tbl.style = "Light Grid Accent 1"
            for r in rows:
                cells = r.find_all(["th", "td"])
                row = tbl.add_row().cells
                for i, c in enumerate(cells[:ncol]):
                    row[i].text = c.get_text(" ", strip=True)
        elif name == "pre":
            par = doc.add_paragraph()
            for i, line in enumerate(el.get_text().rstrip("\n").split("\n")):
                if i:
                    par.add_run().add_break()
                run = par.add_run(line)
                run.font.name = MONO_FONT
                run.font.size = Pt(8.5)
        elif name == "blockquote":
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Inches(0.4)
            add_runs(par, el)
            par.runs and setattr(par.runs[0], "italic", True)

    doc.save(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()