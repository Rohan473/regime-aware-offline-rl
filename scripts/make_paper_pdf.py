"""Render the representation-utility manuscript to a publication-style PDF.

markdown -> HTML (python-markdown) -> PDF (xhtml2pdf). Registers a Unicode TTF
(Arial) so Greek/math glyphs render, embeds the four figures in-text, and
applies a clean 1in-margin print stylesheet.

usage: python scripts/make_paper_pdf.py
Output: paper/manuscript_representation_utility.pdf
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import markdown
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SRC = PAPER / "manuscript_representation_utility.md"
OUT = PAPER / "manuscript_representation_utility.pdf"

def _font_uri(name: str) -> str | None:
    p = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / name
    return p.as_uri() if p.exists() else None


def register_fonts() -> dict:
    """Register a Unicode font family so Greek/math glyphs render."""
    uris = {"reg": _font_uri("DejaVuSans.ttf"), "bold": _font_uri("DejaVuSans-Bold.ttf"),
            "mono": _font_uri("DejaVuSansMono.ttf")}
    try:
        if uris["reg"]:
            pdfmetrics.registerFont(TTFont("Paper", uris["reg"].replace("file:///", "")))
        if uris["bold"]:
            pdfmetrics.registerFont(TTFont("Paper-Bold", uris["bold"].replace("file:///", "")))
            pdfmetrics.registerFontFamily("Paper", normal="Paper", bold="Paper-Bold",
                                          italic="Paper", boldItalic="Paper-Bold")
    except Exception:
        pass
    return uris


def fig_block(src: str, caption: str) -> str:
    return (f'<div class="fig"><img src="{src}"/><p class="cap">{caption}</p></div>')


def insert_before(html: str, anchor: str, block: str) -> str:
    idx = html.find(anchor)
    if idx < 0:
        print(f"[warn] anchor not found: {anchor!r}")
        return html
    hstart = html.rfind("<h", 0, idx)
    return html[:hstart] + block + html[hstart:]


def build_html() -> str:
    text = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list",
                          "md_in_html"])
    # figures (absolute file:// paths so xhtml2pdf resolves them)
    mech = (PAPER / "rep_mechanism_schematic.png").as_uri()
    hyb = (PAPER / "hybrid_mechanism.png").as_uri()
    front = (PAPER / "risk_return_frontier.png").as_uri()
    div = (PAPER / "rep_divergence_regime.png").as_uri()
    body = insert_before(body, "4.6 Why", fig_block(mech, "Figure 2. Decision mechanism: representation -> directional information (terminates, ~chance) / risk-state information -> adaptive exposure -> risk-aware hybrid."))
    body = insert_before(body, "4.7 Exposure", fig_block(hyb, "Figure 2b. Hybrid exposure pathway: a_t = e_t + alpha_t(1 - e_t)."))
    body = insert_before(body, "5. Discussion", fig_block(front, "Figure 4. Risk-return map: constant-exposure family on the buy-and-hold ray; adaptive/hybrid above it."))
    body = insert_before(body, "Appendix A", fig_block(div, "Figure A1. Divergence x regime: absolute per-regime slopes (bull +0.04, bear -1.04, crisis -4.05)."))
    return body


def main() -> None:
    u = register_fonts()
    ff = ""
    if u["reg"]:
        ff += f"@font-face {{ font-family: Paper; src: url('{u['reg']}'); }}\n"
    if u["bold"]:
        ff += f"@font-face {{ font-family: Paper; font-weight: bold; src: url('{u['bold']}'); }}\n"
    if u["mono"]:
        ff += f"@font-face {{ font-family: Mono; src: url('{u['mono']}'); }}\n"
    body = build_html()
    css = f"""
    {ff}
    @page {{ size: letter; margin: 1in; }}
    body {{ font-family: Paper; font-size: 10.5pt; line-height: 1.45; color: #111; }}
    h1 {{ font-size: 19pt; margin: 0 0 6pt 0; }}
    h2 {{ font-size: 14pt; margin: 16pt 0 6pt 0; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }}
    h3 {{ font-size: 11.5pt; margin: 12pt 0 4pt 0; }}
    p {{ margin: 5pt 0; text-align: left; }}
    table {{ border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; }}
    th, td {{ border: 1px solid #bbb; padding: 3pt 5pt; text-align: left; }}
    th {{ background: #eef3fb; }}
    pre {{ font-family: Mono; font-size: 8pt; background: #f6f6f6; padding: 6pt;
           border: 1px solid #ddd; white-space: pre-wrap; }}
    code {{ font-family: Mono; font-size: 9pt; }}
    .fig {{ text-align: center; margin: 12pt 0; }}
    .fig img {{ width: 88%; }}
    .cap {{ font-size: 8.5pt; color: #444; margin-top: 4pt; text-align: center; }}
    .cover {{ text-align: center; margin: 120pt 0 24pt 0; }}
    .cover .t {{ font-size: 22pt; font-weight: bold; }}
    .cover .s {{ font-size: 11pt; color: #555; margin-top: 8pt; }}
    """
    html = (f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>"
            f"<div class='cover'><div class='t'>What determines risk-adjusted utility "
            f"in offline financial reinforcement learning?</div>"
            f"<div class='s'>Representation, decision algorithm, and exposure</div>"
            f"<div class='s'>Manuscript draft</div></div>{body}</body></html>")
    with OUT.open("wb") as fh:
        res = pisa.CreatePDF(src=html, dest=fh, encoding="utf-8")
    if res.err:
        print(f"[error] xhtml2pdf reported {res.err} errors")
        sys.exit(1)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()