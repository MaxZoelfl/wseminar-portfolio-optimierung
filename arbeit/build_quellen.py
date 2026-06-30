#!/usr/bin/env python3
"""
Baut arbeit/Quellen_erklaert.md als .docx nach den W-Seminar-Formalia.

Formalia: Times New Roman 12, Zeilenabstand 1,5; Ränder links 3,5 cm, rechts/oben/unten 2,5 cm;
Überschriften (H1/H2/H3) fett. Konvertierung mit pandoc unter Verwendung einer Referenz-Vorlage.

Aufruf:  venv/bin/python arbeit/build_quellen.py
Voraussetzung: pandoc auf dem PATH (z. B. /opt/homebrew/bin), python-docx.
"""
import os, subprocess
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_LINE_SPACING
from docx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD  = os.path.join(ROOT, "arbeit", "Quellen_erklaert.md")
REF = "/tmp/ref_quellen.docx"
OUT = os.path.join(ROOT, "arbeit", "Quellen_erklaert.docx")


def make_reference():
    subprocess.run(["pandoc", "-o", "/tmp/ref_default_q.docx",
                    "--print-default-data-file", "reference.docx"], check=True)
    doc = Document("/tmp/ref_default_q.docx")

    def set_font(style, size=None, bold=None, name="Times New Roman"):
        style.font.name = name
        rpr = style.element.get_or_add_rPr(); rf = rpr.get_or_add_rFonts()
        for a in ("w:ascii", "w:hAnsi", "w:cs"):
            rf.set(qn(a), name)
        if size is not None: style.font.size = Pt(size)
        if bold is not None: style.font.bold = bold

    S = doc.styles
    n = S["Normal"]; set_font(n, size=12)
    n.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    n.paragraph_format.space_after = Pt(0); n.paragraph_format.space_before = Pt(0)
    names = [s.name for s in S]
    for h, sz in (("Heading 1", 15), ("Heading 2", 13), ("Heading 3", 12)):
        if h in names:
            st = S[h]; set_font(st, size=sz, bold=True)
            st.paragraph_format.space_before = Pt(12); st.paragraph_format.space_after = Pt(6)
            st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    for sec in doc.sections:
        sec.left_margin = Cm(3.5); sec.right_margin = Cm(2.5)
        sec.top_margin = Cm(2.5); sec.bottom_margin = Cm(2.5)
    doc.save(REF)


def main():
    make_reference()
    subprocess.run(["pandoc", MD, "-o", OUT, "--reference-doc=" + REF, "-f", "markdown"],
                   check=True)
    print(f"✓ {OUT}")


if __name__ == "__main__":
    main()
