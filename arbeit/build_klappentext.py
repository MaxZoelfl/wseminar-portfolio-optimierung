#!/usr/bin/env python3
"""
Baut den Klappentext als .docx nach den W-Seminar-Formalien aus arbeit/Klappentext.md.

Formalia (W-Seminararbeit_Formales.pdf):
  - Times New Roman 12, Zeilenabstand 1,5
  - Ränder: links 3,5 cm, rechts/oben/unten 2,5 cm
  - Blocksatz (optional laut Formalia, hier verwendet)

Markdown-Konvention:
  "# ..."  -> Titelzeile (fett, zentriert)
  "## ..." -> Untertitel/Arbeitstitel (kursiv, zentriert)
  sonstige nichtleere Zeile -> Textabsatz (Blocksatz)

Aufruf:  venv/bin/python arbeit/build_klappentext.py
"""
import os
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "arbeit", "Klappentext.md")
OUT = os.path.join(ROOT, "arbeit", "Klappentext.docx")


def set_font(run_or_style, name="Times New Roman"):
    run_or_style.font.name = name
    rpr = run_or_style.element.get_or_add_rPr()
    rf = rpr.get_or_add_rFonts()
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(a), name)


def main():
    doc = Document()

    normal = doc.styles["Normal"]
    normal.font.size = Pt(12)
    set_font(normal)
    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)

    for sec in doc.sections:
        sec.left_margin = Cm(3.5)
        sec.right_margin = Cm(2.5)
        sec.top_margin = Cm(2.5)
        sec.bottom_margin = Cm(2.5)

    lines = open(MD, encoding="utf-8").read().splitlines()
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("## "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(18)
            r = p.add_run(s[3:]); set_font(r); r.font.size = Pt(12); r.italic = True
        elif s.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(6)
            r = p.add_run(s[2:]); set_font(r); r.font.size = Pt(14); r.bold = True
        else:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_after = Pt(6)
            r = p.add_run(s); set_font(r); r.font.size = Pt(12)

    doc.save(OUT)
    words = sum(len(l.split()) for l in lines if l.strip() and not l.strip().startswith("#"))
    print(f"✓ {OUT}")
    print(f"  Textabsätze gebaut | ~{words} Wörter Fließtext")


if __name__ == "__main__":
    main()
