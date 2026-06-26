#!/usr/bin/env python3
"""
Reproduzierbarer Bau der W-Seminararbeit als .docx aus arbeit/arbeit.md.

Pipeline:
  1. Pandoc-Referenz mit Formalia-Stilen erzeugen (TNR 12, 1,5-zeilig, Ränder,
     Fußnoten 10).
  2. Markdown aufbereiten: HTML-Kommentar entfernen, [[FN: ...]] -> ^[...]
     (Pandoc-Inline-Fußnoten).
  3. Pandoc: Markdown -> docx mit Inhaltsverzeichnis (--toc) und echten Fußnoten.
  4. Nachbearbeitung (python-docx): Titelblatt, Abschnittswechsel,
     Seitennummerierung (Titel + Inhaltsverz. ohne Nummer, Text ab S. 3),
     Schlusserklärungs-Platzhalter, Anhang.

Aufruf:  venv/bin/python arbeit/build_docx.py
Voraussetzung: pandoc auf dem PATH (z. B. /opt/homebrew/bin), python-docx.
"""
import re, copy, subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD   = os.path.join(ROOT, "arbeit", "arbeit.md")
REF  = "/tmp/ref.docx"
BODY = "/tmp/arbeit_body.docx"
PMD  = "/tmp/arbeit_pandoc.md"
OUT  = os.path.join(ROOT, "arbeit", "W-Seminararbeit.docx")

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def step1_reference():
    subprocess.run(["pandoc", "-o", "/tmp/ref_default.docx",
                    "--print-default-data-file", "reference.docx"], check=True)
    doc = Document("/tmp/ref_default.docx")

    def set_font(style, name="Times New Roman", size=None, bold=None):
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
    for fn in ("Footnote Text", "Fußnotentext"):
        try:
            st = S[fn]; set_font(st, size=10)
            st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        except KeyError:
            pass
    for t, sz in (("Title", 22), ("Subtitle", 14)):
        try: set_font(S[t], size=sz)
        except KeyError: pass
    for sec in doc.sections:
        sec.left_margin = Cm(3.5); sec.right_margin = Cm(2.5)
        sec.top_margin = Cm(2.5); sec.bottom_margin = Cm(2.5)
    doc.save(REF)


def step2_preprocess():
    src = open(MD, encoding="utf-8").read()
    src = re.sub(r"^<!--.*?-->\s*", "", src, count=1, flags=re.DOTALL)
    n = len(re.findall(r"\[\[FN:", src))
    src = re.sub(r"\[\[FN:\s*(.*?)\]\]", r"^[\1]", src)
    open(PMD, "w", encoding="utf-8").write(src)
    return n


def step3_pandoc():
    subprocess.run(["pandoc", PMD, "-o", BODY, "--reference-doc=" + REF,
                    "--toc", "--toc-depth=3", "-f", "markdown+tex_math_dollars"],
                   check=True)


def step4_postprocess():
    doc = Document(BODY); body = doc.element.body
    paras = doc.paragraphs
    toc_anchor = paras[0]
    einleitung = next(p for p in paras if p.style.name.startswith("Heading")
                      and p.text.strip().startswith("1 Einleitung"))

    def ins(anchor, text, *, size=12, bold=False, align="center", sa=0, italic=False):
        p = anchor.insert_paragraph_before(); p.style = doc.styles["Normal"]
        p.alignment = {"center": WD_ALIGN_PARAGRAPH.CENTER,
                       "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
        p.paragraph_format.space_after = Pt(sa); p.paragraph_format.space_before = Pt(0)
        if text:
            r = p.add_run(text); r.font.name = "Times New Roman"
            r.font.size = Pt(size); r.bold = bold; r.italic = italic
        return p

    ins(toc_anchor, "[Name der Schule]", sa=2)
    ins(toc_anchor, "W-Seminar im Fach [Fach] — [Titel des W-Seminars]")
    for _ in range(4): ins(toc_anchor, "")
    ins(toc_anchor, "Markowitz versus Random Forest", size=22, bold=True, sa=4)
    ins(toc_anchor, "Ein statistisch abgesicherter Vergleich von Portfolio­optimierungs­strategien",
        size=14, italic=True)
    for _ in range(5): ins(toc_anchor, "")
    ins(toc_anchor, "Seminararbeit", bold=True, sa=2)
    ins(toc_anchor, "von [Vorname Nachname]")
    for _ in range(4): ins(toc_anchor, "")
    ins(toc_anchor, "Betreuende Lehrkraft: [Name der Lehrkraft]", sa=2)
    ins(toc_anchor, "Abgabetermin: [TT.MM.JJJJ]", sa=2)
    ins(toc_anchor, "[Ort], [TT.MM.JJJJ]")
    toc_anchor.insert_paragraph_before().add_run().add_break(WD_BREAK.PAGE)
    h = toc_anchor.insert_paragraph_before("Inhaltsverzeichnis"); h.style = doc.styles["Heading 1"]

    # Abschnittswechsel vor Kapitel 1
    final = body.find(qn("w:sectPr"))
    sect1 = copy.deepcopy(final)
    for t in sect1.findall(qn("w:type")): sect1.remove(t)
    typ = OxmlElement("w:type"); typ.set(qn("w:val"), "nextPage"); sect1.insert(0, typ)
    prev = einleitung._p.getprevious()
    while prev is not None and prev.tag != qn("w:p"): prev = prev.getprevious()
    prev.get_or_add_pPr().append(sect1)

    secs = doc.sections
    s2 = secs[1]; s2.footer.is_linked_to_previous = False
    fp = s2.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    for kind in ("begin", "text", "end"):
        if kind == "text":
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
            it.text = " PAGE "; run._r.append(it)
        else:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); run._r.append(fc)
    run.font.name = "Times New Roman"; run.font.size = Pt(11)
    secs[0].footer.is_linked_to_previous = False

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.add_paragraph("Schlusserklärung").style = doc.styles["Heading 1"]
    doc.add_paragraph("[Hier fügt der Verfasser die unterschriebene Schlusserklärung gemäß den "
                      "Formalien ein (inkl. Angabe der verwendeten KI-Tools und ihres Verwendungszwecks).]")
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.add_paragraph("Anhang").style = doc.styles["Heading 1"]
    doc.add_paragraph("Der vollständige Quellcode der Implementierung, sämtliche erzeugten Abbildungen "
                      "(kumulierte Renditen, Gewichtsverläufe, Effizienzlinie, rollierender Sharpe, "
                      "Feature-Importance/SHAP, Stress-Test, Turnover-Analyse) sowie die Protokolldateien "
                      "(CSV/JSON) befinden sich im begleitenden Repository. Die genauen Wortlaute der "
                      "verwendeten KI-Prompts sind hier aufzuführen.")
    doc.save(OUT)
    return len(doc.sections), len(doc.tables)


if __name__ == "__main__":
    step1_reference()
    nfn = step2_preprocess()
    step3_pandoc()
    nsec, ntbl = step4_postprocess()
    print(f"✓ {OUT}")
    print(f"  Fußnoten: {nfn} | Abschnitte: {nsec} | Tabellen: {ntbl}")
