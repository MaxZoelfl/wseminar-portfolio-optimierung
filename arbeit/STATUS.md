# STATUS / Handover — W-Seminararbeit „Mathematik an der Börse"

Stand: 27.06.2026. Diese Notiz orientiert eine **neue Session** in 1 Minute.

## Worum es geht
W-Seminararbeit (bayer. Gymnasium, Leitthema **„Mathematik an der Börse"**):
empirischer Vergleich von **Markowitz-MVO, Random Forest, Equal Weight (1/N) und
Risk Parity** über einen Walk-Forward-Backtest (2015–2024), mit **statistischer
Signifikanzprüfung** (Ledoit-Wolf 2008, Holm, Deflated Sharpe, Purged CV).
Kernbefund: **keine** aktive Strategie schlägt 1/N statistisch signifikant.

## Wo was liegt
- **Manuskript:** `arbeit/arbeit.md` (Markdown, Kap. 1–9 + Literaturverzeichnis).
  **Zitierweise: APA (7. Aufl.)** — In-Text-Kurzbelege `(Autor, Jahr, S. X)`,
  **keine Fußnoten mehr** (frühere `[[FN: …]]`-Marker wurden ersetzt). ~8.000 Wörter,
  117 Belege, 26 Quellen, 3 Abbildungen (aus `output1.6/`, via `--resource-path`).
  Roadmap: `arbeit/ROADMAP.md`. Hinweis: APA weicht vom Schul-Zitierformat ab.
- **Fertiges Dokument:** `arbeit/W-Seminararbeit.docx` (Formalia: TNR 12, 1,5-zeilig,
  Ränder L3,5/R2,5/O2,5/U2,5, echte Word-Fußnoten 10pt, Titelblatt mit Platzhaltern,
  automatisches Inhaltsverzeichnis, Seitenzahlen ab S. 3).
- **Build:** `venv/bin/python arbeit/build_docx.py` (braucht `pandoc` auf PATH).
  Erzeugt das .docx neu aus `arbeit.md` (Pandoc + python-docx). **Nach jeder
  Manuskript-Änderung neu ausführen.**
- **Code (kanonisch):** Paket `portfolio/` (config, data, indicators, optimizers,
  metrics, significance, cross_validation, backtest, dashboard, plots, run).
  `projekt1.6.py` = eingefrorene v4.1-Referenz. Ergebnisse in `output1.6/`.
- **GitHub:** https://github.com/maxzoelfl-source/wseminar-portfolio-optimierung (privat, Branch `main`).
- **Quellen:** `Quellen/*.pdf` (24 Paper) — **via .gitignore NICHT im Repo** (Urheberrecht).
- **Notion:** verbunden; Projektseite „Portfoliooptimierung: Markowitz vs.
  KI-basierte Strategien" + Quellen-PDFs + Notizen vorhanden (noch nicht abgeglichen).

## Was FERTIG ist
- Code refaktoriert, **31 pytest-Tests grün**, v4.1-Methodikfixes (einfache Renditen,
  fairer Turnover), Signifikanzmodul + Purged CV, voller Lauf erzeugt `output1.6/`.
- Fließtext Kap. 1–9 vollständig, mathematisch ausgebaut (Lagrange-Herleitung von
  Effizienzrand/Tangentialportfolio in 2.4, Varianz/Diversifikation in 2.3,
  ERC-Mathematik in 2.10).
- **Beleg-Audit abgeschlossen:** alle Behauptungen gegen den Quellen-Volltext
  geprüft — inhaltlich korrekt; keine Falschdarstellungen.
- **Quellen-Migration (27.06.) abgeschlossen:** Working-Paper-Belege auf die
  begutachteten Endfassungen umgestellt (BSB = via Bayer. Staatsbibliothek
  beschaffte Verlagsfassungen). Neu gemappte Fußnoten-Seiten: GKX → RFS 2020
  (S. 2223–2234), DeMiguel → RFS 2009 (S. 1915/1916/1942; Simulationszahlen
  korrigiert: 3000/6000 Monate statt „50/500/1000 Jahre"), Jagannathan/Ma → JoF
  2003 (S. 1651/1657), Wolf → **Ledoit/Wolf 2008** JEF (S. 850–854, Autorwechsel).
  Chopra/Ziemba: 1 Seitenkorrektur (S. 9 → S. 10). Literaturverzeichnis +
  Notion-DB „Quellen wSeminar" (Eigenschaft *Inhalt*, 29 Quellen) entsprechend
  gepflegt. **Noch nicht committet.** Hinweis: Jegadeesh/Titman ist in der
  Notion-DB doppelt angelegt (Duplikat ggf. löschen).

## Was OFFEN ist (Priorität)
1. **KÜRZEN auf ≤ 20 Seiten Text.** Durch den Mathe-Ausbau ist der Fließtext
   vermutlich bei ~21–23 Seiten (Formalia: 15–20, ohne Anhang). Straffen v. a. in
   Kap. 4 (Methodik) und 6 (Implementierung) und bei Wiederholungen; **Mathematik
   und Belege erhalten**. (Echte Seitenzahl in Word prüfen.)
2. **Titelblatt-Platzhalter** ausfüllen: `[Name der Schule]`, `[Fach]`,
   `[Titel des W-Seminars]`, `[Vorname Nachname]`, `[Name der Lehrkraft]`, Daten, `[Ort]`.
3. **Schlusserklärung** (macht der Verfasser selbst; Platzhalter im docx) — inkl.
   **KI-Offenlegung** (Pflicht: Claude/Anthropic + Verwendungszweck).
4. In Word: **Inhaltsverzeichnis aktualisieren** (F9) — sonst leer.
5. Optional: **Notion-Projektseite abgleichen** (offizielle Vorgaben/Gliederung?).

## Gotchas / Werkzeug-Hinweise
- `/tmp`-PDF-Extrakte werden zwischen Sessions gelöscht → für Quellenarbeit neu
  extrahieren: `pdftotext -layout "Quellen/X.pdf" /tmp/x.txt` (poppler unter
  `/opt/homebrew/bin`; ggf. `export PATH="/opt/homebrew/bin:$PATH"`).
- Gescannte PDFs (Chopra/Ziemba, Michaud, Elton) haben keine Textebene → mit
  `pdftoppm -png -r 140 ...` als Bild rendern und per Read-Tool lesen.
- Zitier-Hinweise (im Literaturverzeichnis vermerkt): Sharpe = Online-Nachdruck
  (Seiten 1–15 statt JPM 49–58). Nach der Migration sind nur noch wenige Quellen
  als Arbeitspapier-/Online-Fassung zitiert (Harvey/Liu/Zhu, Merton, Maillard,
  Bailey et al. PBO, Qian, Sharpe) → deren Fußnoten-Seiten passen zu DIESEN
  Fassungen; alle übrigen verweisen auf die Verlags-Endfassung.
- Headless: `MPLBACKEND=Agg venv/bin/python ...` für Tests/Plots.
