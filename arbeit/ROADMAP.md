# ROADMAP — W-Seminararbeit „Mathematik an der Börse"

Stand: 27.06.2026 (nach Abschluss der Quellen-Migration auf begutachtete Endfassungen).
Priorität: **[P1]** = Pflicht vor Abgabe · **[P2]** = empfohlene Verbesserung · **[P3]** = optional/Kür.
„(KI)" = kann ich (Claude) übernehmen, sag einfach Bescheid.

---

## A. Pflicht vor Abgabe

- [ ] **[P1] Seitenzahl prüfen & ggf. kürzen.** Bayer. Vorgabe i. d. R. **15–20 Seiten Text**
  (ohne Titelblatt, Verzeichnisse, Anhang). Manuskript ~8.000 Wörter; durch den Mathe-Ausbau
  vermutlich ~21–23 S. → **in Word/Pages zählen** (kein LibreOffice lokal, daher keine
  Auto-Zählung möglich). Falls Kürzung nötig: v. a. Kap. 4 (Methodik) und 6 (Implementierung)
  straffen, Wiederholungen entfernen — **Mathematik und Belege erhalten**. (KI: Kürzungsvorschläge)
- [ ] **[P1] Titelblatt-Platzhalter ausfüllen:** `[Name der Schule]`, `[Fach]`,
  `[Titel des W-Seminars]`, `[Vorname Nachname]`, `[Name der Lehrkraft]`, `[Ort]`, `[TT.MM.JJJJ]`.
- [ ] **[P1] Schlusserklärung einfügen & unterschreiben** — inkl. **KI-Offenlegung** (Pflicht:
  Nennung von Claude/Anthropic + Verwendungszweck: Recherche-/Schreibassistenz, Quellen-/Beleg­prüfung,
  Code). (KI: Entwurf der KI-Offenlegung)
- [ ] **[P1] In Word: Inhaltsverzeichnis aktualisieren** (F9) — sonst leer; Fußnoten und
  Seitenzahlen (ab S. 3) gegenprüfen.
- [ ] **[P1] Finale Korrekturlesung.** Besonders die in dieser Session geänderten Stellen:
  §2.9 (DeMiguel-Absatz, neue 3000/6000-Monate-Zahlen) und §5.1/§5.2 (Autorwechsel
  „Ledoit und Wolf"). Rechtschreibung, Zeichensetzung, einheitliche Begriffe.

## B. Inhaltliche Verbesserungen (empfohlen)

- [ ] **[P2] Abbildungen einbinden.** In `output1.6/` liegen fertige Grafiken; der Text hat bisher
  nur **eine Tabelle**. 1–3 zentrale Plots mit Bildunterschrift einbinden, z. B.
  `01_kumulierte_renditen.png` (Equity-Kurven), `06_rollierender_sharpe.png`,
  `02/03_gewichte_*.png` oder `07_feature_importance.png`. Stärkt Kap. 7 erheblich. (KI)
- [ ] **[P2] Neu vorhandene Quellen direkt belegen.** Black/Litterman (1992) als Fußnote dort,
  wo das Modell genannt wird (§2.9 und Ausblick §9.2); optional Ledoit/Wolf 2017
  (Nonlinear Shrinkage) im Ausblick als belegte Erweiterung. Macht bislang nur namentliche
  Erwähnungen überprüfbar. (KI)
- [ ] **[P2] Limitation „kleines Universum" ergänzen.** Nur 15 Titel — explizit als Grenze der
  externen Validität in Kap. 8 nennen (neben Survivorship Bias, Pauschalkosten, Einzelpfad). (KI)
- [ ] **[P3] Kurzzusammenfassung/Abstract** am Anfang, falls vom Seminar gewünscht.

## C. Reproduzierbarkeit / Zahlen-Konsistenz

- [ ] **[P1] Ergebniszahlen verifizieren.** Im Text genannte Werte (Sharpe 1,01 / 0,92 / 0,91 / 0,79;
  CAGR/Drawdown; p-Werte; DSR) **exakt** gegen `output1.6/performance_metrics.csv` u. a.
  abgleichen, damit Tabelle 7.1 und Fließtext mit dem letzten Lauf übereinstimmen. (KI)
- [ ] **[P3] Repro-Anhang.** Kurzer Anhang mit GitHub-Link, Kernparametern und Seeds für die
  Nachvollziehbarkeit (wissenschaftliche Redlichkeit).

## D. Notion / Quellenverwaltung

- [ ] **[P2] Autor-Feld korrigieren.** Beim Eintrag „Robust Performance Hypothesis Testing …"
  Olivier Ledoit als Koautor ergänzen (Konsistenz mit der zitierten Fassung Ledoit/Wolf 2008). (KI)
- [ ] **[P3] Jegadeesh/Titman-Duplikat** endgültig löschen (war beim letzten Stand archiviert).
- [ ] **[P3] Working-Paper-PDFs** bei den migrierten Einträgen (GKX, DeMiguel, Jagannathan/Ma)
  im „PDF"-Feld optional entfernen/kennzeichnen, damit die begutachtete Fassung führend ist.

## E. Backup / Abgabe-Logistik

- [ ] **[P1] `git push`** nach GitHub — `main` ist aktuell **20 Commits vor `origin/main`**
  (Backup hinkt nach). (KI, auf Freigabe)
- [ ] **[P1] Finale Abgabe-PDF** aus dem .docx erzeugen (Word/Pages „Als PDF exportieren";
  kein LibreOffice lokal).

---

### Erledigt (zur Orientierung)
- Fließtext Kap. 1–9 vollständig, mathematisch ausgebaut.
- Beleg-Audit: alle Behauptungen am Quellen-Volltext geprüft.
- **Quellen-Migration:** Working-Paper-Belege auf begutachtete Endfassungen umgestellt
  (GKX, DeMiguel, Jagannathan/Ma, Wolf→Ledoit/Wolf 2008), Seitenzahlen neu verifiziert,
  DeMiguel-Simulationszahlen korrigiert, Chopra/Ziemba-Seitenfehler behoben.
- Notion-DB „Quellen wSeminar": Inhalt, Relevanz (Auswahl) und Begründung für alle 29 Quellen gepflegt.
