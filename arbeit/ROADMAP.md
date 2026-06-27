# ROADMAP — W-Seminararbeit „Mathematik an der Börse"

Stand: 27.06.2026. Priorität: **[P1]** Pflicht vor Abgabe · **[P2]** empfohlen · **[P3]** optional.

---

## ✅ In dieser Session erledigt (automatisiert)
- **Quellen-Migration** auf begutachtete Endfassungen (GKX, DeMiguel, Jagannathan/Ma,
  Wolf→Ledoit/Wolf 2008); alle Seitenzahlen am Volltext verifiziert; DeMiguel-Simulationszahlen
  korrigiert (3000/6000 Monate); Chopra/Ziemba-Seitenfehler behoben.
- **Ergebniszahlen verifiziert:** Tabelle 7.1, p-Werte (0,69/0,53/0,95/0,004/0,48),
  Holm (0,018) und DSR (0,997/0,995/0,994/0,986) stimmen exakt mit `output1.6/` überein.
- **Drei Abbildungen eingebunden:** Abb. 1 Feature-Importance (§4.2), Abb. 2 kumulierte
  Renditen (§7.1), Abb. 3 rollierender Sharpe (§7.2). Build bettet die PNGs ins .docx ein.
- **Neue Quellen direkt belegt:** Black/Litterman 1992 (§2.9, §9.2) und Ledoit/Wolf 2017 (§9.2);
  Literaturverzeichnis ergänzt (jetzt 26 Quellen, 117 Fußnoten).
- **Limitation „nur 15 Titel"** in §8.3 ergänzt.
- **Notion-DB:** Inhalt/Relevanz/Begründung für alle 29 Quellen gepflegt; Olivier Ledoit als
  Koautor beim Eintrag „Robust Performance Hypothesis Testing…" ergänzt.

---

## ⚠️ Was DU noch tun musst (kann ich nicht selbst)

### [P1] 1. Titelblatt-Platzhalter ausfüllen
In der fertigen `arbeit/W-Seminararbeit.docx` ersetzen:
`[Name der Schule]`, `[Fach]`, `[Titel des W-Seminars]`, `[Vorname Nachname]`,
`[Name der Lehrkraft]`, `[Ort]`, `[TT.MM.JJJJ]` (Abgabetermin + Datum).
> Der Arbeitstitel im Dokument lautet aktuell „Markowitz versus Random Forest" — bei Bedarf anpassen.

### [P1] 2. Schlusserklärung + KI-Offenlegung einfügen & unterschreiben
Im .docx ist dafür ein Platzhalter. Pflichtbestandteil ist die Offenlegung der KI-Nutzung.
**Fertiger Entwurf zum Anpassen/Einfügen:**

> **Schlusserklärung**
> Hiermit erkläre ich, dass ich die vorliegende Seminararbeit selbstständig und ohne fremde
> Hilfe verfasst, keine anderen als die angegebenen Quellen und Hilfsmittel benutzt und die
> den benutzten Quellen wörtlich oder inhaltlich entnommenen Stellen als solche kenntlich
> gemacht habe.
>
> **Offenlegung KI-gestützter Hilfsmittel:** Zur Unterstützung wurde das KI-Sprachmodell
> Claude (Anthropic) eingesetzt. Verwendungszweck: Recherche- und Schreibassistenz,
> Strukturierung und sprachliche Überarbeitung des Fließtexts, Abgleich von Behauptungen mit
> den Originalquellen samt Verifikation der Seitenangaben sowie Unterstützung bei der
> Programmierung des Backtests (Python). Sämtliche inhaltlichen Aussagen, die Auswahl der
> Quellen und die wissenschaftliche Verantwortung liegen bei mir; alle Belege wurden anhand
> der Primärquellen überprüft.
>
> _______________________   ______________________________
> Ort, Datum                  Unterschrift

### [P1] 3. In Word öffnen und finalisieren
- **Inhaltsverzeichnis aktualisieren:** Rechtsklick aufs Verzeichnis → „Feld aktualisieren" →
  „Gesamtes Verzeichnis" (oder Cursor ins TOC, **F9**). Sonst bleibt es leer.
- **Seitenzahl prüfen** (kein LibreOffice lokal, daher nicht automatisch zählbar): Bayer.
  Vorgabe i. d. R. **15–20 Seiten Text** (ohne Titelblatt/Verzeichnisse/Anhang). Falls über
  Limit → siehe Punkt 4.
- **Abbildungen kontrollieren:** Größe/Umbruch der drei Grafiken prüfen; ggf. skalieren.

### [P1] 4. Falls über Seitenlimit: kürzen ODER Abbildungen verschieben
- Straffen v. a. Kap. 4 (Methodik) und 6 (Implementierung) sowie Wiederholungen —
  **Mathematik und Belege erhalten**. (Sag Bescheid, ich liefere konkrete Kürzungsvorschläge.)
- Alternativ 1–2 Abbildungen in den **Anhang** verschieben (zählt nicht zum Textlimit).

### [P1] 5. Abgabe-PDF erzeugen
In Word/Pages „Als PDF exportieren" (kein LibreOffice lokal verfügbar).

---

## Optionale Verbesserungen (P2/P3) — sag Bescheid, dann übernehme ich
- **[P2]** Korrekturlesung: vollständiger Stil-/Tippfehler-Durchgang über den Fließtext.
- **[P3]** Weitere Abbildungen (Gewichtsverläufe, Effizienzlinie, Stress-Test) in den Anhang.
- **[P3]** Repro-Anhang: GitHub-Link, Kernparameter, Seeds explizit auflisten.
- **[P3]** Notion: bei den migrierten Einträgen die alten Working-Paper-PDFs im „PDF"-Feld
  entfernen, damit die begutachtete Fassung führend ist.
