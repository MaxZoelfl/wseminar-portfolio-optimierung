# Die Nachrechnungs-Skripte

Jede Zahl der Seminararbeit lässt sich hier **ohne den großen Backtest**
nachvollziehen: aus den eingefrorenen Kursen und den Ergebnisdateien, die der
Backtest einmal erzeugt hat. Je Kapitel gibt es ein Skript. Es geht die Zahlen
**in derselben Reihenfolge wie die Arbeit** durch, rechnet jede eigene Zahl aus
Kursen, Ergebnisdateien oder Programmcode aus und zeigt, wie sie entsteht.
Zahlen aus der Literatur erscheinen an ihrer Stelle mit ihrer Fußnote.

---

## Schnellstart

Im Stammordner des Repositories, bei aktivierter virtueller Umgebung
(Einrichtung: siehe [README im Stammordner](../README.md#loslegen)):

```bash
python skripte/alle_kapitel.py
```

Das dauert rund anderthalb Minuten, liest nur und schreibt keine Datei. Am Ende
steht eine Übersicht, woher die Zahlen jedes Kapitels stammen:

```
                berechnet  Literatur  Definition/Annahme
  Kapitel 1             1         11                   0
  Kapitel 2            56          6                   0
  …
  zusammen            529         32                   7
```

---

## Wie man die Ausgabe liest

Jede Zahl steht in einer eigenen Zeile: links der Wert, den die **Rechnung**
ergibt, gerundet auf so viele Stellen wie in der Arbeit, rechts seine Bedeutung.
Eine Zeile mit `↳` darunter erklärt, wie die Zahl entsteht, woher sie kommt oder
wie sie ungerundet lautet:

```
  § 2.2 Diversifikation und ihre Grenze
  ──────────────────────────────────────────────────────────────────────
  17,8 %             Volatilität von Coca-Cola
  29,7 %             Volatilität von Caterpillar
  8,9 %              Rendite von Coca-Cola allein
  15                 Zahl der untersuchten Titel
  0,88 Pp            Abstand zwischen 15 Titeln und Grenzwert
                     ↳ 17,56 % − 16,68 %
  …
  § 2.5 Schätzfehler als Kernproblem
  ──────────────────────────────────────────────────────────────────────
  10 Jahre           Beobachtungsdauer, gerundet
                     ↳ ungerundet 9,90 Jahre
  0,08               Jobson/Korkie: Sharpe-Quotient des geschätzten Optimums
                     ↳ aus der Literatur (Fn 38)
```

Jeder Eintrag gehört zu einer von drei Arten:

| Art | Bedeutung |
|---|---|
| **berechnet** | aus Kursen, Ergebnisdateien oder Programmcode gerechnet. Steht eine Einstellung nur im Code (etwa die Mindestzahl von 252 Handelstagen), wird sie dort ausgelesen. |
| **Literatur** | eine Zahl aus einer Quelle; sie wird nicht gerechnet, sondern mit ihrer Fußnote genannt. |
| **Definition/Annahme** | etwa die Schwellen 70/30 des RSI, ein Rechenbeispiel oder die elf Sektoren des GICS-Standards, jeweils mit Erklärung. |

**Gerundet wird kaufmännisch und einmal**, vom genauen Wert aus: 8,946 wird auf
eine Stelle zu 8,9, 8,95 zu 9,0; 0,602499… auf drei Stellen zu 0,602 (nicht erst
auf vier Stellen zu 0,6025 und dann zu 0,603). Bei Angaben wie „rund“ oder
„etwa“ steht unter der Zahl zusätzlich ihr ungerundeter Wert.

**Jede Zahl erscheint einmal je Abschnitt.** Nennt ein Abschnitt dieselbe Zahl
zweimal (etwa in der Bildunterschrift und im Fließtext), steht sie beim ersten Mal.

---

## Die Dateien

| Datei | Inhalt | Dauer |
|---|---|---|
| **`alle_kapitel.py`** | startet alle Kapitel nacheinander und fasst zusammen | ~1,5 min |
| `kapitel_1_einleitung.py` | § 1.1–1.2: Zahlen aus der Literatur und die Umrechnung 3000 Monate = 250 Jahre | <1 s |
| `kapitel_2_theorie.py` | § 2.2–2.6: Zwei-Anlagen-Fall, Diversifikationsgrenze, Effizienzrand, t-Werte, Konfidenzintervalle, Shrinkage in den 119 Schätzfenstern | ~5 s |
| `kapitel_3_random_forest.py` | § 3.1–3.5: Prognosefehler, Lernstoff je Termin, Einzelbäume der Tiefe 1 und 14 an allen Terminen, Korrelation der Bäume, gewählte Baumtiefen | ~15 s |
| ↳ `… --baumkorrelation` | rechnet die Korrelation der Bäume (§ 3.3) neu, statt sie aus der Referenzdatei zu lesen, und vergleicht beide | +55 min |
| `kapitel_4_methodik.py` | § 4.1–4.5: Einstellungen, Termine, Schutzvorkehrungen, Lernstoff, Umschlagschranke, Vergleichslauf ohne Purging | ~10 s |
| `kapitel_5_signifikanz.py` | § 5.1–5.4: Standardfehler, Autokorrelation, Testeinstellungen, Gegenrechnung des Tests so, wie Ledoit/Wolf ihn beschreiben | ~30 s |
| `kapitel_6_ergebnisse.py` | § 6.1–6.6: Kennzahlen, Portfoliostruktur, Signifikanztabelle, Kostensensitivität, Zinsschwellen, NVIDIA-Effekt | ~40 s |
| `kapitel_7_fazit.py` | § 7.1: wiederholte p-Werte, Vergleich mit DeMiguel u. a. | <1 s |
| `anhang.py` | Anhang A–C: Bildunterschriften (für A3 wird der zuletzt trainierte Wald nachgebaut), Tabellen A1–A3 und B1–B4, Parameter C1, Angaben in C2 samt Vergleich mit einem zweiten, unabhängigen Lauf | ~35 s |
| `abbildung_kostensensitivitaet.py` | **erzeugt** Abbildung 5 (`output/13_kosten_sensitivitaet.png`) und `output/kosten_sensitivitaet.csv` neu | ~5 s |
| `gegenrechnung_drift.py` | beziffert die festen Gewichte innerhalb des Monats ([LIMITATIONS.md](../LIMITATIONS.md) § 11.1); nicht Teil von `alle_kapitel.py`, weil diese Zahlen nicht in der Arbeit stehen | ~15 s |
| `_werkzeuge.py` | gemeinsame Bausteine (Daten laden, Zahlen aus dem Code lesen, runden, ausgeben). Wird nicht selbst gestartet. | — |

Jedes Kapitel-Skript lässt sich auch einzeln starten, etwa
`python skripte/kapitel_6_ergebnisse.py`. Alle funktionieren aus jedem Ordner heraus.
Nur `abbildung_kostensensitivitaet.py` schreibt Dateien, alle anderen lesen nur.

---

## Welche Formeln die Skripte benutzen

**Keine eigenen.** Jede Formel kommt aus dem Paket `portfolio/` — aus genau dem
Code, der auch den Backtest und die Abbildungen rechnet:

| Formel der Arbeit | steht in |
|---|---|
| Sharpe-Quotient (13/14), CAGR, Volatilität, Drawdown, Calmar, Sortino (Tabelle B1) | `portfolio/metrics.py` |
| Formeln 2 bis 11 und 17 (Zwei-Anlagen-Fall, Diversifikationsgrenze, Effizienzrand, Konfidenzintervalle) | `portfolio/theory_plots.py`, Abschnitt „Die Formeln aus Kapitel 2" |
| rollierender Sharpe-Quotient (Abbildung A4) | `portfolio/plots.py`, `rollierender_sharpe` |
| Test von Ledoit/Wolf (28–30), Holm (32), Deflated Sharpe Ratio (33/34) | `portfolio/significance.py` |
| Turnover (26), Risk Parity (19), Merkmale des Waldes (Tabelle A2) | `portfolio/backtest.py`, `optimizers.py`, `indicators.py` |

Drei Stellen stehen bewusst noch einmal in den Skripten, weil das Paket sie
nicht einzeln herausgibt und sein Rechenkern unverändert bleiben soll — alle
mit Begründung im Code:

- **Formel 35** (effektive Titelzahl) in `_werkzeuge.py`: Der Backtest braucht
  sie nicht, nur Kapitel 6.
- **Formel 28/29** in `kapitel_5_signifikanz.py`: für die Gegenrechnung „wie in
  der Quelle“. Das Skript prüft an den Originaldaten, dass beide Fassungen
  dasselbe ergeben.
- **Der Lernstoff des Waldes**: `kapitel_3_random_forest.py` baut ihn Zeile für
  Zeile so nach, wie `backtest.py` ihn ausschneidet (genutzt auch von Kapitel 4
  und vom Anhang).

---

## Woher die Daten kommen

Alle Skripte lesen dieselben Dateien, die der große Backtest (`python -m portfolio`)
einmal erzeugt hat:

| Datei | Inhalt |
|---|---|
| `data/prices.pkl` | eingefrorene Schlusskurse der 15 Titel und des SPY, Abruf 15.08.2026 |
| `output/daily_returns.csv` | Tagesrenditen der vier Strategien |
| `output/weights_markowitz.csv`, `weights_rf.csv`, `weights_risk_parity.csv` | Gewichte je Rebalancing-Termin |
| `output/turnover.csv` | Umschlag je Strategie und Termin |
| `output/experiment_log.json` | Einstellungen, Kennzahlen und Signifikanztests mit voller Genauigkeit |

Dazu kommen drei kleine Referenzdateien in `skripte/daten/`:

- **`rf_hyperparameter.csv`**: Die Einstellungen, die die Hyperparametersuche an
  jedem der 119 Termine gewählt hat (Zahl der Bäume, Tiefe, Blattgröße). Sie
  stammen aus dem Laufprotokoll des Laufs vom 15.08.2026, dessen Ergebnisdateien
  mit denen des maßgeblichen Laufs bitgleich sind. Das Protokoll selbst liegt
  nicht im Repository, weil es rechnerspezifische Pfade enthält.
- **`baumkorrelation.csv`**: je Termin die Korrelation der Baumprognosen, die
  Fehlerkorrelation nach Breiman und das Verhältnis von Wald- zu Baumfehler
  (§ 3.3). Erzeugt mit `kapitel_3_random_forest.py --baumkorrelation --speichern`,
  weil die Rechnung rund 55 Minuten dauert. Die Spalte `wie_lauf` zeigt, dass
  die neu getunten Wälder dieselben Einstellungen wählen wie im Backtest.
- **`vergleichslauf_ohne_purging.csv`**: Tagesrenditen eines Vergleichslaufs mit
  gewöhnlicher zeitlicher Kreuzvalidierung statt Purged & Embargoed CV, sonst
  identisch mit dem maßgeblichen Lauf (dieselbe Kursdatei, einkernig). Er ist die
  Grundlage der Zahl 0,961 in § 4.3 (wie viel Informationsleckage der Schutz
  verhindert). Gleichgewichtung und Risk Parity stimmen mit dem maßgeblichen Lauf
  bitgenau überein, Markowitz auf drei Nachkommastellen (siehe LIMITATIONS.md,
  Abschnitt 12).

**Zur Genauigkeit:** `output/daily_returns.csv` ist auf sechs Nachkommastellen
gerundet. Für Renditen und Volatilitäten genügt das. Für die vierte Stelle eines
Sharpe-Quotienten nicht immer, deshalb verwenden die Skripte dort die exakten
Werte aus `output/experiment_log.json`.

---

## Wenn der Backtest neu läuft

Dann ändern sich die Ergebnisdateien in `output/`, und die Skripte rechnen
automatisch mit den neuen Werten. Mit unveränderter `data/prices.pkl`, denselben
Paketversionen und unter macOS 26.6.1 reproduziert der Backtest die Dateien in
`output/` bitgenau. In einer anderen Umgebung können einzelne Zahlen in der
letzten Stelle abweichen (siehe [LIMITATIONS.md](../LIMITATIONS.md),
Abschnitt 12). Die Referenzdatei `baumkorrelation.csv` muss nach einem Lauf mit
anderen Einstellungen neu erzeugt werden.

---

## Für den Verfasser: Abgleich mit dem Wortlaut der Arbeit

Die Skripte lesen die Word-Datei nicht. In jedem Skript steht zu jeder Zahl
auch, wie sie in der Arbeit geschrieben ist, zum Beispiel in `kapitel_2_theorie.py`:

```python
p.zahl("Volatilität von Coca-Cola", "17,8", sA * 100, "%")
#       ↑ was die Zahl bedeutet     ↑ so in der Arbeit  ↑ die Rechnung
```

Aus diesem Eintrag liest das Werkzeug die Stellenzahl, auf die gerundet wird.
Zwei Schalter vergleichen außerdem Rechnung und Arbeit:

```bash
python skripte/alle_kapitel.py --abgleich
python skripte/alle_kapitel.py --text pfad/zur/Arbeit.docx
```

`--abgleich` zeigt statt des gerechneten Werts den eingetragenen Wortlaut und
markiert jede Zahl: ✅ stimmt · ≈ stimmt ungefähr (bei „rund“, „etwa“) ·
❌ weicht ab · ⏭ nicht gerechnet. `--text` tut dasselbe und meldet am Ende jede
eingetragene Zahl, die im zugehörigen Abschnitt der angegebenen Fassung nicht
vorkommt. Für `.docx` braucht es [pandoc](https://pandoc.org); eine
Markdown-Fassung geht ohne. Ändert sich eine Zahl in der Arbeit, wird nur der
mittlere Eintrag angepasst; die Rechnung bleibt, wie sie ist.
