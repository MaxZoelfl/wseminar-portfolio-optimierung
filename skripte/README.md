# Die Nachrechnungs-Skripte

Jede Zahl der Seminararbeit lässt sich hier **ohne den großen Backtest**
nachprüfen: aus den eingefrorenen Kursen und den Ergebnisdateien, die der
Backtest einmal erzeugt hat. Je Kapitel gibt es ein Skript. Es geht die Zahlen
**in derselben Reihenfolge wie der Text** durch, rechnet jede eigene Zahl aus
und vergleicht sie mit dem Wortlaut. Zahlen aus der Literatur erscheinen an
ihrer Stelle mit ihrer Fußnote.

Wer nur wissen will, ob noch alles stimmt, braucht einen einzigen Befehl.

---

## Schnellstart

Im Stammordner des Repositories, bei aktivierter virtueller Umgebung
(Einrichtung: siehe [README im Stammordner](../README.md#loslegen)):

```bash
python skripte/alle_kapitel.py
```

Das dauert gut eine Minute, liest nur und schreibt keine Datei. Am Ende steht
eine Übersicht wie diese (die Zahlen hier sind nur ein Beispiel):

```
                 ✅ stimmt  ≈ ungefähr   ❌ weicht ab   ⏭ offen
  Kapitel 1             1           0             0        11
  Kapitel 2            56           0             0         6
  …
```

---

## Wie man die Ausgabe liest

Jede Zahl steht in einer eigenen Zeile — zuerst **genau so, wie sie im Text
steht**, dahinter, was sie bedeutet:

```
  § 2.2 Diversifikation und ihre Grenze
  ──────────────────────────────────────────────────────────────────────
  ✅  17,8 %           Volatilität von Coca-Cola
  ✅  29,7 %           Volatilität von Caterpillar
  ✅  8,9 %            Rendite von Coca-Cola allein
  ✅  fünfzehn         Zahl der untersuchten Titel
  …
  § 2.5 Schätzfehler als Kernproblem
  ──────────────────────────────────────────────────────────────────────
  ⏭   0,08             Jobson/Korkie: Sharpe-Quotient des geschätzten Optimums
                       ↳ Zahl aus der Quelle (Fn 38)
```

| Zeichen | Bedeutung | Was zu tun ist |
|:---:|---|---|
| ✅ | Die Rechnung ergibt, auf dieselbe Stellenzahl gerundet, **genau** die Zahl im Text. | nichts |
| ≈ | Der Text sagt „rund", „etwa" oder „gut", und die Rechnung liegt nah genug daran. Dahinter steht der genaue Wert. | nichts |
| ❌ | Die Rechnung ergibt etwas anderes als der Text. Dahinter steht der gerechnete Wert mit mehr Stellen. | Textstelle prüfen |
| ⏭ | Die Zahl wird hier nicht gerechnet — meist eine Zahl aus der Literatur. Die Zeile darunter sagt, warum. | nichts |

Eine Zeile mit `↳` darunter erklärt die Zahl genauer, etwa welcher Titel gemeint
ist oder wie der Text gerundet hat.

**Gerundet wird kaufmännisch**, also so wie von Hand: 8,946 wird auf eine Stelle
zu 8,9, 8,95 zu 9,0. Die Stellenzahl liest das Skript aus dem Text ab: Steht
dort „17,56", wird auf zwei Stellen gerundet, steht dort „17,6", auf eine.
Zahlwörter („fünfzehn", „zehn Basispunkte") versteht es ebenfalls.

**Gerundet wird einmal, vom genauen Wert aus:** 0,602499… wird auf drei Stellen
zu 0,602 — nicht erst auf vier Stellen zu 0,6025 und dann zu 0,603.

**Jede Zahl erscheint einmal je Abschnitt.** Nennt ein Abschnitt dieselbe Zahl
zweimal (etwa in der Bildunterschrift und im Text), wird sie beim ersten Mal
geprüft.

---

## Die Dateien

| Datei | Prüft | Dauer |
|---|---|---|
| **`alle_kapitel.py`** | startet alle Kapitel nacheinander und fasst zusammen | ~1 min |
| `kapitel_1_einleitung.py` | § 1.1–1.2: nur Zahlen aus der Literatur (⏭) und die Umrechnung „3000 Monate, also 250 Jahre" | <1 s |
| `kapitel_2_theorie.py` | § 2.2–2.6: Zwei-Anlagen-Fall, Diversifikationsgrenze, Effizienzrand, t-Werte, Konfidenzintervalle, Shrinkage in den 119 Schätzfenstern | ~5 s |
| `kapitel_3_random_forest.py` | § 3.1–3.5: Prognosefehler, Lernstoff je Termin, Einzelbäume der Tiefe 1 und 14 an allen Terminen, Korrelation der Bäume, gewählte Baumtiefen | ~15 s |
| ↳ `… --baumkorrelation` | rechnet die Korrelation der Bäume (§ 3.3) neu, statt sie aus der Referenzdatei zu lesen, und vergleicht | +55 min |
| `kapitel_4_methodik.py` | § 4.1–4.5: Einstellungen, Termine, Schutzvorkehrungen, Lernstoff, Umschlagschranke | ~10 s |
| `kapitel_5_signifikanz.py` | § 5.1–5.4: Standardfehler, Autokorrelation, Testeinstellungen, Gegenrechnung des Tests so, wie Ledoit/Wolf ihn beschreiben | ~30 s |
| `kapitel_6_ergebnisse.py` | § 6.1–6.6: Kennzahlen, Portfoliostruktur, Signifikanztabelle, Kostensensitivität, Zinsschwellen, NVIDIA-Effekt | ~40 s |
| `kapitel_7_fazit.py` | § 7.1: wiederholte p-Werte, Vergleich mit DeMiguel u. a. | <1 s |
| `anhang.py` | Anhang A–C: Bildunterschriften, Tabellen A1–A3 und B1–B4, Parameter C1, Angaben in C2 | ~5 s |
| `abbildung_kostensensitivitaet.py` | **erzeugt** Abbildung 5 (`output/13_kosten_sensitivitaet.png`) und `output/kosten_sensitivitaet.csv` neu | ~5 s |
| `gegenrechnung_drift.py` | beziffert die festen Gewichte innerhalb des Monats ([LIMITATIONS.md](../LIMITATIONS.md) § 11.1); nicht Teil von `alle_kapitel.py`, weil diese Zahlen nicht im Text stehen | ~15 s |
| `_werkzeuge.py` | gemeinsame Bausteine (Daten laden, Runden, Soll-Ist-Vergleich, Wortlaut-Abgleich). Wird nicht selbst gestartet. | — |

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
  der Quelle". Das Skript prüft an den Originaldaten, dass beide Fassungen
  dasselbe ergeben.
- Außerdem baut `kapitel_3_random_forest.py` den Lernstoff des Waldes Zeile für
  Zeile so nach, wie `backtest.py` ihn ausschneidet.

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
  stammen aus dem Laufprotokoll des maßgeblichen Laufs. Das Protokoll selbst
  liegt nicht im Repository, weil es rechnerspezifische Pfade enthält.
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

## Wenn sich der Text ändert

**Die Skripte lesen die Word-Datei nicht.** Die Zahl, wie sie im Text steht, ist
im Skript als Zeichenkette eingetragen, zum Beispiel in `kapitel_2_theorie.py`:

```python
p.zahl("Volatilität von Coca-Cola", "17,8", sA * 100, "%")
#       ↑ was die Zahl bedeutet     ↑ so steht es im Text   ↑ die Rechnung
```

Ändert sich eine Zahl im Text, wird nur die Zeichenkette in der Mitte angepasst.
Die Rechnung bleibt, wie sie ist. Neue Sätze mit Zahlen bekommen eine neue
Zeile an der passenden Stelle — die Reihenfolge im Skript folgt dem Text.

Damit kein Eintrag veraltet, ohne dass es auffällt, gibt es den Wortlaut-Abgleich:

```bash
python skripte/alle_kapitel.py --text pfad/zur/Arbeit.docx
```

Am Ende der Ausgabe steht dann jede eingetragene Zahl, die **im zugehörigen
Abschnitt** dieser Fassung der Arbeit nicht vorkommt — meist ein Zeichen, dass
die Stelle im Text geändert wurde. Für `.docx` braucht es
[pandoc](https://pandoc.org); eine Markdown-Fassung des Textes geht ohne. Mit
`--text` endet das Skript mit Fehlercode 1, wenn etwas fehlt.

## Wenn der Backtest neu läuft

Dann ändern sich die Ergebnisdateien in `output/`, und die Skripte rechnen
automatisch mit den neuen Werten. Mit unveränderter `data/prices.pkl`, denselben
Paketversionen und unter macOS 26.6.1 reproduziert der Backtest die Dateien in
`output/` bitgenau, und alle Prüfungen bleiben, wie sie sind. In einer anderen
Umgebung können einzelne Zahlen in der letzten Stelle abweichen (siehe
[LIMITATIONS.md](../LIMITATIONS.md), Abschnitt 12). Die Referenzdatei `baumkorrelation.csv` muss nach einem Lauf mit
anderen Einstellungen neu erzeugt werden.
