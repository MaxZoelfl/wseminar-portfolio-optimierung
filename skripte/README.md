# Die Nachrechnungs-Skripte

Jede Zahl, die die Seminararbeit aus eigenen Daten gewinnt, lässt sich hier
**ohne den großen Backtest** nachrechnen: aus den eingefrorenen Kursen und den
Ergebnisdateien, die der Backtest einmal erzeugt hat. Die Skripte rechnen die
Zahl aus und vergleichen sie mit dem Wortlaut im Text.

Wer nur wissen will, ob noch alles stimmt, braucht einen einzigen Befehl.

---

## Schnellstart

Im Stammordner des Repositories, bei aktivierter virtueller Umgebung
(Einrichtung: siehe [README im Stammordner](../README.md#loslegen)):

```bash
python skripte/alle_kapitel.py
```

Das dauert rund eineinhalb Minuten, liest nur und schreibt keine Datei. Am Ende steht
eine Übersicht wie diese (die Zahlen hier sind nur ein Beispiel):

```
                 ✅ stimmt  ≈ ungefähr   ❌ weicht ab   ⏭ offen
  Kapitel 2            44           0             1         3
  Kapitel 3            18           0             2        10
  …
```

---

## Wie man die Ausgabe liest

Jede geprüfte Zahl steht in einer eigenen Zeile:

```
  ✅ Volatilität Coca-Cola                        Text: 17,8 %      gerechnet: 17,8 %
  ❌ Rendite Coca-Cola allein                     Text: 9,0 %       gerechnet: 8,946 %
```

| Zeichen | Bedeutung | Was zu tun ist |
|:---:|---|---|
| ✅ | Die Rechnung ergibt, auf dieselbe Stellenzahl gerundet, **genau** die Zahl im Text. | nichts |
| ≈ | Der Text sagt „rund", „etwa" oder „gut", und die Rechnung liegt nah genug daran. | nichts |
| ❌ | Die Rechnung ergibt etwas anderes als der Text. Die Zeile zeigt beide Werte mit mehr Nachkommastellen. | Textstelle prüfen |
| ⏭ | Die Zahl wird hier nicht gerechnet. Die Zeile darunter sagt, warum. | nichts |

**Gerundet wird kaufmännisch**, also so wie von Hand: 8,946 wird auf eine Stelle
zu 8,9, 8,95 zu 9,0. Die Stellenzahl liest das Skript aus dem Text ab: Steht
dort „17,56", wird auf zwei Stellen gerundet, steht dort „17,6", auf eine.

**Ein häufiger Grund für ❌ ist doppeltes Runden:** Eine Zahl wird erst auf vier
Stellen gerundet (0,60250) und dann noch einmal auf drei (0,603). Richtig ist,
vom genauen Wert aus nur einmal zu runden (0,602499… → 0,602).

---

## Die Dateien

| Datei | Prüft | Dauer |
|---|---|---|
| **`alle_kapitel.py`** | startet alle Kapitel nacheinander und fasst zusammen | ~1,5 min |
| `kapitel_2_theorie.py` | § 2.2–2.6: Zwei-Anlagen-Fall, Diversifikationsgrenze, Effizienzrand, t-Werte, Konfidenzintervalle, Ledoit-Wolf-Schrumpfung | ~15 s |
| `kapitel_3_random_forest.py` | § 3.1–3.5: Prognosefehler, Lernstoff je Termin, Einzelbäume der Tiefe 1 und 14, gewählte Baumtiefen | ~10 s |
| ↳ `… --baumkorrelation` | zusätzlich § 3.3: Korrelation der Bäume, Breimans Schranke | +55 min |
| `kapitel_4_methodik.py` | § 4.1–4.5: Einstellungen, Termine, Schutzvorkehrungen, Umschlagschranke; beziffert zwei Vereinfachungen des Backtests und testet die Fassung mit Kursdrift (LIMITATIONS.md § 13) | ~15 s |
| `kapitel_5_signifikanz.py` | § 5.1–5.4: Standardfehler, Autokorrelation, Testeinstellungen; Gegenprobe des Tests so, wie Ledoit/Wolf ihn beschreiben | ~20 s |
| `kapitel_6_ergebnisse.py` | § 6.1–6.6: Kennzahlen, Portfoliostruktur, Signifikanztabelle, Kostensensitivität, NVIDIA-Effekt, Zinsschwellen | ~30 s |
| `kapitel_7_fazit.py` | § 7.1: wiederholte p-Werte, Vergleich mit DeMiguel u. a. | ~1 s |
| `anhang.py` | Anhang A–C: Bildunterschriften, Tabellen A2, A3, B2–B4, Parameter C1, Angaben in C2 | ~5 s |
| `abbildung_kostensensitivitaet.py` | **erzeugt** Abbildung 5 (`output/13_kosten_sensitivitaet.png`) und `output/kosten_sensitivitaet.csv` neu | ~5 s |
| `_werkzeuge.py` | gemeinsame Bausteine (Daten laden, Runden, Soll-Ist-Vergleich). Wird nicht selbst gestartet. | — |

Jedes Kapitel-Skript lässt sich auch einzeln starten, etwa
`python skripte/kapitel_6_ergebnisse.py`. Alle funktionieren aus jedem Ordner heraus.
Nur `abbildung_kostensensitivitaet.py` schreibt Dateien, alle anderen lesen nur.

---

## Welche Zahlen nicht gerechnet werden — und warum

- **Kapitel 1** hat kein Skript. Es enthält ausschließlich Zahlen aus der
  Literatur (etwa „3000 Monate bei 25 Titeln" nach DeMiguel u. a.). Belegt sind
  sie durch die Fußnoten, nicht durch den Backtest.
- **Zahlen aus Quellen** in den übrigen Kapiteln (etwa Qians 60/40-Beispiel oder
  die Ergebnisse von Gu, Kelly und Xiu) erscheinen als ⏭ mit dem Vermerk
  „Zahl aus der Quelle".
- **Die lange Rechnung zu Kapitel 3** (Korrelation der Bäume, § 3.3) läuft nur mit
  dem Schalter `--baumkorrelation`, rund 55 Minuten. Sie tunt die Wälder an allen
  119 Terminen neu, führt aber ebenfalls nicht den großen Backtest aus.

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

Dazu kommen zwei kleine Referenzdateien in `skripte/daten/`:

- **`rf_hyperparameter.csv`**: Die Einstellungen, die die Hyperparametersuche an
  jedem der 119 Termine gewählt hat (Zahl der Bäume, Tiefe, Blattgröße). Sie
  stammen aus dem Laufprotokoll des maßgeblichen Laufs. Das Protokoll selbst
  liegt nicht im Repository, weil es rechnerspezifische Pfade enthält.
- **`vergleichslauf_ohne_purging.csv`**: Tagesrenditen eines Vergleichslaufs mit
  gewöhnlicher zeitlicher Kreuzvalidierung statt Purged & Embargoed CV. Er ist
  die Grundlage der Zahl 0,980 in § 4.3 (wie viel Informationsleckage der
  Schutz verhindert). Der Lauf entstand vor dem Einfrieren der Kursdatei und vor
  dem Einkern-Modus. Die übrigen drei Strategien stimmen mit dem maßgeblichen Lauf
  auf drei Nachkommastellen überein.

**Zur Genauigkeit:** `output/daily_returns.csv` ist auf sechs Nachkommastellen
gerundet. Für Renditen und Volatilitäten genügt das. Für die vierte Stelle eines
Sharpe-Quotienten nicht immer, deshalb verwenden die Skripte dort die exakten
Werte aus `output/experiment_log.json`.

---

## Wenn sich der Text ändert

**Die Skripte lesen die Word-Datei nicht.** Die Zahl, wie sie im Text steht, ist
im Skript als Zeichenkette eingetragen, zum Beispiel in `kapitel_2_theorie.py`:

```python
p.zahl("Volatilität Coca-Cola", "17,8", sA * 100, "%")
#       ↑ Beschriftung          ↑ so steht es im Text   ↑ die Rechnung
```

Ändert sich eine Zahl im Text, wird nur die Zeichenkette in der Mitte angepasst.
Die Rechnung bleibt, wie sie ist.

Damit kein Eintrag veraltet, ohne dass es auffällt, gibt es den Wortlaut-Abgleich:

```bash
python skripte/alle_kapitel.py --text pfad/zur/Arbeit.docx
```

Am Ende der Ausgabe steht dann jede eingetragene Zahl, die in dieser Fassung der
Arbeit **nicht** vorkommt — meist ein Zeichen, dass die Stelle im Text geändert
wurde. Für `.docx` braucht es [pandoc](https://pandoc.org); eine Markdown-Fassung
des Textes geht ohne. Der Abgleich sucht jede Zahl im ganzen Text, nicht an ihrer
Stelle: Kurze Zahlen wie „4" findet er fast immer, geänderte Dezimalzahlen und
Daten fängt er zuverlässig ab. Mit `--text` endet das Skript mit Fehlercode 1,
wenn etwas fehlt.

## Wenn der Backtest neu läuft

Dann ändern sich die Ergebnisdateien in `output/`, und die Skripte rechnen
automatisch mit den neuen Werten. Solange `data/prices.pkl` unverändert bleibt,
reproduziert der Backtest die bisherigen Dateien bitgenau (siehe
[LIMITATIONS.md](../LIMITATIONS.md), Abschnitt 12), und alle Prüfungen bleiben
wie sie sind.
