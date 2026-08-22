# Anleitung: Wie führe ich den Code aus?

Diese Anleitung erklärt Schritt für Schritt, wie man das Projekt startet —
auch ohne Programmiererfahrung. Alle Befehle werden im **Terminal**
eingegeben (macOS: Programm „Terminal" öffnen, z. B. über die
Spotlight-Suche mit `⌘ + Leertaste` → „Terminal" tippen).

Was der Code *inhaltlich* tut, steht in [Handbook.md](Handbook.md);
was das Projekt *ist*, in [README.md](README.md).

---

## Schritt 0: In den Projektordner wechseln

```bash
cd /Users/max/Desktop/Schule/wSeminar/Code
```

Alle folgenden Befehle gehen davon aus, dass du in diesem Ordner stehst
(`cd` = "change directory", also Ordner wechseln).

**Wichtig:** Wir benutzen immer `venv/bin/python` statt nur `python`.
Das ist die projekteigene Python-Umgebung („virtual environment"), in der
alle benötigten Zusatzbibliotheken (pandas, scikit-learn, matplotlib, …)
bereits installiert sind. Das systemweite `python` kennt diese Pakete
nicht und würde mit Fehlermeldungen abbrechen.

Fehlt der Ordner `venv/`, baut man ihn so neu auf:

```bash
python3 -m venv venv && venv/bin/pip install -r requirements-dev.txt
```

---

## 1. Prüfen, ob alles heil ist (schnell und ungefährlich)

```bash
MPLBACKEND=Agg venv/bin/python -m pytest tests/ -q
```

Prüft in rund 4 Sekunden, ob die gesamte Mathematik des Projekts noch stimmt.
Erwartetes Ergebnis: **`37 passed`** (37 Prüfungen bestanden).

- Der Vorsatz `MPLBACKEND=Agg` sorgt dafür, dass Diagramme nur unsichtbar
  im Speicher gezeichnet werden — es poppen keine Grafikfenster auf.
- `-q` heißt „quiet": kompakte Ausgabe.

Das ist der richtige Befehl, wenn man nur wissen will „ist alles heil?" —
er lädt keine Daten, verändert nichts und ist völlig gefahrlos.

---

## 2. Den Haupt-Backtest starten (das eigentliche Experiment)

```bash
venv/bin/python -m portfolio
```

Der Befehl bedeutet: „Führe das Paket `portfolio` als Programm aus."
Das startet `portfolio/__main__.py`, welches den „Dirigenten" `main()`
in `portfolio/run.py` aufruft. Was dann nacheinander passiert:

1. **Daten laden** — die Tageskurse der 15 Aktien und des S&P-500-Fonds
   (SPY). Sie kommen aus der eingefrorenen Datei `data/prices.pkl`;
   **Internet wird also nicht gebraucht**, solange diese Datei existiert.
2. **Simulation** — die Monat-für-Monat-Zeitmaschine 2015–2024 läuft über
   119 Rebalancing-Termine: Jeden Monat berechnen alle vier Strategien
   (Markowitz, Random Forest, Equal Weight, Risk Parity) ihre Depotgewichte
   neu. Auf einem Rechner mit Bildschirm öffnet sich dabei das
   **Live-Dashboard**, in dem man den Fortschritt beobachten kann.
3. **Auswertung** — am Ende erscheinen die Kennzahlen-Tabelle und die
   Signifikanztests im Terminal; alle Abbildungen (PNG), Tabellen (CSV)
   und das Experimentprotokoll (JSON) landen im Ordner `output/`.

**Dauer: rund 77 Minuten.** 98 % davon entfallen auf den Random Forest, der
in jedem Monat neu getunt wird — und zwar bewusst einkernig, damit der Lauf
bitgenau wiederholbar bleibt.

> ⚠️ **Warnung vor dem Ausführen:** Ein neuer Lauf **überschreibt den
> Ordner `output/`** — und genau daraus stammen die Abbildungen und Zahlen
> der Seminararbeit. Deshalb: Den Backtest nur laufen lassen, wenn man ihn
> wirklich braucht — oder vorher eine Sicherheitskopie anlegen:
>
> ```bash
> cp -r output output_backup
> ```
>
> Zum bloßen Ausprobieren gibt es den gefahrlosen Probelauf (Abschnitt 5),
> der nach `output_probelauf/` schreibt.

---

## 3. Einzelne Bausteine nachrechnen (Sekunden statt Stunden)

Diese vier Skripte brauchen den Backtest **nicht**. Sie lesen die
gespeicherten Ergebnisse bzw. die eingefrorenen Kurse und rechnen daraus
nach. Ideal, um eine Zahl im Text zu überprüfen.

```bash
venv/bin/python signifikanz.py             # Signifikanzblock nachrechnen
venv/bin/python kosten_sensitivitaet.py    # Abbildung 13 + CSV neu erzeugen
venv/bin/python nachrechnen_kapitel2.py    # Kontrollrechnung zu § 2.1
venv/bin/python nachrechnen_kapitel3.py    # Kontrollrechnung zu § 3.1/§ 3.2
```

Zwei Hinweise dazu:

- `signifikanz.py` liest die auf sechs Nachkommastellen gerundete
  `output/daily_returns.csv` und weicht deshalb in der vierten Stelle des
  p-Werts ab. **Maßgeblich für die Arbeit bleibt
  `output/experiment_log.json`.** Das Skript schreibt bewusst keine Datei.
- `nachrechnen_kapitel3.py` kennt zwei lange Zusatzschritte, die man
  einzeln anfordern muss:

  ```bash
  venv/bin/python nachrechnen_kapitel3.py --sweep            # + Tiefensweep,   ~7 min
  venv/bin/python nachrechnen_kapitel3.py --baumkorrelation  # + Baumkorrelation, ~55 min
  ```

### Nur die Theorie-Abbildungen neu zeichnen

Die fünf Bilder `output/14`–`18` gehören zu Kapitel 2 und zeigen keine
Strategieergebnisse, sondern Eigenschaften der Kursdaten. Sie entstehen bei
jedem vollen Lauf mit, lassen sich aber in zwei Sekunden einzeln erneuern:

```bash
venv/bin/python -c "import pandas as pd; from portfolio.theory_plots import create_theory_plots; from portfolio.config import TICKERS, OUTPUT_DIR; ar=pd.read_pickle('data/prices.pkl')[TICKERS].pct_change().dropna(); rd=pd.read_csv('output/daily_returns.csv',index_col=0,parse_dates=True); create_theory_plots(ar,rd,OUTPUT_DIR)"
```

---

## 4. Einstellungen ändern, ohne Code anzufassen

Alle Stellschrauben des Experiments (Aktienliste, Zeitraum, Kosten, …)
sind in `portfolio/config.py` gesammelt und dort ausführlich kommentiert.
Überschreiben geht auf zwei Wegen:

```bash
# Weg A: Datei config.json im Projektordner ablegen — wird automatisch geladen
cp config.example.json config.json     # dann darin die gewünschten Werte ändern

# Weg B: eine beliebige Datei per Umgebungsvariable benennen
PORTFOLIO_CONFIG=config.schnell.json venv/bin/python -m portfolio
```

Nur die Schlüssel, die in der Datei stehen, werden überschrieben — alles
andere bleibt auf dem Standardwert. `config.json` wieder löschen =
Originalverhalten.

`config.example.json` enthält **alle Standardwerte** und dokumentiert damit
genau den Lauf, aus dem die Zahlen der Arbeit stammen. Es ist also keine
„Beispielabweichung", sondern die Nachschlagefassung des kanonischen Laufs.

---

## 5. Der gefahrlose Probelauf

```bash
PORTFOLIO_CONFIG=config.schnell.json venv/bin/python -m portfolio
```

`config.schnell.json` ändert drei Dinge: Der Random Forest sucht nur alle
6 Monate neue Hyperparameter statt jeden Monat, das Dashboard wird nur alle
5 Schritte gezeichnet, und geschrieben wird nach **`output_probelauf/`**
statt nach `output/`.

Ergebnis: ein Mehrfaches schneller, **`output/` bleibt unangetastet** — die
Zahlen weichen dafür von denen der Arbeit ab. Genau richtig, um zu sehen,
*dass* alles läuft, ohne die maßgeblichen Ergebnisse zu riskieren.

---

## 6. In VS Code: alles per Knopfdruck

Im Debug-Reiter (Play-Symbol mit Käfer, links) stehen fertige
Startkonfigurationen bereit — auswählen und `F5` drücken. VS Code nimmt
automatisch die venv:

| Eintrag | Was er tut |
|---|---|
| **Tests (alle 37)** | die Prüfung aus Abschnitt 1 |
| **Backtest komplett (⚠ ÜBERSCHREIBT output/)** | der volle Lauf, rund 77 min |
| **Backtest schnell (→ output_probelauf/)** | der Probelauf aus Abschnitt 5 |
| **Signifikanz nachrechnen** | `signifikanz.py` |
| **Kostensensitivität (Abb. 13)** | `kosten_sensitivitaet.py` |
| **Kontrollrechnung Kapitel 3** | `nachrechnen_kapitel3.py` |
| **Aktuelle Datei ausführen** | die gerade geöffnete Datei, z. B. `nachrechnen_kapitel2.py` |

---

## 7. Zum Word-Dokument — bitte lesen, bevor du etwas baust

**Die Arbeit wird nicht mehr aus diesem Ordner gebaut.** Geschrieben wird
von Hand in `../Abgabe/W-Seminararbeit_FINAL.docx`.

Der Ordner `arbeit/` enthält die **überholte APA-Fassung** aus dem Juli:
das Manuskript `arbeit/arbeit.md`, das Bauskript `arbeit/build_docx.py` und
die daraus erzeugten `.docx`-Dateien. Sie sind als Nachschlagewerk
aufgehoben, nicht als Arbeitsstand.

Wer `arbeit/build_docx.py` trotzdem startet, sollte zwei Dinge wissen:

1. Es baut die **alte** Fassung mit APA-Kurzbelegen — nicht die Schulform
   mit Fußnoten, auf die 26.07. umgestellt wurde.
2. Die drei eingebundenen Abbildungen verweisen in `arbeit/arbeit.md` noch
   auf den Ordner `output1.6/`, den es nicht mehr gibt. Sie fehlen im
   Ergebnis, ohne dass der Bau abbricht.

Falls es doch einmal gebraucht wird — `pandoc` muss auffindbar sein:

```bash
export PATH="/opt/homebrew/bin:$PATH"
venv/bin/python arbeit/build_docx.py
```

Die Nebendokumente lassen sich analog bauen:

```bash
venv/bin/python arbeit/build_quellen.py       # Quellen-Kompendium → .docx
venv/bin/python arbeit/build_klappentext.py   # Klappentext → .docx
```

---

## Kurzfassung

| Ich möchte …                       | Befehl                                               | Dauer     |
|------------------------------------|------------------------------------------------------|-----------|
| prüfen, ob alles funktioniert      | `MPLBACKEND=Agg venv/bin/python -m pytest tests/ -q` | ~4 s      |
| eine Zahl der Arbeit nachrechnen   | `venv/bin/python signifikanz.py`                     | Sekunden  |
| gefahrlos ausprobieren             | `PORTFOLIO_CONFIG=config.schnell.json venv/bin/python -m portfolio` | Minuten |
| das volle Experiment laufen lassen | `venv/bin/python -m portfolio`                       | ~77 min   |

(Vorher jeweils: `cd /Users/max/Desktop/Schule/wSeminar/Code`)
