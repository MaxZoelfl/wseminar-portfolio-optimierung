# Anleitung: Wie führe ich den Code aus?

Diese Anleitung erklärt Schritt für Schritt, wie man das Projekt startet —
auch ohne Programmiererfahrung. Alle Befehle werden im **Terminal**
eingegeben (macOS: Programm „Terminal" öffnen, z. B. über die
Spotlight-Suche mit `⌘ + Leertaste` → „Terminal" tippen).

---

## Schritt 0: In den Projektordner wechseln

```bash
cd /Users/max/Desktop/Schule/wSeminar/Claude
```

Alle folgenden Befehle gehen davon aus, dass du in diesem Ordner stehst
(`cd` = "change directory", also Ordner wechseln).

**Wichtig:** Wir benutzen immer `venv/bin/python` statt nur `python`.
Das ist die projekteigene Python-Umgebung („virtual environment"), in der
alle benötigten Zusatzbibliotheken (pandas, scikit-learn, matplotlib, …)
bereits installiert sind. Das systemweite `python` kennt diese Pakete
nicht und würde mit Fehlermeldungen abbrechen.

---

## 1. Den Haupt-Backtest starten (das eigentliche Experiment)

```bash
venv/bin/python -m portfolio
```

Der Befehl bedeutet: „Führe das Paket `portfolio` als Programm aus."
Das startet `portfolio/__main__.py`, welches den „Dirigenten" `main()`
in `portfolio/run.py` aufruft. Was dann nacheinander passiert:

1. **Daten laden** — die Tageskurse der 15 Aktien und des S&P-500-Fonds
   (SPY) werden von Yahoo Finance heruntergeladen → **Internet nötig!**
2. **Simulation** — die Monat-für-Monat-Zeitmaschine 2015–2024 läuft:
   Jeden Monat berechnen alle vier Strategien (Markowitz, Random Forest,
   Equal Weight, Risk Parity) ihre Depotgewichte neu. Auf einem Rechner
   mit Bildschirm öffnet sich dabei das **Live-Dashboard**, in dem man
   den Fortschritt beobachten kann.
3. **Auswertung** — am Ende erscheinen die Kennzahlen-Tabelle und die
   Signifikanztests im Terminal; alle Abbildungen (PNG), Tabellen (CSV)
   und das Experimentprotokoll (JSON) landen im Ordner `output1.6/`.

**Dauer: grob 30–90 Minuten.** Der teure Teil ist der Random Forest,
der in jedem Monat neu trainiert und getunt wird.

> ⚠️ **Warnung vor dem Ausführen:** Ein neuer Lauf **überschreibt den
> Ordner `output1.6/`** — und genau daraus stammen die Abbildungen, die
> im fertigen Word-Dokument der Seminararbeit eingebunden sind. Da Yahoo
> Kursdaten gelegentlich nachträglich korrigiert, können die Ergebnisse
> danach minimal von den Zahlen im Text abweichen. Deshalb: Den Backtest
> nur laufen lassen, wenn man ihn wirklich braucht — oder vorher eine
> Sicherheitskopie anlegen, z. B. mit:
>
> ```bash
> cp -r output1.6 output1.6_backup
> ```

---

## 2. Nur die Tests laufen lassen (schnell und ungefährlich)

```bash
MPLBACKEND=Agg venv/bin/python -m pytest tests/ -q
```

Prüft in ~5 Sekunden, ob die gesamte Mathematik des Projekts noch stimmt.
Erwartetes Ergebnis: **`31 passed`** (31 Prüfungen bestanden).

- Der Vorsatz `MPLBACKEND=Agg` sorgt dafür, dass Diagramme nur unsichtbar
  im Speicher gezeichnet werden — es poppen keine Grafikfenster auf.
- `-q` heißt „quiet": kompakte Ausgabe.

Das ist der richtige Befehl, wenn man nur wissen will „ist alles heil?" —
er lädt keine Daten, verändert nichts und ist völlig gefahrlos.

---

## 3. Das Word-Dokument der Arbeit neu bauen

```bash
export PATH="/opt/homebrew/bin:$PATH"
venv/bin/python arbeit/build_docx.py
```

- Die erste Zeile macht das Konvertierungsprogramm **pandoc** auffindbar
  (es liegt im Homebrew-Ordner, den das Terminal nicht immer automatisch
  kennt). Sie muss **einmal pro Terminal-Sitzung** eingegeben werden.
- Die zweite Zeile erzeugt `arbeit/W-Seminararbeit.docx` frisch aus dem
  Manuskript `arbeit/arbeit.md` — inklusive Titelblatt, Formalia und
  Seitenzahlen.

**Nach jeder Änderung am Manuskript einmal ausführen.** Danach in Word
noch das Inhaltsverzeichnis aktualisieren (ins Verzeichnis klicken → `F9`).

Analog für die Nebendokumente:

```bash
venv/bin/python arbeit/build_quellen.py       # Quellen-Kompendium → .docx
venv/bin/python arbeit/build_klappentext.py   # Klappentext → .docx
```

---

## Bonus: Einstellungen ändern, ohne Code anzufassen

Alle Stellschrauben des Experiments (Aktienliste, Zeitraum, Kosten, …)
sind in `portfolio/config.py` gesammelt und dort ausführlich kommentiert.
Man kann sie überschreiben, indem man eine Datei `config.json` in den
Projektordner legt — zum Beispiel für einen **schnellen Probelauf**:

```json
{ "rf_retune_every": 6 }
```

Damit sucht der Random Forest nur alle 6 Monate neue Hyperparameter statt
jeden Monat → der Backtest wird um ein Mehrfaches schneller, die
Ergebnisse weichen dann aber von denen in der Arbeit ab. Datei wieder
löschen = Originalverhalten.

---

## Kurzfassung

| Ich möchte …                          | Befehl                                            | Dauer      |
|---------------------------------------|---------------------------------------------------|------------|
| nur prüfen, ob alles funktioniert      | `MPLBACKEND=Agg venv/bin/python -m pytest tests/ -q` | ~5 s    |
| das volle Experiment laufen lassen     | `venv/bin/python -m portfolio`                    | 30–90 min  |
| das Word-Dokument neu erzeugen         | `venv/bin/python arbeit/build_docx.py`            | Sekunden   |

(Vorher jeweils: `cd /Users/max/Desktop/Schule/wSeminar/Claude`)
