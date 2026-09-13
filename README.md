# W-Seminar: Portfolio-Optimierung — Markowitz vs. Random Forest

Vergleich klassischer und ML-gestützter Portfolio-Optimierung über einen
Walk-Forward-Backtest (2015–2024) auf 15 US-Large-Caps.

**Strategien:** Markowitz MVO (Ledoit-Wolf) · Random Forest · Equal Weight (1/N) · Risk Parity (ERC)

**Paketversion:** 4.2.0 · **Tests:** 37 grün · **Maßgeblicher Lauf:** `output/` (15.08.2026, 77 min; am 25.08.2026 mit identischen Kennzahlen reproduziert)

Dieses Repository ist der Begleitcode zu einer W-Seminararbeit (Mathematik,
Abiturjahrgang 2026/27). Es enthält den vollständigen Quellcode des Backtests,
die eingefrorenen Kursdaten und alle Ergebnisdateien, auf die sich die Arbeit
stützt — jede dort genannte Zahl lässt sich hier nachrechnen.

> **Ein Code, ein Ergebnis.** Es gibt genau **eine** Konfiguration — die
> Standardwerte in `portfolio/config.py` — und genau **einen** Ergebnisordner:
> `output/`. Frühere Entwicklungsläufe sind nicht Teil des Repositories; es
> waren Vorstufen mit inzwischen behobenen Fehlern, keine methodischen
> Alternativen.

Diese Datei erklärt, **was** das Projekt ist und **wie man es bedient**. Die
*Funktionsweise* des Codes steht in [Handbook.md](Handbook.md), die
wissenschaftlichen Grenzen in [LIMITATIONS.md](LIMITATIONS.md), nicht umgesetzte
Ideen samt Status in [Ideen-Backlog.md](Ideen-Backlog.md).

## Projektstruktur

```
Code/
├── portfolio/      das Paket — hier steckt die gesamte Implementierung
├── tests/          37 Unit-Tests, ohne Netzwerk, ~4 s
├── data/           prices.pkl — eingefrorene Kurse, Abruf 15.08.2026
├── output/         DER Ergebnisordner — 20 Abbildungen, 8 CSV, 1 JSON-Protokoll
├── arbeit/         frühere Manuskriptfassung, nicht mehr gepflegt
├── archive/        eingefrorene v4.1-Baseline als Einzeldatei
├── venv/           Python-Umgebung (nicht versioniert, ~547 MB)
├── .vscode/        Interpreter-, Test- und Startkonfiguration
└── (Stammordner)   4 Kontrollskripte · 2 Konfigurationen · 4 Doku-Dateien
```

### Das Paket `portfolio/`

In Ausführungsreihenfolge. Die Schicht sagt, worauf ein Modul aufbauen darf —
jedes importiert nur nach unten, deshalb gibt es keine Zyklen
(Schaubild in [Handbook.md § 2](Handbook.md#2-architektur-schichten-und-abhängigkeiten)).

| Modul | Aufgabe | Schicht |
|---|---|:---:|
| `config.py` | Typisierte Config (`dataclass`), Logging, Plot-Design, Backend-Wahl | 0 |
| `data.py` | Marktdaten laden (yfinance), Kursspeicher, einfache Renditen | 1 |
| `indicators.py` | Technische Indikatoren, Monats-Aggregation, CS-Ränge, Feature-Listen | 2 |
| `metrics.py` | CAGR, Sharpe, Sortino, Drawdown, Calmar, VaR, `@timer` | 1 |
| `cross_validation.py` | Purged & Embargoed CV (López de Prado 2018) | 0 |
| `optimizers.py` | `MarkowitzLedoitWolf`, `RiskParityPortfolio`, `RFPortfolioOptimizer` | 2 |
| `significance.py` | Ledoit-Wolf-2008-Test, Holm-Bonferroni, Deflated Sharpe Ratio | 0 |
| `backtest.py` | Rollierender Walk-Forward-Backtest — **die Hauptschleife** | 3 |
| `dashboard.py` | Live-Training-Dashboard (optional, nur mit Bildschirm) | 2 |
| `plots.py` | Ergebnis-Abbildungen 01–12, CSV- und JSON-Export | 3 |
| `theory_plots.py` | Theorie-Abbildungen 14–18 zu Kapitel 2 der Arbeit | 3 |
| `run.py` | Orchestrierung: `main()` mit den Abschnitten A–I | 3 |
| `__init__.py` · `__main__.py` | Paketkennung und Einstiegspunkt für `python -m portfolio` | — |

### Kontrollskripte im Stammordner

Keines braucht den Backtest — sie lesen `output/` bzw. `data/prices.pkl`.

| Skript | Rechnet nach | Dauer |
|---|---|---|
| `signifikanz.py` | Signifikanzblock aus `output/daily_returns.csv` | Sekunden |
| `kosten_sensitivitaet.py` | Sharpe über Kostensätze 0–100 bp → Abb. 13 + CSV | Sekunden |
| `nachrechnen_kapitel2.py` | § 2.1 der Arbeit, Schritt für Schritt | ~2 s |
| `nachrechnen_kapitel3.py` | § 3.1–3.3 der Arbeit | ~10 s · `--sweep` 7 min · `--baumkorrelation` 55 min |

### Daten, Ergebnisse, Altbestand

| Pfad | Inhalt |
|---|---|
| `data/prices.pkl` | Schlusskurse vom 15.08.2026, splitt- und dividendenbereinigt. **Ohne sie ist der Backtest nicht reproduzierbar** — Yahoo liefert bei jedem Abruf minimal andere Werte. |
| `output/` | Der maßgebliche Lauf: 19 PNG + 1 GIF, 8 CSV und `experiment_log.json` — Letzteres ist die **verbindliche Quelle** für jede Zahl der Arbeit. |
| `run.log` | Protokoll dieses Laufs (15.08.2026, 77 min). `nachrechnen_kapitel3.py` liest daraus die gewählten Baumtiefen. |
| `archive/projekt1.6.py` | Eingefrorene v4.1-Baseline als Einzeldatei. Enthält **nicht** die späteren Erweiterungen — keine Purged CV, keine Fairness-Optionen, die alten Sharpe- und Sortino-Definitionen. Referenz, nicht Alternative. |
| `arbeit/` | Manuskript, Bauskripte und `.docx` der überholten APA-Fassung. Näheres am Ende dieser Datei. |

### Konfiguration und Dokumentation

| Datei | Inhalt |
|---|---|
| `config.example.json` | Alle Standardwerte — dokumentiert damit den kanonischen Lauf |
| `config.schnell.json` | Gefahrloser Probelauf → `output_probelauf/` |
| `requirements.txt` · `requirements-dev.txt` | Abhängigkeiten für Lauf bzw. Tests |
| `conftest.py` | pytest-Setup: headless Matplotlib, Projektwurzel auf `sys.path` |
| `README.md` | diese Datei — was das Projekt ist und wie man es bedient |
| [`Handbook.md`](Handbook.md) | wie der Code funktioniert, mit 13 Diagrammen |
| [`LIMITATIONS.md`](LIMITATIONS.md) | wissenschaftliche Limitationen und Literatur |
| [`Ideen-Backlog.md`](Ideen-Backlog.md) | Ideensammlung vom Mai, je Punkt mit Umsetzungsstatus |

> Frühere Entwicklungsstufen (projekt1.0–1.5 sowie der abgebrochene 2.x-Zweig)
> wurden entfernt; sie bleiben über den ersten Commit (`Initial snapshot`) in der
> Git-Historie erhalten und sind bei Bedarf wiederherstellbar.


## Loslegen

Voraussetzung ist Python 3 (entwickelt und getestet mit 3.14). Alle Befehle
werden im Terminal aus dem Stammordner des Repositories heraus ausgeführt:

```bash
git clone https://github.com/MaxZoelfl/wseminar-portfolio-optimierung.git
cd wseminar-portfolio-optimierung
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

Die virtuelle Umgebung hält pandas, scikit-learn, matplotlib und den Rest vom
System getrennt. Alle weiteren Befehle setzen voraus, dass sie aktiviert ist —
`python` meint dann automatisch das Python der Umgebung.

### 1. Prüfen, ob alles funktioniert (ungefährlich, ~4 s)

```bash
MPLBACKEND=Agg python -m pytest tests/ -q
```

Erwartet: **`37 passed`**. `MPLBACKEND=Agg` zeichnet Diagramme nur unsichtbar im
Speicher, es poppen also keine Fenster auf; `-q` heißt „quiet". Der Befehl lädt
keine Daten und verändert nichts — der richtige erste Griff, wenn man nur wissen
will, ob alles funktioniert.

### 2. Den vollständigen Backtest starten

```bash
python -m portfolio
```

„Führe das Paket `portfolio` als Programm aus." Das startet `__main__.py`,
welches den Dirigenten `main()` in `portfolio/run.py` aufruft. Was passiert:

1. **Daten laden** — Tageskurse der 15 Aktien und des S&P-500-Fonds aus der
   eingefrorenen Datei `data/prices.pkl`.
2. **Simulation** — die Monat-für-Monat-Zeitmaschine 2015–2024 über
   119 Rebalancing-Termine. Auf einem Rechner mit Bildschirm öffnet sich dabei
   das **Live-Dashboard**.
3. **Auswertung** — Kennzahlen und Signifikanztests erscheinen im Terminal,
   Abbildungen, Tabellen und das Experimentprotokoll landen in `output/`.

**Laufzeit rund 77 Minuten** (gemessen am maßgeblichen Lauf).
**98 % davon** entfallen auf die Hyperparametersuche des Random Forest, die in
jedem der 119 Rebalancing-Monate neu läuft — und zwar einkernig, weil
`deterministic=True` bitgenaue Reproduzierbarkeit erzwingt (siehe
[Methodikoptionen](#methodikoptionen-alle-standardmäßig-an)).

> ⚠️ Ein neuer Lauf **überschreibt `output/`** — genau daraus stammen die
> Abbildungen und Zahlen der Seminararbeit. Vorher sichern:
>
> ```bash
> cp -r output output_backup
> ```
>
> Zum bloßen Ausprobieren gibt es den gefahrlosen Probelauf (siehe unten), der
> nach `output_probelauf/` schreibt und `output/` unangetastet lässt.

**Internet wird nicht gebraucht**, solange `data/prices.pkl` existiert: Die
Kurse werden von dort gelesen. Das Paket `yfinance` muss trotzdem installiert
sein, sonst bricht `main()` früh ab. Frischer Download: Datei löschen oder
`price_cache` auf `""` setzen — dann ist der Lauf allerdings nicht mehr mit den
Zahlen der Arbeit vergleichbar (Yahoo liefert bei jedem Abruf minimal andere
Kurse, siehe LIMITATIONS.md § 12).

### 3. Einzelne Bausteine nachrechnen (Sekunden statt Stunden)

Damit lässt sich jede in der Arbeit genannte Zahl einzeln nachrechnen — was die
vier Skripte jeweils tun, steht [oben in der Übersicht](#kontrollskripte-im-stammordner).

```bash
python signifikanz.py                # Signifikanzblock nachrechnen (Sekunden)
python kosten_sensitivitaet.py       # Abb. 13 + kosten_sensitivitaet.csv
python nachrechnen_kapitel2.py       # Kontrollrechnung § 2.1
python nachrechnen_kapitel3.py       # Kontrollrechnung § 3.1–3.2
python nachrechnen_kapitel3.py --sweep            # + Tiefensweep (~7 min)
python nachrechnen_kapitel3.py --baumkorrelation  # + Baumkorrelation (~55 min)
```

`signifikanz.py` liest die auf sechs Nachkommastellen gerundete
`output/daily_returns.csv` und weicht deshalb in der vierten Stelle des p-Werts
ab (gemessen 2·10⁻⁴, exakt eine von 4999 Bootstrap-Ziehungen). **Maßgeblich
bleibt `output/experiment_log.json`.** Das Skript schreibt bewusst keine Datei,
damit im Ergebnisordner nur eine Signifikanzquelle liegt.

Die fünf Theorie-Abbildungen (`output/14`–`18`) zeigen keine Strategieergebnisse,
sondern Eigenschaften der Kursdaten (Modul `portfolio/theory_plots.py`). Sie
entstehen bei jedem vollen Lauf mit, brauchen aber nur `data/prices.pkl` und
`output/daily_returns.csv` — und lassen sich deshalb in zwei Sekunden einzeln
neu erzeugen:

```bash
python -c "import pandas as pd; from portfolio.theory_plots import create_theory_plots; from portfolio.config import TICKERS, OUTPUT_DIR; ar=pd.read_pickle('data/prices.pkl')[TICKERS].pct_change().dropna(); rd=pd.read_csv('output/daily_returns.csv',index_col=0,parse_dates=True); create_theory_plots(ar,rd,OUTPUT_DIR)"
```

## Konfiguration

Alle einstellbaren Parameter liegen typisiert in der `dataclass` `Config`
(`portfolio/config.py`). Überschreiben ohne Code-Änderung:

```bash
# Weg A: Datei config.json im Projektordner ablegen — wird automatisch geladen
cp config.example.json config.json    # darin die gewünschten Werte ändern
python -m portfolio

# Weg B: eine beliebige Datei per Umgebungsvariable benennen
PORTFOLIO_CONFIG=config.schnell.json python -m portfolio
```

`config.json` wieder löschen = Originalverhalten. `config.example.json` enthält
**alle Standardwerte** und dokumentiert damit genau den Lauf, aus dem die Zahlen
der Arbeit stammen — es ist keine „Beispielabweichung", sondern die
Nachschlagefassung des kanonischen Laufs.

Nur die angegebenen Schlüssel werden überschrieben; der Rest bleibt auf Default.
Unbekannte Schlüssel werden gemeldet und ignoriert, eine kaputte Datei bricht den
Lauf nicht ab.

| Schlüssel | Default | Bedeutung |
|---|---|---|
| `tickers` | 15 US-Large-Caps | Anlageuniversum |
| `spy_ticker` | `"SPY"` | Marktproxy für Alpha/Beta |
| `start_date` / `end_date` | `2013-01-01` / `2024-12-31` | Datenzeitraum (inkl. 2 Jahre Indikator-Warmup) |
| `backtest_start` | `2015-01-01` | ab hier wird „investiert" |
| `output_dir` | `"./output"` | Zielordner aller Ergebnisdateien |
| `price_cache` | `"./data/prices.pkl"` | Kursspeicher; `""` erzwingt frischen Download |
| `risk_free_rate` | `0.04` | annualisierter risikofreier Zins |
| `train_years` | `3` | Länge des rollierenden Trainingsfensters |
| `n_frontier` | `120` | Punkte auf der gezeichneten Effizienzlinie |
| `rf_n_iter` | `30` | Kandidaten der RandomizedSearchCV |
| `rf_cv_splits` | `5` | Zahl der CV-Folds im RF-Tuning |
| `max_weight` | `0.20` | Positionsobergrenze je Titel |
| `transaction_cost` | `0.0010` | 10 bp auf den Handelsumsatz |
| `rf_turnover_limit` | `0.30` | max. einseitiger Turnover/Monat (RF) |
| `rf_retune_every` | `1` | Hyperparametersuche nur alle *k* Monate |
| `dashboard_update_every` | `1` | Live-Dashboard nur alle *k* Schritte rendern |
| `deterministic` | `true` | einkernig rechnen → bitgenau wiederholbar |
| `use_purged_cv` | `true` | Purged & Embargoed CV statt TimeSeriesSplit |
| `cv_embargo` | `0.02` | Embargo-Anteil (nur bei `use_purged_cv`) |
| `mvo_turnover_limit` | `0.30` | dasselbe Turnover-Limit auch für Markowitz |
| `turnover_ref_drifted` | `true` | Turnover-Schranke gegen gedriftete Vorgängergewichte |
| `min_variance_fallback` | `true` | bei entarteter Sharpe-Maximierung auf Min-Varianz ausweichen |

Nicht über JSON einstellbar (strukturell festgelegt): `RANK_COLS`,
`FEATURE_COLS`, `COLORS`, `STRATEGIES`.

### Methodikoptionen (alle standardmäßig an)

Die folgenden fünf Schalter waren zunächst als abschaltbare Korrekturen
eingeführt und sind inzwischen der kanonische Standard. Wer sie ausschaltet,
erhält die früheren, fehlerbehafteten Entwicklungsläufe; der Random Forest
erscheint dann um rund 0,044 Sharpe besser, als er ist. Hintergrund und
Literatur: [LIMITATIONS.md](LIMITATIONS.md) § 5 und § 10–12.

- **`mvo_turnover_limit = 0.30`** — früher hatte nur der Random Forest ein
  Turnover-Limit. Auf denselben Wert wie `rf_turnover_limit` gesetzt,
  unterscheiden sich die beiden Strategien wirklich **nur** im Renditeschätzer.
  `null` = kein Limit (früheres Verhalten).
- **`turnover_ref_drifted = true`** — die Turnover-Schranke *im Optimierer*
  misst gegen die kursgedrifteten Vorgängergewichte, also dieselbe Referenz, die
  auch der ausgewiesene Turnover benutzt. Mit `false` überschritt der RF sein
  30-%-Limit in 100 von 118 Monaten.
- **`min_variance_fallback = true`** — erwartet kein zulässiges Portfolio mehr
  als den risikofreien Zins, ist die Sharpe-Maximierung entartet: Der Optimierer
  würde die Volatilität *maximieren*. Dann weicht er auf das
  Minimum-Varianz-Portfolio aus. Der Fall wird **immer** ins Log geschrieben
  (`Max-Sharpe entartet: …`), auch wenn die Option aus ist — im Zeitraum
  2015–2024 trat er in keinem der 119 Monate ein.
- **`use_purged_cv = true`** — Purging und Embargo im RF-Tuning statt
  `TimeSeriesSplit`. Nötig, weil die Merkmale aus überlappenden Fenstern stammen
  (Momentum über 252 Tage) und das Monatslabel in den Folgemonat reicht.
- **`deterministic = true`** — `n_jobs=1` in Wald und Hyperparametersuche.
  Gleitkommaaddition ist nicht assoziativ; bei paralleler Reduktion kürte die
  Suche in 4 von 119 Monaten einen anderen Sieger und verschob den RF-Sharpe um
  rund 0,010. Kostet Laufzeit, kauft Reproduzierbarkeit.

### Performance-Hebel

Standardwerte (`1`) lassen das Verhalten exakt wie im maßgeblichen Lauf:

- **`rf_retune_every`** — RF-Hyperparametersuche nur alle *k* Monate, dazwischen
  nur Refit auf dem aktuellen Fenster. `3` ≈ 2–3× schneller. **Achtung:** Werte
  größer als 1 verändern die Ergebnisse (andere Hyperparameter über die Zeit).
- **`dashboard_update_every`** — Live-Dashboard nur alle *k* Schritte rendern.
  Rein kosmetisch, **kein** Einfluss auf Kennzahlen.

### Der gefahrlose Probelauf

```bash
PORTFOLIO_CONFIG=config.schnell.json python -m portfolio
```

`config.schnell.json` ändert drei Dinge: Hyperparametersuche nur alle 6 Monate,
Dashboard nur alle 5 Schritte — und geschrieben wird nach **`output_probelauf/`**
statt nach `output/`. Ergebnis: ein Mehrfaches schneller, **`output/` bleibt
unangetastet**, die Zahlen weichen dafür von denen der Arbeit ab. Genau richtig,
um zu sehen, *dass* alles läuft, ohne die maßgeblichen Ergebnisse zu riskieren.

## In VS Code: alles per Knopfdruck

Im Debug-Reiter (Play-Symbol mit Käfer, links) stehen fertige
Startkonfigurationen bereit — auswählen und `F5` drücken. VS Code nimmt
automatisch die venv:

| Eintrag | Was er tut |
|---|---|
| **Tests (alle 37)** | die Prüfung von oben |
| **Backtest komplett (⚠ ÜBERSCHREIBT output/)** | der volle Lauf, rund 77 min |
| **Backtest schnell (→ output_probelauf/)** | der Probelauf |
| **Signifikanz nachrechnen** | `signifikanz.py` |
| **Kostensensitivität (Abb. 13)** | `kosten_sensitivitaet.py` |
| **Kontrollrechnung Kapitel 3** | `nachrechnen_kapitel3.py` |
| **Aktuelle Datei ausführen** | die geöffnete Datei, z. B. `nachrechnen_kapitel2.py` |

## Tests

```bash
pip install -r requirements-dev.txt
MPLBACKEND=Agg python -m pytest tests/ -q
```

**37 Unit-Tests** (ohne Netzwerk, ~4 s) prüfen:

| Datei | Prüft |
|---|---|
| `test_metrics.py` (11) | Kennzahlen gegen analytische Werte; Sharpe-Nenner-Schutz; Sortino zählt alle Tage, nicht nur Verlusttage |
| `test_optimizers.py` (9) | Σw = 1, Positionsobergrenze, Turnover-Limit, Risk-Parity-Eigenschaft, Minimum-Varianz-Eigenschaft, entarteter Sharpe-Fall, RF-Refit |
| `test_indicators.py` (5) | RSI-Wertebereich, Momentum = rollierende Summe, CS-Ränge, Monats-Aufzinsung, Look-Ahead-Schutz (Ziel = Folgemonat) |
| `test_cross_validation.py` (5) | Fold-Anzahl, Panel-Gruppierung, Purge/Embargo entfernen die Nachbarperioden, Fehler bei zu wenigen Perioden |
| `test_significance.py` (7) | Holm-Monotonie, Bootstrap erkennt echte und ignoriert unechte Unterschiede, DSR-Wertebereich |

## Wissenschaftliche Analyse

`main()` führt nach dem Backtest eine literaturgestützte Signifikanzanalyse durch:

- **Sharpe-Differenz-Test** nach Ledoit & Wolf (2008) — HAC-studentisierter
  Circular-Block-Bootstrap (4999 Ziehungen, `seed=42`), robust gegen
  Autokorrelation & Vol-Clustering.
- **Holm-Bonferroni-Korrektur** über **alle sechs** paarweisen Vergleiche
  (Holm 1979). Bewusst vollständig: Welche Paare geprüft werden, wäre sonst ein
  Freiheitsgrad des Auswertenden (Harvey/Liu/Zhu 2016).
- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) je Strategie.
- **Purged & Embargoed Cross-Validation** (López de Prado 2018) im RF-Tuning.

### Ergebnisse des maßgeblichen Laufs (`output/`, 15.08.2026)

| Strategie | CAGR | Vola | **Sharpe** | Sortino | Max. DD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Markowitz MVO | 23,14 % | 21,00 % | 0,9063 | 1,2956 | −34,28 % | 0,675 |
| Random Forest | 22,12 % | 19,03 % | **0,9357** | 1,3502 | −33,77 % | 0,655 |
| Equal Weight | 20,03 % | 17,56 % | 0,9005 | 1,2830 | −33,25 % | 0,603 |
| Risk Parity | 16,92 % | 16,45 % | 0,7901 | 1,1214 | −32,44 % | 0,522 |

| Vergleich | ΔSharpe | p (LW) | p (Holm) | Befund |
|---|---:|---:|---:|---|
| RF vs. MVO | +0,029 | 0,866 | 1,000 | n. s. |
| RF vs. EW | +0,035 | 0,773 | 1,000 | n. s. |
| RF vs. RP | +0,146 | 0,257 | 1,000 | n. s. |
| MVO vs. EW | +0,006 | 0,971 | 1,000 | n. s. |
| MVO vs. RP | +0,116 | 0,489 | 1,000 | n. s. |
| **Risk Parity vs. EW** | **−0,110** | **0,004** | **0,022** | **signifikant** |

**Zentraler Befund (robust):** Nach Holm-Korrektur schlägt *kein* aktiver Ansatz
die Equal-Weight-Benchmark signifikant. Signifikant ist allein, dass Risk Parity
*schlechter* abschneidet als 1/N — konsistent mit DeMiguel, Garlappi & Uppal
(2009). Markowitz erzielt zwar die höchste CAGR, aber einen schlechteren
Sharpe-Quotienten als 1/N; der Random Forest gewinnt ausschließlich
risikoadjustiert, und auch das nicht signifikant.

Die Deflated Sharpe Ratio liegt für alle vier Strategien über 0,95 (Hürde
SR₀ = 0,067 annualisiert) — die Sharpe-Werte selbst sind also nicht bloßes
Auswahlglück; nur die *Unterschiede* zwischen ihnen sind es.

Eine ausführliche Diskussion der Grenzen (u. a. **Survivorship Bias**) inkl.
Literaturverzeichnis steht in **[LIMITATIONS.md](LIMITATIONS.md)**.

## Methodische Korrekturen v4.1/v4.2

- **Einfache (arithmetische) Renditen** statt Log-Renditen: die Portfoliorendite
  ist exakt die gewichtete Summe der Asset-Renditen und konsistent zur
  Kennzahl-Berechnung. Monatsrenditen werden korrekt aufgezinst.
- **Driftbewusster Turnover:** neue Zielgewichte werden mit den über die letzte
  Halteperiode gedrifteten Vorgängergewichten verglichen. Equal Weight erhält so
  einen realistischen Rebalancing-Turnover (vorher fälschlich 0).
- **Sharpe und Sortino auf die Lehrbuchdefinition umgestellt:**
  Sharpe rechnet arithmetisch (nicht mehr CAGR im Zähler) und stimmt damit exakt
  mit `significance.py` überein; Sortinos Downside-Abweichung mittelt über alle
  Tage statt nur über die Verlusttage.
- **Theorie-Abbildungen reproduzierbar:** Die Bilder zu Kapitel 2 entstehen in
  `theory_plots.py` aus derselben eingefrorenen Kursdatei wie der Backtest —
  nicht als vorgefertigte Grafiken ohne erzeugenden Code.

## Erzeugte Dateien (`output/`)

| Datei | Inhalt |
|---|---|
| `00_live_dashboard_final.png` | Endzustand des Live-Dashboards |
| `01_kumulierte_renditen.png` | Depotwertverlauf + Drawdown-Panel |
| `02`/`03`/`03b_gewichte_*.png` | Gewichts-Heatmaps MVO / RF / Risk Parity |
| `04_efficient_frontier.png` | Effizienzlinie mit allen vier Portfolios |
| `05_performance_kennzahlen.png` | Sechspanel-Balkendiagramm der Kennzahlen |
| `06_rollierender_sharpe.png` | rollierender 1-Jahres-Sharpe |
| `07_feature_importance.png` | MDI-Wichtigkeit der RF-Merkmale |
| `08_frontier_evolution.png` | Wanderung der Effizienzlinie über die Zeit |
| `09_stress_test.png` | COVID-Crash 2020 und Zinswende 2021/22 |
| `10_turnover_performance.png` | Handelsumsatz gegen Folgemonatsrendite |
| `11_frontier_animation.gif` | Abbildung 8 als Film |
| `12_shap_explainability.png` | SHAP-Erklärbarkeit (nur wenn `shap` installiert) |
| `13_kosten_sensitivitaet.png` | aus `kosten_sensitivitaet.py`, nicht aus dem Lauf |
| `14`–`18_theorie_*.png` | Theorie-Abbildungen zu Kapitel 2 |
| `daily_returns.csv`, `cumulative_returns.csv` | Tagesrenditen und Depotwertverlauf |
| `performance_metrics.csv` | Kennzahlentabelle |
| `weights_markowitz.csv`, `weights_rf.csv`, `weights_risk_parity.csv` | Monatsgewichte |
| `turnover.csv` | Handelsumsatz je Strategie und Monat |
| `kosten_sensitivitaet.csv` | Sharpe über 0–100 bp Kostensatz |
| `experiment_log.json` | **das Laborprotokoll**: Parameter + Kennzahlen + Signifikanz |

## Der Ordner `arbeit/`

Er enthält eine **frühere Manuskriptfassung** der Seminararbeit (Markdown, mit
Bauskripten nach `.docx`) und wird nicht mehr gepflegt: Die Abgabefassung
entsteht außerhalb dieses Repositories in einer anderen Zitierweise. Die
Bauskripte verweisen zudem auf Abbildungen aus einem nicht mehr vorhandenen
Ergebnisordner. Der Ordner bleibt als Nachschlagewerk erhalten; für die
Ergebnisse der Arbeit ist er ohne Bedeutung.

## Kurzfassung

| Ich möchte … | Befehl | Dauer |
|---|---|---|
| prüfen, ob alles funktioniert | `MPLBACKEND=Agg python -m pytest tests/ -q` | ~4 s |
| eine Zahl der Arbeit nachrechnen | `python signifikanz.py` | Sekunden |
| gefahrlos ausprobieren | `PORTFOLIO_CONFIG=config.schnell.json python -m portfolio` | Minuten |
| das volle Experiment laufen lassen | `python -m portfolio` | ~77 min |

(jeweils im Stammordner des Repositories, bei aktivierter virtueller Umgebung)
