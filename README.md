# W-Seminar: Portfolio-Optimierung — Markowitz vs. Random Forest

Vergleich klassischer und ML-gestützter Portfolio-Optimierung über einen
Walk-Forward-Backtest (2015–2024) auf 15 US-Aktien.

**Strategien:** Markowitz MVO (Ledoit-Wolf) · Random Forest · Equal Weight (1/N) · Risk Parity (ERC)

## Projektstruktur

```
portfolio/            ← kanonische, modularisierte Codebasis (aktiv weiterentwickelt)
  config.py           Typisierte Config (dataclass) + Umgebungs-Setup
  metrics.py          Kennzahlen (CAGR, Sharpe, Sortino, Drawdown …)
  significance.py     Ledoit-Wolf-2008-Test, Holm-Bonferroni, Deflated Sharpe Ratio
  cross_validation.py Purged & Embargoed CV (López de Prado 2018)
  indicators.py       Technische Indikatoren, Monats-Aggregation, Feature-Spalten
  optimizers.py       MarkowitzLedoitWolf, RiskParityPortfolio, RFPortfolioOptimizer
  data.py             Marktdaten laden (yfinance) + einfache Renditen
  dashboard.py        Live-Training-Dashboard
  backtest.py         Rollierender Walk-Forward-Backtest
  plots.py            Visualisierungen + CSV-/JSON-Export
  run.py              Orchestrierung (main)

archive/projekt1.6.py ← eingefrorene v4.1-Baseline (Einzeldatei-Referenz; enthält
                        NICHT die späteren Paket-Erweiterungen)
LIMITATIONS.md        ← wissenschaftliche Limitationen & Literatur
output/               ← erzeugte Grafiken, CSVs, GIF, JSON
requirements.txt      ← Python-Abhängigkeiten
```

> Frühere Entwicklungsstufen (projekt1.0–1.5 sowie der abgebrochene 2.x-Zweig)
> wurden entfernt; sie bleiben über den ersten Commit (`Initial snapshot`) in der
> Git-Historie erhalten und sind bei Bedarf wiederherstellbar.
>
> **Hinweis:** `archive/projekt1.6.py` ist die eingefrorene v4.1-Baseline. Die
> wissenschaftlichen Erweiterungen (robuste Signifikanztests, Purged CV) sowie die
> Performance-Hebel liegen ausschließlich im Paket `portfolio/`.

## Ausführen

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m portfolio        # kanonische, aktuelle Version (empfohlen)
# oder die eingefrorene Baseline:
python archive/projekt1.6.py  # v4.1-Referenz (ohne spätere Erweiterungen)
```

Laufzeit ~16 min (Random-Forest-Tuning je Rebalancing-Monat). Ergebnisse landen
in `./output/`.

## Konfiguration

Alle einstellbaren Parameter liegen typisiert in der `dataclass` `Config`
(`portfolio/config.py`). Überschreiben ohne Code-Änderung:

```bash
cp config.example.json config.json   # gewünschte Werte anpassen
python -m portfolio                   # config.json wird automatisch geladen
# oder eigener Pfad:
PORTFOLIO_CONFIG=/pfad/zu/meiner.json python -m portfolio
```

Überschreibbare Schlüssel: `tickers`, `spy_ticker`, `start_date`, `end_date`,
`backtest_start`, `output_dir`, `risk_free_rate`, `train_years`, `n_frontier`,
`rf_n_iter`, `rf_cv_splits`, `max_weight`, `transaction_cost`, `rf_turnover_limit`,
`rf_retune_every`, `dashboard_update_every`, `use_purged_cv`, `cv_embargo`,
`mvo_turnover_limit`, `turnover_ref_drifted`, `min_variance_fallback`.
Nur die angegebenen Schlüssel werden überschrieben; der Rest bleibt auf Default.

### Fairness des Strategievergleichs

Drei Optionen machen den Vergleich Markowitz ↔ Random Forest strenger. Alle sind
**standardmäßig aus**, damit der dokumentierte Lauf in `output1.6/` reproduzierbar
bleibt; eingeschaltet verändern sie die Ergebnisse. Hintergrund und Literatur in
[LIMITATIONS.md](LIMITATIONS.md), Abschnitte 10–11.

- **`mvo_turnover_limit`** (Default `null`): Bisher hatte nur der Random Forest ein
  Turnover-Limit. Auf denselben Wert wie `rf_turnover_limit` gesetzt, unterscheiden
  sich die beiden Strategien wirklich **nur** im Renditeschätzer.
- **`turnover_ref_drifted`** (Default `false`): Die Turnover-Schranke *im Optimierer*
  gegen die kursgedrifteten Vorgängergewichte messen — dieselbe Referenz, die auch
  der ausgewiesene Turnover benutzt. Ohne dies kann der realisierte Turnover das
  nominelle Limit überschreiten.
- **`min_variance_fallback`** (Default `false`): Erwartet kein zulässiges Portfolio
  mehr als den risikofreien Zins, ist die Sharpe-Maximierung entartet — der
  Optimierer würde die Volatilität *maximieren*. Mit dieser Option weicht er auf das
  Minimum-Varianz-Portfolio aus. Der Fall wird **immer** ins Log geschrieben
  (`Max-Sharpe entartet: …`), auch wenn die Option aus ist.

### Performance-Hebel

Standardwerte (`1`) lassen das Verhalten exakt wie zuvor. Zum Beschleunigen:

- **`rf_retune_every`** (Default `1`): RF-Hyperparametersuche nur alle *k* Monate,
  dazwischen nur Refit auf dem aktuellen Fenster. `3` ≈ 2–3× schneller. **Achtung:**
  Werte > 1 verändern die Ergebnisse (andere Hyperparameter über die Zeit).
- **`dashboard_update_every`** (Default `1`): Live-Dashboard nur alle *k* Schritte
  rendern. Rein kosmetisch, **kein** Einfluss auf Kennzahlen.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

34 Unit-Tests (ohne Netzwerk) prüfen Kennzahlen gegen analytische Werte,
Optimierer-Constraints (Σw=1, Positionsobergrenze, Turnover-Limit, Risk-Parity-
Eigenschaft, Minimum-Varianz-Eigenschaft und den entarteten Sharpe-Fall),
Indikatoren sowie die korrekte Monats-Aufzinsung und den Look-Ahead-Schutz
(Ziel = Folgemonat).

## Wissenschaftliche Analyse

`main()` führt nach dem Backtest eine literaturgestützte Signifikanzanalyse durch:

- **Sharpe-Differenz-Test** nach Ledoit & Wolf (2008) — HAC-studentisierter
  Circular-Block-Bootstrap, robust gegen Autokorrelation & Vol-Clustering.
- **Holm-Bonferroni-Korrektur** für die paarweisen Vergleiche (Holm 1979).
- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) je Strategie.

Optional aktivierbar: **Purged & Embargoed Cross-Validation** (López de Prado
2018) im RF-Tuning via `use_purged_cv` in der Konfiguration.

**Zentrales Ergebnis (robust):** Nach Holm-Korrektur schlägt *kein* aktiver
Ansatz die Equal-Weight-Benchmark signifikant; nur „Risk Parity < Equal Weight"
ist signifikant — konsistent mit DeMiguel, Garlappi & Uppal (2009).

Eine ausführliche Diskussion der Grenzen (u. a. **Survivorship Bias**) inkl.
Literaturverzeichnis steht in **[LIMITATIONS.md](LIMITATIONS.md)**.

## Methodische Korrekturen v4.1

- **Einfache (arithmetische) Renditen** statt Log-Renditen: die Portfoliorendite
  ist exakt die gewichtete Summe der Asset-Renditen und konsistent zur
  Kennzahl-Berechnung. Monatsrenditen werden korrekt aufgezinst.
- **Driftbewusster Turnover:** neue Zielgewichte werden mit den über die letzte
  Halteperiode gedrifteten Vorgängergewichten verglichen. Equal Weight erhält so
  einen realistischen Rebalancing-Turnover (vorher fälschlich 0).
