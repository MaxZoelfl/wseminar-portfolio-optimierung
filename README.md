# W-Seminar: Portfolio-Optimierung — Markowitz vs. Random Forest

Vergleich klassischer und ML-gestützter Portfolio-Optimierung über einen
Walk-Forward-Backtest (2015–2024) auf 15 US-Aktien.

**Strategien:** Markowitz MVO (Ledoit-Wolf) · Random Forest · Equal Weight (1/N) · Risk Parity (ERC)

## Projektstruktur

```
portfolio/            ← kanonische, modularisierte Codebasis (v4.1)
  config.py           Konstanten + Umgebungs-Setup (Logging, Matplotlib, optionale Pakete)
  metrics.py          Kennzahlen (CAGR, Sharpe, Sortino, Drawdown …) + Bootstrap-Test
  indicators.py       Technische Indikatoren, Monats-Aggregation, Feature-Spalten
  optimizers.py       MarkowitzLedoitWolf, RiskParityPortfolio, RFPortfolioOptimizer
  data.py             Marktdaten laden (yfinance) + einfache Renditen
  dashboard.py        Live-Training-Dashboard
  backtest.py         Rollierender Walk-Forward-Backtest
  plots.py            Visualisierungen + CSV-/JSON-Export
  run.py              Orchestrierung (main)

projekt1.6.py         ← Legacy-Monolith (identische Logik, dient als Referenz)
projekt1.0 … 2.1.py   ← ältere/alternative Versionen (historisch)
output1.6/            ← erzeugte Grafiken, CSVs, GIF, JSON
requirements.txt      ← Python-Abhängigkeiten
```

## Ausführen

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m portfolio        # modularisierte Version (empfohlen)
# oder, identisch:
python projekt1.6.py       # Legacy-Monolith
```

Laufzeit ~16 min (Random-Forest-Tuning je Rebalancing-Monat). Ergebnisse landen
in `./output1.6/`.

## Methodische Korrekturen v4.1

- **Einfache (arithmetische) Renditen** statt Log-Renditen: die Portfoliorendite
  ist exakt die gewichtete Summe der Asset-Renditen und konsistent zur
  Kennzahl-Berechnung. Monatsrenditen werden korrekt aufgezinst.
- **Driftbewusster Turnover:** neue Zielgewichte werden mit den über die letzte
  Halteperiode gedrifteten Vorgängergewichten verglichen. Equal Weight erhält so
  einen realistischen Rebalancing-Turnover (vorher fälschlich 0).
