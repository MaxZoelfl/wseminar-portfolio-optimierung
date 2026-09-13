# Verbesserungsvorschläge & Feature-Ideen für das Portfolio-Optimierungsprojekt

**Vorbemerkung** (Umsetzungsstatus geprüft am 22.08.2026)

Diese Liste entstand am **09.05.2026, also vor der Umsetzung**. Sie liest sich wie
eine To-do-Liste, ist aber inzwischen zur Hälfte eine Erledigt-Liste:
**10 der 22 Punkte sind gebaut, 5 teilweise, 7 offen.**

Der ursprüngliche Wortlaut steht unverändert da. Ergänzt ist nur unter jedem Punkt
eine eingerückte Statuszeile — ✅ umgesetzt · 🟡 teilweise · ⬜ offen — mit dem Ort im
Code. So bleibt nachvollziehbar, was geplant war, was daraus wurde und was bewusst
liegen blieb.

| Status | Punkte |
|---|---|
| ✅ **umgesetzt** | 2.2 Purged CV · 3.1 Risk Parity · 4.1 CS-Ränge · 5.2 Signifikanztests · 6.2 Experimentprotokoll · 6.3 Config-System · 7.1 Turnover-Plot · 7.2 Frontier-GIF · 7.3 SHAP · 7.4 Stresstest |
| 🟡 **teilweise** | 1.1 Walk-Forward · 2.1 Leakage · 4.2 Volatilitäts-Features · 5.1 Bootstrap · 6.1 Performance |
| ⬜ **offen** | 1.2 Target-Variable · 1.3 Kovarianz · 2.3 mehr Modelle · 2.4 Ensemble · 3.2 CVaR · 3.3 Regime Switching · 4.3 Korrelationsfeatures |

**Einordnung:** Die offenen Punkte sind keine Versäumnisse, sondern
Abgrenzungen. Auf die Frage nach XGBoost ist die ehrliche Antwort, dass schon der
Random Forest die 1/N-Benchmark nicht signifikant schlägt — ein zweites Modell
hätte am Befund nichts geändert, aber die Zahl der getesteten Strategien erhöht
und damit über die Deflated Sharpe Ratio die Hürde für alle anderen mit angehoben.

---

# 1. Methodische Verbesserungen (wichtigster Hebel)

## 1.1 Walk-Forward strikter machen

> 🟡 **teilweise umgesetzt.** Das rollierende Schema läuft genau so — Training `month_end − 3 Jahre` bis `month_end`, danach eine Halteperiode bis zum Folgemonat (`portfolio/backtest.py`). Die zeitliche Vermischung ist über drei Sicherungen geschlossen (Handbook § 7). **`rf_retune_every` gibt es**, es steht im maßgeblichen Lauf aber bewusst auf `1`: Werte darüber verändern die Ergebnisse und wären keine Reproduktion mehr.

Aktuell: Rolling Window + monatliches Rebalancing

Problem:
- RF wird zu häufig neu getuned
- leichte zeitliche Vermischung in Aggregationen möglich

Verbesserung:
- echtes Walk-Forward Schema:
  - Train: t-36M → t
  - Validate: t → t+1M
  - Test: t+1M
- RF nur alle 3–6 Monate neu trainieren

Vorteil:
- realistischeres Backtesting
- weniger Overfitting

---

## 1.2 Target Variable verbessern (Noise-Problem)

> ⬜ **offen.** Ziel ist weiterhin die einfache Monatsrendite (`target_next_month` in `portfolio/indicators.py`). Weder Excess Return gegen SPY noch eine Klassifikation Up/Down/Neutral.

Aktuell:
- monatliche Rendite (sehr noisy)

Verbesserungen:
- Excess Return vs SPY
- risk-adjusted return proxy

Alternative (besser):
- Klassifikation:
  - Up / Down / Neutral

Vorteil:
- stabilere ML-Ergebnisse
- oft bessere RF-Performance

---

## 1.3 Kovarianz konsistenter machen

> ⬜ **offen.** Es bleibt bei Ledoit-Wolf je Fenster (`MarkowitzLedoitWolf.estimate_covariance`). Kein EWMA, kein OAS.

Aktuell:
- Ledoit-Wolf pro Fenster

Upgrade:
- Exponentially Weighted Covariance
- oder OAS (Oracle Approximating Shrinkage)

Vorteil:
- stabilere Portfolio-Gewichte
- weniger Extremallokationen

---

# 2. ML-Verbesserungen

## 2.1 Feature Leakage absichern

> 🟡 **teilweise umgesetzt.** Drei Sicherungen statt eines pauschalen Lags: Trainingsdaten werden bei `month_end` abgeschnitten, der letzte Trainingsmonat wird ausgelassen (sein Label wäre die gesuchte Rendite), und die Kreuzvalidierung purged. Als Test festgenagelt in `test_target_is_next_month_and_last_is_nan`. Explizite Leakage-Assertions über alle Features gibt es nicht.

Problem:
- mögliche implizite Zukunftsinformation in Features

Fix:
- alle Features 1 Monat laggen

oder:
- explizite Leakage-Assertions

---

## 2.2 Purged Cross-Validation (Lopez de Prado)

> ✅ **umgesetzt und Standard.** `portfolio/cross_validation.py`, aktiv über `use_purged_cv = true` und `cv_embargo = 0.02`. Gemessener Effekt: Ohne Purging erscheint der Random Forest um rund 0,044 Sharpe besser, als er ist (LIMITATIONS § 5).

Upgrade von TimeSeriesSplit:

- Purged K-Fold
- Embargo Period

Vorteil:
- verhindert Overlap zwischen Train/Test
- Standard in quantitativer Forschung

---

## 2.3 Mehr Modelle vergleichen

> ⬜ **offen.** Nur Random Forest. Ridge oder XGBoost als Baseline wären der naheliegendste nächste Schritt — und die naheliegendste Rückfrage an die Arbeit.

Aktuell:
- Random Forest

Ergänzen:
- Ridge Regression (Baseline)
- Elastic Net
- XGBoost / LightGBM

Vorteil:
- wissenschaftlich sauberer Vergleich
- stärkere Benchmark-Analyse

---

## 2.4 Ensemble statt Einzelmodell

> ⬜ **offen.** Kein Ensemble, kein Meta-Modell.

Statt nur RF:

- RF + XGBoost + Ridge
- Meta-Modell: Linear Regression

Vorteil:
- stabilere Prognosen
- weniger Overfitting

---

# 3. Portfolio-Optimierung

## 3.1 Risk Parity Benchmark hinzufügen

> ✅ **umgesetzt.** `RiskParityPortfolio` in `portfolio/optimizers.py`, seit v4 als vierte Strategie im Vergleich. Ergebnis: Sharpe 0,79 — die einzige Strategie, die 1/N **signifikant** unterliegt.

Fehlt aktuell

Vorteil:
- Standard-Benchmark in Finance
- sehr gute Vergleichsbasis

---

## 3.2 CVaR / Expected Shortfall

> ⬜ **offen.** Optimiert wird die Sharpe Ratio, nicht CVaR. Als Kennzahl wird der VaR 95 % immerhin ausgewiesen (`compute_metrics`).

Statt Sharpe-Maximierung:

- Minimiere Tail Risk (CVaR 95%)

Vorteil:
- realistischeres Risiko-Modell
- bessere Krisen-Performance

---

## 3.3 Regime Switching

> ⬜ **offen.** Keine Regimeerkennung. Der Stresstest (7.4) betrachtet Krisenphasen nur nachträglich, er steuert nichts.

- Bull / Bear Markt erkennen (HMM oder SMA)
- unterschiedliche Gewichte je Regime

Vorteil:
- adaptive Strategie
- deutlich realistischer

---

# 4. Feature Engineering

## 4.1 Cross-Sectional Features

> ✅ **umgesetzt.** `add_cross_sectional_ranks` in `portfolio/indicators.py` vergibt je Monat Perzentilränge unter den 15 Titeln — für RSI, Momentum 1M/3M/12M und Alpha (`RANK_COLS`). Aus 15 Basis-Merkmalen werden so 20.

Aktuell:
- jedes Asset isoliert

Upgrade:
- Ranking Features:
  - Momentum Rank
  - RSI Rank
  - Return Rank

Vorteil:
- verbessert relative Stärke Modellierung

---

## 4.2 Volatilitäts-Regime Features

> 🟡 **teilweise umgesetzt.** Realisierte Volatilität über 21 und 63 Tage ist als Merkmal drin (`compute_volatility_features`). GARCH und Volatility Spikes nicht.

- GARCH Volatility
- Volatility Spikes
- Realized Volatility

---

## 4.3 Korrelationsfeatures

> ⬜ **offen.** Keine Durchschnittskorrelation, keine Eigenwerte, kein Diversifikationsscore als Merkmal. Die Korrelationsstruktur steckt nur in der Kovarianzmatrix des Optimierers.

- Durchschnittskorrelation je Asset
- Eigenwerte Korrelationsmatrix
- Diversifikationsscore

---

# 5. Backtest-Verbesserungen

## 5.1 Bootstrap Backtesting

> 🟡 **teilweise umgesetzt.** Es gibt einen Bootstrap mit 4999 Ziehungen — aber auf den Renditereihen im Signifikanztest (Circular Block, Ledoit-Wolf 2008), nicht als 1000 vollständig neu gerechnete Backtests. Die Sharpe-Verteilung entsteht also aus Resampling der Ergebnisse, nicht aus Resampling des Experiments.

- 1000 resampled Backtests

Outputs:
- Sharpe Distribution
- Wahrscheinlichkeit RF > MVO

Vorteil:
- statistische Robustheit

---

## 5.2 Signifikanztests

> ✅ **umgesetzt, und strenger als hier vorgeschlagen.** `portfolio/significance.py`: HAC-studentisierter Block-Bootstrap nach Ledoit & Wolf (2008) statt eines einfachen t-Tests, dazu Holm-Bonferroni über alle sechs Paare und die Deflated Sharpe Ratio. Ein t-Test auf Renditen wäre bei autokorrelierten Reihen zu großzügig gewesen.

- Dieese Test (Sharpe Vergleich)
- t-Test auf Returns
- Confidence Intervals

---

# 6. Engineering / Struktur

## 6.1 Performance Optimierung

> 🟡 **teilweise — und bewusst wieder zurückgenommen.** `rf_retune_every` spart Rechenzeit. Die Parallelisierung wurde dagegen **absichtlich abgeschaltet** (`deterministic = true`, `n_jobs=1`): Gleitkommaaddition ist nicht assoziativ, weshalb parallele Läufe in 4 von 119 Monaten andere Hyperparameter kürten. Reproduzierbarkeit schlägt Geschwindigkeit (LIMITATIONS § 12).

- RF Training caching
- Vektorisierte Features
- Parallelisierung

---

## 6.2 Experiment Tracking

> ✅ **umgesetzt.** `save_experiment_json` schreibt `output/experiment_log.json` mit Zeitstempel, allen Parametern, Kennzahlen und Signifikanzergebnissen. Kein MLflow — für ein Experiment mit genau einer maßgeblichen Konfiguration wäre das Overhead.

Einführen:

- MLflow oder JSON Logging

Beispiel:
```json
{
  "window": "36M",
  "sharpe_rf": 1.23,
  "params": {}
}
```

---

## 6.3 Config System

> ✅ **umgesetzt, mit JSON statt YAML.** Die `Config`-dataclass in `portfolio/config.py` hält alle Parameter; überschreiben über `config.json` oder `PORTFOLIO_CONFIG`. Nichts ist mehr hardcodiert. JSON, weil es ohne Zusatzpaket auskommt.

Aktuell hardcoded
Besser:
- YAML Konfiguration:
    - Tickers
    - Window Sizes
    - Costs
    - Model Params

---

## 7. „Wow“-Features (Top-Bewertung)

### 7.1 Turnover vs Performance Plot

> ✅ **umgesetzt.** `plot_turnover_performance` → Abbildung 10. Zeigt den Handelsumsatz gegen die Folgemonatsrendite.


- x: Turnover
- y: Sharpe Ratio

zeigt echte Effizienz der Strategie

---

## 7.2 Animated Efficient Frontier

> ✅ **umgesetzt.** `create_animated_frontier_gif` → Abbildung 11, ein Einzelbild je Rebalancing-Termin.


- Zeitliche Entwicklung als GIF/Video

--- 

## 7.3 SHAP Explainability

> ✅ **umgesetzt.** `plot_shap_values` → Abbildung 12. Optional: fehlt das Paket `shap`, entfällt nur diese Abbildung.


- Feature Contribution im RF

Vorteil:

- interpretierbares ML-Modell

---

## 7.4 Stress Testing

> ✅ **umgesetzt — mit einer Abweichung.** `plot_stress_test` → Abbildung 9 prüft den **COVID-Crash 2020** und die **Zinswende 2021/22**. Die Finanzkrise 2008 ist nicht dabei und kann es nicht sein: Die Kursdaten beginnen 2013.


- 2008 Finanzkrise
- COVID Crash

Vorteil:

- Robustheitsanalyse