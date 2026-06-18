# Limitationen & methodische Diskussion

Dieses Dokument fasst die wissenschaftlichen Grenzen der Untersuchung zusammen
und ordnet sie in die Literatur ein. Es ist als Grundlage für das Kapitel
„Limitationen / kritische Würdigung" der Seminararbeit gedacht. Mehrere Punkte
wurden im Code bereits adressiert (siehe Markierungen *[behoben]* / *[optional]*);
andere bleiben grundsätzliche Einschränkungen, die ehrlich zu benennen sind.

---

## 1. Survivorship Bias (wichtigste Limitation)

Das Anlageuniversum besteht aus **15 heute existierenden Large-Cap-Aktien**
(`config.py` → `tickers`), die über den gesamten Zeitraum 2015–2024 gehandelt
wurden und überwiegend stark performt haben (u. a. AAPL, MSFT, NVDA). Damit ist
das Universum **rückblickend selektiert**: Unternehmen, die in diesem Zeitraum
ausschieden, fusionierten oder scheiterten, fehlen vollständig.

**Folge:** Die absoluten Renditen *aller* Strategien — inklusive der
Equal-Weight-Benchmark — sind **systematisch nach oben verzerrt**. Die zentrale
Frage „schlägt eine aktive Strategie den Markt?" wird dadurch angreifbar, weil
schon die Grundgesamtheit ein Überlebenden-Portfolio ist.

**Literatur:** Brown, Goetzmann, Ibbotson & Ross (1992); Elton, Gruber & Blake
(1996).

**Abmilderung:**
- Ideal: ein **point-in-time** zusammengesetztes Universum (z. B. die
  S&P-100/500-Mitglieder *zum jeweiligen Zeitpunkt*), das ausgeschiedene Titel
  enthält. Solche Daten sind frei oft nicht verfügbar.
- Pragmatisch: das Universum ist über `config.json` (`tickers`) frei wählbar —
  ein breiteres, weniger kuratiertes Set reduziert die Verzerrung.
- Mindestens: den Bias **explizit benennen** und Ergebnisse als *relativ*
  (Strategie vs. Strategie), nicht als *absolut* interpretieren. Der relative
  Vergleich ist vom Survivorship Bias deutlich weniger betroffen, da alle
  Strategien dasselbe verzerrte Universum nutzen.

---

## 2. Schätzfehler in den erwarteten Renditen

Die Markowitz-Optimierung verwendet **historische Mittelwerte** als Schätzer der
erwarteten Renditen. Erwartungswerte sind notorisch schwer zu schätzen, und die
Mean-Variance-Optimierung reagiert extrem sensibel auf Fehler in diesen Inputs
(„Fehler-Maximierer"). Fehler in den Mittelwerten kosten empirisch **rund 10-mal
mehr** als Fehler in den (Ko-)Varianzen.

**Literatur:** Merton (1980); Michaud (1989); Best & Grauer (1991);
Chopra & Ziemba (1993).

**Einordnung:** Genau hier setzt der Random Forest an (bessere Renditeprognose).
Dass RF und MVO am Ende kaum auseinanderliegen, ist **konsistent mit der
Literatur**: ML-Renditeprognosen haben typischerweise sehr kleine prädiktive R²
(Gu, Kelly & Xiu 2020). Die Positionsobergrenze (`max_weight = 0.20`) und das
Long-only-Constraint wirken zudem **wie eine implizite Regularisierung** und
verbessern die Out-of-Sample-Stabilität (Jagannathan & Ma 2003) — eine bewusste,
literaturgedeckte Designentscheidung.

**Mögliche Erweiterung:** Shrinkage der Mittelwerte (James-Stein) oder
Black-Litterman (1992) als zusätzliche Baseline.

---

## 3. Statistische Signifikanz & multiples Testen  *[behoben in v4.1+]*

Ein früherer i.i.d.-Bootstrap vernachlässigte Autokorrelation und
Volatilitäts-Clustering und korrigierte nicht für multiples Testen.

**Umgesetzt** (`portfolio/significance.py`):
- **Ledoit & Wolf (2008):** HAC-studentisierter Circular-Block-Bootstrap-Test
  für Sharpe-Differenzen (Block-Bootstrap nach Politis & Romano 1994).
- **Holm-Bonferroni** (Holm 1979) zur Korrektur der 5 paarweisen Vergleiche
  (Motivation: Harvey, Liu & Zhu 2016).

**Ergebnis (robust):** Nach Korrektur ist nur „Risk Parity schlechter als
Equal Weight" signifikant; **kein** aktiver Ansatz schlägt Equal Weight
signifikant — konsistent mit DeMiguel, Garlappi & Uppal (2009).

---

## 4. Backtest-Overfitting & einzelner historischer Pfad  *[teilweise behoben]*

Die Optimierung erfolgt über **einen** historischen Pfad (2015–2024, überwiegend
Bullenmarkt). Werden viele Strategien/Hyperparameter über denselben Pfad gewählt,
droht Backtest-Overfitting.

**Umgesetzt:** **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) in
`significance.py` — korrigiert die Sharpe für die Anzahl der Versuche und für
Nicht-Normalität (Schiefe/Kurtosis). *Hinweis:* Die hier verwendete Versuchszahl
`N` = Anzahl der Strategien ist konservativ niedrig; streng genommen müsste `N`
auch die Hyperparameter-Suchen einschließen, was die Hürde erhöhen würde.

**Verbleibende Limitation / Erweiterung:** Robustheit über mehrere Pfade via
stationärem Bootstrap (Politis & Romano 1994) sowie Sub-Perioden-Analysen
(z. B. COVID-Crash 2020, Zinswende 2022); minimale Backtest-Länge / PBO
(Bailey et al. 2017).

---

## 5. Informationsleck in der Kreuzvalidierung  *[optional behoben]*

Das monatliche Target wird aus **überlappenden** Tagesfenstern gebildet
(Momentum 252d, rollierendes Alpha/Beta), und das Label eines Monats reicht in
den Folgemonat. Standard-`TimeSeriesSplit` entfernt diese Überlappung nicht →
optimistisch verzerrte CV-Scores.

**Umgesetzt** (`portfolio/cross_validation.py`, aktivierbar via
`use_purged_cv=true`): **Purged & Embargoed Cross-Validation** nach
López de Prado (2018) — entfernt überlappende Label-Perioden (Purging) und eine
Embargo-Zone nach dem Test-Fenster.

---

## 6. Sharpe-Definition & Annualisierung

Die ausgewiesenen Kennzahlen (`metrics.py`) verwenden im Zähler die **geometrische**
Rendite (CAGR), während die klassische Definition (Sharpe 1994) den
**arithmetischen** Mittelwert der Überschussrenditen nutzt; die
Signifikanzanalyse (`significance.py`) verwendet konsistent die arithmetische
Sharpe. Zudem unterstellt die Annualisierung mit √252 **i.i.d.-Renditen** —
bei Autokorrelation ist sie verzerrt.

**Literatur:** Sharpe (1994); Lo (2002).

---

## 7. Konstanter risikofreier Zins

`risk_free_rate = 0.04` ist über 2015–2024 konstant, obwohl die US-Zinsen von
nahe 0 % (2015–2021) auf über 4 % (2023/24) stiegen. Das verzerrt die Sharpe
periodenübergreifend. **Erweiterung:** tatsächliche 3-Monats-T-Bill-Reihe
(Fama-French-Risikofreisatz).

---

## 8. Transaktionskosten

Die pauschalen 10 Basispunkte auf den Turnover (`transaction_cost = 0.0010`)
ignorieren Geld-Brief-Spannen und Market Impact. **Literatur:** Almgren & Chriss
(2000). **Erweiterung:** Sensitivitätsanalyse über mehrere Kostenniveaus.

---

## 9. Bereits behobene methodische Fehler (v4.1)

- **Einfache statt logarithmischer Renditen** in der Portfolio-Aggregation
  (gewichtete Summe nur für einfache Renditen gültig).
- **Driftbewusster Turnover**: Equal Weight erhält einen realistischen
  Rebalancing-Turnover statt fälschlich 0.

---

## Literaturverzeichnis

- Almgren, R. & Chriss, N. (2000): Optimal Execution of Portfolio Transactions. *Journal of Risk* 3(2), 5–40.
- Bailey, D. H. & López de Prado, M. (2014): The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5), 94–107.
- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2017): The Probability of Backtest Overfitting. *Journal of Computational Finance* 20(4), 39–69.
- Best, M. J. & Grauer, R. R. (1991): On the Sensitivity of Mean-Variance-Efficient Portfolios. *Review of Financial Studies* 4(2), 315–342.
- Black, F. & Litterman, R. (1992): Global Portfolio Optimization. *Financial Analysts Journal* 48(5), 28–43.
- Brown, S. J., Goetzmann, W. N., Ibbotson, R. G. & Ross, S. A. (1992): Survivorship Bias in Performance Studies. *Review of Financial Studies* 5(4), 553–580.
- Chopra, V. K. & Ziemba, W. T. (1993): The Effect of Errors in Means, Variances, and Covariances on Optimal Portfolio Choice. *Journal of Portfolio Management* 19(2), 6–11.
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009): Optimal Versus Naive Diversification. *Review of Financial Studies* 22(5), 1915–1953.
- Elton, E. J., Gruber, M. J. & Blake, C. R. (1996): Survivorship Bias and Mutual Fund Performance. *Review of Financial Studies* 9(4), 1097–1120.
- Gu, S., Kelly, B. & Xiu, D. (2020): Empirical Asset Pricing via Machine Learning. *Review of Financial Studies* 33(5), 2223–2273.
- Harvey, C. R., Liu, Y. & Zhu, H. (2016): … and the Cross-Section of Expected Returns. *Review of Financial Studies* 29(1), 5–68.
- Holm, S. (1979): A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics* 6(2), 65–70.
- Jagannathan, R. & Ma, T. (2003): Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps. *Journal of Finance* 58(4), 1651–1683.
- Ledoit, O. & Wolf, M. (2004): A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. *Journal of Multivariate Analysis* 88(2), 365–411.
- Ledoit, O. & Wolf, M. (2008): Robust Performance Hypothesis Testing with the Sharpe Ratio. *Journal of Empirical Finance* 15(5), 850–859.
- Lo, A. W. (2002): The Statistics of Sharpe Ratios. *Financial Analysts Journal* 58(4), 36–52.
- López de Prado, M. (2018): *Advances in Financial Machine Learning*. Wiley.
- Merton, R. C. (1980): On Estimating the Expected Return on the Market. *Journal of Financial Economics* 8(4), 323–361.
- Michaud, R. O. (1989): The Markowitz Optimization Enigma: Is 'Optimized' Optimal? *Financial Analysts Journal* 45(1), 31–42.
- Politis, D. N. & Romano, J. P. (1994): The Stationary Bootstrap. *Journal of the American Statistical Association* 89(428), 1303–1313.
- Sharpe, W. F. (1994): The Sharpe Ratio. *Journal of Portfolio Management* 21(1), 49–58.
