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

## 5. Informationsleck in der Kreuzvalidierung  *[behoben, Standard an]*

Das monatliche Target wird aus **überlappenden** Tagesfenstern gebildet
(Momentum 252d, rollierendes Alpha/Beta), und das Label eines Monats reicht in
den Folgemonat. Standard-`TimeSeriesSplit` entfernt diese Überlappung nicht →
optimistisch verzerrte CV-Scores.

**Umgesetzt** (`portfolio/cross_validation.py`, seit 14.08.2026 **Standard**,
`use_purged_cv = True`, `cv_embargo = 0.02`): **Purged & Embargoed
Cross-Validation** nach López de Prado (2018) — entfernt überlappende
Label-Perioden (Purging) und eine Embargo-Zone nach dem Test-Fenster.

**Ausmaß des Lecks, gemessen** (Lauf mit `use_purged_cv=false` gegen den
heutigen Standard; alles andere identisch):

| | Sharpe RF | CAGR RF | Vola RF | max. DD RF |
|---|---:|---:|---:|---:|
| `TimeSeriesSplit` | 1,0068 | 23,32 % | 19,19 % | −35,57 % |
| **Purged & Embargoed** | **0,9626** | **22,31 %** | **19,02 %** | **−33,76 %** |

Rund **0,04 Sharpe** und ein Prozentpunkt CAGR des scheinbaren
Random-Forest-Vorsprungs waren also Leckage. (Zur Genauigkeit dieser Zahl siehe
Abschnitt 12: Die Lauf-zu-Lauf-Streuung des Random Forest lag vor Einführung des
deterministischen Modus bei rund 0,010 — der Effekt ist also etwa viermal so
groß wie das Rauschen, aber keine exakte Größe.) Markowitz, Equal Weight und Risk
Parity bleiben auf vier Nachkommastellen unverändert — sie werden nicht
kreuzvalidiert. Das ist zugleich die Kontrolle, dass die Änderung nur dort
wirkt, wo sie wirken soll. Der Holm-Befund ändert sich nicht: einzig
„Risk Parity < Equal Weight" bleibt signifikant (p = 0,018).

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

## 10. Fairness des Vergleichs Markowitz ↔ Random Forest  *[behoben, Standard an]*

### Vorbemerkung: Der Random Forest maximiert **nicht** die Rendite

Ein naheliegender Einwand lautet, die KI-Strategie ziele auf Renditemaximierung
ohne Rücksicht auf das Risiko — der Vergleich mit Markowitz sei deshalb schief.
Das trifft auf diese Implementierung **nicht** zu. `RFPortfolioOptimizer.optimize()`
ruft denselben Markowitz-Löser auf wie die klassische Strategie: dieselbe
Zielfunktion (Sharpe-Maximierung, das Risiko steht als σ_p im Nenner), dieselbe
Ledoit-Wolf-Kovarianzmatrix (in `backtest.py` **einmal** geschätzt und an alle
Strategien weitergereicht), dieselben Nebenbedingungen (long-only, Σw = 1,
`max_weight`). Der Random Forest ersetzt ausschließlich die **Renditeschätzung**.

Damit ist der Vergleich als *Ceteris-paribus-Experiment* angelegt: Er isoliert
genau die Größe, an der die MVO laut Literatur scheitert (μ, vgl. Abschnitt 2).
Empirisch bestätigt sich das auch — der RF ist mit 19,3 % annualisierter
Volatilität **weniger** schwankungsanfällig als die klassische MVO (21,0 %).

### Verbleibende Asymmetrie (Turnover-Restriktion)

Streng genommen unterschieden sich die beiden Strategien bisher in **zwei**
Punkten statt in einem: Nur der Random Forest bekam ein Turnover-Limit
(`rf_turnover_limit = 0.30`), die klassische MVO wurde unrestringiert optimiert.
Das ist eine Verletzung des Ceteris-paribus-Prinzips. Sie wirkt in beide
Richtungen — die Schranke senkt einerseits die Handelskosten des RF, wirkt
andererseits aber wie eine zusätzliche Regularisierung und schränkt sein
Optimum ein. Praktisch fällt sie im dokumentierten Lauf gering aus: Der RF
handelt mit Ø 30,8 % Turnover ohnehin **mehr** als die MVO (Ø 13,6 %), das Limit
bindet also selten.

**Umgesetzt** (`config.mvo_turnover_limit`, seit 14.08.2026 **Standard 0.30**):
Dieselbe Handelsrestriktion gilt für beide Strategien; erst dann unterscheiden
sie sich ausschließlich im Renditeschätzer.

### Bezugspunkt der Turnover-Schranke

Der *ausgewiesene* Turnover wird gegen die über die Halteperiode
**kursgedrifteten** Vorgängergewichte gemessen (korrekt — so steht das Depot zum
Rebalancing-Zeitpunkt tatsächlich da). Die Schranke *im Optimierer* verglich
dagegen mit den **ungedrifteten** Zielgewichten des Vormonats. Beide Größen
messen dadurch nicht dasselbe, und der realisierte Turnover kann das nominelle
Limit überschreiten — im dokumentierten Lauf liegt der RF-Mittelwert mit 30,8 %
über dem Limit von 30 %.

**Umgesetzt** (`config.turnover_ref_drifted`, seit 14.08.2026 **Standard `true`**):
beide Größen nutzen dieselbe Referenz.

**Ausmaß im echten Backtest 2015–2024** (archivierte Läufe in
`Archiv/Robustheitslaeufe/`):
Der Random Forest überschritt sein nominelles Limit von 30 % in **100 von 118**
Monaten, im Extremfall mit 34,9 % Turnover. Mit der korrigierten Referenz sind
es **null** Überschreitungen, der Maximalwert liegt exakt auf 30,0 %. Der
Fehler war also nicht kosmetisch — die Restriktion war über weite Strecken des
Backtests faktisch wirkungslos.

---

## 11. Entartete Sharpe-Maximierung bei negativer Überrendite  *[behoben, Standard an]*

Die Zielfunktion `max_sharpe` maximiert (μ_p − r_f) / σ_p. Das ist nur sinnvoll,
solange der **Zähler positiv** ist. Erwartet kein zulässiges Portfolio mehr als
den risikofreien Zins (unter long-only mit Σw = 1 ist μ_p ein gewichteter
Mittelwert der μ_i, also nie größer als max μ_i), wird der Zähler negativ — und
dann macht ein **größeres** σ_p im Nenner den Bruch weniger negativ. Der
Optimierer maximiert in dieser Lage also das Risiko statt es zu minimieren.

Numerische Kontrolle (8 Assets, r_f = 4 %, identische Kovarianzmatrix):

| Szenario | Volatilität der „Max-Sharpe"-Lösung | Minimum-Varianz |
|---|---|---|
| μ ≈ +12 % (Normalfall) | 0,154 | 0,146 |
| μ ≈ −5 % (Baisse) | **0,471** | 0,146 |

Im Baisse-Fall wählt der Optimierer also die gut **dreifache** Volatilität.

**Theoretischer Hintergrund (Merton 1972).** Das ist keine Eigenart dieser
Implementierung, sondern ein seit 1972 bewiesenes Ergebnis. Merton leitet den
Effizienzrand geschlossen her und zeigt, dass ein **Tangentialportfolio** — also die
Lösung der Sharpe-Maximierung bei Vorhandensein einer risikofreien Anlage — nur dann
als *effizientes* Portfolio existiert, wenn

$$r_f < \bar E = A/C$$

gilt, wobei Ē die erwartete Rendite des Minimum-Varianz-Portfolios ist
(S. 1865, Theorem II: „… if and only if R < Ē"). Bei r_f = Ē gibt es überhaupt keinen
Berührpunkt — die Kapitalmarktlinien sind dann genau die Asymptoten des Rands —, bei
r_f > Ē liegt der Berührpunkt auf dem **ineffizienten** unteren Ast. Mertons Fazit
dazu (S. 1868): „Under no condition can one construct the entire frontier (with the
riskless security included) by drawing tangent lines to the upper and lower parts of
the frontier for risky assets only." Die Entartung in der Tabelle oben ist also die
numerische Erscheinungsform eines bekannten Struktursatzes, und der Rückfall auf das
Minimum-Varianz-Portfolio ist die theoretisch konsistente Antwort darauf.

⚠ **Zwei Schwellen nicht verwechseln.** Mertons Grenze Ē = A/C ist **ohne**
Nichtnegativitätsbedingung hergeleitet — er lässt Leerverkäufe und Kreditaufnahme
ausdrücklich zu (S. 1852, Fn. 3: „the only constraint on the x_i is that they sum to
unity"). Unter long-only mit Σw = 1 und Kappe 20 % ist A/C daher **nicht** die
maßgebliche Grenze; hier greift die Entartung erst, wenn kein *zulässiges* Portfolio
mehr μ_p > r_f erreicht, also wenn max μ_i ≤ r_f — und genau das prüft der Code.
Merton belegt deshalb die **Struktur** des Problems (die Sharpe-Maximierung verliert
oberhalb einer Zinsschwelle ihren Sinn), nicht den Auslöser der Abfrage in
`optimizers.py`. In dieser Abgrenzung ist der Beleg auch im Kolloquium haltbar.
Betroffen sind grundsätzlich beide Strategien; praktisch trifft es eher den
Random Forest, weil seine Prognosen konditional sind und in Abschwüngen negativ
werden können, während der gleitende 3-Jahres-Mittelwert der MVO im Sample
2015–2024 fast durchgehend deutlich über 4 % liegt.

**Umgesetzt** (`portfolio/optimizers.py`): Der Fall wird erkannt und **immer**
protokolliert (`Max-Sharpe entartet: …`) — auch bei ausgeschalteter Option, damit
sich im Laufprotokoll nachzählen lässt, wie oft er auftrat. Mit
`config.min_variance_fallback` (seit 14.08.2026 **Standard `true`**) weicht die Strategie dann auf das
**Minimum-Varianz-Portfolio** aus (unter denselben Nebenbedingungen inklusive
Turnover-Schranke). Ökonomisch ist das die konsistente Wahl: Wenn die
Renditeschätzung keine Kompensation für Risiko verspricht, ist Risiko-
minimierung die einzige verbleibende Zielgröße — dieselbe Logik, aus der auch
das Minimum-Varianz-Portfolio als renditeschätzungsfreie Strategie beliebt ist.

**Befund für 2015–2024: der Fall trat in keinem einzigen der 119 Monate ein.**
Das ist plausibel — das Sample ist überwiegend Bullenmarkt, und selbst 2022 lag
weder der gleitende 3-Jahres-Mittelwert noch die RF-Prognose des besten Titels
unter 4 %. Die Option ist damit **reine Absicherung**: Sie ändert an den
Ergebnissen dieser Arbeit nichts, verhindert aber ein ökonomisch unsinniges
Verhalten in Stichproben mit ausgeprägten Baissephasen (etwa 2000–2002 oder
2008). Genau deshalb ist sie als Limitation interessant und nicht als Fehler in
den Zahlen.

---

## 11a. Ergebnis des strengen Vergleichslaufs

Alle drei Korrekturen wurden gemeinsam durchgerechnet (heute Standard, `output/`,
Konfiguration `config.fair.json`); Universum, Zeitraum, Kosten, Features und
Hyperparameterraum sind identisch zum Referenzlauf. **Der zentrale Befund bleibt
unverändert.**

| Sharpe Ratio | Referenz | streng | Δ |
|---|---|---|---|
| Markowitz MVO | 0,9151 | 0,9114 | −0,0037 |
| Random Forest | 1,0119 | 1,0068 | −0,0051 |
| Equal Weight | 0,9131 | 0,9131 | 0 |
| Risk Parity | 0,7858 | 0,7858 | 0 |

Equal Weight und Risk Parity sind definitionsgemäß unberührt (keine
Renditeschätzung, keine Turnover-Schranke). MVO und RF verlieren je rund 0,004
bis 0,005 Sharpe-Punkte — die schärfere Turnover-Bindung kostet ein wenig
Flexibilität, ohne die Rangfolge zu ändern.

Auch die Signifikanzanalyse verschiebt sich praktisch nicht:

| Test | p (Referenz) | p (streng) |
|---|---|---|
| RF vs. MVO | 0,688 | 0,697 |
| RF vs. Equal Weight | 0,528 | 0,542 |
| MVO vs. Equal Weight | 0,953 | 0,971 |
| **Risk Parity vs. Equal Weight** | **0,004** | **0,004** |
| MVO vs. Risk Parity | 0,476 | 0,489 |

Nach Holm-Korrektur ist weiterhin **allein** „Risk Parity schlechter als Equal
Weight" signifikant; kein aktiver Ansatz schlägt die naive 1/N-Benchmark.

**Interpretation für die Arbeit:** Der Befund ist *robust gegenüber der
Vergleichsmethodik*. Er beruht nicht darauf, dass der Random Forest heimlich
bevorzugt wurde — auch unter streng identischen Handelsrestriktionen bleibt
sein Vorsprung statistisch ununterscheidbar von Null. Das ist ein stärkeres
Argument als das ursprüngliche Ergebnis allein, weil der naheliegendste
Einwand gegen das Versuchsdesign damit empirisch ausgeräumt ist.

---

## 12. Bit-genaue Reproduzierbarkeit  *[behoben und nachgewiesen]*

`RandomForestRegressor(random_state=42, n_jobs=-1)` fixiert zwar den
Zufallsgenerator, **nicht** aber die Reihenfolge, in der die parallelen Threads
ihre Teilergebnisse aufsummieren. Gleitkommaaddition ist nicht assoziativ,
weshalb zwei Läufe desselben Codes minimal auseinanderliegen — an einem
synthetischen Backtest über 47 Monate gemessen: max. 3·10⁻¹⁰ in den
Tagesrenditen.

### ⚠ Korrektur vom 15.08.2026: Der Effekt ist **nicht** vernachlässigbar

Frühere Fassungen dieses Abschnitts nannten die Abweichung „weit unterhalb der
Rundung irrelevant". Das gilt für die Tagesrenditen — **nicht aber für den Weg
über die Hyperparametersuche.** Zwei Läufe mit identischer Konfiguration wurden
verglichen:

| | Sharpe RF | CAGR RF |
|---|---:|---:|
| Lauf A | 0,9626 | 22,31 % |
| Lauf B | 0,9527 | 22,12 % |

Ursache: `RandomizedSearchCV` vergleicht 30 Kandidaten anhand ihres
CV-Scores. Liegen zwei Kandidaten nahezu gleichauf, genügt eine Abweichung in
der 10. Nachkommastelle, um einen **anderen** Sieger zu küren — und dann
unterscheiden sich nicht Rundungsstellen, sondern Baumtiefen. Nachgezählt an den
Laufprotokollen: In **4 von 119 Monaten** wurde ein anderer Parametersatz
gewählt, in einem Fall `depth=3` statt `depth=8`. Von dort pflanzt sich der
Unterschied über Prognosen, Gewichte und Umschlag bis in die Kennzahlen fort.

**Größenordnung: rund 0,010 im Sharpe-Quotienten des Random Forest.** Die
anderen drei Strategien sind davon nicht betroffen — sie haben keine
Hyperparameter und reproduzieren sich exakt.

**Umgesetzt** (`config.deterministic`, seit 15.08.2026 **Standard `true`**):
Random Forest und Hyperparametersuche laufen einkernig (`n_jobs=1`); damit ist
der Lauf bitgenau wiederholbar. Kosten: die Tuning-Zeit steigt von rund 12 auf
rund 41 Minuten. Für einen einmaligen Referenzlauf ist das der richtige Tausch —
eine Arbeit, deren Anhang „Reproduzierbarkeit" verspricht, muss reproduzierbar sein.

### Die größere Ursache: die Rohdaten selbst

Der Threading-Effekt war nur die halbe Wahrheit. Ein zweiter, **um Größenordnungen
stärkerer** Störfaktor liegt vor dem Code: **Yahoo Finance liefert bei jedem Abruf
leicht andere Kurse.** Zwei Downloads desselben Zeitraums im Abstand von
20 Sekunden verglichen:

| | Wert |
|---|---|
| abweichende Kurswerte | **36 553 von 45 285** |
| maximale absolute Abweichung | 2,4·10⁻⁴ |
| maximale **relative** Abweichung | **1,3·10⁻⁶** |

Das ist rund das **Zehnmilliardenfache** des Threading-Effekts — und es trifft
alle Strategien, nicht nur den Random Forest. Nachweisbar an den Markowitz-
Gewichten: Sie weichen bereits im **ersten** Rebalancing-Monat um 1,4·10⁻³ ab,
obwohl Markowitz weder kreuzvalidiert noch einen Zufallsgenerator benutzt.

Gegenprobe mit **festen** Kursen, jeweils drei getrennte Prozesse:

| Prüfung | Ergebnis |
|---|---|
| Markowitz-Optimierung, feste Daten | bitgleich |
| RF-Tuning, feste Daten, `n_jobs=1` | bitgleich |
| RF-Tuning, feste Daten, `n_jobs=-1` | Abweichung ab der 16. Stelle |

**Umgesetzt** (`config.price_cache`, Standard `./data/prices.pkl`): Der erste
Download wird abgelegt, alle weiteren Läufe lesen von dort. Neu laden = Datei
löschen.

### ✅ Nachweis

Zwei vollständige, unabhängig gestartete Läufe mit `deterministic = true` und
fester Kursdatei erzeugen **bitgleiche** Ergebnisdateien (alle sieben CSV,
MD5-Vergleich). Reproduzierbarkeit ist damit nicht behauptet, sondern gezeigt.

**Konsequenz für die Interpretation:** Der in Abschnitt 5 genannte Effekt der
Purged Cross-Validation liegt bei rund 0,05 Sharpe und damit deutlich über der
früheren Lauf-zu-Lauf-Streuung von etwa 0,010. Er ist real, sollte im Text aber
als „rund 0,05" und nicht als exakte Zahl geführt werden.

**Für die Archivierung:** Ohne `data/prices.pkl` ist der Backtest grundsätzlich
nicht nachrechenbar — wer die Kurse später neu lädt, erhält andere Werte. Die
Datei gehört deshalb mit Abrufdatum zum Projekt.

---

## Literaturverzeichnis

**Verfügbarkeit** (Stand der Online-Prüfung Juni 2026):
📄 = kostenfreier Volltext verlinkt · 🔒 = nur kostenpflichtig/Bibliothek (DOI verlinkt).

> **Wichtiger Hinweis zur Zitierweise:** Die folgenden bibliografischen Angaben
> (Band, Heft, Seitenbereich, DOI) wurden online verifiziert. **Seitengenaue
> Belege im Fließtext** (z. B. für wörtliche Zitate) sind jedoch **am Original zu
> prüfen** — DOI/Volltext-Links führen zur jeweiligen Quelle.

- Almgren, R. & Chriss, N. (2000): Optimal Execution of Portfolio Transactions. *Journal of Risk* 3(2), 5–39. 📄 [Volltext (Semantic Scholar)](https://www.semanticscholar.org/paper/Optimal-execution-of-portfolio-trans-actions-Almgren-Chriss/4ea1885d7f00dc2ba59be2d6cc62923de23599ce)
- Bailey, D. H. & López de Prado, M. (2014): The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. *Journal of Portfolio Management* 40(5), 94–107. 📄 [PDF (davidhbailey.com)](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) · [SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2017): The Probability of Backtest Overfitting. *Journal of Computational Finance* 20(4), 39–69. DOI [10.21314/JCF.2016.322](https://doi.org/10.21314/JCF.2016.322) · 📄 [SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
- Best, M. J. & Grauer, R. R. (1991): On the Sensitivity of Mean-Variance-Efficient Portfolios to Changes in Asset Means. *Review of Financial Studies* 4(2), 315–342. DOI [10.1093/rfs/4.2.315](https://doi.org/10.1093/rfs/4.2.315) · 🔒 [JSTOR](https://www.jstor.org/stable/2962107)
- Black, F. & Litterman, R. (1992): Global Portfolio Optimization. *Financial Analysts Journal* 48(5), 28–43. 🔒 DOI [10.2469/faj.v48.n5.28](https://doi.org/10.2469/faj.v48.n5.28)
- Brown, S. J., Goetzmann, W. N., Ibbotson, R. G. & Ross, S. A. (1992): Survivorship Bias in Performance Studies. *Review of Financial Studies* 5(4), 553–580. DOI [10.1093/rfs/5.4.553](https://doi.org/10.1093/rfs/5.4.553) · 📄 [PDF (UMD)](https://terpconnect.umd.edu/~wermers/ftpsite/FAME/Brown_Goetzmann_Ibbotson_Ross.pdf)
- Chopra, V. K. & Ziemba, W. T. (1993): The Effect of Errors in Means, Variances, and Covariances on Optimal Portfolio Choice. *Journal of Portfolio Management* 19(2), 6–11. DOI [10.3905/jpm.1993.409440](https://doi.org/10.3905/jpm.1993.409440) · 📄 [PDF (Duke)](https://people.duke.edu/~charvey/Teaching/BA453_2006/Chopra_The_effect_of_1993.pdf)
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009): Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy? *Review of Financial Studies* 22(5), 1915–1953. DOI [10.1093/rfs/hhm075](https://doi.org/10.1093/rfs/hhm075) · 📄 [LBS Research (PDF)](https://lbsresearch.london.edu/id/eprint/407/)
- Elton, E. J., Gruber, M. J. & Blake, C. R. (1996): Survivor Bias and Mutual Fund Performance. *Review of Financial Studies* 9(4), 1097–1120. DOI [10.1093/rfs/9.4.1097](https://doi.org/10.1093/rfs/9.4.1097) · 📄 [PDF](https://finance.martinsewell.com/fund-performance/EltonGruberBlake1996a.pdf)
- Gu, S., Kelly, B. & Xiu, D. (2020): Empirical Asset Pricing via Machine Learning. *Review of Financial Studies* 33(5), 2223–2273. DOI [10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009) · 📄 [NBER w25398](https://www.nber.org/papers/w25398)
- Harvey, C. R., Liu, Y. & Zhu, H. (2016): … and the Cross-Section of Expected Returns. *Review of Financial Studies* 29(1), 5–68. DOI [10.1093/rfs/hhv059](https://doi.org/10.1093/rfs/hhv059) · 📄 [NBER w20592](https://www.nber.org/papers/w20592)
- Holm, S. (1979): A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics* 6(2), 65–70. 📄 [PDF](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf) · [JSTOR 4615733](https://www.jstor.org/stable/4615733)
- Jagannathan, R. & Ma, T. (2003): Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps. *Journal of Finance* 58(4), 1651–1684. DOI [10.1111/1540-6261.00580](https://doi.org/10.1111/1540-6261.00580) · 📄 [NBER w8922](https://www.nber.org/papers/w8922)
- Ledoit, O. & Wolf, M. (2004): A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. *Journal of Multivariate Analysis* 88(2), 365–411. DOI [10.1016/S0047-259X(03)00096-4](https://doi.org/10.1016/S0047-259X(03)00096-4) · 📄 [PDF](https://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf)
- Ledoit, O. & Wolf, M. (2008): Robust Performance Hypothesis Testing with the Sharpe Ratio. *Journal of Empirical Finance* 15(5), 850–859. DOI [10.1016/j.jempfin.2008.03.002](https://doi.org/10.1016/j.jempfin.2008.03.002) · 📄 [CORE (PDF)](https://core.ac.uk/outputs/11251901/)
- Lo, A. W. (2002): The Statistics of Sharpe Ratios. *Financial Analysts Journal* 58(4), 36–52. DOI [10.2469/faj.v58.n4.2453](https://doi.org/10.2469/faj.v58.n4.2453) · 📄 [Preprint (Semantic Scholar)](https://www.semanticscholar.org/paper/The-Statistics-of-Sharpe-Ratios-Lo/05561b77acfdd034a585c32048819cc9ba6d1434)
- López de Prado, M. (2018): *Advances in Financial Machine Learning*. Hoboken, NJ: Wiley. ISBN 978-1-119-48208-6. 🔒 [Wiley](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) · Kap. 7 (CV) als Vorschau: [SSRN 3104847](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3104847)
- Merton, R. C. (1972): An Analytic Derivation of the Efficient Portfolio Frontier. *Journal of Financial and Quantitative Analysis* 7(4), 1851–1872. 🔒 [JSTOR 2329621](https://www.jstor.org/stable/2329621) — Volltext liegt als `Quellen/An_Analytic_Derivation_of_the_Efficient_Portfolio_Frontier_BSB.pdf` vor
- Merton, R. C. (1980): On Estimating the Expected Return on the Market: An Exploratory Investigation. *Journal of Financial Economics* 8(4), 323–361. DOI [10.1016/0304-405X(80)90007-0](https://doi.org/10.1016/0304-405X(80)90007-0) · 📄 [NBER w0444](https://www.nber.org/papers/w0444)
- Michaud, R. O. (1989): The Markowitz Optimization Enigma: Is 'Optimized' Optimal? *Financial Analysts Journal* 45(1), 31–42. DOI [10.2469/faj.v45.n1.31](https://doi.org/10.2469/faj.v45.n1.31) · 📄 [SSRN 2387669](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2387669)
- Politis, D. N. & Romano, J. P. (1994): The Stationary Bootstrap. *Journal of the American Statistical Association* 89(428), 1303–1313. 🔒 DOI [10.1080/01621459.1994.10476870](https://doi.org/10.1080/01621459.1994.10476870)
- Sharpe, W. F. (1994): The Sharpe Ratio. *Journal of Portfolio Management* 21(1), 49–58. DOI [10.3905/jpm.1994.409501](https://doi.org/10.3905/jpm.1994.409501) · 📄 [Volltext (Stanford)](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)
