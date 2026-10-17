# Limitationen und methodische Einordnung

Dieses Dokument beschreibt die Grenzen der Untersuchung, ordnet sie in die
Literatur ein und beziffert, wo das möglich ist, ihren Einfluss auf die
Ergebnisse. Alle Ergebnisse stammen aus **einem** Lauf mit den Standardwerten in
`portfolio/config.py` (Verzeichnis `output/`, 25.08.2026). Paragrafenzeichen
(§) verweisen auf die Seminararbeit.

| Abschnitt | Thema | Einfluss auf die Ergebnisse |
|---|---|---|
| 1 | Survivorship Bias | absolute Renditen zu hoch, Vergleich weniger betroffen |
| 2 | Schätzfehler in den erwarteten Renditen | Kernproblem der Optimierung, Gegenstand der Arbeit |
| 3 | Statistische Absicherung | Tests nach Ledoit/Wolf, Holm und Deflated Sharpe Ratio |
| 4 | Ein historischer Pfad | Rangfolge nur für 2015–2024 belegt |
| 5 | Informationsleck in der Kreuzvalidierung | durch Purged CV ausgeschlossen |
| 6 | Annualisierung | Jahreswerte als Näherung |
| 7 | Konstanter risikofreier Zins | Schwellen beziffert (§ 6.6) |
| 8 | Transaktionskosten | Sensitivität 0–100 Basispunkte (§ 6.4) |
| 9 | Gleiche Regeln für Markowitz und Random Forest | Vergleich unter sonst gleichen Bedingungen |
| 10 | Rückfall auf das Minimum-Varianz-Portfolio | im Zeitraum nie ausgelöst |
| 11 | Vereinfachungen der Umsetzung | beziffert, keine Aussage ändert sich |
| 12 | Bit-genaue Reproduzierbarkeit | nachgewiesen |

---

## 1. Survivorship Bias

Das Anlageuniversum besteht aus **15 heute existierenden Standardwerten**
(`config.py` → `tickers`), ausgewählt nach Ablauf des Untersuchungszeitraums
(§ 4.1). Unternehmen, die zwischen 2015 und 2024 ausschieden, fusionierten oder
scheiterten, fehlen.

**Folge:** Die absoluten Renditen *aller* Strategien, auch der Gleichgewichtung,
sind nach oben verzerrt. Der Vergleich der Strategien untereinander ist weniger
betroffen, weil alle dasselbe Universum nutzen. Eine zweite Folge benennt § 6.6:
Mit NVIDIA ist ein Titel enthalten, dessen Erfolg bei der Auswahl bereits
feststand; er trägt zum Vorsprung vor der Gleichgewichtung bei Markowitz
6,8 und beim Random Forest 5,2 Prozentpunkte arithmetischer Jahresrendite bei.

**Literatur:** Brown, Goetzmann, Ibbotson & Ross (1992); Elton, Gruber & Blake
(1996).

**Erweiterung:** ein zum jeweiligen Zeitpunkt zusammengesetztes Universum
(etwa die damaligen Mitglieder des S&P 500) einschließlich ausgeschiedener
Titel. Der Backtest rechnet mit jedem Universum (`config.json`, `tickers`) und
legt dafür eine eigene Kursdatei und einen eigenen Ergebnisordner an. Die
Nachrechnungsskripte erklären die Zahlen dieser Arbeit; bei einem anderen
Universum entfallen Abschnitte über bestimmte Titel und Zahlen aus den
Referenzdateien des Laufs der Arbeit mit einem Hinweis.

---

## 2. Schätzfehler in den erwarteten Renditen

Die Markowitz-Optimierung schätzt die erwarteten Renditen als **historische
Mittelwerte**. Erwartungswerte sind schwer zu schätzen, und die
Mean-Variance-Optimierung verstärkt Fehler in diesen Eingaben („Fehler-Maximierer“).
Fehler in den Mittelwerten wiegen empirisch rund zehnmal so schwer wie Fehler in
den Varianzen.

**Literatur:** Merton (1980); Michaud (1989); Best & Grauer (1991);
Chopra & Ziemba (1993).

**Einordnung:** Genau hier setzt der Random Forest an: Er ersetzt den
historischen Mittelwert durch eine Prognose. Dass Random Forest und Markowitz am
Ende nicht signifikant auseinanderliegen, passt zur Literatur:
Renditeprognosen des maschinellen Lernens erreichen typischerweise nur ein sehr
kleines R² außerhalb der Stichprobe (Gu, Kelly & Xiu 2020). Positionsobergrenze
(`max_weight = 0.20`) und Long-only-Bedingung wirken zudem wie eine
Regularisierung und stabilisieren die Gewichte (Jagannathan & Ma 2003).

**Erweiterung:** Shrinkage der Mittelwerte (James-Stein) oder Black-Litterman
(1992) als zusätzliche Vergleichsstrategie.

---

## 3. Statistische Absicherung

Umgesetzt in `portfolio/significance.py`:

- **Ledoit & Wolf (2008):** studentisierter Test für die Differenz zweier
  Sharpe-Quotienten mit HAC-Standardfehler und zirkulärem Block-Bootstrap
  (4999 Ziehungen, Blocklänge 14 Handelstage; zirkulärer Block-Bootstrap nach
  Politis & Romano 1992, zitiert nach Ledoit & Wolf 2008, S. 854).
- **Holm (1979)** über **alle sechs** paarweisen Vergleiche (Motivation:
  Harvey, Liu & Zhu 2016).
- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014), siehe Abschnitt 4.

**Ergebnis:** Nach Holm ist allein „Risk Parity schlechter als die
Gleichgewichtung“ signifikant (p = 0,0216); kein aktives Verfahren schlägt die
Gleichgewichtung signifikant — im Einklang mit DeMiguel, Garlappi & Uppal (2009).

### 3.1 Vier Details einfacher als in der Quelle

`significance.py` folgt dem Aufbau von Ledoit & Wolf (2008), vereinfacht aber vier
Details (§ 5.2):

| | Umsetzung | Ledoit & Wolf (2008) |
|---|---|---|
| p-Wert | k / M | (k + 1) / (M + 1), Gl. (9) |
| Standardfehler im Bootstrap | HAC-Schätzer wie für die Originaldaten | aus Blocksummen (Götze/Künsch), S. 854 |
| Kern des HAC-Schätzers | Bartlett, Bandbreite nach Newey/West (1994) | vorgeweißter QS-Kern (Andrews/Monahan 1992) |
| Blocklänge | Faustregel ⌈T^(1/3)⌉ = 14 | Kalibrierung, Algorithmus 3.1 |

Die Verfahren der rechten Spalte sind zitiert nach Ledoit & Wolf (2008),
S. 852–855; die Originalarbeiten stehen nicht im Literaturverzeichnis unten.

Mit der p-Wert-Formel der Quelle läge der kleinste p-Wert bei 0,0038 statt
0,0036 (§ 5.2). Eine Gegenrechnung mit Blocksummen-Standardfehler **und**
Gl. (9) bei Blocklängen von 7, 14 und 21 Handelstagen führt zu denselben
Testentscheidungen: Allein Risk Parity gegen Gleichgewichtung ist nach Holm
signifikant, alle übrigen Paare liegen deutlich darüber (p ≥ 0,26).
Nicht nachgerechnet ist der QS-Kern; er betrifft nur die zeitversetzten
Kovarianzen, die die Standardfehler um höchstens 6,5 % ändern (§ 5.2).
Nachrechnen: `skripte/kapitel_5_signifikanz.py`.

### 3.2 Holm: „<“ statt „≤“

`holm_bonferroni` verwirft bei angepasstem p-Wert < α, Holm (1979) bei ≤ α.
Bei den angepassten p-Werten der Arbeit (0,0216 und fünfmal 1,000) macht das
keinen Unterschied.

---

## 4. Ein historischer Pfad und Backtest-Overfitting

Alle Strategien werden über **einen** historischen Verlauf verglichen
(2015–2024; acht der zehn Jahre schlossen für das Universum positiv, § 6.6). Ob
die Rangfolge in einem anderen Marktumfeld dieselbe wäre, ist nicht geprüft.

Die **Deflated Sharpe Ratio** in `significance.py` korrigiert den
Sharpe-Quotienten für die Zahl der Versuche und für Schiefe und Wölbung. Sie zählt
N = 4 Versuche, die vier Strategien. Die Entscheidungen, die zu genau diesen
vieren führten, sind darin nicht enthalten (§ 6.6); streng genommen läge N höher
und die Hürde damit ebenfalls.

**Erweiterung:** Robustheit über viele Pfade mit dem stationären Bootstrap
(Politis & Romano 1994), Teilzeiträume (etwa 2020 und 2022), minimale
Backtest-Länge und Wahrscheinlichkeit der Überanpassung (Bailey et al. 2017).

---

## 5. Informationsleck in der Kreuzvalidierung

Die Merkmale stammen aus **überlappenden** Tagesfenstern (Momentum über bis zu
252 Tage, rollierendes Alpha und Beta), und der Zielwert eines Monats ist die
Rendite des Folgemonats. Eine rein chronologische Aufteilung
(`TimeSeriesSplit`) schließt Look-Ahead aus, nicht aber dieses Leck:
Benachbarte Lern- und Prüfabschnitte teilen Information, die
Kreuzvalidierung fällt zu optimistisch aus (§ 4.3).

**Umgesetzt** (`portfolio/cross_validation.py`, `use_purged_cv = true`,
`cv_embargo = 0.02`): **Purged and Embargoed Cross-Validation** nach
López de Prado (2018). Sie entfernt Beobachtungen, deren Zielfenster das
Prüffenster überschneidet, und eine Sperrzone von 2 % der Perioden danach.

**Ausmaß des Lecks** (§ 4.3): Ein sonst identischer Vergleichslauf mit
`use_purged_cv = false` (dieselbe Kursdatei, einkernig; Tagesrenditen in
`skripte/daten/vergleichslauf_ohne_purging.csv`) liefert für den Random Forest einen
Sharpe-Quotienten von 0,961 statt 0,936; rund 0,025 des scheinbar besseren
Ergebnisses beruhten also auf Informationsleckage. Markowitz, Gleichgewichtung und
Risk Parity werden nicht kreuzvalidiert; ihre Ergebnisse bleiben gleich.

---

## 6. Sharpe-Definition und Annualisierung

`metrics.py` und `significance.py` rechnen einheitlich mit dem **arithmetischen**
Mittel der Überschussrenditen nach Sharpe (1994) (Formeln 13 und 14). Die
Hochrechnung auf ein Jahr mit √252 unterstellt **unkorrelierte** Tagesrenditen.
Die Tagesrenditen der vier Strategien haben eine Autokorrelation erster Ordnung
zwischen −0,10 und −0,14 (§ 2.4, § 5.1); die Jahreswerte sind deshalb als
Näherung zu lesen.

**Literatur:** Sharpe (1994); Lo (2002).

---

## 7. Konstanter risikofreier Zins

`risk_free_rate = 0.04` gilt für den ganzen Zeitraum, obwohl die kurzfristigen
US-Zinsen dazwischen von nahe 0 % (2015/16 und 2020/21) bis über 4 % (2023/24)
reichten. § 6.6 beziffert, wie viel
davon abhängt: Unterhalb von 3,4 % tauschen Markowitz und Gleichgewichtung im
Sharpe-Quotienten die Plätze; unterhalb von rund 0,8 % wäre der Rückstand von
Risk Parity nach Holm nicht mehr signifikant. Nachrechnen:
`skripte/kapitel_6_ergebnisse.py`.

**Erweiterung:** die tatsächliche Reihe der dreimonatigen US-Schatzwechsel.

---

## 8. Transaktionskosten

Die pauschalen zehn Basispunkte auf den Umschlag (`transaction_cost = 0.0010`)
enthalten keine Marktbeeinflussung (Almgren & Chriss 2000). Die Kosten gehen
nicht in die Optimierung ein; das benachteiligt die aktiven Verfahren gegenüber
der Gleichgewichtung (§ 4.4).

**Sensitivität** (Abbildung 5, § 6.4, `skripte/abbildung_kostensensitivitaet.py`):
Bei 0 bis 100 Basispunkten fällt Markowitz ab rund 20, der Random Forest ab rund
31 Basispunkten unter die Gleichgewichtung. Weil die Kosten nicht in die
Optimierung eingehen, hängen die Gewichte nicht vom Kostensatz ab; die Kurven
sind eine Umrechnung der gespeicherten Renditen, keine wiederholten Backtests.

---

## 9. Gleiche Regeln für Markowitz und Random Forest

### Der Random Forest maximiert nicht die Rendite

`RFPortfolioOptimizer.optimize()` ruft denselben Löser auf wie die
Markowitz-Strategie: dieselbe Zielfunktion (Sharpe-Maximierung, das Risiko steht
als σ_p im Nenner), dieselbe Kovarianzmatrix nach Ledoit & Wolf (2004) (in
`backtest.py` einmal je Termin geschätzt und an Markowitz, Random Forest und Risk
Parity weitergereicht),
dieselben Nebenbedingungen (long-only, Σw = 1, `max_weight`). Der Random Forest
ersetzt ausschließlich die **Renditeschätzung**. Der Vergleich isoliert damit
genau die Größe, an der die Markowitz-Optimierung laut Literatur scheitert
(Abschnitt 2). Der Random Forest schwankt mit 19,0 % Jahresvolatilität
schwächer als Markowitz (21,0 %).

### Dieselbe Umschlagsschranke

Beide aktiven Strategien dürfen höchstens 30 % des Depots je Monat umschichten
(`rf_turnover_limit`, `mvo_turnover_limit`). Die Schranke im Optimierer bezieht
sich auf dieselben, mit den Kursen gedrifteten Gewichte, gegen die auch der
ausgewiesene Umschlag gemessen wird (`turnover_ref_drifted = true`, Formel 26).
Der gemessene Umschlag überschreitet die Schranke deshalb nie; sein Höchstwert
ist 30,0 %.

Die Schranke gilt für beide gleich, wirkt aber einseitig: Beim Random Forest
greift sie in 101 von 118 Monaten, bei Markowitz ein einziges Mal (§ 4.4). Im
Mittel schichtet der Random Forest 29,0 % des Depots je Monat um, Markowitz
13,1 %.

---

## 10. Rückfall auf das Minimum-Varianz-Portfolio

Die Zielfunktion maximiert (μ_p − r_f) / σ_p. Das ist nur sinnvoll, solange der
**Zähler positiv** ist. Erwartet kein zulässiges Portfolio mehr als den
risikofreien Zins (unter long-only mit Σw = 1 ist μ_p ein gewichteter Mittelwert
der μ_i), wird der Zähler negativ, und dann macht ein **größeres** σ_p den Bruch
weniger negativ: Der Optimierer würde das Risiko maximieren statt es zu
minimieren.

Kontrolle mit den Daten aus `tests/test_optimizers.py` (8 Titel, r_f = 4 %,
Obergrenze 20 %): Erwarten alle Titel −5 %, liefert die ungeschützte
Sharpe-Maximierung ein Portfolio mit 66,6 % Volatilität, das
Minimum-Varianz-Portfolio eines mit 34,3 %.

**Theoretischer Hintergrund (Merton 1972).** Merton leitet den Effizienzrand
geschlossen her und zeigt, dass das Tangentialportfolio nur dann ein
*effizientes* Portfolio ist, wenn der risikofreie Zins unter der erwarteten
Rendite Ē = A/C des Minimum-Varianz-Portfolios liegt (S. 1865, Theorem II:
„… if and only if R < Ē“). Bei r_f = Ē gibt es keinen Berührpunkt, bei r_f > Ē
liegt er auf dem ineffizienten unteren Ast des Rands (S. 1868).

**Zwei Schwellen nicht verwechseln.** Mertons Grenze Ē = A/C gilt **ohne**
Nichtnegativitätsbedingung; er lässt Leerverkäufe ausdrücklich zu (S. 1852,
Fn. 3: „the only constraint on the x_i is that they sum to unity“). Unter
long-only mit Obergrenze 20 % greift die Entartung erst, wenn kein *zulässiges*
Portfolio mehr μ_p > r_f erreicht. Wegen der Obergrenze kann das schon eintreten,
während einzelne Titel noch μ_i > r_f haben, denn das renditestärkste zulässige
Portfolio verteilt sich auf mindestens fünf Titel. Der Code prüft deshalb direkt
die gefundene Lösung: erwartete Überrendite w·μ − r_f ≤ 0. Merton belegt die
**Struktur** des Problems, nicht den Auslöser der Abfrage in `optimizers.py`.

**Umgesetzt** (`portfolio/optimizers.py`, `min_variance_fallback = true`): Der
Fall wird erkannt und immer im Laufprotokoll vermerkt (`Max-Sharpe entartet: …`).
Die Strategie weicht dann auf das **Minimum-Varianz-Portfolio** aus, unter
denselben Nebenbedingungen einschließlich Umschlagsschranke. Verspricht die
Renditeschätzung keine Vergütung für Risiko, bleibt Risikominimierung als einzige
sinnvolle Zielgröße.

**Befund 2015–2024:** Der Fall trat an keinem der 119 Termine ein, bei keiner
der beiden Strategien (Laufprotokoll ohne Eintrag `Max-Sharpe entartet`). Die
Option ist Vorsorge für Zeiträume mit ausgeprägten Baissephasen und ändert an
den Ergebnissen der Arbeit nichts.

---

## 11. Vereinfachungen der Umsetzung

Die folgenden Punkte stecken im Lauf der Arbeit. Keiner von ihnen ändert eine
Aussage der Arbeit; § 4.3 und § 4.4 nennen die ersten beiden.

### 11.1 Feste Gewichte innerhalb des Monats

`backtest.py` (Abschnitt 6b) bildet jede Tagesrendite mit den **Zielgewichten**
des Monats, als würde das Depot täglich kostenlos auf diese Gewichte
zurückgesetzt (§ 4.4). Der Umschlag (Formel 26) wird dagegen gegen die mit den
Kursen gedrifteten Gewichte gemessen, so wie sich ein gekauftes und gehaltenes
Depot entwickelt.

Gegenrechnung mit Kursdrift innerhalb des Monats (gleiche Gewichte, gleicher
Umschlag, gleiche Kosten; `skripte/gegenrechnung_drift.py`): Der
Sharpe-Quotient jeder Strategie sinkt um 0,005 bis 0,007, die CAGR um 0,18 bis
0,19 Prozentpunkte. Rangfolge und alle sechs Testentscheidungen bleiben gleich,
die Deflated Sharpe Ratios sinken um höchstens 0,001.

### 11.2 Lernstoff des Random Forest endet an 50 Terminen einen Monat früher

Damit der Zielwert des laufenden Monats nicht im Lernstoff steht, schneidet
`backtest.py` die Trainingszeilen bei „Entscheidungstag minus ein
Kalendermonat“ ab (§ 4.3). Bei Monaten mit höchstens 30 Tagen (Februar, April,
Juni, September, November) liegt diese Grenze vor dem Ende des Vormonats, etwa
30.04. − 1 Monat = 30.03. < 31.03. Dann fällt auch die Zeile des Vormonats heraus,
obwohl ihr Zielwert schon bekannt wäre. Das betrifft 50 der 119 Termine: Der
Lernstoff endet dort beim vorvorletzten statt beim vorletzten Monat (35 statt 36
Monatszeilen je Titel). Kein Blick in die Zukunft, nur ein Monat weniger
Lernstoff. Nachrechnen: `skripte/kapitel_4_methodik.py`.

### 11.3 Alpha und Beta

`indicators.compute_alpha_beta` (Tabelle A2) berechnet Beta als Kovarianz von
Titel- und Marktrendite geteilt durch die Varianz der Marktrendite über die 63
Handelstage **vor** dem jeweiligen Tag. `np.cov` teilt dabei durch n − 1,
`np.var` durch n; Beta liegt deshalb um den festen Faktor 63/62 (1,6 %) über der
Steigung der Kleinste-Quadrate-Regression.

- **Beta** geht ungerankt in den Random Forest ein. Ein Entscheidungsbaum teilt
  nach der Reihenfolge der Werte, und ein fester positiver Faktor ändert sie
  nicht; für die Prognosen ist der Faktor ohne Belang.
- **Alpha** (Achsenabschnitt, × 252) verschiebt sich dadurch an den
  Monatsenden im Median um 0,27 Prozentpunkte pro Jahr. Die Alphas der Titel
  streuen mit einer Standardabweichung von 37,6 Prozentpunkten.

Wie empfindlich der Random Forest auf so kleine Änderungen der Merkmale
reagiert, zeigt eine Gegenrechnung mit exakter Regressionssteigung und dem Tag
selbst im Fenster: Sein Sharpe-Quotient steigt um rund 0,02; Rangfolge, alle
sechs Testentscheidungen und alle Deflated Sharpe Ratios bleiben gleich. Die
Verschiebung liegt weit unter den Standardfehlern der Sharpe-Differenzen
(0,12 bis 0,18 bei den Vergleichen mit dem Random Forest, § 6.3).

---

## 12. Bit-genaue Reproduzierbarkeit

Drei Quellen können zwei Läufe desselben Codes auseinanderbringen. Alle drei
sind ausgeschaltet.

**Kursdaten.** Yahoo Finance liefert bei jedem Abruf leicht andere Kurse: Zwei
Abrufe desselben Zeitraums im Abstand von 20 Sekunden unterschieden sich in
36 553 von 45 285 Kurswerten, relativ um höchstens 1,3·10⁻⁶. Das trifft alle
Strategien, auch die, die weder kreuzvalidiert noch Zufallszahlen benutzt.
**Umgesetzt** (`price_cache = "./data/prices.pkl"`): Die Kurse des Abrufs vom
15.08.2026 liegen eingefroren im Repository; jeder Lauf liest von dort. Ohne
diese Datei ist der Backtest nicht nachrechenbar, wer die Kurse neu lädt, erhält
andere Werte.

**Parallelverarbeitung.** `RandomForestRegressor(random_state=42)` fixiert den
Zufallsgenerator, nicht aber die Reihenfolge, in der parallele Threads ihre
Teilergebnisse aufsummieren. Gleitkommaaddition ist nicht assoziativ. Liegen in
der Hyperparametersuche (30 Kandidaten) zwei Kandidaten nahezu gleichauf, genügt
eine Abweichung in der zehnten Nachkommastelle für einen anderen Sieger, und dann
unterscheiden sich Baumtiefen statt Rundungsstellen. Parallel gerechnet schwankt
der Sharpe-Quotient des Random Forest deshalb von Lauf zu Lauf um rund 0,01.
**Umgesetzt** (`deterministic = true`): Random Forest und Hyperparametersuche
laufen einkernig (`n_jobs = 1`). Der Lauf wird dadurch um ein Mehrfaches
langsamer, dafür bitgenau wiederholbar.

**Programmumgebung.** Gleitkommaergebnisse hängen auch von den Versionen der
Programmbibliotheken ab. Anhang C2 der Arbeit nennt die Versionen des Laufs.

**Nachweis.** Zwei vollständige, unabhängig gestartete Läufe mit
`deterministic = true` und eingefrorener Kursdatei erzeugen **bitgleiche**
Ergebnisdateien (alle sieben CSV-Dateien, MD5-Vergleich). Die Dateien des ersten
Laufs (15.08.2026) liegen im Commit b8eeea9; `skripte/anhang.py` vergleicht sie
mit denen in `output/` (Lauf vom 25.08.2026).

Der Nachweis gilt für dieselbe Umgebung: Der Lauf vom 15.08.2026 (Python 3.14.6)
und der maßgebliche Lauf vom 25.08.2026 (Python 3.14.7), beide unter macOS 26.6.1,
sind bitgleich. NumPy rechnet seine Matrixoperationen mit Apples Bibliothek
Accelerate, die zum Betriebssystem gehört — die naheliegende Ursache dafür, dass
ein Neulauf unter macOS 27 (15.10.2026) in den letzten Stellen abweicht: Die Gewichte von Markowitz (in einem
Monat) und Random Forest (an 44 Terminen) unterscheiden sich um höchstens
1,7·10⁻³. Von den Zahlen der Arbeit ändern sich dadurch sieben in ihrer letzten
angegebenen Stelle um eins: drei p-Werte und ein t-Wert in der Tabelle in § 6.3
sowie drei Sharpe-Werte in Tabelle B3. Alle Testentscheidungen, Rangfolgen,
Deflated Sharpe Ratios und alle übrigen Zahlen bleiben gleich (Abgleich mit
`skripte/alle_kapitel.py --abgleich` auf den Ergebnisdateien dieses Neulaufs). Bitgleich ist
ein Neulauf nur mit denselben Programmversionen (`requirements.txt`) unter
macOS 26.6.1.

---

## Literaturverzeichnis

**Verfügbarkeit** (Stand der Online-Prüfung Juni 2026):
📄 = kostenfreier Volltext verlinkt · 🔒 = nur kostenpflichtig/Bibliothek (DOI verlinkt).

> Die bibliografischen Angaben (Band, Heft, Seitenbereich, DOI) sind online
> geprüft. Seitengenaue Belege im Fließtext sind am Original zu prüfen; DOI und
> Volltext-Links führen zur jeweiligen Quelle.

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
- Merton, R. C. (1972): An Analytic Derivation of the Efficient Portfolio Frontier. *Journal of Financial and Quantitative Analysis* 7(4), 1851–1872. 🔒 [JSTOR 2329621](https://www.jstor.org/stable/2329621)
- Merton, R. C. (1980): On Estimating the Expected Return on the Market: An Exploratory Investigation. *Journal of Financial Economics* 8(4), 323–361. DOI [10.1016/0304-405X(80)90007-0](https://doi.org/10.1016/0304-405X(80)90007-0) · 📄 [NBER w0444](https://www.nber.org/papers/w0444)
- Michaud, R. O. (1989): The Markowitz Optimization Enigma: Is 'Optimized' Optimal? *Financial Analysts Journal* 45(1), 31–42. DOI [10.2469/faj.v45.n1.31](https://doi.org/10.2469/faj.v45.n1.31) · 📄 [SSRN 2387669](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2387669)
- Politis, D. N. & Romano, J. P. (1994): The Stationary Bootstrap. *Journal of the American Statistical Association* 89(428), 1303–1313. 🔒 DOI [10.1080/01621459.1994.10476870](https://doi.org/10.1080/01621459.1994.10476870)
- Sharpe, W. F. (1994): The Sharpe Ratio. *Journal of Portfolio Management* 21(1), 49–58. DOI [10.3905/jpm.1994.409501](https://doi.org/10.3905/jpm.1994.409501) · 📄 [Volltext (Stanford)](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)
