# Kolloquium — Verteidigungsbausteine zu den Limitationen

Kurze, sprechfertige Antworten auf die Frage *„Warum hast du diese Limitation nicht behoben?"*.
Leitgedanke: Meine Forschungsfrage ist **relativ** (Strategie vs. Strategie auf demselben
Universum). Alles, was alle vier Strategien gleich trifft, verändert die Rangfolge kaum.

Die Limitationen zerfallen in drei Sorten:
- **A – bereits gelöst** (Signifikanztests, Purged CV, Renditen/Turnover): kein offener Mangel, sondern eine Stärke.
- **B – Datengrenze**, die den relativen Vergleich nicht kippt (Survivorship, Zins, Kosten).
- **C – bewusste Designentscheidung**, deren „Behebung" das Experiment selbst untergraben würde (Mittelwerte, einzelner Pfad).

---

## 1. Survivorship Bias  *(Sorte B)*
Mein Universum besteht nur aus heute existierenden Titeln, weil ein survivorship-freies Universum die historischen Index-Zusammensetzungen inklusive ausgeschiedener Firmen erfordert – solche Daten sind kostenpflichtig und über Yahoo Finance nicht frei verfügbar. Entscheidend ist, dass ich die Ergebnisse *relativ* interpretiere: Der Bias hebt die absoluten Renditen aller vier Strategien gleichermaßen an und lässt den Strategie-Vergleich, um den es mir geht, weitgehend unberührt.

## 2. Schätzfehler in den erwarteten Renditen  *(Sorte C)*
Ich verwende bewusst die historischen Mittelwerte als Markowitz-Input, weil meine Arbeit gerade prüft, ob der Random Forest genau diese Schätzung verbessern kann – hätte ich die Mittelwerte vorab korrigiert (z. B. Black-Litterman), hätte ich den Vergleich verwässert, den ich messen will. Dass Markowitz und Random Forest am Ende kaum auseinanderliegen, ist kein Fehler, sondern der von Gu, Kelly & Xiu (2020) vorhergesagte Befund sehr kleiner prädiktiver R².

## 3. Statistische Signifikanz & multiples Testen  *(Sorte A – gelöst)*
Diesen Punkt habe ich nicht offengelassen, sondern gelöst: Ich teste Sharpe-Differenzen mit dem robusten Block-Bootstrap-Verfahren von Ledoit & Wolf (2008) und korrigiere die fünf paarweisen Vergleiche nach Holm (1979). Damit ist der Kernbefund gegen Autokorrelation, dicke Verteilungsränder und multiples Testen abgesichert.

## 4. Backtest-Overfitting & einzelner historischer Pfad  *(Sorte C / teils gelöst)*
Es gibt nur einen realen Marktverlauf 2015–2024 – mehr echte Geschichte kann niemand erzeugen –, deshalb sichere ich mich mit der Deflated Sharpe Ratio (Bailey & López de Prado 2014) ab, die für die Zahl der Versuche und für Nicht-Normalität korrigiert. Eine zusätzliche Robustheitsprüfung über Sub-Perioden oder die Probability of Backtest Overfitting wäre eine sinnvolle Erweiterung, aber die Korrektur eines echten Fehlers ist sie nicht.

## 5. Informationsleck in der Kreuzvalidierung  *(Sorte A – gelöst)*
Ich habe die von López de Prado (2018) empfohlene Purged & Embargoed Cross-Validation implementiert, die überlappende Labels aus dem Training entfernt; im berichteten Lauf nutze ich die – ebenfalls zeitrespektierende – Zeitreihen-Kreuzvalidierung und weise die strengere Variante als zuschaltbare Option aus. Genau so steht es in der Arbeit, Aussage und Rechnung stimmen also überein.

## 6. Sharpe-Definition & Annualisierung  *(dokumentiert)*
Der thesenentscheidende Signifikanztest verwendet die korrekte arithmetische Sharpe-Definition nach Sharpe (1994); die deskriptive Kennzahlentabelle nutzt die CAGR-basierte Variante, was ich offenlege. Die Annualisierung mit √12 setzt zudem i.i.d.-Renditen voraus – auf genau diese Einschränkung weise ich mit Lo (2002) ausdrücklich hin.

## 7. Konstanter risikofreier Zins  *(Sorte B)*
Ich unterstelle einen konstanten Zins von 4 %, weil ein zeitvariabler Satz vor allem die absoluten Sharpe-Niveaus verschiebt. Da alle Strategien denselben Zins verwenden, bleibt die Rangfolge – und damit meine eigentliche Aussage – davon unberührt.

## 8. Transaktionskosten  *(Sorte B)*
Ich modelliere Transaktionskosten als pauschale 10 Basispunkte auf den Umschlag, weil ein realistisches Market-Impact-Modell (Almgren & Chriss 2000) Ordergrößen, Liquidität und Geld-Brief-Spannen bräuchte, die für ein hypothetisches Portfolio nicht vorliegen. Das Modell trifft alle Strategien gleich und benachteiligt insbesondere die umschlagsarme 1/N-Benchmark nicht künstlich.

---

## Bonus: bereits behobene Fehler als Stärke benennen
Falls gefragt wird, was ich methodisch *korrigiert* habe: Ich rechne mit einfachen (arithmetischen) statt logarithmischen Renditen, weil nur dann die Portfoliorendite exakt die gewichtete Summe der Einzelrenditen ist, und ich messe den Turnover *driftbewusst*, sodass auch die Equal-Weight-Benchmark einen realistischen Rebalancing-Umschlag erhält. Das zeigt, dass ich die Methodik nicht nur angewandt, sondern kritisch geprüft habe.
