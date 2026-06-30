# Die Quellen verständlich erklärt
### Begleitdokument zur W-Seminararbeit „Mathematik an der Börse"

Dieses Dokument fasst **jede in der Seminararbeit verwendete Quelle** zusammen und erklärt ihren
Inhalt **so, dass man kein Vorwissen braucht**. Fachbegriffe werden jeweils in Alltagssprache erklärt.

**Wie zuverlässig ist das?** Die Zusammenfassungen beruhen auf den **Originalarbeiten selbst**
(Abstract, Einleitung, Kernergebnisse, die aus den PDFs gelesen wurden). Es werden **keine Zahlen
oder Aussagen erfunden**. Wo eine Quelle nur als eingescanntes Bild ohne maschinenlesbaren Text
vorliegt, ist das vermerkt; die Erklärung stützt sich dann auf die gesicherten Kernaussagen.

**Aufbau pro Quelle:** *In einem Satz* · *Das Problem* · *Was die Arbeit macht und herausfindet* ·
*Schlüsselbegriffe einfach erklärt* · *Warum das für die Seminararbeit zählt*.

Die Reihenfolge ist **thematisch** (nicht alphabetisch), damit die Konzepte aufeinander aufbauen.

---

## Inhaltsübersicht

**Teil A – Die Grundidee: Rendite, Risiko, Diversifikation**
1. Markowitz (1952) – Portfolio Selection
2. Markowitz (1959) – Portfolio Selection (Buch)

**Teil B – Wie misst man Erfolg? Das Sharpe-Maß**
3. Sharpe (1966) – Mutual Fund Performance
4. Sharpe (1994) – The Sharpe Ratio
5. Lo (2002) – The Statistics of Sharpe Ratios

**Teil C – Das Kernproblem: Schätzfehler**
6. Chopra & Ziemba (1993) · 7. Michaud (1989) · 8. Best & Grauer (1991) · 9. Merton (1980)

**Teil D – Gegenmittel: bessere Schätzung und Beschränkungen**
10. Ledoit & Wolf (2004) · 11. Ledoit & Wolf (2017) · 12. Jagannathan & Ma (2003) · 13. Black & Litterman (1992)

**Teil E – Einfache Alternativen als Maßstab**
14. DeMiguel, Garlappi & Uppal (2009) · 15. Qian (2005) · 16. Maillard, Roncalli & Teïletche (2009/2010)

**Teil F – Maschinelles Lernen für Prognosen**
17. Breiman (2001) · 18. Gu, Kelly & Xiu (2020) · 19. Jegadeesh & Titman (1993)

**Teil G – Ist der Unterschied echt? Statistik und Robustheit**
20. Ledoit & Wolf (2008) · 21. Politis & Romano (1994) · 22. Holm (1979) · 23. Harvey, Liu & Zhu (2014/2016) ·
24. Bailey & López de Prado (2014) · 25. Bailey, Borwein, López de Prado & Zhu (2015/2016) · 26. López de Prado (2018)

**Teil H – Fallstricke der Praxis**
27. Almgren & Chriss (2000) · 28. Brown, Goetzmann, Ibbotson & Ross (1992) · 29. Elton, Gruber & Blake (1996)

---

# Teil A – Die Grundidee: Rendite, Risiko, Diversifikation

## 1. Markowitz (1952): „Portfolio Selection"
*The Journal of Finance, Bd. 7, Nr. 1, S. 77–91.*

**In einem Satz:** Markowitz begründet, dass man bei der Geldanlage nicht nur auf die erwartete
Rendite schauen darf, sondern Rendite und Risiko **gemeinsam** betrachten muss – und dass clevere
Streuung (Diversifikation) das Risiko senkt.

**Das Problem:** Vor Markowitz gab es keine saubere Mathematik dafür, wie man Geld auf mehrere
Wertpapiere verteilt. Eine naheliegende Regel wäre: „Kauf einfach das, was am meisten Gewinn
verspricht." Markowitz zeigt, dass diese Regel unsinnig ist, weil sie dazu führen würde, das gesamte
Vermögen auf eine einzige Aktie zu setzen – etwas, das kein vernünftiger Anleger tut, weil es viel zu
riskant ist.

**Was die Arbeit macht und herausfindet:** Markowitz beschreibt jede Anlage durch zwei Zahlen: die
**erwartete Rendite** (der durchschnittlich erwartete Gewinn) und die **Varianz** (wie stark die
Rendite schwankt – das Risiko). Sein Kernbefund: Das Risiko eines ganzen Portfolios hängt nicht nur
davon ab, wie riskant die einzelnen Aktien sind, sondern vor allem davon, **wie sie sich zueinander
bewegen**. Aktien, die immer gleichzeitig steigen und fallen, helfen wenig; Aktien, die sich
unterschiedlich entwickeln, gleichen ihre Schwankungen teilweise aus – das Gesamtrisiko sinkt, ohne
dass man Rendite opfern muss. Dieses „gemeinsame Schwanken" misst er über die **Kovarianz**. Aus
diesen Überlegungen ergibt sich eine Menge bester Mischungen – die **Effizienzlinie**: Für jedes
Risikoniveau gibt es genau ein Portfolio mit der höchstmöglichen erwarteten Rendite.

**Schlüsselbegriffe einfach erklärt:**
- *Rendite:* Gewinn oder Verlust einer Anlage, meist in Prozent.
- *Erwartete Rendite:* der im Durchschnitt erwartete Gewinn.
- *Varianz / Standardabweichung:* Maß dafür, wie stark eine Rendite um ihren Mittelwert schwankt – hier gleichgesetzt mit „Risiko".
- *Diversifikation:* das Verteilen des Geldes auf viele verschiedene Anlagen, um Risiko zu senken.
- *Kovarianz:* misst, ob sich zwei Anlagen tendenziell gleichzeitig in dieselbe Richtung bewegen.
- *Effizienzlinie (efficient frontier):* alle Portfolios, die bei gegebenem Risiko die höchstmögliche erwartete Rendite bieten.

**Warum das für die Seminararbeit zählt:** Dieser Aufsatz ist das **Fundament** der ganzen Arbeit – die
Geburtsstunde der modernen Portfoliotheorie (Markowitz erhielt dafür 1990 den Wirtschaftsnobelpreis).
Er liefert die „Markowitz-Optimierung", die in der Arbeit gegen modernere Verfahren antritt.

## 2. Markowitz (1959): „Portfolio Selection: Efficient Diversification of Investments" (Buch)
*Monografie. Das vorliegende PDF ist der Nachdruck der **Yale University Press (1971)**; der Inhalt
entspricht der Erstausgabe von 1959 (John Wiley & Sons).*

**In einem Satz:** Das Buch baut die kurze Idee von 1952 zu einem vollständigen Lehrwerk aus – mit
allen Details, Beispielen und der dahinterstehenden Entscheidungstheorie.

**Das Problem:** Der Aufsatz von 1952 ist nur rund 15 Seiten lang und skizziert die Idee bloß. Für die
praktische Anwendung und die theoretische Absicherung brauchte es eine ausführliche Darstellung.

**Was die Arbeit macht:** Markowitz erklärt Schritt für Schritt, wie man Erwartungswert und Varianz
eines Portfolios berechnet, wie sich die Effizienzlinie konkret bestimmen lässt und warum ein gutes
Portfolio „ein ausgewogenes Ganzes" ist und nicht bloß „eine lange Liste guter Aktien". Er verbindet
die Methode außerdem mit der **Nutzentheorie** – der Frage, wie ein vernünftiger Anleger rational
zwischen Risiko und Rendite abwägt.

**Schlüsselbegriffe einfach erklärt:**
- *Monografie:* ein eigenständiges Buch zu einem einzigen Thema (anders als ein kurzer Zeitschriftenaufsatz).
- *Nutzentheorie:* mathematische Beschreibung, wie Menschen unter Unsicherheit entscheiden und Risiko bewerten.

**Warum das für die Seminararbeit zählt:** Es ist die vertiefte Quelle hinter der Grundtheorie. In der
Arbeit selbst wird der grundlegende Aufsatz von 1952 herangezogen; das Buch dient als Hintergrund.

---

# Teil B – Wie misst man Erfolg? Das Sharpe-Maß

## 3. Sharpe (1966): „Mutual Fund Performance"
*The Journal of Business, Bd. 39, Nr. 1, S. 119–138.*

**In einem Satz:** Sharpe erfindet eine einfache Kennzahl dafür, wie gut eine Geldanlage ihr Risiko in
Rendite „umsetzt" – den direkten Vorläufer der heute berühmten Sharpe Ratio.

**Das Problem:** Wenn Fonds A 10 % Rendite bringt und Fonds B 8 %, ist A dann besser? Nicht unbedingt –
vielleicht ist A viel riskanter. Man braucht ein Maß, das **Rendite und Risiko zusammen** bewertet,
um Anlagen fair zu vergleichen.

**Was die Arbeit macht:** Sharpe untersucht Investmentfonds und schlägt ein Maß vor, das er
**reward-to-variability ratio** nennt (wörtlich „Ertrag-zu-Schwankung-Verhältnis"): die Mehrrendite
über einer sicheren Anlage, geteilt durch die Schwankung (Volatilität) der Anlage. Anschaulich: „Wie
viel zusätzlichen Gewinn bekomme ich pro Einheit eingegangenen Risikos?" Je höher der Wert, desto
effizienter verwandelt die Anlage Risiko in Ertrag.

**Schlüsselbegriffe einfach erklärt:**
- *Investmentfonds:* ein Topf, in den viele Anleger einzahlen und der das Geld gebündelt in viele Wertpapiere investiert.
- *Volatilität:* wie stark der Wert einer Anlage schwankt (gemessen über die Standardabweichung).
- *Risikofreie / sichere Anlage:* z. B. eine sehr sichere Staatsanleihe; ihre Rendite dient als Vergleichsmaßstab.

**Warum das für die Seminararbeit zählt:** Hier entsteht die Idee der Sharpe Ratio, des **zentralen
Vergleichsmaßes** der Arbeit. In der Arbeit wird die spätere, ausführlichere Fassung von 1994 zitiert;
dies ist die historische Ursprungsquelle.

## 4. Sharpe (1994): „The Sharpe Ratio"
*The Journal of Portfolio Management, Bd. 21, Nr. 1, S. 49–58; zitiert nach dem Online-Nachdruck.*

**In einem Satz:** Sharpe präzisiert und verallgemeinert sein Maß von 1966 und gibt ihm den heute
gebräuchlichen Namen „Sharpe Ratio".

**Das Problem:** Über die Jahre war das Maß unter vielen Namen und in leicht verschiedenen Versionen
verwendet worden. Sharpe räumt auf und liefert eine klare, allgemeine Definition.

**Was die Arbeit macht und herausfindet:** Die Sharpe Ratio ist die **erwartete Differenzrendite**
(Rendite der Anlage minus Rendite einer Vergleichsanlage) geteilt durch deren **Standardabweichung**.
Sharpe unterscheidet zwei Sichtweisen: die **ex-ante**-Version (auf Basis von Erwartungen für die
Zukunft) und die **ex-post**-Version (auf Basis tatsächlich beobachteter, historischer Daten). Er
betont Feinheiten: etwa dass das Maß die Wechselbeziehung einer Anlage zu anderen Anlagen nicht
berücksichtigt und dass man beim Umrechnen auf Jahreswerte vorsichtig sein muss.

**Schlüsselbegriffe einfach erklärt:**
- *Differenzrendite / Überschussrendite:* Rendite über einer Vergleichsgröße (oft der sicheren Anlage).
- *ex ante / ex post:* „vorher/erwartet" gegenüber „nachher/tatsächlich beobachtet".

**Warum das für die Seminararbeit zählt:** Dies ist die in der Arbeit zitierte Definition der Sharpe
Ratio – des Maßes, mit dem alle vier Strategien verglichen werden.

## 5. Lo (2002): „The Statistics of Sharpe Ratios"
*Financial Analysts Journal, Bd. 58, Nr. 4, S. 36–52.*

**In einem Satz:** Lo zeigt, dass die Sharpe Ratio selbst nur ein **geschätzter** Wert mit Unsicherheit
ist – und dass eine verbreitete Umrechnung auf Jahreswerte oft falsch ist.

**Das Problem:** Die Sharpe Ratio wird aus historischen Daten berechnet. Historische Daten sind aber nur
eine Stichprobe – den „wahren" Wert kennt man nie exakt. Trotzdem werden Sharpe Ratios oft auf die
Nachkommastelle verglichen, als wären sie genaue Größen.

**Was die Arbeit macht und herausfindet:** Lo leitet mathematisch her, wie genau (oder ungenau) eine
geschätzte Sharpe Ratio ist – also wie groß ihr **Schätzfehler** ausfällt. Zwei wichtige Befunde:
(1) Größere Sharpe Ratios sind tendenziell unpräziser geschätzt. (2) Die übliche Faustregel, eine
monatliche Sharpe Ratio durch Multiplikation mit der Wurzel aus 12 auf ein Jahr „hochzurechnen",
stimmt nur, wenn die Renditen voneinander unabhängig sind. Hängen Renditen zeitlich zusammen (z. B.
Trends), kann diese naive Umrechnung den Wert stark verfälschen – in seinem Beispiel wird die
Jahres-Sharpe-Ratio eines Hedgefonds dadurch um **bis zu 65 %** zu hoch ausgewiesen.

**Schlüsselbegriffe einfach erklärt:**
- *Schätzfehler / Standardfehler:* wie weit ein aus Daten berechneter Wert vom „wahren" Wert abweichen kann.
- *Annualisieren:* einen Wert (z. B. von Monats- auf Jahresbasis) umrechnen.
- *i.i.d. (unabhängig und identisch verteilt):* die Annahme, dass aufeinanderfolgende Renditen nichts miteinander zu tun haben (wie unabhängige Würfelwürfe).
- *Serielle Korrelation / Autokorrelation:* wenn der heutige Wert mit dem gestrigen zusammenhängt.

**Warum das für die Seminararbeit zählt:** Lo begründet, warum man Sharpe Ratios nicht naiv vergleichen
darf, sondern ihre Unsicherheit beachten muss. Das ist die Brücke zu den **statistischen
Signifikanztests** im Hauptteil der Arbeit.

---

# Teil C – Das Kernproblem: Schätzfehler

## 6. Chopra & Ziemba (1993): „The Effect of Errors in Means, Variances, and Covariances on Optimal Portfolio Choice"
*The Journal of Portfolio Management, Bd. 19, Nr. 2, S. 6–11.*

**In einem Satz:** Die Autoren messen, welcher der drei Bausteine der Optimierung – erwartete Renditen,
Schwankungen oder Zusammenhänge – am empfindlichsten auf Schätzfehler reagiert. Die Antwort: die
erwarteten Renditen, mit großem Abstand.

**Das Problem:** Markowitz' Optimierung braucht drei Zutaten: die erwarteten Renditen, die Varianzen
(Schwankungen) und die Kovarianzen (Zusammenhänge) der Aktien. Diese Zutaten kennt niemand exakt – man
muss sie aus Daten **schätzen**, und Schätzungen sind fehlerbehaftet. Die Frage ist: Wo tut ein Fehler
am meisten weh?

**Was die Arbeit macht und herausfindet:** Chopra und Ziemba bauen gezielt Fehler in die Zutaten ein
und messen, wie viel „Nutzen" der Anleger dadurch verliert – über eine Kennzahl namens **Cash
Equivalent Loss** (CEL, grob: „Wie viel Geld ist mir der Fehler wert?"). Ergebnis: Fehler in den
**erwarteten Renditen** schaden weit am meisten. Konkret, bei mittlerer Risikobereitschaft, ist ein
Fehler in den Mittelwerten rund **elfmal** so schädlich wie derselbe Fehler in den Varianzen und etwa
**22-mal** so schädlich wie in den Kovarianzen. Daraus folgt eine klare Empfehlung: Wer begrenzte Zeit
und Mittel hat, sollte sie vor allem in **gute Renditeschätzungen** stecken.

**Schlüsselbegriffe einfach erklärt:**
- *Schätzen:* einen unbekannten Wert aus Daten näherungsweise bestimmen.
- *Cash Equivalent Loss (CEL):* der in Geld ausgedrückte Nutzenverlust, wenn man statt des perfekten ein fehlerbehaftetes Portfolio hält.
- *Risikobereitschaft / Risikotoleranz:* wie viel Schwankung ein Anleger für mehr Rendite zu akzeptieren bereit ist.

**Warum das für die Seminararbeit zählt:** Dieser Befund ist der **rote Faden**: Weil die erwarteten
Renditen die heikelste Zutat sind, lohnt es sich, gerade hier anzusetzen – was genau die Aufgabe des
Random Forest in der Arbeit ist (er soll bessere Renditeprognosen liefern).

## 7. Michaud (1989): „The Markowitz Optimization Enigma: Is ‚Optimized' Optimal?"
*Financial Analysts Journal, Bd. 45, Nr. 1, S. 31–42. (Eingescannt; per Bild gelesen.)*

**In einem Satz:** Michaud erklärt das „Rätsel", warum die theoretisch überzeugende
Markowitz-Optimierung in der Praxis so selten genutzt wird – und nennt sie sinngemäß einen
„Fehler-Maximierer".

**Das Problem:** Die Optimierung ist mathematisch elegant, aber viele Praktiker misstrauen ihr. Warum?

**Was die Arbeit macht und herausfindet:** Michaud benennt das Kernproblem klar: Die MV-Optimierung
neigt dazu, **die Wirkung von Fehlern in den Eingangsdaten zu vergrößern** statt zu dämpfen. Sie
schaufelt das Geld bevorzugt in genau die Wertpapiere, deren Rendite zufällig zu hoch und deren Risiko
zufällig zu niedrig geschätzt wurde – also dorthin, wo die Schätzung am meisten danebenliegt. Die Folge:
Eine **uneingeschränkte** Optimierung kann sogar schlechter abschneiden als eine simple
Gleichverteilung. Michaud betont aber auch, dass sich der praktische Wert der Methode durch sinnvolle
**Beschränkungen** (z. B. keine Leerverkäufe, Obergrenzen) deutlich verbessern lässt.

**Schlüsselbegriffe einfach erklärt:**
- *Eingangsdaten / Inputs:* die geschätzten Zutaten der Optimierung (Renditen, Risiken, Zusammenhänge).
- *Fehler-Maximierer:* anschauliche Bezeichnung dafür, dass das Verfahren Schätzfehler verstärkt statt ausgleicht.
- *Beschränkungen (Constraints):* Regeln, die dem Optimierer Grenzen setzen (z. B. „kein Wertpapier über 20 %").
- *Leerverkauf (Short):* auf fallende Kurse setzen / ein Wertpapier mit negativem Anteil halten.

**Warum das für die Seminararbeit zählt:** Michaud liefert die berühmte Kritik, die erklärt, warum die
Optimierung instabil ist – und begründet zwei Designentscheidungen der Arbeit: das Verbot von
Leerverkäufen und eine Obergrenze je Titel.

## 8. Best & Grauer (1991): „On the Sensitivity of Mean-Variance-Efficient Portfolios to Changes in Asset Means"
*The Review of Financial Studies, Bd. 4, Nr. 2, S. 315–342.*

**In einem Satz:** Die Autoren zeigen mathematisch und in Berechnungen, wie extrem empfindlich
„optimale" Portfolios auf kleinste Änderungen der geschätzten Renditen reagieren.

**Das Problem:** Wenn man die geschätzte erwartete Rendite einer einzigen Aktie nur leicht verändert –
ändert sich das „optimale" Portfolio dann auch nur leicht? Oder kippt es komplett?

**Was die Arbeit macht und herausfindet:** Best und Grauer untersuchen genau diese Frage analytisch
(mit Formeln) und numerisch (mit Beispielrechnungen). Ergebnis: Schon **winzige** Änderungen in den
geschätzten Mittelwerten können die optimalen Portfoliogewichte **drastisch** verschieben – einzelne
Wertpapiere können von „stark gewichtet" zu „gar nicht enthalten" springen. Die scheinbar präzise
„optimale" Lösung steht also auf wackligem Boden.

**Schlüsselbegriffe einfach erklärt:**
- *Portfoliogewichte:* die Anteile, mit denen die einzelnen Wertpapiere im Portfolio vertreten sind.
- *analytisch / numerisch:* „mit allgemeingültigen Formeln" gegenüber „mit konkreten Beispielrechnungen am Computer".
- *Sensitivität / Empfindlichkeit:* wie stark sich ein Ergebnis ändert, wenn man eine Eingangsgröße leicht verändert.

**Warum das für die Seminararbeit zählt:** Best und Grauer liefern den **strengen Beleg** für das, was
Chopra/Ziemba und Michaud beschreiben: Die Optimierung ist überempfindlich gegenüber den (unsicheren)
Renditeschätzungen. Das untermauert, warum einfache Strategien außerhalb der Stichprobe so schwer zu
schlagen sind.

## 9. Merton (1980): „On Estimating the Expected Return on the Market: An Exploratory Investigation"
*NBER Working Paper Nr. 444 (1980); später im Journal of Financial Economics erschienen.*

**In einem Satz:** Merton zeigt eine grundlegende Asymmetrie: Das Risiko (die Schwankung) einer Anlage
lässt sich gut schätzen – die erwartete Rendite dagegen kaum, selbst mit sehr vielen Daten.

**Das Problem:** Für fast jede Finanzentscheidung braucht man die erwartete Rendite des Marktes. Üblich
ist, einfach den historischen Durchschnitt zu nehmen. Aber wie verlässlich ist dieser Wert?

**Was die Arbeit macht und herausfindet:** Merton untersucht mathematisch, wie genau sich erwartete
Rendite und Schwankung aus Daten schätzen lassen. Sein zentraler Befund: Die **Schwankung
(Varianz/Volatilität)** lässt sich sehr präzise bestimmen, wenn man nur oft genug misst (z. B. täglich
statt jährlich) – mehr Messpunkte innerhalb desselben Zeitraums helfen. Die **erwartete Rendite**
hingegen lässt sich so **nicht** verbessern: Hier zählt allein, wie **lang** der beobachtete Zeitraum
insgesamt ist – und selbst über Jahrzehnte bleibt die Schätzung sehr ungenau.

**Schlüsselbegriffe einfach erklärt:**
- *Erwartete Rendite des Marktes:* der durchschnittlich erwartete Gewinn des Gesamtmarkts (z. B. eines breiten Aktienindex).
- *Höher frequente Daten:* häufigere Messungen (z. B. Tages- statt Jahresdaten).
- *Asymmetrie:* hier, dass sich die eine Größe gut, die andere schlecht schätzen lässt.

**Warum das für die Seminararbeit zählt:** Merton erklärt die **tiefere Ursache** für alles bisher
Gesagte: Gerade die wichtigste Zutat der Optimierung – die erwartete Rendite – ist die am schwersten zu
schätzende. Das ist der Grund, warum Renditeprognosen so schwierig sind und warum auch ein Random Forest
hier an Grenzen stößt.

---

# Teil D – Gegenmittel: bessere Schätzung und Beschränkungen

## 10. Ledoit & Wolf (2004): „A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices"
*Journal of Multivariate Analysis, Bd. 88, Nr. 2, S. 365–411.*

**In einem Satz:** Die Autoren liefern eine stabilere Methode, die Zusammenhänge zwischen vielen Aktien
zu schätzen (die „Kovarianzmatrix") – durch eine clevere Mischung aus Beobachtung und einer einfachen
Standardannahme.

**Das Problem:** Die Optimierung braucht die **Kovarianzmatrix** – eine große Tabelle, die für jedes
Aktienpaar angibt, wie stark sich die beiden gemeinsam bewegen. Schätzt man diese Tabelle einfach aus
historischen Daten, wird sie bei vielen Aktien instabil („schlecht konditioniert"): Kleine Datenfehler
werden beim Weiterrechnen (genauer: beim „Invertieren", das die Optimierung benötigt) stark aufgebläht.
Hat man mehr Aktien als Beobachtungstage, lässt sie sich sogar gar nicht mehr verwenden.

**Was die Arbeit macht und herausfindet:** Ledoit und Wolf schlagen einen **Shrinkage-Schätzer** vor
(„Schrumpfung"). Die Idee: Man mischt die aus Daten geschätzte (unzuverlässige) Tabelle mit einer
einfachen, stabilen „Standardtabelle" – ein gewichtetes Mittel aus beidem. Dadurch werden extreme, vom
Zufall getriebene Werte zur Mitte hin gezogen. Sie berechnen sogar die **optimale** Mischung über eine
einfache Formel. Das Ergebnis ist stabil, asymptotisch genauer als die reine Datenschätzung und kommt
ohne Annahmen über die Verteilung der Renditen aus.

**Schlüsselbegriffe einfach erklärt:**
- *Kovarianzmatrix:* Tabelle, die für alle Aktienpaare das gemeinsame Schwanken angibt.
- *Schlecht konditioniert:* mathematisch instabil – kleine Eingabefehler führen zu großen Ergebnisfehlern.
- *Invertieren:* eine Rechenoperation mit Matrizen (vergleichbar dem „Teilen" bei Zahlen), die die Optimierung benötigt.
- *Shrinkage (Schrumpfung):* extreme geschätzte Werte in Richtung eines stabilen Mittelwerts ziehen, um Zufallsausschläge zu dämpfen.

**Warum das für die Seminararbeit zählt:** Dieser Schätzer ist die **Kovarianz-Grundlage aller
risikobasierten Strategien** in der Arbeit (Markowitz-Optimierung und Risk Parity). Er macht die
Optimierung deutlich robuster.

## 11. Ledoit & Wolf (2017): „Nonlinear Shrinkage of the Covariance Matrix … Markowitz Meets Goldilocks"
*The Review of Financial Studies, Bd. 30, Nr. 12, S. 4349–4388.*

**In einem Satz:** Eine verfeinerte Version der Shrinkage-Idee von 2004, die noch flexibler und
treffsicherer ist – „nicht zu viel, nicht zu wenig" Schrumpfung (daher „Goldilocks").

**Das Problem:** Die lineare Shrinkage von 2004 schrumpft alle Teile der Kovarianzmatrix nach einer
einzigen, gleichmäßigen Regel. Manchmal ist das zu grob – verschiedene Teile bräuchten unterschiedlich
starke Korrektur.

**Was die Arbeit macht:** Ledoit und Wolf entwickeln eine **nichtlineare** Shrinkage, die die einzelnen
„Bausteine" der Matrix (fachlich: die Eigenwerte) **individuell** und passgenau korrigiert – jeden gerade
so stark, wie es nötig ist. Der Titelzusatz „Markowitz Meets Goldilocks" spielt darauf an: Wie im Märchen
ist die Korrektur „genau richtig" dosiert.

**Schlüsselbegriffe einfach erklärt:**
- *linear / nichtlinear:* „nach einer einzigen festen Regel" gegenüber „flexibel, je nach Situation unterschiedlich".
- *Eigenwerte:* mathematische Kennzahlen einer Matrix; vereinfacht die „Grundbausteine" des Risikos, die hier einzeln justiert werden.

**Warum das für die Seminararbeit zählt:** Dieses Verfahren wird in der Arbeit nicht direkt verwendet,
sondern im **Ausblick** als mögliche Weiterentwicklung der Kovarianzschätzung genannt. Es zeigt, dass die
Forschung in diese Richtung weitergeht.

## 12. Jagannathan & Ma (2003): „Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps"
*The Journal of Finance, Bd. 58, Nr. 4, S. 1651–1683.*

**In einem Satz:** Die Autoren lösen ein überraschendes Rätsel: Warum verbessern „falsche" Regeln (wie
ein Verbot von Leerverkäufen) die Portfolio-Ergebnisse, obwohl sie die Optimierung doch einschränken?

**Das Problem:** Eigentlich müsste jede zusätzliche Einschränkung die „optimale" Lösung verschlechtern –
schließlich nimmt man dem Optimierer Freiheit. In der Praxis ist aber oft das **Gegenteil** zu
beobachten: Mit einem Leerverkaufsverbot werden Portfolios besser. Warum?

**Was die Arbeit macht und herausfindet:** Jagannathan und Ma zeigen mathematisch, dass ein
**Leerverkaufsverbot** dasselbe bewirkt wie eine **Shrinkage** der Kovarianzmatrix: Es verkleinert genau
die übergroßen, vom Zufall aufgeblähten Werte, die sonst zu extremen Wetten führen würden. Die „falsche"
Einschränkung dämpft also Schätzfehler – und senkt damit das Risiko. Noch überraschender: Mit dieser
Einschränkung schneidet sogar die **einfache** Kovarianzschätzung so gut ab wie viel aufwändigere
Verfahren.

**Schlüsselbegriffe einfach erklärt:**
- *Leerverkaufsverbot (Long-only):* die Regel, dass man Wertpapiere nur kaufen, aber nicht „short" verkaufen darf (keine negativen Anteile).
- *Shrinkage:* siehe Quelle 10 – das Dämpfen extremer geschätzter Werte.

**Warum das für die Seminararbeit zählt:** Das ist die direkte Begründung für zwei Designentscheidungen
der Arbeit: das **Long-only-Verbot** und die **Obergrenze von 20 %** je Titel. Sie machen die Optimierung
stabiler – nicht trotz, sondern wegen der Einschränkung.

## 13. Black & Litterman (1992): „Global Portfolio Optimization"
*Financial Analysts Journal, Bd. 48, Nr. 5, S. 28–43.*

**In einem Satz:** Black und Litterman entwerfen ein Verfahren, das die unzuverlässigen
Renditeschätzungen stabilisiert, indem es eine „neutrale" Marktmeinung mit den eigenen Einschätzungen
des Anlegers kombiniert.

**Das Problem:** Die normale Optimierung verlangt, dass der Anleger für **jede** Anlage eine erwartete
Rendite angibt. Das ist unrealistisch – die meisten Anleger haben nur zu wenigen Märkten eine fundierte
Meinung. Und kleine Fehler in diesen Zahlen führen (siehe Quellen 6–8) zu extremen, unsinnigen
Portfolios.

**Was die Arbeit macht und herausfindet:** Black und Litterman drehen den Spieß um. Statt selbst alle
Renditen zu raten, starten sie mit einer **neutralen Ausgangslage**: jenen Renditen, die das aktuelle
Marktgleichgewicht (die tatsächlichen Marktgewichte) „erklären" würden. Auf diese neutrale Basis legt der
Anleger dann nur seine **eigenen, gezielten Meinungen** („Ich glaube, Markt A schlägt Markt B") –
gewichtet danach, wie sicher er sich ist. Heraus kommen stabile, vernünftige Renditeschätzungen und damit
ausgewogenere Portfolios.

**Schlüsselbegriffe einfach erklärt:**
- *Marktgleichgewicht / Gleichgewichtsrenditen:* die Renditen, die zu den aktuell am Markt beobachteten Gewichtungen passen (eine „neutrale" Annahme).
- *Views (Meinungen):* die eigenen Einschätzungen des Anlegers zu einzelnen Märkten.

**Warum das für die Seminararbeit zählt:** Black-Litterman ist eines der bekanntesten **Gegenmittel**
gegen das Schätzproblem. In der Arbeit wird es als solches genannt und im Ausblick als mögliche
Erweiterung vorgeschlagen.

---

# Teil E – Einfache Alternativen als Maßstab

## 14. DeMiguel, Garlappi & Uppal (2009): „Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy?"
*The Review of Financial Studies, Bd. 22, Nr. 5, S. 1915–1953.*

**In einem Satz:** Die Autoren testen, ob die ausgeklügelten Optimierungsverfahren die kinderleichte
Regel „verteile das Geld gleichmäßig auf alle Anlagen" tatsächlich schlagen – und finden: meistens nicht.

**Das Problem:** Lohnt sich der ganze Aufwand der Optimierung überhaupt? Oder ist die simpelste denkbare
Regel – jedem der N Wertpapiere denselben Anteil 1/N geben – in der Praxis genauso gut? (Schon der
Talmud empfiehlt sinngemäß, das Vermögen zu dritteln.)

**Was die Arbeit macht und herausfindet:** Sie vergleichen **14** verschiedene Optimierungsmodelle
(Markowitz und viele Verbesserungen) über **sieben** echte Datensätze – und zwar fair „out-of-sample",
also auf Daten, die das Modell beim Rechnen noch nicht kannte. Ergebnis: **Kein einziges** Modell schlägt
die naive 1/N-Regel verlässlich – weder bei der Sharpe Ratio noch bei anderen Maßen. Der Grund ist der
bekannte Schätzfehler: Die Optimierung sieht in der Vergangenheit gut aus, doch ihre Vorteile verpuffen
in der Zukunft. Wie groß das Problem ist, zeigt eine Rechnung: Damit sich die Optimierung wirklich lohnt,
bräuchte man – je nach Anzahl der Anlagen – etwa **3 000 bis 6 000 Monate** an Daten (also 250 bis 500
Jahre!).

**Schlüsselbegriffe einfach erklärt:**
- *1/N-Regel / naive Diversifikation:* jedem Wertpapier denselben Anteil geben (bei 10 Aktien also je 10 %).
- *out-of-sample (außerhalb der Stichprobe):* auf Daten testen, die das Modell beim Lernen nicht gesehen hat – der ehrliche Test.
- *Benchmark:* eine Vergleichsgröße, an der sich andere Strategien messen lassen müssen.

**Warum das für die Seminararbeit zählt:** Dies ist die **Schlüsselquelle** und liefert die zentrale
Benchmark der Arbeit (Equal Weight). Der Leitbefund der Seminararbeit – dass keine aktive Strategie 1/N
statistisch signifikant schlägt – ist genau die Bestätigung dieses berühmten Ergebnisses an einem
eigenen Datensatz.

## 15. Qian (2005): „Risk Parity Portfolios: Efficient Portfolios Through True Diversification"
*Forschungspapier, PanAgora Asset Management (nicht in einer Fachzeitschrift begutachtet).*

**In einem Satz:** Qian stellt eine Strategie vor, die nicht das Geld, sondern das **Risiko** gleichmäßig
verteilt – „Risk Parity".

**Das Problem:** Verteilt man das Geld gleichmäßig (1/N), heißt das nicht, dass auch das Risiko gleich
verteilt ist. Steckt z. B. die Hälfte des Geldes in sehr schwankungsanfälligen Aktien, dominieren diese
das Gesamtrisiko – die „Diversifikation" ist dann nur eine Illusion.

**Was die Arbeit macht:** Qian schlägt vor, die Anteile so zu wählen, dass **jede Anlage gleich viel zum
Gesamtrisiko beiträgt**. Schwankungsanfällige Anlagen bekommen dadurch automatisch ein kleineres Gewicht,
ruhige Anlagen ein größeres. So entsteht eine „echte" Diversifikation, bei der kein einzelner Baustein
das Risiko bestimmt.

**Schlüsselbegriffe einfach erklärt:**
- *Risk Parity (Risikoparität):* Anteile so wählen, dass alle Anlagen denselben Risikobeitrag leisten.
- *Risikobeitrag:* der Anteil einer einzelnen Anlage am gesamten Portfoliorisiko.

**Warum das für die Seminararbeit zählt:** Risk Parity ist eine der vier verglichenen Strategien. Wie die
1/N-Regel kommt sie ohne die heikle Renditeschätzung aus – nutzt aber im Gegensatz zu 1/N die
Risikoinformationen der Anlagen.

## 16. Maillard, Roncalli & Teïletche (2009/2010): „On the Properties of Equally-Weighted Risk Contributions Portfolios"
*Arbeitspapier 2009; veröffentlicht im Journal of Portfolio Management, Bd. 36, Nr. 4 (2010), S. 60–70.*

**In einem Satz:** Diese Arbeit liefert die saubere Mathematik hinter Qians Risk-Parity-Idee – das
„Equal Risk Contribution"-Portfolio (ERC).

**Das Problem:** Qians Idee „jede Anlage trägt gleich viel Risiko bei" klingt einfach, aber wie berechnet
man die Gewichte dafür genau? Und welche Eigenschaften hat das Ergebnis?

**Was die Arbeit macht und herausfindet:** Maillard, Roncalli und Teïletche definieren präzise, was der
„Risikobeitrag" einer Anlage ist, und zeigen mithilfe eines mathematischen Satzes (dem Satz von Euler),
dass sich das Gesamtrisiko **vollständig** und ohne Rest in die Beiträge der einzelnen Anlagen zerlegen
lässt. Das **ERC-Portfolio** ist dann jenes, in dem alle diese Beiträge gleich groß sind. Sie zeigen
außerdem: Sein Risiko liegt stets **zwischen** dem des risikoärmsten Portfolios (Minimum-Varianz) und dem
der naiven 1/N-Regel – ein guter Kompromiss.

**Schlüsselbegriffe einfach erklärt:**
- *Equal Risk Contribution (ERC):* das Portfolio, in dem jede Anlage exakt denselben Beitrag zum Gesamtrisiko leistet (die formale Umsetzung von Risk Parity).
- *Satz von Euler:* ein mathematischer Satz, mit dem sich das Gesamtrisiko sauber in Einzelbeiträge aufteilen lässt.
- *Minimum-Varianz-Portfolio:* das Portfolio mit dem kleinstmöglichen Risiko.

**Warum das für die Seminararbeit zählt:** Diese Quelle liefert die **Formeln**, mit denen die
Risk-Parity-Strategie in der Arbeit tatsächlich berechnet wird.

---

# Teil F – Maschinelles Lernen für Prognosen

## 17. Breiman (2001): „Random Forests"
*Machine Learning, Bd. 45, Nr. 1, S. 5–32.*

**In einem Satz:** Breiman erfindet den „Random Forest" – ein Verfahren des maschinellen Lernens, das
viele einfache Entscheidungsbäume kombiniert und dadurch erstaunlich treffsichere und stabile
Vorhersagen liefert.

**Das Problem:** Ein einzelner **Entscheidungsbaum** (eine Folge von Ja/Nein-Fragen, die zu einer
Vorhersage führt) ist anschaulich, aber unzuverlässig: Schon kleine Änderungen in den Daten können einen
ganz anderen Baum ergeben, und einzelne Bäume „lernen" oft das zufällige Rauschen der Trainingsdaten mit
(Überanpassung).

**Was die Arbeit macht und herausfindet:** Breimans Idee: Statt eines Baumes baut man **sehr viele** Bäume
und lässt sie abstimmen (bei Zahlen bildet man den Durchschnitt). Damit die Bäume sich unterscheiden,
bekommt jeder (a) eine zufällige Stichprobe der Daten und (b) an jeder Verzweigung nur eine zufällige
Auswahl der Merkmale zur Auswahl. Breiman beweist zwei wichtige Dinge: (1) Mehr Bäume führen **nie** zu
Überanpassung – der Fehler nähert sich einem festen Grenzwert. (2) Der Wald ist umso besser, je
**treffsicherer** die einzelnen Bäume und je **unähnlicher** (unkorrelierter) sie zueinander sind. Genau
dafür sorgt die eingebaute Zufälligkeit.

**Schlüsselbegriffe einfach erklärt:**
- *Maschinelles Lernen:* Computerverfahren, die aus Beispieldaten Muster lernen, um neue Fälle vorherzusagen.
- *Entscheidungsbaum:* ein Modell, das durch eine Folge von Ja/Nein-Fragen zu einer Vorhersage kommt.
- *Random Forest (Zufallswald):* viele zufällig leicht verschiedene Bäume, deren Vorhersagen gemittelt werden.
- *Überanpassung (Overfitting):* wenn ein Modell die Trainingsdaten samt Zufallsrauschen „auswendig lernt" und neue Fälle dann schlecht vorhersagt.

**Warum das für die Seminararbeit zählt:** Der Random Forest ist das **KI-Verfahren** der Arbeit. Er soll
die schwierigste Zutat – die erwartete Rendite – aus vielen Merkmalen prognostizieren.

## 18. Gu, Kelly & Xiu (2020): „Empirical Asset Pricing via Machine Learning"
*The Review of Financial Studies, Bd. 33, Nr. 5, S. 2223–2273.*

**In einem Satz:** Eine große Vergleichsstudie, die testet, wie gut sich Aktienrenditen mit modernen
Verfahren des maschinellen Lernens vorhersagen lassen – mit ernüchternd kleinen, aber doch wertvollen
Ergebnissen.

**Das Problem:** Taugen Methoden des maschinellen Lernens überhaupt, um die notorisch schwer
prognostizierbaren Aktienrenditen vorherzusagen? Und welche Methode ist die beste?

**Was die Arbeit macht und herausfindet:** Gu, Kelly und Xiu vergleichen zahlreiche ML-Verfahren auf
riesigen US-Aktiendaten. Ergebnis: **Baum- und netzbasierte** Verfahren (darunter der Random Forest)
schneiden am besten ab, weil sie **nichtlineare Wechselwirkungen** zwischen Merkmalen erkennen, die
einfache Methoden übersehen. Aber: Die Prognosegüte ist **winzig** – das monatliche „Bestimmtheitsmaß"
(R², wie viel der Schwankung erklärt wird) liegt für einzelne Aktien nur zwischen 0,33 % und 0,40 %. Das
spiegelt das „notorisch niedrige Signal-Rausch-Verhältnis" der Renditeprognose wider. Trotz dieser
winzigen statistischen Erklärungskraft können die Prognosen wirtschaftlich nützlich sein. Die
informativsten Signale sind Varianten von **Momentum, Liquidität und Volatilität**.

**Schlüsselbegriffe einfach erklärt:**
- *Bestimmtheitsmaß (R²):* Anteil der Schwankung, den ein Modell erklärt; 100 % = perfekt, 0 % = nichts erklärt.
- *Signal-Rausch-Verhältnis:* wie viel echte Information (Signal) im Vergleich zum Zufall (Rauschen) in den Daten steckt; bei Renditen sehr ungünstig.
- *nichtlineare Wechselwirkung:* wenn zwei Merkmale nur in Kombination (nicht einzeln) etwas aussagen.
- *Liquidität:* wie leicht sich ein Wertpapier ohne große Kursbewegung handeln lässt.

**Warum das für die Seminararbeit zählt:** Diese Studie belegt beides – dass ML zur Renditeprognose
**taugt**, aber nur **sehr begrenzt**. Sie rechtfertigt den Einsatz des Random Forest und die Auswahl der
Merkmale (Momentum, Volatilität), dämpft aber zugleich die Erwartungen.

## 19. Jegadeesh & Titman (1993): „Returns to Buying Winners and Selling Losers"
*The Journal of Finance, Bd. 48, Nr. 1, S. 65–91. (Eingescannt; per Bild gelesen.)*

**In einem Satz:** Die Autoren entdecken den „Momentum-Effekt": Aktien, die zuletzt gut liefen, laufen
tendenziell weiter gut – und umgekehrt.

**Das Problem:** Nach der gängigen Theorie (Markteffizienz) sollten vergangene Kurse nichts über die
Zukunft verraten. Stimmt das?

**Was die Arbeit macht und herausfindet:** Jegadeesh und Titman zeigen, dass Strategien, die vergangene
**„Gewinner" kaufen** und vergangene **„Verlierer" verkaufen**, über Halteperioden von **drei bis zwölf
Monaten** deutlich positive Überschussrenditen erzielen. Dieser Effekt lässt sich nicht einfach durch
höheres Risiko erklären. (Ein Teil des Vorsprungs verschwindet allerdings in den darauffolgenden zwei
Jahren wieder.) Das ist ein Beleg gegen die strenge Form der Markteffizienz und einer der robustesten
Befunde der Finanzforschung.

**Schlüsselbegriffe einfach erklärt:**
- *Momentum:* die Tendenz, dass sich eine jüngste Kursentwicklung kurzfristig fortsetzt.
- *Markteffizienz:* die Theorie, dass in Kursen bereits alle Informationen stecken und man mit vergangenen Kursen keinen Vorteil erzielen kann.
- *Halteperiode:* der Zeitraum, über den man die Aktien nach dem Kauf behält.

**Warum das für die Seminararbeit zählt:** Der Momentum-Effekt ist die **wissenschaftliche Grundlage**
dafür, dass die Arbeit dem Random Forest momentumbasierte Merkmale (Kursentwicklung über 1, 3, 6 und 12
Monate) als Eingangsdaten gibt.

---

# Teil G – Ist der Unterschied echt? Statistik und Robustheit

## 20. Ledoit & Wolf (2008): „Robust Performance Hypothesis Testing with the Sharpe Ratio"
*Journal of Empirical Finance, Bd. 15, Nr. 5, S. 850–859.*

**In einem Satz:** Die Autoren liefern einen statistischen Test, der zuverlässig prüft, ob der
Unterschied zwischen den Sharpe Ratios zweier Strategien echt ist oder nur Zufall – auch wenn die
Renditen „unangenehme" Eigenschaften haben.

**Das Problem:** Strategie A hat eine Sharpe Ratio von 1,01, Strategie B von 0,91 – ist A wirklich
besser? Es gibt einen klassischen Test dafür (von Jobson/Korkie). Doch dieser setzt voraus, dass die
Renditen sich „brav" verhalten: normalverteilt und zeitlich unabhängig. Echte Finanzrenditen tun das
nicht – sie haben „schwere Ränder" (extreme Ausschläge kommen häufiger vor als bei einer Glockenkurve)
und hängen zeitlich zusammen (ruhige und stürmische Phasen wechseln sich ab). Dann ist der klassische
Test **ungültig**.

**Was die Arbeit macht und herausfindet:** Ledoit und Wolf entwickeln einen **robusten** Test, der diese
Probleme verkraftet. Kern ist ein Verfahren namens **Bootstrap** (siehe Quelle 21): Statt sich auf
fragwürdige Annahmen zu verlassen, „zieht" man immer wieder neue Stichproben aus den vorhandenen Daten
und beobachtet, wie stark der gemessene Unterschied schwankt. Daraus ergibt sich ein verlässliches
Konfidenzintervall: Enthält es die Null nicht, gilt der Unterschied als echt (statistisch signifikant).

**Schlüsselbegriffe einfach erklärt:**
- *statistischer Test / Signifikanz:* ein Verfahren, das prüft, ob ein beobachteter Unterschied wahrscheinlich echt ist oder bloß Zufall.
- *Normalverteilung („Glockenkurve"):* eine idealtypische, symmetrische Verteilung; viele Tests setzen sie voraus.
- *schwere Ränder (fat tails):* extreme Ereignisse treten häufiger auf, als die Glockenkurve vorhersagt.
- *Konfidenzintervall:* ein Bereich, in dem der „wahre" Wert mit hoher Wahrscheinlichkeit liegt.

**Warum das für die Seminararbeit zählt:** Dieser Test ist das **zentrale Werkzeug** der Arbeit, um zu
entscheiden, ob eine Strategie eine andere wirklich schlägt. Er ist der Grund, warum die Arbeit
„statistisch abgesichert" im Untertitel trägt.

## 21. Politis & Romano (1994): „The Stationary Bootstrap"
*Journal of the American Statistical Association, Bd. 89, Nr. 428, S. 1303–1313.*

**In einem Satz:** Die Autoren entwickeln eine Variante des „Bootstrap"-Verfahrens, die speziell für
zeitlich zusammenhängende Daten (Zeitreihen) gemacht ist.

**Das Problem:** Beim **Bootstrap** zieht man zufällig immer wieder neue „Pseudo-Datensätze" aus den
vorhandenen Daten, um die Unsicherheit einer Schätzung einzuschätzen. Das normale Bootstrap zieht
**einzelne** Beobachtungen zufällig – das funktioniert aber nur, wenn die Beobachtungen unabhängig sind.
Bei Finanzdaten hängt der heutige Wert mit dem gestrigen zusammen; zieht man einzeln, **zerstört** man
diese Struktur.

**Was die Arbeit macht:** Politis und Romano schlagen vor, statt einzelner Beobachtungen ganze **Blöcke**
aufeinanderfolgender Beobachtungen zu ziehen. So bleibt der zeitliche Zusammenhang (etwa, dass auf ruhige
Tage oft weitere ruhige Tage folgen) erhalten. Beim „stationären Bootstrap" haben diese Blöcke zufällige
Längen, was das Verfahren besonders gut für stationäre Zeitreihen macht.

**Schlüsselbegriffe einfach erklärt:**
- *Bootstrap:* ein Trick, bei dem man aus den vorhandenen Daten viele neue Zufallsstichproben „zieht", um die Unsicherheit einer Schätzung zu messen – ohne starke Annahmen.
- *Zeitreihe:* Daten in zeitlicher Reihenfolge (z. B. tägliche Kurse).
- *Block-Resampling:* das Ziehen ganzer zusammenhängender Datenstücke statt einzelner Punkte.
- *stationär:* eine Zeitreihe, deren statistische Eigenschaften über die Zeit ungefähr gleich bleiben.

**Warum das für die Seminararbeit zählt:** Dieses Verfahren ist der **technische Unterbau** des robusten
Sharpe-Tests (Quelle 20). Ohne den Block-Bootstrap ließe sich der zeitliche Zusammenhang der Renditen
nicht korrekt berücksichtigen.

## 22. Holm (1979): „A Simple Sequentially Rejective Multiple Test Procedure"
*Scandinavian Journal of Statistics, Bd. 6, Nr. 2, S. 65–70.*

**In einem Satz:** Holm liefert eine einfache, aber wirkungsvolle Methode, um beim **gleichzeitigen**
Prüfen vieler Hypothesen Fehlalarme zu vermeiden, ohne unnötig streng zu sein.

**Das Problem:** Wer **viele** Tests gleichzeitig durchführt, findet fast zwangsläufig irgendwo einen
„signifikanten" Treffer – rein zufällig. (Bild: Wer oft genug würfelt, wirft auch mal fünfmal
hintereinander eine Sechs.) Man muss also strenger werden. Die einfachste Lösung (Bonferroni) ist aber
**zu** streng und übersieht dadurch echte Effekte.

**Was die Arbeit macht und herausfindet:** Holm schlägt ein **schrittweises** Verfahren vor: Man ordnet
die Testergebnisse vom „stärksten" zum „schwächsten" und prüft sie der Reihe nach mit einer angepassten
Hürde. Das hält die Gefahr eines Fehlalarms insgesamt zuverlässig unter Kontrolle, ist dabei aber
**trennschärfer** als Bonferroni – es findet also mehr echte Effekte, ohne mehr Fehlalarme zu erzeugen.

**Schlüsselbegriffe einfach erklärt:**
- *Hypothese:* eine zu prüfende Behauptung (z. B. „Strategie A schlägt B").
- *multiples Testen:* viele Hypothesen gleichzeitig prüfen.
- *Fehlalarm (Fehler 1. Art):* etwas für echt halten, das in Wahrheit Zufall ist.
- *Bonferroni-Korrektur:* die einfachste (aber strenge) Methode, die Hürde bei vielen Tests anzuheben.
- *Trennschärfe (Power):* die Fähigkeit eines Tests, echte Effekte tatsächlich zu erkennen.

**Warum das für die Seminararbeit zählt:** Die Arbeit vergleicht mehrere Strategien gleichzeitig (fünf
paarweise Vergleiche). Die **Holm-Korrektur** sorgt dafür, dass ein als „signifikant" ausgewiesener
Unterschied auch dieser Mehrfachprüfung standhält.

## 23. Harvey, Liu & Zhu (2014/2016): „… and the Cross-Section of Expected Returns"
*NBER Working Paper Nr. 20592 (2014); veröffentlicht in The Review of Financial Studies, Bd. 29, Nr. 1 (2016), S. 5–68.*

**In einem Satz:** Die Autoren warnen, dass die Finanzforschung über die Jahre so viele „Renditefaktoren"
getestet hat, dass die meisten „Entdeckungen" wahrscheinlich nur Zufall sind – und fordern strengere
Maßstäbe.

**Das Problem:** Hunderte von Studien haben hunderte von „Faktoren" vorgeschlagen, die angeblich
Aktienrenditen erklären. Bei so viel Suchen (Data-Mining) findet man zwangsläufig viele
Scheinzusammenhänge. Die übliche Signifikanzhürde (ein **t-Wert über 2,0**) ist dafür viel zu lasch.

**Was die Arbeit macht und herausfindet:** Harvey, Liu und Zhu wenden die Logik des multiplen Testens auf
die gesamte Faktorforschung an. Ihr Ergebnis: Ein neu „entdeckter" Faktor müsste eine **deutlich höhere**
Hürde nehmen – einen **t-Wert über 3,0** statt 2,0. Daraus folgt eine unbequeme Schlussfolgerung: Viele
der in der Literatur behaupteten Befunde halten dieser strengeren Prüfung **nicht** stand und sind
vermutlich falsch.

**Schlüsselbegriffe einfach erklärt:**
- *Faktor:* eine Eigenschaft (z. B. Unternehmensgröße, Momentum), die angeblich Renditen erklärt.
- *t-Wert:* eine statistische Kennzahl; je größer, desto unwahrscheinlicher ein reiner Zufallstreffer. Faustregel war lange „über 2,0".
- *Data-Mining:* das (oft unkontrollierte) Durchsuchen vieler Möglichkeiten, bis man einen „signifikanten" Treffer findet.

**Warum das für die Seminararbeit zählt:** Diese Quelle begründet, **warum** die Arbeit so viel Wert auf
strenge Statistik legt: Ohne diese Vorsicht hält man leicht Zufall für Können. Sie motiviert direkt die
Korrektur für multiples Testen.

## 24. Bailey & López de Prado (2014): „The Deflated Sharpe Ratio"
*The Journal of Portfolio Management, Bd. 40, Nr. 5, S. 94–107.*

**In einem Satz:** Die Autoren bauen eine „entschärfte" Version der Sharpe Ratio, die berücksichtigt,
dass eine hohe Sharpe Ratio oft nur deshalb hoch ist, weil man viele Strategien ausprobiert hat – und
nicht, weil die Strategie wirklich gut ist.

**Das Problem:** Wenn man **viele** Strategien testet und am Ende die beste auswählt, ist deren Sharpe
Ratio fast automatisch beeindruckend – selbst wenn alle Strategien in Wahrheit nichts taugen. (Bild:
Lässt man 1 000 Affen Aktien auswählen, hat der „beste" Affe eine tolle Bilanz – durch puren Zufall.)
Außerdem täuscht die Sharpe Ratio bei kurzen Datenreihen und „schiefen" Renditen eine höhere Qualität
vor, als wirklich vorhanden ist.

**Was die Arbeit macht und herausfindet:** Die **Deflated Sharpe Ratio (DSR)** korrigiert genau diese
Verzerrungen. Sie misst die beobachtete Sharpe Ratio nicht gegen null, sondern gegen die Messlatte, die
man **allein durch Zufall** erwarten würde, wenn man so viele Strategien ausprobiert hat. Zusätzlich
rechnet sie die Länge der Datenreihe sowie Schiefe und „Spitzigkeit" der Renditeverteilung ein. Ergebnis
ist eine Wahrscheinlichkeit dafür, dass die Strategie **wirklich** Können besitzt (also eine echte
positive Sharpe Ratio hat).

**Schlüsselbegriffe einfach erklärt:**
- *Selektionsverzerrung (selection bias):* die Verzerrung, die entsteht, wenn man aus vielen Versuchen den besten auswählt und nur diesen betrachtet.
- *Schiefe / Kurtosis:* Maße dafür, wie unsymmetrisch bzw. wie „spitz/randlastig" eine Verteilung ist (Abweichungen von der Glockenkurve).
- *Backtest:* die Simulation einer Strategie auf historischen Daten.

**Warum das für die Seminararbeit zählt:** Die Arbeit berechnet die DSR für jede ihrer vier Strategien,
um sicherzustellen, dass deren Sharpe Ratios nicht bloß ein Produkt von Zufall und Strategiesuche sind.

## 25. Bailey, Borwein, López de Prado & Zhu (2015/2016): „The Probability of Backtest Overfitting"
*Arbeitspapier 2015; veröffentlicht im Journal of Computational Finance, Bd. 20, Nr. 4 (2016), S. 39–69.*

**In einem Satz:** Die Autoren liefern eine Methode, um zu messen, wie wahrscheinlich es ist, dass eine
im Backtest tolle Strategie in Wahrheit nur „überangepasst" ist und in Zukunft versagt.

**Das Problem:** Viele Investmentfirmen wählen Strategien anhand von **Backtests** (Simulationen auf
historischen Daten) aus. Aber wer lange genug an einer Strategie feilt, bis sie auf den historischen
Daten glänzt, hat oft nur das **zufällige Rauschen** der Vergangenheit nachgebaut – nicht echtes Können.
Solche Strategien sehen rückblickend großartig aus und enttäuschen dann in der Realität.

**Was die Arbeit macht und herausfindet:** Sie führen die **Probability of Backtest Overfitting (PBO)**
ein – ein Maß dafür, wie wahrscheinlich es ist, dass die im Rückblick beste Strategie außerhalb der
Stichprobe sogar unterdurchschnittlich abschneidet. Der Kernmechanismus: Je mehr Strategien man
ausprobiert, desto höher wird die beste Sharpe Ratio – ganz von allein, **auch wenn keine** der
Strategien echtes Können hat. Wer aus vielen die „Beste" pickt, unterliegt also dem „Fluch des
Gewinners".

**Schlüsselbegriffe einfach erklärt:**
- *Backtest-Overfitting (Überanpassung):* wenn eine Strategie an die Vergangenheit überangepasst ist und nur deren Zufallsmuster abbildet.
- *Hold-out:* das übliche Verfahren, einen Teil der Daten zum Testen zurückzuhalten – laut den Autoren bei Finanz-Backtests unzuverlässig.
- *„Fluch des Gewinners":* die Tendenz, dass die scheinbar beste aus vielen Auswahlen systematisch überschätzt wird.

**Warum das für die Seminararbeit zählt:** Diese Quelle liefert die **theoretische Begründung** für die
Overfitting-Gefahr, gegen die die Deflated Sharpe Ratio (Quelle 24) absichert – ein zentraler Baustein
der statistischen Vorsicht in der Arbeit.

## 26. López de Prado (2018): „Advances in Financial Machine Learning" (Buch)
*Monografie, John Wiley & Sons, 2018 (in der Arbeit v. a. Kapitel 7).*

**In einem Satz:** Ein praxisnahes Lehrbuch, das zeigt, wie man maschinelles Lernen in der Finanzwelt
**richtig** einsetzt – und warum die üblichen Methoden hier oft in die Irre führen.

**Das Problem:** Beim maschinellen Lernen testet man Modelle normalerweise mit **Kreuzvalidierung**: Man
teilt die Daten in Stücke, trainiert auf den einen und testet auf den anderen. Bei Finanzdaten geht das
aber schief, weil aufeinanderfolgende Datenpunkte zeitlich zusammenhängen. Dann „sieht" das Modell im
Test indirekt schon die Antwort – ein verstecktes **Informationsleck** (Leakage), das zu trügerisch
guten Ergebnissen führt.

**Was die Arbeit macht und herausfindet:** López de Prado erklärt, warum die übliche
„k-Fold"-Kreuzvalidierung in der Finanzwelt versagt, und schlägt als Lösung die **Purged K-Fold
Cross-Validation** mit „Embargo" vor: Aus den Trainingsdaten werden alle Beobachtungen entfernt, die sich
zeitlich mit den Testdaten überlappen (Purging), plus eine kleine Pufferzone unmittelbar danach
(Embargo). So wird das Leck geschlossen, und die Bewertung des Modells wird ehrlich.

**Schlüsselbegriffe einfach erklärt:**
- *Kreuzvalidierung (Cross-Validation):* ein Verfahren, ein Modell zu testen, indem man die Daten abwechselnd in Trainings- und Testteile aufteilt.
- *Informationsleck (Leakage):* wenn Wissen aus den Testdaten ungewollt ins Training sickert und das Ergebnis schönt.
- *Purging / Embargo:* das gezielte Entfernen zeitlich überlappender (Purging) und unmittelbar folgender (Embargo) Datenpunkte, um das Leck zu schließen.

**Warum das für die Seminararbeit zählt:** Die Arbeit nutzt genau dieses Verfahren, um den Random Forest
**leckfrei** zu trainieren und seine Einstellungen (Hyperparameter) ehrlich zu wählen.

---

# Teil H – Fallstricke der Praxis

## 27. Almgren & Chriss (2000): „Optimal Execution of Portfolio Transactions"
*Journal of Risk, Bd. 3, Nr. 2, S. 5–39.*

**In einem Satz:** Die Autoren zeigen, dass der Kauf oder Verkauf großer Aktienmengen selbst Kosten
verursacht – und wie man diese Handelskosten klug steuert.

**Das Problem:** In einfachen Modellen tut man so, als könne man Aktien zu einem festen Preis beliebig
kaufen und verkaufen. In Wirklichkeit **bewegt** ein großer Auftrag den Preis: Wer schnell viel verkauft,
drückt den Kurs gegen sich selbst. Diese „Markteinfluss"-Kosten werden oft unterschätzt.

**Was die Arbeit macht und herausfindet:** Almgren und Chriss zerlegen die Handelskosten in zwei Teile:
den **permanenten** Markteinfluss (der Preis bleibt dauerhaft verschoben) und den **temporären** (eine
vorübergehende Verschlechterung, weil man zu schnell handelt). Sie zeigen einen Zielkonflikt: Handelt man
langsam, sinken die Kosten, aber man trägt länger das Risiko von Kursschwankungen; handelt man schnell,
ist es umgekehrt. Daraus leiten sie optimale Handelsstrategien ab. Wichtig: Die Kosten hängen von
**Handelsgröße und Liquidität** ab.

**Schlüsselbegriffe einfach erklärt:**
- *Transaktionskosten:* alle Kosten, die beim Handeln entstehen (Gebühren, Geld-Brief-Spanne, Markteinfluss).
- *Markteinfluss (market impact):* die Kursbewegung, die der eigene Kauf/Verkauf auslöst.
- *Liquidität:* wie leicht sich ein Wertpapier handeln lässt, ohne den Kurs stark zu bewegen.

**Warum das für die Seminararbeit zählt:** Die Arbeit modelliert Transaktionskosten vereinfacht als
pauschalen Satz. Almgren und Chriss zeigen, dass reale Kosten komplexer sind – die Grundlage für eine
ehrliche **Limitation** in der kritischen Würdigung der Arbeit.

## 28. Brown, Goetzmann, Ibbotson & Ross (1992): „Survivorship Bias in Performance Studies"
*The Review of Financial Studies, Bd. 5, Nr. 4, S. 553–580.*

**In einem Satz:** Die Autoren warnen vor einer tückischen Datenfalle: Wenn man nur die „Überlebenden"
untersucht, entstehen scheinbare Muster, die in Wirklichkeit gar nicht existieren.

**Das Problem:** Untersucht man z. B. nur Fonds (oder Aktien), die es **heute noch** gibt, hat man die
gescheiterten automatisch ausgeklammert. Das verzerrt das Bild – denn die Schwachen sind verschwunden.
Manche Studien fanden so eine „Vorhersagbarkeit" von Renditen. Ist die echt?

**Was die Arbeit macht und herausfindet:** Brown, Goetzmann, Ibbotson und Ross zeigen mit Berechnungen,
dass eine durch das **Überleben „abgeschnittene" Stichprobe** ganz von allein scheinbare Zusammenhänge
erzeugt – etwa den Eindruck, dass gute Vergangenheitsleistung künftige Leistung vorhersagt. Dieser Effekt
kann so stark sein, dass er die behauptete „Vorhersagbarkeit" vollständig erklärt, ohne dass dahinter ein
echter Mechanismus steckt.

**Schlüsselbegriffe einfach erklärt:**
- *Survivorship Bias (Überlebensverzerrung):* die Verzerrung, die entsteht, wenn man nur die „Überlebenden" betrachtet und die Gescheiterten ignoriert.
- *abgeschnittene / truncierte Stichprobe:* ein Datensatz, aus dem ein Teil (hier: die Ausgeschiedenen) systematisch fehlt.

**Warum das für die Seminararbeit zählt:** Das Anlageuniversum der Arbeit besteht nur aus **heute noch
existierenden** Aktien. Brown et al. liefern die theoretische Grundlage dafür, diesen Survivorship Bias
als wichtigste **Einschränkung** der Arbeit offen zu benennen.

## 29. Elton, Gruber & Blake (1996): „Survivorship Bias and Mutual Fund Performance"
*The Review of Financial Studies, Bd. 9, Nr. 4, S. 1097–1120. (Eingescannt; per Bild gelesen.)*

**In einem Satz:** Die Autoren messen konkret, wie stark der Survivorship Bias die gemessene Leistung von
Investmentfonds verfälscht.

**Das Problem:** Fonds verschwinden meist nicht zufällig, sondern weil sie **schlecht** liefen (sie werden
geschlossen oder mit anderen verschmolzen). Lässt eine Datenbank diese verschwundenen Fonds weg, sieht
der Durchschnitt der „Überlebenden" zu rosig aus.

**Was die Arbeit macht und herausfindet:** Elton, Gruber und Blake schätzen die **Größe** dieser
Verzerrung sorgfältig ab. Dabei berücksichtigen sie sogar die Bedingungen, zu denen verschwundene Fonds
mit anderen verschmolzen wurden, um die Renditen korrekt einzurechnen. Ergebnis: Der Survivorship Bias
ist real und messbar – er verschiebt die durchschnittlich gemessene Fondsperformance systematisch nach
oben.

**Schlüsselbegriffe einfach erklärt:**
- *Fonds-Attrition / -Mortalität:* das „Aussterben" von Fonds durch Schließung oder Fusion.
- *Fusion (Merger):* das Verschmelzen eines (oft schwachen) Fonds mit einem anderen.

**Warum das für die Seminararbeit zählt:** Diese Quelle ist ein konkretes **Anwendungsbeispiel** für den
Survivorship Bias aus Quelle 28 – und untermauert thematisch, warum die Arbeit ihre eigene
Survivorship-Verzerrung ernst nimmt.

---

# Abschluss: der rote Faden

Damit sind **alle 29 Quellen** erklärt. Sie ergeben zusammen eine logische Kette:

1. **Markowitz** zeigt, wie man Rendite und Risiko abwägt (Teil A).
2. Die **Sharpe Ratio** misst, wie gut eine Anlage Risiko in Ertrag verwandelt (Teil B).
3. Doch **Schätzfehler** – besonders bei den erwarteten Renditen – machen die schöne Theorie zerbrechlich (Teil C).
4. Dagegen helfen bessere Schätzer und Beschränkungen (Teil D); oft schlägt aber schon die einfache **1/N-Regel** alles Aufwändige (Teil E).
5. **Maschinelles Lernen** kann die Renditeprognose nur wenig verbessern (Teil F).
6. Deshalb braucht es **strenge Statistik**, um echtes Können von Zufall zu trennen (Teil G).
7. Und in der Praxis lauern **Handelskosten und Datenfallen** (Teil H).

Genau diese Kette führt zum Kernbefund der Seminararbeit: **Keine** der aufwändigen Strategien schlägt die
denkbar einfachste Regel – die Gleichverteilung – in statistisch belastbarer Weise.
