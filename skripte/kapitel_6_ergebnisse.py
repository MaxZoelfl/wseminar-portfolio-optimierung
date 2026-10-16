"""
Kapitel 6 — Ergebnisse und Diskussion: jede Zahl des Kapitels, in der Reihenfolge der Arbeit.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Kapitel 6 berichtet, wie die vier Strategien abgeschnitten haben, und ordnet es
ein. Fast jede Zahl dort ist ein Ergebnis des großen Backtests. Dieses Skript
rechnet sie aus den gespeicherten Ergebnisdateien nach:

  output/daily_returns.csv     Tagesrenditen → Rendite, Volatilität, Sharpe, Drawdown
  output/weights_*.csv         Monatsgewichte → effektive Titelzahl, NVIDIA-Anteil
  output/turnover.csv          Umschlag → Kosten, Sensitivität (§ 6.4)
  output/experiment_log.json   Signifikanztests mit voller Genauigkeit (§ 6.3)

Alle Kennzahlen mit den Formeln aus portfolio/metrics.py, alle Tests mit
portfolio/significance.py — also genau so, wie der Backtest sie gerechnet hat.

Einzige etwas längere Rechnung: die Zinsschwelle in § 6.6 wiederholt den Test von
Ledoit und Wolf für mehrere Zinssätze (rund 30 Sekunden).

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_6_ergebnisse.py        # rund 40 Sekunden
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from portfolio.metrics import cagr, annualized_vol, max_drawdown, calmar_ratio   # noqa: E402
from portfolio.plots import rollierender_sharpe       # noqa: E402  (dieselbe Rechnung wie Abbildung A4)
from portfolio.significance import sharpe_difference_test, holm_bonferroni   # noqa: E402
from portfolio.theory_plots import konfidenzintervalle, diversifikation, gleichgewichtete_varianz   # noqa: E402
import abbildung_kostensensitivitaet as kosten        # noqa: E402

# Zeilen der Signifikanztabelle in § 6.3 → Schlüssel im Laufprotokoll, dazu die Werte, wie sie in der
# Tabelle stehen (sie legen die Stellenzahl der Ausgabe fest): Δ (ann. Sharpe), s(Δ), t, p, p nach Holm.
TABELLE_63 = [
    ("RF vs. Markowitz",             "RF vs. MVO",         ("0,029", "0,178", "0,165", "0,866", "1,000")),
    ("RF vs. Equal Weight",          "RF vs. EW",          ("0,035", "0,121", "0,291", "0,773", "1,000")),
    ("RF vs. Risk Parity",           "RF vs. RP",          ("0,146", "0,130", "1,122", "0,257", "1,000")),
    ("Markowitz vs. Equal Weight",   "MVO vs. EW",         ("0,006", "0,146", "0,040", "0,971", "1,000")),
    ("Markowitz vs. Risk Parity",    "MVO vs. RP",         ("0,116", "0,160", "0,724", "0,489", "1,000")),
    ("Equal Weight vs. Risk Parity", "Risk Parity vs. EW", ("0,110", "0,038", "2,874", "0,0036", "0,0216")),
]
# Tabelle in § 6.1, Spalten wie in der Arbeit: CAGR, Vola, Sharpe, max. DD, Calmar.
TABELLE_61 = {"Random Forest": ("22,12", "19,03", "0,936", "−33,77", "0,655"),
              "Markowitz MVO": ("23,14", "21,00", "0,906", "−34,28", "0,675"),
              "Equal Weight":  ("20,03", "17,56", "0,900", "−33,25", "0,602"),
              "Risk Parity":   ("16,92", "16,45", "0,790", "−32,44", "0,522")}


def tagesgewichte(g: pd.DataFrame, tage: pd.DatetimeIndex) -> pd.DataFrame:
    """Gewicht je Handelstag wie in portfolio/backtest.py: Das am Monatsende
    festgelegte Gewicht gilt ab dem folgenden Handelstag bis zum nächsten Monatsende."""
    termine = g.index
    stelle = termine.searchsorted(tage, side="left") - 1      # letzter Termin VOR dem Tag
    return pd.DataFrame(g.values[stelle], index=tage, columns=g.columns)


def zinsschwelle(s: pd.DataFrame) -> float:
    """§ 6.6: Zinssatz in Prozent, unter dem 'Risk Parity < 1/N' nach Holm nicht
    mehr signifikant ist. Test genau wie im Backtest (4999 Ziehungen, alle sechs
    Paare, Holm), nur mit einem anderen risikofreien Zins; zwischen dem letzten
    signifikanten und dem ersten nicht signifikanten Zins wird linear interpoliert."""
    paare = [("Random Forest", "Markowitz MVO"), ("Random Forest", "Equal Weight"),
             ("Random Forest", "Risk Parity"), ("Markowitz MVO", "Equal Weight"),
             ("Markowitz MVO", "Risk Parity"), ("Risk Parity", "Equal Weight")]
    punkte = []
    for zins in (0.010, 0.009, 0.008, 0.007):
        pw = [sharpe_difference_test(s[a], s[b], rf=zins, n_boot=4999)["p_value"] for a, b in paare]
        punkte.append((zins, holm_bonferroni(pw)["p_adjusted"][-1]))   # Holm-p von RP gegen 1/N
    for (z1, h1), (z2, h2) in zip(punkte, punkte[1:]):
        if h1 < 0.05 <= h2:
            return (z2 + (z1 - z2) * (0.05 - h2) / (h1 - h2)) * 100
    return float("nan")


def pruefe() -> Pruefung:
    """Alle Zahlen aus Kapitel 6, in der Reihenfolge der Arbeit."""
    p = Pruefung("Kapitel 6 — Ergebnisse und Diskussion")
    s = strategierenditen()
    log = protokoll()
    meta = log["experiment_meta"]
    rf = meta["risk_free_rate"]
    G = gewichte()
    U = umschlag()
    regulaer = U.iloc[1:]                             # Umschlag ohne den Depotaufbau
    wert = (1 + s).cumprod()                          # Depotwert je Tag, Start 1 Euro
    test = log["significance"]["sharpe_difference_tests"]

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("6.1 Rendite und Risikokennzahlen")
    K = pd.DataFrame({n: dict(cagr=cagr(s[n]) * 100, vola=annualized_vol(s[n]) * 100,
                              sharpe=sharpe_exakt(n), dd=max_drawdown(s[n]) * 100,
                              calmar=calmar_ratio(s[n])) for n in STRATEGIEN}).T
    p.aussage("Strategien mit positiver Rendite nach Kosten", "positive Rendite",
              bool((K.cagr > 0).all()), f"{int((K.cagr > 0).sum())} von {len(K)}",
              notiz="CAGR (Formel B1) aus output/daily_returns.csv", woertlich=True)
    for name, (c, v, sh, dd, cal) in TABELLE_61.items():
        p.zahl(f"Tabelle, {name}: CAGR", c, K.cagr[name], "%")
        p.zahl(f"Tabelle, {name}: Volatilität", v, K.vola[name], "%")
        p.zahl(f"Tabelle, {name}: Sharpe-Quotient", sh, K.sharpe[name])
        p.zahl(f"Tabelle, {name}: maximaler Drawdown", dd, K.dd[name], "%")
        p.zahl(f"Tabelle, {name}: Calmar-Quotient", cal, K.calmar[name])
    for name, text in [("Markowitz MVO", "7,85"), ("Random Forest", "7,23"),
                       ("Equal Weight", "6,10"), ("Risk Parity", "4,70")]:
        p.zahl(f"Endvermögen aus einem Euro, {name}", text, wert[name].iloc[-1], "Euro",
               notiz="Produkt aller (1 + Tagesrendite)")
    p.aussage("höchste Rendite", "Markowitz", K.cagr.idxmax() == "Markowitz MVO", K.cagr.idxmax(),
              notiz=" > ".join(f"{KURZ[n]} {de(K.cagr[n], 2)} %" for n in K.cagr.sort_values(ascending=False).index))
    reihe = list(K.sharpe.sort_values(ascending=False).index)
    p.aussage("Markowitz risikoadjustiert (Sharpe) an", "zweiter Stelle", reihe.index("Markowitz MVO") == 1,
              f"{reihe.index('Markowitz MVO') + 1}. Stelle",
              notiz=" > ".join(f"{KURZ[n]} {de(K.sharpe[n], 3)}" for n in reihe), woertlich=True)
    p.zahl("Volatilität von Markowitz, die höchste", "21,0", K.vola["Markowitz MVO"], "%",
           notiz=" > ".join(f"{KURZ[n]} {de(K.vola[n], 2)} %" for n in K.vola.sort_values(ascending=False).index))
    p.aussage("höchster Sharpe-Quotient", "Random Forest", K.sharpe.idxmax() == "Random Forest",
              K.sharpe.idxmax())
    p.aussage("geringste Volatilität und flachster Drawdown", "Risk Parity",
              K.vola.idxmin() == "Risk Parity" and K.dd.idxmax() == "Risk Parity",
              K.vola.idxmin() if K.vola.idxmin() == K.dd.idxmax() else f"{K.vola.idxmin()}, {K.dd.idxmax()}",
              notiz=f"Volatilität {de(K.vola.min(), 2)} %, maximaler Drawdown {de(K.dd.max(), 2)} %")
    # Bildunterschrift Abbildung 4
    p.zahl("Abb. 4: Anfangskapital", "einem", wert.iloc[0].div(1 + s.iloc[0]).iloc[0], "Euro",
           notiz="Depotwert vor dem ersten Handelstag")
    p.zahl("Abb. 4: Transaktionskosten", "zehn", meta["transaction_cost"] * 10_000, "Basispunkte",
           notiz="transaction_cost im Laborprotokoll")
    p.aussage("Abb. 4: Datenzeitraum", "Februar 2015 bis Dezember 2024",
              (s.index[0].year, s.index[0].month, s.index[-1].year, s.index[-1].month) == (2015, 2, 2024, 12),
              f"{s.index[0]:%d.%m.%Y} bis {s.index[-1]:%d.%m.%Y}", woertlich=True)
    vorn = wert.idxmax(axis=1).value_counts(normalize=True) * 100   # wer liegt je Tag vorn?
    for name, text in [("Markowitz MVO", "86,8"), ("Random Forest", "11,8"), ("Equal Weight", "1,4")]:
        p.zahl(f"Anteil der Handelstage in Führung, {name}", text, vorn.get(name, 0.0), "%",
               notiz="Tage, an denen sein Depotwert der höchste der vier ist")
    tiefpunkt = {n: ((wert[n] / wert[n].cummax()) - 1).idxmin() for n in STRATEGIEN}
    for name, datum, iso in [("Random Forest", "23. März 2020", "2020-03-23"),
                             ("Equal Weight", "23. März 2020", "2020-03-23"),
                             ("Risk Parity", "23. März 2020", "2020-03-23"),
                             ("Markowitz MVO", "30. September 2022", "2022-09-30")]:
        ist = tiefpunkt[name]
        p.aussage(f"Tag des maximalen Drawdowns, {name}", datum, ist == pd.Timestamp(iso),
                  f"{ist:%d.%m.%Y}", notiz="tiefster Abstand des Depotwerts zu seinem bisherigen Höchststand",
                  woertlich=True)
    p.aussage("führt nach dem Calmar-Quotienten", "Markowitz", K.calmar.idxmax() == "Markowitz MVO",
              K.calmar.idxmax(),
              notiz=" > ".join(f"{KURZ[n]} {de(K.calmar[n], 3)}" for n in K.calmar.sort_values(ascending=False).index))
    p.zahl("so viel stärker schwankt Markowitz als der Random Forest", "zehn",
           (K.vola["Markowitz MVO"] / K.vola["Random Forest"] - 1) * 100, "%", ungefaehr=True,
           notiz=f"{de(K.vola['Markowitz MVO'], 2)} % ÷ {de(K.vola['Random Forest'], 2)} % − 1")
    p.zahl("so viel tiefer ist sein tiefster Absturz", "anderthalb",
           (K.dd["Markowitz MVO"] / K.dd["Random Forest"] - 1) * 100, "%", ungefaehr=True,
           notiz=f"{de(K.dd['Markowitz MVO'], 2)} % ÷ {de(K.dd['Random Forest'], 2)} % − 1")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("6.2 Struktur der Portfolios")
    neff = {n: effektive_titelzahl(G[n]) for n in STRATEGIEN}   # Formel 35 je Termin
    eins = pd.DataFrame([[1.0] + [0.0] * 14], columns=TICKERS)  # alles in einem Titel
    p.zahl("effektive Titelzahl bei 15 gleichen Gewichten", "15", neff["Equal Weight"].iloc[0],
           notiz="Formel 35: 1 ÷ Σ w², hier 1 ÷ (15 · (1/15)²)")
    p.zahl("effektive Titelzahl bei vollständiger Konzentration", "1", effektive_titelzahl(eins).iloc[0],
           notiz="alles in einem Titel: 1 ÷ 1²")
    for name, text in [("Markowitz MVO", "5,88"), ("Random Forest", "6,16"),
                       ("Risk Parity", "14,05"), ("Equal Weight", "15,00")]:
        p.zahl(f"effektive Titelzahl im Mittel, {name}", text, neff[name].mean(),
               notiz="Mittel über die 119 Termine, aus output/weights_*.csv" if name != "Equal Weight" else "")
    leer = {n: (G[n] < 0.0005).sum(axis=1).mean() for n in ["Markowitz MVO", "Random Forest"]}
    p.aussage("so viel des Universums lassen beide optimierenden Verfahren leer", "rund die Hälfte",
              all(6.5 <= v <= 8.5 for v in leer.values()),
              " bzw. ".join(de(v, 1) for v in leer.values()) + " von 15 Titeln",
              notiz=" · ".join(f"{KURZ[n]} {de(v, 2)}" for n, v in leer.items())
              + " Titel mit Gewicht unter 0,05 %, Mittel je Termin", woertlich=True)
    grenze = {n: (G[n] >= 0.1995).sum(axis=1).mean() for n in ["Markowitz MVO", "Random Forest"]}
    p.aussage("Titel an der Obergrenze (Mittel je Termin)", "etwa drei",
              all(abs(v - 3) <= 0.5 for v in grenze.values()), " bzw. ".join(de(v, 1) for v in grenze.values()),
              notiz=" · ".join(f"{KURZ[n]} {de(v, 2)}" for n, v in grenze.items())
              + " Titel mit Gewicht ≥ 19,95 %", woertlich=True)
    p.zahl("Obergrenze je Titel", "20", meta["max_weight"] * 100, "%")
    p.zahl("Monate, in denen Markowitz NVIDIA an der Obergrenze hält", "81",
           int((G["Markowitz MVO"]["NVDA"] >= 0.1999).sum()))
    p.zahl("von Terminen", "119", len(G["Markowitz MVO"]))
    mu_ci, halb, _ = konfidenzintervalle(titelrenditen())   # Formel 17, wie Abbildung 3
    getrennt = [(i, j) for i, j in combinations(TICKERS, 2) if abs(mu_ci[i] - mu_ci[j]) > halb[i] + halb[j]]
    in_allen = [t for t in TICKERS if all(t in paar for paar in getrennt)]
    p.aussage("einziger Titel, dessen Erwartungswert sich trennen lässt (§ 2.5)", "NVIDIA",
              in_allen == ["NVDA"], ", ".join(in_allen) or "keiner",
              notiz=f"steckt in allen {len(getrennt)} Paaren mit getrennten 95-%-Intervallen (Formel 17)")
    for name, text in [("Random Forest", "29,0"), ("Markowitz MVO", "13,1")]:
        p.zahl(f"mittlerer monatlicher Umschlag, {name}", text, regulaer[UMSCHLAG_SPALTE[name]].mean() * 100, "%",
               notiz="Formel 26, Mittel über die 118 Termine nach dem Depotaufbau")
    klein = {n: regulaer[UMSCHLAG_SPALTE[n]].mean() * 100 for n in ["Equal Weight", "Risk Parity"]}
    p.aussage("Umschlag der beiden übrigen", "gut zwei Prozent", all(2.0 <= v < 2.5 for v in klein.values()),
              " bzw. ".join(de(v, 1) for v in klein.values()) + " %",
              notiz=" · ".join(f"{KURZ[n]} {de(v, 2)} %" for n, v in klein.items()), woertlich=True)
    p.zahl("effektive Titelzahl Markowitz, kleinste", "5,02", neff["Markowitz MVO"].min())
    p.zahl("effektive Titelzahl Markowitz, größte", "8,48", neff["Markowitz MVO"].max())

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("6.3 Ergebnisse der Signifikanzprüfung")
    p.zahl("geprüfte Paarungen", "sechs", len(test))
    for zeile, schluessel, (d_, se_, st_, p_, h_) in TABELLE_63:
        t = test[schluessel]
        p.zahl(f"Tabelle, {zeile}: Δ (Sharpe, jährlich)", d_, abs(t["diff_annual"]))
        p.zahl(f"Tabelle, {zeile}: Standardfehler s(Δ)", se_, t["se_annual"])
        p.zahl(f"Tabelle, {zeile}: Teststatistik t", st_, abs(t["statistic"]))
        p.zahl(f"Tabelle, {zeile}: p", p_, t["p_value"])
        p.zahl(f"Tabelle, {zeile}: p nach Holm", h_, t["p_holm"])
    p.zahl("p Random Forest gegen Gleichgewichtung", "0,77", test["RF vs. EW"]["p_value"])
    p.zahl("p Random Forest gegen Markowitz", "0,87", test["RF vs. MVO"]["p_value"])
    p.zahl("p Markowitz gegen Gleichgewichtung", "0,97", test["MVO vs. EW"]["p_value"])
    signifikant = [k for k, v in test.items() if v["significant_holm"]]
    p.zahl("Vergleiche, die nach Holm die Fünf-Prozent-Schwelle überstehen", "ein", len(signifikant),
           notiz=", ".join(signifikant).replace(" vs. EW", " gegen Gleichgewichtung"))
    p.zahl("p nach Holm, Risk Parity gegen Gleichgewichtung", "0,022", test["Risk Parity vs. EW"]["p_holm"],
           notiz=f"auf vier Stellen {de(test['Risk Parity vs. EW']['p_holm'], 4)}")
    p.zahl("Korrelation Gleichgewichtung und Risk Parity", "0,993", s["Equal Weight"].corr(s["Risk Parity"]),
           notiz="Korrelation der Tagesrenditen")
    vola_diff = {(a, b): (s[a] - s[b]).std() * np.sqrt(HANDELSTAGE) * 100 for a, b in combinations(STRATEGIEN, 2)}
    p.zahl("Volatilität der Differenzreihe Gleichgewichtung − Risk Parity", "2,3",
           vola_diff[("Equal Weight", "Risk Parity")], "%",
           notiz="Standardabweichung der täglichen Renditedifferenz · √252")
    p.zahl("Volatilität der Differenzreihe Markowitz − Risk Parity", "10,4",
           vola_diff[("Markowitz MVO", "Risk Parity")], "%")
    dsr = log["significance"]["deflated_sharpe"]
    p.zahl("Versuche in der Deflated Sharpe Ratio", "vier", log["significance"]["n_strategies"])
    p.zahl("Zufallsschwelle (Sharpe, jährlich)", "0,067", next(iter(dsr.values()))["sr0_annual"],
           notiz="Formel 33, aus dem Laborprotokoll")
    werte = {k: v["deflated_sr"] for k, v in dsr.items()}
    p.zahl("Deflated Sharpe Ratio, kleinster Wert", "0,988", min(werte.values()), notiz=min(werte, key=werte.get))
    p.zahl("Deflated Sharpe Ratio, größter Wert", "0,997", max(werte.values()), notiz=max(werte, key=werte.get))

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("6.4 Sensibilität gegenüber Transaktionskosten")
    piv = kosten.sharpe_tabelle(kosten.tabelle(s, U))   # dieselbe Rechnung wie Abbildung 5
    p.zahl("angesetzter Kostensatz", "zehn", meta["transaction_cost"] * 10_000, "Basispunkte")
    p.zahl("untersuchte Spanne: von", "null", piv.index.min(), "Basispunkte")
    p.zahl("untersuchte Spanne: bis", "hundert", piv.index.max(), "Basispunkte")
    rf_unter = kosten.schnittpunkt(piv, "Random Forest")
    p.zahl("Random Forest fällt unter die Gleichgewichtung bei", "31", rf_unter, "bp", ungefaehr=True,
           notiz="Schnittpunkt der Kurven in Abbildung 5, linear zwischen den 5-bp-Stufen interpoliert")
    p.zahl("Markowitz fällt darunter bei", "20", kosten.schnittpunkt(piv, "Markowitz MVO"), "bp", ungefaehr=True,
           notiz="ebenso")
    # 1/N führt, sobald alle anderen unter 1/N gefallen sind: späteste Schnittstelle.
    ab = max(kosten.schnittpunkt(piv, n) or 0 for n in ["Random Forest", "Markowitz MVO", "Risk Parity"])
    p.zahl("ab hier hat die Gleichgewichtung den höchsten Sharpe-Quotienten", "31", ab, "bp", ungefaehr=True,
           notiz="spätester der drei Schnittpunkte")
    verlust = piv.loc[0] - piv.loc[100]
    p.zahl("Sharpe-Verlust über die Spanne, Random Forest", "0,187", verlust["Random Forest"],
           notiz="Sharpe-Quotient bei 0 bp minus bei 100 bp")
    p.zahl("Sharpe-Verlust über die Spanne, Markowitz", "0,079", verlust["Markowitz MVO"], notiz="ebenso")
    p.aussage("Markowitz hat weniger als halb so hohen Umschlag wie der Random Forest", "weniger als halb",
              regulaer["turnover_mvo"].mean() < regulaer["turnover_rf"].mean() / 2,
              f"{de(regulaer['turnover_mvo'].mean() * 100, 1)} % gegen {de(regulaer['turnover_rf'].mean() * 100, 1)} %",
              notiz="mittlerer monatlicher Umschlag nach dem Depotaufbau", woertlich=True)
    for name in ["Equal Weight", "Risk Parity"]:
        p.zahl(f"Sharpe-Verlust über die Spanne, {name}", "0,020", verlust[name], ungefaehr=True,
               toleranz=0.002, notiz="Sharpe-Quotient bei 0 bp minus bei 100 bp")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("6.5 Warum es keinen Gewinner gibt")
    r_tag = titelrenditen()
    C = (r_tag.cov() * HANDELSTAGE).values
    var_quer, cov_quer = diversifikation(C)            # Formel 6, wie in § 2.2
    abstand = (np.sqrt(gleichgewichtete_varianz(var_quer, cov_quer, len(TICKERS))) - np.sqrt(cov_quer)) * 100
    p.zahl("Abstand der Gleichgewichtung zur Diversifikationsgrenze", "0,88", abstand, "Pp",
           notiz="wie in § 2.2: Volatilität bei 15 Titeln minus √C̄ (Formel 6)")
    p.zahl("Länge des Zeitraums", "zehn", (s.index[-1] - s.index[0]).days / 365.25, "Jahre",
           ungefaehr=True, toleranz=0.5, notiz=f"{s.index[0]:%d.%m.%Y} bis {s.index[-1]:%d.%m.%Y}")
    # Beobachtungsdauer, um eine Renditedifferenz Δ zweier typischer Titel abzusichern
    # (Formel 17 für die Differenz): 1,96 · σ̄·√(2(1 − ρ̄)) / √h = Δ  →  h = (1,96·σ̄/Δ)² · 2(1 − ρ̄),
    # σ̄ = mittlere Volatilität, ρ̄ = mittlere Paarkorrelation der fünfzehn Titel.
    sig = r_tag.std() * np.sqrt(HANDELSTAGE)
    rho_q = r_tag.corr().values[np.triu_indices(len(TICKERS), k=1)].mean()
    h_noetig = (1.96 * sig.mean() / 0.05) ** 2 * 2 * (1 - rho_q)
    p.zahl("nötige Beobachtungsdauer", "120", h_noetig, "Jahre", ungefaehr=True,
           notiz=f"h = (1,96 · σ̄ ÷ Δ)² · 2(1 − ρ̄) mit σ̄ = {de(sig.mean() * 100, 1)} %, Δ = 5 Pp, ρ̄ = {de(rho_q, 2)}")
    p.offen("Annahme: zwei typische Titel, Renditedifferenz", "5 Pp",
            "Annahme des Gedankenexperiments; „typisch“ = mittlere Volatilität und mittlere Korrelation")
    p.merke("Annahme: zwei typische Titel, Renditedifferenz", "fünf Prozentpunkten")
    # Gleichgewichtung auf Markowitz' Volatilität gehebelt (Hebel zum risikofreien Zins):
    hebel = K.vola["Markowitz MVO"] / K.vola["Equal Weight"]
    gehebelt = hebel * s["Equal Weight"] + (1 - hebel) * rf / HANDELSTAGE
    p.zahl("Volatilität von Markowitz", "21,0", K.vola["Markowitz MVO"], "%")
    p.zahl("Rendite der Gleichgewichtung, ungehebelt", "20,03", K.cagr["Equal Weight"], "%")
    p.zahl("… auf Markowitz' Volatilität gehebelt", "22,98", cagr(gehebelt) * 100, "%",
           notiz=f"Hebel {de(hebel, 3)} = {de(K.vola['Markowitz MVO'], 2)} % ÷ {de(K.vola['Equal Weight'], 2)} %, "
                 "Rest zum risikofreien Zins; gehebelte Tagesrenditen aufgezinst")
    vorsprung = K.cagr["Markowitz MVO"] - K.cagr["Equal Weight"]
    rest = K.cagr["Markowitz MVO"] - cagr(gehebelt) * 100
    p.zahl("ursprünglicher Renditevorsprung von Markowitz", "3,11", vorsprung, "Pp",
           notiz="CAGR Markowitz minus CAGR Gleichgewichtung")
    p.zahl("Vorsprung nach Angleichen des Risikos", "0,15", rest, "Pp",
           notiz="CAGR Markowitz minus CAGR der gehebelten Gleichgewichtung")
    p.zahl("Anteil des Vorsprungs, der verschwindet", "95", (1 - rest / vorsprung) * 100, "%",
           ungefaehr=True, notiz=f"1 − {de(rest, 2)} ÷ {de(vorsprung, 2)}")
    p.zahl("entspricht einer Sharpe-Differenz von", "0,0059", test["MVO vs. EW"]["diff_annual"],
           notiz="Differenz der ungerundeten Sharpe-Quotienten, aus dem Laborprotokoll")
    p.zahl("Handelstage mit Markowitz in Führung (aus § 6.1)", "86,8", vorn["Markowitz MVO"], "%")
    roll = rollierender_sharpe(s).dropna()            # dieselbe Rechnung wie Abbildung A4
    p.zahl("Zwölfmonatsfenster mit Markowitz risikoadjustiert vorn", "50,1",
           (roll["Markowitz MVO"] > roll["Equal Weight"]).mean() * 100, "%")
    t_rf = test["RF vs. EW"]
    p.zahl("Vorsprung des Random Forest vor der Gleichgewichtung", "0,035", t_rf["diff_annual"])
    p.zahl("sein Standardfehler", "0,121", t_rf["se_annual"])
    # Wie viele Jahre bräuchte es, damit t = 1,96? t wächst mit √T.
    jahre = len(s) * (1.96 / abs(t_rf["statistic"])) ** 2 / HANDELSTAGE
    p.zahl("Daten, um ihn abzusichern", "450", jahre, "Jahre", ungefaehr=True,
           notiz=f"t wächst mit √T: 2495 Tage · (1,96 ÷ {de(abs(t_rf['statistic']), 3)})² ÷ 252")
    w1n = np.ones(len(TICKERS)) / len(TICKERS)
    rb = w1n * (C @ w1n) / (w1n @ C @ w1n) * 100      # Risikobeiträge im 1/N-Depot (Formel 19), in %
    p.zahl("Risikobeitrag im gleichgewichteten Depot, kleinster", "3,6", rb.min(), "%",
           notiz=TICKERS[int(np.argmin(rb))])
    p.zahl("Risikobeitrag im gleichgewichteten Depot, größter", "11,3", rb.max(), "%",
           notiz=TICKERS[int(np.argmax(rb))])
    p.quelle("Qians 60/40-Beispiel: Risikobeiträge unterscheiden sich um mehr als das …", "Dreizehnfache",
             "Qian 2005, S. 1; wie § 2.6, Fn 48")
    p.zahl("effektive Titelzahl von Risk Parity", "14,05", neff["Risk Parity"].mean())
    p.zahl("… von", "fünfzehn", len(TICKERS))
    p.zahl("Renditekorrelation Risk Parity und Gleichgewichtung", "0,993", s["Equal Weight"].corr(s["Risk Parity"]))
    p.zahl("Risk Parity spart an Schwankung", "einen", K.vola["Equal Weight"] - K.vola["Risk Parity"], "Pp",
           ungefaehr=True, toleranz=0.2, notiz="Volatilität Gleichgewichtung minus Risk Parity")
    p.zahl("… und kostet an Rendite", "drei", K.cagr["Equal Weight"] - K.cagr["Risk Parity"], "Pp",
           ungefaehr=True, toleranz=0.2, notiz="CAGR Gleichgewichtung minus Risk Parity")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("6.6 Grenzen der Untersuchung")
    jahr = (1 + s["Equal Weight"]).groupby(s.index.year).prod() - 1   # Jahresrendite des Universums (1/N)
    p.zahl("Jahre, die für das Anlageuniversum positiv schlossen", "Acht", int((jahr > 0).sum()))
    p.zahl("von Jahren", "zehn", len(jahr))
    p.zahl("risikofreier Zins", "vier", rf * 100, "%")
    # Bei welchem Zins haben Markowitz und 1/N denselben Sharpe-Quotienten?
    # (μ_M − r)/σ_M = (μ_E − r)/σ_E  →  r = (μ_E σ_M − μ_M σ_E)/(σ_M − σ_E), je Tag.
    mM, mE = s["Markowitz MVO"].mean(), s["Equal Weight"].mean()
    sM, sE = s["Markowitz MVO"].std(), s["Equal Weight"].std()
    p.zahl("Zins, unter dem Markowitz und Gleichgewichtung die Plätze tauschen", "3,4",
           (mE * sM - mM * sE) / (sM - sE) * HANDELSTAGE * 100, "%",
           notiz="Zins, bei dem beide Sharpe-Quotienten gleich sind")
    p.zahl("Zins, unter dem Risk Parity nach Holm nicht mehr signifikant zurückliegt", "0,8", zinsschwelle(s),
           "%", ungefaehr=True, toleranz=0.1,
           notiz="Test von Ledoit/Wolf mit Holm für eine Reihe von Zinssätzen wiederholt")
    p.zahl("Versuche in der Deflated Sharpe Ratio", "vier", log["significance"]["n_strategies"])
    p.zahl("Standardwerte im Universum", "fünfzehn", len(TICKERS))
    # Beitrag jedes Titels zum Renditeabstand: Gewichtsdifferenz × Titelrendite, je Tag,
    # gemittelt und auf ein Jahr hochgerechnet (arithmetisch).
    tage = s.index
    r_t = titelrenditen().loc[tage]
    tg = {n: tagesgewichte(G[n], tage) for n in STRATEGIEN}
    kosten_je_tag = {}
    for name in STRATEGIEN:                           # Kosten so verbuchen wie der Backtest
        kk = pd.Series(0.0, index=tage)
        for termin in U.index:
            erster = tage[tage > termin]
            if len(erster):
                kk.loc[erster[0]] = U.loc[termin, UMSCHLAG_SPALTE[name]] * meta["transaction_cost"]
        kosten_je_tag[name] = kk

    def beitrag(a, b):
        """Differenz der arithmetischen Jahresrenditen je Titel, in Prozentpunkten."""
        return ((tg[a] - tg[b]) * r_t).mean() * HANDELSTAGE * 100

    for name, t_nv in [("Markowitz MVO", "6,8"), ("Random Forest", "5,2")]:
        b_ = beitrag(name, "Equal Weight")
        # Kontrolle: Beiträge minus Kostenunterschied = Differenz der gespeicherten Renditen
        kontrolle = (s[name] - s["Equal Weight"]).mean() * HANDELSTAGE * 100
        rechnung = b_.sum() - (kosten_je_tag[name] - kosten_je_tag["Equal Weight"]).mean() * HANDELSTAGE * 100
        assert abs(kontrolle - rechnung) < 0.01, "Zerlegung stimmt nicht mit den Renditen überein"
        p.zahl(f"Beitrag von NVIDIA zum Vorsprung vor 1/N, {name}", t_nv, b_["NVDA"], "Pp",
               notiz="Gewichtsdifferenz zu 1/N mal Tagesrendite von NVIDIA, gemittelt, mal 252")
    p.zahl("übrige Titel", "vierzehn", len(TICKERS) - 1)
    for name, t_rest in [("Markowitz MVO", "3,4"), ("Random Forest", "2,9")]:
        p.zahl(f"Rückstand auf den übrigen vierzehn Titeln, {name}", t_rest,
               -beitrag(name, "Equal Weight").drop("NVDA").sum(), "Pp")
    b_rp = beitrag("Equal Weight", "Risk Parity")
    anteil_nv = b_rp["NVDA"] / b_rp.sum() * 100
    p.aussage("Anteil von NVIDIA am Rückstand von Risk Parity", "drei Fünfteln", abs(anteil_nv - 60) <= 2,
              f"{de(anteil_nv, 1)} %", notiz="drei Fünftel = 60 %", woertlich=True)
    p.aussage("Titel, den Risk Parity am geringsten gewichtet", "am geringsten",
              G["Risk Parity"].mean().idxmin() == "NVDA", G["Risk Parity"].mean().idxmin(),
              notiz=f"mittleres Gewicht {de(G['Risk Parity'].mean().min() * 100, 1)} %", woertlich=True)
    p.aussage("schwankungsstärkster Titel", "schwankungsstärksten", sig.idxmax() == "NVDA",
              sig.idxmax(), notiz=f"Volatilität {de(sig.max() * 100, 1)} %", woertlich=True)
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
