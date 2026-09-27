"""
Kapitel 6 — Ergebnisse und Diskussion: jede eigene Zahl nachgerechnet.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Kapitel 6 berichtet, wie die vier Strategien abgeschnitten haben, und ordnet es
ein. Fast jede Zahl dort ist ein Ergebnis des großen Backtests. Dieses Skript
rechnet sie aus den gespeicherten Ergebnisdateien nach:

  output/daily_returns.csv     Tagesrenditen → Rendite, Volatilität, Sharpe, Drawdown
  output/weights_*.csv         Monatsgewichte → effektive Titelzahl, NVIDIA-Anteil
  output/turnover.csv          Umschlag → Kosten, Sensitivität (§ 6.4)
  output/experiment_log.json   Signifikanztests mit voller Genauigkeit (§ 6.3)

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

from portfolio.metrics import (cagr, annualized_vol, sharpe_ratio,   # noqa: E402
                               max_drawdown, calmar_ratio)
from portfolio.significance import sharpe_difference_test, holm_bonferroni   # noqa: E402
import abbildung_kostensensitivitaet as kosten        # noqa: E402

# Reihenfolge und Beschriftung der Signifikanztabelle in § 6.3 → Schlüssel im Laufprotokoll.
TABELLE_63 = [("RF vs. Markowitz", "RF vs. MVO"), ("RF vs. Equal Weight", "RF vs. EW"),
              ("RF vs. Risk Parity", "RF vs. RP"), ("Markowitz vs. Equal Weight", "MVO vs. EW"),
              ("Markowitz vs. Risk Parity", "MVO vs. RP"),
              ("Equal Weight vs. Risk Parity", "Risk Parity vs. EW")]


def tagesgewichte(g: pd.DataFrame, tage: pd.DatetimeIndex) -> pd.DataFrame:
    """Gewicht je Handelstag wie in portfolio/backtest.py: Das am Monatsende
    festgelegte Gewicht gilt ab dem folgenden Handelstag bis zum nächsten Monatsende."""
    termine = g.index
    stelle = termine.searchsorted(tage, side="left") - 1      # letzter Termin VOR dem Tag
    return pd.DataFrame(g.values[stelle], index=tage, columns=g.columns)


def pruefe() -> Pruefung:
    p = Pruefung("Kapitel 6 — Ergebnisse und Diskussion")
    s = strategierenditen()
    log = protokoll()
    rf = log["experiment_meta"]["risk_free_rate"]
    G = gewichte()
    U = umschlag()
    wert = (1 + s).cumprod()                          # Depotwert je Tag, Start 1 Euro

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("6.1 Rendite und Risikokennzahlen")
    tabelle = {"Random Forest": ("22,12", "19,03", "0,936", "−33,77", "0,655"),
               "Markowitz MVO": ("23,14", "21,00", "0,906", "−34,28", "0,675"),
               "Equal Weight":  ("20,03", "17,56", "0,900", "−33,25", "0,602"),
               "Risk Parity":   ("16,92", "16,45", "0,790", "−32,44", "0,522")}
    k = {}
    for name, (c, v, sh, dd, cal) in tabelle.items():
        r = s[name]
        k[name] = dict(cagr=cagr(r) * 100, vola=annualized_vol(r) * 100,
                       sharpe=sharpe_exakt(name), dd=max_drawdown(r) * 100, calmar=calmar_ratio(r))
        kurz = KURZ[name]
        p.zahl(f"{kurz}: CAGR", c, k[name]["cagr"], "%")
        p.zahl(f"{kurz}: Volatilität", v, k[name]["vola"], "%")
        p.zahl(f"{kurz}: Sharpe", sh, k[name]["sharpe"])
        p.zahl(f"{kurz}: maximaler Drawdown", dd, k[name]["dd"], "%")
        p.zahl(f"{kurz}: Calmar", cal, k[name]["calmar"])
    for name, text in [("Markowitz MVO", "7,85"), ("Random Forest", "7,23"),
                       ("Equal Weight", "6,10"), ("Risk Parity", "4,70")]:
        p.zahl(f"Endvermögen aus 1 Euro, {KURZ[name]}", text, wert[name].iloc[-1], "Euro")
    K = pd.DataFrame(k).T
    p.aussage("höchste Rendite", "Markowitz", K.cagr.idxmax() == "Markowitz MVO", KURZ[K.cagr.idxmax()])
    rang = list(K.sharpe.sort_values(ascending=False).index).index("Markowitz MVO") + 1
    p.aussage("Markowitz risikoadjustiert (Sharpe)", "zweite Stelle", rang == 2, f"{rang}. Stelle",
              notiz="Reihenfolge: " + " > ".join(f"{KURZ[n]} {de(K.sharpe[n], 3)}"
                                                    for n in K.sharpe.sort_values(ascending=False).index))
    p.aussage("höchste Volatilität", "Markowitz", K.vola.idxmax() == "Markowitz MVO", KURZ[K.vola.idxmax()])
    p.zahl("Volatilität Markowitz", "21,0", K.vola["Markowitz MVO"], "%")
    p.aussage("höchster Sharpe-Quotient", "Random Forest", K.sharpe.idxmax() == "Random Forest",
              KURZ[K.sharpe.idxmax()])
    p.aussage("geringste Volatilität, flachster Drawdown", "Risk Parity",
              K.vola.idxmin() == "Risk Parity" and K.dd.idxmax() == "Risk Parity",
              f"{KURZ[K.vola.idxmin()]}, {KURZ[K.dd.idxmax()]}")
    vorn = wert.idxmax(axis=1).value_counts(normalize=True) * 100   # wer liegt je Tag vorn?
    for name, text in [("Markowitz MVO", "86,8"), ("Random Forest", "11,8"), ("Equal Weight", "1,4")]:
        p.zahl(f"Anteil der Tage in Führung, {KURZ[name]}", text, vorn.get(name, 0.0), "%")
    tiefpunkt = {n: ((wert[n] / wert[n].cummax()) - 1).idxmin() for n in STRATEGIEN}
    for name, datum in [("Random Forest", "2020-03-23"), ("Equal Weight", "2020-03-23"),
                        ("Risk Parity", "2020-03-23"), ("Markowitz MVO", "2022-09-30")]:
        p.aussage(f"Tag des maximalen Drawdowns, {KURZ[name]}", datum,
                  str(tiefpunkt[name].date()) == datum, str(tiefpunkt[name].date()))
    p.aussage("führt nach Calmar", "Markowitz", K.calmar.idxmax() == "Markowitz MVO", KURZ[K.calmar.idxmax()])
    p.zahl("Markowitz schwankt stärker als RF um", "10",
           (K.vola["Markowitz MVO"] / K.vola["Random Forest"] - 1) * 100, "%", ungefaehr=True,
           notiz="Text: „gut zehn Prozent“")
    p.zahl("tiefster Absturz Markowitz tiefer um", "1,5",
           (K.dd["Markowitz MVO"] / K.dd["Random Forest"] - 1) * 100, "%", ungefaehr=True,
           notiz="Text: „anderthalb Prozent“")
    p.zahl("Kostensatz (Abb. 4)", "10", log["experiment_meta"]["transaction_cost"] * 10_000, "bp")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("6.2 Struktur der Portfolios")
    neff = {n: 1 / (G[n] ** 2).sum(axis=1) for n in STRATEGIEN}   # Formel 35 je Termin
    for name, text in [("Markowitz MVO", "5,88"), ("Random Forest", "6,16"),
                       ("Risk Parity", "14,05"), ("Equal Weight", "15,00")]:
        p.zahl(f"effektive Titelzahl, {KURZ[name]}", text, neff[name].mean())
    for name in ["Markowitz MVO", "Random Forest"]:
        leer = (G[name] < 0.0005).sum(axis=1).mean()
        p.zahl(f"leere Titel im Mittel, {KURZ[name]}", "7,5", leer, ungefaehr=True, toleranz=1.0,
               notiz="Text: „rund die Hälfte des Universums leer“ (7,5 von 15)")
        an_grenze = (G[name] >= 0.1995).sum(axis=1).mean()
        p.zahl(f"Titel an der 20-%-Grenze, {KURZ[name]}", "3", an_grenze, ungefaehr=True, toleranz=0.5,
               notiz="Text: „etwa drei Titel“")
    p.zahl("NVIDIA an der Obergrenze (Markowitz)", "81", int((G["Markowitz MVO"]["NVDA"] >= 0.1999).sum()),
           "Monate")
    p.zahl("von Terminen", "119", len(G["Markowitz MVO"]))
    r_tag = titelrenditen()
    mu, sig = r_tag.mean() * HANDELSTAGE, r_tag.std() * np.sqrt(HANDELSTAGE)
    halb = 1.96 * sig / np.sqrt(len(r_tag) / HANDELSTAGE)
    trennbar = {t for i, j in combinations(TICKERS, 2) if abs(mu[i] - mu[j]) > halb[i] + halb[j] for t in (i, j)}
    in_allen = [t for t in TICKERS if all(t in (i, j) for i, j in combinations(TICKERS, 2)
                                          if abs(mu[i] - mu[j]) > halb[i] + halb[j])]
    p.aussage("einziger trennbarer Titel (§ 2.5)", "NVIDIA", in_allen == ["NVDA"],
              f"in allen trennbaren Paaren: {', '.join(in_allen)}")
    regulaer = U.iloc[1:]
    for name, text in [("Random Forest", "29,0"), ("Markowitz MVO", "13,1")]:
        p.zahl(f"mittlerer Umschlag, {KURZ[name]}", text, regulaer[UMSCHLAG_SPALTE[name]].mean() * 100, "%")
    klein = [regulaer[UMSCHLAG_SPALTE[n]].mean() * 100 for n in ["Equal Weight", "Risk Parity"]]
    p.aussage("Umschlag der beiden übrigen", "gut zwei Prozent", all(2.0 <= v < 2.5 for v in klein),
              " / ".join(de(v, 2) + " %" for v in klein))
    p.zahl("effektive Titelzahl Markowitz, kleinste", "5,02", neff["Markowitz MVO"].min())
    p.zahl("effektive Titelzahl Markowitz, größte", "8,48", neff["Markowitz MVO"].max())

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("6.3 Ergebnisse der Signifikanzprüfung")
    test = log["significance"]["sharpe_difference_tests"]
    soll = {"RF vs. MVO": ("0,029", "0,178", "0,165", "0,866", "1,000"),
            "RF vs. EW": ("0,035", "0,121", "0,291", "0,773", "1,000"),
            "RF vs. RP": ("0,146", "0,130", "1,122", "0,257", "1,000"),
            "MVO vs. EW": ("0,006", "0,146", "0,040", "0,971", "1,000"),
            "MVO vs. RP": ("0,116", "0,160", "0,724", "0,489", "1,000"),
            "Risk Parity vs. EW": ("0,110", "0,038", "2,874", "0,0036", "0,0216")}
    for zeile, schluessel in TABELLE_63:
        t = test[schluessel]
        d_, se_, st_, p_, h_ = soll[schluessel]
        p.zahl(f"{zeile}: Δ", d_, abs(t["diff_annual"]))
        p.zahl(f"{zeile}: s(Δ)", se_, t["se_annual"])
        p.zahl(f"{zeile}: t", st_, abs(t["statistic"]))
        p.zahl(f"{zeile}: p", p_, t["p_value"])
        p.zahl(f"{zeile}: p nach Holm", h_, t["p_holm"])
    p.zahl("im Text: p RF gegen 1/N", "0,77", test["RF vs. EW"]["p_value"])
    p.zahl("im Text: p RF gegen Markowitz", "0,87", test["RF vs. MVO"]["p_value"])
    p.zahl("im Text: p Markowitz gegen 1/N", "0,97", test["MVO vs. EW"]["p_value"])
    p.zahl("im Text: p nach Holm, RP gegen 1/N", "0,022", test["Risk Parity vs. EW"]["p_holm"],
           notiz="Tabelle und § 7.1 nennen 0,0216")
    signifikant = [k_ for k_, v in test.items() if v["significant_holm"]]
    p.aussage("signifikant nach Holm", "nur RP gegen 1/N", signifikant == ["Risk Parity vs. EW"],
              ", ".join(signifikant))
    p.zahl("Faktor der Standardfehler", "4,2", test["MVO vs. RP"]["se_annual"] / test["Risk Parity vs. EW"]["se_annual"])
    p.zahl("Korrelation 1/N und Risk Parity", "0,993", s["Equal Weight"].corr(s["Risk Parity"]))
    vola_diff = lambda a, b: (s[a] - s[b]).std() * np.sqrt(HANDELSTAGE) * 100
    p.zahl("Volatilität der Differenz 1/N − RP", "2,3", vola_diff("Equal Weight", "Risk Parity"), "%")
    p.zahl("Volatilität der Differenz MVO − RP", "10,4", vola_diff("Markowitz MVO", "Risk Parity"), "%")
    dsr = log["significance"]["deflated_sharpe"]
    p.zahl("Zufallsschwelle der DSR", "0,067", next(iter(dsr.values()))["sr0_annual"])
    werte = [v["deflated_sr"] for v in dsr.values()]
    p.zahl("DSR, kleinster Wert", "0,988", min(werte))
    p.zahl("DSR, größter Wert", "0,997", max(werte))

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("6.4 Sensibilität gegenüber Transaktionskosten")
    piv = kosten.sharpe_tabelle(kosten.tabelle(s, U))
    p.zahl("Kostensatz im Lauf", "10", log["experiment_meta"]["transaction_cost"] * 10_000, "bp")
    p.aussage("untersuchte Spanne", "0 bis 100 bp", (piv.index.min(), piv.index.max()) == (0, 100),
              f"{piv.index.min()} bis {piv.index.max()} bp")
    p.zahl("RF fällt unter 1/N bei", "31", kosten.schnittpunkt(piv, "Random Forest"), "bp", ungefaehr=True)
    p.zahl("Markowitz fällt unter 1/N bei", "20", kosten.schnittpunkt(piv, "Markowitz MVO"), "bp", ungefaehr=True)
    # 1/N führt, sobald alle anderen unter 1/N gefallen sind: späteste Schnittstelle.
    ab = max(kosten.schnittpunkt(piv, n) or 0 for n in ["Random Forest", "Markowitz MVO", "Risk Parity"])
    p.zahl("1/N führt das Feld an ab", "31", ab, "bp", ungefaehr=True,
           notiz="danach liegt keine Strategie mehr über 1/N")
    verlust = piv.loc[0] - piv.loc[100]
    p.zahl("Sharpe-Verlust RF über die Spanne", "0,187", verlust["Random Forest"])
    p.zahl("Sharpe-Verlust Markowitz", "0,079", verlust["Markowitz MVO"])
    for name in ["Equal Weight", "Risk Parity"]:
        p.zahl(f"Sharpe-Verlust {KURZ[name]}", "0,020", verlust[name], ungefaehr=True, toleranz=0.002)
    p.aussage("Markowitz: weniger als halber Umschlag des RF", "ja",
              regulaer["turnover_mvo"].mean() < regulaer["turnover_rf"].mean() / 2,
              f"{de(regulaer['turnover_mvo'].mean() * 100, 1)} % gegen {de(regulaer['turnover_rf'].mean() * 100, 1)} %")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("6.5 Warum es keinen Gewinner gibt")
    C = (r_tag.cov() * HANDELSTAGE).values
    N = len(TICKERS)
    w1n = np.ones(N) / N
    grenze = np.sqrt(C[~np.eye(N, dtype=bool)].mean())
    p.zahl("Abstand 1/N zur Diversifikationsgrenze", "0,88", (np.sqrt(w1n @ C @ w1n) - grenze) * 100, "Pp")
    # Beobachtungsdauer, um 5 Pp Renditedifferenz zweier typischer Titel abzusichern:
    # h = (1,96 · σ̄ / Δ)² · 2(1 − ρ̄), σ̄ = mittlere Volatilität, ρ̄ = mittlere Paarkorrelation.
    rho_q = r_tag.corr().values[np.triu_indices(N, k=1)].mean()
    h_noetig = (1.96 * sig.mean() / 0.05) ** 2 * 2 * (1 - rho_q)
    p.zahl("nötige Beobachtungsdauer", "120", h_noetig, "Jahre", ungefaehr=True)
    # Gleichgewichtung auf Markowitz' Volatilität gehebelt (Hebel zum risikofreien Zins):
    hebel = K.vola["Markowitz MVO"] / K.vola["Equal Weight"]
    gehebelt = hebel * s["Equal Weight"] + (1 - hebel) * rf / HANDELSTAGE
    p.zahl("Volatilität Markowitz", "21,0", K.vola["Markowitz MVO"], "%")
    p.zahl("1/N gehebelt, CAGR", "22,98", cagr(gehebelt) * 100, "%")
    p.zahl("1/N ungehebelt, CAGR", "20,03", K.cagr["Equal Weight"], "%")
    vorsprung = K.cagr["Markowitz MVO"] - K.cagr["Equal Weight"]
    rest = K.cagr["Markowitz MVO"] - cagr(gehebelt) * 100
    p.zahl("Vorsprung Markowitz", "3,11", vorsprung, "Pp")
    p.zahl("Vorsprung nach Angleichung", "0,15", rest, "Pp")
    p.zahl("Anteil, der verschwindet", "95", (1 - rest / vorsprung) * 100, "%", ungefaehr=True)
    p.zahl("entspricht Sharpe-Differenz", "0,0059", test["MVO vs. EW"]["diff_annual"],
           notiz="Testwert aus § 6.3, nicht die Differenz der gerundeten Tabellenwerte")
    p.zahl("Tage in Führung, Markowitz", "86,8", vorn["Markowitz MVO"], "%")
    rolling = {n: s[n].rolling(HANDELSTAGE).apply(lambda x: sharpe(pd.Series(x), rf), raw=True).dropna()
               for n in ["Markowitz MVO", "Equal Weight"]}
    p.zahl("Zwölfmonatsfenster mit Markowitz vorn", "50,1",
           (rolling["Markowitz MVO"] > rolling["Equal Weight"]).mean() * 100, "%")
    t_rf = test["RF vs. EW"]
    p.zahl("Vorsprung RF vor 1/N", "0,035", t_rf["diff_annual"])
    p.zahl("sein Standardfehler", "0,121", t_rf["se_annual"])
    jahre = len(s) * (1.96 / abs(t_rf["statistic"])) ** 2 / HANDELSTAGE
    p.zahl("Daten, um ihn abzusichern", "450", jahre, "Jahre", ungefaehr=True)
    rb = w1n * (C @ w1n) / (w1n @ C @ w1n) * 100      # Risikobeiträge im 1/N-Depot (Formel 19)
    p.zahl("Risikobeitrag, kleinster", "3,6", rb.min(), "%", notiz=TICKERS[int(np.argmin(rb))])
    p.zahl("Risikobeitrag, größter", "11,3", rb.max(), "%", notiz=TICKERS[int(np.argmax(rb))])
    p.offen("Qians 60/40-Beispiel: mehr als das Dreizehnfache", "13", "Zahl aus der Quelle")
    p.zahl("effektive Titelzahl Risk Parity", "14,05", neff["Risk Parity"].mean())
    p.zahl("Renditekorrelation RP und 1/N", "0,993", s["Equal Weight"].corr(s["Risk Parity"]))
    p.zahl("Risk Parity spart Schwankung", "1", K.vola["Equal Weight"] - K.vola["Risk Parity"], "Pp",
           ungefaehr=True, toleranz=0.2)
    p.zahl("… und kostet Rendite", "3", K.cagr["Equal Weight"] - K.cagr["Risk Parity"], "Pp",
           ungefaehr=True, toleranz=0.2)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("6.6 Grenzen der Untersuchung")
    jahr = (1 + s["Equal Weight"]).groupby(s.index.year).prod() - 1
    p.zahl("positive Kalenderjahre (1/N)", "8", int((jahr > 0).sum()), notiz=f"von {len(jahr)}")
    p.zahl("risikofreier Zins", "4", rf * 100, "%")
    # Ab welchem Zins haben Markowitz und 1/N denselben Sharpe-Quotienten?
    # (μ_M − r)/σ_M = (μ_E − r)/σ_E  →  r = (μ_E σ_M − μ_M σ_E)/(σ_M − σ_E), je Tag.
    mM, mE = s["Markowitz MVO"].mean(), s["Equal Weight"].mean()
    sM, sE = s["Markowitz MVO"].std(), s["Equal Weight"].std()
    p.zahl("Zins, bei dem MVO und 1/N die Plätze tauschen", "3,4",
           (mE * sM - mM * sE) / (sM - sE) * HANDELSTAGE * 100, "%")
    p.zahl("Zinsschwelle des signifikanten Befunds", "0,8", zinsschwelle(s), "%", ungefaehr=True, toleranz=0.1)
    tage = s.index
    r_t = titelrenditen().loc[tage]
    kosten_je_tag = {}
    for name in STRATEGIEN:
        kk = pd.Series(0.0, index=tage)
        for termin in U.index:
            erster = tage[tage > termin]
            if len(erster):
                kk.loc[erster[0]] = U.loc[termin, UMSCHLAG_SPALTE[name]] * log["experiment_meta"]["transaction_cost"]
        kosten_je_tag[name] = kk
    tg = {n: tagesgewichte(G[n], tage) for n in STRATEGIEN}

    def beitrag(a, b):
        """Differenz der arithmetischen Jahresrenditen je Titel, in Prozentpunkten."""
        return ((tg[a] - tg[b]) * r_t).mean() * HANDELSTAGE * 100

    for name, t_nv, t_rest in [("Markowitz MVO", "6,8", "3,4"), ("Random Forest", "5,2", "2,9")]:
        b_ = beitrag(name, "Equal Weight")
        # Kontrolle: Beiträge minus Kostenunterschied = Differenz der gespeicherten Renditen
        kontrolle = (s[name] - s["Equal Weight"]).mean() * HANDELSTAGE * 100
        rechnung = b_.sum() - (kosten_je_tag[name] - kosten_je_tag["Equal Weight"]).mean() * HANDELSTAGE * 100
        assert abs(kontrolle - rechnung) < 0.01, "Zerlegung stimmt nicht mit den Renditen überein"
        p.zahl(f"Beitrag NVIDIA zum Vorsprung, {KURZ[name]}", t_nv, b_["NVDA"], "Pp")
        p.zahl(f"Rückstand auf den übrigen 14, {KURZ[name]}", t_rest, -b_.drop("NVDA").sum(), "Pp")
    b_rp = beitrag("Equal Weight", "Risk Parity")
    p.zahl("Anteil NVIDIA am Rückstand von RP", "60", b_rp["NVDA"] / b_rp.sum() * 100, "%",
           ungefaehr=True, toleranz=2, notiz="Text: „zu drei Fünfteln“")
    p.aussage("RP gewichtet NVIDIA am geringsten", "ja", G["Risk Parity"].mean().idxmin() == "NVDA",
              G["Risk Parity"].mean().idxmin())
    p.aussage("NVIDIA ist der schwankungsstärkste Titel", "ja", sig.idxmax() == "NVDA", sig.idxmax())
    return p


def zinsschwelle(s: pd.DataFrame) -> float:
    """Zinssatz in Prozent, unter dem 'Risk Parity < 1/N' nach Holm nicht mehr
    signifikant ist. Test wie im Backtest (4999 Ziehungen, alle sechs Paare, Holm);
    zwischen dem letzten signifikanten und dem ersten nicht signifikanten Zins
    wird linear interpoliert."""
    paare = [("Random Forest", "Markowitz MVO"), ("Random Forest", "Equal Weight"),
             ("Random Forest", "Risk Parity"), ("Markowitz MVO", "Equal Weight"),
             ("Markowitz MVO", "Risk Parity"), ("Risk Parity", "Equal Weight")]
    punkte = []
    for zins in (0.010, 0.009, 0.008, 0.007):
        pw = [sharpe_difference_test(s[a], s[b], rf=zins, n_boot=4999)["p_value"] for a, b in paare]
        punkte.append((zins, holm_bonferroni(pw)["p_adjusted"][-1]))
    for (z1, h1), (z2, h2) in zip(punkte, punkte[1:]):
        if h1 < 0.05 <= h2:
            return (z2 + (z1 - z2) * (0.05 - h2) / (h1 - h2)) * 100
    return float("nan")


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
