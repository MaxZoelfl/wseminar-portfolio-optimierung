"""
Kapitel 2 — Theoretische Grundlagen: jede Zahl des Kapitels, in Textreihenfolge.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Kapitel 2 erklärt die Theorie (Diversifikation, Effizienzrand, Sharpe-Quotient,
Schätzfehler) und belegt sie jeweils an den fünfzehn Aktien der Arbeit. Dieses
Skript rechnet all diese Beispielzahlen noch einmal aus und vergleicht sie mit
dem Text. Es braucht dafür nur die eingefrorenen Kurse und die Tagesrenditen
der Strategien, keinen neuen Backtest.

Die Formeln kommen aus portfolio/theory_plots.py — denselben Funktionen, mit
denen auch die Abbildungen 1 bis 3, A1 und A2 gezeichnet werden.

Wichtig für das Verständnis: Die Theoriezahlen in § 2.2, § 2.3 und § 2.5
beziehen sich auf den GESAMTEN Auswertungszeitraum (2495 Handelstage,
02.02.2015 bis 30.12.2024), sind also "im Rückblick" geschätzt. Die
Strategien im Backtest sehen dagegen immer nur die jeweils letzten drei Jahre.
Das ist auch der Grund, warum § 2.6 (Shrinkage) in den 119 Schätzfenstern des
Backtests rechnet und nicht im Gesamtzeitraum.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_2_theorie.py          # rund 15 Sekunden
"""

import inspect
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from scipy.stats import norm                          # noqa: E402  (Quantile der Normalverteilung)
from sklearn.covariance import LedoitWolf             # noqa: E402  (derselbe Schätzer wie im Backtest)

from portfolio import metrics                         # noqa: E402
from portfolio.optimizers import MarkowitzLedoitWolf, RiskParityPortfolio   # noqa: E402
from portfolio.theory_plots import (jahreskennzahlen, varianzminimale_mischung,   # noqa: E402
                                    zwei_anlagen_varianz, diversifikation,
                                    gleichgewichtete_varianz, konfidenzintervalle,
                                    effizienzrand_geschlossen)


def zeitraum(p: Pruefung, r: pd.DataFrame) -> None:
    """Die Bildunterschriften nennen „Tagesdaten 2015–2024“: erstes und letztes
    Jahr der Renditen, aus denen die Abbildung gezeichnet ist."""
    p.aussage("Datenzeitraum der Abbildung", "2015–2024",
              (r.index[0].year, r.index[-1].year) == (2015, 2024),
              f"{r.index[0]:%d.%m.%Y} bis {r.index[-1]:%d.%m.%Y}", woertlich=True)


def shrinkage_in_den_fenstern():
    """§ 2.6: Ledoit-Wolf in jedem der 119 Schätzfenster, genau wie im Backtest.

    Je Termin: Schrumpfungsintensität δ (Formel 18) und die Frage, wie weit die
    Schrumpfung die Gewichte verschiebt. Gemessen an den beiden Strategien, die
    mit der Kovarianzmatrix optimieren und sich ohne gespeicherte Prognosen
    nachrechnen lassen: Markowitz (mit derselben Umschlagschranke und denselben
    gedrifteten Vorgängergewichten wie im Backtest) und Risk Parity. Verglichen
    wird die Lösung mit Ledoit-Wolf-Matrix gegen die mit der rohen
    Stichprobenmatrix. Der Random Forest fehlt, weil seine Renditeprognosen
    nicht gespeichert sind."""
    r = titelrenditen(nur_auswertungszeitraum=False)
    G = gewichte()
    termine = list(G["Risk Parity"].index)
    mvo, rp = MarkowitzLedoitWolf(), RiskParityPortfolio()
    deltas, fenstergroessen, verschiebung = [], [], {"Markowitz MVO": [], "Risk Parity": []}
    for k, termin in enumerate(termine):
        x = r[(r.index >= termin - pd.DateOffset(years=3)) & (r.index <= termin)]   # Schätzfenster
        lw = LedoitWolf().fit(x.values)
        deltas.append(lw.shrinkage_)
        fenstergroessen.append(len(x))
        mit = lw.covariance_ * HANDELSTAGE                                       # wie im Backtest
        ohne = np.cov(x.values, rowvar=False, bias=True) * HANDELSTAGE           # rohe Stichprobenmatrix
        mu = x.mean().values * HANDELSTAGE                                       # historisches Mittel
        # Vorgängergewichte, mit den Kursen der letzten Halteperiode gedriftet (Formel 26)
        if k == 0:
            vorher = np.ones(len(TICKERS)) / len(TICKERS)
        else:
            halte = r[(r.index > termine[k - 1]) & (r.index <= termin)]
            gedriftet = G["Markowitz MVO"].loc[termine[k - 1]].values * (1 + halte).prod().values
            vorher = gedriftet / gedriftet.sum()
        a = mvo.max_sharpe(mu, mit, w_prev=vorher, turnover_limit=0.30)
        b = mvo.max_sharpe(mu, ohne, w_prev=vorher, turnover_limit=0.30)
        verschiebung["Markowitz MVO"].append(np.abs(a - b).max())
        verschiebung["Risk Parity"].append(np.abs(rp.optimize(mit) - rp.optimize(ohne)).max())
    return pd.Series(deltas) * 100, int(np.median(fenstergroessen)), \
        {n: max(v) * 100 for n, v in verschiebung.items()}


def pruefe() -> Pruefung:
    """Alle Zahlen aus Kapitel 2, in der Reihenfolge des Textes."""
    p = Pruefung("Kapitel 2 — Theoretische Grundlagen")
    proto = protokoll()["experiment_meta"]
    rf = proto["risk_free_rate"]

    # Grundgrößen aus dem gesamten Auswertungszeitraum (Formel 2, auf ein Jahr hochgerechnet):
    r = titelrenditen()                               # 2495 Tage × 15 Titel
    mu, sigma, Sigma, Korr = jahreskennzahlen(r)
    N = len(TICKERS)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("2.2 Diversifikation und ihre Grenze")
    # Abbildung 1: Zwei-Anlagen-Fall, A = Coca-Cola, B = Caterpillar (Formel 4 und 5).
    sA, sB, rho = sigma["KO"], sigma["CAT"], Korr.loc["KO", "CAT"]
    gezeichnet = {-1.0, -0.5, 0.0, round(rho, 6), 1.0}      # die Kurven in Abbildung 1
    p.aussage("Abb. 1: Zahl der gezeichneten Korrelationen", "fünf", len(gezeichnet) == 5,
              f"{len(gezeichnet)}", notiz="−1, −0,5, 0, die gemessene und +1", woertlich=True)
    p.zahl("Abb. 1: gemessene Korrelation Coca-Cola/Caterpillar", "0,35", rho)
    zeitraum(p, r)
    p.zahl("Volatilität von Coca-Cola", "17,8", sA * 100, "%")
    p.zahl("Volatilität von Caterpillar", "29,7", sB * 100, "%")
    p.zahl("Schwelle σ_A/σ_B, unter der Beimischen das Risiko senkt", "0,60", sA / sB)
    w_cat = varianzminimale_mischung(sA, sB, rho)              # Formel 5 gleich null
    p.zahl("Anteil Caterpillar in der varianzminimalen Mischung", "15,7", w_cat * 100, "%")
    p.zahl("Volatilität dieser Mischung", "17,2", np.sqrt(zwei_anlagen_varianz(w_cat, sA, sB, rho)) * 100, "%")
    p.zahl("Rendite dieser Mischung", "11,0", ((1 - w_cat) * mu["KO"] + w_cat * mu["CAT"]) * 100, "%")
    p.zahl("Rendite von Coca-Cola allein", "8,9", mu["KO"] * 100, "%")
    # Diversifikationsgrenze (Formel 6)
    var_quer, cov_quer = diversifikation(Sigma.values)
    var_15 = gleichgewichtete_varianz(var_quer, cov_quer, N)
    p.zahl("Zahl der untersuchten Titel", "fünfzehn", N)
    p.zahl("mittlere Kovarianz C̄", "0,0278", cov_quer)
    p.zahl("minimale Volatilität √C̄ (Grenzwert)", "16,68", np.sqrt(cov_quer) * 100, "%")
    p.zahl("Volatilität bei gleichmäßiger Streuung über 15 Titel", "17,56", np.sqrt(var_15) * 100, "%")
    p.zahl("Anteil des Kovarianzterms an dieser Varianz", "84,3", (N - 1) / N * cov_quer / var_15 * 100, "%")
    abstand = (np.sqrt(var_15) - np.sqrt(cov_quer)) * 100
    p.aussage("Abstand zwischen 15 Titeln und Grenzwert", "weniger als 0,9 Pp", abstand < 0.9,
              f"{de(abstand, 2)} Pp", notiz=f"genau {de(abstand, 2)} Prozentpunkte")
    p.merke("Abstand zum Grenzwert", "0,9")
    # Erreichter Teil des Rückgangs: von einem Titel (√σ̄²) bis zu 15, gemessen am Weg bis zum Grenzwert
    anteil = (np.sqrt(var_quer) - np.sqrt(var_15)) / (np.sqrt(var_quer) - np.sqrt(cov_quer))
    p.zahl("schon erreichter Teil der möglichen Risikominderung", "91,5", anteil * 100, "%")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("2.3 Das Optimierungsproblem und der Effizienzrand")
    rand = effizienzrand_geschlossen(mu.values, Sigma.values, rf)   # Formeln 9 bis 11
    p.zahl("Abb. 2: Volatilität des Minimum-Varianz-Portfolios", "14,25", rand["sig_mvp"] * 100, "%")
    zeitraum(p, r)
    p.zahl("risikofreier Zins", "vier", rf * 100, "%")
    p.zahl("Rendite des Minimum-Varianz-Portfolios", "9,8", rand["mu_mvp"] * 100, "%")
    p.zahl("Volatilität des Tangentialportfolios", "52,8", rand["sig_tan"] * 100, "%")
    p.zahl("größter Leerverkauf im Tangentialportfolio", "79", -rand["w_tan"].min() * 100, "%")
    negativ = sorted(rand["w_mvp"][rand["w_mvp"] < 0] * 100)    # der Text nennt genau zwei
    if len(negativ) != 2:
        p.aussage("Zahl der negativen Gewichte im Minimum-Varianz-Portfolio", "zwei", False, str(len(negativ)))
    for text, wert in zip(["-4,9", "-4,6"], negativ):
        p.zahl("negatives Gewicht im Minimum-Varianz-Portfolio", text, wert, "%")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("2.4 Der Sharpe-Quotient als Vergleichsmaß")
    p.quelle("Sharpes Aufsatz zum Reward-to-Variability-Ratio", "1966", "Fn 29")
    p.zahl("risikofreier Jahreszins", "vier", rf * 100, "%")
    freq = inspect.signature(metrics.sharpe_ratio).parameters["freq"].default
    p.zahl("Handelstage je Jahr für die Annualisierung (√252)", "252", freq)
    s = strategierenditen()
    T = len(s)
    auto = s.apply(lambda x: x.autocorr(lag=1))       # Autokorrelation erster Ordnung je Strategie
    p.zahl("Autokorrelation erster Ordnung, schwächste", "−0,10", auto.max())
    p.zahl("Autokorrelation erster Ordnung, stärkste", "−0,14", auto.min())
    # Formel 15: t = Sharpe je Tag · √T, mit der Sharpe-Formel des Pakets
    t_werte = {k: metrics.sharpe_ratio(s[k], rf) / np.sqrt(HANDELSTAGE) * np.sqrt(T) for k in STRATEGIEN}
    p.zahl("kleinster t-Wert der vier Strategien", "2,49", min(t_werte.values()))
    p.zahl("größter t-Wert der vier Strategien", "2,94", max(t_werte.values()))
    p.zahl("Quantil der Standardnormalverteilung (in Prozent)", "97,5", norm.cdf(1.96) * 100)
    p.zahl("zugehörige Schwelle", "1,96", norm.ppf(0.975))
    p.zahl("Niveau eines zweiseitigen Tests", "fünf", (1 - norm.cdf(1.96)) * 2 * 100, "%")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("2.5 Schätzfehler als Kernproblem")
    mu_ci, halb, h = konfidenzintervalle(r)           # Formel 17
    monate = (1 + r).resample("ME").prod().shape[0]   # Monatsrenditen im selben Zeitraum
    p.zahl("Formel 16: Beobachtungsdauer h", "9,90", h, "Jahre")
    p.zahl("Formel 16: Zahl der Monatsrenditen", "119", monate)
    p.zahl("Formel 16: Zahl der Tagesrenditen", "2495", T)

    def se_varianz(n):
        """Standardfehler des Varianzschätzers nach Formel 16 (mit dem Faktor σ²)."""
        return np.sqrt(2 * sigma ** 4 / n + 4 * mu ** 2 * sigma ** 2 * h / n ** 2)
    verhaeltnis = (se_varianz(T) / se_varianz(monate)).median()
    p.aussage("Standardfehler der Varianz sinkt von Monats- auf Tagesdaten auf", "rund ein Fünftel",
              1 / 5.5 < verhaeltnis < 1 / 4.5, f"1/{de(1 / verhaeltnis, 1)}",
              notiz=f"1/{de(1 / verhaeltnis, 1)} im Median der fünfzehn Titel", woertlich=True)
    p.zahl("Abb. 3: Faktor des 95-%-Intervalls", "1,96", norm.ppf(0.975))
    mit_null = [t for t in TICKERS if mu_ci[t] - halb[t] <= 0 <= mu_ci[t] + halb[t]]
    p.zahl("Abb. 3: Titel, deren Intervall die Null enthält", "sieben", len(mit_null),
           notiz=", ".join(mit_null))
    zeitraum(p, r)
    p.zahl("Beobachtungsdauer, gerundet", "zehn", h, "Jahre", ungefaehr=True, toleranz=0.5,
           notiz="Text: „knapp zehn Jahren“")
    p.zahl("Intervall ± beim ruhigsten Titel", "11", halb.min() * 100, "Pp", notiz=halb.idxmin())
    p.zahl("Intervall ± beim unruhigsten Titel", "30", halb.max() * 100, "Pp", notiz=halb.idxmax())
    paare = list(combinations(TICKERS, 2))
    getrennt = [(i, j) for i, j in paare if abs(mu_ci[i] - mu_ci[j]) > halb[i] + halb[j]]
    p.zahl("Titelpaare", "105", len(paare))
    p.zahl("davon mit überlappenden Intervallen", "94", len(paare) - len(getrennt))
    p.zahl("Ausnahmen, sämtlich mit NVIDIA", "elf", len(getrennt),
           notiz="alle mit NVIDIA" if all("NVDA" in x for x in getrennt) else "nicht alle mit NVIDIA!")
    p.quelle("Jobson/Korkie: Sharpe-Quotient des geschätzten Optimums", "0,08", "Fn 38")
    p.quelle("Jobson/Korkie: Sharpe-Quotient der Gleichgewichtung", "0,27", "Fn 38")
    p.quelle("Best/Grauer: Anteil der Titel, die aus dem Portfolio fallen", "die Hälfte", "Fn 40")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("2.6 Antworten auf den Schätzfehler")
    p.zahl("unabhängige Varianz- und Kovarianzparameter bei 15 Titeln", "120", N * (N + 1) // 2)
    delta, n_voll, verschiebung = shrinkage_in_den_fenstern()
    p.zahl("n in p/n = 15/756: Handelstage je volles Schätzfenster", "756", n_voll)
    p.zahl("Verhältnis p/n", "0,02", N / n_voll, ungefaehr=True)
    p.aussage("Schrumpfungsintensität im Mittel", "unter drei Prozent", delta.mean() < 3,
              f"{de(delta.mean(), 2)} %", notiz=f"Mittel {de(delta.mean(), 2)} %, "
                                                 f"Median {de(delta.median(), 2)} %", woertlich=True)
    groesste = max(verschiebung.values())
    p.zahl("größte Gewichtsverschiebung durch die Shrinkage", "einen", groesste, "Pp", ungefaehr=True,
           toleranz=0.5, notiz="Text: „rund einen Prozentpunkt“; " +
           ", ".join(f"{KURZ[n]} {de(v, 2)} Pp" for n, v in verschiebung.items()))
    p.zahl("Obergrenze je Titel", "20", proto["max_weight"] * 100, "%")
    p.quelle("Qian: Kapitalanteil der Aktien im 60/40-Portfolio", "60", "Fn 48")
    p.quelle("Qian: Risikoanteil der Aktien", "93", "Fn 48")
    oben = Korr.values[np.triu_indices(N, k=1)]      # jede Paarkorrelation einmal
    p.zahl("Zahl der Paarkorrelationen", "105", len(oben))
    p.zahl("kleinste Paarkorrelation", "0,10", oben.min())
    p.zahl("größte Paarkorrelation", "0,83", oben.max())
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
