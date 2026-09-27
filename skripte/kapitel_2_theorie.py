"""
Kapitel 2 — Theoretische Grundlagen: jede eigene Zahl nachgerechnet.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Kapitel 2 erklärt die Theorie (Diversifikation, Effizienzrand, Sharpe-Quotient,
Schätzfehler) und belegt sie jeweils an den fünfzehn Aktien der Arbeit. Dieses
Skript rechnet all diese Beispielzahlen noch einmal aus und vergleicht sie mit
dem Text. Es braucht dafür nur die eingefrorenen Kurse und die Tagesrenditen
der Strategien, keinen neuen Backtest.

Wichtig für das Verständnis: Die Theoriezahlen in § 2.2, § 2.3 und § 2.5
beziehen sich auf den GESAMTEN Auswertungszeitraum (2495 Handelstage,
02.02.2015 bis 30.12.2024), sind also "im Rückblick" geschätzt. Die
Strategien im Backtest sehen dagegen immer nur die jeweils letzten drei Jahre.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_2_theorie.py          # rund 20 Sekunden

AUFBAU — entspricht den Abschnitten der Arbeit:
    § 2.2  Zwei-Anlagen-Fall Coca-Cola/Caterpillar, Diversifikationsgrenze
    § 2.3  Minimum-Varianz- und Tangentialportfolio (geschlossene Lösung)
    § 2.4  Autokorrelation und t-Werte der vier Strategien
    § 2.5  Konfidenzintervalle der erwarteten Renditen, Mertons Asymmetrie
    § 2.6  Ledoit-Wolf-Schrumpfung, Paarkorrelationen
    (§ 2.1 enthält keine eigenen Zahlen.)
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from scipy.stats import norm                          # noqa: E402


def pruefe() -> Pruefung:
    p = Pruefung("Kapitel 2 — Theoretische Grundlagen")
    proto = protokoll()["experiment_meta"]

    # Grundgrößen aus dem gesamten Auswertungszeitraum (wie portfolio/theory_plots.py):
    r = titelrenditen()                               # 2495 Tage × 15 Titel
    mu = r.mean() * HANDELSTAGE                       # erwartete Rendite je Titel, p. a.
    Sigma = r.cov() * HANDELSTAGE                     # Kovarianzmatrix, p. a.
    sigma = pd.Series(np.sqrt(np.diag(Sigma)), index=Sigma.index)   # Volatilitäten
    Korr = r.corr()
    N = len(TICKERS)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("2.2 Diversifikation und ihre Grenze")
    # Zwei-Anlagen-Fall: A = Coca-Cola, B = Caterpillar (Formel 4 und 5).
    sA, sB, rho = sigma["KO"], sigma["CAT"], Korr.loc["KO", "CAT"]
    p.zahl("Volatilität Coca-Cola", "17,8", sA * 100, "%")
    p.zahl("Volatilität Caterpillar", "29,7", sB * 100, "%")
    p.zahl("Korrelation KO/CAT (auch Abb. 1)", "0,35", rho)
    p.zahl("Schwelle σ_A/σ_B", "0,60", sA / sB)
    # Varianzminimale Mischung: Ableitung (Formel 5) gleich null gesetzt.
    w_cat = (sA**2 - rho * sA * sB) / (sA**2 + sB**2 - 2 * rho * sA * sB)
    vol_mix = np.sqrt((1 - w_cat)**2 * sA**2 + w_cat**2 * sB**2
                      + 2 * w_cat * (1 - w_cat) * rho * sA * sB)
    p.zahl("Anteil Caterpillar in der Mischung", "15,7", w_cat * 100, "%")
    p.zahl("Volatilität der Mischung", "17,2", vol_mix * 100, "%")
    p.zahl("Rendite der Mischung", "11,0", ((1 - w_cat) * mu["KO"] + w_cat * mu["CAT"]) * 100, "%")
    p.zahl("Rendite Coca-Cola allein", "8,9", mu["KO"] * 100, "%")

    # Diversifikationsgrenze (Formel 6): mittlere Varianz und mittlere Kovarianz.
    C = Sigma.values
    var_quer = np.diag(C).mean()
    cov_quer = C[~np.eye(N, dtype=bool)].mean()
    w_gleich = np.ones(N) / N
    var_15 = w_gleich @ C @ w_gleich                  # = var_quer/N + (N−1)/N·cov_quer
    p.zahl("mittlere Kovarianz C̄", "0,0278", cov_quer)
    p.zahl("Grenze √C̄ (auch Abb. A1)", "16,68", np.sqrt(cov_quer) * 100, "%")
    p.zahl("Volatilität bei N = 15", "17,56", np.sqrt(var_15) * 100, "%")
    p.zahl("Anteil des Kovarianzterms an der Varianz", "84,3",
           (N - 1) / N * cov_quer / var_15 * 100, "%")
    abstand = (np.sqrt(var_15) - np.sqrt(cov_quer)) * 100
    p.aussage("Abstand 1/N zur Grenze", "< 0,9 Pp", abstand < 0.9, f"{de(abstand, 2)} Pp")
    # "Zählgewinn": Rückgang der Volatilität von einem Titel (√σ̄²) bis N = 15,
    # gemessen am Rückgang bis zur Grenze N → ∞.
    anteil = (np.sqrt(var_quer) - np.sqrt(var_15)) / (np.sqrt(var_quer) - np.sqrt(cov_quer))
    p.zahl("erreichter Anteil des Rückgangs", "91,5", anteil * 100, "%")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("2.3 Das Optimierungsproblem und der Effizienzrand")
    # Geschlossene Lösung (Formel 9 und 11), OHNE Leerverkaufsverbot und Obergrenze.
    S_inv = np.linalg.inv(C)
    eins, m = np.ones(N), mu.values
    a = eins @ S_inv @ eins
    w_mvp = S_inv @ eins / a
    p.zahl("Volatilität MVP (Abb. 2)", "14,25", np.sqrt(1 / a) * 100, "%")
    p.zahl("risikofreier Zins", "4", proto["risk_free_rate"] * 100, "%")
    p.zahl("Rendite MVP", "9,8", w_mvp @ m * 100, "%")
    rf = proto["risk_free_rate"]
    w_tan = S_inv @ (m - rf * eins) / (eins @ S_inv @ (m - rf * eins))
    p.zahl("Volatilität Tangentialportfolio (auch Abb. A2)", "52,8",
           np.sqrt(w_tan @ C @ w_tan) * 100, "%")
    p.zahl("größter Leerverkauf im Tangentialportfolio", "79", -w_tan.min() * 100, "%")
    negativ = sorted(w_mvp[w_mvp < 0] * 100)
    p.aussage("Zahl negativer MVP-Gewichte", "zwei", len(negativ) == 2, f"{len(negativ)}")
    for text, wert in zip(["−4,9", "−4,6"], negativ):
        p.zahl("negatives MVP-Gewicht", text, wert, "%")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("2.4 Der Sharpe-Quotient als Vergleichsmaß")
    s = strategierenditen()
    T = len(s)
    auto = s.apply(lambda x: x.autocorr(lag=1))       # Autokorrelation erster Ordnung
    p.zahl("Autokorrelation, schwächste", "−0,10", auto.max())
    p.zahl("Autokorrelation, stärkste", "−0,14", auto.min())
    # Formel 15: t = Sharpe je Tag · √T.
    t_werte = {k: sharpe(s[k], rf) / np.sqrt(HANDELSTAGE) * np.sqrt(T) for k in STRATEGIEN}
    p.zahl("kleinster t-Wert", "2,49", min(t_werte.values()))
    p.zahl("größter t-Wert", "2,94", max(t_werte.values()))
    p.zahl("97,5-%-Quantil der Normalverteilung", "1,96", norm.ppf(0.975))
    p.offen("Sharpes Aufsatz", "1966", "Jahreszahl aus der Quelle (Fußnote)")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("2.5 Schätzfehler als Kernproblem")
    h = T / HANDELSTAGE                               # Beobachtungsdauer in Jahren
    monate = (1 + r).resample("ME").prod().shape[0]   # Monatsrenditen im selben Zeitraum
    p.zahl("Zahl der Monatsrenditen", "119", monate)
    p.zahl("Zahl der Tagesrenditen", "2495", T)
    # Formel 16 (Merton): Var[σ̂²] = 2σ⁴/n + 4μ²σ²h/n². Verhältnis der
    # Standardfehler Tages- gegen Monatsdaten, je Titel.
    def se_var(n):
        return np.sqrt(2 * sigma**4 / n + 4 * mu**2 * sigma**2 * h / n**2)
    verhaeltnis = (se_var(T) / se_var(monate)).median()
    p.aussage("Standardfehler der Varianz sinkt auf", "rund 1/5",
              1 / 5.5 < verhaeltnis < 1 / 4.5, f"1/{de(1 / verhaeltnis, 1)} (Median der Titel)")
    p.zahl("Beobachtungsdauer h (Abb. 3)", "9,90", h, "Jahre")
    halb = 1.96 * sigma / np.sqrt(h)                  # halbe Breite des 95-%-Intervalls
    p.zahl("Intervall ± ruhigster Titel", "11", halb.min() * 100, "Pp",
           notiz=f"{sigma.idxmin()}")
    p.zahl("Intervall ± unruhigster Titel", "30", halb.max() * 100, "Pp",
           notiz=f"{sigma.idxmax()}")
    mit_null = [t for t in TICKERS if mu[t] - halb[t] <= 0 <= mu[t] + halb[t]]
    p.aussage("Titel, deren Intervall die Null enthält", "sieben", len(mit_null) == 7,
              f"{len(mit_null)}: {', '.join(mit_null)}")
    paare = list(combinations(TICKERS, 2))
    getrennt = [(i, j) for i, j in paare if abs(mu[i] - mu[j]) > halb[i] + halb[j]]
    p.zahl("Titelpaare", "105", len(paare))
    p.zahl("davon mit überlappenden Intervallen", "94", len(paare) - len(getrennt))
    p.aussage("alle Ausnahmen betreffen NVIDIA", "elf, alle NVDA",
              len(getrennt) == 11 and all("NVDA" in x for x in getrennt), f"{len(getrennt)}")
    p.offen("Jobson/Korkie: Sharpe 0,08 gegen 0,27", "0,08 / 0,27", "Zahlen aus der Quelle (Michaud)")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("2.6 Antworten auf den Schätzfehler")
    p.zahl("unabhängige Varianz-/Kovarianzparameter", "120", N * (N + 1) // 2)
    # Ledoit-Wolf in jedem der 119 Schätzfenster, genau wie im Backtest.
    from sklearn.covariance import LedoitWolf
    from portfolio.optimizers import MarkowitzLedoitWolf
    r_alle = titelrenditen(nur_auswertungszeitraum=False)
    mvo = MarkowitzLedoitWolf()
    deltas, fenstergroessen, verschiebung = [], [], []
    for termin in gewichte()["Risk Parity"].index:
        x = r_alle[(r_alle.index >= termin - pd.DateOffset(years=3)) & (r_alle.index <= termin)]
        lw = LedoitWolf().fit(x.values)
        deltas.append(lw.shrinkage_)
        fenstergroessen.append(len(x))
        # Wirkung auf die Gewichte, gemessen am Minimum-Varianz-Portfolio mit
        # Leerverkaufsverbot und 20-%-Obergrenze: mit gegen ohne Schrumpfung.
        mit = mvo.min_variance(lw.covariance_ * HANDELSTAGE)
        ohne = mvo.min_variance(np.cov(x.values, rowvar=False, bias=True) * HANDELSTAGE)
        verschiebung.append(np.abs(mit - ohne).max())
    n_voll = int(np.median(fenstergroessen))
    p.zahl("Beobachtungen je volles Fenster n", "756", n_voll)
    p.zahl("Verhältnis p/n", "0,02", N / n_voll, ungefaehr=True)
    d = pd.Series(deltas) * 100
    p.aussage("Schrumpfungsintensität im Mittel", "unter 3 %", d.mean() < 3 and d.median() < 3,
              f"Mittel {de(d.mean(), 2)} %, Median {de(d.median(), 2)} %")
    p.zahl("größte Gewichtsverschiebung", "1", max(verschiebung) * 100, "Pp",
           notiz=f"genau {de(max(verschiebung) * 100, 2)} Pp am Minimum-Varianz-Portfolio — "
                 "streng genommen knapp über einem Prozentpunkt")
    p.zahl("Obergrenze je Titel", "20", proto["max_weight"] * 100, "%")
    p.offen("Qian: 60 % Kapital, 93 % Risiko", "60 / 93", "Zahlen aus der Quelle")
    oben = Korr.values[np.triu_indices(N, k=1)]
    p.zahl("kleinste Paarkorrelation", "0,10", oben.min())
    p.zahl("größte Paarkorrelation", "0,83", oben.max())
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
