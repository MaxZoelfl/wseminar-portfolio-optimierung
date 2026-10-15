"""
Kapitel 3 — Random-Forest-gestützte Renditeprognose: jede eigene Zahl nachgerechnet.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Kapitel 3 erklärt, wie der Random Forest Renditen vorhersagt und warum dabei
wenig zu erwarten ist. Die eigenen Zahlen darin beschreiben den Datensatz
(wie groß ist der Prognosefehler?), den Lernstoff (wie viele Trainingszeilen?)
und die Wälder, die der Backtest tatsächlich gewählt hat (welche Baumtiefe?).

Zwei Stufen, je nach Geduld:
  ohne Schalter        rund 10 Sekunden. Baut die Merkmale neu auf (wie der
                       Backtest, aber ohne Optimierung), liest die gewählten
                       Einstellungen aus skripte/daten/rf_hyperparameter.csv und
                       trainiert an jedem vierten Termin je einen Einzelbaum der
                       Tiefe 1 und 14 (§ 3.2).
  --baumkorrelation    zusätzlich rund 55 Minuten: tunt an allen 119 Terminen den
                       Wald genau wie im Backtest und misst, wie ähnlich sich seine
                       Bäume sind (§ 3.3).
Keiner der Schritte führt den großen Backtest aus.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_3_random_forest.py
    python skripte/kapitel_3_random_forest.py --baumkorrelation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from portfolio.config import START_DATE, END_DATE, TRAIN_YEARS, CV_EMBARGO   # noqa: E402
from portfolio.data import download_data              # noqa: E402
from portfolio.indicators import (build_all_indicators, aggregate_to_monthly,  # noqa: E402
                                  add_cross_sectional_ranks, FEATURE_COLS)

BAUMKORR = "--baumkorrelation" in sys.argv


# ────────────────────────────────────────────────────────────────────────
# Merkmale und Trainingsfenster — dieselbe Kette wie im Backtest
# ────────────────────────────────────────────────────────────────────────
def monatsdaten_aufbauen():
    px, ret, spy, tickers = download_data(TICKERS, SPY_TICKER, START_DATE, END_DATE)
    md = add_cross_sectional_ranks(aggregate_to_monthly(
        build_all_indicators(px, ret, spy, tickers), ret, tickers))
    return md, ret


def trainingsfenster(monatsdaten, monatsende):
    """Merkmale X und Zielwerte y genau so wie backtest.run_backtest()."""
    start = monatsende - pd.DateOffset(years=TRAIN_YEARS)
    fenster = monatsdaten[(monatsdaten.index >= start) & (monatsdaten.index <= monatsende)]
    letzter_train = monatsende - pd.DateOffset(months=1)   # Look-Ahead-Schutz (§ 4.3)
    X, y = [], []
    for t in TICKERS:
        z = fenster[(fenster["ticker"] == t) & (fenster.index <= letzter_train)]
        sp = [c for c in FEATURE_COLS if c in z.columns]
        g = z[sp + ["target_next_month"]].dropna()
        if len(g) < 12:                                     # unter 12 Monatszeilen: Titel fällt weg
            continue
        X.append(g[sp]); y.append(g["target_next_month"])
    return pd.concat(X), pd.concat(y), fenster


def pruefe() -> Pruefung:
    p = Pruefung("Kapitel 3 — Random-Forest-gestützte Renditeprognose")
    md, ret = monatsdaten_aufbauen()
    termine = gewichte()["Risk Parity"].index                # die 119 Entscheidungstermine
    hp = pd.read_csv(DATEN / "rf_hyperparameter.csv", comment="#")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("3.1 Prognose als Regressionsproblem")
    p.zahl("Zahl der Kennzahlen je Titel und Monat", "20", len(FEATURE_COLS))
    # Die Zielgröße: Monatsrenditen aller Titel im Auswertungszeitraum.
    panel = md.loc[(md.index >= "2015-01-31") & (md.index <= "2024-12-31"),
                   ["ticker", "monthly_ret"]].dropna()
    r = panel["monthly_ret"]
    p.zahl("Zahl der Titel", "15", panel["ticker"].nunique())
    # RMSE der Nullprognose = Wurzel des mittleren QUADRATS der Renditen
    # (nicht zentriert, wie im Nenner des R² bei Gu/Kelly/Xiu).
    rmse0 = np.sqrt((r ** 2).mean())
    p.zahl("RMSE der Nullprognose", "7,48", rmse0 * 100, "%")
    p.zahl("RMSE-Rückgang bei R² = 1 %", "0,5", (1 - np.sqrt(1 - 0.01)) * 100, "%", ungefaehr=True)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("3.2 Entscheidungsbäume und das Split-Kriterium")
    groessen = pd.Series([len(trainingsfenster(md, t)[0]) for t in termine])
    median = groessen.median()
    p.zahl("Trainingszeilen je Termin (Median)", "525", median, ungefaehr=True)
    p.zahl("Bereiche bei Tiefe drei", "8", 2 ** 3)
    p.zahl("Zeilen je Bereich bei Tiefe drei", "66", median / 8, ungefaehr=True)
    p.aussage("Zeilen je Bereich bei Tiefe zehn", "weniger als eine", median / 2 ** 10 < 1,
              de(median / 2 ** 10, 2))
    a, b = tiefensweep(md, termine)
    p.zahl("Trainingsfehler Tiefe 14 in % von Tiefe 1", "22", a * 100, "%")
    p.zahl("Anstieg des Prüffehlers Tiefe 1 → 14", "81", b * 100, "%")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("3.3 Bagging als Diversifikation")
    r_tag = titelrenditen()
    oben = r_tag.corr().values[np.triu_indices(len(TICKERS), k=1)]
    p.zahl("mittlere Korrelation der 15 Aktien", "0,42", oben.mean())
    B = hp["n_estimators"].median()
    p.zahl("Bäume je Wald (Median)", "321", B)
    if BAUMKORR:
        k = baumkorrelation(md, ret, termine)
        rho, rho_quer, verhaeltnis, eingehalten = k["rho"], k["rho_quer"], k["verhaeltnis"], k["eingehalten"]
        p.zahl("mittlere Korrelation der Baumprognosen ρ", "0,21", rho, ungefaehr=True)
    else:
        rho = 0.21                                    # Textwert; die Rechnung unten hängt nicht davon ab
        p.offen("mittlere Korrelation der Baumprognosen ρ", "≈ 0,21", "nur mit --baumkorrelation (rund 55 Minuten)")
    # Formel 24: Var = σ²(ρ + (1−ρ)/B). Von einem Baum (σ²) bis zu unendlich
    # vielen (ρσ²) sinkt die Varianz um (1−ρ)σ². Bei B Bäumen ist davon der
    # Anteil [1 − (ρ + (1−ρ)/B)] / (1 − ρ) = 1 − 1/B erreicht — unabhängig von ρ.
    erreicht = (1 - (rho + (1 - rho) / B)) / (1 - rho)
    p.zahl("erreichte Varianzreduktion bei 321 Bäumen", "99,7", erreicht * 100, "%",
           notiz="Anteil der bei unbegrenzt vielen Bäumen möglichen; = 1 − 1/B")
    C = (r_tag.cov() * HANDELSTAGE).values
    N = len(TICKERS)
    vq, cq, v15 = np.diag(C).mean(), C[~np.eye(N, dtype=bool)].mean(), np.ones(N) / N @ C @ (np.ones(N) / N)
    p.zahl("zum Vergleich: bei 15 Titeln (§ 2.2)", "91,5",
           (np.sqrt(vq) - np.sqrt(v15)) / (np.sqrt(vq) - np.sqrt(cq)) * 100, "%")
    p.zahl("Zahl der Termine", "119", len(termine))
    if BAUMKORR:
        p.zahl("Residuenkorrelation ρ̄ (Breiman)", "0,84", rho_quer)
        p.zahl("Fehler des Waldes in % des Einzelbaums", "82", verhaeltnis * 100, "%")
        p.aussage("Schranke an jedem Termin eingehalten", "ja", eingehalten == len(termine),
                  f"{eingehalten} von {len(termine)}")
    else:
        for was, text in [("Residuenkorrelation ρ̄ (Breiman)", "0,84"),
                          ("Fehler des Waldes in % des Einzelbaums", "82 %"),
                          ("Schranke an jedem Termin eingehalten", "ja")]:
            p.offen(was, text, "nur mit --baumkorrelation (rund 55 Minuten)")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("3.4 Merkmale und ihre Begründung")
    import inspect
    from portfolio.indicators import compute_alpha_beta
    fenster = inspect.signature(compute_alpha_beta).parameters["window"].default
    p.zahl("Regressionsfenster für Alpha und Beta", "63", fenster, "Handelstage")
    p.offen("Momentum wirkt über drei bis zwölf Monate", "3–12", "Befund aus der Quelle (Jegadeesh/Titman)")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("3.5 Was vom maschinellen Lernen zu erwarten ist")
    for was, text in [("Gu/Kelly/Xiu: Titel, Jahre, Merkmale", "30 000 / 60 / 920"),
                      ("Gu/Kelly/Xiu: R² Random Forest / bestes Verfahren", "0,33 / 0,40 %"),
                      ("Gu/Kelly/Xiu: gewählte Tiefen", "1–5")]:
        p.offen(was, text, "Zahlen aus der Quelle")
    from portfolio.optimizers import RFPortfolioOptimizer
    tiefe = RFPortfolioOptimizer()._param_grid()["rf__max_depth"]
    kleinste, groesste = tiefe.support()          # randint(3, 15) zieht 3 bis 14
    p.zahl("größte zulässige Tiefe im Suchraum", "14", groesste)
    p.zahl("kleinste zulässige Tiefe", "3", kleinste)
    p.zahl("Monate mit Tiefe drei", "98", int((hp["max_depth"] == 3).sum()))
    p.zahl("von Terminen", "119", len(hp))
    # Der Text sagt "etwa 7,47": Aus den gerundeten Textwerten folgt
    # 7,48 · √(1 − 0,0033) = 7,4676, aus dem genauen RMSE 7,4646.
    p.zahl("RMSE bei R² = 0,33 %", "7,47", np.sqrt((r ** 2).mean() * (1 - 0.0033)) * 100, "%",
           ungefaehr=True, toleranz=0.01,
           notiz="Text: „etwa“; aus den gerundeten Textwerten 7,48 · √(1 − 0,0033) = 7,4676")
    return p


# ────────────────────────────────────────────────────────────────────────
# Langsame Rechnungen (nur auf Wunsch)
# ────────────────────────────────────────────────────────────────────────
def tiefensweep(md, termine):
    """§ 3.2: Einzelbaum je Tiefe 1…14 an jedem vierten Termin, Purged CV wie im Backtest.
    Übrige Einstellungen fest (min_samples_leaf=3), damit allein die Tiefe variiert.
    Rückgabe: (Trainingsfehler Tiefe 14 / Tiefe 1, Prüffehler Tiefe 14 / Tiefe 1 − 1)."""
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import StandardScaler
    from portfolio.cross_validation import purged_kfold_splits
    stichprobe = [termine[i] for i in range(0, len(termine), 4)]
    ergebnis = {t: {"train": [], "cv": []} for t in (1, 14)}
    for termin in stichprobe:
        X, y, _ = trainingsfenster(md, termin)
        folds = list(purged_kfold_splits(X.index, n_splits=5, embargo_pct=CV_EMBARGO))
        Xv, yv = X.values, y.values
        for tiefe in (1, 14):
            tr, cv = [], []
            for i_tr, i_te in folds:
                sk = StandardScaler().fit(Xv[i_tr])
                A, Bm = sk.transform(Xv[i_tr]), sk.transform(Xv[i_te])
                baum = DecisionTreeRegressor(max_depth=tiefe, min_samples_leaf=3,
                                             random_state=42).fit(A, yv[i_tr])
                tr.append(((yv[i_tr] - baum.predict(A)) ** 2).mean())
                cv.append(((yv[i_te] - baum.predict(Bm)) ** 2).mean())
            ergebnis[tiefe]["train"].append(np.mean(tr))
            ergebnis[tiefe]["cv"].append(np.mean(cv))
    a = np.mean(ergebnis[14]["train"]) / np.mean(ergebnis[1]["train"])
    b = np.mean(ergebnis[14]["cv"]) / np.mean(ergebnis[1]["cv"]) - 1
    return a, b


def baumkorrelation(md, ret, termine):
    """§ 3.3: An jedem Termin den Wald wie im Backtest tunen und die Bäume vergleichen.
    ρ   = mittlere Korrelation der Prognosen zweier Bäume (Formel 24)
    ρ̄   = Korrelation der Residuen nach Breiman (Formel 25), PE*(Wald)/PE*(Baum)."""
    from portfolio.optimizers import RFPortfolioOptimizer
    print("\n  Baumkorrelation läuft (rund 55 Minuten) …", flush=True)
    zeilen = []
    for i, termin in enumerate(termine):
        X, y, fenster = trainingsfenster(md, termin)
        aktuell = []
        for t in TICKERS:
            z = fenster[fenster["ticker"] == t]
            sp = [c for c in FEATURE_COLS if c in z.columns]
            g = z[sp].dropna()
            aktuell.append(g.iloc[-1] if len(g) else pd.Series(np.zeros(len(sp)), index=sp))
        X_akt = pd.DataFrame(aktuell)[X.columns]
        ende = termine[i + 1] if i + 1 < len(termine) else ret.index[-1]
        halte = ret[(ret.index > termin) & (ret.index <= ende)]
        y_real = ((1 + halte).prod() - 1).reindex(TICKERS).values
        rfo = RFPortfolioOptimizer()
        rfo.fit_with_tuning(X, y, sample_times=X.index)
        skala, wald = rfo.best_estimator_.named_steps["scaler"], rfo.best_estimator_.named_steps["rf"]
        P = np.array([b.predict(skala.transform(X_akt.values)) for b in wald.estimators_])
        beweglich = P.std(axis=1) > 1e-12            # Bäume mit konstanter Prognose: keine Korrelation
        Q = P[beweglich]
        Cq = np.corrcoef(Q)
        rho = (Cq.sum() - np.trace(Cq)) / (len(Q) * (len(Q) - 1))
        R = y_real[None, :] - P
        sd = np.sqrt((R ** 2).mean(axis=1))          # Breiman: rohes zweites Moment
        pe_baum = (sd ** 2).mean()
        pe_wald = ((y_real - P.mean(axis=0)) ** 2).mean()
        rho_quer = pe_wald / sd.mean() ** 2
        zeilen.append(dict(rho=rho, rho_quer=rho_quer, verh=pe_wald / pe_baum,
                           ok=pe_wald <= rho_quer * pe_baum + 1e-15))
        if i % 10 == 0:
            print(f"    {i:>3} von {len(termine)} …", flush=True)
    D = pd.DataFrame(zeilen)
    return dict(rho=D.rho.mean(), rho_quer=D.rho_quer.mean(),
                verhaeltnis=D.verh.mean(), eingehalten=int(D.ok.sum()))


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
