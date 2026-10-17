"""
Kapitel 3 — Random-Forest-gestützte Renditeprognose: jede Zahl des Kapitels,
in der Reihenfolge der Arbeit.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Kapitel 3 erklärt, wie der Random Forest Renditen vorhersagt und warum dabei
wenig zu erwarten ist. Die eigenen Zahlen darin beschreiben den Datensatz
(wie groß ist der Prognosefehler?), den Lernstoff (wie viele Trainingszeilen?)
und die Wälder, die der Backtest tatsächlich gewählt hat (welche Baumtiefe?).

Woher die Zahlen kommen:
  - Merkmale und Lernstoff baut das Skript genau so auf wie der Backtest
    (dieselben Funktionen aus portfolio/indicators.py, dasselbe Abschneiden
    des Trainingsfensters wie in portfolio/backtest.py).
  - Die gewählten Einstellungen je Termin (Bäume, Tiefe, Blattgröße) stehen in
    skripte/daten/rf_hyperparameter.csv, aus dem Laufprotokoll des Backtests.
  - Die Korrelation der Bäume (§ 3.3) verlangt, an allen 119 Terminen den Wald
    neu zu tunen — rund 55 Minuten. Das Ergebnis liegt deshalb als
    Referenzdatei in skripte/daten/baumkorrelation.csv. Mit dem Schalter
    --baumkorrelation rechnet das Skript sie neu und vergleicht mit der Datei.
  - Beide Referenzdateien gehören zum Lauf der Arbeit. Bei einem anderen
    Universum oder anderen Einstellungen (config.json) entfallen die Zahlen
    daraus mit einem Hinweis; die Korrelation der Bäume lässt sich dann mit
    --baumkorrelation für den neuen Lauf rechnen.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_3_random_forest.py                     # rund 15 Sekunden
    python skripte/kapitel_3_random_forest.py --baumkorrelation   # + 55 Minuten
    python skripte/kapitel_3_random_forest.py --baumkorrelation --speichern
                                     # dasselbe und die Referenzdatei neu schreiben
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from portfolio.config import START_DATE, END_DATE, TRAIN_YEARS, CV_EMBARGO   # noqa: E402
from portfolio.data import download_data              # noqa: E402
from portfolio.indicators import (build_all_indicators, aggregate_to_monthly,  # noqa: E402
                                  add_cross_sectional_ranks, compute_alpha_beta, FEATURE_COLS)
from portfolio.optimizers import RFPortfolioOptimizer  # noqa: E402

BAUMKORR = "--baumkorrelation" in sys.argv            # lange Rechnung erzwingen?
SPEICHERN = "--speichern" in sys.argv                 # Referenzdatei neu schreiben?
REFERENZ = DATEN / "baumkorrelation.csv"

# Die fünf Merkmalsgruppen aus § 3.4, mit den Merkmalen aus Tabelle A2.
GRUPPEN = {
    "Trend und Momentum": ["mom_21d", "mom_63d", "mom_126d", "mom_252d", "monthly_ret"],
    "technische Indikatoren": ["rsi", "macd", "macd_sig", "macd_hist", "bb_pct_b", "bb_width"],
    "Risiko": ["vol_21d", "vol_63d"],
    "Marktbezug": ["alpha_spy", "beta_spy"],
    "Querschnittsränge": ["rsi_rank", "mom_21d_rank", "mom_63d_rank", "mom_252d_rank", "alpha_spy_rank"],
}


# ────────────────────────────────────────────────────────────────────────
# Merkmale und Trainingsfenster — dieselbe Kette wie im Backtest
# ────────────────────────────────────────────────────────────────────────
def monatsdaten_aufbauen():
    """Monatliche Merkmale aller Titel, mit denselben Funktionen wie der Backtest:
    Tagesindikatoren → Monatswerte → Querschnittsränge. Rückgabe: (Monatstabelle,
    Tagesrenditen der Titel)."""
    px, ret, spy, tickers = download_data(TICKERS, SPY_TICKER, START_DATE, END_DATE)
    md = add_cross_sectional_ranks(aggregate_to_monthly(
        build_all_indicators(px, ret, spy, tickers), ret, tickers))
    return md, ret


def trainingsfenster(monatsdaten, monatsende):
    """Merkmale X und Zielwerte y genau so wie portfolio/backtest.py, Abschnitt 4.

    Der Backtest schneidet diese Auswahl inline aus (im Rechenkern, der nicht
    verändert wird); hier steht dieselbe Regel noch einmal, Zeile für Zeile:
    drei Jahre bis zum Entscheidungstag, Zeilen nur bis "Monatsende minus ein
    Monat" (Look-Ahead-Schutz, § 4.3, samt Kalenderdetail), mindestens zwölf
    vollständige Monatszeilen je Titel."""
    start = monatsende - pd.DateOffset(years=TRAIN_YEARS)
    fenster = monatsdaten[(monatsdaten.index >= start) & (monatsdaten.index <= monatsende)]
    letzter_train = monatsende - pd.DateOffset(months=1)
    X, y = [], []
    for t in TICKERS:
        z = fenster[(fenster["ticker"] == t) & (fenster.index <= letzter_train)]
        sp = [c for c in FEATURE_COLS if c in z.columns]
        g = z[sp + ["target_next_month"]].dropna()           # nur vollständige Zeilen
        if len(g) < 12:                                     # unter 12 Monatszeilen: Titel fällt weg
            continue
        X.append(g[sp])
        y.append(g["target_next_month"])
    return pd.concat(X), pd.concat(y), fenster


def tiefensweep(md, termine):
    """§ 3.2: Einzelbaum der Tiefe 1 und 14 an allen 119 Terminen, geprüft mit der
    Purged-Kreuzvalidierung wie im Backtest (fünf Blöcke, Sperrzone 2 %).
    Übrige Einstellungen fest (Blattgröße 3, die kleinste des Suchraums), damit
    allein die Tiefe variiert — "bei sonst festen Einstellungen".
    Rückgabe: (Trainingsfehler Tiefe 14 / Tiefe 1, Prüffehler Tiefe 14 / Tiefe 1 − 1),
    jeweils aus den über alle Termine gemittelten Fehlern."""
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import StandardScaler
    from portfolio.cross_validation import purged_kfold_splits
    fehler = {t: {"train": [], "cv": []} for t in (1, 14)}
    for termin in termine:
        X, y, _ = trainingsfenster(md, termin)
        folds = list(purged_kfold_splits(X.index, n_splits=5, embargo_pct=CV_EMBARGO))
        Xv, yv = X.values, y.values
        for tiefe in (1, 14):
            train, pruef = [], []
            for i_tr, i_te in folds:
                skala = StandardScaler().fit(Xv[i_tr])          # wie die Pipeline im Backtest
                A, B = skala.transform(Xv[i_tr]), skala.transform(Xv[i_te])
                baum = DecisionTreeRegressor(max_depth=tiefe, min_samples_leaf=3,
                                             random_state=42).fit(A, yv[i_tr])
                train.append(((yv[i_tr] - baum.predict(A)) ** 2).mean())   # Fehler auf dem Lernstoff
                pruef.append(((yv[i_te] - baum.predict(B)) ** 2).mean())   # Fehler auf dem Prüfblock
            fehler[tiefe]["train"].append(np.mean(train))
            fehler[tiefe]["cv"].append(np.mean(pruef))
    a = np.mean(fehler[14]["train"]) / np.mean(fehler[1]["train"])
    b = np.mean(fehler[14]["cv"]) / np.mean(fehler[1]["cv"]) - 1
    return a, b


def baumkorrelation(md, ret, termine) -> pd.DataFrame:
    """§ 3.3: An jedem Termin den Wald genau wie im Backtest tunen und seine Bäume
    vergleichen. Je Termin:
      rho        mittlere Korrelation der Prognosen zweier Bäume (Formel 24)
      rho_quer   gewichtete Korrelation ihrer Fehler nach Breiman (Formel 25)
      verh       Fehler des Waldes geteilt durch den mittleren Fehler eines Baums
      ok         hält der Wald Breimans Schranke ein?
      wie_lauf   hat die Suche dieselben Einstellungen gewählt wie im Backtest?"""
    # Die Einstellungen des Laufs (zum Vergleich in der Spalte wie_lauf) gibt es nur
    # für das Universum der Arbeit.
    hp = pd.read_csv(DATEN / "rf_hyperparameter.csv", comment="#") if LAUF_DER_ARBEIT else None
    print("\n  Baumkorrelation läuft (rund 55 Minuten) …", flush=True)
    zeilen = []
    for i, termin in enumerate(termine):
        X, y, fenster = trainingsfenster(md, termin)
        # Prognose-Eingabe wie im Backtest: je Titel die jüngste vollständige Merkmalszeile
        aktuell = []
        for t in TICKERS:
            z = fenster[fenster["ticker"] == t]
            sp = [c for c in FEATURE_COLS if c in z.columns]
            g = z[sp].dropna()
            aktuell.append(g.iloc[-1] if len(g) else pd.Series(np.zeros(len(sp)), index=sp))
        X_akt = pd.DataFrame(aktuell)[X.columns]
        # Was tatsächlich kam: Rendite jedes Titels in der folgenden Halteperiode
        ende = termine[i + 1] if i + 1 < len(termine) else ret.index[-1]
        halte = ret[(ret.index > termin) & (ret.index <= ende)]
        y_real = ((1 + halte).prod() - 1).reindex(TICKERS).values
        rfo = RFPortfolioOptimizer()
        rfo.fit_with_tuning(X, y, sample_times=X.index)   # dieselbe Suche wie im Backtest
        bp = rfo.best_params_
        wie_lauf = None
        if hp is not None:
            soll = hp.iloc[i]
            wie_lauf = (bp["rf__n_estimators"], bp["rf__max_depth"], bp["rf__min_samples_leaf"]) == \
                       (soll["n_estimators"], soll["max_depth"], soll["min_samples_leaf"])
        skala, wald = rfo.best_estimator_.named_steps["scaler"], rfo.best_estimator_.named_steps["rf"]
        P = np.array([b.predict(skala.transform(X_akt.values)) for b in wald.estimators_])  # Baum × Titel
        beweglich = P.std(axis=1) > 1e-12            # Bäume mit konstanter Prognose: keine Korrelation
        Q = P[beweglich]
        Cq = np.corrcoef(Q)
        rho = (Cq.sum() - np.trace(Cq)) / (len(Q) * (len(Q) - 1))   # Mittel aller Paare ohne Diagonale
        R = y_real[None, :] - P                      # Fehler jedes Baums je Titel
        sd = np.sqrt((R ** 2).mean(axis=1))          # Breiman: Wurzel des mittleren quadratischen Fehlers
        pe_baum = (sd ** 2).mean()                   # PE*(Baum): mittlerer Fehler eines Baums
        pe_wald = ((y_real - P.mean(axis=0)) ** 2).mean()   # PE*(Wald): Fehler der gemittelten Prognose
        rho_quer = pe_wald / sd.mean() ** 2          # Breimans gewichtete Korrelation der Fehler
        zeilen.append(dict(termin=termin.date(), rho=rho, rho_quer=rho_quer, verh=pe_wald / pe_baum,
                           ok=bool(pe_wald <= rho_quer * pe_baum + 1e-15), wie_lauf=bool(wie_lauf)))
        if i % 10 == 0:
            print(f"    {i:>3} von {len(termine)} …", flush=True)
    return pd.DataFrame(zeilen)


def pruefe() -> Pruefung:
    """Alle Zahlen aus Kapitel 3, in der Reihenfolge der Arbeit."""
    p = Pruefung("Kapitel 3 — Random-Forest-gestützte Renditeprognose")
    md, ret = monatsdaten_aufbauen()
    termine = gewichte()["Risk Parity"].index                # die 119 Entscheidungstermine
    # Von der Suche gewählte Einstellungen je Termin: nur für den Lauf der Arbeit hinterlegt.
    hp = pd.read_csv(DATEN / "rf_hyperparameter.csv", comment="#") if LAUF_DER_ARBEIT else None

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("3.1 Prognose als Regressionsproblem")
    p.zahl("Kennzahlen je Titel und Monat (Merkmale des Waldes)", "zwanzig", len(FEATURE_COLS))
    # Zielgröße: Monatsrenditen aller Titel im Auswertungszeitraum
    panel = md.loc[(md.index >= "2015-01-31") & (md.index <= "2024-12-31"),
                   ["ticker", "monthly_ret"]].dropna()
    r = panel["monthly_ret"]
    p.zahl("Titel, für die stets null prognostiziert wird", "fünfzehn", panel["ticker"].nunique())
    # RMSE der Nullprognose = Wurzel des mittleren QUADRATS der Renditen
    # (nicht zentriert, wie im Nenner des R² bei Gu/Kelly/Xiu).
    rmse0 = np.sqrt((r ** 2).mean())
    p.zahl("RMSE der Nullprognose", "7,48", rmse0 * 100, "%")
    p.offen("angenommenes R² im Gedankenexperiment", "1 %", "Annahme eines Rechenbeispiels")
    p.merke("angenommenes R² im Gedankenexperiment", "einem Prozent")
    p.zahl("so viel sinkt der RMSE dann", "0,5", (1 - np.sqrt(1 - 0.01)) * 100, "%", ungefaehr=True,
           notiz="RMSE sinkt auf √(1 − R²) seines Werts: 1 − √(1 − 0,01)")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("3.2 Entscheidungsbäume und das Split-Kriterium")
    groessen = pd.Series([len(trainingsfenster(md, t)[0]) for t in termine])
    median = groessen.median()
    p.zahl("Bereiche eines Baums der Tiefe drei", "acht", 2 ** 3, notiz="jede Ebene teilt jeden Bereich in zwei: 2³")
    p.zahl("Zeilen je Bereich bei Tiefe drei", "66", median / 8, ungefaehr=True,
           notiz=f"{median:.0f} Trainingszeilen ÷ 8 Bereiche")
    p.zahl(f"Trainingszeilen je Termin (Median über {len(termine)} Termine)", "525", median, ungefaehr=True)
    p.aussage("Zeilen je Bereich bei Tiefe zehn", "weniger als eine", median / 2 ** 10 < 1,
              de(median / 2 ** 10, 2), notiz=f"{median:.0f} Trainingszeilen auf 2¹⁰ = 1024 Bereiche",
              woertlich=True)
    train, pruef = tiefensweep(md, termine)
    p.zahl("Trainingsfehler bei Tiefe 14 in % des Fehlers bei Tiefe eins", "22", train * 100, "%",
           notiz=f"Einzelbaum, mittlerer quadratischer Fehler auf dem Lernstoff, Mittel über alle {len(termine)} Termine")
    p.zahl("Anstieg des Kreuzvalidierungsfehlers von Tiefe eins auf vierzehn", "75", pruef * 100, "%",
           notiz="derselbe Einzelbaum, Fehler auf den Prüfblöcken der Purged-Kreuzvalidierung, "
                 f"Mittel über alle {len(termine)} Termine")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("3.3 Bagging als Diversifikation")
    herkunft = "neu gerechnet" if BAUMKORR else "aus skripte/daten/baumkorrelation.csv"
    if BAUMKORR:
        D = baumkorrelation(md, ret, termine)
        if SPEICHERN and LAUF_DER_ARBEIT:
            D.to_csv(REFERENZ, index=False)
            print(f"  Referenzdatei geschrieben: {REFERENZ.relative_to(ROOT)}")
        elif SPEICHERN:
            print("  Referenzdatei nicht überschrieben: Sie gehört zum Lauf der Arbeit.")
        elif LAUF_DER_ARBEIT:
            alt = pd.read_csv(REFERENZ)
            gleich = np.allclose(alt[["rho", "rho_quer", "verh"]].values, D[["rho", "rho_quer", "verh"]].values)
            p.aussage("Neurechnung gleich der Referenzdatei", "gleich", gleich,
                      "gleich" if gleich else "weicht ab",
                      notiz="Spalten rho, rho_quer und verh von skripte/daten/baumkorrelation.csv")
    elif LAUF_DER_ARBEIT:
        D = pd.read_csv(REFERENZ)
    else:
        D = None
        p.hinweis("Korrelation der Bäume — skripte/daten/baumkorrelation.csv gehört zum Lauf der Arbeit; "
                  "für diesen Lauf mit --baumkorrelation neu rechnen (rund 55 Minuten)")
    r_tag = titelrenditen()
    oben = r_tag.corr().values[np.triu_indices(len(TICKERS), k=1)]   # jede Paarkorrelation einmal
    if D is not None:
        p.zahl("mittlere Korrelation der Baumprognosen ρ", "0,21", D.rho.mean(), ungefaehr=True,
               notiz=f"Mittel über alle Termine, {herkunft}")
        p.aussage("ρ ist halb so groß wie die Korrelation der Aktien", "halb so viel",
                  0.45 <= D.rho.mean() / oben.mean() <= 0.55, f"Faktor {de(D.rho.mean() / oben.mean(), 2)}",
                  notiz=f"ρ = {de(D.rho.mean(), 3)} gegen {de(oben.mean(), 3)} (mittlere Paarkorrelation der "
                        "Tagesrenditen)", woertlich=True)
    p.zahl(f"mittlere Korrelation der {len(TICKERS)} Aktien", "0,42", oben.mean(),
           notiz="Mittel der paarweisen Korrelationen der Tagesrenditen")
    if hp is not None:
        B = hp["n_estimators"].median()
        p.zahl(f"Bäume je Wald (Median der {len(hp)} gewählten Wälder)", "321", B,
               notiz="aus skripte/daten/rf_hyperparameter.csv")
    else:
        B = None
        p.hinweis("Bäume je Wald — die gewählten Einstellungen je Termin stehen nur im Laufprotokoll; "
                  "skripte/daten/rf_hyperparameter.csv gehört zum Lauf der Arbeit")
    if D is not None and B is not None:
        # Formel 24: Var = σ²(ρ + (1−ρ)/B). Von einem Baum (σ²) bis zu unendlich
        # vielen (ρσ²) sinkt die Varianz um (1−ρ)σ². Bei B Bäumen ist davon der
        # Anteil [1 − (ρ + (1−ρ)/B)] / (1 − ρ) = 1 − 1/B erreicht — unabhängig von ρ.
        rho = D.rho.mean()
        erreicht = (1 - (rho + (1 - rho) / B)) / (1 - rho)
        p.zahl(f"erreichte Varianzreduktion bei {B:.0f} Bäumen", "99,7", erreicht * 100, "%",
               notiz="Anteil der bei unbegrenzt vielen Bäumen möglichen; = 1 − 1/B")
    if D is not None:
        p.zahl("Termine", "119", len(D))
        p.zahl("gewichtete Korrelation der Fehler ρ̄ (Breiman)", "0,84", D.rho_quer.mean(),
               notiz=f"Mittel über alle Termine, {herkunft}")
        p.zahl("Fehler des Waldes in % des Einzelbaumfehlers", "82", D.verh.mean() * 100, "%",
               notiz=f"Mittel über alle Termine, {herkunft}")
        p.aussage("Schranke an jedem Termin eingehalten", "an jedem Termin", bool(D.ok.all()),
                  f"{int(D.ok.sum())} von {len(D)}", notiz="Fehler des Waldes ≤ ρ̄ · mittlerer Fehler eines Baums "
                  "(Formel 25)", woertlich=True)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("3.4 Merkmale und ihre Begründung")
    alle = sorted(sum(GRUPPEN.values(), []))
    p.zahl("Kennzahlen", "20", len(FEATURE_COLS))
    p.aussage("Gruppen, in die sie sich teilen", "fünf", len(GRUPPEN) == 5 and alle == sorted(FEATURE_COLS),
              f"{len(GRUPPEN)}", notiz=" · ".join(f"{g} {len(m)}" for g, m in GRUPPEN.items())
              + " (zusammen genau die Merkmale aus FEATURE_COLS)", woertlich=True)
    p.quelle("Jegadeesh/Titman: Momentum wirkt über drei bis zwölf Monate", "drei bis zwölf", "Fn 68")
    p.quelle("Jegadeesh/Titman: Effekt baut sich ab nach etwa", "einem Jahr", "Fn 69")
    fenster = inspect.signature(compute_alpha_beta).parameters["window"].default
    p.zahl("Regressionsfenster für Alpha und Beta", "63", fenster, "Handelstage")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("3.5 Was ist vom maschinellen Lernen zu erwarten?")
    p.quelle("Gu/Kelly/Xiu: untersuchte Titel", "30 000", "Fn 70")
    p.quelle("Gu/Kelly/Xiu: Jahre", "sechzig", "Fn 70")
    p.quelle("Gu/Kelly/Xiu: davon zur Auswertung", "30", "Fn 70")
    p.quelle("Gu/Kelly/Xiu: Merkmale", "920", "Fn 70")
    p.quelle("Gu/Kelly/Xiu: R² ihres Random Forest (Prozent)", "0,33", "Fn 70")
    p.quelle("Gu/Kelly/Xiu: R² des besten Verfahrens (Prozent)", "0,40", "Fn 70")
    p.quelle("Gu/Kelly/Xiu: gewählte Baumtiefen", "ein bis fünf", "Fn 71")
    tiefe = RFPortfolioOptimizer()._param_grid()["rf__max_depth"]
    kleinste, groesste = tiefe.support()          # randint(3, 15) zieht 3 bis 14
    p.zahl("größte zulässige Tiefe im Suchraum", "14", groesste)
    if hp is not None:
        p.zahl("Monate, in denen die kleinste Tiefe gewählt wird", "98", int((hp["max_depth"] == kleinste).sum()),
               notiz="aus skripte/daten/rf_hyperparameter.csv (von der Suche gewählte Einstellungen je Termin)")
        p.zahl("von Terminen", "119", len(hp))
    else:
        p.hinweis("Monate mit der kleinsten Tiefe — die gewählten Einstellungen je Termin stehen nur im "
                  "Laufprotokoll; skripte/daten/rf_hyperparameter.csv gehört zum Lauf der Arbeit")
    p.zahl("kleinste zulässige Tiefe", "drei", kleinste)
    p.quelle("R², mit dem gerechnet wird: das des Random Forest bei Gu/Kelly/Xiu", "0,33", "Fn 70")
    p.zahl("RMSE der Nullprognose (aus § 3.1)", "7,48", rmse0 * 100, "%")
    # Gerechnet wird wie in der Arbeit mit dem auf zwei Stellen gerundeten RMSE aus § 3.1.
    gerundet = float(runde(rmse0 * 100, 2))
    p.zahl("RMSE bei einem R² von 0,33 %", "7,47", gerundet * np.sqrt(1 - 0.0033), "%",
           notiz=f"{de(gerundet, 2)} % · √(1 − 0,0033) = {de(gerundet * np.sqrt(1 - 0.0033), 4)} %; mit dem "
                 f"ungerundeten RMSE {de(rmse0 * np.sqrt(1 - 0.0033) * 100, 4)} %")
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
