"""
Kontrollrechnung zu § 3.1 und § 3.2 — jede Zahl der Abschnitte nachvollziehbar.

WOZU DIESE DATEI?
Sie rechnet die eigenen Zahlen aus Kapitel 3 noch einmal nach und zeigt die
Zwischenergebnisse. Damit lässt sich jede Angabe im Text überprüfen, ohne den
vollständigen Backtest laufen zu lassen (der braucht knapp 30 Minuten).

Sie ändert NICHTS am Projekt und schreibt keine Datei. Sie liest die
eingefrorenen Kursdaten `data/prices.pkl`, den Ergebnisordner `output/` und
das Laufprotokoll `run.log`.

AUFRUF (aus dem Ordner Code/):
    venv/bin/python nachrechnen_kapitel3.py                    # Schritte 1-3, ~10 Sekunden
    venv/bin/python nachrechnen_kapitel3.py --sweep            # + Schritt 4, ~7 Minuten
    venv/bin/python nachrechnen_kapitel3.py --baumkorrelation  # + Schritt 5, ~55 Minuten

AUFBAU:
    Schritt 1  § 3.1 — das Panel, das der Wald zu lernen versucht
    Schritt 2  § 3.1 — was ein R²_oos von 0,33 % zahlenmäßig bedeutet
    Schritt 3  § 3.2 — Trainingsumfang je Termin und die gewählten Baumtiefen
    Schritt 4  § 3.2 — Tiefensweep: Einzelbaum gegen Wald  (nur mit --sweep)
    Schritt 5  § 3.3 — Korrelation der Bäume im Wald  (nur mit --baumkorrelation)

Gegenstück zu `nachrechnen_kapitel2.py`.
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from portfolio.config import TICKERS, SPY_TICKER, START_DATE, END_DATE, \
    TRAIN_YEARS, CV_EMBARGO
from portfolio.data import download_data
from portfolio.indicators import build_all_indicators, aggregate_to_monthly, \
    add_cross_sectional_ranks, FEATURE_COLS

import logging
logging.getLogger("portfolio_v4").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

SWEEP    = "--sweep" in sys.argv
BAUMKORR = "--baumkorrelation" in sys.argv


def trennlinie(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


# ────────────────────────────────────────────────────────────────────────
# Daten aufbauen — dieselbe Kette wie im Backtest, nur ohne die Optimierer.
# ────────────────────────────────────────────────────────────────────────
asset_px, asset_ret, spy_ret, tickers = download_data(
    TICKERS, SPY_TICKER, START_DATE, END_DATE
)
indikatoren = build_all_indicators(asset_px, asset_ret, spy_ret, tickers)
monatsdaten = add_cross_sectional_ranks(
    aggregate_to_monthly(indikatoren, asset_ret, tickers)
)

alle_monate  = monatsdaten.index.unique()
backtest_mon = alle_monate[alle_monate >= pd.Timestamp("2015-01-01")]


# ────────────────────────────────────────────────────────────────────────
# Schritt 1 — § 3.1: das Panel, das der Wald zu lernen versucht
# ────────────────────────────────────────────────────────────────────────
trennlinie("Schritt 1 — § 3.1: die Zielgröße im Auswertungszeitraum")

panel = monatsdaten.loc[
    (monatsdaten.index >= pd.Timestamp("2015-01-31"))
    & (monatsdaten.index <= pd.Timestamp("2024-12-31")),
    ["ticker", "monthly_ret"],
].dropna()
r = panel["monthly_ret"]

print(f"Monatsrenditen im Panel : {len(r)}  "
      f"({len(tickers)} Titel x {panel.index.nunique()} Monate)")
print(f"Mittelwert              : {r.mean():.4f}  = {100*r.mean():.2f} %")
print(f"Standardabweichung      : {r.std():.4f}  = {100*r.std():.2f} %")
print(f"kleinste / groesste     : {100*r.min():.2f} % / {100*r.max():.2f} %")

# Der Nenner des R²_oos nach Gu/Kelly/Xiu (2020), S. 2246, Gl. (19):
# die Summe der QUADRIERTEN Renditen, ausdruecklich OHNE Zentrierung.
E_r2 = (r ** 2).mean()
print(f"\nE[r^2]                  : {E_r2:.6f}")
print(f"Wurzel daraus           : {np.sqrt(E_r2):.4f}  = {100*np.sqrt(E_r2):.2f} %"
      "   <- der Nenner des R2_oos")

# Zerlegung in gemeinsame Marktbewegung und titelspezifischen Teil.
monatsmittel = r.groupby(level=0).mean()
abweichung   = r - r.index.map(monatsmittel)
anteil_markt = monatsmittel.var() / r.var()
print(f"\nStreuung der Monatsmittel   : {100*monatsmittel.std():.2f} %"
      f"   -> Varianzanteil {100*anteil_markt:.0f} % (gemeinsame Marktbewegung)")
print(f"Streuung um das Monatsmittel: {100*abweichung.std():.2f} %"
      f"   -> Varianzanteil {100*(1-anteil_markt):.0f} % (titelspezifisch)")


# ────────────────────────────────────────────────────────────────────────
# Schritt 2 — § 3.1: was ein R²_oos von 0,33 % zahlenmaessig bedeutet
# ────────────────────────────────────────────────────────────────────────
trennlinie("Schritt 2 — § 3.1: Groessenordnung des erreichbaren R2_oos")

print("Gu/Kelly/Xiu (2020) berichten fuer ihren Random Forest 0,33 % je Monat")
print("(S. 2251), fuer das beste Verfahren ueberhaupt 0,40 % (S. 2252).\n")
print(f"{'R2_oos':>8} {'Wurzelfehler':>14} {'Rueckgang':>12}")
basis = np.sqrt(E_r2)
for R2 in (0.0000, 0.0033, 0.0040, 0.0100):
    neu = np.sqrt(E_r2 * (1 - R2))
    print(f"{100*R2:>7.2f}% {100*neu:>13.2f}% {100*(1-neu/basis):>11.2f}%")
print("\nLesart: Die beste bekannte Prognose veraendert den typischen")
print("Prognosefehler in der zweiten Nachkommastelle.")


# ────────────────────────────────────────────────────────────────────────
# Schritt 3 — § 3.2: Trainingsumfang je Termin, gewaehlte Baumtiefen
# ────────────────────────────────────────────────────────────────────────
trennlinie("Schritt 3 — § 3.2: Lernstoff je Termin und gewaehlte Tiefe")


def trainingsfenster(monatsende):
    """Baut Merkmale und Zielwerte genau so wie backtest.run_backtest()."""
    start          = monatsende - pd.DateOffset(years=TRAIN_YEARS)
    fenster        = monatsdaten[(monatsdaten.index >= start)
                                 & (monatsdaten.index <= monatsende)]
    # Look-Ahead-Schutz aus § 4.3: der letzte Monat des Fensters faellt weg,
    # weil sein Zielwert genau der Monat waere, der prognostiziert werden soll.
    letzter_train  = monatsende - pd.DateOffset(months=1)
    merkmale, ziel = [], []
    for t in TICKERS:
        zeilen = fenster[(fenster["ticker"] == t)
                         & (fenster.index <= letzter_train)]
        spalten = [c for c in FEATURE_COLS if c in zeilen.columns]
        gueltig = zeilen[spalten + ["target_next_month"]].dropna()
        if len(gueltig) < 12:          # unter 12 Monaten Historie: auslassen
            continue
        merkmale.append(gueltig[spalten])
        ziel.append(gueltig["target_next_month"])
    return pd.concat(merkmale), pd.concat(ziel)


groessen = [len(trainingsfenster(m)[0]) for m in backtest_mon[:-1]]
g = pd.Series(groessen)
print(f"Termine                 : {len(g)}")
print(f"Trainingszeilen  Median : {g.median():.0f}"
      f"   ({len(TICKERS)} Titel x {g.median()/len(TICKERS):.0f} Monate)")
print(f"                 Spanne : {g.min():.0f} bis {g.max():.0f}")
print(f"                 Mittel : {g.mean():.0f}")
print(f"Merkmale                : {len(FEATURE_COLS)}")
print(f"Verhaeltnis Zeilen:Merkmale : {g.median()/len(FEATURE_COLS):.0f} : 1")

# Die gewaehlten Hyperparameter stehen im Laufprotokoll des maßgeblichen Laufs.
if os.path.exists("run.log"):
    treffer = re.findall(r"RF-Params: n_est=(\d+), depth=(\d+), leaf=(\d+)",
                         open("run.log", encoding="utf-8", errors="replace").read())
    hp = pd.DataFrame(treffer, columns=["n_est", "depth", "leaf"]).astype(int)
    print(f"\nGewaehlte Baumtiefe ueber {len(hp)} Termine "
          "(Suchraum laut config.py: 3 bis 14):")
    for tiefe, anzahl in hp.depth.value_counts().sort_index().items():
        print(f"   Tiefe {tiefe:>2} : {anzahl:>3} Monate"
              f"   ({100*anzahl/len(hp):.1f} %)")
    print(f"   Median {hp.depth.median():.0f} | Mittelwert {hp.depth.mean():.2f}"
          f" | hoechstens 5 in {100*(hp.depth<=5).mean():.1f} % der Monate")
    print(f"   Baeume je Wald: Median {hp.n_est.median():.0f}, "
          f"Spanne {hp.n_est.min()} bis {hp.n_est.max()}")
    print("   ACHTUNG: Der Suchraum beginnt bei 3. Die Suche waehlt in "
          f"{100*(hp.depth==3).mean():.1f} % der Monate diesen Rand.")
else:
    print("\n(run.log nicht gefunden — Baumtiefen uebersprungen)")

# Wie viele Blaetter eine Tiefe zulaesst, und wie viel Lernstoff je Blatt bleibt.
print(f"\nBlaetter und Belegung bei {g.median():.0f} Trainingszeilen:")
print(f"{'Tiefe':>6} {'max. Blaetter':>14} {'Zeilen je Blatt':>17}")
for tiefe in (3, 6, 10, 14):
    blaetter = 2 ** tiefe
    print(f"{tiefe:>6} {blaetter:>14} {g.median()/blaetter:>17.1f}")


# ────────────────────────────────────────────────────────────────────────
# Schritt 4 — § 3.2: Tiefensweep (nur mit --sweep, Laufzeit ~7 Minuten)
# ────────────────────────────────────────────────────────────────────────
if SWEEP:
    trennlinie("Schritt 4 — § 3.2: Einzelbaum gegen Wald ueber die Tiefe")

    # Absichtlich erst hier importiert und nicht am Dateianfang: sklearn
    # braucht mehrere Sekunden zum Laden. Die Schritte 1-3 kommen ohne aus,
    # und der haeufigste Aufruf ist der ohne Schalter.
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from portfolio.cross_validation import purged_kfold_splits

    TIEFEN  = list(range(1, 15))
    # Jeder vierte Termin — der Sweep kostet sonst eine halbe Stunde.
    STICHPROBE = [backtest_mon[i] for i in range(0, len(backtest_mon) - 1, 4)]
    print(f"{len(STICHPROBE)} Termine, Tiefen 1 bis 14, "
          "Purged CV mit 5 Bloecken wie im Backtest.")
    print("Die uebrigen Stellschrauben sind festgehalten (300 Baeume, "
          "min_samples_leaf=3,\nmax_features=0.45, max_samples=0.8), damit "
          "allein die Tiefe variiert.\n")

    ergebnis = {t: {"train": [], "cv": [], "wald": []} for t in TIEFEN}
    for monatsende in STICHPROBE:
        X, y = trainingsfenster(monatsende)
        folds = list(purged_kfold_splits(X.index, n_splits=5,
                                         embargo_pct=CV_EMBARGO))
        Xv, yv = X.values, y.values
        for tiefe in TIEFEN:
            tr, cv, wd = [], [], []
            for i_train, i_test in folds:
                sk = StandardScaler().fit(Xv[i_train])
                A, B = sk.transform(Xv[i_train]), sk.transform(Xv[i_test])
                baum = DecisionTreeRegressor(
                    max_depth=tiefe, min_samples_leaf=3, random_state=42
                ).fit(A, yv[i_train])
                tr.append(((yv[i_train] - baum.predict(A)) ** 2).mean())
                cv.append(((yv[i_test] - baum.predict(B)) ** 2).mean())
                wald = RandomForestRegressor(
                    n_estimators=300, max_depth=tiefe, min_samples_leaf=3,
                    max_features=0.45, max_samples=0.8,
                    random_state=42, n_jobs=1,
                ).fit(A, yv[i_train])
                wd.append(((yv[i_test] - wald.predict(B)) ** 2).mean())
            ergebnis[tiefe]["train"].append(np.mean(tr))
            ergebnis[tiefe]["cv"].append(np.mean(cv))
            ergebnis[tiefe]["wald"].append(np.mean(wd))
        print(".", end="", flush=True)
    print("\n")

    print(f"{'Tiefe':>6} {'Baum Training':>15} {'Baum CV':>12} "
          f"{'Wald CV':>12} {'Wald/Baum':>11}")
    for tiefe in TIEFEN:
        a = np.mean(ergebnis[tiefe]["train"])
        b = np.mean(ergebnis[tiefe]["cv"])
        c = np.mean(ergebnis[tiefe]["wald"])
        print(f"{tiefe:>6} {a:>15.6f} {b:>12.6f} {c:>12.6f} {c/b:>11.3f}")

    a1, a14 = np.mean(ergebnis[1]["train"]),  np.mean(ergebnis[14]["train"])
    b1, b14 = np.mean(ergebnis[1]["cv"]),     np.mean(ergebnis[14]["cv"])
    c1, c14 = np.mean(ergebnis[1]["wald"]),   np.mean(ergebnis[14]["wald"])
    print(f"\nVon Tiefe 1 auf 14:")
    print(f"  Trainingsfehler des Baums faellt auf {100*a14/a1:.0f} % "
          "— der Baum lernt die Daten auswendig")
    print(f"  Prueffehler des Baums steigt um     {100*(b14/b1-1):.0f} %")
    print(f"  Pruefehler des Waldes steigt um     {100*(c14/c1-1):.0f} % "
          "— das Mitteln faengt die Varianz ab")
else:
    print("\n(Schritt 4 uebersprungen — mit --sweep aufrufen, Laufzeit ~7 Minuten)")


# ────────────────────────────────────────────────────────────────────────
# Schritt 5 — § 3.3: Korrelation der Baeume  (nur mit --baumkorrelation)
# ────────────────────────────────────────────────────────────────────────
if BAUMKORR:
    trennlinie("Schritt 5 — § 3.3: wie stark sich die Baeume eines Waldes gleichen")

    from portfolio.optimizers import RFPortfolioOptimizer   # spaet, siehe Schritt 4

    print("Fuer jeden der 119 Termine wird der Wald genau so getunt wie im Backtest")
    print("und anschliessend jeder einzelne Baum ausgewertet. Laufzeit ~55 Minuten.\n")

    def mittlere_paarkorrelation(P):
        """Mittlere paarweise Pearson-Korrelation der Zeilen (= Baeume) von P.

        Baeume, die auf allen Auswertungspunkten denselben Wert vorhersagen,
        haben keine Streuung und damit keine definierte Korrelation; sie
        werden ausgelassen (kommt bei Tiefe 3 und 15 Punkten gelegentlich vor).
        """
        beweglich = P.std(axis=1) > 1e-12
        Q = P[beweglich]
        if len(Q) < 2:
            return np.nan
        C = np.corrcoef(Q)
        B = len(Q)
        return (C.sum() - np.trace(C)) / (B * (B - 1))

    zeilen = []
    for i, monatsende in enumerate(backtest_mon[:-1]):
        X, y = trainingsfenster(monatsende)

        # Prognosezeilen: je Titel die juengste vollstaendige Merkmalszeile,
        # genau wie backtest.run_backtest() sie baut.
        start   = monatsende - pd.DateOffset(years=TRAIN_YEARS)
        fenster = monatsdaten[(monatsdaten.index >= start)
                              & (monatsdaten.index <= monatsende)]
        aktuell = []
        for t in TICKERS:
            zeil = fenster[fenster["ticker"] == t]
            sp   = [c for c in FEATURE_COLS if c in zeil.columns]
            gue  = zeil[sp].dropna()
            aktuell.append(gue.iloc[-1] if len(gue)
                           else pd.Series(np.zeros(len(sp)), index=sp))
        X_aktuell = pd.DataFrame(aktuell)[X.columns]

        # Realisierte Rendite der Halteperiode — der Zielwert, gegen den
        # Breimans Residuen gemessen werden.
        halte = asset_ret[(asset_ret.index > monatsende)
                          & (asset_ret.index <= backtest_mon[i + 1])]
        y_real = ((1 + halte).prod() - 1).reindex(TICKERS).values

        rfo = RFPortfolioOptimizer()
        rfo.fit_with_tuning(X, y, sample_times=X.index)
        pipe   = rfo.best_estimator_
        skala  = pipe.named_steps["scaler"]
        wald   = pipe.named_steps["rf"]

        # Prognose JEDES einzelnen Baums auf den 15 Entscheidungszeilen.
        P = np.array([b.predict(skala.transform(X_aktuell.values))
                      for b in wald.estimators_])
        B = P.shape[0]
        R = y_real[None, :] - P                       # Residuen, B x 15

        # Breiman (2001), S. 26: sd(Theta) ist die Wurzel des ROHEN zweiten
        # Moments der Residuen, nicht die zentrierte Standardabweichung.
        sd         = np.sqrt((R ** 2).mean(axis=1))
        PE_baum    = (sd ** 2).mean()                 # PE*(tree)
        PE_wald    = ((y_real - P.mean(axis=0)) ** 2).mean()   # PE*(forest)
        rho_quer   = PE_wald / (sd.mean() ** 2)       # Gl. (14), exakt
        jensen     = (sd.mean() ** 2) / PE_baum       # <= 1

        zeilen.append(dict(
            B=B, rho_prognose=mittlere_paarkorrelation(P),
            sigma2=P.var(axis=0, ddof=1).mean(),
            rho_quer=rho_quer, jensen=jensen,
            PE_baum=PE_baum, PE_wald=PE_wald,
            schranke_haelt=PE_wald <= rho_quer * PE_baum + 1e-15,
        ))
        if i % 10 == 0:
            print(f"  {i:>3} von {len(backtest_mon)-1} …", flush=True)

    D = pd.DataFrame(zeilen)

    print("\n--- Bagging-Formel: Var(h_quer) = sigma^2/B + (1-1/B) rho sigma^2 ---")
    sigma = np.sqrt(D.sigma2.mean())
    rho   = D.rho_prognose.mean()
    B_med = D.B.median()
    real  = sigma * np.sqrt(1 / B_med + (1 - 1 / B_med) * rho)
    grenz = sigma * np.sqrt(rho)
    print(f"  sigma (ein Baum)             : {100*sigma:.3f} % je Monat")
    print(f"  rho   (Prognosekorrelation)  : {rho:.4f}"
          f"   Median {D.rho_prognose.median():.4f}")
    print(f"  B     (Median)               : {B_med:.0f}   -> 1/B = {1/B_med:.5f}")
    print(f"  Streuung des Waldes bei B    : {100*real:.3f} %")
    print(f"  Grenzwert fuer B -> unendlich: {100*grenz:.3f} %")
    print(f"  vom erreichbaren Rueckgang realisiert: "
          f"{100*(sigma-real)/(sigma-grenz):.1f} %")
    print(f"\n  Gegenstueck § 2.2: 26,98 % / rho_quer 0,420 / Grenze 16,68 %")
    print(f"  Anteil der Streuung, der bleibt:  Portfolio {16.68/26.98:.3f}"
          f"   Wald {grenz/sigma:.3f}")

    print("\n--- Breimans Schranke: PE*(Wald) <= rho_quer * PE*(Baum) ---")
    print(f"  rho_quer (Residuenkorrelation, Gl. 14): {D.rho_quer.mean():.4f}"
          f"   Median {D.rho_quer.median():.4f}")
    print(f"  tatsaechlich PE*(Wald)/PE*(Baum)      : "
          f"{(D.PE_wald/D.PE_baum).mean():.4f}")
    print(f"  Schranke eingehalten in {D.schranke_haelt.sum()} von {len(D)} Terminen")
    print(f"  Jensen-Faktor (E sd)^2 / E[sd^2]      : {D.jensen.mean():.4f}"
          f"   -> die Schranke ist auf {100*(1-D.jensen.mean()):.1f} % scharf")
    print("\n  ACHTUNG: rho (Prognosen) und rho_quer (Residuen) sind zwei")
    print("  verschiedene Groessen. Nie 'die Baeume sind zu 84 % korreliert'.")
else:
    print("(Schritt 5 uebersprungen — mit --baumkorrelation aufrufen, ~55 Minuten)")

print()
