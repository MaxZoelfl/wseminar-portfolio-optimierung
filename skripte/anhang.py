"""
Anhang A bis C: jede Zahl in Bildunterschriften, Tabellen und Parameterliste,
in der Reihenfolge des Anhangs.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Der Anhang enthält Zahlen an drei Arten von Stellen:
  • in Bildunterschriften (Abbildungen A1–A4)   → aus Kursen und Strategierenditen
  • in Tabellen (A1–A3, B1–B4)                  → aus Gewichten, Renditen, Programmcode
                                                   und dem Laborprotokoll
  • in der Parameterliste (C1) und in C2        → aus den Einstellungen des Laufs, dem
                                                   Programmcode und der Versionsgeschichte (git)
Anhang D (Wortlaut der Prompts) und E (Herleitungen) enthalten keine Ergebniszahlen.

Formeln: dieselben Funktionen wie der Backtest und die Abbildungen
(portfolio/metrics.py, portfolio/theory_plots.py, portfolio/plots.py).

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/anhang.py                      # rund 10 Sekunden
"""

import ast
import inspect
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from portfolio import indicators, metrics, optimizers, significance, theory_plots   # noqa: E402
from portfolio.backtest import run_backtest           # noqa: E402
from portfolio.config import CFG                      # noqa: E402
from portfolio.indicators import FEATURE_COLS         # noqa: E402
from portfolio.plots import rollierender_sharpe       # noqa: E402
from portfolio.theory_plots import (jahreskennzahlen, diversifikation,   # noqa: E402
                                    effizienzrand_geschlossen)
from kapitel_4_methodik import SEKTOR                 # noqa: E402  (Sektorzuordnung nach GICS)

# Der Commit, den Anhang C2 als maßgeblichen Stand nennt. Ändert sich C2 in der
# Arbeit, wird nur diese Zeile angepasst.
C2_COMMIT, C2_DATUM = "67b7a3b", "15.10.2026"

# Module, die zusammen den "Rechenkern" bilden: alles, was eine Zahl der Arbeit verändern kann.
RECHENKERN = ["backtest", "cross_validation", "data", "indicators", "metrics", "optimizers", "significance"]

# Tabelle A3 wie in der Arbeit: Titel → (Markowitz, Random Forest, Risk Parity), mittleres Gewicht in %.
TABELLE_A3 = {
    "NVDA": ("17,3", "11,8", "4,2"), "UNH": ("14,5", "6,2", "6,6"), "MSFT": ("12,0", "5,3", "5,5"),
    "AAPL": ("10,9", "4,9", "5,7"), "PLD": ("7,9", "4,2", "6,4"), "JNJ": ("7,6", "6,7", "8,9"),
    "PG": ("7,2", "10,1", "9,1"), "XOM": ("4,7", "9,3", "6,5"), "CAT": ("4,2", "10,6", "5,4"),
    "LIN": ("4,1", "3,4", "6,5"), "JPM": ("2,7", "2,7", "5,3"), "KO": ("2,6", "7,1", "9,0"),
    "HON": ("2,3", "2,5", "6,1"), "VZ": ("1,7", "10,2", "9,6"), "GS": ("0,2", "4,9", "5,1")}
# Tabellen B2–B4 wie in der Arbeit, Zeilen in der Reihenfolge der Strategien.
TABELLE_B2 = {"Random Forest": ("1,350", "−1,81", "54,91", "623,3"),      # Sortino, VaR, Treffer, gesamt
              "Markowitz MVO": ("1,296", "−1,96", "55,75", "685,1"),
              "Equal Weight":  ("1,283", "−1,59", "55,27", "509,7"),
              "Risk Parity":   ("1,121", "−1,46", "55,31", "370,2")}
TABELLE_B3 = {"Random Forest": ("0,0589", "0,0200", "0,3181", "0,9357", "0,31", "1,56"),
              "Markowitz MVO": ("0,0571", "0,0200", "0,3181", "0,9063", "0,28", "1,53"),
              "Equal Weight":  ("0,0567", "0,0200", "0,3181", "0,9005", "0,28", "1,52"),
              "Risk Parity":   ("0,0498", "0,0200", "0,3180", "0,7901", "0,17", "1,41")}
TABELLE_B4 = {"Random Forest": ("−0,253", "9,68", "1,0112"), "Markowitz MVO": ("−0,290", "13,52", "1,0133"),
              "Equal Weight": ("−0,283", "18,76", "1,0151"), "Risk Parity": ("−0,274", "19,13", "1,0124")}


def ohne_kommentare(quelltext: str) -> str:
    """Programmlogik ohne Dokumentation: Syntaxbaum, aus dem alle Docstrings entfernt
    sind (Kommentare kennt der Syntaxbaum ohnehin nicht)."""
    baum = ast.parse(quelltext)
    for knoten in ast.walk(baum):
        rumpf = getattr(knoten, "body", None)
        if isinstance(rumpf, list) and rumpf and isinstance(rumpf[0], ast.Expr) \
                and isinstance(getattr(rumpf[0], "value", None), ast.Constant) \
                and isinstance(rumpf[0].value.value, str):
            knoten.body = rumpf[1:] or [ast.Pass()]
    return ast.dump(baum)


def git(*argumente) -> str:
    """Einen git-Befehl im Projektordner ausführen und seine Ausgabe zurückgeben."""
    return subprocess.run(["git", *argumente], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def zeitraum(p: Pruefung, r: pd.DataFrame, text: str = "2015–2024") -> None:
    """Die Bildunterschriften nennen „Tagesdaten 2015–2024“: erstes und letztes
    Jahr der Renditen, aus denen die Abbildung gezeichnet ist."""
    von, bis = (int(x) for x in text.split("–"))
    p.aussage("Datenzeitraum der Abbildung", text, (r.index[0].year, r.index[-1].year) == (von, bis),
              f"{r.index[0]:%d.%m.%Y} bis {r.index[-1]:%d.%m.%Y}", woertlich=True)


def pruefe() -> Pruefung:
    """Alle Zahlen aus Anhang A bis C, in der Reihenfolge des Anhangs."""
    p = Pruefung("Anhang A bis C")
    log = protokoll()
    meta, sig = log["experiment_meta"], log["significance"]
    rf = meta["risk_free_rate"]
    s = strategierenditen()
    G = gewichte()
    r = titelrenditen()
    T = len(s)
    N = len(TICKERS)
    mu, sigma, Sigma, _ = jahreskennzahlen(r)          # wie die Theorie-Abbildungen

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("Abbildung A1: Diversifikationsgrenze")
    _, cov_quer = diversifikation(Sigma.values)        # Formel 6, wie das Bild
    p.zahl("Titelzahl, ab der die Kurve extrapoliert", "15", N)
    p.zahl("Grenzwert √C̄", "16,68", np.sqrt(cov_quer) * 100, "%")
    zeitraum(p, r)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("Abbildung A2: Effizienzrand mit Kapitalmarktlinie")
    rand = effizienzrand_geschlossen(mu.values, Sigma.values, rf)   # Formeln 9 bis 11
    p.zahl("Titel", "fünfzehn", N)
    p.zahl("risikofreier Zins", "vier", rf * 100, "%")
    p.zahl("Handelstage im Rückblick", "2495", T)
    p.zahl("Schätzfenster des Backtests (zum Vergleich)", "119", len(G["Risk Parity"]))
    p.zahl("Tangentialportfolio: Volatilität", "52,8", rand["sig_tan"] * 100, "%")
    p.zahl("Tangentialportfolio: erwartete Rendite", "83,4", rand["mu_tan"] * 100, "%")
    p.zahl("Tangentialportfolio: größter Leerverkauf", "79", -rand["w_tan"].min() * 100, "%")
    zeitraum(p, r)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("Abbildung A3: Merkmalswichtigkeit des Random Forests")
    letzter = G["Random Forest"].index[-1]
    p.aussage("Entscheidungstermin des zuletzt trainierten Waldes", "November 2024",
              (letzter.year, letzter.month) == (2024, 11), f"{letzter:%d.%m.%Y}", woertlich=True)
    p.offen("Summe der Wichtigkeiten", "100 Prozent", "Eigenschaft des Maßes: scikit-learn teilt die "
            "Wichtigkeiten so, dass sie sich zu 1 addieren")
    # Welche Tagesdaten stecken im Lernstoff dieses Waldes? Fenster: drei Jahre bis zum
    # Termin. Die älteste Monatszeile enthält das Zwölf-Monats-Momentum, das 252
    # Handelstage zurückreicht. Die jüngste Zeile liegt vor "Termin minus ein Monat"
    # (§ 4.3); ihr Zielwert ist die Rendite des Monats danach.
    alle_tage = titelrenditen(nur_auswertungszeitraum=False).index
    erste_zeile = letzter - pd.DateOffset(years=meta["train_years"])
    erster_tag_im_monat = alle_tage[alle_tage >= erste_zeile - pd.offsets.MonthBegin(1)][0]
    fruehester = alle_tage[alle_tage.get_loc(erster_tag_im_monat) - 251]
    letzte_zeile = pd.offsets.MonthEnd().rollback(letzter - pd.DateOffset(months=1))
    spaetester = alle_tage[alle_tage <= letzte_zeile + pd.offsets.MonthEnd(1)][-1]
    p.aussage("Tagesdaten, aus denen dieser Wald gelernt hat", "November 2020 bis Oktober 2024",
              (fruehester.year, fruehester.month, spaetester.year, spaetester.month) == (2020, 11, 2024, 10),
              f"{fruehester:%d.%m.%Y} bis {spaetester:%d.%m.%Y}",
              notiz="vom Zwölf-Monats-Momentum der ältesten Zeile bis zum Zielwert der jüngsten",
              woertlich=True)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("Abbildung A4: Rollierender Zwölf-Monats-Sharpe-Quotient")
    roll = rollierender_sharpe(s).dropna()            # dieselbe Rechnung wie das Bild (plots.py)
    p.zahl("Fensterlänge", "252", HANDELSTAGE, "Handelstage")
    p.zahl("Zahl der Fenster", "2.244", len(roll))
    p.zahl("Fenster mit höherem Sharpe-Quotienten bei Markowitz als bei 1/N", "50,1",
           (roll["Markowitz MVO"] > roll["Equal Weight"]).mean() * 100, "%")
    wert = (1 + s).cumprod()
    p.zahl("Handelstage, an denen Markowitz' kumulierte Rendite über 1/N liegt", "97,4",
           (wert["Markowitz MVO"] > wert["Equal Weight"]).mean() * 100, "%",
           notiz="nicht zu verwechseln mit 86,8 % in § 6.1 (dort: vor allen drei anderen)")
    zeitraum(p, s)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("Tabelle A1: Das Anlagenuniversum")
    sektoren = {}
    for t in meta["tickers"]:
        sektoren.setdefault(SEKTOR[t], []).append(t)
    p.aussage("Titel und Kürzel wie im Code (portfolio/config.py)", "Kürzel",
              sorted(meta["tickers"]) == sorted(SEKTOR), f"{len(meta['tickers'])} Kürzel",
              notiz=f"{len(sektoren)} Sektoren: " + " · ".join(f"{s_}: {', '.join(t)}" for s_, t in sektoren.items()))

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("Tabelle A2: Die Zwanzig Merkmale des Random Forests")
    std = lambda f, name: inspect.signature(f).parameters[name].default   # Voreinstellung im Code
    p.zahl("Merkmale", "Zwanzig", len(FEATURE_COLS))
    aggr = inspect.getsource(indicators.aggregate_to_monthly)
    mittel_teil = aggr[aggr.index('for col in ["mom_21d"'):aggr.index("feat[\"ticker\"]")]
    p.aussage("Momentum und Volatilität als Monatsmittel, die übrigen als Stichtagswert", "Monatsmittel",
              all(c in mittel_teil for c in ["mom_21d", "mom_63d", "mom_126d", "mom_252d", "vol_21d", "vol_63d"])
              and ".mean()" in mittel_teil, "steht so im Code", notiz="aggregate_to_monthly", woertlich=True)
    for zahl in std(indicators.compute_momentum, "periods"):
        p.zahl("Momentum-Fenster", str(zahl), zahl, "Handelstage")
    p.zahl("RSI-Fenster", "14", std(indicators.compute_rsi, "period"), "Tage")
    p.aussage("RSI auf 0 bis 100 skaliert", "100",
              "100 - (100 / (1 + rs))" in inspect.getsource(indicators.compute_rsi), "steht so im Code",
              notiz="Code: 100 − 100/(1 + RS)", woertlich=True)
    p.offen("RSI: überkauft über / überverkauft unter", "70 / 30", "übliche Deutung, keine Einstellung des Codes")
    p.zahl("MACD: kurzer Durchschnitt", "12", std(indicators.compute_macd, "fast"), "Tage")
    p.zahl("MACD: langer Durchschnitt", "26", std(indicators.compute_macd, "slow"), "Tage")
    p.zahl("MACD-Signal: Glättung", "neun", std(indicators.compute_macd, "signal"), "Tage")
    p.zahl("Bollinger: Abstand der Bänder in Standardabweichungen", "2", std(indicators.compute_bollinger, "n_std"))
    p.zahl("Bollinger: Durchschnitt über", "20", std(indicators.compute_bollinger, "period"), "Tage")
    p.offen("%B: unteres / oberes Band", "0 / 1", "Definition von %B")
    for zahl in std(indicators.compute_volatility_features, "periods"):
        p.zahl("Volatilitäts-Fenster", str(zahl), zahl, "Handelstage")
    p.aussage("Volatilität mit √252 auf ein Jahr hochgerechnet", "252",
              "np.sqrt(252)" in inspect.getsource(indicators.compute_volatility_features),
              "steht so im Code", woertlich=True)
    p.zahl("Regressionsfenster für Beta und Alpha", "63", std(indicators.compute_alpha_beta, "window"), "Handelstage")
    p.offen("Beispiel: Beta von", "1,5", "Rechenbeispiel zur Deutung, keine Zahl aus den Daten")
    p.aussage("Alpha mit 252 auf ein Jahr hochgerechnet", "252",
              "alpha * 252" in inspect.getsource(indicators.compute_alpha_beta), "steht so im Code", woertlich=True)
    p.zahl("Ränge: unter so vielen Titeln", "fünfzehn", N)
    # Perzentilrang mit pandas rank(pct=True): kleinster Wert 1/15, größter 1
    rang = pd.Series(np.arange(N, dtype=float)).rank(pct=True)
    p.aussage("Rang des kleinsten Werts", "1/15", np.isclose(rang.min(), 1 / N), de(rang.min(), 4),
              notiz="pandas rank(pct=True), wie in portfolio/indicators.py", woertlich=True)
    p.zahl("Rang des größten Werts", "1", rang.max())

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("Tabelle A3: Mittlere Portfoliogewichte je Titel")
    mittel = {n: G[n].mean() * 100 for n in STRATEGIEN}
    for titel, werte in TABELLE_A3.items():
        for name, text in zip(["Markowitz MVO", "Random Forest", "Risk Parity"], werte):
            p.zahl(f"{titel}, {name}", text, mittel[name][titel], "%")
        p.zahl(f"{titel}, 1/N", "6,67", mittel["Equal Weight"][titel], "%")
    for name, lo, hi in [("Markowitz MVO", "0,2", "17,3"), ("Random Forest", "2,5", "11,8"),
                         ("Risk Parity", "4,2", "9,6")]:
        p.zahl(f"Spannweite, {name}: kleinstes mittleres Gewicht", lo, mittel[name].min(), "%")
        p.zahl(f"Spannweite, {name}: größtes mittleres Gewicht", hi, mittel[name].max(), "%")
    for name, vier in [("Markowitz MVO", "54,75"), ("Random Forest", "42,75"),
                       ("Risk Parity", "36,64"), ("Equal Weight", "26,67")]:
        p.zahl(f"Summe der größten Vier, {name}", vier, mittel[name].nlargest(4).sum(), "%")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("Tabelle B1: Definition der Kennzahlen")
    p.zahl("CAGR: Handelstage je Jahr im Exponenten 252/T", "252", std(metrics.cagr, "freq"))
    p.zahl("Volatilität: Hochrechnung mit √252", "252", std(metrics.annualized_vol, "freq"))
    p.zahl("Sortino-Quotient: Hochrechnung mit √252", "252", std(metrics.sortino_ratio, "freq"))
    quelle_kz = inspect.getsource(metrics.compute_metrics)
    p.aussage("Value at Risk: 5-%-Quantil der Tagesrenditen", "0,05", "np.percentile(r, 5)" in quelle_kz,
              "steht so im Code", notiz="np.percentile(r, 5)", woertlich=True)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("Tabelle B2: Weitere Kennzahlen")
    for name, (so, var, hit, ges) in TABELLE_B2.items():
        x = s[name]
        p.zahl(f"{name}: Sortino-Quotient", so, metrics.sortino_ratio(x, rf))
        p.zahl(f"{name}: Value at Risk 95 % (täglich)", var, np.percentile(x, 5) * 100, "%")
        p.zahl(f"{name}: Trefferquote", hit, (x > 0).mean() * 100, "%")
        p.zahl(f"{name}: Gesamtrendite", ges, ((1 + x).prod() - 1) * 100, "%")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("Tabelle B3: Standardfehler und Konfidenzintervalle der Sharpe-Quotienten")
    for name, (st, setag, sej, sj, lo, hi) in TABELLE_B3.items():
        S_tag = sharpe_exakt(name, je_tag=True)                # volle Genauigkeit aus dem Lauf
        se_tag = np.sqrt((1 + 0.5 * S_tag ** 2) / T)          # Formel 27
        S_jahr, se_jahr = S_tag * np.sqrt(HANDELSTAGE), se_tag * np.sqrt(HANDELSTAGE)
        p.zahl(f"{name}: Ŝ je Tag", st, S_tag)
        p.zahl(f"{name}: Standardfehler je Tag", setag, se_tag)
        p.zahl(f"{name}: Standardfehler je Jahr", sej, se_jahr)
        p.zahl(f"{name}: Ŝ je Jahr", sj, S_jahr)
        p.zahl(f"{name}: 95-%-Intervall, unten", lo, S_jahr - 1.96 * se_jahr)
        p.zahl(f"{name}: 95-%-Intervall, oben", hi, S_jahr + 1.96 * se_jahr)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("Tabelle B4: Schiefe und Wölbung der Tagesrenditen")
    p.offen("Wölbung der Normalverteilung", "3", "mathematische Konstante")
    p.zahl("geschätzt aus so vielen Tagesrenditen", "2495", T)
    for name, (g3t, g4t, nt) in TABELLE_B4.items():
        v = sig["deflated_sharpe"][name]                       # Werte aus dem Lauf
        g3, g4, S = v["skew"], v["kurtosis"], v["sr_per_period"]
        p.zahl(f"{name}: Schiefe", g3t, g3)
        p.zahl(f"{name}: Wölbung", g4t, g4)
        p.zahl(f"{name}: Nenner der DSR (Formel 34)", nt, np.sqrt(1 - g3 * S + (g4 - 1) / 4 * S ** 2))

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("C1: Parameter des Backtests")
    p.zahl("Anlageuniversum: US-Standardwerte", "15", meta["n_assets"])
    p.aussage("Referenzreihe", "SPY", CFG.spy_ticker == "SPY", CFG.spy_ticker, woertlich=True)
    p.aussage("Datenbestand", "01.01.2013 – 31.12.2024",
              (CFG.start_date, CFG.end_date) == ("2013-01-01", "2024-12-31"),
              f"{CFG.start_date} bis {CFG.end_date}", woertlich=True)
    p.aussage("Beginn des Backtests", "01.01.2015", meta["backtest_start"] == "2015-01-01",
              meta["backtest_start"], woertlich=True)
    p.zahl("Rebalancing-Termine", "119", len(G["Risk Parity"]))
    p.zahl("Schätzfenster", "3", meta["train_years"], "Jahre")
    quelltext = inspect.getsource(run_backtest)
    for zahl, stelle, was in [("252", "< 252", "Handelstage"), ("24", "< 24", "Monate Merkmale"),
                              ("12", "< 12", "Monatszeilen je Titel")]:
        p.aussage(f"Mindesthistorie: {was}", zahl, stelle in quelltext, "steht so im Code",
                  notiz=f"Code: {stelle}", woertlich=True)
    p.zahl("risikofreier Zins", "4", rf * 100, "% p. a.")
    p.zahl("… umgelegt auf Handelstage", "252", std(metrics.sharpe_ratio, "freq"))
    p.zahl("Positionsobergrenze", "20", meta["max_weight"] * 100, "% je Titel")
    p.zahl("Transaktionskosten", "10", meta["transaction_cost"] * 10_000, "bp des Umschlags")
    p.zahl("Umschlagschranke", "30", meta["rf_turnover_limit"] * 100, "% je Monat")
    p.aussage("Rückfall auf Minimum-Varianz", "ja", meta["min_variance_fallback"] is True,
              str(meta["min_variance_fallback"]))
    p.aussage("Kreuzvalidierung: Purged", "Purged", meta["use_purged_cv"] is True, str(meta["use_purged_cv"]),
              woertlich=True)
    p.zahl("Kreuzvalidierung: Blöcke", "5", meta["rf_cv_splits"])
    p.zahl("Kreuzvalidierung: Sperrzone", "2", meta["cv_embargo"] * 100, "%")
    p.zahl("Hyperparametersuche: Kandidaten je Termin", "30", meta["rf_n_iter"])
    gitter = optimizers.RFPortfolioOptimizer()._param_grid()
    bereich = lambda name: gitter[f"rf__{name}"].support()   # kleinster und größter möglicher Wert
    for name, text, was in [("n_estimators", "100–499", "Bäume"), ("max_depth", "3–14", "Tiefe"),
                            ("min_samples_leaf", "3–19", "Blattgröße"),
                            ("max_features", "0,3–0,9", "Merkmalsanteil"),
                            ("max_samples", "0,6–0,95", "Stichprobenanteil")]:
        lo, hi = bereich(name)
        ist = f"{lo:g}–{hi:g}".replace(".", ",")
        p.aussage(f"Suchraum {was} ({name})", text, ist == text, ist, woertlich=True)
    p.aussage("Parallelverarbeitung aus (n_jobs = 1)", "aus", CFG.deterministic is True,
              f"deterministic={CFG.deterministic}")
    p.offen("Kursdatei: Abruf", "15.08.2026", "steht im Laufprotokoll des Abrufs, nicht in den Ergebnisdateien")
    k = kurse()
    p.zahl("Kursdatei: Handelstage", "3019", len(k))
    p.zahl("Kursdatei: Reihen (15 Titel und SPY)", "16", k.shape[1])
    beliebig = next(iter(sig["sharpe_difference_tests"].values()))
    p.zahl("Signifikanztest: Bootstrap-Ziehungen", "4999", beliebig["n_boot"])
    p.zahl("Signifikanztest: T", "2495", beliebig["T"])
    p.zahl("Blocklänge ⌈T^(1/3)⌉", "14", beliebig["block_size"])
    p.zahl("HAC-Bandbreite ⌊4(T/100)^(2/9)⌋", "8", significance._nw_bandwidth(T), "Handelstage")
    startwerte = [inspect.signature(significance.sharpe_difference_test).parameters["seed"].default,
                  theory_plots.THEORY_SEED]
    p.aussage("Startwert aller Zufallsverfahren", "42",
              all(x == 42 for x in startwerte) and "random_state=42" in inspect.getsource(optimizers),
              "Bootstrap, Titelauswahl A1 und Random Forest: 42", woertlich=True)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("C2: Quellcode")
    try:
        for commit, datum in [(C2_COMMIT, C2_DATUM), ("b8eeea9", "15.08.2026")]:
            ist = git("show", "-s", "--format=%ad", "--date=format:%d.%m.%Y", commit).strip()
            p.merke(f"Commit {commit}", commit)
            p.aussage(f"Commit {commit} vom", datum, ist == datum, ist, woertlich=True)
        # Beschreibt der in C2 genannte Commit noch den heutigen Stand? Danach darf sich
        # höchstens diese Datei geändert haben (in ihr steht C2_COMMIT, und ein Commit
        # kann seine eigene Kennung nicht enthalten). Sonst: kein Fehler, aber ein Hinweis.
        danach = [d for d in git("diff", "--name-only", C2_COMMIT, "HEAD").split()
                  if d != "skripte/anhang.py"]
        if danach:
            p.offen("Dateien, die sich nach diesem Commit geändert haben", str(len(danach)),
                    f"C2 nennt {C2_COMMIT}; C2, das Literaturverzeichnis und C2_COMMIT oben in "
                    "diesem Skript auf den neuesten Commit setzen")
        veraendert = []
        for modul in RECHENKERN:                     # Programmlogik heute gegen b8eeea9
            alt = git("show", f"b8eeea9:portfolio/{modul}.py")
            neu = (ROOT / "portfolio" / f"{modul}.py").read_text(encoding="utf-8")
            if ohne_kommentare(alt) != ohne_kommentare(neu):
                veraendert.append(modul)
        p.aussage("Rechenkern seit b8eeea9", "unverändert", not veraendert,
                  "Programmlogik geändert in: " + ", ".join(veraendert),
                  notiz="seitdem nur Kommentare und Dokumentation geändert" if not veraendert else "",
                  woertlich=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        p.offen("Commits und Rechenkern", f"{C2_COMMIT} / b8eeea9", "git nicht verfügbar (kein Klon des Repositorys)")
    p.aussage("Python", "3.14", sys.version.startswith("3.14"), sys.version.split()[0], woertlich=True)
    for paket, text in [("NumPy", "2.4.4"), ("pandas", "3.0.2"), ("SciPy", "1.17.1"),
                        ("scikit-learn", "1.8.0"), ("yfinance", "1.3.0")]:
        p.aussage(paket, text, version(paket.lower()) == text, version(paket.lower()), woertlich=True)
    datum = pd.Timestamp(meta["timestamp"]).strftime("%d.%m.%Y")
    p.aussage("alle Zahlen aus dem Lauf vom", "25.08.2026", datum == "25.08.2026", datum, woertlich=True)
    p.offen("zweiter, unabhängiger Lauf: bitgleich", "zweiter", "nachgewiesen in LIMITATIONS.md, Abschnitt 12 "
            "(MD5-Vergleich der sieben CSV-Dateien)")
    p.aussage("LIMITATIONS.md, Abschnitt 12, erklärt Kursdatei und Parallelverarbeitung", "12",
              "## 12. Bit-genaue Reproduzierbarkeit" in (ROOT / "LIMITATIONS.md").read_text(encoding="utf-8"),
              "Abschnitt fehlt", woertlich=True)
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
