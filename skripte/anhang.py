"""
Anhang A bis C: jede eigene Zahl in Bildunterschriften, Tabellen und Parameterliste nachgerechnet.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Der Anhang enthält Zahlen an drei Arten von Stellen:
  • in Bildunterschriften (Anhang A)            → aus Kursen und Strategierenditen
  • in Ergebnistabellen (Tabelle A3, B2–B4)      → aus Gewichten, Renditen und dem Laborprotokoll
  • in der Parameterliste (C1) und in C2        → aus den Einstellungen des Laufs, dem Programmcode
                                                   und der Versionsgeschichte (git)
Anhang D (Wortlaut der Prompts) und E (Herleitungen) enthalten keine Ergebniszahlen.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/anhang.py                      # wenige Sekunden
"""

import ast
import inspect
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from portfolio import indicators, optimizers, significance, theory_plots   # noqa: E402
from portfolio.backtest import run_backtest           # noqa: E402
from portfolio.config import CFG                      # noqa: E402
from portfolio.indicators import FEATURE_COLS         # noqa: E402
from portfolio.metrics import sortino_ratio           # noqa: E402

# Der Commit, den Anhang C2 als maßgeblichen Stand nennt. Ändert sich C2 in der
# Arbeit, wird nur diese Zeile angepasst.
C2_COMMIT, C2_DATUM = "0b10daa", "27.09.2026"

# Module, die zusammen den "Rechenkern" bilden: alles, was eine Zahl der Arbeit verändern kann.
RECHENKERN = ["backtest", "cross_validation", "data", "indicators", "metrics", "optimizers", "significance"]


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
    return subprocess.run(["git", *argumente], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def pruefe() -> Pruefung:
    p = Pruefung("Anhang A bis C")
    log = protokoll()
    meta, sig = log["experiment_meta"], log["significance"]
    s = strategierenditen()
    G = gewichte()
    r = titelrenditen()
    T = len(s)
    N = len(TICKERS)
    Sigma = r.cov() * HANDELSTAGE
    mu = r.mean() * HANDELSTAGE

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("A Abbildungen A1–A4 (Bildunterschriften)")
    C = Sigma.values
    p.zahl("A1: Grenzwert √C̄", "16,68", np.sqrt(C[~np.eye(N, dtype=bool)].mean()) * 100, "%")
    p.zahl("A1: Zahl der Titel", "15", N)
    p.zahl("A2: risikofreier Zins", "4", meta["risk_free_rate"] * 100, "%")
    p.zahl("A2: Handelstage im Rückblick", "2495", T)
    p.zahl("A2: Schätzfenster des Backtests", "119", len(G["Risk Parity"]))
    S_inv, eins, rf = np.linalg.inv(C), np.ones(N), meta["risk_free_rate"]
    w_tan = S_inv @ (mu.values - rf * eins) / (eins @ S_inv @ (mu.values - rf * eins))
    p.zahl("A2: Tangentialportfolio, Volatilität", "52,8", np.sqrt(w_tan @ C @ w_tan) * 100, "%")
    p.zahl("A2: Tangentialportfolio, Rendite", "83,4", w_tan @ mu.values * 100, "%")
    p.zahl("A2: größter Leerverkauf", "79", -w_tan.min() * 100, "%")
    letzter = G["Random Forest"].index[-1]
    p.aussage("A3: zuletzt trainierter Wald", "November 2024", (letzter.year, letzter.month) == (2024, 11),
              f"Entscheidungstermin {letzter.date()}", woertlich=True)
    p.zahl("A4: Fensterlänge", "252", HANDELSTAGE, "Handelstage")
    p.zahl("A4: Zahl der Fenster", "2.244", T - HANDELSTAGE + 1)
    roll = {n: s[n].rolling(HANDELSTAGE).apply(lambda x: sharpe(pd.Series(x), rf), raw=True).dropna()
            for n in ["Markowitz MVO", "Equal Weight"]}
    p.zahl("A4: Fenster mit Markowitz vorn", "50,1", (roll["Markowitz MVO"] > roll["Equal Weight"]).mean() * 100, "%")
    wert = (1 + s).cumprod()
    p.zahl("A4: Tage mit Markowitz-Kurve über 1/N", "97,4",
           (wert["Markowitz MVO"] > wert["Equal Weight"]).mean() * 100, "%",
           notiz="nicht zu verwechseln mit 86,8 % in § 6.1 (vor allen drei anderen)")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("A Tabelle A2: die zwanzig Merkmale")
    p.zahl("Zahl der Merkmale", "20", len(FEATURE_COLS))
    std = lambda f, name: inspect.signature(f).parameters[name].default
    p.aussage("Momentum-Fenster", "21 / 63 / 126 / 252", std(indicators.compute_momentum, "periods") == [21, 63, 126, 252],
              str(std(indicators.compute_momentum, "periods")))
    p.zahl("RSI-Fenster", "14", std(indicators.compute_rsi, "period"), "Tage")
    p.aussage("MACD-Durchschnitte", "12 / 26 / 9",
              (std(indicators.compute_macd, "fast"), std(indicators.compute_macd, "slow"),
               std(indicators.compute_macd, "signal")) == (12, 26, 9), "12 / 26 / 9")
    p.zahl("Bollinger-Fenster", "20", std(indicators.compute_bollinger, "period"), "Tage")
    p.zahl("Bollinger-Abstand", "2", std(indicators.compute_bollinger, "n_std"), "Standardabw.")
    p.aussage("Volatilitäts-Fenster", "21 / 63", std(indicators.compute_volatility_features, "periods") == [21, 63],
              str(std(indicators.compute_volatility_features, "periods")))
    p.zahl("Regressionsfenster Alpha/Beta", "63", std(indicators.compute_alpha_beta, "window"), "Tage")
    p.aussage("Alpha mit 252 hochgerechnet", "252", "alpha * 252" in inspect.getsource(indicators.compute_alpha_beta),
              "steht so im Code")
    p.offen("RSI über 70 / unter 30", "70 / 30", "übliche Deutung, keine Einstellung des Codes")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("A Tabelle A3: mittlere Portfoliogewichte je Titel")
    soll = {  # Titel: (Markowitz, Random Forest, Risk Parity) in Prozent, wie in der Tabelle
        "NVDA": ("17,3", "11,8", "4,2"), "UNH": ("14,5", "6,2", "6,6"), "MSFT": ("12,0", "5,3", "5,5"),
        "AAPL": ("10,9", "4,9", "5,7"), "PLD": ("7,9", "4,2", "6,4"), "JNJ": ("7,6", "6,7", "8,9"),
        "PG": ("7,2", "10,1", "9,1"), "XOM": ("4,7", "9,3", "6,5"), "CAT": ("4,2", "10,6", "5,4"),
        "LIN": ("4,1", "3,4", "6,5"), "JPM": ("2,7", "2,7", "5,3"), "KO": ("2,6", "7,1", "9,0"),
        "HON": ("2,3", "2,5", "6,1"), "VZ": ("1,7", "10,2", "9,6"), "GS": ("0,2", "4,9", "5,1")}
    mittel = {n: G[n].mean() * 100 for n in STRATEGIEN}
    abweichend = []
    for titel, werte in soll.items():
        for name, text in zip(["Markowitz MVO", "Random Forest", "Risk Parity"], werte):
            if de(mittel[name][titel], 1) != text:
                abweichend.append(f"{titel}/{KURZ[name]}: Text {text}, gerechnet {de(mittel[name][titel], 2)}")
    p.aussage("45 Einzelgewichte", "wie Tabelle", not abweichend,
              "alle 45 stimmen" if not abweichend else "; ".join(abweichend))
    p.zahl("1/N je Titel", "6,67", mittel["Equal Weight"].iloc[0], "%")
    for name, (lo, hi), vier in [("Markowitz MVO", ("0,2", "17,3"), "54,75"),
                                 ("Random Forest", ("2,5", "11,8"), "42,75"),
                                 ("Risk Parity", ("4,2", "9,6"), "36,64"),
                                 ("Equal Weight", (None, None), "26,67")]:
        if lo:
            p.zahl(f"Spannweite {KURZ[name]}, kleinstes", lo, mittel[name].min(), "%")
            p.zahl(f"Spannweite {KURZ[name]}, größtes", hi, mittel[name].max(), "%")
        p.zahl(f"Summe der größten Vier, {KURZ[name]}", vier, mittel[name].nlargest(4).sum(), "%")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("B Tabelle B2: weitere Kennzahlen")
    b2 = {"Random Forest": ("1,350", "−1,81", "54,91", "623,3"),
          "Markowitz MVO": ("1,296", "−1,96", "55,75", "685,1"),
          "Equal Weight":  ("1,283", "−1,59", "55,27", "509,7"),
          "Risk Parity":   ("1,121", "−1,46", "55,31", "370,2")}
    for name, (so, var, hit, ges) in b2.items():
        x = s[name]
        p.zahl(f"{KURZ[name]}: Sortino", so, sortino_ratio(x, rf))
        p.zahl(f"{KURZ[name]}: Value at Risk 95 %", var, np.percentile(x, 5) * 100, "%")
        p.zahl(f"{KURZ[name]}: Trefferquote", hit, (x > 0).mean() * 100, "%")
        p.zahl(f"{KURZ[name]}: Gesamtrendite", ges, ((1 + x).prod() - 1) * 100, "%")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("B Tabelle B3: Standardfehler und Konfidenzintervalle")
    b3 = {"Random Forest": ("0,0589", "0,3181", "0,9357", "0,31", "1,56"),
          "Markowitz MVO": ("0,0571", "0,3181", "0,9063", "0,28", "1,53"),
          "Equal Weight":  ("0,0567", "0,3181", "0,9005", "0,28", "1,52"),
          "Risk Parity":   ("0,0498", "0,3180", "0,7901", "0,17", "1,41")}
    for name, (st, sej, sj, lo, hi) in b3.items():
        S_tag = sharpe_exakt(name, je_tag=True)                # volle Genauigkeit aus dem Lauf
        se_tag = np.sqrt((1 + 0.5 * S_tag ** 2) / T)          # Formel 27
        S_jahr, se_jahr = S_tag * np.sqrt(HANDELSTAGE), se_tag * np.sqrt(HANDELSTAGE)
        p.zahl(f"{KURZ[name]}: Ŝ Tag", st, S_tag)
        p.zahl(f"{KURZ[name]}: SE Tag", "0,0200", se_tag)
        p.zahl(f"{KURZ[name]}: SE Jahr", sej, se_jahr)
        p.zahl(f"{KURZ[name]}: Ŝ Jahr", sj, S_jahr)
        p.zahl(f"{KURZ[name]}: Intervall unten", lo, S_jahr - 1.96 * se_jahr)
        p.zahl(f"{KURZ[name]}: Intervall oben", hi, S_jahr + 1.96 * se_jahr)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("B Tabelle B4: Schiefe und Wölbung")
    p.zahl("Zahl der Tagesrenditen", "2495", T)
    b4 = {"Random Forest": ("−0,253", "9,68", "1,0112"), "Markowitz MVO": ("−0,290", "13,52", "1,0133"),
          "Equal Weight": ("−0,283", "18,76", "1,0151"), "Risk Parity": ("−0,274", "19,13", "1,0124")}
    for name, (g3t, g4t, nt) in b4.items():
        v = sig["deflated_sharpe"][name]                       # Werte aus dem Lauf
        g3, g4, S = v["skew"], v["kurtosis"], v["sr_per_period"]
        p.zahl(f"{KURZ[name]}: Schiefe", g3t, g3)
        p.zahl(f"{KURZ[name]}: Wölbung", g4t, g4)
        p.zahl(f"{KURZ[name]}: Nenner der DSR", nt, np.sqrt(1 - g3 * S + (g4 - 1) / 4 * S ** 2))

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("C C1: Parameter des Backtests")
    p.zahl("Anlageuniversum", "15", meta["n_assets"])
    p.aussage("Referenzreihe", "SPY", CFG.spy_ticker == "SPY", CFG.spy_ticker)
    p.aussage("Datenbestand", "01.01.2013 – 31.12.2024",
              (CFG.start_date, CFG.end_date) == ("2013-01-01", "2024-12-31"), f"{CFG.start_date} bis {CFG.end_date}",
              woertlich=True)
    p.aussage("Beginn des Backtests", "01.01.2015", meta["backtest_start"] == "2015-01-01", meta["backtest_start"],
              woertlich=True)
    p.zahl("Rebalancing-Termine", "119", len(G["Risk Parity"]))
    p.zahl("Schätzfenster", "3", meta["train_years"], "Jahre")
    quelltext = inspect.getsource(run_backtest)
    p.aussage("Mindesthistorie 252 Tage / 24 Monate / 12 Zeilen", "252 / 24 / 12",
              all(x in quelltext for x in ("< 252", "< 24", "< 12")), "steht so im Code")
    p.zahl("risikofreier Zins", "4", rf * 100, "%")
    p.zahl("Positionsobergrenze", "20", meta["max_weight"] * 100, "%")
    p.zahl("Transaktionskosten", "10", meta["transaction_cost"] * 10_000, "bp")
    p.zahl("Umschlagschranke", "30", meta["rf_turnover_limit"] * 100, "%")
    p.aussage("Rückfall auf Minimum-Varianz", "ja", meta["min_variance_fallback"] is True,
              str(meta["min_variance_fallback"]))
    p.aussage("Kreuzvalidierung purged", "ja", meta["use_purged_cv"] is True, str(meta["use_purged_cv"]))
    p.zahl("Blöcke der Kreuzvalidierung", "5", meta["rf_cv_splits"])
    p.zahl("Sperrzone", "2", meta["cv_embargo"] * 100, "%")
    p.zahl("Kandidaten der Hyperparametersuche", "30", meta["rf_n_iter"])
    gitter = optimizers.RFPortfolioOptimizer()._param_grid()
    bereich = lambda name: gitter[f"rf__{name}"].support()
    # max_samples (Stichprobenanteil je Baum) wird ebenfalls gesucht; C1 nennt
    # ihn erst, wenn der Vorschlag vom 14.10.2026 übernommen ist (bis dahin
    # meldet --text diese Zeile als "nicht im Text").
    for name, text in [("n_estimators", "100–499"), ("max_depth", "3–14"), ("min_samples_leaf", "3–19"),
                       ("max_features", "0,3–0,9"), ("max_samples", "0,6–0,95")]:
        lo, hi = bereich(name)
        ist = f"{lo:g}–{hi:g}".replace(".", ",")
        p.aussage(f"Suchraum {name}", text, ist == text, ist, woertlich=True)
    p.aussage("Parallelverarbeitung aus (n_jobs = 1)", "aus", CFG.deterministic is True, f"deterministic={CFG.deterministic}")
    k = kurse()
    p.zahl("Kursdatei: Handelstage", "3019", len(k))
    p.zahl("Kursdatei: Reihen", "16", k.shape[1])
    beliebig = next(iter(sig["sharpe_difference_tests"].values()))
    p.zahl("Bootstrap-Ziehungen", "4999", beliebig["n_boot"])
    p.zahl("T", "2495", beliebig["T"])
    p.zahl("Blocklänge", "14", beliebig["block_size"])
    p.zahl("HAC-Bandbreite", "8", significance._nw_bandwidth(T), "Tage")
    startwerte = [inspect.signature(significance.sharpe_difference_test).parameters["seed"].default,
                  theory_plots.THEORY_SEED]
    p.aussage("Startwert aller Zufallsverfahren", "42",
              all(x == 42 for x in startwerte) and "random_state=42" in inspect.getsource(optimizers),
              "Bootstrap, Titelauswahl A1 und Random Forest: 42")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("C C2: Quellcode")
    try:
        for commit, datum in [(C2_COMMIT, C2_DATUM), ("b8eeea9", "15.08.2026")]:
            ist = git("show", "-s", "--format=%ad", "--date=format:%d.%m.%Y", commit).strip()
            p.merke(f"Commit {commit}", commit)
            p.aussage(f"Commit {commit} vom", datum, ist == datum, ist, woertlich=True)
        # Steht in C2 der neueste Stand? Wenn nicht: kein Fehler, aber ein Hinweis.
        neuester, neu_datum = git("log", "-1", "--format=%h %ad", "--date=format:%d.%m.%Y").split()
        if not neuester.startswith(C2_COMMIT):
            p.offen("neuester Commit im Repository", f"{neuester} ({neu_datum})",
                    f"C2 nennt {C2_COMMIT} — vor der Abgabe C2 und das Literaturverzeichnis "
                    "auf den neuesten Commit setzen und C2_COMMIT oben in diesem Skript nachziehen")
        veraendert = []
        for modul in RECHENKERN:
            alt = git("show", f"b8eeea9:portfolio/{modul}.py")
            neu = (ROOT / "portfolio" / f"{modul}.py").read_text(encoding="utf-8")
            if ohne_kommentare(alt) != ohne_kommentare(neu):
                veraendert.append(modul)
        p.aussage("Rechenkern seit b8eeea9 unverändert", "unverändert", not veraendert,
                  "nur Kommentare und Dokumentation geändert" if not veraendert
                  else "Programmlogik geändert in: " + ", ".join(veraendert))
    except (subprocess.CalledProcessError, FileNotFoundError):
        p.offen("Commits und Rechenkern", f"{C2_COMMIT} / b8eeea9", "git nicht verfügbar (kein Klon des Repositorys)")
    p.aussage("Python", "3.14", sys.version.startswith("3.14"), sys.version.split()[0], woertlich=True)
    for paket, text in [("numpy", "2.4.4"), ("pandas", "3.0.2"), ("scipy", "1.17.1"),
                        ("scikit-learn", "1.8.0"), ("yfinance", "1.3.0")]:
        p.aussage(paket, text, version(paket) == text, version(paket), woertlich=True)
    datum = pd.Timestamp(meta["timestamp"]).strftime("%d.%m.%Y")
    p.aussage("Lauf vom", "25.08.2026", datum == "25.08.2026", datum, woertlich=True)
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
