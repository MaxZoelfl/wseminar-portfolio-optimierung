"""
Kapitel 4 — Methodik des empirischen Vergleichs: jede eigene Zahl nachgerechnet.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Kapitel 4 beschreibt, WIE der Vergleich aufgebaut ist: welche Daten, welche
Regeln, welche Schutzvorkehrungen. Die Zahlen darin sind deshalb meist
Einstellungen des Backtests (etwa 20 % Obergrenze, 10 Basispunkte Kosten) oder
einfache Zählungen (etwa 119 Termine). Dieses Skript liest die Einstellungen
aus dem Laborprotokoll des Laufs (output/experiment_log.json) und aus dem
Programmcode und zählt nach, statt sie abzuschreiben.

Eine Zahl stammt aus einem zweiten Lauf: der Sharpe-Quotient 0,980 "ohne
Purging" in § 4.3. Dessen Tagesrenditen liegen als Referenzdatei in
skripte/daten/vergleichslauf_ohne_purging.csv (siehe skripte/README.md).

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_4_methodik.py          # wenige Sekunden
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from portfolio.backtest import run_backtest           # noqa: E402  (nur der Quelltext wird gelesen)

# Sektorzuordnung nach GICS, wie in Tabelle A1 und in portfolio/config.py kommentiert.
SEKTOR = {"AAPL": "Informationstechnologie", "MSFT": "Informationstechnologie",
          "NVDA": "Informationstechnologie", "JNJ": "Gesundheitswesen", "UNH": "Gesundheitswesen",
          "JPM": "Finanzen", "GS": "Finanzen", "PG": "Basiskonsumgüter", "KO": "Basiskonsumgüter",
          "CAT": "Industrie", "HON": "Industrie", "XOM": "Energie", "VZ": "Kommunikationsdienste",
          "PLD": "Immobilien", "LIN": "Grundstoffe"}
GICS_SEKTOREN = 11


def pruefe() -> Pruefung:
    p = Pruefung("Kapitel 4 — Methodik des empirischen Vergleichs")
    meta = protokoll()["experiment_meta"]
    quelltext = inspect.getsource(run_backtest)       # die Regeln, wie sie im Code stehen
    termine = gewichte()["Risk Parity"].index
    umsch = umschlag()

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("4.1 Datengrundlage und Auswahl der Titel")
    p.zahl("Zahl der Titel", "15", meta["n_assets"])
    p.offen("Abrufdatum der Kurse", "15.08.2026", "dokumentiert in Anhang C1 und im Laborprotokoll")
    from portfolio.config import START_DATE, END_DATE
    p.aussage("Datenzeitraum", "01.01.2013–31.12.2024",
              (START_DATE, END_DATE) == ("2013-01-01", "2024-12-31"), f"{START_DATE} bis {END_DATE}")
    p.aussage("Beginn der Investition", "01.01.2015", meta["backtest_start"] == "2015-01-01",
              meta["backtest_start"])
    n = meta["n_assets"]
    p.zahl("zu schätzende Einträge der Kovarianzmatrix", "120", n * (n + 1) // 2)
    sektoren = {SEKTOR[t] for t in meta["tickers"]}
    p.zahl("abgedeckte GICS-Sektoren", "9", len(sektoren))
    p.zahl("GICS-Sektoren insgesamt", "11", GICS_SEKTOREN)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("4.2 Die vier verglichenen Strategien")
    p.zahl("Obergrenze je Titel", "20", meta["max_weight"] * 100, "%")
    p.zahl("Umschlagschranke Random Forest", "30", meta["rf_turnover_limit"] * 100, "%")
    p.zahl("Umschlagschranke Markowitz", "30", meta["mvo_turnover_limit"] * 100, "%")
    p.zahl("Handelstage je Jahr (Hochrechnung)", "252", HANDELSTAGE)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("4.3 Walk-Forward-Backtesting und die Vermeidung von Look-Ahead-Bias")
    p.zahl("Rebalancing-Termine", "119", len(termine))
    p.zahl("Länge des Trainingsfensters", "3", meta["train_years"], "Jahre")
    # Wie viele Termine haben noch kein volles Dreijahresfenster? Voll heißt: Die
    # Kursdaten reichen bis auf höchstens eine Woche an den Fensteranfang heran.
    erster_tag = titelrenditen(nur_auswertungszeitraum=False).index[0]
    fensteranfang = [t - pd.DateOffset(years=meta["train_years"]) for t in termine]
    unvoll = [t for t, a in zip(termine, fensteranfang) if erster_tag > a + pd.Timedelta(days=7)]
    p.zahl("Termine ohne volles Fenster", "11", len(unvoll))
    laenge = (termine[0] - erster_tag).days / 365.25
    p.aussage("erstes Fenster", "gut zwei Jahre", 2.0 < laenge < 2.25, f"{de(laenge, 2)} Jahre")
    erster_voller = next(t for t, a in zip(termine, fensteranfang) if erster_tag <= a + pd.Timedelta(days=7))
    p.aussage("Fenster vollständig ab", "Ende 2015", erster_voller == pd.Timestamp("2015-12-31"),
              f"{erster_voller.date()}")
    for zahl, stelle in [("252", "len(train_daily) < 252"), ("24", "< 24"), ("12", "len(valid) < 12")]:
        p.aussage(f"Mindestgrenze im Code ({stelle})", zahl, stelle in quelltext, "steht so im Code")
    # Monatszeilen je Titel im vollen Fenster: drei Jahre bis zum vorletzten Monat.
    from portfolio.indicators import FEATURE_COLS
    p.zahl("Merkmale", "20", len(FEATURE_COLS))
    # Monatszeilen je Titel: dieselbe Merkmalskette wie im Backtest (aus kapitel_3).
    from kapitel_3_random_forest import monatsdaten_aufbauen, trainingsfenster
    md, _ = monatsdaten_aufbauen()
    X_voll = trainingsfenster(md, termine[-1])[0]          # letzter Termin: volles Fenster
    p.zahl("Monatszeilen je Titel (volles Fenster)", "36", len(X_voll) / len(TICKERS), ungefaehr=True,
           notiz="drei Jahre = 36 Monate, abzüglich des ausgelassenen letzten Monats")
    p.zahl("Sperrzone (Embargo)", "2", meta["cv_embargo"] * 100, "%")
    vergleich = pd.read_csv(DATEN / "vergleichslauf_ohne_purging.csv", index_col=0,
                            parse_dates=True, comment="#")
    heute = strategierenditen()
    rf = meta["risk_free_rate"]
    ohne, mit = sharpe(vergleich["Random Forest"], rf), sharpe(heute["Random Forest"], rf)
    p.zahl("Sharpe Random Forest ohne Purging", "0,980", ohne, notiz="Referenzdatei des Vergleichslaufs")
    p.zahl("Sharpe Random Forest mit Purging", "0,936", mit)
    p.zahl("Informationsleckage", "0,04", ohne - mit, ungefaehr=True)
    gleich = all(de(sharpe(vergleich[s], rf), 3) == de(sharpe(heute[s], rf), 3)
                 for s in ["Markowitz MVO", "Equal Weight", "Risk Parity"])
    p.aussage("übrige drei Strategien unverändert", "unverändert", gleich,
              "auf drei Nachkommastellen gleich" if gleich else "weichen ab")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("4.4 Transaktionskosten und Umschlag")
    erster = umsch.iloc[0]
    p.aussage("Umschlag am ersten Termin", "100 % (alle Strategien)", bool((erster == 1.0).all()),
              ", ".join(f"{v * 100:.0f} %" for v in erster))
    p.zahl("Kostensatz", "10", meta["transaction_cost"] * 10_000, "bp")
    regulaer = umsch.iloc[1:]                          # ohne den Depotaufbau
    p.zahl("reguläre Termine", "118", len(regulaer))
    bindet = lambda spalte: int((regulaer[spalte] >= meta["rf_turnover_limit"] - 5e-4).sum())
    p.zahl("Schranke bindet beim Random Forest", "101", bindet("turnover_rf"), "Monate")
    p.zahl("Schranke bindet bei Markowitz", "1", bindet("turnover_mvo"), "Monat")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("4.5 Performance-Kennzahlen")
    p.zahl("risikofreier Zins", "4", rf * 100, "%")
    p.zahl("Handelstage für die Umlage", "252", HANDELSTAGE)
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
