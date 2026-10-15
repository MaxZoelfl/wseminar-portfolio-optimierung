"""
Kapitel 4 — Methodik des empirischen Vergleichs: jede Zahl des Kapitels, in
Textreihenfolge.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Kapitel 4 beschreibt, WIE der Vergleich aufgebaut ist: welche Daten, welche
Regeln, welche Schutzvorkehrungen. Die Zahlen darin sind deshalb meist
Einstellungen des Backtests (etwa 20 % Obergrenze, zehn Basispunkte Kosten)
oder einfache Zählungen (etwa 119 Termine). Dieses Skript liest die
Einstellungen aus dem Laborprotokoll des Laufs (output/experiment_log.json)
und aus dem Programmcode und zählt nach, statt sie abzuschreiben.

Eine Zahl stammt aus einem zweiten Lauf: der Sharpe-Quotient 0,961 "ohne
Purging" in § 4.3. Dessen Tagesrenditen liegen als Referenzdatei in
skripte/daten/vergleichslauf_ohne_purging.csv (siehe skripte/README.md).

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_4_methodik.py          # rund 10 Sekunden
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from portfolio.backtest import run_backtest           # noqa: E402  (nur der Quelltext wird gelesen)
from portfolio.config import START_DATE, END_DATE     # noqa: E402
from portfolio.indicators import FEATURE_COLS, compute_momentum   # noqa: E402
from kapitel_3_random_forest import monatsdaten_aufbauen, trainingsfenster   # noqa: E402

# Sektorzuordnung nach GICS, wie in Tabelle A1 und in portfolio/config.py kommentiert.
SEKTOR = {"AAPL": "Informationstechnologie", "MSFT": "Informationstechnologie",
          "NVDA": "Informationstechnologie", "JNJ": "Gesundheitswesen", "UNH": "Gesundheitswesen",
          "JPM": "Finanzen", "GS": "Finanzen", "PG": "Basiskonsumgüter", "KO": "Basiskonsumgüter",
          "CAT": "Industrie", "HON": "Industrie", "XOM": "Energie", "VZ": "Kommunikationsdienste",
          "PLD": "Immobilien", "LIN": "Grundstoffe"}
GICS_SEKTOREN = 11                                    # so viele Sektoren kennt der GICS-Standard


def pruefe() -> Pruefung:
    """Alle Zahlen aus Kapitel 4, in der Reihenfolge des Textes."""
    p = Pruefung("Kapitel 4 — Methodik des empirischen Vergleichs")
    meta = protokoll()["experiment_meta"]
    quelltext = inspect.getsource(run_backtest)       # die Regeln, wie sie im Code stehen
    termine = gewichte()["Risk Parity"].index
    umsch = umschlag()
    rf = meta["risk_free_rate"]

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("4.1 Datengrundlage und Auswahl der Titel")
    p.zahl("US-amerikanische Standardwerte", "15", meta["n_assets"])
    p.offen("Abrufdatum der Kurse", "15.08.2026", "steht im Laufprotokoll des Abrufs und in Anhang C1, "
            "nicht in den Ergebnisdateien")
    p.aussage("Beginn des Datenzeitraums", "01.01.2013", START_DATE == "2013-01-01", START_DATE, woertlich=True)
    p.aussage("Ende des Datenzeitraums", "31.12.2024", END_DATE == "2024-12-31", END_DATE, woertlich=True)
    p.aussage("investiert wird ab", "01.01.2015", meta["backtest_start"] == "2015-01-01",
              meta["backtest_start"], woertlich=True)
    laengstes = max(inspect.signature(compute_momentum).parameters["periods"].default)
    p.zahl("Zwölf-Monats-Momentum: Monate (252 Handelstage ÷ 21)", "Zwölf", laengstes / 21)
    n = meta["n_assets"]
    p.zahl("zu schätzende Einträge der Kovarianzmatrix bei 15 Titeln", "120", n * (n + 1) // 2)
    sektoren = {SEKTOR[t] for t in meta["tickers"]}
    p.zahl("abgedeckte GICS-Sektoren", "neun", len(sektoren))
    p.zahl("GICS-Sektoren insgesamt", "elf", GICS_SEKTOREN)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("4.2 Die vier verglichenen Strategien")
    p.zahl("Obergrenze je Titel", "20", meta["max_weight"] * 100, "%")
    p.zahl("Umschlagschranke Random Forest", "30", meta["rf_turnover_limit"] * 100, "%")
    p.zahl("Umschlagschranke Markowitz (dieselbe)", "30", meta["mvo_turnover_limit"] * 100, "%")
    mu_zeile = [z for z in quelltext.splitlines() if "mu_hist =" in z][0]
    p.aussage("Hochrechnung des historischen Mittels auf ein Jahr", "252",
              "* 252" in mu_zeile, "steht so im Code", notiz=mu_zeile.strip(), woertlich=True)
    p.aussage("Anteil je Titel bei der Gleichgewichtung", "ein Fünfzehntel",
              abs(gewichte()["Equal Weight"].iloc[0, 0] - 1 / 15) < 1e-12,
              "1/15", notiz="1/15 = 6,67 %", woertlich=True)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("4.3 Walk-Forward-Backtesting und die Vermeidung von Look-Ahead-Bias")
    p.zahl("Trainingsfenster: vorangehende Jahre", "drei", meta["train_years"])
    p.aussage("Zeitraum des Backtests", "2015 bis 2024",
              (termine[0].year, umsch.index[-1].year) == (2015, 2024),
              f"{termine[0]:%m/%Y} bis {strategierenditen().index[-1]:%m/%Y}", woertlich=True)
    p.zahl("Rebalancing-Termine", "119", len(termine))
    erster_tag = titelrenditen(nur_auswertungszeitraum=False).index[0]
    p.zahl("Jahr, in dem die Kursdaten beginnen", "2013", erster_tag.year)
    # Wie viele Termine haben noch kein volles Dreijahresfenster? Voll heißt: Die
    # Kursdaten reichen bis auf höchstens eine Woche an den Fensteranfang heran.
    fensteranfang = [t - pd.DateOffset(years=meta["train_years"]) for t in termine]
    unvoll = [t for t, a in zip(termine, fensteranfang) if erster_tag > a + pd.Timedelta(days=7)]
    p.zahl("erste Termine ohne volles Dreijahresfenster", "elf", len(unvoll))
    laenge = (termine[0] - erster_tag).days / 365.25
    p.aussage("Fenster am ersten Termin", "gut zwei Jahren", 2.0 < laenge < 2.25, f"{de(laenge, 2)} Jahre",
              notiz=f"{de(laenge, 2)} Jahre", woertlich=True)
    erster_voller = next(t for t, a in zip(termine, fensteranfang) if erster_tag <= a + pd.Timedelta(days=7))
    p.aussage("Fenster vollständig ab", "Ende 2015", erster_voller == pd.Timestamp("2015-12-31"),
              f"{erster_voller.date()}", notiz=f"erster voller Termin {erster_voller:%d.%m.%Y}", woertlich=True)
    for zahl, stelle, was in [("252", "len(train_daily) < 252", "Untergrenze: Handelstage im Fenster"),
                              ("24", "< 24", "Untergrenze: Monate Merkmalshistorie")]:
        p.aussage(was, zahl, stelle in quelltext, "steht so im Code", notiz=f"Code: {stelle}", woertlich=True)
    # Lernstoff des Random Forest: dieselbe Auswahl wie im Backtest (aus kapitel_3)
    md, _ = monatsdaten_aufbauen()
    lernstoff = {t: trainingsfenster(md, t)[0] for t in termine}
    # Kalenderdetail (Kommentar in backtest.py): "Monatsende minus ein Monat"
    # trifft bei Monaten mit höchstens 30 Tagen nicht das Ende des Vormonats;
    # dessen Zeile fällt mit heraus, der Lernstoff endet einen Monat früher.
    frueher = sum(x.index.max() < t - pd.offsets.MonthEnd(1) for t, x in lernstoff.items())
    p.zahl("Termine, an denen der Lernstoff einen Monat früher endet", "50", frueher)
    p.zahl("von Terminen", "119", len(termine))
    p.aussage("Untergrenze: vollständige Monatszeilen je Titel", "zwölf", "len(valid) < 12" in quelltext,
              "steht so im Code", notiz="Code: len(valid) < 12", woertlich=True)
    zeilen = pd.Series({t: len(x) / len(TICKERS) for t, x in lernstoff.items()})
    voll = zeilen[zeilen.index >= erster_voller]
    p.zahl("Monatszeilen je Titel (volles Fenster)", "36", voll.median(), ungefaehr=True,
           notiz=f"Text: „etwa“; an {int((voll == 36).sum())} Terminen 36, an {int((voll == 35).sum())} "
                 "Terminen 35 (Kalenderdetail)")
    p.zahl("Merkmale", "zwanzig", len(FEATURE_COLS))
    p.zahl("Sperrzone (Embargo) in Prozent der Perioden", "zwei", meta["cv_embargo"] * 100, "%")
    vergleich = pd.read_csv(DATEN / "vergleichslauf_ohne_purging.csv", index_col=0,
                            parse_dates=True, comment="#")
    heute = strategierenditen()
    ohne, mit = sharpe_ratio(vergleich["Random Forest"], rf), sharpe_exakt("Random Forest")
    p.zahl("Sharpe Random Forest ohne Purging (Vergleichslauf)", "0,961", ohne,
           notiz="Referenzdatei skripte/daten/vergleichslauf_ohne_purging.csv")
    p.zahl("Sharpe Random Forest mit Purging (maßgeblicher Lauf)", "0,936", mit)
    p.zahl("so viel beruhte auf Informationsleckage", "0,025", ohne - mit, ungefaehr=True,
           toleranz=0.0005, notiz=f"Text: „rund 0,025“; genau {de(ohne - mit, 4)}")
    gleich = all(de(sharpe_ratio(vergleich[s], rf), 3) == de(sharpe_ratio(heute[s], rf), 3)
                 for s in ["Markowitz MVO", "Equal Weight", "Risk Parity"])
    p.aussage("die übrigen drei Strategien bleiben unverändert", "unverändert", gleich,
              "auf drei Nachkommastellen gleich" if gleich else "weichen ab",
              notiz="auf drei Nachkommastellen gleich" if gleich else "", woertlich=True)

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("4.4 Transaktionskosten und Umschlag")
    erster = umsch.iloc[0]
    p.zahl("Umschlag am ersten Termin (Aufbau des Depots)", "100", erster.min() * 100, "%",
           notiz="bei allen vier Strategien" if (erster == 1.0).all() else "nicht bei allen!")
    p.zahl("Kostensatz", "zehn", meta["transaction_cost"] * 10_000, "Basispunkte")
    regulaer = umsch.iloc[1:]                          # ohne den Depotaufbau
    p.zahl("Umschlagschranke für Markowitz und Random Forest", "30", meta["rf_turnover_limit"] * 100, "%")
    bindet = lambda spalte: int((regulaer[spalte] >= meta["rf_turnover_limit"] - 5e-4).sum())
    p.zahl("Monate, in denen die Schranke beim Random Forest bindet", "101", bindet("turnover_rf"))
    p.zahl("von regulären Terminen (ohne den Depotaufbau)", "118", len(regulaer))
    p.zahl("so oft bindet sie bei Markowitz", "ein", bindet("turnover_mvo"),
           notiz="Text: „nur ein einziges Mal“")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("4.5 Performance-Kennzahlen")
    p.zahl("risikofreier Zins pro Jahr", "vier", rf * 100, "%")
    freq = inspect.signature(sharpe_ratio).parameters["freq"].default
    p.zahl("Handelstage, auf die er umgelegt wird", "252", freq)
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
