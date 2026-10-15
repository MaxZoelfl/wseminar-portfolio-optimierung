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

Zwei Vereinfachungen des Backtests werden hier zusätzlich beziffert (Kommentare
in portfolio/backtest.py, LIMITATIONS.md Abschnitt 13): An wie vielen Terminen
der Lernstoff des Random Forest einen Monat früher endet (§ 4.3), und wie viel
Sharpe-Quotient die festen Gewichte innerhalb des Monats ausmachen (§ 4.4).
Für die Fassung mit Kursdrift laufen zur Gegenprobe auch alle sechs
Signifikanztests noch einmal.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_4_methodik.py          # rund 15 Sekunden
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


def renditen_nachgebaut(drift: bool) -> pd.DataFrame:
    """Tagesrenditen der vier Strategien, nachgebaut aus Kursen, Gewichten und Umschlag.

    drift=False  feste Zielgewichte an jedem Tag des Monats — so rechnet der Backtest
    drift=True   die Gewichte driften im Monat mit den Kursen (gekauft und gehalten),
                 so wie es die Umschlagsmessung (Formel 26) unterstellt
    Die Kosten (Umschlag × Kostensatz) gehen in beiden Fällen am ersten Tag ab."""
    r = titelrenditen(nur_auswertungszeitraum=False)
    G, U = gewichte(), umschlag()
    kostensatz = protokoll()["experiment_meta"]["transaction_cost"]
    termine = list(G["Risk Parity"].index)
    ende = strategierenditen().index[-1]
    spalten = {}
    for s in STRATEGIEN:
        teile = []
        for k, t in enumerate(termine):
            bis = termine[k + 1] if k + 1 < len(termine) else ende
            halte = r[(r.index > t) & (r.index <= bis)]
            w = G[s].loc[t].values
            if drift:
                wert = (w * (1 + halte).cumprod()).sum(axis=1)       # Depotwert, Start bei 1
                tag = wert / wert.shift(1, fill_value=1.0) - 1
            else:
                tag = (halte * w).sum(axis=1)
            tag.iloc[0] -= U.loc[t, UMSCHLAG_SPALTE[s]] * kostensatz
            teile.append(tag)
        spalten[s] = pd.concat(teile)
    return pd.DataFrame(spalten)


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
    lernstoff = {t: trainingsfenster(md, t)[0] for t in termine}
    zeilen = pd.Series({t: len(x) / len(TICKERS) for t, x in lernstoff.items()})
    voll = zeilen[zeilen.index >= erster_voller]
    p.zahl("Monatszeilen je Titel (volles Fenster)", "36", voll.median(), ungefaehr=True,
           notiz=f"an {int((voll == 36).sum())} Terminen 36, an {int((voll == 35).sum())} Terminen 35 "
                 "(Kalenderdetail, siehe nächste Zeile)")
    # Kalenderdetail (Kommentar in backtest.py): "Monatsende minus ein Monat"
    # trifft bei Monaten mit höchstens 30 Tagen nicht das Ende des Vormonats;
    # dessen Zeile fällt mit heraus, der Lernstoff endet einen Monat früher.
    frueher = sum(x.index.max() < t - pd.offsets.MonthEnd(1) for t, x in lernstoff.items())
    p.zahl("Termine, an denen der Lernstoff einen Monat früher endet", "50", frueher)
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
    # Vereinfachung (Kommentar 6b in backtest.py): feste Gewichte im Monat.
    # Gegenrechnung mit Kursdrift; beide Fassungen aus denselben (auf vier
    # Stellen gerundeten) Gewichten nachgebaut, damit sich Rundungen aufheben.
    fest, drift = renditen_nachgebaut(drift=False), renditen_nachgebaut(drift=True)
    rueckgang = pd.Series({s: sharpe(fest[s], rf) - sharpe(drift[s], rf) for s in STRATEGIEN})
    p.zahl("Sharpe-Rückgang mit Kursdrift im Monat, kleinster", "0,005", rueckgang.min(),
           notiz=", ".join(f"{KURZ[s]} {de(v, 4)}" for s, v in rueckgang.items()))
    p.zahl("Sharpe-Rückgang mit Kursdrift im Monat, größter", "0,007", rueckgang.max())
    rang = lambda df: list(pd.Series({s: sharpe(df[s], rf) for s in STRATEGIEN}).sort_values().index)
    p.aussage("Rangfolge mit Kursdrift unverändert", "unverändert", rang(fest) == rang(drift),
              " < ".join(KURZ[s] for s in rang(drift)))
    from portfolio.significance import sharpe_difference_test, holm_bonferroni
    lang = {v: k for k, v in KURZ.items()}
    paare = [("RF", "MVO"), ("RF", "EW"), ("RF", "RP"), ("MVO", "EW"), ("MVO", "RP"), ("RP", "EW")]
    pw = [sharpe_difference_test(drift[lang[x]], drift[lang[y]], rf=rf)["p_value"] for x, y in paare]
    holm = holm_bonferroni(pw)
    signifikant = [f"{x}–{y}" for (x, y), r in zip(paare, holm["reject"]) if r]
    p.aussage("Testentscheidungen mit Kursdrift unverändert", "unverändert",
              signifikant == ["RP–EW"], f"signifikant: {', '.join(signifikant) or 'keins'} "
              f"(Holm {de(holm['p_adjusted'][-1], 4)})")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("4.5 Performance-Kennzahlen")
    p.zahl("risikofreier Zins", "4", rf * 100, "%")
    p.zahl("Handelstage für die Umlage", "252", HANDELSTAGE)
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
