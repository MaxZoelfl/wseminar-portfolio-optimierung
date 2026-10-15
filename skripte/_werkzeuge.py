"""
Gemeinsame Werkzeuge für alle Nachrechnungs-Skripte in diesem Ordner.

FÜR EINSTEIGER — WAS STECKT HIER DRIN?
Jedes Kapitel-Skript (kapitel_2_theorie.py, kapitel_3_random_forest.py, …)
braucht dieselben Grundbausteine: die eingefrorenen Kurse laden, die
Ergebnisdateien des großen Backtests einlesen und am Ende jede berechnete
Zahl mit der Zahl vergleichen, die im Text der Arbeit steht. Damit das nicht
siebenmal geschrieben werden muss, liegt es einmal hier.

Der Unterstrich am Anfang des Dateinamens bedeutet: Diese Datei startet man
nicht selbst, sie wird von den anderen Skripten benutzt.

WOHER DIE DATEN KOMMEN (nichts davon erfordert einen neuen Backtest)
  data/prices.pkl              eingefrorene Schlusskurse, Abruf 15.08.2026
  output/daily_returns.csv     Tagesrenditen der vier Strategien
  output/weights_*.csv         Monatsgewichte von Markowitz, Random Forest, Risk Parity
  output/turnover.csv          Umschlag je Strategie und Monat
  output/experiment_log.json   Parameter, Kennzahlen und Signifikanztests des Laufs
  skripte/daten/               zwei kleine Referenzdateien (siehe skripte/README.md)

WIE DER VERGLEICH MIT DEM TEXT FUNKTIONIERT
Jede Prüfung bekommt die Zahl so, wie sie im Text steht — als Zeichenkette,
etwa "17,56". Aus ihr liest das Werkzeug ab, auf wie viele Nachkommastellen
der Text rundet, rundet die eigene Rechnung genauso (kaufmännisch, also
0,5 wird aufgerundet) und vergleicht. Ergebnis je Zahl:

    ✅  stimmt         gerechnet und gerundet genau wie im Text
    ≈   stimmt ungefähr  nur bei Angaben wie "rund", "etwa", "gut": die
                         Rechnung liegt nah genug am Wortlaut
    ❌  weicht ab      der Text sollte geprüft werden
    ⏭  nicht gerechnet  Zahl stammt aus einer Quelle oder braucht eine
                         lange Rechnung, die nur auf Wunsch läuft

WICHTIG: Der Wortlaut ist in den Skripten EINGETRAGEN — sie lesen die
Word-Datei nicht selbst. Ändert sich der Text, muss der Eintrag nachgezogen
werden. Damit das nicht übersehen wird, prüft
    python skripte/alle_kapitel.py --text DATEI      (.docx oder .md)
zusätzlich, ob jeder eingetragene Wortlaut in DATEI überhaupt vorkommt
(Funktion wortlaut_fehlt unten).
"""

import json
import os
import re
import subprocess
import sys
import warnings
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────
# Pfade. Der Projektstamm liegt eine Ebene über diesem Ordner. Alle Skripte
# wechseln dorthin, weil das Paket "portfolio" seine Kursdatei relativ zum
# Arbeitsverzeichnis sucht — so funktioniert jeder Aufruf, egal von wo.
# ────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
DATEN = Path(__file__).resolve().parent / "daten"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import logging                                    # noqa: E402  (nach sys.path)
logging.getLogger("portfolio_v4").setLevel(logging.WARNING)

from portfolio.config import TICKERS, SPY_TICKER  # noqa: E402

HANDELSTAGE = 252
STRATEGIEN = ["Random Forest", "Markowitz MVO", "Equal Weight", "Risk Parity"]
KURZ = {"Random Forest": "RF", "Markowitz MVO": "MVO",
        "Equal Weight": "EW", "Risk Parity": "RP"}


# ────────────────────────────────────────────────────────────────────────
# Daten laden
# ────────────────────────────────────────────────────────────────────────
def kurse() -> pd.DataFrame:
    """Schlusskurse der 15 Titel und des SPY, nur Tage mit allen 16 Kursen
    (genau wie portfolio/data.py)."""
    roh = pd.read_pickle(ROOT / "data" / "prices.pkl")
    return roh[TICKERS + [SPY_TICKER]].dropna(how="any")


def titelrenditen(nur_auswertungszeitraum: bool = True) -> pd.DataFrame:
    """Einfache Tagesrenditen der 15 Titel (Formel 1 der Arbeit).

    nur_auswertungszeitraum=True schneidet auf die 2495 Handelstage zu, über die
    auch die Strategien bewertet werden (02.02.2015 bis 30.12.2024). Das ist der
    Zeitraum, den die Arbeit mit "Tagesdaten 2015–2024" meint.
    """
    r = kurse().pct_change(fill_method=None).dropna()[TICKERS]
    if nur_auswertungszeitraum:
        s = strategierenditen()
        r = r.loc[s.index[0]:s.index[-1]]
    return r


def strategierenditen() -> pd.DataFrame:
    """Tagesrenditen der vier Strategien aus dem maßgeblichen Lauf."""
    return pd.read_csv(OUTPUT / "daily_returns.csv", index_col=0, parse_dates=True)


def gewichte() -> dict:
    """Monatsgewichte je Strategie; Zeilen = Rebalancing-Termine, Spalten = Titel."""
    g = {}
    for name, datei in [("Markowitz MVO", "weights_markowitz.csv"),
                        ("Random Forest", "weights_rf.csv"),
                        ("Risk Parity", "weights_risk_parity.csv")]:
        g[name] = pd.read_csv(OUTPUT / datei, index_col=0, parse_dates=True)[TICKERS]
    termine = g["Risk Parity"].index
    g["Equal Weight"] = pd.DataFrame(1 / len(TICKERS), index=termine, columns=TICKERS)
    return g


def umschlag() -> pd.DataFrame:
    """Umschlag je Strategie und Termin (Formel 26); erste Zeile = Depotaufbau."""
    return pd.read_csv(OUTPUT / "turnover.csv", index_col=0, parse_dates=True)


UMSCHLAG_SPALTE = {"Markowitz MVO": "turnover_mvo", "Random Forest": "turnover_rf",
                   "Equal Weight": "turnover_ew", "Risk Parity": "turnover_rp"}


def protokoll() -> dict:
    """experiment_log.json: Parameter, Kennzahlen, Signifikanz — mit voller Genauigkeit."""
    return json.loads((OUTPUT / "experiment_log.json").read_text(encoding="utf-8"))


def sharpe_exakt(strategie: str, je_tag: bool = False) -> float:
    """Sharpe-Quotient mit voller Genauigkeit aus dem Laborprotokoll des Laufs.

    daily_returns.csv ist auf sechs Nachkommastellen gerundet. Für Rendite und
    Volatilität reicht das, für die vierte Stelle eines Sharpe-Quotienten nicht
    immer (Risk Parity: 0,79014 exakt, 0,79015 aus der CSV). Wo der Text Sharpe-
    Quotienten auf drei oder vier Stellen nennt, wird deshalb dieser Wert benutzt."""
    eintrag = protokoll()["significance"]["deflated_sharpe"][strategie]
    return eintrag["sr_per_period"] if je_tag else eintrag["sharpe_annual"]


def sharpe(r: pd.Series, rf: float = 0.04) -> float:
    """Sharpe-Quotient wie in Formel 13/14: Mittel der Überschussrendite durch
    ihre Standardabweichung, mal Wurzel 252."""
    d = r - rf / HANDELSTAGE
    return float(d.mean() / d.std(ddof=1) * np.sqrt(HANDELSTAGE))


# ────────────────────────────────────────────────────────────────────────
# Zahlen so schreiben und runden, wie der Text es tut
# ────────────────────────────────────────────────────────────────────────
def runde(x: float, stellen: int) -> Decimal:
    """Kaufmännisch runden (0,5 → aufrunden), wie man es von Hand täte."""
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-stellen), rounding=ROUND_HALF_UP)


def de(x: float, stellen: int) -> str:
    """Zahl in deutscher Schreibweise: 17.5634 → '17,56' (bei stellen=2)."""
    s = f"{runde(x, stellen):f}".replace(".", ",").replace("-", "−")
    return s


def lies(text: str) -> tuple:
    """Zahl aus dem Wortlaut lesen: '−4,9' → (-4.9, 1 Nachkommastelle)."""
    t = text.strip().replace("−", "-").replace(".", "").replace(" ", "")
    stellen = len(t.split(",")[1]) if "," in t else 0
    return float(t.replace(",", ".")), stellen


# ────────────────────────────────────────────────────────────────────────
# Die Prüfliste eines Kapitels
# ────────────────────────────────────────────────────────────────────────
class Pruefung:
    """Sammelt alle Prüfungen eines Kapitels und druckt sie als Liste.

    Benutzung in den Kapitel-Skripten:
        p = Pruefung("Kapitel 2 — Theorie")
        p.abschnitt("2.2 Diversifikation und ihre Grenze")
        p.zahl("Volatilität Coca-Cola", "17,8", 0.17796 * 100, "%")
        ...
        p.abschluss()
    """

    ZEICHEN = {"ok": "✅", "ungefaehr": "≈ ", "falsch": "❌", "offen": "⏭ "}
    # Jeder Wortlaut, der genau so im Text stehen muss, landet hier — über alle
    # Kapitel hinweg. Daraus prüft wortlaut_fehlt(), ob er noch im Text steht.
    WORTLAUT = []

    def __init__(self, titel: str):
        self.titel = titel
        self.zaehler = {k: 0 for k in self.ZEICHEN}
        self.abweichungen = []
        self._aktuell = ""
        print("\n" + "═" * 78)
        print(f"  {titel}")
        print("═" * 78)
        print("  Legende: ✅ stimmt · ≈ stimmt ungefähr (\"rund\", \"etwa\") · "
              "❌ weicht ab · ⏭ nicht gerechnet")

    def merke(self, was: str, text: str) -> None:
        """Wortlaut, der so im Text steht, für die Prüfung mit --text vormerken."""
        Pruefung.WORTLAUT.append((self.titel, self._aktuell, was, text))

    def abschnitt(self, name: str) -> None:
        self._aktuell = name.split()[0]
        print(f"\n  § {name}")
        print("  " + "─" * 74)

    def _zeile(self, status: str, was: str, text: str, gerechnet: str, notiz: str = "") -> None:
        self.zaehler[status] += 1
        print(f"  {self.ZEICHEN[status]} {was:<44} Text: {text:<11} gerechnet: {gerechnet}")
        if notiz:
            print(f"       ↳ {notiz}")
        if status == "falsch":
            self.abweichungen.append(f"§ {self._aktuell}  {was}: Text {text}, gerechnet {gerechnet}"
                                     + (f" — {notiz}" if notiz else ""))

    def zahl(self, was: str, text: str, ist: float, einheit: str = "",
             ungefaehr: bool = False, toleranz: float = None, notiz: str = "") -> None:
        """Vergleicht eine gerechnete Zahl mit dem Wortlaut.

        text       die Zahl, wie sie im Text steht ("17,56", "−4,9", "2.244")
        ist        die gerechnete Zahl, schon in derselben Einheit (Prozent als 17,56, nicht 0,1756)
        ungefaehr  True, wenn der Text "rund/etwa/gut/knapp" sagt; dann gilt die
                   Toleranz (Standard: eine Einheit der letzten Stelle, mindestens 3 %)
        """
        self.merke(was, text)
        soll, stellen = lies(text)
        gerundet = runde(ist, stellen)
        anzeige = de(ist, stellen) + (f" {einheit}" if einheit else "")
        genau = de(ist, max(stellen + 2, 2))
        if float(gerundet) == soll:
            self._zeile("ok", was, text + (f" {einheit}" if einheit else ""), anzeige, notiz)
            return
        if ungefaehr:
            tol = toleranz if toleranz is not None else max(10 ** -stellen, abs(soll) * 0.03)
            if abs(ist - soll) <= tol:
                self._zeile("ungefaehr", was, text + (f" {einheit}" if einheit else ""),
                            f"{genau}{' ' + einheit if einheit else ''}", notiz)
                return
        self._zeile("falsch", was, text + (f" {einheit}" if einheit else ""),
                    f"{genau}{' ' + einheit if einheit else ''}", notiz)

    def aussage(self, was: str, text: str, wahr: bool, gerechnet: str, notiz: str = "",
                woertlich: bool = False) -> None:
        """Prüft eine Aussage in Worten ("sieben der fünfzehn", "unter drei Prozent").

        woertlich=True, wenn `text` genau so in der Arbeit steht (etwa ein Datum
        oder "über die Hälfte"); dann prüft --text auch, ob er noch dort steht."""
        if woertlich:
            self.merke(was, text)
        self._zeile("ok" if wahr else "falsch", was, text, gerechnet, notiz)

    def offen(self, was: str, text: str, grund: str) -> None:
        """Zahl, die hier bewusst nicht gerechnet wird — mit Begründung."""
        self._zeile("offen", was, text, "—", grund)

    def abschluss(self) -> int:
        """Druckt die Zusammenfassung und gibt die Zahl der Abweichungen zurück."""
        z = self.zaehler
        print("\n  " + "─" * 74)
        print(f"  Ergebnis {self.titel}: {z['ok']} ✅   {z['ungefaehr']} ≈   "
              f"{z['falsch']} ❌   {z['offen']} ⏭")
        if self.abweichungen:
            print("\n  Abweichungen, die im Text geprüft werden sollten:")
            for a in self.abweichungen:
                print(f"    • {a}")
        print()
        return z["falsch"]


# ────────────────────────────────────────────────────────────────────────
# Steht der eingetragene Wortlaut noch im Text?
# ────────────────────────────────────────────────────────────────────────
def _vereinheitlicht(text: str) -> str:
    """Schreibvarianten angleichen, die nichts an der Zahl ändern: Minus- und
    Gedankenstriche, geschützte Leerzeichen, Fußnotenmarken des Exports."""
    text = re.sub(r"\[\^\d+\]", "", text)                 # Fußnotenmarken [^12]
    text = text.replace("---", "—").replace("--", "–")   # Striche im Markdown-Export
    for alt, neu in [("\u2212", "-"), ("–", "-"), ("\u00a0", " "), ("\u202f", " "), ("\\", "")]:
        text = text.replace(alt, neu)
    return text


def text_laden(pfad: str) -> str:
    """Text der Arbeit laden: .docx über pandoc als Markdown, sonst als Textdatei."""
    pfad = Path(pfad).expanduser()
    if pfad.suffix.lower() == ".docx":
        try:
            ergebnis = subprocess.run(["pandoc", str(pfad), "-t", "markdown", "--wrap=none"],
                                      capture_output=True, text=True, check=True)
        except FileNotFoundError:
            sys.exit("Für .docx wird pandoc gebraucht (https://pandoc.org). "
                     "Alternativ eine Markdown-Fassung des Textes angeben.")
        return ergebnis.stdout
    return pfad.read_text(encoding="utf-8")


def wortlaut_fehlt(pfad: str) -> list:
    """Alle vorgemerkten Wortlaute, die im Text nicht (mehr) vorkommen.

    Eine Zahl gilt als gefunden, wenn sie irgendwo im Text vollständig
    dasteht: "7,46" wird in "7,46 Prozent" gefunden, aber nicht in "17,46"
    oder "7,465"; "50" nicht in "50,1". Der Abgleich weiß nicht, an
    welcher Stelle die Zahl steht — eine kurze Zahl wie "4" findet er fast
    immer. Er fängt also vor allem geänderte Dezimalzahlen und Daten ab."""
    text = _vereinheitlicht(text_laden(pfad))
    fehlt, gesehen = [], set()
    for titel, abschnitt, was, wortlaut in Pruefung.WORTLAUT:
        muster = (r"(?<![\d,])(?<!\d\.)" + re.escape(_vereinheitlicht(wortlaut).strip())
                  + r"(?!\d)(?![.,]\d)")
        if (titel, abschnitt, wortlaut) in gesehen or re.search(muster, text):
            continue
        gesehen.add((titel, abschnitt, wortlaut))
        fehlt.append((titel, abschnitt, was, wortlaut))
    return fehlt
