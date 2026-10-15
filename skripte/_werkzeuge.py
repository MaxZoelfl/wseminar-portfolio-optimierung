"""
Gemeinsame Werkzeuge für alle Nachrechnungs-Skripte in diesem Ordner.

FÜR EINSTEIGER — WAS STECKT HIER DRIN?
Jedes Kapitel-Skript (kapitel_1_einleitung.py, kapitel_2_theorie.py, …)
braucht dieselben Grundbausteine: die eingefrorenen Kurse laden, die
Ergebnisdateien des großen Backtests einlesen und am Ende jede berechnete
Zahl mit der Zahl vergleichen, die im Text der Arbeit steht. Damit das nicht
in jedem Skript neu geschrieben werden muss, liegt es einmal hier.

Der Unterstrich am Anfang des Dateinamens bedeutet: Diese Datei startet man
nicht selbst, sie wird von den anderen Skripten benutzt.

WOHER DIE DATEN KOMMEN (nichts davon erfordert einen neuen Backtest)
  data/prices.pkl              eingefrorene Schlusskurse, Abruf 15.08.2026
  output/daily_returns.csv     Tagesrenditen der vier Strategien
  output/weights_*.csv         Monatsgewichte von Markowitz, Random Forest, Risk Parity
  output/turnover.csv          Umschlag je Strategie und Monat
  output/experiment_log.json   Parameter, Kennzahlen und Signifikanztests des Laufs
  skripte/daten/               kleine Referenzdateien (siehe skripte/README.md)

WELCHE FORMELN BENUTZT WERDEN
Keine eigenen. Jede Formel kommt aus dem Paket portfolio/ — also aus genau dem
Code, der auch den Backtest und die Abbildungen rechnet: Kennzahlen aus
portfolio/metrics.py, Tests aus portfolio/significance.py, die Formeln aus
Kapitel 2 aus portfolio/theory_plots.py. So kann eine nachgerechnete Zahl nie
mit einer anderen Formel entstehen als die Zahl in der Arbeit.

WIE DER VERGLEICH MIT DEM TEXT FUNKTIONIERT
Jede Prüfung bekommt die Zahl so, wie sie im Text steht — als Zeichenkette,
etwa "17,56" oder "fünfzehn". Aus ihr liest das Werkzeug ab, auf wie viele
Nachkommastellen der Text rundet, rundet die eigene Rechnung genauso
(kaufmännisch, also 0,5 wird aufgerundet) und vergleicht. Ergebnis je Zahl:

    ✅  stimmt            gerechnet und gerundet genau wie im Text
    ≈   stimmt ungefähr   nur bei Angaben wie "rund", "etwa", "gut": die
                          Rechnung liegt nah genug am Wortlaut
    ❌  weicht ab         der Text sollte geprüft werden
    ⏭  nicht gerechnet   Zahl aus einer Quelle (Literatur) oder eine Rechnung,
                          die nur auf Wunsch läuft

Die Prüfungen stehen in den Skripten in derselben Reihenfolge wie die Zahlen im
Text. Die Ausgabe liest sich deshalb wie der Text von oben nach unten.

WICHTIG: Der Wortlaut ist in den Skripten EINGETRAGEN — sie lesen die
Word-Datei nicht selbst. Ändert sich der Text, muss der Eintrag nachgezogen
werden. Damit das nicht übersehen wird, prüft
    python skripte/alle_kapitel.py --text DATEI      (.docx oder .md)
zusätzlich, ob jeder eingetragene Wortlaut im passenden Abschnitt dieser
Fassung der Arbeit steht (Funktion wortlaut_fehlt unten).
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
os.environ.setdefault("MPLBACKEND", "Agg")          # Diagramme nie in einem Fenster öffnen
warnings.filterwarnings("ignore")                   # Bibliothekswarnungen ausblenden

import logging                                    # noqa: E402  (erst nach sys.path)
logging.getLogger("portfolio").setLevel(logging.WARNING)   # Logbuch des Pakets leise stellen

from portfolio.config import TICKERS, SPY_TICKER  # noqa: E402
from portfolio.metrics import sharpe_ratio        # noqa: E402,F401  (die eine Sharpe-Formel, Formel 13/14)

HANDELSTAGE = 252
# Reihenfolge wie in den Tabellen der Arbeit (§ 6.1, Anhang B).
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
    """Monatsgewichte je Strategie; Zeilen = Rebalancing-Termine, Spalten = Titel.
    Equal Weight wird ergänzt (immer 1/15), damit alle vier gleich behandelt werden."""
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


# Spaltenname der Strategie in turnover.csv
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
    Quotienten auf drei oder vier Stellen nennt, wird deshalb dieser Wert benutzt.
    Gerechnet wurde er im Lauf mit derselben Formel 13/14 (sharpe_ratio)."""
    eintrag = protokoll()["significance"]["deflated_sharpe"][strategie]
    return eintrag["sr_per_period"] if je_tag else eintrag["sharpe_annual"]


def effektive_titelzahl(g: pd.DataFrame) -> pd.Series:
    """Formel 35 je Termin: 1 / Σ w_i² (Kehrwert des Herfindahl-Index).
    1 bei einem einzigen Titel, 15 bei gleichen Gewichten auf 15 Titel.
    Der Backtest selbst braucht diese Kennzahl nicht; sie steht nur hier."""
    return 1 / (g ** 2).sum(axis=1)


# ────────────────────────────────────────────────────────────────────────
# Zahlen so lesen, runden und schreiben, wie der Text es tut
# ────────────────────────────────────────────────────────────────────────
# Zahlwörter, wie sie im Fließtext vorkommen ("fünfzehn Titel", "zehn Basispunkte").
ZAHLWORT = {"null": 0, "ein": 1, "eine": 1, "einen": 1, "einem": 1, "einer": 1, "eins": 1, "zwei": 2, "drei": 3,
            "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
            "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15,
            "zwanzig": 20, "fünfundzwanzig": 25, "dreißig": 30, "sechzig": 60, "hundert": 100,
            "anderthalb": 1.5}


def runde(x: float, stellen: int) -> Decimal:
    """Kaufmännisch runden (0,5 → aufrunden), wie man es von Hand täte."""
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-stellen), rounding=ROUND_HALF_UP)


def de(x: float, stellen: int) -> str:
    """Zahl in deutscher Schreibweise: 17.5634 → '17,56' (bei stellen=2)."""
    return f"{runde(x, stellen):f}".replace(".", ",").replace("-", "−")


def lies(text: str) -> tuple:
    """Zahl aus dem Wortlaut lesen: '−4,9' → (-4.9, 1 Nachkommastelle),
    '2.244' → (2244, 0), 'fünfzehn' → (15, 0)."""
    t = text.strip().lower()
    if t in ZAHLWORT:                                  # "anderthalb" hat eine Nachkommastelle
        wert = float(ZAHLWORT[t])
        return wert, 0 if wert.is_integer() else 1
    t = t.replace("−", "-").replace(".", "").replace(" ", "")
    stellen = len(t.split(",")[1]) if "," in t else 0
    return float(t.replace(",", ".")), stellen


# ────────────────────────────────────────────────────────────────────────
# Die Prüfliste eines Kapitels
# ────────────────────────────────────────────────────────────────────────
class Pruefung:
    """Sammelt alle Prüfungen eines Kapitels und druckt sie als Liste.

    Benutzung in den Kapitel-Skripten, immer in der Reihenfolge des Textes:
        p = Pruefung("Kapitel 2 — Theoretische Grundlagen")
        p.abschnitt("2.2 Diversifikation und ihre Grenze")
        p.zahl("Volatilität von Coca-Cola", "17,8", 0.17796 * 100, "%")
        p.quelle("Beobachtungsdauer nach DeMiguel u. a.", "3000", "Fn 6")
        ...
        p.abschluss()

    Jede Zeile der Ausgabe zeigt zuerst die Zahl GENAU SO, wie sie im Text
    steht, dann, was sie bedeutet. Nur wenn die Rechnung etwas anderes ergibt,
    folgt dahinter der gerechnete Wert.
    """

    ZEICHEN = {"ok": "✅", "ungefaehr": "≈ ", "falsch": "❌", "offen": "⏭ "}
    # Jeder Wortlaut, der genau so im Text stehen muss, landet hier — über alle
    # Kapitel hinweg. Daraus prüft wortlaut_fehlt(), ob er noch im Text steht.
    WORTLAUT = []

    def __init__(self, titel: str):
        """Kopf der Liste drucken und die Zähler für ✅/≈/❌/⏭ auf null setzen."""
        self.titel = titel
        self.zaehler = {k: 0 for k in self.ZEICHEN}
        self.abweichungen = []
        self._aktuell = ""
        print("\n" + "═" * 78)
        print(f"  {titel}")
        print("═" * 78)
        print("  ✅ nachgerechnet, ergibt genau die Zahl im Text   ≈ passt zu „rund“/„etwa“")
        print("  ❌ Rechnung ergibt etwas anderes                   ⏭ nicht gerechnet (Grund dahinter)")

    def abschnitt(self, name: str) -> None:
        """Neue Zwischenüberschrift, etwa "2.2 Diversifikation und ihre Grenze".
        Das erste Wort ("2.2", "Tabelle A3", "C1" …) ist zugleich die Adresse
        des Abschnitts für den Abgleich mit --text."""
        w = name.replace(":", " ").split()
        self._aktuell = " ".join(w[:2]) if w[0] in ("Tabelle", "Abbildung") else w[0]
        print(f"\n  § {name}" if name[0].isdigit() else f"\n  {name}")
        print("  " + "─" * 74)

    def merke(self, was: str, text: str) -> None:
        """Wortlaut, der so im Text steht, für die Prüfung mit --text vormerken."""
        Pruefung.WORTLAUT.append((self.titel, self._aktuell, was, text))

    def _zeile(self, status: str, was: str, text: str, gerechnet: str = "", notiz: str = "") -> None:
        """Eine Ausgabezeile: Zeichen, Zahl wie im Text, Bedeutung — und nur bei
        Abweichungen dahinter der gerechnete Wert."""
        self.zaehler[status] += 1
        zeile = f"  {self.ZEICHEN[status]}  {text:<16} {was}"
        if status in ("ungefaehr", "falsch") and gerechnet:
            zeile += f"   → gerechnet {gerechnet}"
        print(zeile)
        if notiz:
            print(f"      {'':<16} ↳ {notiz}")
        if status == "falsch":
            self.abweichungen.append(f"§ {self._aktuell}  {was}: Text {text}, gerechnet {gerechnet}"
                                     + (f" — {notiz}" if notiz else ""))

    def zahl(self, was: str, text: str, ist: float, einheit: str = "",
             ungefaehr: bool = False, toleranz: float = None, notiz: str = "") -> None:
        """Vergleicht eine gerechnete Zahl mit dem Wortlaut.

        was        was die Zahl bedeutet, in einfachen Worten
        text       die Zahl, wie sie im Text steht ("17,56", "−4,9", "2.244", "fünfzehn")
        ist        die gerechnete Zahl, schon in derselben Einheit (Prozent als 17,56, nicht 0,1756)
        ungefaehr  True, wenn der Text "rund/etwa/gut/knapp" sagt; dann gilt die
                   Toleranz (Standard: das Größere aus einer Einheit der letzten
                   Stelle und 3 % des Werts)
        """
        self.merke(was, text)
        soll, stellen = lies(text)
        if text.strip().lower() in ZAHLWORT:            # "vier Prozent" statt "vier %"
            einheit = {"%": "Prozent", "Pp": "Prozentpunkt" if soll == 1 else "Prozentpunkte"}.get(einheit, einheit)
        anzeige = text + (f" {einheit}" if einheit else "")
        genau = de(ist, max(stellen + 2, 2)) + (f" {einheit}" if einheit else "")
        if float(runde(ist, stellen)) == soll:
            self._zeile("ok", was, anzeige, notiz=notiz)
            return
        if ungefaehr:
            tol = toleranz if toleranz is not None else max(10 ** -stellen, abs(soll) * 0.03)
            if abs(ist - soll) <= tol:
                self._zeile("ungefaehr", was, anzeige, genau, notiz)
                return
        self._zeile("falsch", was, anzeige, genau, notiz)

    def aussage(self, was: str, text: str, wahr: bool, gerechnet: str, notiz: str = "",
                woertlich: bool = False) -> None:
        """Prüft eine Aussage in Worten ("sieben der fünfzehn", "Markowitz").

        woertlich=True, wenn `text` genau so in der Arbeit steht (etwa ein Datum
        oder "rund die Hälfte"); dann prüft --text auch, ob er noch dort steht.
        Der gerechnete Befund erscheint nur, wenn die Aussage nicht stimmt — außer
        er steht in der Notiz."""
        if woertlich:
            self.merke(was, text)
        self._zeile("ok" if wahr else "falsch", was, text, gerechnet, notiz)

    def quelle(self, was: str, text: str, beleg: str) -> None:
        """Zahl aus der Literatur: Sie wird nicht gerechnet, sondern belegt.
        Mit --text wird trotzdem geprüft, ob sie noch so im Text steht."""
        self.merke(was, text)
        self._zeile("offen", was, text, notiz=f"Zahl aus der Quelle ({beleg})")

    def offen(self, was: str, text: str, grund: str) -> None:
        """Zahl, die hier bewusst nicht gerechnet wird — mit Begründung."""
        self._zeile("offen", was, text, notiz=grund)

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
    for alt, neu in [("−", "-"), ("–", "-"), (" ", " "), (" ", " "), ("\\", "")]:
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


def abschnitte(text: str) -> dict:
    """Den Text an seinen Überschriften in Abschnitte zerlegen.

    Schlüssel ist die Adresse, wie die Skripte sie benutzen: "2.2" für
    „## 2.2 Diversifikation …", "Tabelle A3" für „### Tabelle A3: …", "C1" für
    „### C1: …". Ein Kapitel ("2") umfasst alle seine Abschnitte."""
    teile, adresse, puffer = {}, "", []
    for zeile in text.split("\n"):
        m = re.match(r"^#{1,4}\s+(?:Anhang [A-E]:\s*)?(.*)$", zeile)
        if m:
            teile.setdefault(adresse, []).extend(puffer)
            puffer, kopf = [], m.group(1).strip()
            w = kopf.replace(":", " ").split()
            adresse = " ".join(w[:2]) if w and w[0] in ("Tabelle", "Abbildung") else (w[0] if w else "")
            puffer.append(zeile)
            continue
        puffer.append(zeile)
    teile.setdefault(adresse, []).extend(puffer)
    ergebnis = {k: "\n".join(v) for k, v in teile.items()}
    for k in list(ergebnis):                      # Kapitel = Summe seiner Abschnitte
        if re.match(r"^\d+\.\d+$", k):
            ergebnis[k.split(".")[0]] = ergebnis.get(k.split(".")[0], "") + "\n" + ergebnis[k]
    return ergebnis


def wortlaut_fehlt(pfad: str) -> list:
    """Alle vorgemerkten Wortlaute, die im zugehörigen Abschnitt des Textes
    nicht (mehr) vorkommen.

    Eine Zahl gilt als gefunden, wenn sie im richtigen Abschnitt vollständig
    dasteht: "7,46" wird in "7,46 Prozent" gefunden, aber nicht in "17,46"
    oder "7,465"; "50" nicht in "50,1". Gesucht wird im Abschnitt, in dem die
    Prüfung steht — so fällt auch eine geänderte kurze Zahl auf, die zufällig
    anderswo im Text vorkommt."""
    teile = {k: _vereinheitlicht(v) for k, v in abschnitte(text_laden(pfad)).items()}
    ganz = _vereinheitlicht(text_laden(pfad))
    fehlt, gesehen = [], set()
    for titel, abschnitt, was, wortlaut in Pruefung.WORTLAUT:
        bereich = teile.get(abschnitt, ganz)
        muster = (r"(?<![\d,])(?<!\d\.)" + re.escape(_vereinheitlicht(wortlaut).strip())
                  + r"(?!\d)(?![.,]\d)")
        if (titel, abschnitt, wortlaut) in gesehen or re.search(muster, bereich, re.IGNORECASE):
            continue
        gesehen.add((titel, abschnitt, wortlaut))
        fehlt.append((titel, abschnitt, was, wortlaut))
    return fehlt
