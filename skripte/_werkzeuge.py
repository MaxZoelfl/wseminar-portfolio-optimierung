"""
Gemeinsame Werkzeuge für alle Nachrechnungs-Skripte in diesem Ordner.

FÜR EINSTEIGER — WAS STECKT HIER DRIN?
Jedes Kapitel-Skript (kapitel_1_einleitung.py, kapitel_2_theorie.py, …)
braucht dieselben Grundbausteine: die eingefrorenen Kurse laden, die
Ergebnisdateien des großen Backtests einlesen und jede Zahl mit ihrer
Herleitung ausgeben. Damit das nicht in jedem Skript neu geschrieben werden
muss, liegt es einmal hier.

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

WAS AUSGEGEBEN WIRD
Jede Zahl eines Kapitels steht in einer eigenen Zeile, in der Reihenfolge, in
der sie in der Arbeit vorkommt:

    17,8 %             Volatilität von Coca-Cola
                       ↳ Herleitung, Quelle oder genauerer Wert

Links steht der Wert, den die RECHNUNG ergibt — aus Kursen, Ergebnisdateien
oder Programmcode, gerundet auf so viele Stellen wie in der Arbeit. Zahlen
aus der Literatur werden nicht gerechnet, sondern mit ihrer Fußnote genannt;
Definitionen und Annahmen (etwa ein Rechenbeispiel) mit einer Erklärung.
Am Ende jedes Kapitels steht, wie viele Zahlen woher stammen.

FÜR DEN VERFASSER: ABGLEICH MIT DEM WORTLAUT
In jedem Skript steht zu jeder Zahl auch, wie sie in der Arbeit geschrieben
ist (etwa "17,8" oder "fünfzehn"). Daraus liest das Werkzeug die Stellenzahl
ab, auf die gerundet wird. Mit dem Schalter --abgleich vergleicht es außerdem
Rechnung und Wortlaut und markiert jede Zahl:

    ✅  stimmt            gerechnet und gerundet genau wie in der Arbeit
    ≈   stimmt ungefähr   nur bei Angaben wie "rund", "etwa", "gut"
    ❌  weicht ab         die Stelle in der Arbeit sollte geprüft werden
    ⏭  nicht gerechnet   Zahl aus der Literatur, Definition oder Annahme

    python skripte/alle_kapitel.py --abgleich
    python skripte/alle_kapitel.py --text DATEI      (.docx oder .md)

--text prüft zusätzlich, ob jeder eingetragene Wortlaut im passenden
Abschnitt dieser Fassung der Arbeit steht (Funktion wortlaut_fehlt unten).
"""

import inspect
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
DATEN = Path(__file__).resolve().parent / "daten"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")          # Diagramme nie in einem Fenster öffnen
warnings.filterwarnings("ignore")                   # Bibliothekswarnungen ausblenden

import logging                                    # noqa: E402  (erst nach sys.path)
logging.getLogger("portfolio").setLevel(logging.WARNING)   # Logbuch des Pakets leise stellen

from dataclasses import fields                    # noqa: E402
from portfolio.config import CFG, Config, TICKERS, SPY_TICKER, OHNE_EINFLUSS_AUF_ERGEBNISSE   # noqa: E402
from portfolio.metrics import sharpe_ratio        # noqa: E402,F401  (die eine Sharpe-Formel, Formel 13/14)

# Ergebnisordner und Kursdatei so, wie die Einstellungen sie festlegen: für die
# Arbeit output/ und data/prices.pkl. Bei einem anderen Universum oder Zeitraum
# vergibt portfolio/config.py eigene Pfade (output_<kennung>, prices_<kennung>.pkl).
OUTPUT = (ROOT / CFG.output_dir).resolve()
KURSDATEI = (ROOT / CFG.price_cache).resolve()
# Für Ausgaben: Pfad relativ zum Projektordner, sofern die Kursdatei darin liegt.
KURSDATEI_ANZEIGE = str(KURSDATEI.relative_to(ROOT)) if KURSDATEI.is_relative_to(ROOT) else str(KURSDATEI)

# Rechnen die Skripte mit dem Lauf der Arbeit? Zwei Fragen:
#  UNIVERSUM_DER_ARBEIT  dieselben Titel, Referenzreihe und derselbe Zeitraum —
#                        Voraussetzung für die Abschnitte über bestimmte Titel
#                        (etwa Coca-Cola und Caterpillar in § 2.2), soweit die
#                        Skripte das nicht ohnehin an den Titeln selbst prüfen;
#  LAUF_DER_ARBEIT       zusätzlich alle übrigen Einstellungen gleich, die Ergebnisse
#                        verändern — Voraussetzung für die Referenzdateien in
#                        skripte/daten/, die aus dem Lauf der Arbeit stammen.
# Trifft eine Frage nicht zu, entfällt der betroffene Teil mit einem Hinweis.
_ARBEIT = Config()
UNIVERSUM_DER_ARBEIT = (sorted(TICKERS), SPY_TICKER, CFG.start_date, CFG.end_date) == \
    (sorted(_ARBEIT.tickers), _ARBEIT.spy_ticker, _ARBEIT.start_date, _ARBEIT.end_date)
ABWEICHENDE_EINSTELLUNGEN = [f.name for f in fields(Config) if f.name not in OHNE_EINFLUSS_AUF_ERGEBNISSE
                             and getattr(CFG, f.name) != getattr(_ARBEIT, f.name)]
LAUF_DER_ARBEIT = not ABWEICHENDE_EINSTELLUNGEN

HANDELSTAGE = 252
# Reihenfolge wie in den Tabellen der Arbeit (§ 6.1, Anhang B).
STRATEGIEN = ["Random Forest", "Markowitz MVO", "Equal Weight", "Risk Parity"]
KURZ = {"Random Forest": "RF", "Markowitz MVO": "MVO",
        "Equal Weight": "EW", "Risk Parity": "RP"}


# ────────────────────────────────────────────────────────────────────────
# Daten laden
# ────────────────────────────────────────────────────────────────────────
def kurse() -> pd.DataFrame:
    """Schlusskurse der Titel und der Referenzreihe (SPY), nur Tage mit allen
    Kursen (genau wie portfolio/data.py). Gelesen wird die Kursdatei, mit der der
    Backtest gerechnet hat."""
    if not KURSDATEI.exists():
        sys.exit(f"Kursdatei {KURSDATEI} fehlt — zuerst den Backtest laufen lassen "
                 "(python -m portfolio).")
    roh = pd.read_pickle(KURSDATEI)
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
    """experiment_log.json: Parameter, Kennzahlen, Signifikanz — mit voller Genauigkeit.

    Bricht mit einer klaren Meldung ab, wenn der Ergebnisordner fehlt oder zu
    anderen Titeln gehört als die Einstellungen (dann passen Gewichte, Renditen
    und Kurse nicht zusammen)."""
    datei = OUTPUT / "experiment_log.json"
    if not datei.exists():
        sys.exit(f"Ergebnisordner {OUTPUT.name}/ fehlt — zuerst den Backtest laufen lassen (python -m portfolio).")
    log = json.loads(datei.read_text(encoding="utf-8"))
    if sorted(log["experiment_meta"]["tickers"]) != sorted(TICKERS):
        sys.exit(f"Die Ergebnisse in {OUTPUT.name}/ gehören zu anderen Titeln als die Einstellungen "
                 "(config.json). Backtest mit diesen Titeln laufen lassen oder config.json anpassen.")
    return log


def sharpe_exakt(strategie: str, je_tag: bool = False) -> float:
    """Sharpe-Quotient mit voller Genauigkeit aus dem Laborprotokoll des Laufs.

    daily_returns.csv ist auf sechs Nachkommastellen gerundet. Für Rendite und
    Volatilität reicht das, für die vierte Stelle eines Sharpe-Quotienten nicht
    immer (Risk Parity: 0,79014 exakt, 0,79015 aus der CSV). Wo der Text Sharpe-
    Quotienten auf drei oder vier Stellen nennt, wird deshalb dieser Wert benutzt.
    Gerechnet wurde er im Lauf mit derselben Formel 13/14 (sharpe_ratio)."""
    eintrag = protokoll()["significance"]["deflated_sharpe"][strategie]
    return eintrag["sr_per_period"] if je_tag else eintrag["sharpe_annual"]


def zahl_im_code(quelltext: str, muster: str) -> float:
    """Eine Zahl aus dem Programmcode lesen, etwa die 252 in "len(train_daily) < 252".

    muster ist ein regulärer Ausdruck mit einer Klammergruppe um die Zahl. So
    stammt eine Einstellung, die nur im Code steht (und nicht im Laborprotokoll),
    trotzdem aus dem Code selbst und nicht aus einer Abschrift."""
    treffer = re.search(muster, quelltext)
    if treffer is None:
        raise ValueError(f"Im Programmcode nicht gefunden: {muster}")
    return float(treffer.group(1))


def std_wert(funktion, parameter: str):
    """Voreinstellung eines Parameters im Programmcode, etwa window=252 in
    rollierender_sharpe — so kommt die Zahl aus dem Code selbst."""
    return inspect.signature(funktion).parameters[parameter].default


def kursdatei_commits() -> list:
    """Alle Commits, die die Kursdatei verändert haben, als (Kennung, Datum),
    der jüngste zuerst. Leer, wenn git nicht verfügbar ist (kein Klon des
    Repositorys) oder die Kursdatei nicht in der Versionsgeschichte liegt."""
    try:
        aus = subprocess.run(["git", "log", "--format=%h %ad", "--date=format:%d.%m.%Y", "--",
                              str(KURSDATEI.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return []
    return [tuple(z.split()) for z in aus.strip().splitlines() if z.strip()]


def effektive_titelzahl(g: pd.DataFrame) -> pd.Series:
    """Formel 35 je Termin: 1 / Σ w_i² (Kehrwert des Herfindahl-Index).
    1 bei einem einzigen Titel, 15 bei gleichen Gewichten auf 15 Titel.
    Der Backtest selbst braucht diese Kennzahl nicht; sie steht nur hier."""
    return 1 / (g ** 2).sum(axis=1)


# ────────────────────────────────────────────────────────────────────────
# Zahlen so lesen, runden und schreiben, wie die Arbeit es tut
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
# Die Zahlenliste eines Kapitels
# ────────────────────────────────────────────────────────────────────────
# Abgleich mit dem Wortlaut der Arbeit nur auf Wunsch (für den Verfasser):
# python skripte/alle_kapitel.py --abgleich  oder  … --text DATEI
ABGLEICH = "--abgleich" in sys.argv or "--text" in sys.argv


class Pruefung:
    """Sammelt alle Zahlen eines Kapitels und druckt sie als Liste.

    Benutzung in den Kapitel-Skripten, immer in der Reihenfolge der Arbeit:
        p = Pruefung("Kapitel 2 — Theoretische Grundlagen")
        p.abschnitt("2.2 Diversifikation und ihre Grenze")
        p.zahl("Volatilität von Coca-Cola", "17,8", 0.17796 * 100, "%")
        p.quelle("Beobachtungsdauer nach DeMiguel u. a.", "3000", "Fn 6")
        ...
        p.abschluss()

    Vier Arten von Einträgen:
      zahl     eine Zahl, die gerechnet wird; ausgegeben wird das Ergebnis der
               Rechnung, gerundet auf die Stellenzahl der Arbeit
      aussage  ein Befund in Worten oder als Datum (etwa "Markowitz",
               "31.12.2015"), ermittelt aus Daten oder Programmcode
      quelle   eine Zahl aus der Literatur; sie wird nicht gerechnet, sondern
               mit ihrer Fußnote genannt
      offen    eine Definition, Konstante oder Annahme, mit Erklärung

    Ohne --abgleich zeigt jede Zeile den gerechneten Wert und seine Bedeutung.
    Mit --abgleich (für den Verfasser) zeigt sie stattdessen den Wortlaut der
    Arbeit und ein Zeichen, ob die Rechnung ihn bestätigt.
    """

    ZEICHEN = {"ok": "✅", "ungefaehr": "≈ ", "falsch": "❌", "offen": "⏭ "}
    # Jeder Wortlaut, der genau so in der Arbeit stehen muss, landet hier — über alle
    # Kapitel hinweg. Daraus prüft wortlaut_fehlt(), ob er noch in der Arbeit steht.
    WORTLAUT = []

    def __init__(self, titel: str):
        """Kopf der Liste drucken und die Zähler auf null setzen."""
        self.titel = titel
        self.zaehler = {k: 0 for k in self.ZEICHEN}                       # für --abgleich
        self.herkunft = {"berechnet": 0, "literatur": 0, "festlegung": 0}  # für die Standardausgabe
        self.abweichungen = []
        self._aktuell = ""
        print("\n" + "═" * 78)
        print(f"  {titel}")
        print("═" * 78)
        if ABGLEICH:
            print("  ✅ nachgerechnet, ergibt genau die Zahl der Arbeit   ≈ passt zu „rund“/„etwa“")
            print("  ❌ Rechnung ergibt etwas anderes                     ⏭ nicht gerechnet (Grund dahinter)")
        else:
            print("  Jede Zahl in der Reihenfolge der Arbeit: links der Wert, den die Rechnung")
            print("  ergibt (gerundet wie in der Arbeit), rechts seine Bedeutung, darunter (↳)")
            print("  Herleitung, Quelle oder genauerer Wert.")

    def abschnitt(self, name: str) -> None:
        """Neue Zwischenüberschrift, etwa "2.2 Diversifikation und ihre Grenze".
        Das erste Wort ("2.2", "Tabelle A3", "C1" …) ist zugleich die Adresse
        des Abschnitts für den Abgleich mit --text."""
        w = name.replace(":", " ").split()
        self._aktuell = " ".join(w[:2]) if w[0] in ("Tabelle", "Abbildung") else w[0]
        print(f"\n  § {name}" if name[0].isdigit() else f"\n  {name}")
        print("  " + "─" * 74)

    def merke(self, was: str, text: str) -> None:
        """Wortlaut, der so in der Arbeit steht, für den Abgleich mit --text vormerken."""
        Pruefung.WORTLAUT.append((self.titel, self._aktuell, was, text))

    def _zeile(self, status: str, herkunft: str, was: str, wert: str, wortlaut: str,
               gerechnet: str = "", notiz: str = "") -> None:
        """Eine Ausgabezeile.

        Standard:    gerechneter Wert · Bedeutung · darunter die Notiz.
        --abgleich:  Zeichen · Wortlaut der Arbeit · Bedeutung · bei Abweichung
                     der gerechnete Wert."""
        self.zaehler[status] += 1
        self.herkunft[herkunft] += 1
        if ABGLEICH:
            zeile = f"  {self.ZEICHEN[status]}  {wortlaut:<16} {was}"
            if status in ("ungefaehr", "falsch") and gerechnet:
                zeile += f"   → gerechnet {gerechnet}"
            print(zeile)
        else:
            print(f"  {wert:<18} {was}" if len(wert) < 18 else f"  {wert}  {was}")
        if notiz:
            print(f"  {'':<18} ↳ {notiz}" if not ABGLEICH else f"      {'':<16} ↳ {notiz}")
        if status == "falsch":
            self.abweichungen.append(f"§ {self._aktuell}  {was}: Arbeit {wortlaut}, gerechnet {gerechnet}"
                                     + (f" — {notiz}" if notiz else ""))

    def zahl(self, was: str, text: str, ist: float, einheit: str = "",
             ungefaehr: bool = False, toleranz: float = None, notiz: str = "") -> None:
        """Eine gerechnete Zahl.

        was        was die Zahl bedeutet, in einfachen Worten
        text       die Zahl, wie sie in der Arbeit steht ("17,56", "−4,9", "2.244",
                   "fünfzehn"); aus ihr kommt die Stellenzahl, auf die gerundet wird
        ist        die gerechnete Zahl, schon in derselben Einheit (Prozent als 17,56, nicht 0,1756)
        ungefaehr  True bei Angaben wie "rund", "etwa", "gut", "knapp": Dann steht
                   unter der Zahl zusätzlich ihr ungerundeter Wert (und --abgleich
                   lässt eine Toleranz zu: das Größere aus einer Einheit der letzten
                   Stelle und 3 % des Werts)
        """
        self.merke(was, text)
        soll, stellen = lies(text)
        genau = de(ist, max(stellen + 2, 2))
        if float(runde(ist, stellen)) == soll:
            status = "ok"
        elif ungefaehr and abs(ist - soll) <= (toleranz if toleranz is not None
                                               else max(10 ** -stellen, abs(soll) * 0.03)):
            status = "ungefaehr"
        else:
            status = "falsch"
        wert = de(ist, stellen) + (f" {einheit}" if einheit else "")
        wortlaut = text
        if text.strip().lower() in ZAHLWORT:            # "vier Prozent" statt "vier %"
            wortlaut += " " + {"%": "Prozent", "Pp": "Prozentpunkt" if soll == 1 else "Prozentpunkte"}.get(einheit, einheit)
        elif einheit:
            wortlaut += f" {einheit}"
        gerundet_anders = float(runde(ist, stellen)) != float(runde(ist, stellen + 2))
        if ungefaehr and gerundet_anders and not ABGLEICH and "ungerundet" not in notiz:
            zusatz = f"ungerundet {genau}" + (f" {einheit}" if einheit else "")
            notiz = f"{notiz}; {zusatz}" if notiz else zusatz
        self._zeile(status, "berechnet", was, wert, wortlaut.strip(), genau + (f" {einheit}" if einheit else ""), notiz)

    def aussage(self, was: str, text: str, wahr: bool, gerechnet: str, notiz: str = "",
                woertlich: bool = False) -> None:
        """Ein Befund in Worten oder als Datum, ermittelt aus Daten oder Programmcode.

        text       wie die Arbeit ihn formuliert ("Markowitz", "31.12.2015")
        wahr       ob die Rechnung ihn bestätigt
        gerechnet  was die Rechnung ergibt — das ist der Wert, der ausgegeben wird
        woertlich  True, wenn `text` genau so in der Arbeit steht; dann prüft --text,
                   ob er dort noch steht"""
        if woertlich:
            self.merke(was, text)
        self._zeile("ok" if wahr else "falsch", "berechnet", was, gerechnet, text, gerechnet, notiz)

    def quelle(self, was: str, text: str, beleg: str) -> None:
        """Zahl aus der Literatur: Sie wird nicht gerechnet, sondern belegt.
        Mit --text wird trotzdem geprüft, ob sie noch so in der Arbeit steht."""
        self.merke(was, text)
        wert = text                                       # Ausgabe wie die übrigen Zahlen: Ziffern,
        try:                                              # Tausender mit Leerzeichen ("30 000")
            zahl_, stellen = lies(text)
            wert = f"{zahl_:,.0f}".replace(",", " ") if zahl_ >= 10_000 else de(zahl_, stellen)
        except ValueError:                                # Wörter wie "die Hälfte" bleiben stehen
            pass
        self._zeile("offen", "literatur", was, wert, text, notiz=f"aus der Literatur ({beleg})")

    def hinweis(self, text: str) -> None:
        """Ein Abschnitt oder eine Zahl entfällt, mit Begründung — etwa weil sie
        sich auf einen bestimmten Titel der Arbeit bezieht, der im gewählten
        Universum fehlt. Wird nicht mitgezählt."""
        print(f"  (entfällt: {text})")

    def offen(self, was: str, text: str, grund: str) -> None:
        """Definition, Konstante oder Annahme, die nicht aus den Daten stammt — mit Erklärung."""
        self._zeile("offen", "festlegung", was, text, text, notiz=grund)

    def abschluss(self) -> int:
        """Druckt die Zusammenfassung. Rückgabe (für den Exit-Code): mit --abgleich die
        Zahl der Abweichungen vom Wortlaut, sonst 0 — ohne Abgleich ist eine andere
        Zahl als in der Arbeit kein Fehler (etwa bei einem anderen Universum)."""
        print("\n  " + "─" * 74)
        if ABGLEICH:
            z = self.zaehler
            print(f"  Ergebnis {self.titel}: {z['ok']} ✅   {z['ungefaehr']} ≈   "
                  f"{z['falsch']} ❌   {z['offen']} ⏭")
            if self.abweichungen:
                print("\n  Abweichungen, die in der Arbeit geprüft werden sollten:")
                for a in self.abweichungen:
                    print(f"    • {a}")
        else:
            h = self.herkunft
            print(f"  {self.titel.split(' — ')[0]}: {sum(h.values())} Zahlen — {h['berechnet']} aus Daten und "
                  f"Programmcode berechnet, {h['literatur']} aus der Literatur, "
                  f"{h['festlegung']} Definitionen und Annahmen")
        print()
        return self.zaehler["falsch"] if ABGLEICH else 0


# ────────────────────────────────────────────────────────────────────────
# Für den Verfasser: Steht der eingetragene Wortlaut noch in der Arbeit?
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
