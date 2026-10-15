"""
Zentrale Konfiguration & Umgebungs-Setup für das Portfolio-Optimierungsprojekt.

════════════════════════════════════════════════════════════════════════════
FÜR EINSTEIGER — WORUM GEHT ES IN DIESEM PROJEKT?
════════════════════════════════════════════════════════════════════════════
Dieses Programm vergleicht vier "Anlagestrategien", also vier verschiedene
Rezepte, wie man Geld auf 15 Aktien (Apple, Microsoft, Coca-Cola, …) verteilt:

  1. "Markowitz MVO"  — die klassische Mathematik-Methode von 1952: Sie sucht
                        die Mischung mit dem besten Verhältnis aus erwartetem
                        Gewinn und Schwankungsrisiko.
  2. "Random Forest"  — eine Methode der Künstlichen Intelligenz: Ein Computer-
                        modell versucht, die Renditen des nächsten Monats
                        vorherzusagen, und mischt danach das Depot.
  3. "Equal Weight"   — die denkbar einfachste Regel: Jede Aktie bekommt genau
                        gleich viel Geld (bei 15 Aktien je 1/15).
  4. "Risk Parity"    — jede Aktie soll gleich viel zum GESAMTRISIKO beitragen
                        (schwankungsarme Aktien bekommen mehr Gewicht).

Der Vergleich läuft als sogenannter "Backtest": Das Programm tut so, als hätte
es von 2015 bis 2024 jeden Monat neu investiert — natürlich nur mit Daten, die
zum jeweiligen Zeitpunkt wirklich bekannt waren. Am Ende wird statistisch
geprüft, ob eine Strategie WIRKLICH besser war oder ob der Unterschied bloß
Zufall sein könnte.

WAS MACHT SPEZIELL DIESE DATEI (config.py)?
Sie ist die "Schaltzentrale" des Projekts: Alle Einstellungen (welche Aktien,
welcher Zeitraum, welche Kostenannahmen, …) stehen hier an EINEM Ort. Alle
anderen Dateien lesen ihre Einstellungen von hier. So muss man zum Ändern
eines Parameters nie im Rechen-Code herumsuchen.

KLEINES PROGRAMMIER-GLOSSAR (gilt für alle Dateien des Projekts):
  - "import X"      : lädt ein fertiges Zusatzpaket (Bibliothek), damit wir
                      dessen Funktionen benutzen können — wie ein Werkzeugkoffer.
  - "def name(...)" : definiert eine FUNKTION, also einen benannten, wieder-
                      verwendbaren Arbeitsschritt (Zutaten rein → Ergebnis raus).
  - "class Name"    : definiert eine KLASSE, also einen Bauplan für Objekte,
                      die Daten UND passende Funktionen ("Methoden") bündeln.
  - "# Text"        : eine Kommentarzeile — wird vom Computer ignoriert und
                      dient nur der Erklärung für menschliche Leser.
  - Ein Text in dreifachen Anführungszeichen (wie dieser hier) heißt
    "Docstring" und beschreibt eine Datei/Funktion/Klasse.
  - "DataFrame"     : eine Tabelle (wie in Excel) aus der Bibliothek pandas;
                      eine "Series" ist eine einzelne Tabellen-Spalte.

Was in dieser Datei sonst noch passiert, in der Reihenfolge von oben:
  - Grafikausgabe einrichten: Diagramme in einem Fenster, wenn ein Bildschirm
    da ist, sonst nur als Dateien
  - Zusatzpakete prüfen (shap, yfinance, Animation): Fehlt eines, läuft das
    Programm trotzdem, nur die betroffene Abbildung entfällt
  - das Logbuch (Protokollausgaben im Terminal) einrichten
  - die Einstellungen selbst (Abschnitt "class Config"); sie sind der einzige
    Ort, an dem ein Wert festgelegt wird
  - Kurznamen in Großbuchstaben (TICKERS, RISK_FREE_RATE, …), unter denen die
    übrigen Dateien die Einstellungen lesen
  - das einheitliche Aussehen aller Diagramme

Einstellungen ändern, ohne den Code anzufassen:
  - eine Datei ``config.json`` im Arbeitsverzeichnis ablegen, ODER
  - die Umgebungsvariable ``PORTFOLIO_CONFIG=/pfad/zu/meiner.json`` setzen.
Beispiel siehe ``config.schnell.json``. Die Datei wird beim ersten Laden dieses
Moduls gelesen und gilt dann für das ganze Projekt.

Diese Datei benutzt selbst keine andere Datei des Projekts; sie ist das
Fundament, auf dem alle anderen aufbauen.
"""

# ---------------------------------------------------------------------------
# IMPORTE: Werkzeugkästen aus der Python-Standardbibliothek laden.
# ---------------------------------------------------------------------------
import os        # Zugriff auf das Betriebssystem (Dateipfade, Umgebungsvariablen)
import json      # Lesen/Schreiben des JSON-Textformats (für die Konfig-Datei)
import warnings  # Steuerung von Warnmeldungen
import logging   # Protokoll-Ausgaben ("Logbuch" des Programms im Terminal)
from dataclasses import dataclass, field, fields, asdict  # bequeme Datencontainer
from typing import Optional  # "Optional[float]" = eine Zahl ODER None ("nicht gesetzt")

import matplotlib  # die zentrale Bibliothek zum Zeichnen von Diagrammen

# ---------------------------------------------------------------------------
# GRAFIK-AUSGABE EINRICHTEN
# Matplotlib kann Diagramme entweder live in einem Fenster anzeigen
# ("interaktives Backend") oder nur unsichtbar in Bilddateien schreiben
# ("Agg"-Backend — nötig auf Servern ohne Bildschirm, sogenannt "headless").
# Wir PROBIEREN zuerst den Fenster-Modus; scheitert das (kein Bildschirm
# vorhanden), schalten wir automatisch auf den unsichtbaren Modus um.
# "try/except" ist Pythons Sicherheitsnetz: Der try-Block wird versucht,
# und NUR wenn dabei ein Fehler auftritt, läuft der except-Block.
# ---------------------------------------------------------------------------
_BACKEND_TESTED = False
INTERACTIVE_DISPLAY = False   # merkt sich: Können wir Live-Fenster anzeigen?
try:
    import matplotlib.pyplot as plt
    _test = plt.figure(figsize=(1, 1))   # Mini-Testbild erzeugen …
    plt.close(_test)                     # … und sofort wieder schließen.
    INTERACTIVE_DISPLAY = True           # Hat geklappt → Live-Anzeige möglich.
    _BACKEND_TESTED = True
except Exception:
    matplotlib.use("Agg")                # Fallback: nur in Dateien zeichnen.
    import matplotlib.pyplot as plt
    INTERACTIVE_DISPLAY = False
    _BACKEND_TESTED = True

# ---------------------------------------------------------------------------
# OPTIONALE ZUSATZPAKETE
# Manche Funktionen des Projekts brauchen Extra-Bibliotheken, die vielleicht
# nicht installiert sind. Statt dann abzustürzen, merken wir uns nur in einer
# Ja/Nein-Variable (True/False-"Flag"), ob das Paket verfügbar ist. Die Namen
# werden IMMER definiert (None bei fehlendem Paket), damit abhängiger Code
# keine NameErrors riskiert und sich allein auf die *_AVAILABLE-Flags stützt.
# ---------------------------------------------------------------------------
try:
    import shap                     # SHAP: erklärt, WARUM das KI-Modell so entscheidet
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

try:
    from matplotlib.animation import FuncAnimation, PillowWriter  # für animierte GIFs
    ANIMATION_AVAILABLE = True
except ImportError:
    FuncAnimation = None
    PillowWriter = None
    ANIMATION_AVAILABLE = False

try:
    import yfinance as yf           # lädt kostenlose Kursdaten von Yahoo Finance
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

# Unwichtige Bibliotheks-Warnungen ausblenden, damit das Protokoll lesbar bleibt.
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# LOGGING
# Das "Logbuch" des Programms: Jede wichtige Aktion wird mit Uhrzeit und
# Wichtigkeitsstufe (INFO, WARNING, ERROR) im Terminal protokolliert.
# Alle anderen Dateien holen sich dieses Logbuch über die Variable ``log``.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,                                  # ab Stufe INFO anzeigen
    format="%(asctime)s | %(levelname)-8s | %(message)s",  # Zeilenaufbau
    datefmt="%H:%M:%S",                                  # Uhrzeit-Format
)
log = logging.getLogger("portfolio")


# ---------------------------------------------------------------------------
# TYPISIERTE KONFIGURATION
# Eine "dataclass" ist ein bequemer Datencontainer: Man listet nur die Felder
# mit ihren Standardwerten auf, Python erzeugt den Rest automatisch.
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Alle einstellbaren Parameter des Backtests an einem Ort.

    Jede Zeile unten ist ein "Stellrad" des Experiments: links der Name,
    dahinter der Standardwert. Wer das Experiment verändern will (andere
    Aktien, anderer Zeitraum, andere Kosten), dreht NUR hier — der übrige
    Code liest diese Werte und bleibt unangetastet.
    """
    # ---- Anlageuniversum & Zeitraum ---------------------------------------
    # "Ticker" = das Kürzel, unter dem eine Aktie an der Börse gehandelt wird
    # (z. B. AAPL für Apple). Die 15 Titel sind bewusst über viele Branchen
    # gestreut, damit nicht alle gleichzeitig fallen (Diversifikation).
    # Sektoren nach GICS, wie in Tabelle A1 der Arbeit.
    tickers: list = field(default_factory=lambda: [
        "AAPL",   # Apple             — Informationstechnologie
        "MSFT",   # Microsoft         — Informationstechnologie
        "NVDA",   # NVIDIA            — Informationstechnologie
        "JNJ",    # Johnson & Johnson — Gesundheitswesen
        "UNH",    # UnitedHealth      — Gesundheitswesen
        "JPM",    # JPMorgan Chase    — Finanzen
        "GS",     # Goldman Sachs     — Finanzen
        "PG",     # Procter & Gamble  — Basiskonsumgüter
        "KO",     # Coca-Cola         — Basiskonsumgüter
        "XOM",    # Exxon Mobil       — Energie
        "CAT",    # Caterpillar       — Industrie
        "HON",    # Honeywell         — Industrie
        "VZ",     # Verizon           — Kommunikationsdienste
        "PLD",    # Prologis          — Immobilien
        "LIN",    # Linde             — Grundstoffe
    ])
    # SPY = börsengehandelter Fonds auf den S&P-500-Index; dient als
    # Vergleichsmaßstab ("der Markt") für einige Kennzahlen.
    spy_ticker: str       = "SPY"
    # Daten werden schon ab 2013 geladen, obwohl der Backtest erst 2015
    # beginnt: Die technischen Indikatoren (z. B. 12-Monats-Momentum) brauchen
    # eine Anlaufzeit ("Warmup") von bis zu einem Jahr Historie.
    start_date: str       = "2013-01-01"   # Extra-Warmup für Indikatoren
    end_date: str         = "2024-12-31"
    backtest_start: str   = "2015-01-01"   # ab hier wird "investiert"
    output_dir: str       = "./output"     # Ordner für alle Ergebnisdateien
    # Kursspeicher: Yahoo liefert bei wiederholtem Abruf leicht abweichende
    # Kurse (relativ bis 1,3e-6). Der erste Download wird hier abgelegt und
    # danach wiederverwendet — sonst ist kein Lauf reproduzierbar.
    # Auf "" setzen erzwingt jedes Mal einen frischen Download.
    price_cache: str      = "./data/prices.pkl"

    # ---- Modell- & Optimierungsparameter -----------------------------------
    # Risikofreier Zins: Rendite einer als sicher geltenden Anlage (etwa
    # US-Staatsanleihen). Wird von der Portfoliorendite abgezogen, um den
    # "Mehrertrag fürs Risiko" zu messen (→ Sharpe Ratio). 0.04 = 4 % pro Jahr.
    risk_free_rate: float    = 0.04        # annualisiert (~US-10J 2024)
    # Wie viele Jahre Vergangenheit jede Strategie zum "Lernen" sieht.
    train_years: int         = 3
    # Anzahl der berechneten Punkte auf der Effizienzlinie (nur fürs Diagramm).
    n_frontier: int          = 120
    # Der Random Forest hat "Hyperparameter" (Stellschrauben wie Baumtiefe).
    # Die Suche probiert rf_n_iter zufällige Kombinationen aus …
    rf_n_iter: int           = 30          # RandomizedSearchCV-Iterationen
    # … und bewertet jede per Kreuzvalidierung mit so vielen Zeit-Abschnitten
    # (Prüfblöcken). Welches Verfahren die Abschnitte bildet, legt
    # use_purged_cv weiter unten fest; Standard ist Purged & Embargoed CV.
    rf_cv_splits: int        = 5           # Prüfblöcke der Kreuzvalidierung
    # Keine Aktie darf mehr als 20 % des Depots ausmachen (Klumpenrisiko-Schutz).
    max_weight: float        = 0.20        # Positionsobergrenze je Titel
    # Handelskosten: 0,10 % des umgeschichteten Betrags gehen bei jedem
    # Umbau des Depots "verloren" (Gebühren, Geld-Brief-Spanne).
    transaction_cost: float  = 0.0010      # 0.10 % auf Handelsumsatz
    # Der Random Forest darf sein Depot pro Monat höchstens zu 30 % umbauen —
    # sonst würden die Handelskosten seine Prognosegewinne auffressen.
    rf_turnover_limit: float = 0.30        # max. einseitiger Turnover/Monat (RF)

    # ---- Stellschrauben für die Rechenzeit (Standard 1 = jeden Monat alles) --
    rf_retune_every: int        = 1  # RF-Hyperparametersuche nur alle k Monate;
                                     # dazwischen wird der Wald mit den zuletzt
                                     # gefundenen Einstellungen nur neu trainiert.
                                     # >1 beschleunigt deutlich, ändert Ergebnisse.
    dashboard_update_every: int = 1  # Live-Dashboard nur alle k Schritte rendern;
                                     # rein kosmetisch (kein Ergebniseinfluss).

    # ---- Reproduzierbarkeit -------------------------------------------------
    # random_state=42 fixiert zwar den Zufallsgenerator, nicht aber die
    # Reihenfolge, in der parallele Threads ihre Teilergebnisse aufsummieren.
    # Gleitkommaaddition ist nicht assoziativ: Parallel gerechnet weichen zwei
    # Läufe minimal voneinander ab, und bei nahezu gleichwertigen Kandidaten
    # kann die Hyperparametersuche dann einen ANDEREN Sieger wählen.
    # True = einkernig rechnen (n_jobs=1) und damit bitgenau reproduzierbar,
    # dafür deutlich langsamer. Siehe LIMITATIONS.md, Abschnitt 12.
    deterministic: bool = True

    # ---- Kreuzvalidierung im RF-Tuning -------------------------------------
    # Purged & Embargoed CV nach López de Prado (2018, Kap. 7) statt
    # TimeSeriesSplit. Nötig, weil die Merkmale aus überlappenden Fenstern
    # stammen (Momentum über 252 Tage) und das Monatsziel in den Folgemonat
    # reicht: benachbarte Trainings- und Testabschnitte teilen dadurch
    # Information. TimeSeriesSplit verhindert Look-Ahead, aber nicht dieses
    # Leck. Auf False gesetzt (nur für den Vergleichslauf in § 4.3 der Arbeit)
    # erscheint der Random Forest um rund 0,025 Sharpe besser, als er ist.
    use_purged_cv: bool = True
    cv_embargo: float    = 0.02  # Embargo-Anteil (nur wirksam bei use_purged_cv).

    # ---- Fairness des Strategievergleichs ----------------------------------
    # Markowitz bekommt dieselbe Umschlagsschranke wie der Random Forest. Dann
    # unterscheiden sich die beiden Strategien in genau EINEM Punkt, dem
    # Renditeschätzer — der Vergleich ist "ceteris paribus" (lateinisch: "unter
    # sonst gleichen Bedingungen"). ``null`` in der JSON bzw. None = keine
    # Schranke für Markowitz.
    mvo_turnover_limit: Optional[float] = 0.30

    # Bezugspunkt der Umschlagsschranke im Optimierer. True = die mit den
    # Kursen gedrifteten Vorgängergewichte, also derselbe Bezugspunkt, gegen
    # den der ausgewiesene Umschlag gemessen wird (Formel 26 der Arbeit). Dann
    # hält der gemessene Umschlag die Schranke exakt ein. False = die
    # Zielgewichte des Vormonats; der gemessene Umschlag kann dann über der
    # Schranke liegen.
    turnover_ref_drifted: bool = True

    # Sharpe-Maximierung ist nur sinnvoll, solange überhaupt ein zulässiges
    # Portfolio eine erwartete Rendite ÜBER dem risikofreien Zins hat. Sonst
    # ist der Zähler (μ − r_f) negativ, und der Optimierer macht die
    # Volatilität im Nenner absichtlich GROSS, um den negativen Quotienten zu
    # verkleinern — er maximiert dann also das Risiko. True = in diesem Fall
    # auf das Minimum-Varianz-Portfolio ausweichen. Im Zeitraum 2015–2024
    # trat der Fall in keinem der 119 Monate ein; die Option ist Vorsorge.
    min_variance_fallback: bool = True


def _load_config() -> Config:
    """Erzeugt die Einstellungen: Standardwerte, dazu optionale Änderungen
    aus einer JSON-Datei (einem einfachen Textformat für Schlüssel-Wert-Paare).

    Ablauf in Worten:
      1. Starte mit den Standardwerten von oben (``Config()``).
      2. Schaue nach, ob eine Datei ``config.json`` existiert (oder eine per
         Umgebungsvariable PORTFOLIO_CONFIG benannte Datei).
      3. Falls ja: Übernimm daraus alle Einträge, deren Name ein echtes
         Config-Feld ist; unbekannte Namen werden nur gemeldet und ignoriert.
    So kann man Experimente variieren, ohne den Programmcode anzufassen.
    """
    cfg = Config()
    # os.environ.get(A, B) liest die Umgebungsvariable A; existiert sie nicht,
    # wird der Ersatzwert B ("config.json" im aktuellen Ordner) benutzt.
    path = os.environ.get("PORTFOLIO_CONFIG", "config.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)                     # JSON-Text → Python-Daten
            valid = {f.name for f in fields(cfg)}        # erlaubte Feldnamen
            applied, ignored = [], []
            for k, v in data.items():
                if k in valid:
                    setattr(cfg, k, v); applied.append(k)   # Wert übernehmen
                else:
                    ignored.append(k)                        # unbekannt → merken
            log.info(f"Konfiguration aus '{path}' geladen — Overrides: {applied or 'keine'}"
                     + (f" | ignoriert: {ignored}" if ignored else ""))
        except Exception as e:
            # Kaputte/unlesbare Datei bricht das Programm nicht ab — wir
            # warnen nur und arbeiten mit den Standardwerten weiter.
            log.warning(f"Konnte '{path}' nicht laden ({e}); nutze Defaults.")
    return cfg


# Die EINE, projektweit gültige Konfigurations-Instanz. Wird genau einmal
# beim ersten Import dieses Moduls erzeugt.
CFG = _load_config()

# ---------------------------------------------------------------------------
# KURZNAMEN DER EINSTELLUNGEN (maßgeblich sind allein die Werte in CFG)
# Die übrigen Dateien lesen die Einstellungen über diese GROSSBUCHSTABEN-Namen
# (Python-Konvention für "Konstanten", also Werte, die sich zur Laufzeit nicht
# ändern). Jeder Name zeigt auf das gleichnamige Feld in CFG.
# ---------------------------------------------------------------------------
TICKERS           = CFG.tickers
SPY_TICKER        = CFG.spy_ticker
START_DATE        = CFG.start_date
END_DATE          = CFG.end_date
BACKTEST_START    = CFG.backtest_start
RISK_FREE_RATE    = CFG.risk_free_rate
TRAIN_YEARS       = CFG.train_years
N_FRONTIER        = CFG.n_frontier
OUTPUT_DIR        = CFG.output_dir
RF_N_ITER         = CFG.rf_n_iter
RF_CV_SPLITS      = CFG.rf_cv_splits
MAX_WEIGHT        = CFG.max_weight
TRANSACTION_COST  = CFG.transaction_cost
RF_TURNOVER_LIMIT = CFG.rf_turnover_limit
RF_RETUNE_EVERY        = CFG.rf_retune_every
DASHBOARD_UPDATE_EVERY = CFG.dashboard_update_every
PRICE_CACHE            = CFG.price_cache
DETERMINISTIC          = CFG.deterministic
USE_PURGED_CV          = CFG.use_purged_cv
CV_EMBARGO             = CFG.cv_embargo
MVO_TURNOVER_LIMIT     = CFG.mvo_turnover_limit
TURNOVER_REF_DRIFTED   = CFG.turnover_ref_drifted
MIN_VARIANCE_FALLBACK  = CFG.min_variance_fallback

# Cross-sectional Ranking: welche Kennzahlen ("Features") monatlich in einen
# Rang UNTER den 15 Aktien umgerechnet werden (Erklärung in indicators.py).
# Strukturell festgelegt, nicht über config.json einstellbar.
RANK_COLS = ["rsi", "mom_21d", "mom_63d", "mom_252d", "alpha_spy"]

# ---------------------------------------------------------------------------
# EINHEITLICHES DIAGRAMM-DESIGN
# rcParams sind die globalen Voreinstellungen von Matplotlib. Einmal hier
# gesetzt, sehen alle Abbildungen des Projekts gleich aus (weiße Fläche,
# dezentes Gitter, Serifenschrift wie in der Seminararbeit).
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
    "axes.grid"         : True,
    "grid.alpha"        : 0.25,     # Gitterlinien nur 25 % deckend (dezent)
    "grid.linewidth"    : 0.6,
    "axes.spines.top"   : False,    # obere/rechte Rahmenlinie ausblenden
    "axes.spines.right" : False,
    "font.family"       : "serif",
    "font.size"         : 11,
    "axes.titlesize"    : 13,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 11,
    "legend.fontsize"   : 9,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
})

# ---------------------------------------------------------------------------
# ABBILDUNGSÜBERSCHRIFTEN — standardmäßig AUS
# ---------------------------------------------------------------------------
# In der Seminararbeit steht unter jeder Abbildung eine Bildunterschrift. Eine
# zusätzliche Überschrift IM Bild wäre doppelt und in wissenschaftlichen
# Arbeiten unüblich. Deshalb werden die Überschriften standardmäßig nicht
# gezeichnet. Wer sie zurückhaben will (etwa für eine Präsentation), setzt vor
# dem Lauf im Terminal:   export PLOT_TITLES=1
# Betroffen sind nur Überschriften GANZER Abbildungen. Beschriftungen einzelner
# Teilbilder (etwa der Krisenname über einem Stresstest-Panel) bleiben immer
# stehen — ohne sie wäre die Abbildung nicht lesbar.
SHOW_TITLES = os.getenv("PLOT_TITLES", "0") == "1"


def bildtitel(ziel, *args, **kwargs):
    """Überschrift einer ganzen Abbildung; zeichnet nur bei PLOT_TITLES=1.

    ``ziel`` ist entweder die ganze Abbildung (dann Gesamtüberschrift über allen
    Teilbildern) oder ein einzelnes Koordinatensystem (dann dessen Titel).
    Benutzt von plots.py und theory_plots.py."""
    if not SHOW_TITLES:
        return
    if hasattr(ziel, "suptitle"):        # ganze Abbildung ("Figure")
        ziel.suptitle(*args, **kwargs)
    else:                                # einzelnes Koordinatensystem ("Axes")
        ziel.set_title(*args, **kwargs)


# Feste Farbzuordnung je Strategie (Hex-Farbcodes), damit z. B. "Random
# Forest" in JEDER Abbildung dieselbe rote Linie hat.
COLORS = {
    "Markowitz MVO" : "#1f77b4",   # blau
    "Random Forest" : "#d62728",   # rot
    "Equal Weight"  : "#2ca02c",   # grün
    "Risk Parity"   : "#ff7f0e",   # orange
    "frontier"      : "#7f7f7f",   # grau   (Effizienzlinie)
    "cml"           : "#9467bd",   # violett (Kapitalmarktlinie)
    "mvp"           : "#e7ba52",   # gold   (Minimum-Varianz-Portfolio)
}

# Die vier verglichenen Strategien in fester Reihenfolge (für Tabellen/Plots).
STRATEGIES = ["Markowitz MVO", "Random Forest", "Equal Weight", "Risk Parity"]
