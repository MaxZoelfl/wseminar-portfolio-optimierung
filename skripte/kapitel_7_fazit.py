"""
Kapitel 7 — Eine Antwort und drei offene Fragen: jede Zahl des Kapitels, in
Textreihenfolge.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Kapitel 7 beantwortet die Forschungsfrage und greift dafür Zahlen aus den
Kapiteln 1, 5 und 6 wieder auf: die p-Werte der Tests, die Länge des
Auswertungszeitraums und den Vergleich mit DeMiguel u. a. Dieses Skript
prüft, ob sie hier genauso stehen, wie sie dort berechnet wurden.
§ 7.2 (Ausblick) enthält keine Zahlen.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_7_fazit.py             # unter einer Sekunde
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from portfolio.significance import holm_bonferroni    # noqa: E402
from kapitel_1_einleitung import DEMIGUEL_MONATE      # noqa: E402  (Zahl aus der Quelle, § 1.1)


def pruefe() -> Pruefung:
    """Alle Zahlen aus Kapitel 7, in der Reihenfolge des Textes."""
    p = Pruefung("Kapitel 7 — Eine Antwort und drei offene Fragen")
    log = protokoll()
    test = log["significance"]["sharpe_difference_tests"]
    s = strategierenditen()

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("7.1 Beantwortung der Forschungsfrage")
    p.zahl("paarweise Vergleiche", "sechs", len(test))
    alpha = inspect.signature(holm_bonferroni).parameters["alpha"].default
    p.zahl("Signifikanzniveau (Voreinstellung im Code)", "fünf", alpha * 100, "%")
    p.zahl("p Random Forest gegen Gleichgewichtung", "0,77", test["RF vs. EW"]["p_value"])
    p.zahl("p Markowitz gegen Gleichgewichtung", "0,97", test["MVO vs. EW"]["p_value"])
    p.zahl("p nach Holm, Risk Parity gegen Gleichgewichtung", "0,0216", test["Risk Parity vs. EW"]["p_holm"])
    p.zahl("aktive Verfahren (alle außer der Gleichgewichtung)", "drei", len(s.columns) - 1)
    p.quelle("DeMiguel u. a.: nötige Beobachtungsdauer in Monaten", "3000", "§ 1.1, Fn 6")
    p.quelle("DeMiguel u. a.: Titel im Portfolio", "25", "§ 1.1, Fn 6")
    monate = len(gewichte()["Risk Parity"])
    p.zahl("Auswertungszeitraum in Monaten", "119", monate)
    p.zahl("so viel kürzer als bei DeMiguel u. a.", "fünfundzwanzig", DEMIGUEL_MONATE / monate,
           ungefaehr=True, toleranz=1, notiz=f"Text: „rund fünfundzwanzigmal“; 3000 ÷ 119 = "
                                              f"{de(DEMIGUEL_MONATE / monate, 1)}")
    p.zahl("Standardwerte", "fünfzehn", log["experiment_meta"]["n_assets"])
    jahre = (s.index[-1] - s.index[0]).days / 365.25
    p.zahl("Jahre", "zehn", jahre, ungefaehr=True, toleranz=0.5, notiz=f"genau {de(jahre, 2)} Jahre")
    p.zahl("Verfahren", "vier", len(s.columns))
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
