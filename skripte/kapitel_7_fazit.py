"""
Kapitel 7 — Eine Antwort und drei offene Fragen: jede eigene Zahl nachgerechnet.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Kapitel 7 fasst zusammen und wiederholt dabei einige Zahlen aus Kapitel 6
(p-Werte) und aus der Einleitung (die Beobachtungsdauer nach DeMiguel u. a.).
Dieses Skript prüft, ob die Wiederholungen mit den Ergebnissen übereinstimmen
und ob der Vergleich "rund fünfundzwanzigmal kürzer" aufgeht.
§ 7.2 (Ausblick) enthält keine eigenen Zahlen.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_7_fazit.py             # eine Sekunde
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from portfolio.significance import holm_bonferroni    # noqa: E402

DEMIGUEL_MONATE = 3000                                # DeMiguel/Garlappi/Uppal (2009), Quelle, § 1.1


def pruefe() -> Pruefung:
    p = Pruefung("Kapitel 7 — Eine Antwort und drei offene Fragen")
    log = protokoll()
    test = log["significance"]["sharpe_difference_tests"]
    s = strategierenditen()

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("7.1 Beantwortung der Forschungsfrage")
    p.zahl("paarweise Vergleiche", "6", len(test))
    import inspect
    alpha = inspect.signature(holm_bonferroni).parameters["alpha"].default
    p.zahl("Niveau α (Voreinstellung im Code)", "5", alpha * 100, "%")
    p.zahl("p Random Forest gegen 1/N", "0,77", test["RF vs. EW"]["p_value"])
    p.zahl("p Markowitz gegen 1/N", "0,97", test["MVO vs. EW"]["p_value"])
    p.zahl("p nach Holm, Risk Parity gegen 1/N", "0,0216", test["Risk Parity vs. EW"]["p_holm"])
    p.offen("beide Vorsprünge gehen auf NVIDIA zurück", "(§ 6.6)",
            "wird in kapitel_6_ergebnisse.py, § 6.6, nachgerechnet")
    p.offen("DeMiguel u. a.: Beobachtungsdauer", "3000 Monate", "Zahl aus der Quelle")
    p.offen("DeMiguel u. a.: Portfolio aus", "25 Titeln", "Zahl aus der Quelle")
    monate = len(gewichte()["Risk Parity"])
    p.zahl("Auswertungszeitraum", "119", monate, "Monate", notiz="Text: „knapp 119 Monate“")
    p.zahl("so viel kürzer", "25", DEMIGUEL_MONATE / monate, "-mal", ungefaehr=True,
           notiz="Text: „rund fünfundzwanzigmal“")
    p.zahl("Standardwerte", "15", log["experiment_meta"]["n_assets"])
    jahre = (s.index[-1] - s.index[0]).days / 365.25
    p.zahl("Jahre", "10", jahre, ungefaehr=True, toleranz=0.5, notiz="Text: „zehn Jahre“")
    p.zahl("Verfahren", "4", len(s.columns))
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
