"""
Kapitel 1 — Eine Regel, ein Optimum, eine Frage: alle Zahlen des Kapitels.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Kapitel 1 stellt die Frage der Arbeit und den Forschungsstand vor. Es enthält
noch keine eigenen Ergebnisse, nur Zahlen aus der Literatur (DeMiguel u. a.,
Gu/Kelly/Xiu). Die lassen sich nicht aus dem Backtest berechnen, sondern sind
durch die Fußnoten belegt. Das Skript listet sie trotzdem in der Reihenfolge
der Arbeit auf, jeweils mit ihrer Fußnote — so ist die Ausgabe aller Skripte
zusammen ein vollständiges Verzeichnis der Zahlen der Arbeit. Eine einzige
Zahl ist eigene Rechnung: 3000 Monate sind 250 Jahre.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_1_einleitung.py        # unter einer Sekunde
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

# Die Beobachtungsdauer nach DeMiguel u. a. (2009), S. 1915 — eine Zahl aus der
# Quelle, die Kapitel 7 wieder aufgreift (dort wird sie mit 119 Monaten verglichen).
DEMIGUEL_MONATE = 3000


def pruefe() -> Pruefung:
    """Alle Zahlen aus Kapitel 1, in der Reihenfolge der Arbeit."""
    p = Pruefung("Kapitel 1 — Eine Regel, ein Optimum, eine Frage")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("1.1 Problemstellung")
    p.quelle("Talmud-Regel: seit dem vierten Jahrhundert überliefert", "vierten", "Fn 1")
    p.quelle("Talmud-Regel: je ein Drittel Grund, Ware, Bargeld", "ein Drittel", "Fn 1")
    p.quelle("Markowitz' Portfoliotheorie", "1952", "Fn 3")
    p.quelle("DeMiguel u. a.: geprüfte optimierende Verfahren", "vierzehn", "Fn 5")
    p.quelle("DeMiguel u. a.: Datensätze", "sieben", "Fn 5")
    p.quelle("DeMiguel u. a.: nötige Beobachtungsdauer in Monaten", "3000", "Fn 6")
    p.quelle("DeMiguel u. a.: bei so vielen Titeln", "25", "Fn 6")
    # Eigene Umrechnung der Quellenzahl: Monate durch 12.
    p.zahl("dieselbe Dauer in Jahren", "250", DEMIGUEL_MONATE / 12, notiz="3000 Monate ÷ 12")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("1.2 Forschungsstand und Abgrenzung")
    p.quelle("DeMiguel u. a.: Vorsprung des besten Verfahrens nur in einem von …", "sieben", "Fn 9")
    p.quelle("Gu/Kelly/Xiu: untersuchte Titel", "30000", "Fn 11")
    p.quelle("Gu/Kelly/Xiu: Jahre", "sechzig", "Fn 11")
    p.quelle("Gu/Kelly/Xiu: mehr als so viele Merkmale", "900", "Fn 11")
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
