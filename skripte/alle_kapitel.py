"""
Alle Nachrechnungen auf einmal — mit einer Übersicht am Ende.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Es startet nacheinander die Prüfungen für Kapitel 2 bis 7 und den Anhang und
fasst am Schluss in einer Tabelle zusammen, wie viele Zahlen je Kapitel
stimmen und wo der Text vom Ergebnis abweicht. Das ist der schnellste Weg,
nach einer Textänderung zu sehen, ob noch alle Zahlen passen.

Es laufen nur die schnellen Stufen (zusammen rund eine Minute). Die lange
Rechnung für Kapitel 3 (Korrelation der Bäume, rund 55 Minuten) startet man bei
Bedarf einzeln:
    python skripte/kapitel_3_random_forest.py --baumkorrelation

Kapitel 1 fehlt mit Absicht: Es enthält nur Zahlen aus der Literatur
(DeMiguel u. a., Gu/Kelly/Xiu), die sich nicht aus dem Backtest berechnen lassen.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/alle_kapitel.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SKRIPTE = [("Kapitel 2", "kapitel_2_theorie"),
           ("Kapitel 3", "kapitel_3_random_forest"),
           ("Kapitel 4", "kapitel_4_methodik"),
           ("Kapitel 5", "kapitel_5_signifikanz"),
           ("Kapitel 6", "kapitel_6_ergebnisse"),
           ("Kapitel 7", "kapitel_7_fazit"),
           ("Anhang",    "anhang")]


def main() -> int:
    ergebnisse = []
    for kapitel, modul in SKRIPTE:
        p = importlib.import_module(modul).pruefe()
        p.abschluss()
        ergebnisse.append((kapitel, p))

    print("═" * 78)
    print("  GESAMTÜBERSICHT")
    print("═" * 78)
    print(f"  {'':<12}{'✅ stimmt':>11}{'≈ ungefähr':>12}{'❌ weicht ab':>14}{'⏭ offen':>10}")
    summe = [0, 0, 0, 0]
    for kapitel, p in ergebnisse:
        z = p.zaehler
        werte = [z["ok"], z["ungefaehr"], z["falsch"], z["offen"]]
        summe = [a + b for a, b in zip(summe, werte)]
        print(f"  {kapitel:<12}{werte[0]:>11}{werte[1]:>12}{werte[2]:>14}{werte[3]:>10}")
    print("  " + "─" * 59)
    print(f"  {'zusammen':<12}{summe[0]:>11}{summe[1]:>12}{summe[2]:>14}{summe[3]:>10}")
    alle = [a for _, p in ergebnisse for a in p.abweichungen]
    if alle:
        print("\n  Alle Abweichungen auf einen Blick:")
        for a in alle:
            print(f"    • {a}")
    print()
    return 1 if summe[2] else 0


if __name__ == "__main__":
    sys.exit(main())
