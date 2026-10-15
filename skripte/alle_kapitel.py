"""
Alle Nachrechnungen auf einmal — mit einer Übersicht am Ende.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Es startet nacheinander die Prüfungen für Kapitel 2 bis 7 und den Anhang und
fasst am Schluss in einer Tabelle zusammen, wie viele Zahlen je Kapitel
stimmen und wo der Text vom Ergebnis abweicht. Das ist der schnellste Weg,
nach einer Textänderung zu sehen, ob noch alle Zahlen passen.

Es laufen nur die schnellen Stufen (zusammen rund eineinhalb Minuten). Die lange
Rechnung für Kapitel 3 (Korrelation der Bäume, rund 55 Minuten) startet man bei
Bedarf einzeln:
    python skripte/kapitel_3_random_forest.py --baumkorrelation

Kapitel 1 fehlt mit Absicht: Es enthält nur Zahlen aus der Literatur
(DeMiguel u. a., Gu/Kelly/Xiu), die sich nicht aus dem Backtest berechnen lassen.

STEHT DER EINGETRAGENE WORTLAUT NOCH IM TEXT?
Die Skripte vergleichen mit dem Wortlaut, der in ihnen eingetragen ist — die
Word-Datei lesen sie nicht. Mit --text prüft dieses Skript am Ende zusätzlich,
ob jede eingetragene Zahl in der angegebenen Fassung der Arbeit vorkommt. So
fällt auf, wenn eine Textstelle geändert wurde, das Skript aber noch die alte
Zahl prüft. Für eine .docx-Datei wird pandoc gebraucht; eine Markdown-Fassung
geht ohne.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/alle_kapitel.py
    python skripte/alle_kapitel.py --text pfad/zur/Arbeit.docx
"""

import argparse
import importlib
import sys
from pathlib import Path

# Aufrufordner merken, BEVOR _werkzeuge geladen wird: Es wechselt in den
# Projektordner, und ein relativer Pfad hinter --text soll sich auf den Ordner
# beziehen, aus dem das Skript gestartet wurde.
STARTORDNER = Path.cwd()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import wortlaut_fehlt                 # noqa: E402

SKRIPTE = [("Kapitel 2", "kapitel_2_theorie"),
           ("Kapitel 3", "kapitel_3_random_forest"),
           ("Kapitel 4", "kapitel_4_methodik"),
           ("Kapitel 5", "kapitel_5_signifikanz"),
           ("Kapitel 6", "kapitel_6_ergebnisse"),
           ("Kapitel 7", "kapitel_7_fazit"),
           ("Anhang",    "anhang")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Alle Zahlen der Arbeit nachrechnen.")
    parser.add_argument("--text", metavar="DATEI",
                        help="Fassung der Arbeit (.docx oder .md): prüfen, ob jeder "
                             "eingetragene Wortlaut darin vorkommt")
    # parse_known_args: Schalter der Kapitel-Skripte (etwa --baumkorrelation) stören nicht.
    argumente, _ = parser.parse_known_args()
    textdatei = (STARTORDNER / argumente.text).resolve() if argumente.text else None

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

    veraltet = []
    if textdatei:
        veraltet = wortlaut_fehlt(textdatei)
        print("\n" + "═" * 78)
        print(f"  WORTLAUT-ABGLEICH mit {textdatei.name}")
        print("═" * 78)
        if veraltet:
            print("  Diese eingetragenen Zahlen stehen so nicht im Text. Entweder wurde")
            print("  die Stelle geändert (dann den Eintrag im Skript nachziehen), oder")
            print("  der Text schreibt sie anders (etwa als Wort):\n")
            for titel, abschnitt, was, wortlaut in veraltet:
                print(f"    • {titel.split(' — ')[0]}, § {abschnitt}: {was} — „{wortlaut}“")
        else:
            print("  Jeder eingetragene Wortlaut steht im Text.")
    print()
    return 1 if summe[2] or veraltet else 0


if __name__ == "__main__":
    sys.exit(main())
