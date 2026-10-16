"""
Alle Nachrechnungen auf einmal — mit einer Übersicht am Ende.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Es startet nacheinander die Skripte für Kapitel 1 bis 7 und den Anhang. Jedes
listet die Zahlen seines Kapitels in der Reihenfolge der Arbeit auf, mit dem
Wert, den die Rechnung aus Kursen, Ergebnisdateien und Programmcode ergibt,
und mit seiner Herleitung. Am Schluss steht je Kapitel, wie viele Zahlen
berechnet sind, wie viele aus der Literatur stammen und wie viele
Definitionen oder Annahmen sind.

Zusammen dauert das rund anderthalb Minuten. Die lange Rechnung für Kapitel 3
(Korrelation der Bäume, rund 55 Minuten) liest das Skript aus einer
Referenzdatei; neu rechnen lässt sie sich einzeln:
    python skripte/kapitel_3_random_forest.py --baumkorrelation

FÜR DEN VERFASSER: ABGLEICH MIT DEM WORTLAUT
    --abgleich      vergleicht jede gerechnete Zahl mit dem Wortlaut, der im
                    Skript eingetragen ist, und markiert sie (✅ ≈ ❌ ⏭)
    --text DATEI    dasselbe und zusätzlich: Steht jeder eingetragene Wortlaut
                    im passenden Abschnitt dieser Fassung der Arbeit? (.docx
                    braucht pandoc, .md geht ohne)

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/alle_kapitel.py
    python skripte/alle_kapitel.py --abgleich
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
from _werkzeuge import ABGLEICH, wortlaut_fehlt       # noqa: E402

SKRIPTE = [("Kapitel 1", "kapitel_1_einleitung"),
           ("Kapitel 2", "kapitel_2_theorie"),
           ("Kapitel 3", "kapitel_3_random_forest"),
           ("Kapitel 4", "kapitel_4_methodik"),
           ("Kapitel 5", "kapitel_5_signifikanz"),
           ("Kapitel 6", "kapitel_6_ergebnisse"),
           ("Kapitel 7", "kapitel_7_fazit"),
           ("Anhang",    "anhang")]


def main() -> int:
    """Alle Kapitel ausgeben, Übersicht drucken, auf Wunsch mit der Arbeit abgleichen.
    Rückgabe 1, wenn eine Rechnung vom eingetragenen Wortlaut abweicht oder (mit
    --text) ein Wortlaut in der Arbeit fehlt."""
    parser = argparse.ArgumentParser(description="Alle Zahlen der Arbeit nachrechnen.")
    parser.add_argument("--abgleich", action="store_true",
                        help="jede Zahl mit dem eingetragenen Wortlaut der Arbeit vergleichen")
    parser.add_argument("--text", metavar="DATEI",
                        help="Fassung der Arbeit (.docx oder .md): zusätzlich prüfen, ob jeder "
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
    if ABGLEICH:
        spalten = [("ok", "✅ stimmt"), ("ungefaehr", "≈ ungefähr"), ("falsch", "❌ weicht ab"), ("offen", "⏭ offen")]
        zaehler = lambda p: p.zaehler
    else:
        spalten = [("berechnet", "berechnet"), ("literatur", "Literatur"), ("festlegung", "Definition/Annahme")]
        zaehler = lambda p: p.herkunft
    breite = [max(len(t) + 2, 11) for _, t in spalten]
    print(f"  {'':<12}" + "".join(f"{t:>{b}}" for (_, t), b in zip(spalten, breite)))
    summe = {k: 0 for k, _ in spalten}
    for kapitel, p in ergebnisse:
        z = zaehler(p)
        for k, _ in spalten:
            summe[k] += z[k]
        print(f"  {kapitel:<12}" + "".join(f"{z[k]:>{b}}" for (k, _), b in zip(spalten, breite)))
    print("  " + "─" * (10 + sum(breite)))
    print(f"  {'zusammen':<12}" + "".join(f"{summe[k]:>{b}}" for (k, _), b in zip(spalten, breite)))
    abweichungen = sum(p.zaehler["falsch"] for _, p in ergebnisse)
    alle = [a for _, p in ergebnisse for a in p.abweichungen]
    if ABGLEICH and alle:
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
            print("  Diese eingetragenen Zahlen stehen so nicht in der Arbeit. Entweder wurde")
            print("  die Stelle geändert (dann den Eintrag im Skript nachziehen), oder")
            print("  die Arbeit schreibt sie anders (etwa als Wort):\n")
            for titel, abschnitt, was, wortlaut in veraltet:
                print(f"    • {titel.split(' — ')[0]}, § {abschnitt}: {was} — „{wortlaut}“")
        else:
            print("  Jeder eingetragene Wortlaut steht in der Arbeit.")
    print()
    return 1 if abweichungen or veraltet else 0


if __name__ == "__main__":
    sys.exit(main())
