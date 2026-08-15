"""Rechnet den Signifikanzblock aus den gespeicherten Tagesrenditen nach.

Wozu: Schnelle Kontrollrechnung, ohne den Backtest (knapp eine Stunde) zu
wiederholen. Der Test hängt nur von 'daily_returns.csv' ab, der Block-Bootstrap
ist mit seed=42 fest verdrahtet.

⚠ NICHT die maßgebliche Quelle. 'daily_returns.csv' wird auf sechs
Nachkommastellen gerundet gespeichert; die Werte hier weichen deshalb in der
vierten Stelle des p-Werts ab (gemessen: 2e-4, das entspricht genau einer von
4999 Bootstrap-Ziehungen). Die Zahlen für die Arbeit stehen in
'<ordner>/experiment_log.json' — dort rechnet der Lauf mit voller Genauigkeit.

Aufruf:  venv/bin/python signifikanz.py [ordner]     (Standard: output)
Ausgabe: nur auf den Bildschirm; es wird bewusst keine Datei geschrieben,
damit im Ergebnisordner nur EINE Signifikanzquelle liegt.
"""
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

from portfolio.config import RISK_FREE_RATE
from portfolio.significance import (deflated_sharpe_from_strategies,
                                    holm_bonferroni, sharpe_difference_test)

ORDNER = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
KURZ = {"Random Forest": "RF", "Markowitz MVO": "MVO",
        "Equal Weight": "EW", "Risk Parity": "RP"}


def main() -> None:
    r = pd.read_csv(ORDNER / "daily_returns.csv", index_col=0, parse_dates=True)
    namen = [s for s in KURZ if s in r.columns]

    # Alle sechs Paare — nicht eine Auswahl davon.
    tests, pwerte, paare = {}, [], list(combinations(namen, 2))
    for a, b in paare:
        t = sharpe_difference_test(r[a], r[b], rf=RISK_FREE_RATE, n_boot=4999)
        tests[f"{KURZ[a]} vs. {KURZ[b]}"] = t
        pwerte.append(t["p_value"])

    holm = holm_bonferroni(pwerte)
    for schl, p_adj, verwerfen in zip(tests, holm["p_adjusted"], holm["reject"]):
        tests[schl]["p_holm"] = float(p_adj)
        tests[schl]["significant_holm"] = bool(verwerfen)

    print(f"{'Vergleich':<14}{'Δ ann.':>9}{'s(Δ)':>8}{'t':>8}{'p':>9}{'p Holm':>9}   Befund")
    print("─" * 66)
    for schl, t in tests.items():
        print(f"{schl:<14}{t['diff_annual']:>9.4f}{t['se_annual']:>8.4f}"
              f"{t['statistic']:>8.3f}{t['p_value']:>9.4f}{t['p_holm']:>9.4f}"
              f"   {'SIGNIFIKANT' if t['significant_holm'] else 'n. s.'}")

    dsr = {n: deflated_sharpe_from_strategies(r, n, rf=RISK_FREE_RATE) for n in namen}
    print("\nDeflated Sharpe Ratio")
    print("─" * 21)
    for n, v in dsr.items():
        print(f"  {n:<15}{v['deflated_sr']:.4f}")

    print("\n⚠ Kontrollrechnung auf gerundeten Daten — maßgeblich ist experiment_log.json.")


if __name__ == "__main__":
    main()
