"""Sensitivität der Ergebnisse gegenüber dem Transaktionskostensatz.

Die Handelskosten gehen im Backtest NICHT in die Optimierung ein — sie werden
in 'backtest.py' erst nachträglich vom ersten Tag jeder Halteperiode abgezogen
(cost = turnover × TRANSACTION_COST). Gewichte und Umschlag sind also vom
Kostensatz unabhängig, und die Kennzahlen für einen anderen Satz lassen sich
exakt aus den gespeicherten Tagesrenditen zurückrechnen — ohne neuen Backtest.

Aufruf:  venv/bin/python kosten_sensitivitaet.py [ordner]   (Standard: output)
Ergebnis: <ordner>/kosten_sensitivitaet.csv und 13_kosten_sensitivitaet.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from portfolio.config import TRANSACTION_COST
from portfolio.metrics import cagr, annualized_vol, sharpe_ratio, max_drawdown

ORDNER = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
SAETZE = [b / 10_000 for b in range(0, 105, 5)]        # 0 bis 100 bp in 5-bp-Schritten
SPALTEN = {"Markowitz MVO": "turnover_mvo", "Random Forest": "turnover_rf",
           "Equal Weight": "turnover_ew",  "Risk Parity":  "turnover_rp"}


def kostentage(rendite_index: pd.DatetimeIndex,
               umschlag: pd.DataFrame) -> dict:
    """Ordnet jedem Rebalancing-Datum den ersten Handelstag danach zu —
    genau dort zieht der Backtest die Kosten ab."""
    zuordnung = {}
    for stichtag in umschlag.index:
        spaeter = rendite_index[rendite_index > stichtag]
        if len(spaeter):
            zuordnung[stichtag] = spaeter[0]
    return zuordnung


def brutto(renditen: pd.DataFrame, umschlag: pd.DataFrame) -> pd.DataFrame:
    """Rechnet die im Lauf abgezogenen Kosten wieder heraus."""
    roh = renditen.copy()
    tag = kostentage(renditen.index, umschlag)
    for name, spalte in SPALTEN.items():
        for stichtag, ziel in tag.items():
            roh.loc[ziel, name] += umschlag.loc[stichtag, spalte] * TRANSACTION_COST
    return roh


def netto(roh: pd.DataFrame, umschlag: pd.DataFrame, satz: float) -> pd.DataFrame:
    """Zieht die Kosten zum Satz 'satz' neu ab."""
    aus = roh.copy()
    tag = kostentage(roh.index, umschlag)
    for name, spalte in SPALTEN.items():
        for stichtag, ziel in tag.items():
            aus.loc[ziel, name] -= umschlag.loc[stichtag, spalte] * satz
    return aus


def main() -> None:
    renditen = pd.read_csv(ORDNER / "daily_returns.csv", index_col=0, parse_dates=True)
    umschlag = pd.read_csv(ORDNER / "turnover.csv",      index_col=0, parse_dates=True)
    referenz = pd.read_csv(ORDNER / "performance_metrics.csv", index_col=0)

    roh = brutto(renditen, umschlag)

    zeilen = []
    for satz in SAETZE:
        n = netto(roh, umschlag, satz)
        for name in SPALTEN:
            r = n[name].dropna()
            zeilen.append({
                "Kostensatz (bp)": round(satz * 10_000),
                "Strategie": name,
                "CAGR (%)":  cagr(r) * 100,
                "Vola (%)":  annualized_vol(r) * 100,
                "Sharpe":    sharpe_ratio(r),
                "Max. DD (%)": max_drawdown(r) * 100,
            })
    tab = pd.DataFrame(zeilen)

    # Kontrolle: bei 10 bp muss der Referenzlauf exakt reproduziert werden.
    probe = tab[tab["Kostensatz (bp)"] == 10].set_index("Strategie")["Sharpe"]
    abw = (probe - referenz["Sharpe Ratio"]).abs().max()
    # 'performance_metrics.csv' ist auf vier Nachkommastellen gerundet — mehr
    # Übereinstimmung als 5e-5 ist deshalb gar nicht prüfbar.
    print(f"Kontrolle bei 10 bp — größte Abweichung zum Lauf: {abw:.2e}")
    print("  ✓ im Rahmen der Rundung exakt reproduziert\n"
          if abw < 1e-4 else "  ⚠ Rekonstruktion weicht ab!\n")

    pivot = tab.pivot(index="Kostensatz (bp)", columns="Strategie", values="Sharpe")
    pivot = pivot[["Random Forest", "Markowitz MVO", "Equal Weight", "Risk Parity"]]
    print("Sharpe Ratio je Kostensatz")
    print("─" * 26)
    print(pivot.loc[[0, 10, 25, 50, 75, 100]].round(4).to_string())

    # Schnittpunkte mit der 1/N-Benchmark linear interpolieren.
    print("\nWo eine Strategie die 1/N-Benchmark unterschreitet")
    print("─" * 50)
    ew = pivot["Equal Weight"]
    for name in ["Random Forest", "Markowitz MVO"]:
        d = pivot[name] - ew
        wechsel = [(a, b) for a, b in zip(d.index[:-1], d.index[1:])
                   if d[a] > 0 >= d[b]]
        if not wechsel:
            lage = "durchgehend darüber" if (d > 0).all() else "durchgehend darunter"
            print(f"  {name:<15} {lage}")
            continue
        a, b = wechsel[0]
        x = a + (b - a) * d[a] / (d[a] - d[b])
        print(f"  {name:<15} bei rund {x:.0f} bp")

    tab.to_csv(ORDNER / "kosten_sensitivitaet.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    stil = {"Random Forest": ("#1f77b4", "-", "o"), "Markowitz MVO": ("#d62728", "-", "s"),
            "Equal Weight": ("#2ca02c", "--", "^"), "Risk Parity": ("#ff7f0e", "-.", "D")}
    for name in pivot.columns:
        farbe, linie, marker = stil[name]
        ax.plot(pivot.index, pivot[name], color=farbe, linestyle=linie,
                marker=marker, linewidth=2, markersize=6, label=name)
    ax.set_xlabel("Transaktionskosten je Umschlag (Basispunkte)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio in Abhängigkeit von den Transaktionskosten\n"
                 "Backtest 2015–2024, 15 US-Large-Caps", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    ziel = ORDNER / "13_kosten_sensitivitaet.png"
    fig.savefig(ziel, dpi=200)
    print(f"\nGespeichert: {ziel}")
    print(f"Gespeichert: {ORDNER / 'kosten_sensitivitaet.csv'}")


if __name__ == "__main__":
    main()
