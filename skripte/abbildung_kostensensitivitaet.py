"""
Abbildung 5 der Arbeit (output/13_kosten_sensitivitaet.png): Sharpe-Quotient je Kostensatz.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Die Handelskosten gehen im Backtest NICHT in die Optimierung ein — sie werden
in portfolio/backtest.py erst nachträglich vom ersten Tag jeder Halteperiode
abgezogen (Kosten = Umschlag × Kostensatz, § 4.4). Gewichte und Umschlag sind
also vom Kostensatz unabhängig. Deshalb lassen sich die Kennzahlen für jeden
anderen Kostensatz exakt aus den gespeicherten Tagesrenditen zurückrechnen,
ohne neuen Backtest:
  1. die im Lauf abgezogenen zehn Basispunkte wieder dazurechnen ("brutto"),
  2. den neuen Kostensatz abziehen ("netto"),
  3. die Kennzahlen mit denselben Formeln wie im Backtest berechnen
     (portfolio/metrics.py).
Das geschieht für 0 bis 100 Basispunkte in Schritten von fünf.

Als einziges Skript in diesem Ordner SCHREIBT es Dateien:
  <ordner>/kosten_sensitivitaet.csv     alle Kennzahlen je Kostensatz
  <ordner>/13_kosten_sensitivitaet.png  die Abbildung
Kapitel 6 (§ 6.4) benutzt dieselben Funktionen, ohne Dateien zu schreiben.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/abbildung_kostensensitivitaet.py [ordner]   # Standard: output/ im Projekt
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                  # nur in Dateien zeichnen, kein Fenster
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import ROOT, STRATEGIEN, UMSCHLAG_SPALTE   # noqa: E402  (Pfade, Namen)

from portfolio.config import TRANSACTION_COST, COLORS, bildtitel   # noqa: E402
from portfolio.metrics import cagr, annualized_vol, sharpe_ratio, max_drawdown   # noqa: E402

ORDNER = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
    else ROOT / "output"
SAETZE = [b / 10_000 for b in range(0, 105, 5)]        # 0 bis 100 bp in 5-bp-Schritten


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
    """Rechnet die im Lauf abgezogenen Kosten (Umschlag × zehn Basispunkte)
    wieder heraus: Tagesrenditen vor Kosten."""
    roh = renditen.copy()
    tag = kostentage(renditen.index, umschlag)
    for name, spalte in UMSCHLAG_SPALTE.items():
        for stichtag, ziel in tag.items():
            roh.loc[ziel, name] += umschlag.loc[stichtag, spalte] * TRANSACTION_COST
    return roh


def netto(roh: pd.DataFrame, umschlag: pd.DataFrame, satz: float) -> pd.DataFrame:
    """Zieht die Kosten zum Satz 'satz' neu ab (am selben Tag wie der Backtest)."""
    aus = roh.copy()
    tag = kostentage(roh.index, umschlag)
    for name, spalte in UMSCHLAG_SPALTE.items():
        for stichtag, ziel in tag.items():
            aus.loc[ziel, name] -= umschlag.loc[stichtag, spalte] * satz
    return aus


def tabelle(renditen: pd.DataFrame, umschlag: pd.DataFrame) -> pd.DataFrame:
    """Kennzahlen jeder Strategie bei jedem Kostensatz von 0 bis 100 bp,
    mit den Formeln aus portfolio/metrics.py (Formel 13/14, Tabelle B1).
    Wird auch von kapitel_6_ergebnisse.py benutzt (ohne dass Dateien entstehen)."""
    roh = brutto(renditen, umschlag)
    zeilen = []
    for satz in SAETZE:
        n = netto(roh, umschlag, satz)
        for name in STRATEGIEN:
            r = n[name].dropna()
            zeilen.append({
                "Kostensatz (bp)": round(satz * 10_000),
                "Strategie": name,
                "CAGR (%)":  cagr(r) * 100,
                "Vola (%)":  annualized_vol(r) * 100,
                "Sharpe":    sharpe_ratio(r),
                "Max. DD (%)": max_drawdown(r) * 100,
            })
    return pd.DataFrame(zeilen)


def sharpe_tabelle(tab: pd.DataFrame) -> pd.DataFrame:
    """Sharpe-Quotienten als Tabelle: Zeilen = Kostensatz in bp, Spalten = Strategien."""
    pivot = tab.pivot(index="Kostensatz (bp)", columns="Strategie", values="Sharpe")
    return pivot[STRATEGIEN]


def schnittpunkt(pivot: pd.DataFrame, name: str, gegen: str = "Equal Weight"):
    """Kostensatz (bp), bei dem 'name' unter 'gegen' fällt — zwischen den beiden
    benachbarten 5-bp-Stufen linear interpoliert. None, wenn es keinen Wechsel gibt."""
    d = pivot[name] - pivot[gegen]
    for a, b in zip(d.index[:-1], d.index[1:]):
        if d[a] > 0 >= d[b]:
            return a + (b - a) * d[a] / (d[a] - d[b])
    return None


def main() -> None:
    """Tabelle rechnen, gegen den Lauf kontrollieren, CSV und Abbildung schreiben."""
    renditen = pd.read_csv(ORDNER / "daily_returns.csv", index_col=0, parse_dates=True)
    umschlag = pd.read_csv(ORDNER / "turnover.csv",      index_col=0, parse_dates=True)
    referenz = pd.read_csv(ORDNER / "performance_metrics.csv", index_col=0)

    tab = tabelle(renditen, umschlag)

    # Kontrolle: bei 10 bp muss der Referenzlauf exakt reproduziert werden.
    probe = tab[tab["Kostensatz (bp)"] == 10].set_index("Strategie")["Sharpe"]
    abw = (probe - referenz["Sharpe Ratio"]).abs().max()
    # 'performance_metrics.csv' ist auf vier Nachkommastellen gerundet — mehr
    # Übereinstimmung als 5e-5 ist deshalb gar nicht prüfbar.
    print(f"Kontrolle bei 10 bp (dem Kostensatz des Laufs) — größte Abweichung der Sharpe-Quotienten "
          f"zu output/performance_metrics.csv: {abw:.2e}")
    print("  Das liegt innerhalb der Rundung dieser Datei auf vier Stellen: Die Umrechnung gibt den Lauf wieder.\n"
          if abw < 1e-4 else "  Das liegt außerhalb der Rundung dieser Datei: Die Umrechnung gibt den Lauf nicht wieder.\n")

    pivot = sharpe_tabelle(tab)
    print("Sharpe Ratio je Kostensatz")
    print("─" * 26)
    print(pivot.loc[[0, 10, 25, 50, 75, 100]].round(4).to_string())

    # Schnittpunkte mit der 1/N-Benchmark linear interpolieren.
    print("\nWo eine Strategie die 1/N-Benchmark unterschreitet")
    print("─" * 50)
    for name in ["Random Forest", "Markowitz MVO"]:
        x = schnittpunkt(pivot, name)
        if x is None:
            d = pivot[name] - pivot["Equal Weight"]
            print(f"  {name:<15} {'durchgehend darüber' if (d > 0).all() else 'durchgehend darunter'}")
        else:
            print(f"  {name:<15} bei rund {x:.0f} bp")

    tab.to_csv(ORDNER / "kosten_sensitivitaet.csv", index=False)

    # Die Abbildung: Farben aus config.py (COLORS), damit jede Strategie dieselbe
    # Farbe hat wie in Abbildung 4; Linienart und Markierung unterscheiden die
    # Linien zusätzlich, auch im Schwarzweißdruck.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    stil = {"Random Forest": ("-", "o"), "Markowitz MVO": ("-", "s"),
            "Equal Weight": ("--", "^"), "Risk Parity": ("-.", "D")}
    for name in pivot.columns:
        linie, marker = stil[name]
        ax.plot(pivot.index, pivot[name], color=COLORS[name], linestyle=linie,
                marker=marker, linewidth=2, markersize=6, label=name)
    ax.set_xlabel("Transaktionskosten je Umschlag (Basispunkte)")
    ax.set_ylabel("Sharpe Ratio")
    # Überschrift nur mit PLOT_TITLES=1 (siehe bildtitel in config.py).
    bildtitel(ax, "Sharpe Ratio in Abhängigkeit von den Transaktionskosten\n"
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
