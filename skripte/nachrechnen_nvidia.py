"""
Kontrollrechnung zu § 6.2 und § 6.6 — der NVIDIA-Effekt und die Zinsabhängigkeit.

WOZU DIESE DATEI?
Das Universum der Arbeit wurde nach Ablauf des Untersuchungszeitraums gewählt.
Mit NVIDIA enthält es einen Titel, dessen außergewöhnliche Entwicklung zu diesem
Zeitpunkt schon feststand. Diese Datei misst, wie stark die Ergebnisse an diesem
einen Titel hängen, und wie empfindlich der einzige signifikante Befund der
Arbeit (Risk Parity bleibt hinter der Gleichgewichtung zurück) auf den
angesetzten risikofreien Zins reagiert.

Sie ändert NICHTS am Projekt und schreibt keine Datei. Sie liest nur die
eingefrorenen Kurse `data/prices.pkl` und die Ergebnisse in `output/`.
Es wird KEIN neuer Backtest gerechnet: Alle Zahlen folgen aus den gespeicherten
Monatsgewichten und Tagesrenditen des maßgeblichen Laufs.

AUFRUF (virtuelle Umgebung aktiviert):
    python skripte/nachrechnen_nvidia.py            # Schritte 1-3, gut eine Minute

AUFBAU:
    Schritt 1  § 6.2 — wie stark jede Strategie auf NVIDIA setzt
    Schritt 2  § 6.6 — Zerlegung der Renditeunterschiede nach Titeln
    Schritt 3  § 6.6 — ab welchem Zins der Befund „Risk Parity < 1/N" kippt

Gegenstück zu `nachrechnen_kapitel2.py` und `nachrechnen_kapitel3.py`.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Projektstamm = eine Ebene über skripte/ (Erklärung siehe signifikanz.py).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging
logging.getLogger("portfolio_v4").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

from portfolio.config import TICKERS, TRANSACTION_COST
from portfolio.significance import holm_bonferroni, sharpe_difference_test

AUSGABE = ROOT / "output"
HANDELSTAGE = 252
STRATEGIEN = {                       # Spaltenname → (Kürzel, Gewichtsdatei, Turnover-Spalte)
    "Markowitz MVO": ("MVO", "weights_markowitz.csv",   "turnover_mvo"),
    "Random Forest": ("RF",  "weights_rf.csv",          "turnover_rf"),
    "Equal Weight":  ("EW",  None,                      "turnover_ew"),
    "Risk Parity":   ("RP",  "weights_risk_parity.csv", "turnover_rp"),
}
# Dieselben sechs Paare in derselben Reihenfolge wie in portfolio/run.py.
PAARE = [("Random Forest", "Markowitz MVO"), ("Random Forest", "Equal Weight"),
         ("Random Forest", "Risk Parity"),   ("Markowitz MVO", "Equal Weight"),
         ("Markowitz MVO", "Risk Parity"),   ("Risk Parity",   "Equal Weight")]


def trennlinie(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


# ────────────────────────────────────────────────────────────────────────
# Daten laden — dieselbe Aufbereitung wie in portfolio/data.py
# ────────────────────────────────────────────────────────────────────────
kurse = pd.read_pickle(ROOT / "data" / "prices.pkl")
kurse = kurse[TICKERS + ["SPY"]].dropna(how="any")
titelrenditen = kurse.pct_change(fill_method=None).dropna()[TICKERS]

strategierenditen = pd.read_csv(AUSGABE / "daily_returns.csv", index_col=0, parse_dates=True)
umschlag = pd.read_csv(AUSGABE / "turnover.csv", index_col=0, parse_dates=True)

gewichte = {}
for name, (_, datei, _) in STRATEGIEN.items():
    if datei:
        gewichte[name] = pd.read_csv(AUSGABE / datei, index_col=0, parse_dates=True)[TICKERS]
termine = gewichte["Risk Parity"].index
gewichte["Equal Weight"] = pd.DataFrame(1 / len(TICKERS), index=termine, columns=TICKERS)


def tagesgewichte(name: str) -> pd.DataFrame:
    """Gewichte je Handelstag, gebildet wie in portfolio/backtest.py.

    Das am Monatsende d festgelegte Gewicht gilt für alle Tage NACH d bis
    einschließlich zum nächsten Monatsende. Innerhalb der Halteperiode rechnet
    der Backtest mit festen Gewichten (Tagesrendite = Σ w_i · r_i).
    """
    stuecke = []
    for i, d in enumerate(termine):
        ende = termine[i + 1] if i + 1 < len(termine) else strategierenditen.index[-1]
        tage = titelrenditen.index[(titelrenditen.index > d) & (titelrenditen.index <= ende)]
        stuecke.append(pd.DataFrame([gewichte[name].loc[d].values] * len(tage),
                                    index=tage, columns=TICKERS))
    return pd.concat(stuecke)


tagesgew = {name: tagesgewichte(name) for name in STRATEGIEN}
tage = tagesgew["Equal Weight"].index
r = titelrenditen.loc[tage]


# ────────────────────────────────────────────────────────────────────────
trennlinie("SCHRITT 1 — § 6.2: Wie stark setzt jede Strategie auf NVIDIA?")
# ────────────────────────────────────────────────────────────────────────
print(f"{'':<16}{'mittleres Gewicht':>18}{'an der 20-%-Grenze':>22}")
for name, (kurz, _, _) in STRATEGIEN.items():
    w = gewichte[name]["NVDA"]
    grenze = int((w >= 0.1999).sum())
    print(f"{name:<16}{w.mean() * 100:>16.1f} %{grenze:>12} von {len(w)} Monaten")

rp_mittel = gewichte["Risk Parity"].mean().sort_values()
print(f"\nRisk Parity, drei kleinste mittlere Gewichte: "
      + ", ".join(f"{t} {v * 100:.1f} %" for t, v in rp_mittel.head(3).items()))
vola = titelrenditen.loc[tage].std() * np.sqrt(HANDELSTAGE)
print(f"Schwankungsstärkster Titel im Auswertungszeitraum: {vola.idxmax()} "
      f"({vola.max() * 100:.1f} % p. a.)")


# ────────────────────────────────────────────────────────────────────────
trennlinie("SCHRITT 2 — § 6.6: Woher kommen die Renditeunterschiede?")
# ────────────────────────────────────────────────────────────────────────
# Für zwei Strategien A und B gilt an jedem Tag
#     r_A − r_B = Σ_i (w_A,i − w_B,i) · r_i   −   (Kosten_A − Kosten_B).
# Über alle Tage gemittelt und mit 252 annualisiert, zerfällt die Differenz der
# ARITHMETISCHEN Jahresrenditen damit exakt in einen Beitrag je Titel plus den
# Kostenunterschied. (Für die CAGR gibt es keine solche Zerlegung — nicht mischen.)
kosten = {}
for name, (_, _, spalte) in STRATEGIEN.items():
    k = pd.Series(0.0, index=tage)
    for d in termine:
        erster_tag = tage[tage > d]
        if len(erster_tag):
            k.loc[erster_tag[0]] = umschlag.loc[d, spalte] * TRANSACTION_COST
    kosten[name] = k

print("Differenz der arithmetischen Jahresrenditen in Prozentpunkten:\n")
print(f"{'Vergleich':<10}{'gesamt':>9}{'NVIDIA':>9}{'übrige 14':>11}{'Kosten':>9}"
      f"{'Kontrolle':>11}   größter übriger Beitrag")
for a, b in [("Markowitz MVO", "Equal Weight"), ("Random Forest", "Equal Weight"),
             ("Equal Weight", "Risk Parity"), ("Markowitz MVO", "Random Forest"),
             ("Markowitz MVO", "Risk Parity"), ("Random Forest", "Risk Parity")]:
    beitrag = ((tagesgew[a] - tagesgew[b]) * r).mean() * HANDELSTAGE * 100
    kost = -(kosten[a] - kosten[b]).mean() * HANDELSTAGE * 100
    gesamt = beitrag.sum() + kost
    # Kontrolle: dieselbe Differenz direkt aus den gespeicherten Strategierenditen.
    kontrolle = (strategierenditen[a] - strategierenditen[b]).loc[tage].mean() * HANDELSTAGE * 100
    rest = beitrag.drop("NVDA")
    groesster = rest.abs().idxmax()
    print(f"{STRATEGIEN[a][0] + '−' + STRATEGIEN[b][0]:<10}{gesamt:>+9.3f}{beitrag['NVDA']:>+9.3f}"
          f"{rest.sum():>+11.3f}{kost:>+9.3f}{kontrolle:>+11.3f}   {groesster} {beitrag[groesster]:+.2f}")

ew_rp = ((tagesgew["Equal Weight"] - tagesgew["Risk Parity"]) * r).mean() * HANDELSTAGE
print(f"\nAnteil von NVIDIA am Vorsprung der Gleichgewichtung vor Risk Parity "
      f"(vor Kosten): {ew_rp['NVDA'] / ew_rp.sum() * 100:.0f} %")
print("\nLesart: Ist der NVIDIA-Beitrag größer als der Gesamtvorsprung, liegt die")
print("Strategie auf den übrigen vierzehn Titeln hinter der Vergleichsstrategie.")
print("'Kontrolle' muss mit 'gesamt' übereinstimmen (Abweichung nur durch die auf")
print("vier Nachkommastellen gerundeten Gewichte in output/).")


# ────────────────────────────────────────────────────────────────────────
trennlinie("SCHRITT 3 — § 6.6: Ab welchem Zins kippt „Risk Parity < 1/N\"?")
# ────────────────────────────────────────────────────────────────────────
# Genau das Verfahren aus § 6.3: Test von Ledoit und Wolf für alle sechs Paare,
# 4999 Ziehungen, danach Holm. Nur der risikofreie Zins wird variiert.
# Warum der Zins hier wirkt: Der Sharpe-Quotient zieht ihn im Zähler ab und teilt
# durch die Volatilität. Ein hoher Zins belastet deshalb die schwankungsärmste
# Strategie am stärksten — und das ist Risk Parity.
print(f"{'Zins':>6}{'ΔSharpe RP−EW':>15}{'p':>9}{'p nach Holm':>13}   Befund")
ergebnisse = []
for zins in (0.04, 0.03, 0.02, 0.015, 0.01, 0.009, 0.008, 0.0075, 0.005, 0.0):
    tests = [sharpe_difference_test(strategierenditen[a], strategierenditen[b],
                                    rf=zins, n_boot=4999) for a, b in PAARE]
    holm = holm_bonferroni([t["p_value"] for t in tests])["p_adjusted"]
    t, h = tests[-1], holm[-1]
    ergebnisse.append((zins, h))
    print(f"{zins * 100:>5.2f}%{t['diff_annual']:>+15.4f}{t['p_value']:>9.4f}{h:>13.4f}   "
          f"{'signifikant' if h < 0.05 else 'n. s.'}"
          f"   (übrige Paare nach Holm ≥ {min(holm[:-1]):.2f})")

# Schwelle: letzter Zins, bei dem der Befund noch signifikant ist, und erster,
# bei dem er es nicht mehr ist — dazwischen linear interpoliert.
for (z_hoch, h_hoch), (z_tief, h_tief) in zip(ergebnisse, ergebnisse[1:]):
    if h_hoch < 0.05 <= h_tief:
        schwelle = z_tief + (z_hoch - z_tief) * (0.05 - h_tief) / (h_hoch - h_tief)
        print(f"\nSchwelle: Unterhalb von rund {schwelle * 100:.1f} Prozent ist der Befund "
              f"nach Holm nicht mehr signifikant.")
        break
print("Die Bootstrap-p-Werte schwanken in der dritten Nachkommastelle; die Schwelle")
print("ist deshalb als Größenordnung zu lesen, nicht auf die zweite Stelle genau.")
print("Die übrigen fünf Paare bleiben bei jedem Zins nicht signifikant.")
