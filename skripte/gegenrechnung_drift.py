"""
Gegenrechnung zu LIMITATIONS.md, Abschnitt 11.1: feste Gewichte innerhalb des Monats.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Der Backtest rechnet jede Tagesrendite mit den Zielgewichten des Monats, so als
würde das Depot jeden Tag kostenlos auf diese Gewichte zurückgesetzt. Ein real
gekauftes Depot "driftet" dagegen mit den Kursen: Was steigt, wird schwerer.
§ 4.4 der Arbeit nennt diese Vereinfachung. Wie viel sie ausmacht, steht in
LIMITATIONS.md — und dieses Skript rechnet es nach:

  1. die Tagesrenditen beider Varianten aus Kursen, Gewichten und Umschlag
     nachbauen (Kosten in beiden Fällen gleich, am ersten Tag des Monats),
  2. Sharpe-Quotient und CAGR beider Varianten vergleichen (portfolio/metrics.py),
  3. alle sechs Signifikanztests für die Drift-Variante wiederholen
     (portfolio/significance.py, genau wie im Backtest).

Es gehört nicht zur Nachrechnung der Arbeit (alle_kapitel.py), weil die Zahlen
nicht in der Arbeit stehen, sondern nur in LIMITATIONS.md.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/gegenrechnung_drift.py         # rund 15 Sekunden
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from portfolio.metrics import cagr                    # noqa: E402
from portfolio.significance import sharpe_difference_test, holm_bonferroni, deflated_sharpe_from_strategies   # noqa: E402


def renditen_nachgebaut(drift: bool) -> pd.DataFrame:
    """Tagesrenditen der vier Strategien, nachgebaut aus Kursen, Gewichten und Umschlag.

    drift=False  feste Zielgewichte an jedem Tag des Monats — so rechnet der Backtest
    drift=True   die Gewichte driften im Monat mit den Kursen (gekauft und gehalten),
                 so wie es die Umschlagsmessung (Formel 26) unterstellt
    Die Kosten (Umschlag × Kostensatz) gehen in beiden Fällen am ersten Tag ab."""
    r = titelrenditen(nur_auswertungszeitraum=False)
    G, U = gewichte(), umschlag()
    kostensatz = protokoll()["experiment_meta"]["transaction_cost"]
    termine = list(G["Risk Parity"].index)
    ende = strategierenditen().index[-1]
    spalten = {}
    for s in STRATEGIEN:
        teile = []
        for k, t in enumerate(termine):
            bis = termine[k + 1] if k + 1 < len(termine) else ende
            halte = r[(r.index > t) & (r.index <= bis)]          # die Handelstage dieses Monats
            w = G[s].loc[t].values
            if drift:
                wert = (w * (1 + halte).cumprod()).sum(axis=1)   # Depotwert, Start bei 1
                tag = wert / wert.shift(1, fill_value=1.0) - 1
            else:
                tag = (halte * w).sum(axis=1)                    # gewichtete Summe, wie im Backtest
            tag.iloc[0] -= U.loc[t, UMSCHLAG_SPALTE[s]] * kostensatz
            teile.append(tag)
        spalten[s] = pd.concat(teile)
    return pd.DataFrame(spalten)


def main() -> None:
    """Beide Varianten vergleichen und das Ergebnis als Tabelle ausgeben."""
    rf = protokoll()["experiment_meta"]["risk_free_rate"]
    fest, drift = renditen_nachgebaut(drift=False), renditen_nachgebaut(drift=True)
    # Kontrolle: Die Variante "fest" muss die gespeicherten Renditen wiedergeben
    # (bis auf die Rundung der Gewichte auf vier Stellen in den CSV-Dateien).
    abweichung = (fest - strategierenditen()).abs().max().max()
    print(f"Kontrolle: feste Gewichte gegen daily_returns.csv, größte Abweichung je Tag {abweichung:.1e}\n")
    print(f"{'Strategie':<16}{'Sharpe fest':>12}{'mit Drift':>11}{'Rückgang':>10}{'CAGR-Änderung':>15}")
    for s in STRATEGIEN:
        a, b = sharpe_ratio(fest[s], rf), sharpe_ratio(drift[s], rf)
        print(f"{s:<16}{de(a, 3):>12}{de(b, 3):>11}{de(a - b, 4):>10}"
              f"{de((cagr(drift[s]) - cagr(fest[s])) * 100, 2) + ' Pp':>15}")
    paare = [("Random Forest", "Markowitz MVO"), ("Random Forest", "Equal Weight"),
             ("Random Forest", "Risk Parity"), ("Markowitz MVO", "Equal Weight"),
             ("Markowitz MVO", "Risk Parity"), ("Risk Parity", "Equal Weight")]
    pw = [sharpe_difference_test(drift[a], drift[b], rf=rf)["p_value"] for a, b in paare]
    holm = holm_bonferroni(pw)
    print("\nSignifikanztests mit Drift (Ledoit/Wolf, Holm):")
    for (a, b), p_, h_, r_ in zip(paare, pw, holm["p_adjusted"], holm["reject"]):
        print(f"  {KURZ[a]:>3} gegen {KURZ[b]:<3}  p {de(p_, 4)}  Holm {de(h_, 4)}  {'signifikant' if r_ else '—'}")
    print("\nDeflated Sharpe Ratio (Formel 34):")
    for s in STRATEGIEN:
        a = deflated_sharpe_from_strategies(fest, s, rf=rf)["deflated_sr"]
        b = deflated_sharpe_from_strategies(drift, s, rf=rf)["deflated_sr"]
        print(f"  {s:<16}{de(a, 4):>8}  mit Drift {de(b, 4)}  Rückgang {de(a - b, 4)}")


if __name__ == "__main__":
    main()
