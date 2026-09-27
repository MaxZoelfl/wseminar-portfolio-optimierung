"""
Kapitel 5 — Statistische Signifikanz und Robustheit: jede eigene Zahl nachgerechnet.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Kapitel 5 erklärt die Werkzeuge, mit denen die Arbeit prüft, ob ein Unterschied
zwischen zwei Strategien echt oder Zufall ist: Standardfehler, den Test von
Ledoit und Wolf, die Holm-Korrektur und die Deflated Sharpe Ratio. Die eigenen
Zahlen darin sind Eigenschaften der vier Renditereihen (etwa: wie stark sie
miteinander korrelieren) und Einstellungen des Tests (etwa: 4999 Ziehungen).
Die Testergebnisse selbst stehen in Kapitel 6 und werden dort geprüft.

Alles hier kommt aus output/daily_returns.csv und output/experiment_log.json.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_5_signifikanz.py       # wenige Sekunden
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from portfolio.significance import _diff_and_se, _nw_bandwidth, holm_bonferroni   # noqa: E402


def lo_faktor(rho: float, q: int = HANDELSTAGE) -> float:
    """Korrekter Hochrechnungsfaktor nach Lo (2002), Gl. (22), für Renditen mit
    Autokorrelation erster Ordnung rho (AR(1)). Ohne Autokorrelation wäre er √q."""
    summe = 2 * rho / (1 - rho) ** 2 * ((1 - rho) * q - (1 - rho ** q))
    return q / np.sqrt(q + summe)


def pruefe() -> Pruefung:
    p = Pruefung("Kapitel 5 — Statistische Signifikanz und Robustheit")
    s = strategierenditen()
    log = protokoll()
    rf = log["experiment_meta"]["risk_free_rate"]
    T = len(s)
    d = s - rf / HANDELSTAGE                          # Überschussrenditen je Tag

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("5.1 Warum ein Sharpe-Vergleich allein nicht ausreicht")
    # Formel 27 (Lo): SE(Ŝ_Tag) ≈ √((1 + Ŝ²/2)/T), auf Jahresbasis mal √252.
    S_tag = d.mean() / d.std(ddof=1)
    se_jahr = np.sqrt((1 + 0.5 * S_tag ** 2) / T) * np.sqrt(HANDELSTAGE)
    p.aussage("Standardfehler für alle vier gleich", "0,318",
              all(de(v, 3) == "0,318" for v in se_jahr), ", ".join(de(v, 4) for v in se_jahr))
    S_jahr = S_tag * np.sqrt(HANDELSTAGE)
    bester = S_jahr.idxmax()
    p.zahl("höchster Sharpe-Quotient", "0,94", S_jahr[bester], notiz=bester)
    unten, oben = S_jahr[bester] - 1.96 * se_jahr[bester], S_jahr[bester] + 1.96 * se_jahr[bester]
    p.zahl("95-%-Intervall, untere Grenze", "0,31", unten)
    p.zahl("95-%-Intervall, obere Grenze", "1,56", oben)
    p.aussage("Intervall schließt die übrigen ein", "ja",
              all(unten <= S_jahr[k] <= oben for k in S_jahr.index), "alle drei darin")
    p.aussage("Intervall schließt die Null aus", "ja", unten > 0, f"untere Grenze {de(unten, 2)}")
    auto = s.apply(lambda x: x.autocorr(lag=1))
    p.zahl("Autokorrelation (Mittel der vier)", "−0,13", auto.mean(), ungefaehr=True)
    faktor = pd.Series({k: lo_faktor(auto[k]) / np.sqrt(HANDELSTAGE) for k in auto.index})
    unterschaetzt = (1 - 1 / faktor) * 100            # so viel liegt √252-Wert unter dem korrigierten
    p.aussage("Jahreswerte unterschätzt um", "rund ein Zehntel",
              bool(((unterschaetzt > 8) & (unterschaetzt < 15)).all()),
              f"{de(unterschaetzt.min(), 1)} bis {de(unterschaetzt.max(), 1)} %")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("5.2 Der Test von Ledoit und Wolf")
    korr = s.corr().values[np.triu_indices(4, k=1)]
    p.zahl("Korrelation der Renditereihen, kleinste", "0,84", korr.min())
    p.zahl("Korrelation der Renditereihen, größte", "0,99", korr.max())
    L = _nw_bandwidth(T)
    p.zahl("zeitversetzte Kovarianzen (Bandbreite)", "8", L, "Tage")
    # Wie stark ändern die zeitversetzten Kovarianzen den Standardfehler?
    aenderung = []
    for a, b in combinations(STRATEGIEN, 2):
        xa, xb = d[a].to_numpy(), d[b].to_numpy()
        aenderung.append(_diff_and_se(xa, xb, L=L)[1] / _diff_and_se(xa, xb, L=0)[1] - 1)
    p.zahl("größte Änderung der Standardfehler", "6,5", max(abs(v) for v in aenderung) * 100, "%")
    test = log["significance"]["sharpe_difference_tests"]
    beliebig = next(iter(test.values()))
    p.zahl("Bootstrap-Ziehungen", "4999", beliebig["n_boot"])
    p.zahl("Blocklänge ⌈T^(1/3)⌉", "14", int(np.ceil(T ** (1 / 3))), "Handelstage")
    p.aussage("Blocklänge im Laufprotokoll", "14", beliebig["block_size"] == 14, str(beliebig["block_size"]))
    p.zahl("geprüfte Paare", "6", len(test))

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("5.3 Korrektur für multiples Testen")
    import inspect
    alpha = inspect.signature(holm_bonferroni).parameters["alpha"].default
    p.zahl("Niveau α (Voreinstellung im Code)", "5", alpha * 100, "%")
    p.zahl("Fehlerwahrscheinlichkeit bei 6 unabhängigen Tests", "26,5", (1 - 0.95 ** 6) * 100, "%")
    p.offen("Harvey/Liu/Zhu: 142 von 296 Faktoren", "296 / 142", "Zahlen aus der Quelle")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("5.4 Deflated Sharpe Ratio")
    p.zahl("Zahl der Kandidaten N", "4", log["significance"]["n_strategies"])
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
