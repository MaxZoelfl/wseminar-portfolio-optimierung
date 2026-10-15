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

Zur Gegenprobe rechnet das Skript den Test außerdem so, wie Ledoit und Wolf
(2008) ihn beschreiben: Standardfehler im Bootstrap aus Blocksummen (S. 854),
p-Wert als (k + 1)/(M + 1) (Gl. 9), bei Blocklängen von 7, 14 und 21 Tagen.
Warum das Paket einfacher rechnet: LIMITATIONS.md § 13.3.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_5_signifikanz.py       # rund 20 Sekunden
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from portfolio.significance import _diff_and_se, _nw_bandwidth, holm_bonferroni   # noqa: E402

PAARE = [("RF", "MVO"), ("RF", "EW"), ("RF", "RP"), ("MVO", "EW"), ("MVO", "RP"), ("RP", "EW")]


def p_wert_wie_quelle(a: np.ndarray, b: np.ndarray, block: int, ziehungen: int = 4999,
                      startwert: int = 42) -> float:
    """p-Wert des Ledoit-Wolf-Tests so, wie die Quelle ihn beschreibt.

    Gleich wie im Paket: Differenz und HAC-Standardfehler der Originaldaten,
    zirkulärer Block-Bootstrap, zentriert auf die beobachtete Differenz.
    Anders: Der Standardfehler jeder Ziehung kommt aus den Blocksummen der
    gezogenen Blöcke (Ledoit/Wolf 2008, S. 854), und der p-Wert ist
    (k + 1)/(M + 1) (Gl. 9)."""
    T = len(a)
    diff, se = _diff_and_se(a, b)
    beobachtet = abs(diff) / se
    rng = np.random.default_rng(startwert)
    n_bloecke, l = int(np.ceil(T / block)), T // block
    k = 0
    for _ in range(ziehungen):
        idx = np.concatenate([np.arange(s, s + block) % T for s in rng.integers(0, T, n_bloecke)])[:T]
        x, y = a[idx], b[idx]
        # Differenz und Gradient (Formel 28 und 29) für die gezogene Reihe
        mx, my, qx, qy = x.mean(), y.mean(), (x ** 2).mean(), (y ** 2).mean()
        sx, sy = np.sqrt(qx - mx ** 2), np.sqrt(qy - my ** 2)
        d = mx / sx - my / sy
        g = np.array([qx / sx ** 3, -qy / sy ** 3, -0.5 * mx / sx ** 3, 0.5 * my / sy ** 3])
        # Ψ* aus Blocksummen: je Block die Summe der zentrierten Momente, durch √Blocklänge
        U = np.column_stack([x, y, x ** 2, y ** 2])
        f = (U - U.mean(axis=0))[: l * block].reshape(l, block, 4).sum(axis=1) / np.sqrt(block)
        se_stern = np.sqrt(g @ (f.T @ f / l) @ g / T)
        k += abs(d - diff) / se_stern >= beobachtet
    return (k + 1) / (ziehungen + 1)


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
    halbes_quadrat = 0.5 * S_tag ** 2                 # der Term ½·Ŝ²_Tag in Formel 27
    p.zahl("Term ½·Ŝ² auf Tagesbasis (größter)", "0,002", halbes_quadrat.max(), ungefaehr=True,
           notiz=f"{de(halbes_quadrat.min(), 4)} bis {de(halbes_quadrat.max(), 4)} — Text: „rund“")
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
    # Der Code rechnet den p-Wert als Anteil k/M; Ledoit/Wolf (2008), Gl. (9),
    # als (k + 1)/(M + 1). Wie viel das ausmacht, am kleinsten p-Wert:
    kleinster = min(test.values(), key=lambda t: t["p_value"])
    k = round(kleinster["p_value"] * kleinster["n_boot"])
    p.zahl("kleinster p-Wert, wie im Code (k/M)", "0,0036", kleinster["p_value"])
    p.zahl("derselbe nach Ledoit/Wolf, Gl. (9)", "0,0038", (k + 1) / (kleinster["n_boot"] + 1))

    # ──────────────────────────────────────────────────────────────────
    # Gegenprobe: derselbe Test so, wie die Quelle ihn beschreibt (LIMITATIONS.md § 13.3).
    lang = {v: k for k, v in KURZ.items()}
    for block in (7, 14, 21):
        pw = [p_wert_wie_quelle(d[lang[x]].to_numpy(), d[lang[y]].to_numpy(), block) for x, y in PAARE]
        holm = holm_bonferroni(pw)
        signifikant = [f"{x}–{y}" for (x, y), r in zip(PAARE, holm["reject"]) if r]
        p.aussage(f"wie die Quelle, Blocklänge {block}: nur RP–EW signifikant", "ja",
                  signifikant == ["RP–EW"],
                  f"Holm RP–EW {de(holm['p_adjusted'][-1], 4)}, übrige p ≥ {de(min(pw[:-1]), 2)}")

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
