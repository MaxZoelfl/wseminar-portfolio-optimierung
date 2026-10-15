"""
Kapitel 5 — Statistische Signifikanz und Robustheit: jede Zahl des Kapitels,
in Textreihenfolge.

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Kapitel 5 erklärt die Werkzeuge, mit denen die Arbeit prüft, ob ein Unterschied
zwischen zwei Strategien echt oder Zufall ist: Standardfehler, den Test von
Ledoit und Wolf, die Holm-Korrektur und die Deflated Sharpe Ratio. Die eigenen
Zahlen darin sind Eigenschaften der vier Renditereihen (etwa: wie stark sie
miteinander korrelieren) und Einstellungen des Tests (etwa: 4999 Ziehungen).
Die Testergebnisse selbst stehen in Kapitel 6 und werden dort geprüft.

§ 5.2 nennt außerdem eine Gegenrechnung: den Test so, wie Ledoit und Wolf ihn
beschreiben (Standardfehler im Bootstrap aus Blocksummen, S. 854; p-Wert als
(k + 1)/(M + 1), Gl. 9), bei Blocklängen von 7, 14 und 21 Tagen. Auch die
rechnet dieses Skript nach (rund 20 Sekunden).

Alles hier kommt aus output/daily_returns.csv und output/experiment_log.json.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/kapitel_5_signifikanz.py       # rund 25 Sekunden
"""

import inspect
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from portfolio.significance import _diff_and_se, _nw_bandwidth, holm_bonferroni, EULER_GAMMA   # noqa: E402

# Die sechs Paare in der Reihenfolge der Tabelle in § 6.3.
PAARE = [("RF", "MVO"), ("RF", "EW"), ("RF", "RP"), ("MVO", "EW"), ("MVO", "RP"), ("RP", "EW")]


def p_wert_wie_quelle(a: np.ndarray, b: np.ndarray, block: int, ziehungen: int = 4999,
                      startwert: int = 42) -> float:
    """p-Wert des Ledoit-Wolf-Tests so, wie die Quelle ihn beschreibt.

    Gleich wie im Paket (portfolio/significance.py): Differenz und HAC-
    Standardfehler der Originaldaten, zirkulärer Block-Bootstrap, zentriert auf
    die beobachtete Differenz, derselbe Startwert 42.
    Anders: Der Standardfehler jeder Ziehung kommt aus den Blocksummen der
    gezogenen Blöcke (Ledoit/Wolf 2008, S. 854), und der p-Wert ist
    (k + 1)/(M + 1) (Gl. 9).

    Formel 28 und 29 (Differenz und Gradient) stehen hier ein zweites Mal: Das
    Paket rechnet den Gradienten innerhalb von _diff_and_se und gibt ihn nicht
    heraus, und der Rechenkern soll unverändert bleiben. Dass beide Fassungen
    dasselbe rechnen, prüft pruefe() an den Originaldaten."""
    T = len(a)
    diff, se = _diff_and_se(a, b)                    # Originaldaten: wie im Paket
    beobachtet = abs(diff) / se
    rng = np.random.default_rng(startwert)
    n_bloecke, l = int(np.ceil(T / block)), T // block
    k = 0
    for _ in range(ziehungen):
        # zufällige Blockanfänge, am Reihenende zum Anfang zurück (zirkulär)
        idx = np.concatenate([np.arange(s, s + block) % T for s in rng.integers(0, T, n_bloecke)])[:T]
        d, g = differenz_und_gradient(a[idx], b[idx])
        # Ψ* aus Blocksummen: je Block die Summe der zentrierten Momente, durch √Blocklänge
        U = np.column_stack([a[idx], b[idx], a[idx] ** 2, b[idx] ** 2])
        f = (U - U.mean(axis=0))[: l * block].reshape(l, block, 4).sum(axis=1) / np.sqrt(block)
        se_stern = np.sqrt(g @ (f.T @ f / l) @ g / T)
        k += abs(d - diff) / se_stern >= beobachtet   # mindestens so extrem wie beobachtet?
    return (k + 1) / (ziehungen + 1)


def differenz_und_gradient(x: np.ndarray, y: np.ndarray):
    """Formel 28 (Differenz der Sharpe-Quotienten je Tag) und Formel 29 (ihr
    Gradient nach den vier Momenten a, b, c, d)."""
    a, b, c, d = x.mean(), y.mean(), (x ** 2).mean(), (y ** 2).mean()
    sx, sy = np.sqrt(c - a ** 2), np.sqrt(d - b ** 2)
    return a / sx - b / sy, np.array([c / sx ** 3, -d / sy ** 3, -0.5 * a / sx ** 3, 0.5 * b / sy ** 3])


def pruefe() -> Pruefung:
    """Alle Zahlen aus Kapitel 5, in der Reihenfolge des Textes."""
    p = Pruefung("Kapitel 5 — Statistische Signifikanz und Robustheit")
    s = strategierenditen()
    log = protokoll()
    rf = log["experiment_meta"]["risk_free_rate"]
    T = len(s)
    d = s - rf / HANDELSTAGE                          # Überschussrenditen je Tag

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("5.1 Warum ein Sharpe-Vergleich allein nicht ausreicht")
    p.zahl("Formel 27: Zahl der Handelstage T", "2495", T)
    # Formel 27 (Lo): SE(Ŝ_Tag) ≈ √((1 + ½Ŝ²)/T), auf Jahresbasis mal √252.
    S_tag = pd.Series({n: sharpe_exakt(n, je_tag=True) for n in STRATEGIEN})
    se_jahr = np.sqrt((1 + 0.5 * S_tag ** 2) / T) * np.sqrt(HANDELSTAGE)
    p.aussage("Standardfehler, für alle vier fast gleich", "0,318",
              all(de(v, 3) == "0,318" for v in se_jahr), ", ".join(de(v, 4) for v in se_jahr),
              notiz=", ".join(f"{KURZ[n]} {de(v, 4)}" for n, v in se_jahr.items()), woertlich=True)
    halbes_quadrat = 0.5 * S_tag ** 2                 # der Term ½·Ŝ²_Tag in Formel 27
    p.zahl("Term ½·Ŝ² auf Tagesbasis", "0,002", halbes_quadrat.max(), ungefaehr=True,
           notiz=f"Text: „rund“; {de(halbes_quadrat.min(), 4)} bis {de(halbes_quadrat.max(), 4)}")
    S_jahr = pd.Series({n: sharpe_exakt(n) for n in STRATEGIEN})
    bester = S_jahr.idxmax()
    p.zahl("höchster Sharpe-Quotient", "0,94", S_jahr[bester], notiz=bester)
    unten, oben = S_jahr[bester] - 1.96 * se_jahr[bester], S_jahr[bester] + 1.96 * se_jahr[bester]
    p.zahl("sein 95-%-Intervall, untere Grenze", "0,31", unten)
    p.zahl("sein 95-%-Intervall, obere Grenze", "1,56", oben)
    alle_drin = all(unten <= S_jahr[k] <= oben for k in S_jahr.index)
    p.aussage("Intervalle der übrigen liegen in diesem Bereich", "nahezu vollständig", alle_drin,
              "ja" if alle_drin else "nein", notiz="alle drei Sharpe-Quotienten liegen darin")
    mit_null = [n for n in STRATEGIEN if S_jahr[n] - 1.96 * se_jahr[n] <= 0]
    p.aussage("alle Intervalle schließen die Null aus", "die Null aus", not mit_null,
              ", ".join(mit_null) or "keines enthält die Null", notiz="keines enthält die Null",
              woertlich=True)
    auto = s.apply(lambda x: x.autocorr(lag=1))
    p.zahl("Autokorrelation der Renditen (Mittel der vier)", "−0,13", auto.mean(), ungefaehr=True,
           notiz="Text: „etwa“")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("5.2 Der Test von Ledoit und Wolf")
    korr = s.corr().values[np.triu_indices(4, k=1)]
    p.zahl("Korrelation der vier Renditereihen, kleinste", "0,84", korr.min())
    p.zahl("Korrelation der vier Renditereihen, größte", "0,99", korr.max())
    L = _nw_bandwidth(T)
    p.zahl("zeitversetzte Kovarianzen: über so viele Tage", "acht", L)
    # Wie stark ändern die zeitversetzten Kovarianzen den Standardfehler?
    aenderung = []
    for a, b in combinations(STRATEGIEN, 2):
        xa, xb = d[a].to_numpy(), d[b].to_numpy()
        aenderung.append(_diff_and_se(xa, xb, L=L)[1] / _diff_and_se(xa, xb, L=0)[1] - 1)
    p.zahl("größte Änderung der Standardfehler dadurch", "6,5", max(abs(v) for v in aenderung) * 100, "%")
    test = log["significance"]["sharpe_difference_tests"]
    beliebig = next(iter(test.values()))
    p.zahl("künstliche Historien (Bootstrap-Ziehungen)", "4999", beliebig["n_boot"])
    p.zahl("Blocklänge ⌈T^(1/3)⌉", "14", int(np.ceil(T ** (1 / 3))), "Handelstage",
           notiz=f"im Laufprotokoll: {beliebig['block_size']}")
    # Gegenrechnung wie in der Quelle (LIMITATIONS.md, Abschnitt 3.1)
    kleinster = min(test.values(), key=lambda t: t["p_value"])
    k = round(kleinster["p_value"] * kleinster["n_boot"])
    p.zahl("kleinster p-Wert nach der Formel der Quelle (k + 1)/(M + 1)", "0,0038",
           (k + 1) / (kleinster["n_boot"] + 1))
    p.zahl("kleinster p-Wert, wie im Code gerechnet (k/M)", "0,0036", kleinster["p_value"])
    lang = {v: k_ for k_, v in KURZ.items()}
    # Kontrolle: Die zweite Fassung von Formel 28 rechnet dasselbe wie das Paket.
    xa, xb = d["Random Forest"].to_numpy(), d["Equal Weight"].to_numpy()
    assert abs(differenz_und_gradient(xa, xb)[0] - _diff_and_se(xa, xb)[0]) < 1e-15
    for block in (7, 14, 21):
        pw = [p_wert_wie_quelle(d[lang[x]].to_numpy(), d[lang[y]].to_numpy(), block) for x, y in PAARE]
        holm = holm_bonferroni(pw)
        signifikant = [f"{x}–{y}" for (x, y), r in zip(PAARE, holm["reject"]) if r]
        im_lauf = [f"{x}–{y}" for (x, y), t in zip(PAARE, test.values()) if t["significant_holm"]]
        p.aussage(f"Gegenrechnung nach der Quelle mit Blocklänge {block}: dieselben Testentscheidungen",
                  str(block), signifikant == im_lauf, f"signifikant: {', '.join(signifikant) or 'keins'}",
                  notiz=f"signifikant nach Holm: {', '.join(signifikant) or 'keins'} "
                        f"(Holm-p RP–EW {de(holm['p_adjusted'][-1], 4)}; "
                        f"übrige Paare p ≥ {de(min(pw[:-1]), 2)})", woertlich=True)
    p.zahl("geprüfte Paare", "sechs", len(test))

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("5.3 Korrektur für multiples Testen")
    alpha = inspect.signature(holm_bonferroni).parameters["alpha"].default
    p.zahl("Vergleiche", "sechs", len(test))
    p.zahl("Niveau α je Vergleich (Voreinstellung im Code)", "5", alpha * 100, "%")
    p.zahl("Wahrscheinlichkeit mindestens eines Fehlschlusses bei 6 unabhängigen Tests",
           "26,5", (1 - (1 - alpha) ** len(test)) * 100, "%")
    p.quelle("Harvey/Liu/Zhu: veröffentlichte Faktoren", "296", "Fn 98")
    p.quelle("Harvey/Liu/Zhu: nach Holm nicht mehr signifikant", "142", "Fn 98")

    # ──────────────────────────────────────────────────────────────────
    p.abschnitt("5.4 Deflated Sharpe Ratio")
    p.zahl("Formel 33: Zahl der Kandidaten N", "4", log["significance"]["n_strategies"])
    p.zahl("Formel 33: Euler-Mascheroni-Konstante γ (im Code)", "0,5772", EULER_GAMMA)
    p.zahl("Formel 34: Zahl der Beobachtungen T", "2495", next(iter(log["significance"]["deflated_sharpe"].values()))["T"])
    p.offen("Formel 34: Wölbung der Normalverteilung", "3", "mathematische Konstante; im Code "
            "kurtosis(…, fisher=False), also Wölbung ohne Abzug von 3")
    return p


if __name__ == "__main__":
    sys.exit(1 if pruefe().abschluss() else 0)
