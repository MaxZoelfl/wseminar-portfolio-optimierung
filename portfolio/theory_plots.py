"""
Die vier Theorie-Abbildungen zu Kapitel 2 der Seminararbeit.

FÜR EINSTEIGER — WAS MACHT DIESE DATEI?
Die übrigen Abbildungen des Projekts (in ``plots.py``) zeigen ERGEBNISSE:
wie sich die vier Strategien geschlagen haben. Die Bilder hier zeigen
dagegen keine Strategie, sondern EIGENSCHAFTEN DER KURSDATEN SELBST — also
Dinge, die schon feststehen, bevor irgendeine Strategie gerechnet wird.

Sie illustrieren die Theorie aus Kapitel 2 der Arbeit an genau den Kursen,
mit denen später auch der Backtest läuft:

  14  Zwei-Anlagen-Fall     → § 2.2, Abbildung 1 der Arbeit
      Alle Mischungen aus zwei Aktien, gezeichnet für fünf verschiedene
      Korrelationen. Zeigt: Je kleiner die Korrelation, desto weiter
      wölbt sich die Kurve nach links — und Wölbung nach links IST der
      Diversifikationsgewinn.

  15  Diversifikationsgrenze → § 2.2, Anhangsabbildung
      Wie stark sinkt das Risiko, wenn man immer mehr Aktien gleich
      gewichtet beimischt? Antwort: schnell am Anfang, dann kaum noch —
      es bleibt die mittlere Kovarianz übrig, die sich nicht wegstreuen
      lässt (Markowitz 1959, S. 111).

  16  Schätzunsicherheit    → § 2.5, Abbildung 3 der Arbeit
      Für jede Aktie die geschätzte Jahresrendite samt 95-%-Konfidenz-
      intervall. Zeigt, wie ungenau Erwartungswerte selbst aus zehn
      Jahren Tagesdaten geschätzt sind — das Kernproblem der Arbeit.

  17  Effizienzrand         → § 2.3, Abbildung 2 der Arbeit
      Die Hyperbel der geschlossenen Lösung mit dem Minimum-Varianz-
      Portfolio als linkestem Punkt. Bewusst OHNE die Beschränkungen der
      Arbeit: § 2.3 behandelt das klassische Problem, Leerverkaufsverbot
      und Obergrenze folgen erst in § 2.6.

WARUM DIESE DATEI ÜBERHAUPT ENTSTAND
Die drei Bilder lagen bis zum 17.08.2026 nur als fertige PNG-Dateien im
Ordner ``Abbildungen/`` vor, ohne erzeugenden Code. Sie waren damit weder
reproduzierbar noch korrigierbar — ein Widerspruch zu Anhang C der Arbeit,
der Reproduzierbarkeit für alles behauptet. Jetzt entstehen sie bei jedem
Backtestlauf neu, aus derselben eingefrorenen Kursdatei.

WICHTIG — WELCHE DATEN HINEINGEHEN
Alle drei Funktionen erwarten ``asset_returns`` bereits eingeschränkt auf den
AUSWERTUNGSZEITRAUM des Backtests (erster bis letzter Tag von ``returns_df``,
also 02.02.2015 – 30.12.2024). Nicht den vollen Datenbestand ab 2013 nehmen —
der enthält die zwei Vorlaufjahre, die nur zum Anlernen der Indikatoren dienen,
und lieferte andere Zahlen als die im Text genannten.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize   # numerischer Löser für den zulässigen Rand
from .config import *

# Ein Börsenjahr hat rund 252 Handelstage. Damit wird von Tages- auf
# Jahreswerte umgerechnet ("annualisiert").
TRADING_DAYS = 252

# Fester Startwert des Zufallsgenerators für die Stichprobe in Abbildung 15.
# Ohne ihn sähe das Bild bei jedem Lauf anders aus — und die Arbeit behauptet
# in Anhang C bitgenaue Reproduzierbarkeit.
THEORY_SEED = 42


def _annualised_moments(asset_returns: pd.DataFrame):
    """Hilfsfunktion: annualisierte Renditen, Volatilitäten, Kovarianzmatrix.

    Wird von allen drei Abbildungen gebraucht, deshalb einmal zentral.
    Rückgabe (jeweils auf Jahresbasis):
      mu    – erwartete Rendite je Titel     (Mittelwert  × 252)
      sigma – Volatilität je Titel           (Standardabw. × √252)
      cov   – Kovarianzmatrix                (Kovarianzen × 252)
      corr  – Korrelationsmatrix             (dimensionslos, unverändert)
    """
    mu    = asset_returns.mean() * TRADING_DAYS
    cov   = asset_returns.cov() * TRADING_DAYS
    sigma = pd.Series(np.sqrt(np.diag(cov)), index=cov.index)
    corr  = asset_returns.corr()
    return mu, sigma, cov, corr


def plot_two_asset_diversification(asset_returns: pd.DataFrame,
                                   output_path: str,
                                   asset_a: str = "KO",
                                   asset_b: str = "CAT") -> None:
    """Abbildung 14 → Abbildung 1 der Arbeit (§ 2.2): Zwei-Anlagen-Fall.

    Gezeichnet wird die Menge aller Mischungen aus zwei Aktien im
    Risiko-Rendite-Diagramm, und zwar fünfmal: für die tatsächlich gemessene
    Korrelation und für vier hypothetische Werte von −1 bis +1.

    Die zugrundeliegende Formel (§ 2.2 der Arbeit) für einen Anteil w in B:
        sigma_P^2(w) = (1-w)^2 sigma_A^2 + w^2 sigma_B^2
                       + 2 w (1-w) rho sigma_A sigma_B
    Die Rendite ist dagegen schlicht linear: mu_P(w) = (1-w) mu_A + w mu_B.
    Genau dieser Unterschied — Rendite gerade, Risiko gekrümmt — erzeugt die
    nach links gewölbten Kurven.

    ``asset_a`` ist die Basis, ``asset_b`` die Beimischung.
    """
    log.info(f"Theorieplot 14: Zwei-Anlagen-Fall ({asset_a}/{asset_b}) …")
    mu, sigma, _, corr = _annualised_moments(asset_returns)

    for t in (asset_a, asset_b):
        if t not in mu.index:
            log.warning(f"  Titel {t} nicht in den Daten – Abbildung übersprungen.")
            return

    sA, sB = sigma[asset_a], sigma[asset_b]
    mA, mB = mu[asset_a], mu[asset_b]
    rho_emp = corr.loc[asset_a, asset_b]

    # w läuft von 0 (nur A) bis 1 (nur B); 400 Stützstellen ergeben glatte Kurven.
    w = np.linspace(0.0, 1.0, 400)
    mu_p = (1 - w) * mA + w * mB          # linear, unabhängig von rho

    fig, ax = plt.subplots(figsize=(11, 7))

    # Vier hypothetische Korrelationen plus die gemessene. Der empirische Wert
    # wird kräftig rot gezeichnet, die übrigen dünn und gestrichelt — der Leser
    # soll auf einen Blick sehen, was real ist und was nicht.
    kurven = [
        (-1.00, "#999999", ":",  1.4, f"ρ = −1,00"),
        (-0.50, "#1f77b4", "--", 1.6, f"ρ = −0,50"),
        ( 0.00, "#2ca02c", "-.", 1.6, f"ρ =  0,00"),
        # Dezimalkomma statt -punkt, damit die Legende zum deutschen Text passt.
        (rho_emp, "#d62728", "-", 2.8,
         f"ρ = {rho_emp:+.2f}".replace(".", ",") + "  (empirisch)"),
        ( 1.00, "#333333", ":",  1.4, f"ρ = +1,00"),
    ]
    for rho, farbe, stil, breite, beschriftung in kurven:
        var_p = ((1 - w) ** 2 * sA ** 2 + w ** 2 * sB ** 2
                 + 2 * w * (1 - w) * rho * sA * sB)
        ax.plot(np.sqrt(var_p) * 100, mu_p * 100,
                color=farbe, linestyle=stil, linewidth=breite, label=beschriftung)

    # Die beiden reinen Anlagen als Punkte markieren und beschriften.
    for tick, s, m in ((asset_a, sA, mA), (asset_b, sB, mB)):
        ax.scatter(s * 100, m * 100, s=90, color="black", zorder=5)
        ax.annotate(f"  {tick}", (s * 100, m * 100),
                    fontsize=13, va="center", zorder=5)

    ax.set_title("Diversifikationseffekt im Zwei-Anlagen-Fall\n"
                 f"{asset_a} und {asset_b}, Tagesdaten "
                 f"{asset_returns.index[0]:%d.%m.%Y} – {asset_returns.index[-1]:%d.%m.%Y}",
                 fontweight="bold")
    ax.set_xlabel("Volatilität σ (% p. a.)")
    ax.set_ylabel("Erwartete Rendite μ (% p. a.)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_diversification_limit(asset_returns: pd.DataFrame,
                               output_path: str,
                               n_draws: int = 400) -> None:
    """Abbildung 15 → Anhangsabbildung zu § 2.2: die Diversifikationsgrenze.

    Zwei Dinge in einem Bild:

    1. Die MODELLKURVE nach Markowitz (1959, S. 111). Verteilt man das Geld
       gleichmäßig auf N Titel, so zerfällt die Portfoliovarianz in
           Var = (mittlere Varianz)/N + (N-1)/N · (mittlere Kovarianz).
       Der erste Summand geht gegen null, der zweite bleibt übrig. Die Wurzel
       aus der mittleren Kovarianz ist deshalb die Grenze, unter die man durch
       bloßes Hinzunehmen weiterer Titel nicht kommt.

    2. Die SIMULATION als Gegenprobe: Für jede Titelzahl N von 1 bis 15 werden
       ``n_draws`` zufällige Auswahlen aus dem Universum gezogen, jeweils gleich
       gewichtet, und die tatsächliche Volatilität gemittelt. Liegen die Punkte
       auf der Kurve, ist die Formel bestätigt.

    ACHTUNG: Die Grenze gilt für GLEICHGEWICHTETE Portfolios bei wachsender
    Titelzahl. Sie ist KEINE untere Schranke für ein optimiertes Portfolio aus
    fester Titelzahl — Umgewichten kommt darunter. Der Titel des Bildes ist
    entsprechend vorsichtig formuliert.
    """
    log.info("Theorieplot 15: Diversifikationsgrenze …")
    _, sigma, cov, _ = _annualised_moments(asset_returns)

    C = cov.values
    n_assets = C.shape[0]
    if n_assets < 2:
        log.warning("  Weniger als zwei Titel – Abbildung übersprungen.")
        return

    # Mittlere Varianz = Durchschnitt der Diagonale.
    # Mittlere Kovarianz = Durchschnitt aller Einträge NEBEN der Diagonale.
    nebendiagonale = ~np.eye(n_assets, dtype=bool)
    var_quer = np.diag(C).mean()
    cov_quer = C[nebendiagonale].mean()
    grenze   = np.sqrt(cov_quer)

    # Modellkurve bis N = 100 (über das eigene Universum hinaus, damit der
    # Grenzwert sichtbar wird).
    n_modell  = np.arange(1, 101)
    var_modell = var_quer / n_modell + (n_modell - 1) / n_modell * cov_quer

    # Simulation: eigener Zufallsgenerator mit festem Startwert, damit das Bild
    # bei jedem Lauf identisch ausfällt und den Hauptlauf nicht beeinflusst.
    rng = np.random.default_rng(THEORY_SEED)
    n_sim, vol_sim = [], []
    for n in range(1, n_assets + 1):
        vols = []
        for _ in range(n_draws):
            auswahl = rng.choice(n_assets, size=n, replace=False)
            w = np.ones(n) / n
            teilmatrix = C[np.ix_(auswahl, auswahl)]
            vols.append(np.sqrt(w @ teilmatrix @ w))
        n_sim.append(n)
        vol_sim.append(np.mean(vols))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(n_modell, np.sqrt(var_modell) * 100, color="#1f77b4", linewidth=2.5,
            label=r"Modell: $\sqrt{\bar{V}/N + \frac{N-1}{N}\bar{C}}$")
    ax.scatter(n_sim, np.array(vol_sim) * 100, color="#d62728", s=42, zorder=5,
               label=f"Simulation (Ø über {n_draws} Zufallsauswahlen)")
    ax.axhline(grenze * 100, color="#2ca02c", linestyle="--", linewidth=2,
               label=rf"Diversifikationsgrenze $\sqrt{{\bar{{C}}}}$ = {grenze*100:.1f} %")

    ax.set_title("Diversifikation senkt das Risiko – aber nur bis zu einer Grenze",
                 fontweight="bold")
    ax.set_xlabel("Anzahl gleichgewichteter Aktien N")
    ax.set_ylabel("Portfoliovolatilität (% p. a.)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    log.info(f"  mittlere Kovarianz C̄ = {cov_quer:.6f} → √C̄ = {grenze*100:.3f} % | "
             f"1/N über alle {n_assets} Titel = "
             f"{np.sqrt(np.ones(n_assets)/n_assets @ C @ (np.ones(n_assets)/n_assets))*100:.3f} %")


def plot_estimation_uncertainty(asset_returns: pd.DataFrame,
                                output_path: str) -> None:
    """Abbildung 16 → Abbildung 3 der Arbeit (§ 2.5): Schätzunsicherheit von μ.

    Für jede Aktie wird die geschätzte Jahresrendite als Punkt gezeichnet, dazu
    als waagerechter Balken das 95-%-Konfidenzintervall des Schätzers:

        Standardfehler(mu_jährlich) = sigma_jährlich / √h        (h = Jahre)
        Intervall                   = mu ± 1,96 · Standardfehler

    Das ist genau Mertons Ergebnis (1980, Anhang A, Gl. A.3): Die Genauigkeit
    des Erwartungswert-Schätzers hängt allein an der KALENDARISCHEN LÄNGE h des
    Beobachtungszeitraums — nicht daran, wie fein man abtastet. Aus Tagesdaten
    statt Monatsdaten gewinnt man für mu deshalb nichts.

    Die Botschaft des Bildes: Die Intervalle sind dutzende Prozentpunkte breit
    und überlappen einander fast alle. Wer auf solchen Schätzungen optimiert,
    optimiert zu einem großen Teil auf Rauschen (§ 2.5, Michaud 1989).
    """
    log.info("Theorieplot 16: Schätzunsicherheit der Erwartungswerte …")
    mu, sigma, _, _ = _annualised_moments(asset_returns)

    # Länge des Beobachtungszeitraums in Jahren.
    jahre = len(asset_returns) / TRADING_DAYS
    standardfehler = sigma / np.sqrt(jahre)
    halbe_breite   = 1.96 * standardfehler

    # Aufsteigend sortieren, damit der höchste Wert oben landet.
    reihenfolge = mu.sort_values().index
    y = np.arange(len(reihenfolge))

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.errorbar(mu[reihenfolge] * 100, y,
                xerr=halbe_breite[reihenfolge] * 100,
                fmt="o", markersize=9, color="#1f77b4",
                ecolor="#888888", elinewidth=1.8, capsize=4)
    ax.axvline(0, color="black", linewidth=1.2)    # Nulllinie zum Vergleich

    ax.set_yticks(y)
    ax.set_yticklabels(reihenfolge)
    ax.set_title("Erwartungswerte lassen sich kaum schätzen\n"
                 "Punktschätzer und Unsicherheit, "
                 f"{jahre:.0f} Jahre Tagesdaten "
                 f"({asset_returns.index[0]:%Y}–{asset_returns.index[-1]:%Y})",
                 fontweight="bold")
    ax.set_xlabel("Geschätzte Jahresrendite mit 95-%-Konfidenzintervall (%)")
    ax.grid(alpha=0.3, axis="x")

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Zwei Zahlen fürs Protokoll, die im Text von § 2.5 gebraucht werden.
    breiteste = (2 * halbe_breite).idxmax()
    log.info(f"  Beobachtungsdauer h = {jahre:.2f} Jahre | "
             f"breitestes Intervall: {breiteste} "
             f"± {halbe_breite[breiteste]*100:.1f} Prozentpunkte")


def plot_efficient_frontier_theory(asset_returns: pd.DataFrame,
                                   output_path: str,
                                   show_constrained: bool = False,
                                   max_weight: float = 0.20,
                                   n_points: int = 90) -> None:
    """Abbildung 17 → Abbildung 2 der Arbeit (§ 2.3): der Effizienzrand.

    ACHTUNG — NICHT verwechseln mit ``plot_efficient_frontier`` in plots.py.
    Jene Abbildung (04) gehört zum ERGEBNISTEIL: Sie zeichnet die Schätzungen
    des letzten Backtest-Monats und trägt die vier Strategien ein. Diese hier
    ist eine reine THEORIEABBILDUNG zum allgemeinen Markowitz-Problem: derselbe
    Auswertungszeitraum wie der übrige Kapitel 2, keine Strategien, keine
    Kennzahlenlegende.

    GEZEIGT WIRD DIE GESCHLOSSENE LÖSUNG, also der Rand unter der einzigen
    Nebenbedingung, dass die Gewichte sich zu eins summieren:

        sigma^2(mu) = (a mu^2 - 2 b mu + c) / d
        mit a = 1'S^-1 1,  b = 1'S^-1 mu,  c = mu'S^-1 mu,  d = ac - b^2

    Das ist eine Hyperbel. Ihr linkester Punkt ist das Minimum-Varianz-Portfolio
    mit sigma = 1/sqrt(a) und mu = b/a. Oberhalb davon liegen die EFFIZIENTEN
    Portfolios (kräftig gezeichnet), unterhalb die ineffizienten (dünn): Zu jedem
    von ihnen gibt es ein Portfolio mit gleichem Risiko und höherer Rendite.

    WARUM OHNE BESCHRÄNKUNGEN?
    § 2.3 behandelt das klassische Problem. Leerverkaufsverbot und Obergrenze
    je Titel werden erst in § 2.6 eingeführt — dort als Mittel gegen den
    Schätzfehler, nicht als technische Randbedingung. Eine Abbildung, die sie
    vorwegnähme, erklärte in Kapitel 2 etwas, das der Text erst drei Abschnitte
    später begründet. Wer den Vergleich dennoch braucht (etwa für § 2.6 oder den
    Anhang), setzt ``show_constrained=True``: dann kommt der zulässige Rand als
    zweite Kurve hinzu. Der Unterschied liegt weniger in der Lage der Kurven —
    sie liegen fast aufeinander — als in ihrer Länge, weil die Obergrenze den
    erreichbaren Renditebereich beschneidet.

    KEINE KAPITALMARKTLINIE. Das Tangentialportfolio dieser Daten liegt bei
    rund 53 % Volatilität und verlangt Leerverkäufe bis −79 %; es läge weit
    ausserhalb jedes brauchbaren Ausschnitts. Diese Zahlen gehören als Satz in
    den Text — dort sagen sie mehr als eine Linie, die aus dem Bild läuft.
    """
    log.info("Theorieplot 17: Effizienzrand …")
    mu_v, sigma, cov, _ = _annualised_moments(asset_returns)
    mu_a, S = mu_v.values, cov.values
    n = len(mu_a)
    one = np.ones(n)

    # ---- Geschlossene Lösung: die vier Skalare und das MVP --------------
    S_inv = np.linalg.inv(S)
    a = one @ S_inv @ one
    b = one @ S_inv @ mu_a
    c = mu_a @ S_inv @ mu_a
    d = a * c - b * b
    mu_mvp, sig_mvp = b / a, np.sqrt(1.0 / a)

    fig, ax = plt.subplots(figsize=(11, 7))

    # ---- Ausschnitt -----------------------------------------------------
    # An der Hyperbel ausgerichtet: vom MVP aus so weit nach oben, dass der
    # Bogen gut sichtbar ist. Die volatilsten Titel liegen bewusst ausserhalb —
    # sie würden die Kurve sonst ins linke Fünftel drängen.
    y_max = 36.0
    y_min = max(0.0, mu_mvp * 100 - 7.0)
    mu_g = np.linspace(y_min / 100, y_max / 100, 400)
    sig_g = np.sqrt((a * mu_g ** 2 - 2 * b * mu_g + c) / d)
    x_max = sig_g.max() * 100 * 1.30

    # ---- Der Rand: effizienter Ast kräftig, ineffizienter dünn ----------
    ob = mu_g >= mu_mvp
    ax.plot(sig_g[ob] * 100, mu_g[ob] * 100, color="#1f77b4", linewidth=3.2,
            label="Effizienzrand (effizienter Ast)", zorder=4)
    ax.plot(sig_g[~ob] * 100, mu_g[~ob] * 100, color="#9ec5e8", linewidth=1.5,
            linestyle="--", label="ineffizienter Ast", zorder=4)

    # ---- Optional: der Rand unter den Beschränkungen der Arbeit ---------
    if show_constrained:
        k = int(np.ceil(1.0 / max_weight))
        mu_sort = np.sort(mu_a)
        bounds = [(0.0, max_weight)] * n
        eq_sum = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        zs, ss = [], []
        for ziel in np.linspace(mu_sort[:k].mean(), mu_sort[-k:].mean(), n_points):
            cons = (eq_sum, {"type": "eq", "fun": lambda w, z=ziel: w @ mu_a - z})
            res = minimize(lambda w: w @ S @ w, one / n, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"ftol": 1e-12, "maxiter": 800})
            if res.success:
                ss.append(np.sqrt(res.fun)); zs.append(ziel)
        ax.plot(np.array(ss) * 100, np.array(zs) * 100, color="#d62728",
                linewidth=2.2, zorder=3,
                label=f"zulässig: long only, höchstens {max_weight:.0%} je Titel")

    # ---- Einzeltitel ----------------------------------------------------
    ax.scatter(sigma.values * 100, mu_a * 100, s=34, color="#b0c4de",
               edgecolor="#5b7ea6", linewidth=0.6, zorder=5, label="Einzeltitel")
    draussen = []
    for t, s_i, m_i in zip(mu_v.index, sigma.values, mu_a):
        if s_i * 100 > x_max or m_i * 100 > y_max or m_i * 100 < y_min:
            draussen.append(t)
        elif s_i * 100 > 0.86 * x_max:
            ax.annotate(f"{t} ", (s_i * 100, m_i * 100), fontsize=8,
                        color="#5b7ea6", va="center", ha="right", zorder=5)
        else:
            ax.annotate(f" {t}", (s_i * 100, m_i * 100), fontsize=8,
                        color="#5b7ea6", va="center", zorder=5)

    # ---- Das Minimum-Varianz-Portfolio ----------------------------------
    ax.scatter([sig_mvp * 100], [mu_mvp * 100], marker="D", s=80, color="#e7ba52",
               edgecolor="black", linewidth=0.9, zorder=6,
               label=f"Minimum-Varianz-Portfolio ({sig_mvp*100:.2f} %)")
    ax.annotate(rf"$\sigma = 1/\sqrt{{a}}$", (sig_mvp * 100, mu_mvp * 100),
                xytext=(-12, -22), textcoords="offset points", fontsize=9,
                color="#7a6220", ha="right", zorder=6)

    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Der Effizienzrand nach Markowitz\n"
                 f"{n} Titel, Tagesdaten "
                 f"{asset_returns.index[0]:%d.%m.%Y} – {asset_returns.index[-1]:%d.%m.%Y}",
                 fontweight="bold")
    ax.set_xlabel("Volatilität σ (% p. a.)")
    ax.set_ylabel("Erwartete Rendite μ (% p. a.)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)

    if draussen:
        ax.annotate("außerhalb des Ausschnitts: " + ", ".join(draussen),
                    xy=(0.99, 0.015), xycoords="axes fraction", ha="right",
                    fontsize=8.5, color="#666666", style="italic")

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    log.info(f"  a = {a:.3f} | MVP: mu = {mu_mvp*100:.2f} %, sigma = {sig_mvp*100:.3f} %"
             + (f" | ausserhalb: {', '.join(draussen)}" if draussen else ""))

def create_theory_plots(asset_returns: pd.DataFrame, returns_df: pd.DataFrame,
                        output_dir: str) -> None:
    """Erzeugt alle drei Theorie-Abbildungen in den Ergebnisordner.

    Diese eine Funktion ruft ``run.py`` auf. Sie übernimmt auch das Zuschneiden
    der Kursrenditen auf den Auswertungszeitraum des Backtests: ``asset_returns``
    beginnt schon 2013 (Vorlauf für die Indikatoren), bewertet wird aber erst ab
    2015. ``returns_df`` — die Renditen der Strategien — gibt den richtigen
    Ausschnitt vor.
    """
    zeitraum = asset_returns.loc[returns_df.index[0]:returns_df.index[-1]]
    log.info(f"Theorie-Abbildungen: {len(zeitraum)} Handelstage, "
             f"{zeitraum.index[0]:%d.%m.%Y} – {zeitraum.index[-1]:%d.%m.%Y}")

    plot_two_asset_diversification(
        zeitraum, os.path.join(output_dir, "14_theorie_zwei_anlagen.png"))
    plot_diversification_limit(
        zeitraum, os.path.join(output_dir, "15_theorie_diversifikationsgrenze.png"))
    plot_estimation_uncertainty(
        zeitraum, os.path.join(output_dir, "16_theorie_schaetzunsicherheit.png"))
    plot_efficient_frontier_theory(
        zeitraum, os.path.join(output_dir, "17_theorie_effizienzrand.png"))
