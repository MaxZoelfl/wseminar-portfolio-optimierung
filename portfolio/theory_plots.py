"""
Die drei Theorie-Abbildungen zu Kapitel 2 der Seminararbeit.

FÜR EINSTEIGER — WAS MACHT DIESE DATEI?
Die übrigen Abbildungen des Projekts (in ``plots.py``) zeigen ERGEBNISSE:
wie sich die vier Strategien geschlagen haben. Die drei Bilder hier zeigen
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
