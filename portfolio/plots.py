"""
Alle Visualisierungen sowie CSV-/JSON-Export.

FÜR EINSTEIGER — WAS MACHT DIESE DATEI?
Nach dem Backtest liegen die Ergebnisse als nackte Zahlenreihen vor. Diese
Datei übersetzt sie in die Abbildungen der Seminararbeit (nummerierte
PNG-Bilddateien im Output-Ordner) und exportiert die Rohdaten als CSV-
Tabellen (für Excel) sowie ein JSON-Protokoll (für die Reproduzierbarkeit).

Wiederkehrendes Matplotlib-Grundmuster in fast jeder Funktion:
  1. fig, ax = plt.subplots(...)   → leere Leinwand ("Figure") mit einem
                                     oder mehreren Koordinatensystemen ("Axes")
  2. ax.plot / ax.bar / ax.scatter → Daten als Linien/Balken/Punkte einzeichnen
  3. Beschriftungen: set_title (Überschrift), set_xlabel/set_ylabel (Achsen),
     legend (Erklärkästchen, welche Farbe welche Strategie ist)
  4. plt.savefig(...)              → als PNG-Datei speichern
     (dpi = Auflösung; bbox_inches="tight" = Ränder knapp zuschneiden)
  5. plt.close()                   → Speicher wieder freigeben

Außerdem oft benutzt:
  - (1 + r).cumprod(): macht aus Tagesrenditen den Depotwert-Verlauf
    ("aus 1 € wurden …") — das kumulative Produkt der Wachstumsfaktoren.
  - "* 100": rechnet Anteile (0.05) in Prozent (5 %) für die Anzeige um.
  - COLORS / STRATEGIES: zentrale Farb- und Namenslisten aus config.py,
    damit jede Strategie in allen Bildern dieselbe Farbe hat.
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick          # Zahlenformate an den Achsen (z. B. "12.3%")
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize as MplNorm   # bildet Zahlen auf Farbskalen ab
import seaborn as sns                      # für die Heatmaps
from .config import *
from .metrics import *
from .indicators import *
from .optimizers import *

def plot_cumulative_returns(returns_df: pd.DataFrame, output_path: str) -> None:
    """Abbildung 1: Kumulierte Portfoliorenditen mit Drawdown-Panel.

    Das wichtigste Bild der Arbeit: Oben der Depotwert-Verlauf aller vier
    Strategien ("aus 1 € wurden über 10 Jahre …"), unten dazu synchron der
    "Drawdown" — wie weit jede Strategie zu jedem Zeitpunkt unter ihrem
    bisherigen Höchststand lag (Krisen erscheinen als tiefe Täler).
    """
    log.info("Plot 1: Kumulierte Renditen …")
    # Zwei übereinanderliegende Diagramme, oben 3× so hoch wie unten;
    # sharex=True koppelt die Zeitachsen (Zoom/Ausschnitt immer synchron).
    fig, (ax_main, ax_dd) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    for col in returns_df.columns:            # eine Linie je Strategie
        color = COLORS.get(col, "gray")
        r     = returns_df[col].dropna()
        cum   = (1 + r).cumprod()             # Tagesrenditen → Depotwert-Verlauf
        ax_main.plot(cum.index, cum.values, label=col, color=color,
                     linewidth=2, zorder=3)
        # Zarte Schattierung zwischen Kurve und bisherigem Hoch (zeigt
        # "verlorenes Terrain" direkt im oberen Bild):
        ax_main.fill_between(cum.index, cum, cum.cummax(),
                              color=color, alpha=0.07)
        # Drawdown fürs untere Panel: Abstand zum bisherigen Hoch in %.
        roll_max = cum.cummax()
        dd       = (cum - roll_max) / roll_max * 100
        ax_dd.fill_between(dd.index, dd.values, 0, color=color, alpha=0.4)
        ax_dd.plot(dd.index, dd.values, color=color, linewidth=0.8, alpha=0.7)

    # Referenzlinie beim Startkapital (1 €):
    ax_main.axhline(1, color="black", linewidth=0.8, linestyle="--",
                    alpha=0.4, label="Startkapital")
    ax_main.set_ylabel("Kumulierter Wert (Start = 1 €)")
    ax_main.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.2f} €"))
    ax_main.legend(loc="upper left", framealpha=0.9)
    ax_main.set_title(
        "Kumulierte Portfoliorenditen im Vergleich\n"
        "(MVO | Random Forest | Equal Weight | Risk Parity | "
        "monatliches Rebalancing)",
        pad=12,
    )
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Datum")
    ax_dd.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax_dd.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.0f}%"))
    plt.tight_layout()                        # Abstände automatisch entzerren
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → {output_path}")


def plot_weight_heatmap(weights_df: pd.DataFrame, title: str,
                         output_path: str) -> None:
    """Abbildungen 2, 3, 3b: Heatmap der Portfoliogewichtungen über Zeit.

    Eine "Heatmap" ist eine Tabelle, deren Zellen nach Wert eingefärbt sind:
    Zeilen = Aktien, Spalten = Rebalancing-Monate, Farbe = Gewicht in %
    (hellgelb ≈ 0 %, dunkelrot = hoch). So sieht man auf einen Blick, ob eine
    Strategie breit streut (gleichmäßig blass) oder konzentriert wettet
    (einzelne dunkle Bänder).
    """
    log.info(f"Plot Heatmap: {title} …")
    data       = (weights_df * 100).T          # Anteile → Prozent; Tabelle drehen
    # Spaltenbeschriftungen als "Mrz 2015"-Datumstexte:
    col_labels = [
        c.strftime("%b %Y") if hasattr(c, "strftime") else str(c)
        for c in data.index
    ]
    # Bildbreite wächst mit der Monatszahl, damit die Zellen lesbar bleiben:
    fig, ax = plt.subplots(figsize=(max(14, len(col_labels) * 0.55), 7))
    sns.heatmap(
        data.T, ax=ax, cmap="YlOrRd", linewidths=0.35, linecolor="white",
        annot=True, fmt=".1f", annot_kws={"size": 8},   # Zahlwert in jede Zelle
        cbar_kws={"label": "Gewicht (%)", "shrink": 0.75},
        vmin=0, vmax=60,                                # feste Farbskala 0–60 %
    )
    # Nur jede zweite Monatsbeschriftung anzeigen (sonst überlappen sie):
    visible = [lbl if j % 2 == 0 else "" for j, lbl in enumerate(col_labels)]
    ax.set_xticklabels(visible, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(f"Portfolio-Gewichtungen: {title}\n(% | je Rebalancing-Monat)", pad=12)
    ax.set_xlabel("Rebalancing-Datum")
    ax.set_ylabel("Asset")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → {output_path}")


def plot_efficient_frontier(mu_hist: np.ndarray, cov_ann: np.ndarray,
                             tickers: list, w_mvo: np.ndarray,
                             w_rf: np.ndarray, w_rp: np.ndarray,
                             w_ew: np.ndarray, mu_rf: np.ndarray,
                             rf: float, output_path: str) -> None:
    """Abbildung 4: Efficient Frontier mit CML und allen Portfolio-Punkten.

    Das Lehrbuch-Bild der Markowitz-Theorie, gezeichnet mit den Schätzungen
    des LETZTEN Backtest-Monats. Koordinaten: x = Risiko (Volatilität),
    y = erwartete Rendite. Eingezeichnet werden:
      - die Effizienzlinie (Kurve der bestmöglichen Portfolios),
      - die Capital Market Line (CML): Gerade vom risikofreien Zins durch das
        Tangentialportfolio — Mischungen aus sicherer Anlage und Depot,
      - alle 15 Einzelaktien (kleine Punkte, stets UNTER der Kurve — einzeln
        ist man nie effizient, das ist der Diversifikationsgewinn),
      - die vier Strategie-Portfolios und das Minimum-Varianz-Portfolio (MVP).
    """
    log.info("Plot 4: Efficient Frontier …")
    # Effizienzlinie mit derselben 20-%-Obergrenze berechnen wie im Backtest,
    # damit das Markowitz-Portfolio wirklich AUF der Kurve liegt:
    mvo_opt  = MarkowitzLedoitWolf(rf=rf)
    frontier = mvo_opt.efficient_frontier(mu_hist, cov_ann, max_weight=MAX_WEIGHT)

    fig, ax = plt.subplots(figsize=(12, 8))

    if not frontier.empty:
        # Kurvenpunkte nach ihrer Sharpe Ratio einfärben (Farbskala "viridis"):
        scatter = ax.scatter(
            frontier["vol"] * 100, frontier["ret"] * 100,
            c=frontier["sr"], cmap="viridis", s=18, zorder=2, alpha=0.85,
            label="Efficient Frontier",
        )
        plt.colorbar(scatter, ax=ax, pad=0.01, shrink=0.8).set_label(
            "Sharpe Ratio", fontsize=9
        )
        ax.plot(frontier["vol"] * 100, frontier["ret"] * 100,
                color=COLORS["frontier"], linewidth=1.5, alpha=0.6, zorder=1)

    # Capital Market Line: Gerade ab (0, rf) durch das Tangentialportfolio.
    ret_tan, vol_tan, _ = portfolio_perf(w_mvo, mu_hist, cov_ann, rf)
    if vol_tan > 0:
        cml_vols = np.linspace(0, vol_tan * 1.6, 120)
        cml_rets = rf + (ret_tan - rf) / vol_tan * cml_vols   # Geradengleichung
        ax.plot(cml_vols * 100, cml_rets * 100,
                color=COLORS["cml"], linestyle="--", linewidth=1.6, alpha=0.9,
                label=f"Capital Market Line (rf = {rf*100:.1f}%)", zorder=3)
        ax.scatter([0], [rf * 100], marker="*", s=180, color=COLORS["cml"],
                   zorder=7, label=f"Risikoloser Zinssatz ({rf*100:.1f}%)")

    # Die 15 Einzelaktien als beschriftete Punkte:
    for i, t in enumerate(tickers):
        a_vol = np.sqrt(cov_ann[i, i]) * 100    # Diagonale der Matrix = Varianz je Aktie
        a_ret = mu_hist[i] * 100
        ax.scatter(a_vol, a_ret, color="lightsteelblue", s=55, zorder=4,
                   alpha=0.8, edgecolors="steelblue", linewidths=0.5)
        ax.annotate(t, (a_vol, a_ret), textcoords="offset points",
                    xytext=(5, 2), fontsize=7.5, color="dimgrey")

    def _add_pt(w, mu, label, color, marker, size=230):
        """Hilfsfunktion: ein Strategie-Portfolio als großen Marker setzen,
        mit Rendite/Vola/Sharpe direkt im Legendentext."""
        ret, vol, sr = portfolio_perf(w, mu, cov_ann, rf)
        ax.scatter(vol * 100, ret * 100, color=color, marker=marker, s=size,
                   zorder=8, edgecolors="black", linewidths=1.0,
                   label=f"{label}\n  Rendite: {ret*100:.1f}%  "
                         f"Vola: {vol*100:.1f}%  Sharpe: {sr:.2f}")

    # Beachte: Der RF-Punkt wird mit den PROGNOSTIZIERTEN Renditen (mu_rf)
    # bewertet, die anderen mit den historischen (mu_hist).
    _add_pt(w_mvo, mu_hist, "Markowitz MVO", COLORS["Markowitz MVO"], "D")
    _add_pt(w_rf,  mu_rf,   "Random Forest", COLORS["Random Forest"], "^")
    _add_pt(w_rp,  mu_hist, "Risk Parity",   COLORS["Risk Parity"],   "P")
    _add_pt(w_ew,  mu_hist, "Equal Weight",  COLORS["Equal Weight"],  "s", 190)

    # Minimum-Varianz-Portfolio: der Kurvenpunkt mit dem kleinsten Risiko
    # (das "linke Ende" der Effizienzlinie).
    if not frontier.empty:
        mvp = frontier.loc[frontier["vol"].idxmin()]
        ax.scatter(mvp["vol"] * 100, mvp["ret"] * 100,
                   color=COLORS["mvp"], marker="P", s=210, zorder=8,
                   edgecolors="black", linewidths=0.9,
                   label=f"MVP  Vol: {mvp['vol']*100:.1f}%  "
                         f"Sharpe: {mvp['sr']:.2f}")

    ax.set_title(
        "Effizienzkurve (Efficient Frontier) mit Capital Market Line\n"
        "Markowitz (1952) | Kovarianz: Ledoit-Wolf (2004)", pad=12,
    )
    ax.set_xlabel("Annualisierte Volatilität (%)")
    ax.set_ylabel("Annualisierte Erwartungsrendite (%)")
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}%"))
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}%"))
    # Legende rechts NEBEN das Diagramm setzen (bbox_to_anchor), weil sie
    # mit den Kennzahlen-Zeilen recht groß ist:
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92,
              bbox_to_anchor=(1.18, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → {output_path}")


def plot_performance_metrics(metrics_df: pd.DataFrame, output_path: str) -> None:
    """Abbildung 5: Sechspanel-Balkendiagramm aller Kennzahlen.

    Sechs kleine Diagramme nebeneinander (2 Zeilen × 3 Spalten), jedes zeigt
    EINE Kennzahl als horizontale Balken — ein Balken je Strategie, Zahlwert
    direkt am Balkenende. So lassen sich alle "Noten" auf einen Blick
    vergleichen (Bedeutung der Kennzahlen: siehe metrics.py).
    """
    log.info("Plot 5: Performance-Kennzahlen …")
    # Paare (Spaltenname in der Tabelle, Anzeigetitel im Diagramm):
    display = [
        ("CAGR (%)",                "CAGR (% p.a.)"),
        ("Sharpe Ratio",            "Sharpe Ratio"),
        ("Sortino Ratio",           "Sortino Ratio"),
        ("Max. Drawdown (%)",       "Max. Drawdown (%)"),
        ("Calmar Ratio",            "Calmar Ratio"),
        ("Annualisierte Vola. (%)", "Volatilität (% p.a.)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    palette   = [COLORS.get(c, "#aaaaaa") for c in metrics_df.index]

    # zip verheiratet die 6 Teildiagramme mit den 6 Kennzahlen:
    for ax, (key, label) in zip(axes.flatten(), display):
        vals = metrics_df[key]
        bars = ax.barh(vals.index, vals.values,          # barh = horizontale Balken
                       color=palette[:len(vals)], height=0.5, edgecolor="white")
        # "span" = größter Absolutwert; dient als Maßstab für Textabstände
        # und Achsengrenzen (damit die Beschriftung nie am Rand klebt).
        span = vals.abs().max() if vals.abs().max() > 0 else 1
        for bar, val in zip(bars, vals):
            sign = "+" if val > 0 else ""
            ax.text(val + span * 0.03, bar.get_y() + bar.get_height() / 2,
                    f"{sign}{val:.2f}", va="center", ha="left",
                    fontsize=9, fontweight="bold")
        ax.set_title(label, fontweight="bold", pad=8)
        ax.axvline(0, color="black", linewidth=0.7, alpha=0.4)   # Nulllinie
        ax.set_xlim(vals.min() - span * 0.3, vals.max() + span * 0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Performance-Kennzahlen im Vergleich | Rollierendes Backtest (2015–2024)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → {output_path}")


def plot_rolling_sharpe(returns_df: pd.DataFrame, output_path: str,
                         window: int = 252) -> None:
    """Abbildung 6: Rollierender 1-Jahres-Sharpe Ratio.

    Statt EINER Sharpe-Zahl für 10 Jahre: die Sharpe Ratio im gleitenden
    252-Tage-Fenster, als Zeitverlauf. Zeigt, dass die Rangfolge der
    Strategien über die Marktphasen wechselt — ein zentrales Argument der
    Arbeit gegen vorschnelle "Strategie X ist besser"-Schlüsse.
    """
    log.info("Plot 6: Rollierender Sharpe Ratio …")
    rf_daily = RISK_FREE_RATE / 252     # Jahreszins auf einen Tag herunterbrechen
    fig, ax  = plt.subplots(figsize=(13, 5))

    for col in returns_df.columns:
        r  = returns_df[col]
        # Je Zeitpunkt: (Ø Tagesrendite − Tageszins) / Streuung, annualisiert.
        rs = (r.rolling(window).mean() - rf_daily) / r.rolling(window).std() * np.sqrt(252)
        ax.plot(rs.index, rs.values, label=col,
                color=COLORS.get(col, "gray"), linewidth=1.8, alpha=0.9)

    # Orientierungslinien: 0 = kein Mehrwert übers Sparbuch, 1 = sehr gut.
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(1, color="grey",  linewidth=0.6, linestyle=":",  alpha=0.5)
    ax.set_title(f"Rollierender Sharpe Ratio ({window}-Tage-Fenster ≈ 1 Jahr)", pad=10)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Sharpe Ratio (annualisiert)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → {output_path}")


def plot_feature_importance(rf_optimizer: RFPortfolioOptimizer,
                             output_path: str) -> None:
    """Abbildung 7: Feature Importance des Random Forest (MDI).

    Beantwortet: Auf welche Merkmale hat der Random Forest beim Vorhersagen
    am meisten geachtet? "MDI" (Mean Decrease in Impurity) ist das eingebaute
    Wichtigkeitsmaß des Waldes: Es misst, wie stark ein Merkmal über alle
    Bäume hinweg die Prognosen verbessert hat. Ausgabe: horizontale Balken,
    wichtigstes Merkmal oben.
    """
    log.info("Plot 7: Feature Importance …")
    if rf_optimizer.best_estimator_ is None:
        log.warning("RF nicht trainiert, überspringe.")
        return
    try:
        # Aus der Pipeline den eigentlichen Wald herausgreifen und dessen
        # Wichtigkeitswerte auslesen (ein Wert je Merkmal, Summe = 1):
        rf_step = rf_optimizer.best_estimator_.named_steps["rf"]
        imps    = rf_step.feature_importances_
        # Merkmalsnamen möglichst aus dem Modell selbst rekonstruieren:
        feat_names = [c for c in FEATURE_COLS
                      if c in rf_optimizer.best_estimator_.feature_names_in_
                      ] if hasattr(rf_optimizer.best_estimator_, "feature_names_in_") \
                        else FEATURE_COLS[:len(imps)]

        importance_df = pd.DataFrame({"Feature": feat_names, "Importance": imps})
        # Interne Spaltennamen durch lesbare Beschriftungen ersetzen:
        importance_df["Feature"] = importance_df["Feature"].map(
            FEATURE_DISPLAY_NAMES).fillna(importance_df["Feature"])
        importance_df = importance_df.sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(9, max(7, len(importance_df) * 0.4)))
        bars = ax.barh(importance_df["Feature"], importance_df["Importance"] * 100,
                       color="#1f77b4", edgecolor="white", height=0.65)
        for bar, val in zip(bars, importance_df["Importance"] * 100):
            ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", ha="left", fontsize=9)
        ax.set_title(
            "Random Forest: Feature Importance (v4 mit Cross-sectional Ranks)\n"
            "Mean Decrease in Impurity | letzter Trainingsschritt", pad=12,
        )
        ax.set_xlabel("Relative Wichtigkeit (%)")
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}%"))
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        log.info(f"  → {output_path}")
    except Exception as e:
        log.warning(f"Feature Importance Fehler: {e}")


def plot_frontier_evolution(frontier_snapshots: list, output_path: str) -> None:
    """Abbildung 8: Evolution der Efficient Frontier über den Backtestzeitraum.

    Alle im Backtest gesammelten "Fotos" der Effizienzlinie (eines pro
    Quartal) übereinandergelegt, farblich von dunkel (2015) nach hell (2024)
    gestaffelt. Die Kernbotschaft: Die Kurve wandert mit jeder Neuschätzung
    kräftig — die "optimalen" Portfolios von gestern sind es morgen nicht
    mehr. Genau das ist das Schätzfehler-Problem der Markowitz-Methode.
    """
    if not frontier_snapshots:
        return
    log.info("Plot 8: Frontier-Evolution …")
    all_snaps = pd.concat(frontier_snapshots, ignore_index=True)
    dates     = sorted(all_snaps["date"].unique())
    n_dates   = len(dates)
    cmap      = plt.cm.viridis                 # Farbskala dunkelviolett → gelb
    norm      = MplNorm(vmin=0, vmax=n_dates - 1)   # Snapshot-Nr. → Farbposition

    fig, ax = plt.subplots(figsize=(13, 8))
    for idx, date in enumerate(dates):
        snap  = all_snaps[all_snaps["date"] == date].copy()
        color = cmap(norm(idx))
        # Jüngere Kurven decken kräftiger (höheres alpha), ältere verblassen:
        alpha = 0.20 + 0.70 * (idx / max(n_dates - 1, 1))
        ax.plot(snap["vol"] * 100, snap["ret"] * 100,
                color=color, linewidth=1.2, alpha=alpha, zorder=2)
        # Den Tangentialpunkt (max. Sharpe) jeder Kurve als Raute markieren:
        tang_idx = snap["sr"].idxmax()
        ax.scatter(snap.loc[tang_idx, "vol"] * 100, snap.loc[tang_idx, "ret"] * 100,
                   color=color, marker="D", s=18, alpha=max(0.4, alpha),
                   zorder=3, linewidths=0)

    # Farbbalken als Zeitleiste (beschriftet mit Jahreszahlen):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Rebalancing-Zeitpunkt (früh → spät)", fontsize=9)
    tick_idx = np.linspace(0, n_dates - 1, min(6, n_dates), dtype=int)
    cbar.set_ticks(tick_idx)
    cbar.set_ticklabels([str(dates[i].year) for i in tick_idx])

    # Erste und letzte Kurve extra dick nachzeichnen und in die Legende nehmen:
    for idx, label, lw in [
        (0,          f"Frontier {dates[0].strftime('%b %Y')} (Start)", 2.2),
        (n_dates - 1, f"Frontier {dates[-1].strftime('%b %Y')} (Ende)", 2.2),
    ]:
        snap  = all_snaps[all_snaps["date"] == dates[idx]]
        ax.plot(snap["vol"] * 100, snap["ret"] * 100,
                color=cmap(norm(idx)), linewidth=lw, alpha=1.0, label=label, zorder=5)

    ax.set_title(
        "Evolution der Effizienzlinie über den Backtestzeitraum (2015–2024)\n"
        "Jede Kurve = Frontier zu einem Rebalancing-Termin | Rauten = Tangentialpunkte",
        pad=12,
    )
    ax.set_xlabel("Annualisierte Volatilität (%)")
    ax.set_ylabel("Annualisierte Erwartungsrendite (%)")
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.0f}%"))
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.0f}%"))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    # Erklärkasten direkt ins Bild (für Leser ohne Bildunterschrift):
    ax.annotate(
        "Die Frontier verschiebt sich mit jeder Neuschätzung\n"
        "von μ und Σ (Ledoit-Wolf) — sie ist kein statisches Objekt.",
        xy=(0.02, 0.97), xycoords="axes fraction", fontsize=8, va="top",
        color="dimgrey",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → {output_path}")


def plot_stress_test(returns_df: pd.DataFrame, output_path: str) -> None:
    """
    Abbildung 9 (NEU v4): Stress-Test — Krisenperioden-Analyse.
    Zeigt relative Wertentwicklung (normiert auf 1.0 am Periodenanfang)
    und maximalen Drawdown je Strategie in zwei Stressperioden:
      - COVID-Crash (Januar–Juni 2020)
      - Fed-Zinserhöhungsperiode (November 2021 – Dezember 2022)

    Idee: Gute Schönwetter-Performance kann Schwächen verdecken — hier wird
    gezielt hineingezoomt, wie sich jede Strategie in den zwei größten
    Krisen des Zeitraums geschlagen hat. Der tiefste Punkt jeder Kurve wird
    mit dem jeweiligen Maximalverlust beschriftet (z. B. "↓−31.2%").
    """
    log.info("Plot 9: Stress-Tests …")
    # Die zwei Krisenfenster als Name → (Start, Ende):
    stress_periods = {
        "COVID-Crash (Jan–Jun 2020)": ("2020-01-01", "2020-06-30"),
        "Fed-Zinserhöhungen (Nov 2021–Dez 2022)": ("2021-11-01", "2022-12-31"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))   # ein Panel je Krise
    fig.suptitle(
        "Stress-Test: Krisenresistenz der Portfoliostrategien",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, (title, (start, end)) in zip(axes, stress_periods.items()):
        period = returns_df.loc[start:end].copy()     # nur die Krisen-Tage ausschneiden
        if period.empty:
            ax.text(0.5, 0.5, "Keine Daten für diesen Zeitraum",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        legend_lines = []
        for col in returns_df.columns:
            if col not in period.columns:
                continue
            r   = period[col].dropna()
            if len(r) == 0:
                continue
            cum = (1 + r).cumprod()
            cum = cum / cum.iloc[0]     # auf 1.0 am Krisenbeginn normieren
            line, = ax.plot(cum.index, cum.values, label=col,
                            color=COLORS.get(col, "gray"), linewidth=2.2)
            legend_lines.append(line)

            # Max. Drawdown annotieren: tiefsten Punkt finden (idxmin) und
            # den Verlust als Pfeil-Beschriftung daruntersetzen.
            mdd = (cum / cum.cummax() - 1).min()
            idx = (cum / cum.cummax() - 1).idxmin()
            ax.annotate(
                f"↓{mdd*100:.1f}%",
                xy=(idx, cum[idx]),
                xytext=(0, -22), textcoords="offset points",
                fontsize=7.5, color=COLORS.get(col, "gray"),
                arrowprops=dict(arrowstyle="-", color=COLORS.get(col, "gray"),
                                lw=0.8, alpha=0.6),
                ha="center",
            )

        ax.axhline(1, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylabel("Relativer Portfoliowert (Start = 1.0)")
        ax.set_xlabel("Datum")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.2f}"))
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → {output_path}")


def plot_turnover_performance(turnover_df: pd.DataFrame,
                               returns_df: pd.DataFrame,
                               output_path: str) -> None:
    """
    Abbildung 10 (NEU v4): Turnover vs. Rendite — Kosten-Effizienz-Analyse.

    Zeigt für jede Strategie den Zusammenhang zwischen dem monatlichen
    Handelsumsatz (Turnover) und der nachfolgenden Monatsrendite.
    Hoher Turnover bei niedriger Rendite = schlechte Kosten-Effizienz.
    (Links: Punktwolke, ein Punkt = ein Monat einer Strategie.
     Rechts: durchschnittlicher Turnover je Strategie als Balken —
     zeigt, wie viel "Hin und Her" jede Strategie produziert.)

    Wissenschaftliche Relevanz:
      Frazzini, A., Israel, R., Moskowitz, T. (2015): Trading Costs.
      → Transaktionskosten sind für aktive Strategien der wichtigste
        Performance-Treiber nach Gebühren.
    """
    log.info("Plot 10: Turnover vs. Performance …")

    # Tagesrenditen zu Monatsrenditen aufzinsen (multiplikativ, vgl. indicators):
    monthly_returns = (1 + returns_df).resample("ME").prod() - 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Turnover-Effizienz-Analyse\n"
        "Monatlicher Handelsumsatz vs. nachfolgende Monatsrendite",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # Panel 1: Scatter Turnover vs. nächste Monatsrendite
    ax = axes[0]
    # Zuordnung Strategiename → Spaltenname im Turnover-Protokoll:
    strategy_to_col = {
        "Markowitz MVO" : "turnover_mvo",
        "Random Forest" : "turnover_rf",
        "Risk Parity"   : "turnover_rp",
        "Equal Weight"  : "turnover_ew",
    }

    for strat, to_col in strategy_to_col.items():
        if to_col not in turnover_df.columns or strat not in monthly_returns.columns:
            continue
        # reindex richtet die Turnover-Daten am Monatsraster der Renditen aus;
        # danach nur Monate behalten, die in BEIDEN Reihen vorkommen:
        to   = turnover_df[to_col].reindex(monthly_returns.index).dropna() * 100
        ret  = monthly_returns[strat].reindex(to.index).dropna() * 100
        common = to.index.intersection(ret.index)
        ax.scatter(to[common], ret[common], label=strat,
                   color=COLORS.get(strat, "gray"), alpha=0.55, s=40,
                   edgecolors="none")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Handelsumsatz / Turnover (%)")
    ax.set_ylabel("Monatsrendite (%)")
    ax.set_title("Scatter: Turnover vs. Monatsrendite", pad=8)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}%"))
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}%"))

    # Panel 2: Durchschnittlicher Turnover je Strategie (Balken)
    ax = axes[1]
    avg_turnover = {}
    for strat, to_col in strategy_to_col.items():
        if to_col in turnover_df.columns:
            avg_turnover[strat] = turnover_df[to_col].mean() * 100

    strategies_list = list(avg_turnover.keys())
    values          = list(avg_turnover.values())
    colors_list     = [COLORS.get(s, "gray") for s in strategies_list]

    bars = ax.barh(strategies_list, values, color=colors_list,
                   height=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=9,
                fontweight="bold")

    ax.set_title("Ø Monatlicher Turnover je Strategie", pad=8)
    ax.set_xlabel("Durchschnittlicher Handelsumsatz (%)")
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}%"))

    # Erinnerung an die Kostenformel direkt im Bild:
    ax.annotate(
        f"Transaktionskosten: {TRANSACTION_COST*100:.2f}% × Turnover",
        xy=(0.04, 0.04), xycoords="axes fraction", fontsize=8,
        color="dimgrey",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → {output_path}")


def create_animated_frontier_gif(frontier_snapshots: list,
                                  output_path: str) -> None:
    """
    Abbildung 11 (NEU v4): Animierte GIF der Frontier-Evolution.
    Jeder Frame = ein Rebalancing-Zeitpunkt. Pillow wird benötigt.

    Dasselbe wie Abbildung 8, aber als Film: Die Effizienzlinie wandert
    Bild für Bild durch die Zeit; ältere Kurven bleiben blass im Hintergrund
    stehen. Technik: FuncAnimation ruft für jedes Einzelbild die Funktion
    _update(frame) auf, die die Liniendaten austauscht; PillowWriter fügt
    die Bilder zu einer GIF-Datei zusammen.
    """
    if not ANIMATION_AVAILABLE:
        log.warning("matplotlib.animation nicht verfügbar, GIF übersprungen.")
        return
    if not frontier_snapshots:
        log.warning("Keine Frontier-Snapshots, GIF übersprungen.")
        return
    log.info("Plot 11: Animierte Frontier-GIF …")

    try:
        all_snaps = pd.concat(frontier_snapshots, ignore_index=True)
        dates     = sorted(all_snaps["date"].unique())
        n_dates   = len(dates)

        # Achsengrenzen einmalig berechnen (über ALLE Frames hinweg fest,
        # damit das Bild während der Animation nicht springt):
        vol_min = all_snaps["vol"].min() * 100 * 0.95
        vol_max = all_snaps["vol"].max() * 100 * 1.05
        ret_min = all_snaps["ret"].min() * 100 * 1.10
        ret_max = all_snaps["ret"].max() * 100 * 1.10

        fig, ax = plt.subplots(figsize=(10, 7))
        cmap    = plt.cm.viridis
        norm    = MplNorm(vmin=0, vmax=n_dates - 1)

        # Alle Zeichen-Objekte VORAB leer anlegen; die Animation füllt sie
        # später nur noch mit Daten (viel schneller als jedes Mal neu zeichnen):
        # 1) Hintergrund-Linien für alle früheren Frontiers (anfangs grau/leer)
        bg_lines = []
        for _ in range(n_dates):
            line, = ax.plot([], [], color="lightgray", linewidth=0.8, alpha=0.5, zorder=1)
            bg_lines.append(line)

        # 2) die aktuell hervorgehobene Frontier, 3) ihr Tangentialpunkt,
        # 4) die Datumsanzeige oben links:
        active_line, = ax.plot([], [], color="blue", linewidth=2.5, zorder=4)
        tang_pt      = ax.scatter([], [], color="red", marker="D", s=80,
                                   zorder=5, edgecolors="black", linewidths=0.8)
        date_text    = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                               fontsize=11, va="top", fontweight="bold",
                               bbox=dict(boxstyle="round,pad=0.4",
                                         facecolor="lightyellow", alpha=0.9))

        ax.set_xlim(vol_min, vol_max)
        ax.set_ylim(ret_min, ret_max)
        ax.set_xlabel("Annualisierte Volatilität (%)")
        ax.set_ylabel("Annualisierte Erwartungsrendite (%)")
        ax.set_title(
            "Efficient Frontier — Evolution (2015–2024)\n"
            "Jeder Frame = ein Rebalancing-Termin | "
            "Raute = Tangentialpunkt (max. Sharpe)",
            pad=10,
        )
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.0f}%"))
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.0f}%"))
        ax.grid(True, alpha=0.2)

        def _update(frame):
            """Wird von FuncAnimation für jedes Einzelbild (frame = 0,1,2,…)
            aufgerufen und aktualisiert nur die Dateninhalte der Objekte."""
            # Vorherige Frontiers im Hintergrund einblenden:
            for j, line in enumerate(bg_lines):
                if j < frame:
                    snap = all_snaps[all_snaps["date"] == dates[j]]
                    line.set_data(snap["vol"] * 100, snap["ret"] * 100)
                    line.set_color(cmap(norm(j)))
                    line.set_alpha(0.20 + 0.40 * (j / max(n_dates - 1, 1)))
                else:
                    line.set_data([], [])   # zukünftige Kurven bleiben unsichtbar

            # Aktuelle Frontier hervorheben:
            snap = all_snaps[all_snaps["date"] == dates[frame]]
            active_line.set_data(snap["vol"] * 100, snap["ret"] * 100)
            active_line.set_color(cmap(norm(frame)))

            # Tangentialpunkt (Kurvenpunkt mit maximaler Sharpe Ratio):
            tang_idx = snap["sr"].idxmax()
            tang_pt.set_offsets(
                [[snap.loc[tang_idx, "vol"] * 100,
                  snap.loc[tang_idx, "ret"] * 100]]
            )

            date_text.set_text(dates[frame].strftime("%b %Y"))
            return [active_line, tang_pt, date_text] + bg_lines

        anim = FuncAnimation(
            fig, _update,
            frames=n_dates,
            interval=200,   # ms zwischen Frames
            blit=True,      # Optimierung: nur veränderte Bildteile neu zeichnen
        )
        anim.save(output_path, writer=PillowWriter(fps=4),
                  dpi=100, progress_callback=lambda i, n: None)
        plt.close(fig)
        log.info(f"  → {output_path}")
    except Exception as e:
        log.warning(f"GIF-Erstellung fehlgeschlagen: {e}")


def plot_shap_values(rf_optimizer: RFPortfolioOptimizer,
                     X_train: pd.DataFrame, output_path: str) -> None:
    """
    Abbildung 12 (NEU v4, optional): SHAP-Erklärbarkeit des Random Forest.

    SHAP (SHapley Additive exPlanations) erklärt, wie viel jedes Feature
    zur Renditeprognose beiträgt — sowohl im Durchschnitt (Importance)
    als auch für einzelne Beobachtungen (Richtung des Beitrags).
    Die Grundidee stammt aus der Spieltheorie (Shapley-Werte): Der Beitrag
    jedes Merkmals wird fair "ausbezahlt", so als wären die Merkmale
    Mitspieler in einem Team, dessen Gewinn (die Prognose) aufgeteilt wird.

    Im Gegensatz zu MDI-Feature Importance (Abbildung 7):
      - SHAP zeigt die Richtung des Einflusses (positiv/negativ)
      - SHAP ist konsistenter und berücksichtigt Interaktionseffekte
      - Standard in ML-Erklärbarkeitsforschung (Lundberg & Lee, 2017)

    Referenz: Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to
    Interpreting Model Predictions. NeurIPS 2017.
    """
    if not SHAP_AVAILABLE:
        log.info("SHAP nicht verfügbar (pip install shap). Plot übersprungen.")
        return
    if rf_optimizer.best_estimator_ is None:
        log.warning("RF nicht trainiert, SHAP übersprungen.")
        return

    log.info("Plot 12: SHAP-Werte …")
    try:
        # Wald und Skalierer aus der Pipeline holen; die Daten müssen genauso
        # skaliert werden wie beim Training, sonst passen sie nicht zum Modell:
        rf_step  = rf_optimizer.best_estimator_.named_steps["rf"]
        scaler   = rf_optimizer.best_estimator_.named_steps["scaler"]
        X_scaled = scaler.transform(X_train.values)

        # SHAP TreeExplainer: effizient für baumbasierte Modelle.
        # Zur Rechenzeit-Begrenzung höchstens 500 Beobachtungen analysieren:
        explainer   = shap.TreeExplainer(rf_step)
        n_sample    = min(500, len(X_scaled))
        X_sub       = X_scaled[:n_sample]
        shap_values = explainer.shap_values(X_sub)

        # Feature-Namen in lesbare Beschriftungen übersetzen:
        feat_names = [FEATURE_DISPLAY_NAMES.get(c, c) for c in X_train.columns]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Linkes Panel — "Bee-Swarm" (Bienenschwarm): ein Punkt je Beobachtung
        # und Merkmal. Position auf der x-Achse = Beitrag zur Prognose
        # (links = drückt die Prognose, rechts = hebt sie), Farbe = war der
        # Merkmalswert hoch oder niedrig. plt.sca(ax1) lenkt die SHAP-eigene
        # Zeichenfunktion in unser linkes Teilbild.
        plt.sca(ax1)
        shap.summary_plot(
            shap_values, X_sub,
            feature_names=feat_names,
            show=False, plot_size=None, max_display=15,
        )
        ax1.set_title(
            "SHAP Summary (Bee-Swarm)\nFarbe = Feature-Wert | x = Beitrag zur Prognose",
            fontsize=10, pad=8,
        )

        # Rechtes Panel — mittlere absolute SHAP-Werte: die globale Rangliste
        # der Merkmale (wie stark beeinflusst jedes im Schnitt die Prognose,
        # egal in welche Richtung). Nur die Top 15 anzeigen.
        mean_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_shap)
        plt.sca(ax2)
        ax2.barh(
            [feat_names[i] for i in sorted_idx[-15:]],
            mean_shap[sorted_idx[-15:]],
            color="#1f77b4", edgecolor="white", height=0.65,
        )
        ax2.set_title(
            "SHAP Feature Importance\nMittlerer absoluter SHAP-Wert (global)",
            fontsize=10, pad=8,
        )
        ax2.set_xlabel("|SHAP-Wert| (Einfluss auf Renditeprognose)")

        fig.suptitle(
            "SHAP-Erklärbarkeit des Random Forest (Lundberg & Lee, 2017)",
            fontsize=13, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        log.info(f"  → {output_path}")
    except Exception as e:
        log.warning(f"SHAP-Plot fehlgeschlagen: {e}")


def save_all_csv(returns_df: pd.DataFrame, metrics_df: pd.DataFrame,
                 weights_mvo: pd.DataFrame, weights_rf: pd.DataFrame,
                 weights_rp: pd.DataFrame, turnover_df: pd.DataFrame,
                 output_dir: str) -> None:
    """Exportiert alle Ergebnistabellen als CSV-Dateien.

    CSV ("comma-separated values") ist das einfachste Tabellenformat — mit
    Excel, Numbers oder jedem Texteditor zu öffnen. float_format legt die
    Anzahl der Nachkommastellen fest. So sind alle Zahlen der Arbeit auch
    außerhalb von Python nachprüfbar.
    """
    log.info("Speichere CSV-Dateien …")
    cum = (1 + returns_df).cumprod()          # zusätzlich den Depotwert-Verlauf ableiten
    cum.to_csv(           os.path.join(output_dir, "cumulative_returns.csv"),  float_format="%.6f")
    returns_df.to_csv(    os.path.join(output_dir, "daily_returns.csv"),       float_format="%.6f")
    metrics_df.to_csv(    os.path.join(output_dir, "performance_metrics.csv"), float_format="%.4f")
    weights_mvo.T.to_csv( os.path.join(output_dir, "weights_markowitz.csv"),   float_format="%.4f")
    weights_rf.T.to_csv(  os.path.join(output_dir, "weights_rf.csv"),          float_format="%.4f")
    weights_rp.T.to_csv(  os.path.join(output_dir, "weights_risk_parity.csv"), float_format="%.4f")
    turnover_df.to_csv(   os.path.join(output_dir, "turnover.csv"),            float_format="%.4f")
    log.info("  → 7 CSV-Dateien gespeichert.")


def save_experiment_json(metrics_df: pd.DataFrame,
                          significance_results: dict,
                          tickers: list,
                          output_path: str) -> None:
    """
    Speichert Experimentparameter + Ergebnisse als JSON (v4.1).
    Dient der Reproduzierbarkeit und wissenschaftlichen Dokumentation.
    Enthält die robuste Signifikanzanalyse (Ledoit-Wolf 2008, Holm,
    Deflated Sharpe Ratio) statt des früheren i.i.d.-Bootstraps.

    JSON ist ein maschinen- UND menschenlesbares Textformat. In der Datei
    steht das komplette "Laborprotokoll": Wer später wissen will, mit welchen
    Einstellungen genau diese Ergebnisse entstanden sind, findet hier alles —
    Zeitstempel, Parameter, Kennzahlen und Signifikanztests.
    """
    record = {
        "experiment_meta": {
            "version"           : "4.2",
            "timestamp"         : datetime.now().isoformat(),
            "backtest_start"    : BACKTEST_START,
            "backtest_end"      : END_DATE,
            "risk_free_rate"    : RISK_FREE_RATE,
            "train_years"       : TRAIN_YEARS,
            "max_weight"        : MAX_WEIGHT,
            "transaction_cost"  : TRANSACTION_COST,
            "rf_turnover_limit" : RF_TURNOVER_LIMIT,
            "rf_n_iter"         : RF_N_ITER,
            "rf_cv_splits"      : RF_CV_SPLITS,
            # Fairness-/Robustheitsoptionen: gehören ins Laborprotokoll, weil
            # sie die Ergebnisse verändern und ein Lauf sonst nicht
            # reproduzierbar wäre.
            "use_purged_cv"         : USE_PURGED_CV,
            "cv_embargo"            : CV_EMBARGO if USE_PURGED_CV else None,
            "mvo_turnover_limit"    : MVO_TURNOVER_LIMIT,
            "turnover_ref_drifted"  : TURNOVER_REF_DRIFTED,
            "min_variance_fallback" : MIN_VARIANCE_FALLBACK,
            "n_assets"          : len(tickers),
            "tickers"           : tickers,
            "feature_cols"      : FEATURE_COLS,
        },
        "performance_metrics": metrics_df.to_dict(),
        "significance"       : significance_results,
    }

    # indent=2 rückt schön ein (lesbar); ensure_ascii=False erhält Umlaute;
    # default=str macht auch Datumsobjekte u. ä. speicherbar.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"  → {output_path}")
