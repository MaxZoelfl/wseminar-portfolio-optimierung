"""Alle Visualisierungen sowie CSV-/JSON-Export."""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize as MplNorm
import seaborn as sns
from .config import *
from .metrics import *
from .indicators import *
from .optimizers import *

def plot_cumulative_returns(returns_df: pd.DataFrame, output_path: str) -> None:
    """Abbildung 1: Kumulierte Portfoliorenditen mit Drawdown-Panel."""
    log.info("Plot 1: Kumulierte Renditen …")
    fig, (ax_main, ax_dd) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    for col in returns_df.columns:
        color = COLORS.get(col, "gray")
        r     = returns_df[col].dropna()
        cum   = (1 + r).cumprod()
        ax_main.plot(cum.index, cum.values, label=col, color=color,
                     linewidth=2, zorder=3)
        ax_main.fill_between(cum.index, cum, cum.cummax(),
                              color=color, alpha=0.07)
        roll_max = cum.cummax()
        dd       = (cum - roll_max) / roll_max * 100
        ax_dd.fill_between(dd.index, dd.values, 0, color=color, alpha=0.4)
        ax_dd.plot(dd.index, dd.values, color=color, linewidth=0.8, alpha=0.7)

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
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → {output_path}")


def plot_weight_heatmap(weights_df: pd.DataFrame, title: str,
                         output_path: str) -> None:
    """Abbildungen 2, 3, 3b: Heatmap der Portfoliogewichtungen über Zeit."""
    log.info(f"Plot Heatmap: {title} …")
    data       = (weights_df * 100).T
    col_labels = [
        c.strftime("%b %Y") if hasattr(c, "strftime") else str(c)
        for c in data.index
    ]
    fig, ax = plt.subplots(figsize=(max(14, len(col_labels) * 0.55), 7))
    sns.heatmap(
        data.T, ax=ax, cmap="YlOrRd", linewidths=0.35, linecolor="white",
        annot=True, fmt=".1f", annot_kws={"size": 8},
        cbar_kws={"label": "Gewicht (%)", "shrink": 0.75},
        vmin=0, vmax=60,
    )
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
    """Abbildung 4: Efficient Frontier mit CML und allen Portfolio-Punkten."""
    log.info("Plot 4: Efficient Frontier …")
    mvo_opt  = MarkowitzLedoitWolf(rf=rf)
    frontier = mvo_opt.efficient_frontier(mu_hist, cov_ann, max_weight=MAX_WEIGHT)

    fig, ax = plt.subplots(figsize=(12, 8))

    if not frontier.empty:
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

    ret_tan, vol_tan, _ = portfolio_perf(w_mvo, mu_hist, cov_ann, rf)
    if vol_tan > 0:
        cml_vols = np.linspace(0, vol_tan * 1.6, 120)
        cml_rets = rf + (ret_tan - rf) / vol_tan * cml_vols
        ax.plot(cml_vols * 100, cml_rets * 100,
                color=COLORS["cml"], linestyle="--", linewidth=1.6, alpha=0.9,
                label=f"Capital Market Line (rf = {rf*100:.1f}%)", zorder=3)
        ax.scatter([0], [rf * 100], marker="*", s=180, color=COLORS["cml"],
                   zorder=7, label=f"Risikoloser Zinssatz ({rf*100:.1f}%)")

    for i, t in enumerate(tickers):
        a_vol = np.sqrt(cov_ann[i, i]) * 100
        a_ret = mu_hist[i] * 100
        ax.scatter(a_vol, a_ret, color="lightsteelblue", s=55, zorder=4,
                   alpha=0.8, edgecolors="steelblue", linewidths=0.5)
        ax.annotate(t, (a_vol, a_ret), textcoords="offset points",
                    xytext=(5, 2), fontsize=7.5, color="dimgrey")

    def _add_pt(w, mu, label, color, marker, size=230):
        ret, vol, sr = portfolio_perf(w, mu, cov_ann, rf)
        ax.scatter(vol * 100, ret * 100, color=color, marker=marker, s=size,
                   zorder=8, edgecolors="black", linewidths=1.0,
                   label=f"{label}\n  Rendite: {ret*100:.1f}%  "
                         f"Vola: {vol*100:.1f}%  Sharpe: {sr:.2f}")

    _add_pt(w_mvo, mu_hist, "Markowitz MVO", COLORS["Markowitz MVO"], "D")
    _add_pt(w_rf,  mu_rf,   "Random Forest", COLORS["Random Forest"], "^")
    _add_pt(w_rp,  mu_hist, "Risk Parity",   COLORS["Risk Parity"],   "P")
    _add_pt(w_ew,  mu_hist, "Equal Weight",  COLORS["Equal Weight"],  "s", 190)

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
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92,
              bbox_to_anchor=(1.18, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → {output_path}")


def plot_performance_metrics(metrics_df: pd.DataFrame, output_path: str) -> None:
    """Abbildung 5: Sechspanel-Balkendiagramm aller Kennzahlen."""
    log.info("Plot 5: Performance-Kennzahlen …")
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

    for ax, (key, label) in zip(axes.flatten(), display):
        vals = metrics_df[key]
        bars = ax.barh(vals.index, vals.values,
                       color=palette[:len(vals)], height=0.5, edgecolor="white")
        span = vals.abs().max() if vals.abs().max() > 0 else 1
        for bar, val in zip(bars, vals):
            sign = "+" if val > 0 else ""
            ax.text(val + span * 0.03, bar.get_y() + bar.get_height() / 2,
                    f"{sign}{val:.2f}", va="center", ha="left",
                    fontsize=9, fontweight="bold")
        ax.set_title(label, fontweight="bold", pad=8)
        ax.axvline(0, color="black", linewidth=0.7, alpha=0.4)
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
    """Abbildung 6: Rollierender 1-Jahres-Sharpe Ratio."""
    log.info("Plot 6: Rollierender Sharpe Ratio …")
    rf_daily = RISK_FREE_RATE / 252
    fig, ax  = plt.subplots(figsize=(13, 5))

    for col in returns_df.columns:
        r  = returns_df[col]
        rs = (r.rolling(window).mean() - rf_daily) / r.rolling(window).std() * np.sqrt(252)
        ax.plot(rs.index, rs.values, label=col,
                color=COLORS.get(col, "gray"), linewidth=1.8, alpha=0.9)

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
    """Abbildung 7: Feature Importance des Random Forest (MDI)."""
    log.info("Plot 7: Feature Importance …")
    if rf_optimizer.best_estimator_ is None:
        log.warning("RF nicht trainiert, überspringe.")
        return
    try:
        rf_step = rf_optimizer.best_estimator_.named_steps["rf"]
        imps    = rf_step.feature_importances_
        feat_names = [c for c in FEATURE_COLS
                      if c in rf_optimizer.best_estimator_.feature_names_in_
                      ] if hasattr(rf_optimizer.best_estimator_, "feature_names_in_") \
                        else FEATURE_COLS[:len(imps)]

        importance_df = pd.DataFrame({"Feature": feat_names, "Importance": imps})
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
    """Abbildung 8: Evolution der Efficient Frontier über den Backtestzeitraum."""
    if not frontier_snapshots:
        return
    log.info("Plot 8: Frontier-Evolution …")
    all_snaps = pd.concat(frontier_snapshots, ignore_index=True)
    dates     = sorted(all_snaps["date"].unique())
    n_dates   = len(dates)
    cmap      = plt.cm.viridis
    norm      = MplNorm(vmin=0, vmax=n_dates - 1)

    fig, ax = plt.subplots(figsize=(13, 8))
    for idx, date in enumerate(dates):
        snap  = all_snaps[all_snaps["date"] == date].copy()
        color = cmap(norm(idx))
        alpha = 0.20 + 0.70 * (idx / max(n_dates - 1, 1))
        ax.plot(snap["vol"] * 100, snap["ret"] * 100,
                color=color, linewidth=1.2, alpha=alpha, zorder=2)
        tang_idx = snap["sr"].idxmax()
        ax.scatter(snap.loc[tang_idx, "vol"] * 100, snap.loc[tang_idx, "ret"] * 100,
                   color=color, marker="D", s=18, alpha=max(0.4, alpha),
                   zorder=3, linewidths=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Rebalancing-Zeitpunkt (früh → spät)", fontsize=9)
    tick_idx = np.linspace(0, n_dates - 1, min(6, n_dates), dtype=int)
    cbar.set_ticks(tick_idx)
    cbar.set_ticklabels([str(dates[i].year) for i in tick_idx])

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
    """
    log.info("Plot 9: Stress-Tests …")
    stress_periods = {
        "COVID-Crash (Jan–Jun 2020)": ("2020-01-01", "2020-06-30"),
        "Fed-Zinserhöhungen (Nov 2021–Dez 2022)": ("2021-11-01", "2022-12-31"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Stress-Test: Krisenresistenz der Portfoliostrategien",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, (title, (start, end)) in zip(axes, stress_periods.items()):
        period = returns_df.loc[start:end].copy()
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
            cum = cum / cum.iloc[0]
            line, = ax.plot(cum.index, cum.values, label=col,
                            color=COLORS.get(col, "gray"), linewidth=2.2)
            legend_lines.append(line)

            # Max. Drawdown annotieren
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

    Wissenschaftliche Relevanz:
      Frazzini, A., Israel, R., Moskowitz, T. (2015): Trading Costs.
      → Transaktionskosten sind für aktive Strategien der wichtigste
        Performance-Treiber nach Gebühren.
    """
    log.info("Plot 10: Turnover vs. Performance …")

    # Monatliche Renditen aggregieren
    monthly_returns = (1 + returns_df).resample("ME").prod() - 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Turnover-Effizienz-Analyse\n"
        "Monatlicher Handelsumsatz vs. nachfolgende Monatsrendite",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # Panel 1: Scatter Turnover vs. nächste Monatsrendite
    ax = axes[0]
    strategy_to_col = {
        "Markowitz MVO" : "turnover_mvo",
        "Random Forest" : "turnover_rf",
        "Risk Parity"   : "turnover_rp",
        "Equal Weight"  : "turnover_ew",
    }

    for strat, to_col in strategy_to_col.items():
        if to_col not in turnover_df.columns or strat not in monthly_returns.columns:
            continue
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

        # Achsengrenzen einmalig berechnen
        vol_min = all_snaps["vol"].min() * 100 * 0.95
        vol_max = all_snaps["vol"].max() * 100 * 1.05
        ret_min = all_snaps["ret"].min() * 100 * 1.10
        ret_max = all_snaps["ret"].max() * 100 * 1.10

        fig, ax = plt.subplots(figsize=(10, 7))
        cmap    = plt.cm.viridis
        norm    = MplNorm(vmin=0, vmax=n_dates - 1)

        # Alle früheren Frontiers als Hintergrund (grau)
        bg_lines = []
        for _ in range(n_dates):
            line, = ax.plot([], [], color="lightgray", linewidth=0.8, alpha=0.5, zorder=1)
            bg_lines.append(line)

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
            # Vorherige Frontiers im Hintergrund
            for j, line in enumerate(bg_lines):
                if j < frame:
                    snap = all_snaps[all_snaps["date"] == dates[j]]
                    line.set_data(snap["vol"] * 100, snap["ret"] * 100)
                    line.set_color(cmap(norm(j)))
                    line.set_alpha(0.20 + 0.40 * (j / max(n_dates - 1, 1)))
                else:
                    line.set_data([], [])

            # Aktuelle Frontier hervorheben
            snap = all_snaps[all_snaps["date"] == dates[frame]]
            active_line.set_data(snap["vol"] * 100, snap["ret"] * 100)
            active_line.set_color(cmap(norm(frame)))

            # Tangentialpunkt
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
            blit=True,
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

    Im Gegensatz zu MDI-Feature Importance:
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
        rf_step  = rf_optimizer.best_estimator_.named_steps["rf"]
        scaler   = rf_optimizer.best_estimator_.named_steps["scaler"]
        X_scaled = scaler.transform(X_train.values)

        # SHAP TreeExplainer: effizient für baumbasierte Modelle
        explainer   = shap.TreeExplainer(rf_step)
        n_sample    = min(500, len(X_scaled))
        X_sub       = X_scaled[:n_sample]
        shap_values = explainer.shap_values(X_sub)

        # Feature-Namen
        feat_names = [FEATURE_DISPLAY_NAMES.get(c, c) for c in X_train.columns]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Summary Plot: Bee-Swarm (Feature-Beiträge je Beobachtung)
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

        # Bar Plot: Mittlere absolute SHAP-Werte (globale Importance)
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
    log.info("Speichere CSV-Dateien …")
    cum = (1 + returns_df).cumprod()
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
            "n_assets"          : len(tickers),
            "tickers"           : tickers,
            "feature_cols"      : FEATURE_COLS,
        },
        "performance_metrics": metrics_df.to_dict(),
        "significance"       : significance_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"  → {output_path}")
