"""
Orchestrierung: Daten -> Indikatoren -> Backtest -> Kennzahlen -> Plots.

FÜR EINSTEIGER — WAS MACHT DIESE DATEI?
Dies ist der "Dirigent" des Projekts: Die Funktion main() ruft alle anderen
Bausteine in der richtigen Reihenfolge auf — von A wie "Daten laden" bis
I wie "Zusammenfassung drucken". Die Abschnitte sind im Code mit A–I
beschriftet. Wer verstehen will, wie das Gesamtsystem zusammenspielt,
liest am besten genau diese Datei von oben nach unten.

Gestartet wird das Ganze im Terminal mit:
    python -m portfolio
(das führt __main__.py aus, welches wiederum main() von hier aufruft).
"""

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from .config import *
from .metrics import *
from .indicators import *
from .optimizers import *
from .data import *
from .dashboard import *
from .backtest import *
from .plots import *
from .significance import (
    sharpe_difference_test, holm_bonferroni, deflated_sharpe_from_strategies,
)

def main():
    """Führt das komplette Experiment einmal von Anfang bis Ende aus."""
    t0_main = time.perf_counter()   # Gesamt-Stoppuhr
    log.info("=" * 70)
    log.info("  PORTFOLIO-OPTIMIERUNG V4 | MARKOWITZ vs. RF vs. EW vs. RISK PARITY")
    log.info("  Neu v4: Risk Parity | CS-Ranking | Live-Dashboard | SHAP | GIF")
    log.info(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    # Ohne yfinance gibt es keine Kursdaten → früh und verständlich abbrechen.
    if not YFINANCE_AVAILABLE:
        log.error("yfinance fehlt. Bitte: pip install yfinance")
        return

    # Ergebnisordner anlegen (exist_ok=True: kein Fehler, wenn er schon existiert).
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info(f"Output-Ordner: {os.path.abspath(OUTPUT_DIR)}")
    log.info(f"Interaktives Display: {INTERACTIVE_DISPLAY} | "
             f"SHAP: {SHAP_AVAILABLE} | Animation: {ANIMATION_AVAILABLE}")

    # ------------------------------------------------------------------
    # A: Daten laden — Kurse von Yahoo Finance holen, Renditen berechnen.
    # ------------------------------------------------------------------
    asset_prices, asset_returns, spy_returns, tickers = download_data(
        TICKERS, SPY_TICKER, START_DATE, END_DATE
    )

    # ------------------------------------------------------------------
    # B: Technische Indikatoren — die Merkmale für den Random Forest.
    # ------------------------------------------------------------------
    indicator_dict = build_all_indicators(
        asset_prices, asset_returns, spy_returns, tickers
    )

    # ------------------------------------------------------------------
    # C: Backtest (mit Live-Dashboard) — die eigentliche Simulation;
    #    das ist der mit Abstand rechenintensivste Schritt.
    # ------------------------------------------------------------------
    results = run_backtest(
        asset_prices, asset_returns, spy_returns, tickers, indicator_dict
    )

    # Das Ergebnis-dict in einzelne, sprechend benannte Variablen auspacken:
    returns_df         = results["returns"]
    weights_mvo        = results["weights_mvo"]
    weights_rf         = results["weights_rf"]
    weights_rp         = results["weights_rp"]
    turnover_df        = results["turnover_df"]
    mu_hist            = results["mu_hist"]
    mu_rf              = results["mu_rf"]
    cov_ann            = results["cov_ann"]
    w_mvo_last         = results["w_mvo_last"]
    w_rf_last          = results["w_rf_last"]
    w_rp_last          = results["w_rp_last"]
    frontier_snapshots = results["frontier_snapshots"]
    rfo                = results["rfo"]
    w_ew               = np.ones(len(tickers)) / len(tickers)   # 1/N, immer gleich

    # ------------------------------------------------------------------
    # D: Kennzahlen — die "Noten"-Tabelle aller Strategien berechnen
    #    und im Terminal ausgeben.
    # ------------------------------------------------------------------
    metrics_df = compute_metrics(returns_df)
    print("\n" + "=" * 70)
    print("  PERFORMANCE-KENNZAHLEN — VOLLSTÄNDIGE ÜBERSICHT (v4)")
    print("=" * 70)
    print(metrics_df.to_string())
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # E: Signifikanz- & Overfitting-Analyse (wissenschaftlich robust)
    #    Ledoit & Wolf (2008) + Holm-Bonferroni + Deflated Sharpe Ratio
    #    → Sind die Unterschiede ECHT oder Zufall? (Details: significance.py)
    # ------------------------------------------------------------------
    log.info("Signifikanzanalyse: Ledoit-Wolf-2008-Test + Holm-Bonferroni + Deflated Sharpe …")
    r_mvo = returns_df["Markowitz MVO"]
    r_rf  = returns_df["Random Forest"]
    r_rp  = returns_df["Risk Parity"]
    r_ew  = returns_df["Equal Weight"]

    # Die fünf paarweisen Vergleiche der Arbeit (Name, Reihe A, Reihe B):
    pairs = [
        ("RF vs. MVO",        r_rf,  r_mvo),
        ("RF vs. EW",         r_rf,  r_ew),
        ("MVO vs. EW",        r_mvo, r_ew),
        ("Risk Parity vs. EW",r_rp,  r_ew),
        ("MVO vs. RP",        r_mvo, r_rp),
    ]
    # Für jedes Paar den Sharpe-Differenz-Test rechnen, p-Werte sammeln:
    sharpe_tests = {}
    pvals = []
    for label, ra, rb in pairs:
        t = sharpe_difference_test(ra, rb, rf=RISK_FREE_RATE, n_boot=4999)
        sharpe_tests[label] = t
        pvals.append(t["p_value"])
    # Weil es FÜNF Tests sind: p-Werte per Holm-Bonferroni verschärfen
    # (sonst steigt die Chance auf Zufallstreffer, s. significance.py).
    holm = holm_bonferroni(pvals)
    for (label, _, _), p_adj, rej in zip(pairs, holm["p_adjusted"], holm["reject"]):
        sharpe_tests[label]["p_holm"] = float(p_adj)
        sharpe_tests[label]["significant_holm"] = bool(rej)
        log.info(
            f"  {label:<20} ΔSharpe={sharpe_tests[label]['diff_annual']:>+.3f} | "
            f"p_LW={sharpe_tests[label]['p_value']:.3f} | p_Holm={p_adj:.3f} | "
            f"{'SIGNIFIKANT' if rej else 'n.s.'}"
        )

    # Deflated Sharpe Ratio für jede Strategie (Schutz vor "Auswahlglück"):
    strat_cols = [c for c in STRATEGIES if c in returns_df.columns]
    dsr_results = {
        col: deflated_sharpe_from_strategies(returns_df, col, rf=RISK_FREE_RATE)
        for col in strat_cols
    }
    # Alles für das JSON-Protokoll (Abschnitt H) bündeln:
    significance_results = {
        "sharpe_difference_tests": sharpe_tests,
        "deflated_sharpe": dsr_results,
        "n_strategies": len(strat_cols),
        "method": "Ledoit-Wolf (2008) HAC block bootstrap; Holm-Bonferroni; "
                  "Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)",
    }

    # Testergebnisse übersichtlich im Terminal ausgeben
    # (✓ = statistisch signifikant, ✗ = nicht; "n.s." = nicht signifikant):
    print("\n  SHARPE-DIFFERENZ-TEST (Ledoit & Wolf 2008, HAC-Block-Bootstrap; "
          "Holm-korrigiert):")
    print("  " + "-" * 70)
    for label, _, _ in pairs:
        t = sharpe_tests[label]
        mark = "✓" if t["significant_holm"] else "✗"
        print(f"  {mark} {label:<20} ΔSharpe={t['diff_annual']:>+.3f}  "
              f"p={t['p_value']:.3f}  p(Holm)={t['p_holm']:.3f}")
    print("\n  DEFLATED SHARPE RATIO (Bailey & López de Prado 2014, "
          f"N={significance_results['n_strategies']} Strategien):")
    print("  " + "-" * 70)
    for col, d in dsr_results.items():
        mark = "✓" if d["significant"] else "✗"
        print(f"  {mark} {col:<16} SR={d['sharpe_annual']:.2f}  "
              f"Hürde SR0={d['sr0_annual']:.2f}  DSR={d['deflated_sr']:.3f}")
    print()

    # ------------------------------------------------------------------
    # F: Visualisierungen — alle Abbildungen als PNG in den Output-Ordner
    #    (die Funktionen sind in plots.py erklärt).
    # ------------------------------------------------------------------
    log.info("Erstelle finale Abbildungen …")

    plot_cumulative_returns(
        returns_df, os.path.join(OUTPUT_DIR, "01_kumulierte_renditen.png"))

    if not weights_mvo.empty:
        plot_weight_heatmap(weights_mvo, "Markowitz MVO (Ledoit-Wolf)",
            os.path.join(OUTPUT_DIR, "02_gewichte_markowitz.png"))

    if not weights_rf.empty:
        plot_weight_heatmap(weights_rf, "Random Forest MVO",
            os.path.join(OUTPUT_DIR, "03_gewichte_random_forest.png"))

    if not weights_rp.empty:
        plot_weight_heatmap(weights_rp, "Risk Parity (ERC)",
            os.path.join(OUTPUT_DIR, "03b_gewichte_risk_parity.png"))

    plot_efficient_frontier(
        mu_hist=mu_hist, cov_ann=cov_ann, tickers=tickers,
        w_mvo=w_mvo_last, w_rf=w_rf_last, w_rp=w_rp_last, w_ew=w_ew,
        mu_rf=mu_rf, rf=RISK_FREE_RATE,
        output_path=os.path.join(OUTPUT_DIR, "04_efficient_frontier.png"),
    )

    plot_performance_metrics(
        metrics_df, os.path.join(OUTPUT_DIR, "05_performance_kennzahlen.png"))

    plot_rolling_sharpe(
        returns_df, os.path.join(OUTPUT_DIR, "06_rollierender_sharpe.png"))

    # Feature Importance (letztes RF-Modell aus Backtest)
    if rfo.best_estimator_ is not None:
        plot_feature_importance(
            rfo, os.path.join(OUTPUT_DIR, "07_feature_importance.png"))

    plot_frontier_evolution(
        frontier_snapshots,
        os.path.join(OUTPUT_DIR, "08_frontier_evolution.png"),
    )

    # NEU v4: Stress-Test
    plot_stress_test(
        returns_df,
        os.path.join(OUTPUT_DIR, "09_stress_test.png"),
    )

    # NEU v4: Turnover vs. Performance
    if not turnover_df.empty:
        plot_turnover_performance(
            turnover_df, returns_df,
            os.path.join(OUTPUT_DIR, "10_turnover_performance.png"),
        )

    # NEU v4: Animiertes GIF
    create_animated_frontier_gif(
        frontier_snapshots,
        os.path.join(OUTPUT_DIR, "11_frontier_animation.gif"),
    )

    # NEU v4: SHAP (optional — nur wenn das shap-Paket installiert ist).
    # Für die SHAP-Analyse werden die Monats-Features noch einmal frisch
    # aufgebaut und alle Aktien mit genug Historie zusammengesammelt:
    if SHAP_AVAILABLE and rfo.best_estimator_ is not None:
        monthly_data_shap = aggregate_to_monthly(indicator_dict, asset_returns, tickers)
        monthly_data_shap = add_cross_sectional_ranks(monthly_data_shap)
        feat_rows_shap, _ = [], []
        for ticker in tickers:
            t_rows = monthly_data_shap[monthly_data_shap["ticker"] == ticker]
            avail  = [c for c in FEATURE_COLS if c in t_rows.columns]
            valid  = t_rows[avail].dropna()
            if len(valid) >= 12:
                feat_rows_shap.append(valid)
        if feat_rows_shap:
            X_for_shap = pd.concat(feat_rows_shap).dropna()
            if not X_for_shap.empty:
                avail_for_shap = [c for c in FEATURE_COLS if c in X_for_shap.columns]
                plot_shap_values(
                    rfo, X_for_shap[avail_for_shap],
                    os.path.join(OUTPUT_DIR, "12_shap_explainability.png"),
                )

    # ------------------------------------------------------------------
    # G: CSV-Export — alle Ergebnistabellen für Excel & Co.
    # ------------------------------------------------------------------
    save_all_csv(returns_df, metrics_df, weights_mvo, weights_rf,
                 weights_rp, turnover_df, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # H: JSON-Experimentprotokoll — Parameter + Ergebnisse zur Reproduktion.
    # ------------------------------------------------------------------
    save_experiment_json(
        metrics_df, significance_results, tickers,
        os.path.join(OUTPUT_DIR, "experiment_log.json"),
    )

    # ------------------------------------------------------------------
    # I: Kompakte Zusammenfassung — das Wichtigste noch einmal ins Terminal.
    # ------------------------------------------------------------------
    total = time.perf_counter() - t0_main
    log.info("=" * 70)
    log.info(f"  FERTIG | Gesamtlaufzeit: {total/60:.1f} min ({total:.0f}s)")
    log.info(f"  Alle Dateien in: {os.path.abspath(OUTPUT_DIR)}/")
    log.info("=" * 70)

    print("=" * 60)
    print("  KURZ-ZUSAMMENFASSUNG (v4 — 4 Strategien)")
    print("=" * 60)
    for strat in metrics_df.index:
        m = metrics_df.loc[strat]
        print(f"\n  [{strat}]")
        print(f"    CAGR          : {m['CAGR (%)']:>7.2f} %")
        print(f"    Gesamtrendite : {m['Gesamtrendite (%)']:>7.2f} %")
        print(f"    Sharpe Ratio  : {m['Sharpe Ratio']:>7.4f}")
        print(f"    Sortino Ratio : {m['Sortino Ratio']:>7.4f}")
        print(f"    Max. Drawdown : {m['Max. Drawdown (%)']:>7.2f} %")
    print("=" * 60)
    print(f"\n  Ausgabe-Ordner: {os.path.abspath(OUTPUT_DIR)}/\n")

    print("\n  SIGNIFIKANZ-ÜBERSICHT (Holm-korrigiert, Ledoit-Wolf 2008):")
    for label, _, _ in pairs:
        t = sharpe_tests[label]
        sig_str = "SIGNIFIKANT (p_Holm < 5%)" if t["significant_holm"] else "n.s."
        print(f"    {label:<35} {sig_str}")
    print()

# Dieser Standard-Baustein sorgt dafür, dass main() nur läuft, wenn die
# Datei direkt gestartet wird — nicht schon beim bloßen Importieren.
if __name__ == "__main__":
    main()
