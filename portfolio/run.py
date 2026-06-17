"""Orchestrierung: Daten -> Indikatoren -> Backtest -> Kennzahlen -> Plots."""

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

def main():
    t0_main = time.perf_counter()
    log.info("=" * 70)
    log.info("  PORTFOLIO-OPTIMIERUNG V4 | MARKOWITZ vs. RF vs. EW vs. RISK PARITY")
    log.info("  Neu v4: Risk Parity | CS-Ranking | Live-Dashboard | SHAP | GIF")
    log.info(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    if not YFINANCE_AVAILABLE:
        log.error("yfinance fehlt. Bitte: pip install yfinance")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info(f"Output-Ordner: {os.path.abspath(OUTPUT_DIR)}")
    log.info(f"Interaktives Display: {INTERACTIVE_DISPLAY} | "
             f"SHAP: {SHAP_AVAILABLE} | Animation: {ANIMATION_AVAILABLE}")

    # ------------------------------------------------------------------
    # A: Daten laden
    # ------------------------------------------------------------------
    asset_prices, asset_returns, spy_returns, tickers = download_data(
        TICKERS, SPY_TICKER, START_DATE, END_DATE
    )

    # ------------------------------------------------------------------
    # B: Technische Indikatoren
    # ------------------------------------------------------------------
    indicator_dict = build_all_indicators(
        asset_prices, asset_returns, spy_returns, tickers
    )

    # ------------------------------------------------------------------
    # C: Backtest (mit Live-Dashboard)
    # ------------------------------------------------------------------
    results = run_backtest(
        asset_prices, asset_returns, spy_returns, tickers, indicator_dict
    )

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
    w_ew               = np.ones(len(tickers)) / len(tickers)

    # ------------------------------------------------------------------
    # D: Kennzahlen
    # ------------------------------------------------------------------
    metrics_df = compute_metrics(returns_df)
    print("\n" + "=" * 70)
    print("  PERFORMANCE-KENNZAHLEN — VOLLSTÄNDIGE ÜBERSICHT (v4)")
    print("=" * 70)
    print(metrics_df.to_string())
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # E: Bootstrap-Signifikanztests (v4: jetzt aufgerufen!)
    # ------------------------------------------------------------------
    log.info("Führe Bootstrap-Signifikanztests durch (Bailey et al. 2014) …")
    bootstrap_results = {}
    r_mvo = returns_df["Markowitz MVO"]
    r_rf  = returns_df["Random Forest"]
    r_rp  = returns_df["Risk Parity"]
    r_ew  = returns_df["Equal Weight"]

    pairs = [
        ("RF vs. MVO",        r_rf,  r_mvo),
        ("RF vs. EW",         r_rf,  r_ew),
        ("MVO vs. EW",        r_mvo, r_ew),
        ("Risk Parity vs. EW",r_rp,  r_ew),
        ("MVO vs. RP",        r_mvo, r_rp),
    ]
    for label, ra, rb in pairs:
        res = bootstrap_paired_test(ra, rb, n_bootstrap=500, metric="sharpe")
        bootstrap_results[label] = res
        sig = "✓ SIGNIFIKANT (p < 0.05)" if res["significant"] else "✗ nicht signifikant"
        log.info(
            f"  {label:<30} | ΔSharpe: {res['observed_diff']:>+.4f} | "
            f"p = {res['p_value']:.4f} | {sig}"
        )

    print("\n  BOOTSTRAP-SIGNIFIKANZTESTS (Sharpe-Vergleich, 500 Resamples):")
    print("  " + "-" * 60)
    for label, res in bootstrap_results.items():
        sig = "✓" if res["significant"] else "✗"
        print(f"  {sig} {label:<30} ΔSharpe={res['observed_diff']:>+.4f}  "
              f"p={res['p_value']:.4f}  "
              f"95%-KI: [{res['ci_95'][0]:>+.4f}, {res['ci_95'][1]:>+.4f}]")
    print()

    # ------------------------------------------------------------------
    # F: Visualisierungen
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

    # NEU v4: SHAP (optional)
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
    # G: CSV-Export
    # ------------------------------------------------------------------
    save_all_csv(returns_df, metrics_df, weights_mvo, weights_rf,
                 weights_rp, turnover_df, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # H: JSON-Experimentprotokoll
    # ------------------------------------------------------------------
    save_experiment_json(
        metrics_df, bootstrap_results, tickers,
        os.path.join(OUTPUT_DIR, "experiment_log.json"),
    )

    # ------------------------------------------------------------------
    # I: Kompakte Zusammenfassung
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

    print("\n  BOOTSTRAP-SIGNIFIKANZ ÜBERSICHT:")
    for label, res in bootstrap_results.items():
        sig_str = "SIGNIFIKANT (p < 5%)" if res["significant"] else "n.s."
        print(f"    {label:<35} {sig_str}")
    print()

if __name__ == "__main__":
    main()
