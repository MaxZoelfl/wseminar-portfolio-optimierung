"""
Rollierender Walk-Forward-Backtest mit monatlichem Rebalancing.

FÜR EINSTEIGER — WAS MACHT DIESE DATEI?
Dies ist das Herzstück des Experiments: die Zeitmaschinen-Simulation.
Ein "Backtest" spielt durch, wie sich die vier Strategien entwickelt HÄTTEN,
wenn man sie von 2015 bis 2024 real angewendet hätte.

"Walk-Forward" (vorwärtsschreitend) bedeutet dabei:
  - Wir stellen uns an ein Monatsende, z. B. den 31.01.2015.
  - Jede Strategie darf NUR die letzten 3 Jahre Vergangenheit sehen
    (das "Trainingsfenster") und entscheidet damit ihre Depotgewichte.
  - Dann wird einen Monat "gehalten" und die tatsächlich eingetretenen
    Renditen des Februars gutgeschrieben (abzüglich Handelskosten).
  - Einen Monat weiterrücken, alles wiederholen — ca. 120-mal bis Ende 2024.

Der springende Punkt: Zu keinem Zeitpunkt fließt Wissen über die Zukunft in
die Entscheidungen ein (kein "Look-Ahead-Bias"). Genau daran scheitern viele
unsaubere Backtests — deshalb stecken unten mehrere explizite Schutzvorkehrungen
im Code (siehe die Kommentare "v3-FIX" und den Abschnitt Transaktionskosten).

Weitere Begriffe:
  - "Rebalancing": das monatliche Umschichten des Depots auf die neuen Zielgewichte.
  - "Turnover" (Handelsumsatz): wie viel Prozent des Depots dabei umgeschlagen
    werden — daran hängen die Transaktionskosten.
  - "Halteperiode": die Handelstage zwischen zwei Rebalancing-Terminen.
"""

import os
import time
import numpy as np
import pandas as pd
from .config import *
from .metrics import *
from .indicators import *
from .optimizers import *
from .dashboard import *

def run_backtest(asset_prices: pd.DataFrame,
                 asset_returns: pd.DataFrame,
                 spy_returns: pd.Series,
                 tickers: list,
                 indicator_dict: dict) -> dict:
    """
    Rollierender Backtest mit monatlichem Rebalancing.

    Ablauf je Rebalancing-Monat:
      1. Trainingsdaten extrahieren (letzte TRAIN_YEARS Jahre)
      2. Markowitz MVO: Ledoit-Wolf Sigma + Sharpe-Maximierung
      3. Risk Parity: Equal Risk Contribution (NEU v4)
      4. Random Forest: Features → Tuning → Vorhersage → Sharpe-Maximierung
      5. Equal Weight: 1/N
      6. Portfoliorenditen der Halteperiode berechnen (inkl. Transaktionskosten)
      7. Live-Dashboard aktualisieren

    Rückgabe: ein dict (Sammelbehälter) mit allen Ergebnissen — Renditereihen,
    Gewichtsverläufen, Turnover-Protokoll usw. — für Kennzahlen und Plots.
    """
    # ---- Vorbereitung: Tagesdaten zu Monats-Features verdichten ----------
    log.info("Aggregiere Features täglich → monatlich …")
    monthly_data = aggregate_to_monthly(indicator_dict, asset_returns, tickers)
    monthly_data = add_cross_sectional_ranks(monthly_data)    # v4: CS-Ranks

    # Liste aller Monatsenden, ab dem konfigurierten Backtest-Start (2015).
    backtest_months = monthly_data.index.unique()
    backtest_months = backtest_months[backtest_months >= BACKTEST_START]
    n               = len(tickers)

    # Die drei "aktiven" Optimierer einmal anlegen (Equal Weight braucht
    # keinen — seine Gewichte sind immer 1/N).
    mvo = MarkowitzLedoitWolf()
    rp  = RiskParityPortfolio()   # NEU v4
    rfo = RFPortfolioOptimizer()

    # Sammel-Listen, die im Lauf der Schleife gefüllt werden:
    results_mvo, results_rf, results_ew, results_rp = [], [], [], []  # Tagesrenditen
    hist_w_mvo, hist_w_rf, hist_w_rp               = [], [], []       # Gewichte je Monat
    frontier_snapshots                              = []              # Effizienzlinien-Fotos
    turnover_log                                    = []   # v4: Turnover je Step

    w_prev_rf = np.ones(n) / n   # Start: Equal Weight

    # Wachstumsfaktoren der jeweils letzten Halteperiode (für driftbewussten
    # Turnover, Erklärung unten im Kosten-Abschnitt). None = es gibt noch
    # keine Vorperiode.
    prev_hold_growth = None

    # RF-Tuning-Zähler (Performance-Hebel RF_RETUNE_EVERY) und Index des letzten
    # Backtest-Schritts (für gezielte Dashboard-Updates).
    rf_tune_count = 0
    _last_i = len(backtest_months) - 2

    # Live-Dashboard initialisieren (zeigt den Fortschritt in einem Fenster;
    # auf Rechnern ohne Display deaktiviert es sich selbst).
    dashboard = LiveDashboard(tickers, len(backtest_months) - 1)

    log.info(f"Starte Backtest: {len(backtest_months)} Rebalancing-Monate …")

    # ======================================================================
    # DIE HAUPTSCHLEIFE: ein Durchlauf = ein Rebalancing-Monat.
    # "[:-1]" lässt den letzten Monat aus, weil auf ihn kein Haltemonat mehr
    # folgt, dessen Rendite man messen könnte. enumerate liefert zusätzlich
    # den Zähler i (0, 1, 2, …).
    # ======================================================================
    for i, month_end in enumerate(backtest_months[:-1]):
        t0 = time.perf_counter()   # Stoppuhr für die Schrittdauer

        # ---- 1) Trainingsfenster ausschneiden -----------------------------
        # Alles zwischen (month_end − 3 Jahre) und month_end — mehr darf
        # keine Strategie sehen.
        train_start   = month_end - pd.DateOffset(years=TRAIN_YEARS)
        train_daily   = asset_returns[
            (asset_returns.index >= train_start) &
            (asset_returns.index <= month_end)
        ]
        train_monthly = monthly_data[
            (monthly_data.index >= train_start) &
            (monthly_data.index <= month_end)
        ]

        # Qualitätskontrolle: mindestens ~1 Jahr Tagesdaten und 24 Monate
        # Feature-Historie, sonst wäre jede Schätzung unseriös → überspringen.
        if len(train_daily) < 252 or len(train_monthly.index.unique()) < 24:
            log.warning(f"  [{month_end.date()}] Zu wenige Daten, überspringe.")
            continue

        # Die Halteperiode: alle Handelstage vom Tag NACH dem Rebalancing bis
        # zum nächsten Monatsende. Deren Renditen entscheiden über Erfolg.
        next_month  = backtest_months[i + 1]
        hold_period = asset_returns[
            (asset_returns.index > month_end) &
            (asset_returns.index <= next_month)
        ]
        if hold_period.empty:
            continue

        # ---- Kovarianzmatrix (Ledoit-Wolf) --------------------------------
        # Gemeinsame Risiko-Schätzgrundlage für MVO, Risk Parity und RF.
        cov_ann = mvo.estimate_covariance(train_daily)
        # Historische Durchschnittsrenditen, auf Jahresbasis (×252):
        # die Renditeschätzung der KLASSISCHEN Markowitz-Strategie.
        mu_hist = train_daily.mean().values * 252

        # ---- 2) Markowitz MVO ---------------------------------------------
        # try/except als Sicherheitsnetz: Falls der Optimierer in einem Monat
        # scheitert (selten, z. B. keine Konvergenz), wird neutral auf 1/N
        # ausgewichen statt den ganzen Backtest abzubrechen.
        try:
            w_mvo = mvo.max_sharpe(mu_hist, cov_ann)
        except Exception as e:
            log.warning(f"  MVO-Fehler {month_end.date()}: {e}")
            w_mvo = np.ones(n) / n

        # ---- 3) Risk Parity (NEU v4) ---------------------------------------
        try:
            w_rp = rp.optimize(cov_ann)
        except Exception as e:
            log.warning(f"  RP-Fehler {month_end.date()}: {e}")
            w_rp = np.ones(n) / n

        # ---- 4) Random Forest ----------------------------------------------
        # v3-FIX: Letzter Trainingsmonat ausgeschlossen (Look-Ahead-Bias).
        # Grund: Das Label ("target_next_month") des letzten Trainingsmonats
        # ist die Rendite GENAU des Monats, den wir gleich vorhersagen wollen.
        # Ihn im Training zu lassen hieße, die Lösung vorab zu verraten.
        last_train_month = month_end - pd.DateOffset(months=1)
        feat_rows, target_rows = [], []
        for ticker in tickers:
            # Trainingszeilen dieser Aktie bis einschließlich Vormonat:
            t_rows = train_monthly[
                (train_monthly["ticker"] == ticker) &
                (train_monthly.index <= last_train_month)
            ]
            # Nur vollständige Zeilen (Features + Zielwert vorhanden) verwenden:
            valid = t_rows[[c for c in FEATURE_COLS if c in t_rows.columns]
                           + ["target_next_month"]].dropna()
            if len(valid) < 12:
                continue   # unter 12 Monaten Historie: Aktie diesmal auslassen
            avail_feats = [c for c in FEATURE_COLS if c in valid.columns]
            feat_rows.append(valid[avail_feats])
            target_rows.append(valid["target_next_month"])

        if len(feat_rows) < n // 2:
            # Zu wenige Aktien mit brauchbaren Daten → RF diesmal neutral:
            # Gleichgewichtung und historische Renditen als Ersatz.
            w_rf  = np.ones(n) / n
            mu_rf = mu_hist.copy()
        else:
            # Trainingsdaten aller Aktien zu einer großen Tabelle stapeln
            # ("gepoolt": EIN Modell lernt aktienübergreifende Muster).
            X_train_rf = pd.concat(feat_rows)
            y_train_rf = pd.concat(target_rows)
            try:
                # Hyperparameter nur alle RF_RETUNE_EVERY Monate neu suchen;
                # dazwischen das Modell nur auf das aktuelle Fenster refitten.
                # (Default RF_RETUNE_EVERY=1 → jeden Monat volle Suche.)
                if rf_tune_count % RF_RETUNE_EVERY == 0 or rfo.best_estimator_ is None:
                    rfo.fit_with_tuning(X_train_rf, y_train_rf,
                                        sample_times=X_train_rf.index)
                else:
                    rfo.refit(X_train_rf, y_train_rf)
                rf_tune_count += 1
                # Prognose-Eingabe bauen: je Aktie die JÜNGSTE vollständige
                # Merkmalszeile (= Zustand am aktuellen Monatsende).
                X_current_rows = []
                for ticker in tickers:
                    t_rows = train_monthly[train_monthly["ticker"] == ticker]
                    avail  = [c for c in FEATURE_COLS if c in t_rows.columns]
                    valid  = t_rows[avail].dropna()
                    if len(valid) > 0:
                        X_current_rows.append(valid.iloc[-1])
                    else:
                        # Notlösung bei komplett fehlenden Daten: neutrale Nullen
                        # (nach der Skalierung = "durchschnittliche Aktie").
                        X_current_rows.append(
                            pd.Series(np.zeros(len(avail)), index=avail)
                        )
                X_current = pd.DataFrame(X_current_rows)
                # Sicherstellen, dass alle Trainingsspalten vorhanden sind —
                # das Modell erwartet exakt dieselben Spalten in derselben
                # Reihenfolge wie beim Training.
                for col in X_train_rf.columns:
                    if col not in X_current.columns:
                        X_current[col] = 0.0
                X_current = X_current[X_train_rf.columns]
                # KI-Prognose der Monatsrenditen (annualisiert) → damit dann
                # dieselbe Sharpe-Maximierung wie bei Markowitz:
                mu_rf = rfo.predict_monthly_returns(X_current)
                w_rf  = rfo.optimize(mu_rf, cov_ann, w_prev=w_prev_rf)
            except Exception as e:
                log.warning(f"  RF-Fehler {month_end.date()}: {e}")
                w_rf  = np.ones(n) / n
                mu_rf = mu_hist.copy()

        # ---- Efficient Frontier Snapshot (jeden 3. Monat) ----------------
        # Nur für die Abbildungen 8/11: gelegentlich ein "Foto" der aktuellen
        # Effizienzlinie aufheben. Hat keinerlei Einfluss auf die Ergebnisse.
        if i % 3 == 0:
            try:
                snap = mvo.efficient_frontier(mu_hist, cov_ann, n_points=60,
                                              max_weight=MAX_WEIGHT)
                if not snap.empty:
                    snap["date"]   = month_end
                    snap["w_tang"] = [w_mvo] * len(snap)
                    frontier_snapshots.append(snap)
            except Exception:
                pass

        # ---- 5) Equal Weight ----------------------------------------------
        # Die Vergleichs-Basislinie: stur 1/N in jede Aktie, jeden Monat.
        w_ew = np.ones(n) / n

        # ---- 6a) Transaktionskosten ----------------------------------------
        # Turnover = 0.5 * Σ|w_neu − w_alt|. Als "w_alt" werden die über die
        # letzte Halteperiode GEDRIFTETEN Vorgängergewichte verwendet (nicht
        # die ursprünglichen Zielgewichte). "Drift" heißt: Auch ohne Handel
        # verschieben sich die Gewichte, weil die Aktien unterschiedlich
        # gestiegen/gefallen sind (was gut lief, macht nun automatisch mehr
        # vom Depot aus). Dadurch erhält auch Equal Weight einen realistischen
        # Rebalancing-Turnover (Rückführung der Kursdrift auf 1/N), und alle
        # Strategien werden konsistent und FAIR bewertet.
        if i == 0 or prev_hold_growth is None:
            turnover_mvo = 1.0   # Erstaufbau des Portfolios: 100 % Kaufumsatz
            turnover_rf  = 1.0
            turnover_rp  = 1.0
            turnover_ew  = 1.0
        else:
            def _drift(w_prev):
                # Alte Gewichte mit den Wachstumsfaktoren der Halteperiode
                # fortschreiben und wieder auf Summe 1 normieren.
                wd = np.asarray(w_prev, dtype=float) * prev_hold_growth
                s  = wd.sum()
                return wd / s if s > 0 else np.asarray(w_prev, dtype=float)
            prev_w_mvo = _drift(hist_w_mvo[-1].values) if hist_w_mvo else np.ones(n) / n
            prev_w_rf  = _drift(hist_w_rf[-1].values)  if hist_w_rf  else np.ones(n) / n
            prev_w_rp  = _drift(hist_w_rp[-1].values)  if hist_w_rp  else np.ones(n) / n
            prev_w_ew  = _drift(np.ones(n) / n)
            # Halbe Absolutsumme, weil jeder Verkauf zugleich ein Kauf ist.
            turnover_mvo = np.abs(w_mvo - prev_w_mvo).sum() / 2
            turnover_rf  = np.abs(w_rf  - prev_w_rf ).sum() / 2
            turnover_rp  = np.abs(w_rp  - prev_w_rp ).sum() / 2
            turnover_ew  = np.abs(w_ew  - prev_w_ew ).sum() / 2

        # Kosten = Umsatzanteil × Kostensatz (0,10 % laut config).
        cost_mvo = turnover_mvo * TRANSACTION_COST
        cost_rf  = turnover_rf  * TRANSACTION_COST
        cost_rp  = turnover_rp  * TRANSACTION_COST
        cost_ew  = turnover_ew  * TRANSACTION_COST

        # ---- 6b) Portfoliorenditen der Halteperiode ------------------------
        # Tagesrendite des Portfolios = gewichtete Summe der Aktienrenditen
        # (hold_period * w).sum(axis=1): jede Spalte mit ihrem Gewicht
        # multiplizieren, dann zeilenweise (= je Tag) aufsummieren.
        ret_mvo = (hold_period * w_mvo).sum(axis=1).copy()
        ret_rf  = (hold_period * w_rf ).sum(axis=1).copy()
        ret_rp  = (hold_period * w_rp ).sum(axis=1).copy()
        ret_ew  = (hold_period * w_ew ).sum(axis=1).copy()

        # Die Handelskosten werden am ersten Tag der Halteperiode abgezogen —
        # genau dann fällt der Depotumbau ja an.
        if len(ret_mvo) > 0:
            ret_mvo.iloc[0] -= cost_mvo
            ret_rf.iloc[0]  -= cost_rf
            ret_rp.iloc[0]  -= cost_rp
            ret_ew.iloc[0]  -= cost_ew

        # Monatsergebnisse an die Sammel-Listen anhängen:
        results_mvo.append(ret_mvo)
        results_rf.append( ret_rf)
        results_rp.append( ret_rp)
        results_ew.append( ret_ew)

        hist_w_mvo.append(pd.Series(w_mvo, index=tickers, name=month_end))
        hist_w_rf.append( pd.Series(w_rf,  index=tickers, name=month_end))
        hist_w_rp.append( pd.Series(w_rp,  index=tickers, name=month_end))

        # Turnover-Log für Scatter-Plot (Abbildung 10)
        turnover_log.append({
            "date"         : month_end,
            "turnover_mvo" : turnover_mvo,
            "turnover_rf"  : turnover_rf,
            "turnover_rp"  : turnover_rp,
            "turnover_ew"  : turnover_ew,
        })

        # Zustände für die nächste Runde merken:
        w_prev_rf = w_rf.copy()

        # Wachstumsfaktoren dieser Halteperiode je Asset (prod(1+r)) für den
        # driftbewussten Turnover der nächsten Iteration sichern.
        prev_hold_growth = (1 + hold_period).prod(axis=0).reindex(tickers).to_numpy()

        # ---- 7) Live Dashboard aktualisieren --------------------------------
        # Nur rendern, wenn das Dashboard aktiv ist und der Schritt fällig ist
        # (alle DASHBOARD_UPDATE_EVERY Schritte sowie im letzten Schritt, damit
        # das gespeicherte PNG den Endzustand zeigt). Spart die wachsende
        # concat-Aggregation in headless-Läufen. Default 1 → jeder Schritt.
        if dashboard.active and (i % DASHBOARD_UPDATE_EVERY == 0 or i == _last_i):
            returns_so_far = pd.DataFrame({
                "Markowitz MVO" : pd.concat(results_mvo).sort_index(),
                "Random Forest" : pd.concat(results_rf ).sort_index(),
                "Equal Weight"  : pd.concat(results_ew ).sort_index(),
                "Risk Parity"   : pd.concat(results_rp ).sort_index(),
            })
            dashboard.update(i + 1, month_end, returns_so_far, w_mvo, w_rf)

        # Fortschrittszeile ins Logbuch (Monat, MVO-Sharpe, Turnover, Dauer):
        elapsed = time.perf_counter() - t0
        ret_m, vol_m, sr_m = portfolio_perf(w_mvo, mu_hist, cov_ann)
        log.info(
            f"  [{i+1:>3}/{len(backtest_months)-1}] {month_end.date()} | "
            f"MVO-SR: {sr_m:.3f} | "
            f"TO-MVO: {turnover_mvo*100:.1f}% | "
            f"TO-RF: {turnover_rf*100:.1f}% | "
            f"{elapsed:.1f}s"
        )

    # ======================================================================
    # NACH DER SCHLEIFE: Aufräumen und alles zusammenpacken.
    # ======================================================================
    # Dashboard finalisieren (letzten Stand als PNG sichern, Fenster schließen)
    dashboard.finalize(os.path.join(OUTPUT_DIR, "00_live_dashboard_final.png"))
    dashboard.close()

    # Alle Monats-Stücke zu durchgehenden Tagesrendite-Reihen verketten —
    # eine Spalte je Strategie, über den gesamten Zeitraum 2015–2024.
    returns_df = pd.DataFrame({
        "Markowitz MVO" : pd.concat(results_mvo).sort_index(),
        "Random Forest" : pd.concat(results_rf ).sort_index(),
        "Equal Weight"  : pd.concat(results_ew ).sort_index(),
        "Risk Parity"   : pd.concat(results_rp ).sort_index(),
    })

    # Gewichtsverläufe als Tabellen (Zeilen = Aktien, Spalten = Monate):
    weights_mvo_df = pd.DataFrame(hist_w_mvo).T
    weights_rf_df  = pd.DataFrame(hist_w_rf ).T
    weights_rp_df  = pd.DataFrame(hist_w_rp ).T
    turnover_df    = pd.DataFrame(turnover_log).set_index("date")

    # Alles in einem dict zurückgeben; die Einträge mit "_last" bzw. mu/cov
    # stammen aus dem LETZTEN Monat (für die Momentaufnahme-Plots wie die
    # Efficient Frontier).
    return {
        "returns"            : returns_df,
        "weights_mvo"        : weights_mvo_df,
        "weights_rf"         : weights_rf_df,
        "weights_rp"         : weights_rp_df,
        "turnover_df"        : turnover_df,
        "mu_hist"            : mu_hist,
        "mu_rf"              : mu_rf,
        "cov_ann"            : cov_ann,
        "w_mvo_last"         : w_mvo,
        "w_rf_last"          : w_rf,
        "w_rp_last"          : w_rp,
        "frontier_snapshots" : frontier_snapshots,
        "rfo"                : rfo,     # RF-Objekt für SHAP / Feature Importance
    }
