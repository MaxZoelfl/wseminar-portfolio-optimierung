"""Constraints und Eigenschaften der Portfolio-Optimierer.

FÜR EINSTEIGER: Automatische Prüfprogramme (Erklärung des Testprinzips im
Kopf von tests/test_metrics.py). Diese Datei prüft die Strategie-"Rezepte"
aus portfolio/optimizers.py: Halten die Optimierer ihre Spielregeln ein
(voll investiert, keine Leerverkäufe, 20-%-Obergrenze, Turnover-Limit)?
Da die exakten optimalen Gewichte kaum von Hand nachzurechnen sind, prüfen
die Tests EIGENSCHAFTEN der Lösung statt konkreter Zahlen.
"""
import numpy as np

from portfolio.optimizers import MarkowitzLedoitWolf, RiskParityPortfolio
from portfolio.config import MAX_WEIGHT


def _spd_cov(n, seed=0):
    """Symmetrisch positiv definite Kovarianzmatrix.

    Hilfsfunktion: erzeugt eine zufällige, aber mathematisch GÜLTIGE
    Kovarianzmatrix. (Der Trick A·Aᵀ garantiert die nötigen Eigenschaften
    "symmetrisch" und "positiv definit" — jede echte Kovarianzmatrix hat
    sie; das kleine +0.01 auf der Diagonale stabilisiert numerisch.)
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T / n + np.eye(n) * 0.01


def test_markowitz_weights_valid():
    # Prüft die drei Grundregeln jeder Markowitz-Lösung. Die winzigen
    # Toleranzen (1e-6 etc.) erlauben unvermeidliche Rundungsreste des Lösers.
    n = 8
    rng = np.random.default_rng(1)
    mu = rng.normal(0.10, 0.05, n)
    w = MarkowitzLedoitWolf().max_sharpe(mu, _spd_cov(n))
    assert abs(w.sum() - 1.0) < 1e-6          # voll investiert
    assert (w >= -1e-9).all()                 # long-only
    assert (w <= MAX_WEIGHT + 1e-6).all()     # Positionsobergrenze


def test_markowitz_turnover_constraint():
    # Startdepot = Gleichgewichtung; erlaubter Umbau höchstens 10 %.
    # Die neue Lösung darf also maximal 10 % vom alten Depot abweichen.
    n = 6
    rng = np.random.default_rng(4)
    mu = rng.normal(0.10, 0.05, n)
    cov = _spd_cov(n, 2)
    w_prev = np.ones(n) / n
    w = MarkowitzLedoitWolf().max_sharpe(mu, cov, w_prev=w_prev, turnover_limit=0.10)
    turnover = np.abs(w - w_prev).sum() / 2
    assert turnover <= 0.10 + 1e-4


def _rc_dispersion(w, cov):
    """Variationskoeffizient der Risikobeiträge (0 = perfekte Risk Parity).

    Misst, wie UNGLEICH die Risikobeiträge der Aktien sind (Streuung geteilt
    durch Mittelwert). Je kleiner, desto "risiko-gleicher" das Portfolio.
    """
    port_vol = np.sqrt(w @ cov @ w)
    rc = w * (cov @ w) / port_vol
    return rc.std() / rc.mean()


def test_risk_parity_more_equal_than_equal_weight():
    # Kern-Eigenschaft von Risk Parity: Seine Risikobeiträge müssen
    # gleichmäßiger sein als bei naiver Geld-Gleichverteilung (1/N).
    n = 8
    cov = _spd_cov(n, seed=3)
    w_rp = RiskParityPortfolio().optimize(cov)
    assert abs(w_rp.sum() - 1.0) < 1e-6
    assert (w_rp > 0).all()
    # Risk Parity gleicht die Risikobeiträge deutlich stärker an als naive 1/N.
    w_ew = np.ones(n) / n
    assert _rc_dispersion(w_rp, cov) < _rc_dispersion(w_ew, cov)


def test_rf_refit_keeps_params_and_predicts():
    # Performance-Hebel: refit nutzt zuletzt getunte Parameter ohne neue Suche.
    # Ablauf: einmal voll tunen (fit_with_tuning), dann auf neuen Daten nur
    # refitten — die gemerkten Einstellungen dürfen sich dabei NICHT ändern,
    # und das Modell muss anschließend gültige (endliche) Prognosen liefern.
    import pandas as pd
    from portfolio.optimizers import RFPortfolioOptimizer
    rng = np.random.default_rng(0)
    cols = ["f1", "f2", "f3"]
    X1 = pd.DataFrame(rng.normal(size=(120, 3)), columns=cols)
    y1 = pd.Series(rng.normal(size=120))
    o = RFPortfolioOptimizer(n_iter=4, cv_splits=3)   # klein gehalten: Test soll schnell sein
    o.fit_with_tuning(X1, y1)
    params = dict(o.best_params_)
    X2 = pd.DataFrame(rng.normal(size=(120, 3)), columns=cols)
    y2 = pd.Series(rng.normal(size=120))
    o.refit(X2, y2)
    assert o.best_params_ == params                 # refit ändert Params nicht
    pred = o.predict_monthly_returns(X2.iloc[:5])
    assert pred.shape == (5,) and np.isfinite(pred).all()


def test_rf_refit_falls_back_without_tuning():
    # Ohne vorheriges Tuning fällt refit auf die volle Suche zurück.
    # (Sicherheitsnetz: refit direkt als Erstes aufzurufen darf nicht crashen.)
    import pandas as pd
    from portfolio.optimizers import RFPortfolioOptimizer
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.normal(size=120))
    o = RFPortfolioOptimizer(n_iter=4, cv_splits=3)
    o.refit(X, y)
    assert o.best_estimator_ is not None


def test_efficient_frontier_nonempty_and_sorted():
    # Die Effizienzlinie muss Punkte liefern, die erwarteten Spalten haben
    # und von links (wenig Rendite) nach rechts (viel Rendite) sortiert sein.
    n = 5
    rng = np.random.default_rng(2)
    mu = rng.normal(0.10, 0.05, n)
    fr = MarkowitzLedoitWolf().efficient_frontier(mu, _spd_cov(n, 5), n_points=20)
    assert len(fr) > 0
    assert {"ret", "vol", "sr"}.issubset(fr.columns)
    # Zielrenditen monoton steigend
    assert (fr["ret"].diff().dropna() >= -1e-9).all()
