"""Kennzahlen gegen analytisch bekannte Werte.

FÜR EINSTEIGER — WAS SIND DIESE TESTS?
Die Dateien im Ordner tests/ sind automatische Prüfprogramme ("Unit-Tests").
Jede Funktion, deren Name mit ``test_`` beginnt, wird vom Werkzeug pytest
ausgeführt. Das Grundmuster ist immer gleich:
  1. Ein winziges Beispiel konstruieren, dessen richtiges Ergebnis man von
     Hand ausrechnen kann.
  2. Die Projekt-Funktion darauf loslassen.
  3. Mit ``assert`` ("stelle sicher, dass …") prüfen, ob beides übereinstimmt.
     Stimmt eine assert-Bedingung nicht, schlägt der Test ROT fehl.
So merkt man sofort, wenn eine spätere Code-Änderung versehentlich die
Mathematik kaputt macht. Häufig steht statt ``a == b`` die Form
``abs(a - b) < 1e-9``: Computer runden bei Kommazahlen minimal, deshalb
prüft man "praktisch gleich" statt "exakt gleich".

Diese Datei prüft die Finanz-Kennzahlen aus portfolio/metrics.py.
"""
import numpy as np
import pandas as pd

from portfolio.metrics import (
    cagr, annualized_vol, sharpe_ratio, max_drawdown,
    sortino_ratio, portfolio_perf,
)


def _series(vals):
    """Hilfsfunktion: baut aus einer Zahlenliste eine Rendite-Zeitreihe mit
    Börsentags-Datumsindex (bdate_range = nur Werktage, wie an der Börse)."""
    return pd.Series(vals, index=pd.bdate_range("2020-01-01", periods=len(vals)))


def test_cagr_zero_returns():
    # Ein Jahr lang 0 % Tagesrendite → das Jahreswachstum muss exakt 0 sein.
    assert cagr(_series([0.0] * 252)) == 0.0


def test_cagr_known_value():
    # 252 Handelstage konstantes r -> (1+r)^252 - 1 (genau 1 Jahr)
    # Das Ergebnis lässt sich mit dem Taschenrechner nachprüfen.
    r = 0.001
    expected = (1 + r) ** 252 - 1
    assert abs(cagr(_series([r] * 252)) - expected) < 1e-9


def test_max_drawdown_known():
    # +10 % dann -50 %  ->  maximaler Drawdown -50 %
    # (Der Absturz zählt vom Hoch NACH dem +10%-Tag aus.)
    assert abs(max_drawdown(_series([0.10, -0.50])) - (-0.5)) < 1e-9


def test_max_drawdown_monotonic_up_is_zero():
    # Nur steigende Kurse → es gibt nie einen Rückgang vom Hoch → Drawdown 0.
    assert abs(max_drawdown(_series([0.01] * 10))) < 1e-12


def test_annualized_vol_constant_is_near_zero():
    # konstante Renditen -> Vola ~0 (bis auf Floating-Point-Rauschen)
    # (Volatilität misst SCHWANKUNG — wer immer gleich rentiert, schwankt nicht.)
    assert annualized_vol(_series([0.005] * 100)) < 1e-9


def test_sharpe_zero_vol_guard():
    # Ueberschussrendite konstant -> Standardabweichung 0 -> Sharpe 0.
    # Prueft gezielt den "if sd > 0"-Schutz in sharpe_ratio().
    assert sharpe_ratio(_series([0.0] * 252), rf=0.0) == 0.0


def test_sharpe_penalises_risk_free_shortfall():
    # Eine Strategie, die konstant 0 % liefert, waehrend der sichere Zins 4 %
    # betraegt, hat eine konstant NEGATIVE Ueberschussrendite. Die Streuung
    # ist null, der Schutz greift also weiterhin.
    assert sharpe_ratio(_series([0.0] * 252), rf=0.04) == 0.0


def test_sortino_no_downside_guard():
    # Keine Abweichung unter die Zielrendite -> Downside-Abweichung 0 -> 0.
    assert sortino_ratio(_series([0.0] * 252), rf=0.0) == 0.0


def test_sortino_counts_shortfall_against_target():
    # Liegt die Rendite dauerhaft UNTER dem sicheren Zins, ist das sehr wohl
    # Downside-Risiko: Sortino muss negativ werden, nicht null.
    assert sortino_ratio(_series([0.0] * 252), rf=0.04) < 0


def test_sortino_uses_all_days_not_only_losers():
    # Downside-Abweichung mittelt ueber ALLE Tage. Haengt man an eine Reihe
    # mit wenigen Verlusten viele Gewinntage an, sinkt die Downside-Abweichung
    # und der Sortino-Quotient steigt. Die alte Fassung (Streuung INNERHALB
    # der Verlusttage) reagierte darauf nicht.
    wenig = _series([-0.01] * 5 + [0.01] * 5)
    viel  = _series([-0.01] * 5 + [0.01] * 95)
    assert sortino_ratio(viel, rf=0.0) > sortino_ratio(wenig, rf=0.0)


def test_portfolio_perf_matches_formula():
    # Zwei Aktien, 50/50 gemischt, ohne Korrelation (Nebendiagonale = 0):
    # Rendite = 0.5·10 % + 0.5·20 % = 15 %; Varianz = 0.25·0.04 + 0.25·0.09.
    # Genau das müssen die Matrixformeln in portfolio_perf liefern.
    mu = np.array([0.1, 0.2])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    w = np.array([0.5, 0.5])
    ret, vol, sr = portfolio_perf(w, mu, cov, rf=0.0)
    assert abs(ret - 0.15) < 1e-12
    assert abs(vol - np.sqrt(0.25 * 0.04 + 0.25 * 0.09)) < 1e-12
    assert abs(sr - ret / vol) < 1e-12
