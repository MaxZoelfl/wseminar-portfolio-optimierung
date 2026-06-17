"""Kennzahlen gegen analytisch bekannte Werte."""
import numpy as np
import pandas as pd

from portfolio.metrics import (
    cagr, annualized_vol, sharpe_ratio, max_drawdown,
    sortino_ratio, portfolio_perf,
)


def _series(vals):
    return pd.Series(vals, index=pd.bdate_range("2020-01-01", periods=len(vals)))


def test_cagr_zero_returns():
    assert cagr(_series([0.0] * 252)) == 0.0


def test_cagr_known_value():
    # 252 Handelstage konstantes r -> (1+r)^252 - 1 (genau 1 Jahr)
    r = 0.001
    expected = (1 + r) ** 252 - 1
    assert abs(cagr(_series([r] * 252)) - expected) < 1e-9


def test_max_drawdown_known():
    # +10 % dann -50 %  ->  maximaler Drawdown -50 %
    assert abs(max_drawdown(_series([0.10, -0.50])) - (-0.5)) < 1e-9


def test_max_drawdown_monotonic_up_is_zero():
    assert abs(max_drawdown(_series([0.01] * 10))) < 1e-12


def test_annualized_vol_constant_is_near_zero():
    # konstante Renditen -> Vola ~0 (bis auf Floating-Point-Rauschen)
    assert annualized_vol(_series([0.005] * 100)) < 1e-9


def test_sharpe_zero_vol_guard():
    # Vola exakt 0 -> Sharpe 0 (Division-durch-0-Schutz)
    assert sharpe_ratio(_series([0.0] * 252)) == 0.0


def test_sortino_no_downside_guard():
    # keine negativen Renditen -> Downside-Vola 0/NaN -> Sortino 0 (Schutz)
    assert sortino_ratio(_series([0.0] * 252)) == 0.0


def test_portfolio_perf_matches_formula():
    mu = np.array([0.1, 0.2])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    w = np.array([0.5, 0.5])
    ret, vol, sr = portfolio_perf(w, mu, cov, rf=0.0)
    assert abs(ret - 0.15) < 1e-12
    assert abs(vol - np.sqrt(0.25 * 0.04 + 0.25 * 0.09)) < 1e-12
    assert abs(sr - ret / vol) < 1e-12
