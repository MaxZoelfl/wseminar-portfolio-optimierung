"""Performance-Kennzahlen, Hilfsfunktionen (Timer) und Signifikanztests."""

import time
import numpy as np
import pandas as pd
from .config import *

def timer(func):
    """Dekorator: misst und protokolliert Laufzeit."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        log.info(f"[TIMER] {func.__name__:<40} {time.perf_counter()-t0:>7.2f}s")
        return result
    return wrapper


def cagr(returns: pd.Series, freq: int = 252) -> float:
    """Compound Annual Growth Rate aus täglichen Renditen."""
    n = len(returns) / freq
    return (1 + returns).prod() ** (1 / n) - 1 if n > 0 else 0.0


def annualized_vol(returns: pd.Series, freq: int = 252) -> float:
    return returns.std() * np.sqrt(freq)


def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE,
                 freq: int = 252) -> float:
    r = cagr(returns, freq)
    v = annualized_vol(returns, freq)
    return (r - rf) / v if v > 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    cum      = (1 + returns).cumprod()
    roll_max = cum.cummax()
    dd       = (cum - roll_max) / roll_max
    return dd.min()


def calmar_ratio(returns: pd.Series, freq: int = 252) -> float:
    mdd = abs(max_drawdown(returns))
    return cagr(returns, freq) / mdd if mdd > 0 else 0.0


def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE,
                  freq: int = 252) -> float:
    r        = cagr(returns, freq)
    down     = returns[returns < 0]
    down_vol = down.std() * np.sqrt(freq)
    return (r - rf) / down_vol if down_vol > 0 else 0.0


def portfolio_perf(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                   rf: float = RISK_FREE_RATE):
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sr  = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sr


def bootstrap_paired_test(returns_a: pd.Series, returns_b: pd.Series,
                           n_bootstrap: int = 500,
                           metric: str = "sharpe") -> dict:
    """
    Bootstrap-Paarvergleich zweier Portfoliostrategien.
    H0: Strategie A ist nicht besser als Strategie B (gemessen an Sharpe/CAGR).
    Methodik: Bailey, D.H., et al. (2014). The Probability of Backtest Overfitting.

    Parameter:
      n_bootstrap : Anzahl Resampling-Durchläufe (500 genügt für p < 0.05)
      metric      : "sharpe" oder "cagr"

    Rückgabe:
      p_value     : Wahrscheinlichkeit, dass Differenz durch Zufall entstanden
      significant : True wenn p < 0.05
      ci_95       : 95%-Konfidenzintervall der Differenz
    """
    aligned = pd.DataFrame({"a": returns_a, "b": returns_b}).dropna()
    ra, rb  = aligned["a"].values, aligned["b"].values
    n       = len(ra)

    def _metric(r):
        s = pd.Series(r)
        if metric == "sharpe":
            v = r.std() * np.sqrt(252)
            return (cagr(s) - RISK_FREE_RATE) / v if v > 0 else 0.0
        return cagr(s)

    observed_diff   = _metric(ra) - _metric(rb)
    bootstrap_diffs = []

    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx  = rng.integers(0, n, n)
        diff = _metric(ra[idx]) - _metric(rb[idx])
        bootstrap_diffs.append(diff)

    bd      = np.array(bootstrap_diffs)
    p_value = float((bd <= 0).mean() if observed_diff > 0 else (bd >= 0).mean())
    ci_low  = float(np.percentile(bd, 2.5))
    ci_high = float(np.percentile(bd, 97.5))

    return {
        "observed_diff" : round(observed_diff, 6),
        "p_value"       : round(p_value, 4),
        "ci_95"         : (round(ci_low, 6), round(ci_high, 6)),
        "significant"   : p_value < 0.05,
        "n_bootstrap"   : n_bootstrap,
        "metric"        : metric,
    }


def compute_metrics(returns_df: pd.DataFrame,
                    rf: float = RISK_FREE_RATE) -> pd.DataFrame:
    """Vollständige Performance-Analyse aller Portfoliostrategien."""
    metrics = {}
    for col in returns_df.columns:
        r   = returns_df[col].dropna()
        cum = (1 + r).cumprod()
        metrics[col] = {
            "CAGR (%)"                   : round(cagr(r) * 100, 2),
            "Gesamtrendite (%)"          : round((cum.iloc[-1] - 1) * 100, 2),
            "Annualisierte Vola. (%)"    : round(annualized_vol(r) * 100, 2),
            "Sharpe Ratio"               : round(sharpe_ratio(r, rf), 4),
            "Sortino Ratio"              : round(sortino_ratio(r, rf), 4),
            "Max. Drawdown (%)"          : round(max_drawdown(r) * 100, 2),
            "Calmar Ratio"               : round(calmar_ratio(r), 4),
            "VaR 95 % (tägl., %)"        : round(np.percentile(r, 5) * 100, 2),
            "Hit Rate (%)"               : round((r > 0).mean() * 100, 2),
        }
    return pd.DataFrame(metrics).T
