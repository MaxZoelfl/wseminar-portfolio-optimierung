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
