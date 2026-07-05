"""
=============================================================================
PORTFOLIO-OPTIMIERUNG: MARKOWITZ (LEDOIT-WOLF) vs. RANDOM FOREST (ML)
W-Seminararbeit | Bayerisches Gymnasium
Version 4.1 — methodische Korrekturen (einfache Renditen, fairer Turnover)
=============================================================================
Strategien:
  1. Markowitz MVO + Ledoit-Wolf Kovarianzschätzung
  2. Random-Forest-gestützte Portfoliooptimierung
  3. Equal Weight (1/N Benchmark)
  4. Risk Parity / Equal Risk Contribution  ← NEU v4

Verbesserungen gegenüber v3:
  A) Risk Parity Benchmark (Equal Risk Contribution)
     Jedes Asset trägt gleich viel zum Gesamtrisiko bei.
     Referenz: Qian, E. (2005). Risk Parity Portfolios.
  B) Cross-sectional Ranking Features für den Random Forest
     RSI-Rang, Momentum-Rang, Alpha-Rang relativ zu allen anderen Assets
     Referenz: Jegadeesh & Titman (1993). Returns to Buying Winners.
  C) Live-Training-Dashboard
     Echtzeit-Visualisierung während des Backtests (4 Panels).
  D) Bootstrap-Signifikanztest in main() aufgerufen
     p-Werte für alle paarweisen Strategievergleiche.
  E) Stress-Test-Visualisierung
     COVID-Crash (Feb–Apr 2020) + Zinsanstieg 2022.
  F) Animierte Efficient-Frontier-Evolution als GIF (Pillow-Fallback).
  G) Turnover vs. Performance Scatter (Kosten-Effizienz-Analyse).
  H) SHAP-Erklärbarkeit des Random Forest (optionale shap-Bibliothek).
  I) JSON-Experimentprotokoll für Reproduzierbarkeit.

Beibehaltene Korrekturen aus v3:
  - Look-Ahead-Bias behoben: letzter Trainingsmonat von RF-Training ausgeschl.
  - Positionsobergrenze 20% (MAX_WEIGHT = 0.20)
  - Transaktionskosten 0.10% auf Handelsumsatz (Turnover)
  - TimeSeriesSplit für Cross-Validation

Methodische Korrekturen v4.1:
  - Einfache (arithmetische) statt logarithmischer Renditen. Damit ist die
    Portfoliorendite exakt die gewichtete Summe der Asset-Renditen und
    konsistent zur Kennzahl-Berechnung über (1 + r).cumprod(). Monatsrenditen
    werden korrekt aufgezinst ( prod(1+r) - 1 ) statt summiert.
  - Driftbewusster Turnover: Vergleich der neuen Zielgewichte mit den über die
    letzte Halteperiode gedrifteten Vorgängergewichten. Equal Weight erhält so
    einen realistischen Rebalancing-Turnover (vorher fälschlich 0).

Datenquelle  : Yahoo Finance via yfinance
Zeitraum     : 2015-01-01 bis 2024-12-31 (Warmup für Indikatoren ab 2013-01-01)
Rebalancing  : Monatlich (Ende jedes Monats)
Training     : Rollierendes 3-Jahres-Fenster
Outputs      : PNG-Grafiken + CSV + GIF + JSON im Ordner ./output1.6/
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0. IMPORTS
# ---------------------------------------------------------------------------
import time
import warnings
import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# Matplotlib: interaktives Backend für Live-Dashboard versuchen,
# Fallback auf Agg (headless/serverless Umgebungen)
import matplotlib
_BACKEND_TESTED = False
INTERACTIVE_DISPLAY = False
try:
    # Versuche interaktives Backend zu aktivieren
    import matplotlib.pyplot as plt
    _test = plt.figure(figsize=(1, 1))
    plt.close(_test)
    INTERACTIVE_DISPLAY = True
    _BACKEND_TESTED = True
except Exception:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INTERACTIVE_DISPLAY = False
    _BACKEND_TESTED = True

import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize as MplNorm
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import randint, uniform

from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Optionale Abhängigkeiten mit graceful Fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from matplotlib.animation import FuncAnimation, PillowWriter
    ANIMATION_AVAILABLE = True
except ImportError:
    ANIMATION_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("portfolio_v4")

# ---------------------------------------------------------------------------
# GLOBALE KONFIGURATION
# ---------------------------------------------------------------------------

TICKERS = [
    "AAPL",   # Apple            — Technologie
    "MSFT",   # Microsoft        — Technologie
    "NVDA",   # NVIDIA           — Halbleiter
    "JNJ",    # Johnson & Johnson — Gesundheit
    "UNH",    # UnitedHealth     — Gesundheit
    "JPM",    # JPMorgan         — Finanzen
    "GS",     # Goldman Sachs    — Finanzen
    "PG",     # Procter & Gamble — Konsum (nicht-zyklisch)
    "KO",     # Coca-Cola        — Konsum (nicht-zyklisch)
    "XOM",    # Exxon Mobil      — Energie
    "CAT",    # Caterpillar      — Industrie
    "HON",    # Honeywell        — Industrie
    "VZ",     # Verizon          — Telekommunikation
    "PLD",    # Prologis         — Immobilien (REIT)
    "LIN",    # Linde            — Grundstoffe
]

SPY_TICKER        = "SPY"
START_DATE        = "2013-01-01"   # Extra-Warmup für Indikatoren
END_DATE          = "2024-12-31"
BACKTEST_START    = "2015-01-01"
RISK_FREE_RATE    = 0.04           # Annualisiert (~US-10J 2024)
TRAIN_YEARS       = 3
N_FRONTIER        = 120
OUTPUT_DIR        = "./output1.6"

RF_N_ITER         = 30             # RandomizedSearchCV Iterationen
RF_CV_SPLITS      = 5              # TimeSeriesSplit Folds

MAX_WEIGHT        = 0.20           # Positionsobergrenze je Asset
TRANSACTION_COST  = 0.0010         # 0.10% auf Handelsumsatz
RF_TURNOVER_LIMIT = 0.30           # Max. 30% einseitiger Turnover/Monat (RF)

# Cross-sectional Ranking: welche Features werden gerankt (v4)
RANK_COLS = ["rsi", "mom_21d", "mom_63d", "mom_252d", "alpha_spy"]

# Matplotlib-Design
plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
    "axes.grid"         : True,
    "grid.alpha"        : 0.25,
    "grid.linewidth"    : 0.6,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "font.family"       : "serif",
    "font.size"         : 11,
    "axes.titlesize"    : 13,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 11,
    "legend.fontsize"   : 9,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
})

COLORS = {
    "Markowitz MVO" : "#1f77b4",
    "Random Forest" : "#d62728",
    "Equal Weight"  : "#2ca02c",
    "Risk Parity"   : "#ff7f0e",   # NEU v4
    "frontier"      : "#7f7f7f",
    "cml"           : "#9467bd",
    "mvp"           : "#e7ba52",
}

STRATEGIES = ["Markowitz MVO", "Random Forest", "Equal Weight", "Risk Parity"]


# ---------------------------------------------------------------------------
# 1. HILFSFUNKTIONEN & DEKORATOREN
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. TECHNISCHE INDIKATOREN (täglich)
# ---------------------------------------------------------------------------

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI nach Wilder (1978): Überkauft/überverkauft Signal."""
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(prices: pd.Series, fast: int = 12,
                 slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD nach Appel (1979): Trendfolge-Indikator."""
    ema_f  = prices.ewm(span=fast,   adjust=False).mean()
    ema_s  = prices.ewm(span=slow,   adjust=False).mean()
    macd   = ema_f - ema_s
    sig    = macd.ewm(span=signal,   adjust=False).mean()
    return pd.DataFrame({"macd": macd, "macd_sig": sig, "macd_hist": macd - sig})


def compute_bollinger(prices: pd.Series, period: int = 20,
                      n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bänder nach Bollinger (1983): Volatilitäts-Indikator."""
    sma   = prices.rolling(period).mean()
    std   = prices.rolling(period).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    pct_b = (prices - lower) / (upper - lower + 1e-10)
    bw    = (upper - lower) / (sma + 1e-10)
    return pd.DataFrame({"bb_pct_b": pct_b, "bb_width": bw})


def compute_momentum(returns: pd.Series,
                     periods: list = [21, 63, 126, 252]) -> pd.DataFrame:
    """Preismomentum über verschiedene Horizonte (Jegadeesh & Titman, 1993)."""
    return pd.DataFrame({f"mom_{p}d": returns.rolling(p).sum() for p in periods})


def compute_volatility_features(returns: pd.Series,
                                 periods: list = [21, 63]) -> pd.DataFrame:
    """Rollende historische Volatilität als Risikowarnsignal."""
    return pd.DataFrame({
        f"vol_{p}d": returns.rolling(p).std() * np.sqrt(252) for p in periods
    })


def compute_alpha_beta(asset_returns: pd.Series,
                       market_returns: pd.Series,
                       window: int = 63) -> pd.DataFrame:
    """
    Rollierendes Alpha & Beta vs. S&P 500 (SPY).
    OLS-Regression im rollierenden Fenster: r_i = α + β·r_m + ε
    """
    alphas, betas = [], []
    idx = asset_returns.index

    for i in range(len(idx)):
        if i < window:
            alphas.append(np.nan)
            betas.append(np.nan)
            continue
        ra   = asset_returns.iloc[i-window:i].values
        rm   = market_returns.reindex(asset_returns.index[i-window:i]).values
        mask = ~(np.isnan(ra) | np.isnan(rm))
        if mask.sum() < window // 2:
            alphas.append(np.nan)
            betas.append(np.nan)
            continue
        ra, rm = ra[mask], rm[mask]
        beta   = np.cov(ra, rm)[0, 1] / (np.var(rm) + 1e-12)
        alpha  = ra.mean() - beta * rm.mean()
        alphas.append(alpha * 252)
        betas.append(beta)

    return pd.DataFrame({"alpha_spy": alphas, "beta_spy": betas}, index=idx)


@timer
def build_all_indicators(daily_prices: pd.DataFrame,
                          daily_returns: pd.DataFrame,
                          spy_returns: pd.Series,
                          tickers: list) -> dict:
    """Berechnet alle technischen Indikatoren für jedes Asset (täglich)."""
    log.info("Berechne technische Indikatoren für alle Assets …")
    indicator_dict = {}

    for ticker in tickers:
        prices = daily_prices[ticker]
        rets   = daily_returns[ticker]
        combined = pd.concat([
            compute_rsi(prices).rename("rsi"),
            compute_macd(prices),
            compute_bollinger(prices),
            compute_momentum(rets),
            compute_volatility_features(rets),
            compute_alpha_beta(rets, spy_returns),
        ], axis=1)
        indicator_dict[ticker] = combined

    log.info(f"  Indikatoren: {len(indicator_dict)} Assets, "
             f"{combined.shape[1]} Features je Asset")
    return indicator_dict


# ---------------------------------------------------------------------------
# 3. FEATURE-AGGREGATION: TÄGLICH → MONATLICH + CROSS-SECTIONAL RANKS
# ---------------------------------------------------------------------------

def aggregate_to_monthly(indicator_dict: dict,
                          daily_returns: pd.DataFrame,
                          tickers: list) -> pd.DataFrame:
    """
    Aggregiert tägliche Indikatoren zu monatlichen Feature-Vektoren.
    Aggregationslogik:
      - Endwert (last):  RSI, MACD, Bollinger, Alpha, Beta
      - Durchschnitt (mean): Volatilität, Momentum
    """
    rows = []
    for ticker in tickers:
        ind_df     = indicator_dict[ticker].copy()
        ret_col    = daily_returns[ticker]
        # Monatsrendite korrekt aufzinsen: einfache Renditen sind über die
        # ZEIT multiplikativ ( prod(1+r) - 1 ), nicht additiv.
        monthly_r  = (1 + ret_col).resample("ME").prod() - 1

        feat = pd.DataFrame(index=monthly_r.index)
        for col in ["rsi", "macd", "macd_sig", "macd_hist",
                    "bb_pct_b", "bb_width", "alpha_spy", "beta_spy"]:
            if col in ind_df.columns:
                feat[col] = ind_df[col].resample("ME").last()
        for col in ["mom_21d", "mom_63d", "mom_126d", "mom_252d",
                    "vol_21d", "vol_63d"]:
            if col in ind_df.columns:
                feat[col] = ind_df[col].resample("ME").mean()

        feat["ticker"]            = ticker
        feat["monthly_ret"]       = monthly_r
        feat["target_next_month"] = monthly_r.shift(-1)
        rows.append(feat)

    combined = pd.concat(rows).sort_index()
    combined = combined.replace([np.inf, -np.inf], np.nan)
    return combined


def add_cross_sectional_ranks(monthly_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional Ranking Features (v4).
    Für jeden Monat t: jedes Asset erhält einen Perzentil-Rang (0–1)
    relativ zu ALLEN anderen Assets in diesem Monat.

    Wissenschaftliche Begründung (Jegadeesh & Titman, 1993):
      Das relative Momentum (Cross-sectional Momentum) erklärt Aktienrenditen
      besser als absolutes Momentum. Gewinner-Assets tendieren zur
      Outperformance gegenüber Verlierern über 3–12 Monate.
    """
    result = monthly_data.copy()
    for col in RANK_COLS:
        if col not in monthly_data.columns:
            continue
        result[f"{col}_rank"] = (
            monthly_data.groupby(level=0)[col]
            .rank(pct=True, na_option="keep")
        )
    return result


# ---------------------------------------------------------------------------
# 4. FEATURE-SPALTEN
# ---------------------------------------------------------------------------

FEATURE_COLS_BASE = [
    "rsi", "macd", "macd_sig", "macd_hist",
    "bb_pct_b", "bb_width",
    "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_21d", "vol_63d",
    "alpha_spy", "beta_spy",
    "monthly_ret",
]

FEATURE_COLS = FEATURE_COLS_BASE + [f"{c}_rank" for c in RANK_COLS]

# Lesbarer Name für Plots
FEATURE_DISPLAY_NAMES = {
    "rsi"           : "RSI (14)",
    "macd"          : "MACD-Linie",
    "macd_sig"      : "MACD-Signal",
    "macd_hist"     : "MACD-Histogramm",
    "bb_pct_b"      : "Bollinger %B",
    "bb_width"      : "Bollinger Bandwidth",
    "mom_21d"       : "Momentum 1M",
    "mom_63d"       : "Momentum 3M",
    "mom_126d"      : "Momentum 6M",
    "mom_252d"      : "Momentum 12M",
    "vol_21d"       : "Volatilität 1M",
    "vol_63d"       : "Volatilität 3M",
    "alpha_spy"     : "Alpha vs. SPY",
    "beta_spy"      : "Beta vs. SPY",
    "monthly_ret"   : "Vormonatsrendite",
    "rsi_rank"      : "RSI-Rang (CS)",
    "mom_21d_rank"  : "Momentum 1M Rang (CS)",
    "mom_63d_rank"  : "Momentum 3M Rang (CS)",
    "mom_252d_rank" : "Momentum 12M Rang (CS)",
    "alpha_spy_rank": "Alpha-Rang (CS)",
}


# ---------------------------------------------------------------------------
# 5. MARKOWITZ MVO MIT LEDOIT-WOLF SHRINKAGE
# ---------------------------------------------------------------------------

class MarkowitzLedoitWolf:
    """
    Klassische Mean-Variance Optimization mit Ledoit-Wolf Kovarianzschätzung.
    Referenz: Ledoit & Wolf (2004). Journal of Multivariate Analysis.
    """

    def __init__(self, rf: float = RISK_FREE_RATE):
        self.rf = rf

    def estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        lw = LedoitWolf()
        lw.fit(returns.values)
        return lw.covariance_ * 252

    def _neg_sharpe(self, weights, mu, cov):
        ret = float(weights @ mu)
        vol = float(np.sqrt(weights @ cov @ weights))
        return -(ret - self.rf) / vol if vol > 0 else 0.0

    def max_sharpe(self, mu: np.ndarray, cov: np.ndarray,
                   w_prev: np.ndarray = None,
                   turnover_limit: float = None) -> np.ndarray:
        """
        Maximiert Sharpe Ratio (Tangential-Portfolio).
        Long-Only, vollständig investiert. Positionsobergrenze: MAX_WEIGHT.
        Optional: Turnover-Constraint nach Garleanu & Pedersen (2013).
        """
        n  = len(mu)
        w0 = w_prev if w_prev is not None else np.ones(n) / n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds      = [(0.0, MAX_WEIGHT)] * n

        if w_prev is not None and turnover_limit is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, wp=w_prev, lim=turnover_limit:
                       lim - np.abs(w - wp).sum() / 2,
            })

        result = minimize(
            fun=self._neg_sharpe, x0=w0, args=(mu, cov),
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        w = np.maximum(result.x, 0.0)
        w /= w.sum()
        return w

    def efficient_frontier(self, mu: np.ndarray, cov: np.ndarray,
                            n_points: int = N_FRONTIER) -> pd.DataFrame:
        """Berechnet N Punkte auf der Effizienzlinie (Long-Only)."""
        n      = len(mu)
        bounds = [(0.0, 1.0)] * n

        res_mvp = minimize(
            lambda w: float(w @ cov @ w), np.ones(n) / n,
            method="SLSQP", bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        )
        ret_min = float(res_mvp.x @ mu)
        ret_max = float(mu.max())

        frontier = []
        for target in np.linspace(ret_min, ret_max, n_points):
            res = minimize(
                lambda w: float(w @ cov @ w), np.ones(n) / n,
                method="SLSQP", bounds=bounds,
                constraints=[
                    {"type": "eq", "fun": lambda w: w.sum() - 1},
                    {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
                ],
                options={"maxiter": 500, "ftol": 1e-12},
            )
            if res.success:
                vol = float(np.sqrt(res.x @ cov @ res.x))
                sr  = (target - self.rf) / vol if vol > 0 else 0.0
                frontier.append({"ret": target, "vol": vol, "sr": sr, "w": res.x.copy()})

        return pd.DataFrame(frontier)


# ---------------------------------------------------------------------------
# 6. RISK PARITY PORTFOLIO (NEU v4)
# ---------------------------------------------------------------------------

class RiskParityPortfolio:
    """
    Equal Risk Contribution (Risk Parity) Portfolio.

    Jedes Asset trägt gleich viel zum gesamten Portfoliorisiko bei.
    Formell: RC_i = w_i * (Sigma w)_i / sqrt(w' Sigma w) = 1/N für alle i

    Vorteile gegenüber Equal Weight:
      - Berücksichtigt Asset-Volatilität und Korrelationen
      - Weniger Konzentration in risikoreichen Assets
      - Empirisch bessere risikobereinigte Renditen (Qian, 2005)

    Referenz: Qian, E. (2005). Risk Parity Portfolios: Efficient Portfolios
    through True Diversification. PanAgora Asset Management.
    """

    def optimize(self, cov: np.ndarray, max_weight: float = MAX_WEIGHT) -> np.ndarray:
        n = cov.shape[0]

        def _objective(w):
            """Minimiert die Streuung der Risikobeiträge."""
            portfolio_vol = np.sqrt(max(w @ cov @ w, 1e-10))
            rc = w * (cov @ w) / portfolio_vol
            target_rc = portfolio_vol / n
            return float(np.sum((rc - target_rc) ** 2))

        w0 = np.ones(n) / n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0.005, max_weight)] * n

        result = minimize(
            fun=_objective, x0=w0,
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 3000, "ftol": 1e-14},
        )

        w = np.maximum(result.x, 0.0)
        w /= w.sum()
        return w


# ---------------------------------------------------------------------------
# 7. RANDOM FOREST MIT TIMESERIESPLIT + RANDOMIZEDSEARCHCV
# ---------------------------------------------------------------------------

class RFPortfolioOptimizer:
    """
    Random-Forest-gestützte Portfoliooptimierung mit monatlichem Rebalancing.
    Referenz: Fischer & Krauss (2018). European Journal of Operational Research.
    """

    def __init__(self, rf: float = RISK_FREE_RATE,
                 n_iter: int = RF_N_ITER,
                 cv_splits: int = RF_CV_SPLITS):
        self.rf               = rf
        self.n_iter           = n_iter
        self.cv_splits        = cv_splits
        self.best_estimator_  = None
        self.best_params_     = {}
        self._mvo             = MarkowitzLedoitWolf(rf=rf)

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestRegressor(random_state=42, n_jobs=-1)),
        ])

    def _param_grid(self) -> dict:
        return {
            "rf__n_estimators"     : randint(100, 500),
            "rf__max_depth"        : randint(3, 15),
            "rf__min_samples_leaf" : randint(3, 20),
            "rf__max_features"     : uniform(0.3, 0.6),
            "rf__max_samples"      : uniform(0.6, 0.35),
        }

    @timer
    def fit_with_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Training mit RandomizedSearchCV + TimeSeriesSplit.
        TimeSeriesSplit verhindert Look-Ahead-Bias: Testdaten liegen immer
        zeitlich NACH den Trainingsdaten.
        """
        tscv   = TimeSeriesSplit(n_splits=self.cv_splits)
        search = RandomizedSearchCV(
            estimator=self._build_pipeline(),
            param_distributions=self._param_grid(),
            n_iter=self.n_iter, scoring="neg_mean_squared_error",
            cv=tscv, random_state=42, n_jobs=-1, refit=True,
        )
        search.fit(X_train.values, y_train.values)
        self.best_estimator_ = search.best_estimator_
        self.best_params_    = search.best_params_
        log.info(
            f"    RF-Params: n_est={self.best_params_.get('rf__n_estimators','?')}, "
            f"depth={self.best_params_.get('rf__max_depth','?')}, "
            f"leaf={self.best_params_.get('rf__min_samples_leaf','?')}"
        )

    def predict_monthly_returns(self, X_current: pd.DataFrame) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Zuerst fit_with_tuning() aufrufen.")
        return self.best_estimator_.predict(X_current.values) * 12

    def optimize(self, mu_predicted: np.ndarray, cov: np.ndarray,
                 w_prev: np.ndarray = None) -> np.ndarray:
        return self._mvo.max_sharpe(
            mu_predicted, cov,
            w_prev=w_prev, turnover_limit=RF_TURNOVER_LIMIT,
        )


# ---------------------------------------------------------------------------
# 8. DATEN LADEN
# ---------------------------------------------------------------------------

@timer
def download_data(tickers: list, spy_ticker: str, start: str, end: str):
    log.info(f"Lade Marktdaten: {len(tickers)} Assets + SPY | {start} → {end}")
    raw = yf.download(
        tickers + [spy_ticker], start=start, end=end,
        auto_adjust=True, progress=False,
    )["Close"]

    missing   = [t for t in tickers + [spy_ticker] if t not in raw.columns]
    if missing:
        log.warning(f"Fehlende Tickers: {missing}")

    available = [t for t in tickers if t in raw.columns]
    prices    = raw[available + [spy_ticker]].dropna(how="any")
    log.info(f"  Verfügbare Assets: {len(available)} | Handelstage: {len(prices)}")

    # Einfache (arithmetische) Renditen: P_t / P_{t-1} - 1.
    # Bewusst KEINE Log-Renditen: Einfache Renditen sind über Assets additiv,
    # d.h. die Portfoliorendite ist exakt die gewichtete Summe der Asset-
    # Renditen ( sum_i w_i * r_i ). Log-Renditen sind das NICHT — ihre
    # gewichtete Summe über Assets ergibt keine gültige Portfoliorendite und
    # wäre inkonsistent mit der Kennzahl-Berechnung via (1 + r).cumprod().
    daily_returns = prices.pct_change(fill_method=None).dropna()
    spy_ret   = daily_returns[spy_ticker]
    asset_ret = daily_returns[available]
    asset_px  = prices[available]
    return asset_px, asset_ret, spy_ret, available


# ---------------------------------------------------------------------------
# 9. LIVE-TRAINING-DASHBOARD
# ---------------------------------------------------------------------------

class LiveDashboard:
    """
    Echtzeit-Visualisierung während des Backtests.

    Zeigt 4 Panels:
      1. Kumulierte Renditen aller Strategien (wird mit jedem Step erweitert)
      2. Aktuelle MVO-Gewichtung (Balkendiagramm)
      3. Rollierender Sharpe Ratio (gleitend, letztes verfügbares Fenster)
      4. Fortschritt & aktuelle Kennzahlen (Texttafel)

    Hinweis: Funktioniert nur mit interaktivem Matplotlib-Backend.
    In headless-Umgebungen (z.B. Server ohne Display) wird das Dashboard
    automatisch deaktiviert ohne den Backtest zu unterbrechen.
    """

    def __init__(self, tickers: list, n_steps: int):
        self.tickers  = tickers
        self.n_steps  = n_steps
        self.fig      = None
        self.axes     = {}
        self.active   = False
        self._init_time = datetime.now()
        self._setup()

    def _setup(self):
        if not INTERACTIVE_DISPLAY:
            log.info("Live-Dashboard: Kein interaktives Display verfügbar "
                     "(headless), überspringe.")
            return
        try:
            plt.ion()
            self.fig = plt.figure(figsize=(18, 10))
            self.fig.patch.set_facecolor("white")
            self.fig.suptitle(
                "Portfolio-Optimierung v4 | Live-Training-Dashboard",
                fontsize=14, fontweight="bold", y=0.98
            )
            gs = gridspec.GridSpec(
                2, 3, figure=self.fig,
                hspace=0.40, wspace=0.35,
                left=0.06, right=0.97, top=0.93, bottom=0.07,
            )
            self.axes["returns"]  = self.fig.add_subplot(gs[0, :2])
            self.axes["weights"]  = self.fig.add_subplot(gs[0, 2])
            self.axes["sharpe"]   = self.fig.add_subplot(gs[1, :2])
            self.axes["progress"] = self.fig.add_subplot(gs[1, 2])

            # Initiale Beschriftungen
            self.axes["returns"].set_title("Kumulierte Renditen", fontsize=11)
            self.axes["weights"].set_title("MVO-Gewichte (aktuell)", fontsize=11)
            self.axes["sharpe"].set_title("Rollierender Sharpe Ratio", fontsize=11)
            self.axes["progress"].axis("off")

            plt.pause(0.05)
            self.active = True
            log.info("Live-Dashboard aktiviert.")
        except Exception as e:
            log.warning(f"Live-Dashboard Initialisierungsfehler: {e}")
            self.active = False

    def update(self, step: int, month, returns_so_far: pd.DataFrame,
               w_mvo: np.ndarray, w_rf: np.ndarray):
        """Aktualisiert alle 4 Dashboard-Panels nach jedem Rebalancing-Schritt."""
        if not self.active:
            return
        try:
            elapsed = (datetime.now() - self._init_time).seconds

            # ---- Panel 1: Kumulierte Renditen ----------------------------
            ax = self.axes["returns"]
            ax.clear()
            if not returns_so_far.empty:
                for col in STRATEGIES:
                    if col not in returns_so_far.columns:
                        continue
                    cum = (1 + returns_so_far[col].dropna()).cumprod()
                    ax.plot(cum.index, cum.values, label=col,
                            color=COLORS.get(col, "gray"), linewidth=1.8)
            ax.axhline(1, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
            ax.set_title("Kumulierte Renditen (laufend)", fontsize=10, pad=6)
            ax.set_ylabel("Wert (Start = 1 €)")
            ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
            ax.grid(True, alpha=0.2)

            # ---- Panel 2: Aktuelle MVO-Gewichte -------------------------
            ax = self.axes["weights"]
            ax.clear()
            n  = len(self.tickers)
            x  = np.arange(n)
            ax.bar(x, w_mvo * 100, color=COLORS["Markowitz MVO"],
                   alpha=0.75, edgecolor="white", label="MVO", width=0.4)
            ax.bar(x + 0.4, w_rf * 100, color=COLORS["Random Forest"],
                   alpha=0.75, edgecolor="white", label="RF", width=0.4)
            ax.axhline(MAX_WEIGHT * 100, color="red", linestyle="--",
                       linewidth=0.9, alpha=0.6, label=f"Max {MAX_WEIGHT*100:.0f}%")
            ax.set_xticks(x + 0.2)
            ax.set_xticklabels(self.tickers, rotation=45, ha="right", fontsize=6)
            ax.set_title("Aktuelle Gewichte: MVO vs. RF (%)", fontsize=10, pad=6)
            ax.set_ylabel("Gewicht (%)")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.2, axis="y")

            # ---- Panel 3: Rollierender Sharpe ----------------------------
            ax = self.axes["sharpe"]
            ax.clear()
            if not returns_so_far.empty and len(returns_so_far) > 60:
                rf_d   = RISK_FREE_RATE / 252
                window = min(252, max(60, len(returns_so_far) // 3))
                for col in STRATEGIES:
                    if col not in returns_so_far.columns:
                        continue
                    r  = returns_so_far[col].dropna()
                    rs = (r.rolling(window).mean() - rf_d) / r.rolling(window).std() * np.sqrt(252)
                    ax.plot(rs.index, rs.values, label=col,
                            color=COLORS.get(col, "gray"), linewidth=1.4, alpha=0.9)
                ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
                ax.axhline(1, color="gray",  linewidth=0.5, linestyle=":",  alpha=0.4)
            ax.set_title(f"Rollierender Sharpe Ratio", fontsize=10, pad=6)
            ax.set_ylabel("Sharpe Ratio")
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.2)

            # ---- Panel 4: Fortschritt & Kennzahlen ----------------------
            ax = self.axes["progress"]
            ax.clear()
            ax.axis("off")

            progress = step / max(self.n_steps, 1) * 100
            eta_s    = elapsed / max(step, 1) * (self.n_steps - step)
            eta_min  = eta_s / 60

            lines = [
                f"Fortschritt: {step}/{self.n_steps}  ({progress:.1f}%)",
                f"Datum:       {str(month)[:10]}",
                f"Laufzeit:    {elapsed//60:.0f}m {elapsed%60:.0f}s",
                f"ETA:         ~{eta_min:.1f} min",
                "",
            ]

            if not returns_so_far.empty:
                lines.append("Kennzahlen (bisher):")
                lines.append("─" * 30)
                for col in STRATEGIES:
                    if col not in returns_so_far.columns:
                        continue
                    r = returns_so_far[col].dropna()
                    if len(r) > 0:
                        sr = sharpe_ratio(r)
                        c  = cagr(r)
                        lines.append(f"{col[:14]:<14}")
                        lines.append(f"  CAGR: {c*100:>6.1f}%  SR: {sr:>5.2f}")

            ax.text(0.04, 0.97, "\n".join(lines),
                    transform=ax.transAxes, fontsize=8.5,
                    va="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.5",
                              facecolor="#f0f4ff", alpha=0.9, edgecolor="#c0c8e0"))

            self.fig.canvas.draw_idle()
            plt.pause(0.03)

        except Exception:
            pass   # Dashboard-Fehler dürfen den Backtest nie unterbrechen

    def finalize(self, output_path: str):
        """Speichert den finalen Dashboard-Status als PNG."""
        if not self.active:
            return
        try:
            self.fig.savefig(output_path, dpi=150, bbox_inches="tight")
            log.info(f"  Live-Dashboard gespeichert: {output_path}")
            plt.ioff()
        except Exception as e:
            log.warning(f"Dashboard-Speicherung fehlgeschlagen: {e}")

    def close(self):
        if self.active and self.fig is not None:
            try:
                plt.ioff()
                plt.close(self.fig)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 10. BACKTEST ENGINE
# ---------------------------------------------------------------------------

@timer
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
    """
    log.info("Aggregiere Features täglich → monatlich …")
    monthly_data = aggregate_to_monthly(indicator_dict, asset_returns, tickers)
    monthly_data = add_cross_sectional_ranks(monthly_data)    # v4: CS-Ranks

    backtest_months = monthly_data.index.unique()
    backtest_months = backtest_months[backtest_months >= BACKTEST_START]
    n               = len(tickers)

    mvo = MarkowitzLedoitWolf()
    rp  = RiskParityPortfolio()   # NEU v4
    rfo = RFPortfolioOptimizer()

    results_mvo, results_rf, results_ew, results_rp = [], [], [], []
    hist_w_mvo, hist_w_rf, hist_w_rp               = [], [], []
    frontier_snapshots                              = []
    turnover_log                                    = []   # v4: Turnover je Step

    w_prev_rf = np.ones(n) / n   # Start: Equal Weight

    # Wachstumsfaktoren der jeweils letzten Halteperiode (für driftbewussten
    # Turnover). None = es gibt noch keine Vorperiode.
    prev_hold_growth = None

    # Live-Dashboard initialisieren
    dashboard = LiveDashboard(tickers, len(backtest_months) - 1)

    log.info(f"Starte Backtest: {len(backtest_months)} Rebalancing-Monate …")

    for i, month_end in enumerate(backtest_months[:-1]):
        t0 = time.perf_counter()

        train_start   = month_end - pd.DateOffset(years=TRAIN_YEARS)
        train_daily   = asset_returns[
            (asset_returns.index >= train_start) &
            (asset_returns.index <= month_end)
        ]
        train_monthly = monthly_data[
            (monthly_data.index >= train_start) &
            (monthly_data.index <= month_end)
        ]

        if len(train_daily) < 252 or len(train_monthly.index.unique()) < 24:
            log.warning(f"  [{month_end.date()}] Zu wenige Daten, überspringe.")
            continue

        next_month  = backtest_months[i + 1]
        hold_period = asset_returns[
            (asset_returns.index > month_end) &
            (asset_returns.index <= next_month)
        ]
        if hold_period.empty:
            continue

        # ---- Kovarianzmatrix (Ledoit-Wolf) --------------------------------
        cov_ann = mvo.estimate_covariance(train_daily)
        mu_hist = train_daily.mean().values * 252

        # ---- Markowitz MVO ------------------------------------------------
        try:
            w_mvo = mvo.max_sharpe(mu_hist, cov_ann)
        except Exception as e:
            log.warning(f"  MVO-Fehler {month_end.date()}: {e}")
            w_mvo = np.ones(n) / n

        # ---- Risk Parity (NEU v4) -----------------------------------------
        try:
            w_rp = rp.optimize(cov_ann)
        except Exception as e:
            log.warning(f"  RP-Fehler {month_end.date()}: {e}")
            w_rp = np.ones(n) / n

        # ---- Random Forest ------------------------------------------------
        # v3-FIX: Letzter Trainingsmonat ausgeschlossen (Look-Ahead-Bias)
        last_train_month = month_end - pd.DateOffset(months=1)
        feat_rows, target_rows = [], []
        for ticker in tickers:
            t_rows = train_monthly[
                (train_monthly["ticker"] == ticker) &
                (train_monthly.index <= last_train_month)
            ]
            valid = t_rows[[c for c in FEATURE_COLS if c in t_rows.columns]
                           + ["target_next_month"]].dropna()
            if len(valid) < 12:
                continue
            avail_feats = [c for c in FEATURE_COLS if c in valid.columns]
            feat_rows.append(valid[avail_feats])
            target_rows.append(valid["target_next_month"])

        if len(feat_rows) < n // 2:
            w_rf  = np.ones(n) / n
            mu_rf = mu_hist.copy()
        else:
            X_train_rf = pd.concat(feat_rows)
            y_train_rf = pd.concat(target_rows)
            try:
                rfo.fit_with_tuning(X_train_rf, y_train_rf)
                X_current_rows = []
                for ticker in tickers:
                    t_rows = train_monthly[train_monthly["ticker"] == ticker]
                    avail  = [c for c in FEATURE_COLS if c in t_rows.columns]
                    valid  = t_rows[avail].dropna()
                    if len(valid) > 0:
                        X_current_rows.append(valid.iloc[-1])
                    else:
                        X_current_rows.append(
                            pd.Series(np.zeros(len(avail)), index=avail)
                        )
                X_current = pd.DataFrame(X_current_rows)
                # Sicherstellen, dass alle Trainingsspalten vorhanden sind
                for col in X_train_rf.columns:
                    if col not in X_current.columns:
                        X_current[col] = 0.0
                X_current = X_current[X_train_rf.columns]
                mu_rf = rfo.predict_monthly_returns(X_current)
                w_rf  = rfo.optimize(mu_rf, cov_ann, w_prev=w_prev_rf)
            except Exception as e:
                log.warning(f"  RF-Fehler {month_end.date()}: {e}")
                w_rf  = np.ones(n) / n
                mu_rf = mu_hist.copy()

        # ---- Efficient Frontier Snapshot (jeden 3. Monat) ----------------
        if i % 3 == 0:
            try:
                snap = mvo.efficient_frontier(mu_hist, cov_ann, n_points=60)
                if not snap.empty:
                    snap["date"]   = month_end
                    snap["w_tang"] = [w_mvo] * len(snap)
                    frontier_snapshots.append(snap)
            except Exception:
                pass

        # ---- Equal Weight ------------------------------------------------
        w_ew = np.ones(n) / n

        # ---- Transaktionskosten ------------------------------------------
        # Turnover = 0.5 * Σ|w_neu − w_alt|. Als "w_alt" werden die über die
        # letzte Halteperiode GEDRIFTETEN Vorgängergewichte verwendet (nicht
        # die ursprünglichen Zielgewichte). Dadurch erhält auch Equal Weight
        # einen realistischen Rebalancing-Turnover (Rückführung der Kursdrift
        # auf 1/N), und alle Strategien werden konsistent bewertet.
        if i == 0 or prev_hold_growth is None:
            turnover_mvo = 1.0   # Erstaufbau des Portfolios
            turnover_rf  = 1.0
            turnover_rp  = 1.0
            turnover_ew  = 1.0
        else:
            def _drift(w_prev):
                wd = np.asarray(w_prev, dtype=float) * prev_hold_growth
                s  = wd.sum()
                return wd / s if s > 0 else np.asarray(w_prev, dtype=float)
            prev_w_mvo = _drift(hist_w_mvo[-1].values) if hist_w_mvo else np.ones(n) / n
            prev_w_rf  = _drift(hist_w_rf[-1].values)  if hist_w_rf  else np.ones(n) / n
            prev_w_rp  = _drift(hist_w_rp[-1].values)  if hist_w_rp  else np.ones(n) / n
            prev_w_ew  = _drift(np.ones(n) / n)
            turnover_mvo = np.abs(w_mvo - prev_w_mvo).sum() / 2
            turnover_rf  = np.abs(w_rf  - prev_w_rf ).sum() / 2
            turnover_rp  = np.abs(w_rp  - prev_w_rp ).sum() / 2
            turnover_ew  = np.abs(w_ew  - prev_w_ew ).sum() / 2

        cost_mvo = turnover_mvo * TRANSACTION_COST
        cost_rf  = turnover_rf  * TRANSACTION_COST
        cost_rp  = turnover_rp  * TRANSACTION_COST
        cost_ew  = turnover_ew  * TRANSACTION_COST

        # ---- Portfoliorenditen -------------------------------------------
        ret_mvo = (hold_period * w_mvo).sum(axis=1).copy()
        ret_rf  = (hold_period * w_rf ).sum(axis=1).copy()
        ret_rp  = (hold_period * w_rp ).sum(axis=1).copy()
        ret_ew  = (hold_period * w_ew ).sum(axis=1).copy()

        if len(ret_mvo) > 0:
            ret_mvo.iloc[0] -= cost_mvo
            ret_rf.iloc[0]  -= cost_rf
            ret_rp.iloc[0]  -= cost_rp
            ret_ew.iloc[0]  -= cost_ew

        results_mvo.append(ret_mvo)
        results_rf.append( ret_rf)
        results_rp.append( ret_rp)
        results_ew.append( ret_ew)

        hist_w_mvo.append(pd.Series(w_mvo, index=tickers, name=month_end))
        hist_w_rf.append( pd.Series(w_rf,  index=tickers, name=month_end))
        hist_w_rp.append( pd.Series(w_rp,  index=tickers, name=month_end))

        # Turnover-Log für Scatter-Plot
        turnover_log.append({
            "date"         : month_end,
            "turnover_mvo" : turnover_mvo,
            "turnover_rf"  : turnover_rf,
            "turnover_rp"  : turnover_rp,
            "turnover_ew"  : turnover_ew,
        })

        w_prev_rf = w_rf.copy()

        # Wachstumsfaktoren dieser Halteperiode je Asset (prod(1+r)) für den
        # driftbewussten Turnover der nächsten Iteration sichern.
        prev_hold_growth = (1 + hold_period).prod(axis=0).reindex(tickers).to_numpy()

        # ---- Live Dashboard aktualisieren --------------------------------
        returns_so_far = pd.DataFrame({
            "Markowitz MVO" : pd.concat(results_mvo).sort_index(),
            "Random Forest" : pd.concat(results_rf ).sort_index(),
            "Equal Weight"  : pd.concat(results_ew ).sort_index(),
            "Risk Parity"   : pd.concat(results_rp ).sort_index(),
        })
        dashboard.update(i + 1, month_end, returns_so_far, w_mvo, w_rf)

        elapsed = time.perf_counter() - t0
        ret_m, vol_m, sr_m = portfolio_perf(w_mvo, mu_hist, cov_ann)
        log.info(
            f"  [{i+1:>3}/{len(backtest_months)-1}] {month_end.date()} | "
            f"MVO-SR: {sr_m:.3f} | "
            f"TO-MVO: {turnover_mvo*100:.1f}% | "
            f"TO-RF: {turnover_rf*100:.1f}% | "
            f"{elapsed:.1f}s"
        )

    # Dashboard finalisieren
    dashboard.finalize(os.path.join(OUTPUT_DIR, "00_live_dashboard_final.png"))
    dashboard.close()

    returns_df = pd.DataFrame({
        "Markowitz MVO" : pd.concat(results_mvo).sort_index(),
        "Random Forest" : pd.concat(results_rf ).sort_index(),
        "Equal Weight"  : pd.concat(results_ew ).sort_index(),
        "Risk Parity"   : pd.concat(results_rp ).sort_index(),
    })

    weights_mvo_df = pd.DataFrame(hist_w_mvo).T
    weights_rf_df  = pd.DataFrame(hist_w_rf ).T
    weights_rp_df  = pd.DataFrame(hist_w_rp ).T
    turnover_df    = pd.DataFrame(turnover_log).set_index("date")

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


# ---------------------------------------------------------------------------
# 11. PERFORMANCE-KENNZAHLEN
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 12. VISUALISIERUNGEN
# ---------------------------------------------------------------------------

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
    frontier = mvo_opt.efficient_frontier(mu_hist, cov_ann)

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


# ---------------------------------------------------------------------------
# 13. CSV-EXPORT
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 14. JSON-EXPERIMENTPROTOKOLL
# ---------------------------------------------------------------------------

def save_experiment_json(metrics_df: pd.DataFrame,
                          bootstrap_results: dict,
                          tickers: list,
                          output_path: str) -> None:
    """
    Speichert Experimentparameter + Ergebnisse als JSON (v4).
    Dient der Reproduzierbarkeit und wissenschaftlichen Dokumentation.
    """
    record = {
        "experiment_meta": {
            "version"           : "4.0",
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
        "bootstrap_tests"    : bootstrap_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"  → {output_path}")


# ---------------------------------------------------------------------------
# 15. HAUPTPROGRAMM
# ---------------------------------------------------------------------------

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