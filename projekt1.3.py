"""
=============================================================================
PORTFOLIO-OPTIMIERUNG: MARKOWITZ (LEDOIT-WOLF) vs. RANDOM FOREST (ML)
W-Seminararbeit | Bayerisches Gymnasium
Version 3.0 — methodisch bereinigt
=============================================================================
Vergleich zweier Portfoliooptimierungsstrategien:
  1. Klassische Markowitz MVO mit Ledoit-Wolf Kovarianzschätzung
  2. Random-Forest-gestützte Portfoliooptimierung
     - Technische Indikatoren als Features (täglich -> monatlich aggregiert)
     - Hyperparameter-Tuning via RandomizedSearchCV + TimeSeriesSplit
     - Monatliche Vorhersage -> monatliches Rebalancing (konsistente Architektur)
     - FIX v3: Look-Ahead-Bias behoben (Zielvariable sauber abgetrennt)
  3. Equal-Weight Benchmark (Blindprobe)

Korrekturen gegenueber v2:
  A) LOOK-AHEAD-BIAS BEHOBEN: Der letzte Monat des Trainingsfensters wird beim
     Aufbau der RF-Trainingsdaten ausgeschlossen, da seine Zielvariable
     (target_next_month) die Rendite der Halteperiode enthaelt (Zukunft).
     Begruendung: Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q.
     (2014). The Probability of Backtest Overfitting. Journal of Computational
     Finance.
  B) POSITIONSOBERGRENZE: Max. 20% je Asset (MAX_WEIGHT = 0.20), gilt fuer
     MVO und RF gleichermassen. Entspricht gaengiger Praxis und UCITS-Richtlinie.
  C) TRANSAKTIONSKOSTEN: 0.10% auf den Handelsumsatz (Turnover) je
     Rebalancing-Termin. Konservative Schaetzung fuer liquide US-Large-Caps
     (Bid-Ask-Spread + Kommission). Kosten werden vom ersten Handelstag der
     neuen Halteperiode abgezogen.
  3. Equal-Weight Benchmark (Blindprobe)

Datenquelle  : Yahoo Finance via yfinance
Zeitraum     : 2015-01-01 bis 2024-12-31
Rebalancing  : Monatlich (Ende jedes Monats)
Training     : Rollierendes 3-Jahres-Fenster
Outputs      : PNG-Grafiken + CSV-Dateien im Ordner ./output/

Wissenschaftliche Grundlagen:
  - Markowitz, H. (1952): Portfolio Selection. Journal of Finance, 7(1), 77-91.
  - Ledoit, O. & Wolf, M. (2004): A well-conditioned estimator for
    large-dimensional covariance matrices. Journal of Multivariate Analysis.
  - Breiman, L. (2001): Random Forests. Machine Learning, 45, 5-32.
  - Sharpe, W. F. (1966/1994): Mutual Fund Performance / The Sharpe Ratio.
  - Krauss, C., Do, X.A., Huck, N. (2017): Deep neural networks, gradient-
    boosted trees, random forests: Statistical arbitrage on the S&P 500.
    European Journal of Operational Research, 259(2), 689-702.
  - DeMiguel, V., Garlappi, L., Uppal, R. (2009): Optimal versus Naive
    Diversification. Review of Financial Studies, 22(5).
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0. IMPORTS
# ---------------------------------------------------------------------------
import time
import warnings
import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import randint, uniform

from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
log = logging.getLogger("portfolio_v2")

# ---------------------------------------------------------------------------
# GLOBALE KONFIGURATION
# ---------------------------------------------------------------------------

TICKERS = [
    "AAPL",   # Apple            — Technologie
    "MSFT",   # Microsoft        — Technologie
    "NVDA",   # NVIDIA           — Halbleiter
    "JNJ",    # Johnson&Johnson  — Gesundheit
    "UNH",    # UnitedHealth     — Gesundheit
    "JPM",    # JPMorgan         — Finanzen
    "GS",     # Goldman Sachs    — Finanzen
    "PG",     # Procter&Gamble   — Konsum (nicht-zyklisch)
    "KO",     # Coca-Cola        — Konsum (nicht-zyklisch)
    "XOM",    # Exxon Mobil      — Energie
    "CAT",    # Caterpillar      — Industrie
    "HON",    # Honeywell        — Industrie
    "VZ",     # Verizon          — Telekommunikation
    "PLD",    # Prologis         — Immobilien (REIT)
    "LIN",    # Linde            — Grundstoffe
]

SPY_TICKER     = "SPY"          # S&P 500 ETF (Benchmark fuer Alpha-Berechnung)
START_DATE     = "2013-01-01"   # Extra 2 Jahre fuer Indikatoren-Warmup
END_DATE       = "2024-12-31"
BACKTEST_START = "2015-01-01"   # Eigentlicher Backtest-Beginn
RISK_FREE_RATE = 0.04           # Annualisierter risikoloser Zinssatz (~US-10J-2024)
TRAIN_YEARS    = 3              # Rollierendes Trainingsfenster
N_FRONTIER     = 120            # Punkte auf der Effizienzlinie
OUTPUT_DIR     = "./output1.3"

# RandomizedSearchCV: Anzahl der getesteten Parameterkombinationen
# Hoeher = genauer, aber langsamer. 30 ist gut fuer Seminararbeiten.
RF_N_ITER      = 30
RF_CV_SPLITS   = 5              # Anzahl TimeSeriesSplit-Faelten

# --- v3: Positionsobergrenze & Transaktionskosten -----------------------
# Max. Gewicht je Asset: verhindert extreme Konzentration (z.B. 100% NVDA).
# 20% entspricht gaengiger institutioneller Praxis und UCITS-Vorgaben.
MAX_WEIGHT     = 0.20

# Transaktionskosten: 0.10% auf den Handelsumsatz (Turnover) je Rebalancing.
# Begruendung: Bid-Ask-Spread (~0.05%) + Kommission (~0.05%) fuer US-Large-Caps.
# Quelle: Frazzini, A., Israel, R., Moskowitz, T. (2015): Trading Costs.
TRANSACTION_COST = 0.0010      # 0.10% pro Umsatz-Einheit

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
    "frontier"      : "#7f7f7f",
    "cml"           : "#9467bd",
    "mvp"           : "#e7ba52",
}


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
    """Compound Annual Growth Rate aus taegl. Renditen."""
    n = len(returns) / freq
    return (1 + returns).prod() ** (1 / n) - 1 if n > 0 else 0.0


def annualized_vol(returns: pd.Series, freq: int = 252) -> float:
    return returns.std() * np.sqrt(freq)


def sharpe_ratio(returns: pd.Series,
                 rf: float = RISK_FREE_RATE,
                 freq: int = 252) -> float:
    r  = cagr(returns, freq)
    v  = annualized_vol(returns, freq)
    return (r - rf) / v if v > 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    cum        = (1 + returns).cumprod()
    roll_max   = cum.cummax()
    dd         = (cum - roll_max) / roll_max
    return dd.min()


def calmar_ratio(returns: pd.Series, freq: int = 252) -> float:
    mdd = abs(max_drawdown(returns))
    return cagr(returns, freq) / mdd if mdd > 0 else 0.0


def sortino_ratio(returns: pd.Series,
                  rf: float = RISK_FREE_RATE,
                  freq: int = 252) -> float:
    """Sortino Ratio: bestraft nur negative Volatilitaet (Downside-Risiko)."""
    r           = cagr(returns, freq)
    down        = returns[returns < 0]
    down_vol    = down.std() * np.sqrt(freq)
    return (r - rf) / down_vol if down_vol > 0 else 0.0


def portfolio_perf(weights: np.ndarray,
                   mu: np.ndarray,
                   cov: np.ndarray,
                   rf: float = RISK_FREE_RATE):
    """Annualisierte Rendite, Volatilitaet und Sharpe eines Portfolios."""
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sr  = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sr


# ---------------------------------------------------------------------------
# 2. TECHNISCHE INDIKATOREN (taeglich)
# ---------------------------------------------------------------------------

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) nach Wilder (1978).
    Misst die Staerke eines Preistrends; Werte > 70 = ueberkauft, < 30 = ueberverkauft.
    Signal fuer Mean-Reversion-Strategien.
    """
    delta  = prices.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(prices: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence) nach Appel (1979).
    Trendfolge-Indikator: MACD-Linie = EMA(12) - EMA(26).
    Signal-Linie = EMA(9) des MACD. Histogramm = Differenz.
    """
    ema_fast    = prices.ewm(span=fast,   adjust=False).mean()
    ema_slow    = prices.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return pd.DataFrame({
        "macd"      : macd_line,
        "macd_sig"  : signal_line,
        "macd_hist" : histogram,
    })


def compute_bollinger(prices: pd.Series,
                      period: int = 20,
                      n_std:  float = 2.0) -> pd.DataFrame:
    """
    Bollinger Baender nach Bollinger (1983).
    Mittleres Band = SMA(20), Baender = +/- 2 Standardabweichungen.
    %B-Indikator zeigt Position des Kurses relativ zu den Baendern (0-1).
    """
    sma       = prices.rolling(period).mean()
    std       = prices.rolling(period).std()
    upper     = sma + n_std * std
    lower     = sma - n_std * std
    pct_b     = (prices - lower) / (upper - lower + 1e-10)
    bandwidth = (upper - lower) / (sma + 1e-10)
    return pd.DataFrame({
        "bb_upper"    : upper,
        "bb_lower"    : lower,
        "bb_pct_b"    : pct_b,
        "bb_width"    : bandwidth,
    })


def compute_momentum(returns: pd.Series,
                     periods: list = [21, 63, 126, 252]) -> pd.DataFrame:
    """
    Preismomentum ueber verschiedene Rueckblickfenster.
    Empirisch: Momentum-Effekt (Jegadeesh & Titman, 1993):
    vergangene Gewinner tendieren zur Outperformance (1-12 Monate).
    """
    out = {}
    for p in periods:
        out[f"mom_{p}d"] = returns.rolling(p).sum()
    return pd.DataFrame(out)


def compute_volatility_features(returns: pd.Series,
                                 periods: list = [21, 63]) -> pd.DataFrame:
    """
    Rollende historische Volatilitaet. Hoehere Volatilitaet = hoehere
    Unsicherheit; kann als Risikowarnsignal dienen.
    """
    out = {}
    for p in periods:
        out[f"vol_{p}d"] = returns.rolling(p).std() * np.sqrt(252)
    return pd.DataFrame(out)


def compute_alpha_beta(asset_returns: pd.Series,
                       market_returns: pd.Series,
                       window: int = 63) -> pd.DataFrame:
    """
    Rollierendes Alpha und Beta gegenueber dem S&P 500 (SPY).
    Berechnet via linearer Regression im rollierenden Fenster:
      r_i = alpha + beta * r_market + epsilon
    Alpha: abnormale Rendite bereinigt um Marktrisiko (CAPM).
    Beta: Marktrisiko-Sensitivitaet.
    """
    alphas  = []
    betas   = []
    idx     = asset_returns.index

    for i in range(len(idx)):
        if i < window:
            alphas.append(np.nan)
            betas.append(np.nan)
            continue
        ra = asset_returns.iloc[i-window:i].values
        rm = market_returns.reindex(asset_returns.index[i-window:i]).values
        mask = ~(np.isnan(ra) | np.isnan(rm))
        if mask.sum() < window // 2:
            alphas.append(np.nan)
            betas.append(np.nan)
            continue
        ra, rm = ra[mask], rm[mask]
        # OLS: beta = Cov(r_i, r_m) / Var(r_m)
        beta  = np.cov(ra, rm)[0, 1] / (np.var(rm) + 1e-12)
        alpha = ra.mean() - beta * rm.mean()
        alphas.append(alpha * 252)   # Annualisiert
        betas.append(beta)

    return pd.DataFrame({
        "alpha_spy" : alphas,
        "beta_spy"  : betas,
    }, index=idx)


@timer
def build_all_indicators(daily_prices: pd.DataFrame,
                          daily_returns: pd.DataFrame,
                          spy_returns: pd.Series,
                          tickers: list) -> dict:
    """
    Berechnet alle technischen Indikatoren fuer jedes Asset (taeglich).
    Gibt ein Dict {ticker: DataFrame mit allen Indikatoren} zurueck.
    """
    log.info("Berechne technische Indikatoren fuer alle Assets …")
    indicator_dict = {}

    for ticker in tickers:
        prices  = daily_prices[ticker]
        rets    = daily_returns[ticker]

        rsi     = compute_rsi(prices).rename("rsi")
        macd_df = compute_macd(prices)
        boll_df = compute_bollinger(prices)
        mom_df  = compute_momentum(rets)
        vol_df  = compute_volatility_features(rets)
        ab_df   = compute_alpha_beta(rets, spy_returns)

        combined = pd.concat([
            rsi, macd_df, boll_df, mom_df, vol_df, ab_df
        ], axis=1)

        indicator_dict[ticker] = combined

    log.info(f"  Indikatoren berechnet: {len(indicator_dict)} Assets, "
             f"{combined.shape[1]} Features je Asset")
    return indicator_dict


# ---------------------------------------------------------------------------
# 3. FEATURE-AGGREGATION: TAEGLICH → MONATLICH
# ---------------------------------------------------------------------------

def aggregate_to_monthly(indicator_dict: dict,
                          daily_returns: pd.DataFrame,
                          tickers: list) -> pd.DataFrame:
    """
    Aggregiert taeglich berechnete Indikatoren zu monatlichen Feature-Vektoren.

    Aggregationslogik:
      - Monats-Endwert (last):  RSI, %B, MACD-Linie, Alpha, Beta
        → repraesentiert den Zustand am Monatsende
      - Monatsdurchschnitt (mean): Volatilität, Momentum
        → stabiler als Einzeltageswert
      - Monatliche Rendite (sum der Log-Returns): Zielvariable t+1
    """
    rows = []

    for ticker in tickers:
        ind_df  = indicator_dict[ticker].copy()
        ret_col = daily_returns[ticker]

        # Monatliche Rendite als Zielvariable (naechster Monat)
        monthly_ret = ret_col.resample("ME").sum()

        # Feature-Aggregation pro Monat
        monthly_feat = pd.DataFrame(index=monthly_ret.index)

        # Endwerte (Zustand am Monatsende)
        for col in ["rsi", "macd", "macd_sig", "macd_hist",
                    "bb_pct_b", "bb_width", "alpha_spy", "beta_spy"]:
            if col in ind_df.columns:
                monthly_feat[col] = ind_df[col].resample("ME").last()

        # Durchschnittswerte (stabiler)
        for col in ["mom_21d", "mom_63d", "mom_126d", "mom_252d",
                    "vol_21d", "vol_63d"]:
            if col in ind_df.columns:
                monthly_feat[col] = ind_df[col].resample("ME").mean()

        monthly_feat["ticker"]          = ticker
        monthly_feat["monthly_ret"]     = monthly_ret
        # Zielvariable: Rendite im NAECHSTEN Monat (shift(-1))
        monthly_feat["target_next_month"] = monthly_ret.shift(-1)

        rows.append(monthly_feat)

    combined = pd.concat(rows).sort_index()
    combined = combined.replace([np.inf, -np.inf], np.nan)
    return combined


# ---------------------------------------------------------------------------
# 4. FEATURE-SPALTEN DEFINIEREN
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Technische Indikatoren
    "rsi",
    "macd", "macd_sig", "macd_hist",
    "bb_pct_b", "bb_width",
    # Momentum (verschiedene Horizonte)
    "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    # Volatilitaet
    "vol_21d", "vol_63d",
    # Alpha und Beta vs. SPY
    "alpha_spy", "beta_spy",
    # Vormonatliche Rendite (Kurzfrist-Signal)
    "monthly_ret",
]


# ---------------------------------------------------------------------------
# 5. MARKOWITZ MVO MIT LEDOIT-WOLF SHRINKAGE
# ---------------------------------------------------------------------------

class MarkowitzLedoitWolf:
    """
    Klassische Mean-Variance Optimization mit Ledoit-Wolf Kovarianzschaetzung.

    Problem der klassischen Stichproben-Kovarianzmatrix:
      Mit T Beobachtungen und N Assets wird die Matrix Sigma fuer T ~ N
      schlecht konditioniert. Kleine Schaetzfehler fuehren zu extremen,
      instabilen Portfolio-Gewichten ("Error-Maximizer"-Problem).

    Ledoit-Wolf (2004) Loesung: Shrinkage-Schaetzung
      Sigma_LW = (1-alpha) * Sigma_sample + alpha * Sigma_target
      wobei Sigma_target eine stabile, strukturierte Zielmatrix ist
      (z.B. diagonale Matrix, skalengewichtete Identitaet).
      Das optimale alpha wird analytisch bestimmt (kein manuelles Tuning).

    Referenz: Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator
    for large-dimensional covariance matrices. Journal of Multivariate
    Analysis, 88(2), 365-411.
    """

    def __init__(self, rf: float = RISK_FREE_RATE):
        self.rf = rf

    def estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Schaetzt die Kovarianzmatrix mit Ledoit-Wolf Shrinkage.
        Ergebnis: annualisierte Kovarianzmatrix.
        """
        lw = LedoitWolf()
        lw.fit(returns.values)
        cov_daily = lw.covariance_
        return cov_daily * 252   # Annualisierung

    def _neg_sharpe(self, weights, mu, cov):
        ret = float(weights @ mu)
        vol = float(np.sqrt(weights @ cov @ weights))
        return -(ret - self.rf) / vol if vol > 0 else 0.0

    def max_sharpe(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Maximiert die Sharpe Ratio (Tangential-Portfolio).
        Long-Only, vollstaendig investiert (Summe = 1).
        Positionsobergrenze: MAX_WEIGHT (Standard: 20%) je Asset.
        Solver: SLSQP (Sequential Least Squares Quadratic Programming).

        v3-Aenderung: bounds = (0, MAX_WEIGHT) statt (0, 1.0).
        Begruendung: Verhindert extreme Konzentration in Einzeltiteln,
        entspricht institutioneller Praxis (UCITS-Richtlinie: max. 10-20%).
        """
        n  = len(mu)
        w0 = np.ones(n) / n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds      = [(0.0, MAX_WEIGHT)] * n   # v3: Positionsobergrenze

        result = minimize(
            fun         = self._neg_sharpe,
            x0          = w0,
            args        = (mu, cov),
            method      = "SLSQP",
            bounds      = bounds,
            constraints = constraints,
            options     = {"maxiter": 2000, "ftol": 1e-12},
        )

        w = np.maximum(result.x, 0.0)
        w /= w.sum()
        return w

    def efficient_frontier(self,
                            mu: np.ndarray,
                            cov: np.ndarray,
                            n_points: int = N_FRONTIER) -> pd.DataFrame:
        """
        Berechnet N Punkte auf der Effizienzlinie.
        Je Renditeziel gamma: minimiere Varianz unter Rendite-Nebenbedingung.
        """
        n      = len(mu)
        bounds = [(0.0, 1.0)] * n

        # Untere Grenze: Minimum-Varianz-Portfolio
        res_mvp = minimize(
            lambda w: float(w @ cov @ w),
            np.ones(n) / n,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}],
        )
        ret_min = float(res_mvp.x @ mu)
        ret_max = float(mu.max())

        frontier = []
        for target in np.linspace(ret_min, ret_max, n_points):
            res = minimize(
                lambda w: float(w @ cov @ w),
                np.ones(n) / n,
                method      = "SLSQP",
                bounds      = bounds,
                constraints = [
                    {"type": "eq", "fun": lambda w: w.sum() - 1},
                    {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
                ],
                options = {"maxiter": 500, "ftol": 1e-12},
            )
            if res.success:
                vol = float(np.sqrt(res.x @ cov @ res.x))
                sr  = (target - self.rf) / vol if vol > 0 else 0.0
                frontier.append({
                    "ret" : target,
                    "vol" : vol,
                    "sr"  : sr,
                    "w"   : res.x.copy(),
                })
        return pd.DataFrame(frontier)


# ---------------------------------------------------------------------------
# 6. RANDOM FOREST MIT TIMESERIESPLIT + RANDOMIZEDSEARCHCV
# ---------------------------------------------------------------------------

class RFPortfolioOptimizer:
    """
    Random-Forest-gestuetzte Portfoliooptimierung.

    Architektur (konsistent monatlich):
      1. Features: monatlich aggregierte technische Indikatoren
      2. Zielvariable: Rendite im naechsten Monat (t+1)
      3. Training: Rolling Window (TRAIN_YEARS * 12 Monate)
      4. Hyperparameter-Tuning: RandomizedSearchCV mit TimeSeriesSplit
         -> vermeidet Look-Ahead-Bias (keine zufaellige Train/Test-Aufteilung)
      5. Vorhersage: mu_RF fuer jeden Asset im naechsten Monat
      6. Portfoliooptimierung: Sharpe-Maximierung mit mu_RF als Inputvektor
         (analog Markowitz, aber mit ML-Prognosen statt historischem Mittelwert)

    Methodische Begruendung TimeSeriesSplit:
      Standard-K-Fold-CV muss fuer Zeitreihen verboten werden: zufaellige
      Aufteilung fuehrt dazu, dass Zukunftsdaten die Vergangenheit trainieren
      (Look-Ahead-Bias). TimeSeriesSplit garantiert: Testdaten liegen immer
      zeitlich nach den Trainingsdaten.

    Referenz: Fischer, T. & Krauss, C. (2018). Deep learning with long
    short-term memory networks for financial market predictions.
    European Journal of Operational Research, 270(2), 654-669.
    """

    def __init__(self, rf: float = RISK_FREE_RATE,
                 n_iter: int = RF_N_ITER,
                 cv_splits: int = RF_CV_SPLITS):
        self.rf        = rf
        self.n_iter    = n_iter
        self.cv_splits = cv_splits
        self.best_estimator_  = None
        self.best_params_     = {}
        self._mvo             = MarkowitzLedoitWolf(rf=rf)

    def _build_pipeline(self) -> Pipeline:
        """
        sklearn Pipeline: StandardScaler -> RandomForestRegressor.
        Der Scaler normalisiert alle Features auf N(0,1) — wichtig da
        Features sehr unterschiedliche Skalen haben (RSI: 0-100, Alpha: ~0.01).
        """
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestRegressor(random_state=42, n_jobs=-1)),
        ])

    def _param_grid(self) -> dict:
        """
        Hyperparameter-Suchraum fuer RandomizedSearchCV.
        Begruendung der Ranges:
          n_estimators: mehr Baeume = stabiler, aber langsamer (100-500 ok)
          max_depth: begrenzt Overfitting (5-15 typisch fuer Finanzdaten)
          min_samples_leaf: Mindestgroesse Blatt -> Regularisierung
          max_features: Anteil Features je Split -> Diversitaet der Baeume
          max_samples: Bootstrap-Anteil -> Datenvarianz reduzieren
        """
        return {
            "rf__n_estimators"     : randint(100, 500),
            "rf__max_depth"        : randint(3, 15),
            "rf__min_samples_leaf" : randint(3, 20),
            "rf__max_features"     : uniform(0.3, 0.6),
            "rf__max_samples"      : uniform(0.6, 0.35),
        }

    @timer
    def fit_with_tuning(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series) -> None:
        """
        Trainiert den RF mit Hyperparameter-Tuning via RandomizedSearchCV.

        TimeSeriesSplit garantiert: im i-ten Split liegen alle Testdaten
        zeitlich nach allen Trainingsdaten dieses Splits.
        Scoring: negatives MSE (sklearn-Konvention: hoeherer Wert = besser).
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        search = RandomizedSearchCV(
            estimator           = self._build_pipeline(),
            param_distributions = self._param_grid(),
            n_iter              = self.n_iter,
            scoring             = "neg_mean_squared_error",
            cv                  = tscv,
            random_state        = 42,
            n_jobs              = -1,
            refit               = True,   # Bestes Modell auf gesamten Trainingsdaten
        )

        search.fit(X_train.values, y_train.values)
        self.best_estimator_ = search.best_estimator_
        self.best_params_    = search.best_params_

        log.info(f"    Beste RF-Params: n_est={self.best_params_.get('rf__n_estimators','?')}, "
                 f"depth={self.best_params_.get('rf__max_depth','?')}, "
                 f"leaf={self.best_params_.get('rf__min_samples_leaf','?')}")

    def predict_monthly_returns(self,
                                 X_current: pd.DataFrame) -> np.ndarray:
        """
        Gibt monatliche Renditeprognosen je Asset zurueck (annualisiert * 12).
        """
        if self.best_estimator_ is None:
            raise RuntimeError("Zuerst fit_with_tuning() aufrufen.")
        preds = self.best_estimator_.predict(X_current.values)
        return preds * 12   # Monatlich -> annualisiert

    def optimize(self,
                 mu_predicted: np.ndarray,
                 cov: np.ndarray) -> np.ndarray:
        """Sharpe-Maximierung mit ML-Renditeprognosen als Inputvektor."""
        return self._mvo.max_sharpe(mu_predicted, cov)


# ---------------------------------------------------------------------------
# 7. DATEN LADEN
# ---------------------------------------------------------------------------

@timer
def download_data(tickers: list,
                  spy_ticker: str,
                  start: str,
                  end: str):
    """Laedt adjustierte Schlusskurse und berechnet Log-Renditen."""
    log.info(f"Lade Marktdaten: {len(tickers)} Assets + SPY | {start} -> {end}")

    all_tickers = tickers + [spy_ticker]
    raw = yf.download(all_tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]

    # Fehlende Assets pruefen
    missing = [t for t in all_tickers if t not in raw.columns]
    if missing:
        log.warning(f"Fehlende Tickers: {missing}")

    available = [t for t in tickers if t in raw.columns]
    prices    = raw[available + [spy_ticker]].dropna(how="any")

    log.info(f"  Verfuegbare Assets: {len(available)} | "
             f"Handelstage: {len(prices)}")

    daily_returns = np.log(prices / prices.shift(1)).dropna()

    spy_ret   = daily_returns[spy_ticker]
    asset_ret = daily_returns[available]
    asset_px  = prices[available]

    return asset_px, asset_ret, spy_ret, available


# ---------------------------------------------------------------------------
# 8. BACKTEST ENGINE
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
      1. Trainingsdaten der letzten TRAIN_YEARS Jahre extrahieren
      2. Markowitz: Ledoit-Wolf Sigma schatzen + Sharpe maximieren
      3. RF: Features aufbauen, Tuning, Vorhersage, Sharpe maximieren
      4. Equal Weight: 1/N Gewichtung
      5. Portfoliorenditen der naechsten 30 Tage berechnen
    """
    log.info("Baue Feature-Matrix auf (taeglich -> monatlich) …")
    monthly_data = aggregate_to_monthly(indicator_dict, asset_returns, tickers)

    # Nur Monate ab BACKTEST_START verwenden
    backtest_months = monthly_data.index.unique()
    backtest_months = backtest_months[backtest_months >= BACKTEST_START]

    n   = len(tickers)
    mvo = MarkowitzLedoitWolf()
    rfo = RFPortfolioOptimizer()

    results_mvo, results_rf, results_ew = [], [], []
    hist_w_mvo, hist_w_rf               = [], []

    log.info(f"Starte Backtest: {len(backtest_months)} Rebalancing-Monate …")

    for i, month_end in enumerate(backtest_months[:-1]):
        t0 = time.perf_counter()

        train_start = month_end - pd.DateOffset(years=TRAIN_YEARS)

        # ---- Trainingsdaten extrahieren -----------------------------------
        train_daily = asset_returns[
            (asset_returns.index >= train_start) &
            (asset_returns.index <= month_end)
        ]
        train_monthly = monthly_data[
            (monthly_data.index >= train_start) &
            (monthly_data.index <= month_end)
        ]

        # Mindest-Datencheck
        if len(train_daily) < 252 or len(train_monthly.index.unique()) < 24:
            log.warning(f"  [{month_end.date()}] Zu wenige Daten, ueberspringe.")
            continue

        # Naechster Rebalancing-Termin (Halteperiode)
        next_month  = backtest_months[i + 1]
        hold_period = asset_returns[
            (asset_returns.index > month_end) &
            (asset_returns.index <= next_month)
        ]
        if hold_period.empty:
            continue

        # ---- Kovarianzmatrix (Ledoit-Wolf) --------------------------------
        cov_ann = mvo.estimate_covariance(train_daily)

        # ---- Markowitz MVO ------------------------------------------------
        mu_hist = train_daily.mean().values * 252   # Historische Annualisierung

        try:
            w_mvo = mvo.max_sharpe(mu_hist, cov_ann)
        except Exception as e:
            log.warning(f"  MVO-Fehler {month_end.date()}: {e}")
            w_mvo = np.ones(n) / n

        # ---- Random Forest ------------------------------------------------
        # v3-FIX LOOK-AHEAD-BIAS:
        # Die Zielvariable target_next_month fuer Monat T = Rendite von T+1.
        # Wenn train_monthly bis month_end geht, enthaelt der letzte Monat
        # als Ziel die Rendite der Halteperiode (= Zukunft fuer das Modell).
        # Loesung: RF-Training nur auf Monate VOR month_end (< month_end).
        # Der letzte Monat (month_end) dient nur als Prognose-Input (X_current),
        # nicht als Trainingsbeobachtung.
        last_train_month = month_end - pd.DateOffset(months=1)

        feat_rows   = []
        target_rows = []

        for ticker in tickers:
            # Trainingszeilen: nur bis last_train_month (exkl. month_end)
            t_rows = train_monthly[
                (train_monthly["ticker"] == ticker) &
                (train_monthly.index <= last_train_month)
            ]
            valid  = t_rows[FEATURE_COLS + ["target_next_month"]].dropna()
            if len(valid) < 12:
                continue
            feat_rows.append(valid[FEATURE_COLS])
            target_rows.append(valid["target_next_month"])

        if len(feat_rows) < n // 2:
            w_rf   = np.ones(n) / n
            mu_rf  = mu_hist.copy()
        else:
            X_train_rf = pd.concat(feat_rows)
            y_train_rf = pd.concat(target_rows)

            try:
                rfo.fit_with_tuning(X_train_rf, y_train_rf)

                # Aktuelle Feature-Zeile: letzter verfuegbarer Monat je Asset
                X_current_rows = []
                for ticker in tickers:
                    t_rows = train_monthly[train_monthly["ticker"] == ticker]
                    valid  = t_rows[FEATURE_COLS].dropna()
                    if len(valid) > 0:
                        X_current_rows.append(valid.iloc[-1])
                    else:
                        X_current_rows.append(
                            pd.Series(np.zeros(len(FEATURE_COLS)),
                                      index=FEATURE_COLS)
                        )
                X_current = pd.DataFrame(X_current_rows, columns=FEATURE_COLS)
                mu_rf     = rfo.predict_monthly_returns(X_current)
                w_rf      = rfo.optimize(mu_rf, cov_ann)

            except Exception as e:
                log.warning(f"  RF-Fehler {month_end.date()}: {e}")
                w_rf  = np.ones(n) / n
                mu_rf = mu_hist.copy()

        # ---- Equal Weight -------------------------------------------------
        w_ew = np.ones(n) / n

        # ---- Transaktionskosten (v3) --------------------------------------
        # Turnover = Summe der absoluten Gewichtsaenderungen je Asset.
        # Bei erstem Rebalancing: voller Turnover (Kauf aus Cash).
        # Kosten = Turnover * TRANSACTION_COST, abgezogen als einmaliger
        # Abschlag vom ersten Handelstag der Halteperiode.
        #
        # Methodik: Frazzini et al. (2015): einfacher proportionaler
        # Transaktionskostenansatz, konservativ fuer liquide Large-Caps.

        if i == 0:
            # Erster Rebalancing-Schritt: Kauf aus 100% Cash
            turnover_mvo = w_mvo.sum()    # = 1.0 (volles Investment)
            turnover_rf  = w_rf.sum()
            turnover_ew  = w_ew.sum()
        else:
            prev_w_mvo = hist_w_mvo[-1].values if hist_w_mvo else np.ones(n) / n
            prev_w_rf  = hist_w_rf[-1].values  if hist_w_rf  else np.ones(n) / n
            # Turnover = halbe Summe |Gewichtsaenderung| (einseitig gemessen)
            turnover_mvo = np.abs(w_mvo - prev_w_mvo).sum() / 2
            turnover_rf  = np.abs(w_rf  - prev_w_rf ).sum() / 2
            turnover_ew  = 0.0   # Equal Weight rebalanciert kaum

        cost_mvo = turnover_mvo * TRANSACTION_COST
        cost_rf  = turnover_rf  * TRANSACTION_COST
        cost_ew  = turnover_ew  * TRANSACTION_COST

        # ---- Portfoliorenditen berechnen (nach Kosten) --------------------
        ret_mvo = (hold_period * w_mvo).sum(axis=1).copy()
        ret_rf  = (hold_period * w_rf ).sum(axis=1).copy()
        ret_ew  = (hold_period * w_ew ).sum(axis=1).copy()

        # Kosten einmalig am ersten Handelstag der Halteperiode abziehen
        if len(ret_mvo) > 0:
            ret_mvo.iloc[0] -= cost_mvo
            ret_rf.iloc[0]  -= cost_rf
            ret_ew.iloc[0]  -= cost_ew

        results_mvo.append(ret_mvo)
        results_rf.append( ret_rf)
        results_ew.append( ret_ew)

        hist_w_mvo.append(pd.Series(w_mvo, index=tickers, name=month_end))
        hist_w_rf.append( pd.Series(w_rf,  index=tickers, name=month_end))

        elapsed = time.perf_counter() - t0
        ret_m, vol_m, sr_m = portfolio_perf(w_mvo, mu_hist, cov_ann)
        log.info(
            f"  [{i+1:>3}/{len(backtest_months)-1}] {month_end.date()} | "
            f"MVO-SR: {sr_m:.3f} | "
            f"Turnover MVO: {turnover_mvo*100:.1f}% | "
            f"Kosten MVO: {cost_mvo*100:.3f}% | "
            f"Dauer: {elapsed:.1f}s"
        )

    # Zusammenfuehren
    returns_df = pd.DataFrame({
        "Markowitz MVO" : pd.concat(results_mvo).sort_index(),
        "Random Forest" : pd.concat(results_rf ).sort_index(),
        "Equal Weight"  : pd.concat(results_ew ).sort_index(),
    })

    weights_mvo_df = pd.DataFrame(hist_w_mvo).T
    weights_rf_df  = pd.DataFrame(hist_w_rf ).T

    return {
        "returns"     : returns_df,
        "weights_mvo" : weights_mvo_df,
        "weights_rf"  : weights_rf_df,
        "mu_hist"     : mu_hist,
        "mu_rf"       : mu_rf,
        "cov_ann"     : cov_ann,
        "w_mvo_last"  : w_mvo,
        "w_rf_last"   : w_rf,
    }


# ---------------------------------------------------------------------------
# 9. PERFORMANCE-KENNZAHLEN
# ---------------------------------------------------------------------------

def compute_metrics(returns_df: pd.DataFrame,
                    rf: float = RISK_FREE_RATE) -> pd.DataFrame:
    """
    Vollstaendige Performance-Analyse aller Portfoliostrategien.

    Kennzahlen:
      CAGR            : Jaehrliche Wachstumsrate (geometrisch)
      Gesamtrendite   : Totaler Wertzuwachs im Backtestzeitraum
      Volatilitaet    : Annualisierte Standardabweichung der taegl. Renditen
      Sharpe Ratio    : Risikobereingte Rendite (Sharpe, 1966)
      Sortino Ratio   : Sharpe nur mit Downside-Volatilitaet (Sortino, 1994)
      Max. Drawdown   : Groesstes Wertverlust vom Hoechststand
      Calmar Ratio    : CAGR / |Max. Drawdown| (Risiko-Effizienz-Mass)
      VaR 95 %        : Historischer Value at Risk (95 %-Konfidenz, taegl.)
      Hit Rate        : Anteil positiver Handelstage
    """
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
            "VaR 95 % (taegl., %)"      : round(np.percentile(r, 5) * 100, 2),
            "Hit Rate (%)"               : round((r > 0).mean() * 100, 2),
        }
    return pd.DataFrame(metrics).T


# ---------------------------------------------------------------------------
# 10. VISUALISIERUNGEN
# ---------------------------------------------------------------------------

def plot_cumulative_returns(returns_df: pd.DataFrame,
                             output_path: str) -> None:
    """Abbildung 1: Kumulierte Portfoliorenditen mit Drawdown-Schattierung."""
    log.info("Plot 1: Kumulierte Renditen …")
    fig, (ax_main, ax_dd) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    for col in returns_df.columns:
        color = COLORS[col]
        r     = returns_df[col].dropna()
        cum   = (1 + r).cumprod()

        ax_main.plot(cum.index, cum.values,
                     label=col, color=color, linewidth=2, zorder=3)
        ax_main.fill_between(cum.index, cum, cum.cummax(),
                              color=color, alpha=0.07)

        # Drawdown-Panel
        roll_max = cum.cummax()
        dd       = (cum - roll_max) / roll_max * 100
        ax_dd.fill_between(dd.index, dd.values, 0,
                            color=color, alpha=0.45, label=col)
        ax_dd.plot(dd.index, dd.values,
                   color=color, linewidth=0.8, alpha=0.7)

    ax_main.axhline(1, color="black", linewidth=0.8,
                    linestyle="--", alpha=0.4, label="Startkapital")
    ax_main.set_ylabel("Kumulierter Wert (Startkapital = 1 \u20ac)")
    ax_main.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.2f} \u20ac"))
    ax_main.legend(loc="upper left", framealpha=0.9)
    ax_main.set_title(
        "Kumulierte Portfoliorenditen im Vergleich\n"
        "(Rollierendes Backtest, monatliches Rebalancing, Ledoit-Wolf + Techn. Indikatoren)",
        pad=12,
    )

    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Datum")
    ax_dd.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax_dd.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.0f}%"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  -> {output_path}")


def plot_weight_heatmap(weights_df: pd.DataFrame,
                         title: str,
                         output_path: str) -> None:
    """Abbildungen 2 & 3: Heatmap der Portfoliogewichtungen ueber Zeit."""
    log.info(f"Plot Heatmap: {title} …")

    data = (weights_df * 100).T   # Spalten = Assets, Zeilen = Zeitpunkte
    col_labels = [
        c.strftime("%b %Y") if hasattr(c, "strftime") else str(c)
        for c in data.index
    ]

    fig, ax = plt.subplots(figsize=(max(14, len(col_labels) * 0.55), 7))

    sns.heatmap(
        data.T,
        ax         = ax,
        cmap       = "YlOrRd",
        linewidths = 0.35,
        linecolor  = "white",
        annot      = True,
        fmt        = ".1f",
        annot_kws  = {"size": 8},
        cbar_kws   = {"label": "Gewicht (%)", "shrink": 0.75},
        vmin=0, vmax=60,
    )

    visible = [lbl if j % 2 == 0 else ""
               for j, lbl in enumerate(col_labels)]
    ax.set_xticklabels(visible, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(
        f"Portfolio-Gewichtungen: {title}\n"
        "(Werte in % | je Rebalancing-Monat)",
        pad=12,
    )
    ax.set_xlabel("Rebalancing-Datum")
    ax.set_ylabel("Asset")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  -> {output_path}")


def plot_efficient_frontier(mu_hist: np.ndarray,
                             cov_ann: np.ndarray,
                             tickers: list,
                             w_mvo: np.ndarray,
                             w_rf: np.ndarray,
                             w_ew: np.ndarray,
                             mu_rf: np.ndarray,
                             rf: float,
                             output_path: str) -> None:
    """
    Abbildung 4: Vollstaendige Effizienzlinie (Efficient Frontier).

    Dargestellt:
      - Effizienzkurve (Long-Only MV-Frontier)
      - Minimum-Varianz-Portfolio (MVP)
      - Tangential-Portfolio (Markowitz MVO, max. Sharpe)
      - RF-optimiertes Portfolio
      - Equal-Weight Portfolio
      - Capital Market Line (CML) vom risikolosen Zinssatz
      - Einzelne Assets als Referenzpunkte
    """
    log.info("Plot 4: Efficient Frontier …")
    mvo_opt  = MarkowitzLedoitWolf(rf=rf)
    frontier = mvo_opt.efficient_frontier(mu_hist, cov_ann)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Effizienzkurve
    if not frontier.empty:
        scatter = ax.scatter(
            frontier["vol"] * 100,
            frontier["ret"] * 100,
            c      = frontier["sr"],
            cmap   = "viridis",
            s      = 18,
            zorder = 2,
            alpha  = 0.85,
            label  = "Efficient Frontier",
        )
        cbar = plt.colorbar(scatter, ax=ax, pad=0.01, shrink=0.8)
        cbar.set_label("Sharpe Ratio", fontsize=9)

        ax.plot(
            frontier["vol"] * 100,
            frontier["ret"] * 100,
            color=COLORS["frontier"], linewidth=1.5,
            alpha=0.6, zorder=1,
        )

    # Capital Market Line
    ret_tan, vol_tan, _ = portfolio_perf(w_mvo, mu_hist, cov_ann, rf)
    if vol_tan > 0:
        cml_vols = np.linspace(0, vol_tan * 1.6, 120)
        cml_rets = rf + (ret_tan - rf) / vol_tan * cml_vols
        ax.plot(cml_vols * 100, cml_rets * 100,
                color=COLORS["cml"], linestyle="--", linewidth=1.6,
                alpha=0.9, label=f"Capital Market Line (rf = {rf*100:.1f}%)",
                zorder=3)
        ax.scatter([0], [rf * 100], marker="*", s=180,
                   color=COLORS["cml"], zorder=7,
                   label=f"Risikoloser Zinssatz ({rf*100:.1f}%)")

    # Einzelne Assets
    for i, t in enumerate(tickers):
        a_vol = np.sqrt(cov_ann[i, i]) * 100
        a_ret = mu_hist[i] * 100
        ax.scatter(a_vol, a_ret, color="lightsteelblue",
                   s=55, zorder=4, alpha=0.8, edgecolors="steelblue", linewidths=0.5)
        ax.annotate(t, (a_vol, a_ret),
                    textcoords="offset points", xytext=(5, 2),
                    fontsize=7.5, color="dimgrey")

    # Portfolio-Punkte
    def _add_pt(w, mu, label, color, marker, size=230):
        ret, vol, sr = portfolio_perf(w, mu, cov_ann, rf)
        ax.scatter(vol * 100, ret * 100,
                   color=color, marker=marker, s=size,
                   zorder=8, edgecolors="black", linewidths=1.0,
                   label=(f"{label}\n"
                          f"  Rendite: {ret*100:.1f}%  |  "
                          f"Vola: {vol*100:.1f}%  |  "
                          f"Sharpe: {sr:.2f}"))

    _add_pt(w_mvo, mu_hist, "Markowitz MVO (Tangential)", COLORS["Markowitz MVO"], "D")
    _add_pt(w_rf,  mu_rf,   "Random Forest MVO",          COLORS["Random Forest"], "^")
    _add_pt(w_ew,  mu_hist, "Equal Weight (1/N)",          COLORS["Equal Weight"],  "s", 190)

    # MVP
    if not frontier.empty:
        mvp = frontier.loc[frontier["vol"].idxmin()]
        ax.scatter(mvp["vol"] * 100, mvp["ret"] * 100,
                   color=COLORS["mvp"], marker="P", s=210, zorder=8,
                   edgecolors="black", linewidths=0.9,
                   label=f"Minimum-Varianz-Portfolio (MVP)\n"
                         f"  Vol: {mvp['vol']*100:.1f}%  |  "
                         f"Sharpe: {mvp['sr']:.2f}")

    ax.set_title(
        "Effizienzkurve (Efficient Frontier) mit Capital Market Line\n"
        "Markowitz (1952) | Kovarianzschaetzung: Ledoit-Wolf (2004)",
        pad=12,
    )
    ax.set_xlabel("Annualisierte Volatilitaet (%)")
    ax.set_ylabel("Annualisierte Erwartungsrendite (%)")
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}%"))
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}%"))
    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.92,
        bbox_to_anchor=(1.18, 1),
        borderaxespad=0,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  -> {output_path}")


def plot_performance_metrics(metrics_df: pd.DataFrame,
                              output_path: str) -> None:
    """Abbildung 5: Sechspanel-Balkendiagramm aller Kennzahlen."""
    log.info("Plot 5: Performance-Kennzahlen …")

    display = [
        ("CAGR (%)",                "CAGR (% p.a.)"),
        ("Sharpe Ratio",            "Sharpe Ratio"),
        ("Sortino Ratio",           "Sortino Ratio"),
        ("Max. Drawdown (%)",       "Max. Drawdown (%)"),
        ("Calmar Ratio",            "Calmar Ratio"),
        ("Annualisierte Vola. (%)", "Volatilitaet (% p.a.)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    palette = [COLORS[c] for c in metrics_df.index if c in COLORS]
    if len(palette) < len(metrics_df):
        palette += ["#aaaaaa"] * (len(metrics_df) - len(palette))

    for ax, (key, label) in zip(axes.flatten(), display):
        vals = metrics_df[key]
        bars = ax.barh(vals.index, vals.values,
                       color=palette[:len(vals)],
                       height=0.5, edgecolor="white")
        span = vals.abs().max() if vals.abs().max() > 0 else 1
        for bar, val in zip(bars, vals):
            sign = "+" if val > 0 else ""
            ax.text(
                val + span * 0.03,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.2f}",
                va="center", ha="left", fontsize=9, fontweight="bold",
            )
        ax.set_title(label, fontweight="bold", pad=8)
        ax.axvline(0, color="black", linewidth=0.7, alpha=0.4)
        ax.set_xlim(vals.min() - span * 0.3, vals.max() + span * 0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(
        "Performance-Kennzahlen im Vergleich | Rollierendes Backtest (2015-2024)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  -> {output_path}")


def plot_rolling_sharpe(returns_df: pd.DataFrame,
                         output_path: str,
                         window: int = 252) -> None:
    """Abbildung 6: Rollierender 1-Jahres-Sharpe Ratio."""
    log.info("Plot 6: Rollierender Sharpe Ratio …")
    rf_daily = RISK_FREE_RATE / 252

    fig, ax = plt.subplots(figsize=(13, 5))
    for col in returns_df.columns:
        r  = returns_df[col]
        rs = (r.rolling(window).mean() - rf_daily) / r.rolling(window).std() * np.sqrt(252)
        ax.plot(rs.index, rs.values,
                label=col, color=COLORS[col], linewidth=1.8, alpha=0.9)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(1, color="grey",  linewidth=0.6, linestyle=":",  alpha=0.5)
    ax.set_title(f"Rollierender Sharpe Ratio ({window}-Tage-Fenster = ca. 1 Jahr)", pad=10)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Sharpe Ratio (annualisiert)")
    ax.legend(loc="upper left")
    ax.set_xlim(returns_df.index[0], returns_df.index[-1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  -> {output_path}")


def plot_feature_importance(rf_optimizer: RFPortfolioOptimizer,
                             output_path: str) -> None:
    """
    Abbildung 7: Feature Importance des Random Forest.
    Zeigt welche technischen Indikatoren die staerkste Prognosequalitaet besitzen.
    Methodisch: mittlere Abnahme der Impuritaet (Mean Decrease in Impurity).
    """
    log.info("Plot 7: Feature Importance …")
    if rf_optimizer.best_estimator_ is None:
        log.warning("RF nicht trainiert, ueberspringe Feature Importance.")
        return

    try:
        rf_step = rf_optimizer.best_estimator_.named_steps["rf"]
        imps    = rf_step.feature_importances_

        importance_df = pd.DataFrame({
            "Feature"    : FEATURE_COLS,
            "Importance" : imps,
        }).sort_values("Importance", ascending=True)

        # Lesbarer Feature-Name
        rename = {
            "rsi"        : "RSI (14)",
            "macd"       : "MACD-Linie",
            "macd_sig"   : "MACD-Signal",
            "macd_hist"  : "MACD-Histogramm",
            "bb_pct_b"   : "Bollinger %B",
            "bb_width"   : "Bollinger Bandwidth",
            "mom_21d"    : "Momentum 1M",
            "mom_63d"    : "Momentum 3M",
            "mom_126d"   : "Momentum 6M",
            "mom_252d"   : "Momentum 12M",
            "vol_21d"    : "Volatilitaet 1M",
            "vol_63d"    : "Volatilitaet 3M",
            "alpha_spy"  : "Alpha vs. SPY",
            "beta_spy"   : "Beta vs. SPY",
            "monthly_ret": "Vormonatsrendite",
        }
        importance_df["Feature"] = importance_df["Feature"].map(rename).fillna(
            importance_df["Feature"]
        )

        fig, ax = plt.subplots(figsize=(9, 7))
        bars = ax.barh(
            importance_df["Feature"],
            importance_df["Importance"] * 100,
            color="#1f77b4", edgecolor="white", height=0.65,
        )
        for bar, val in zip(bars, importance_df["Importance"] * 100):
            ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", ha="left", fontsize=9)

        ax.set_title(
            "Random Forest: Feature Importance\n"
            "(Mean Decrease in Impurity | letzter Trainingsschritt)",
            pad=12,
        )
        ax.set_xlabel("Relative Wichtigkeit (%)")
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}%"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        log.info(f"  -> {output_path}")
    except Exception as e:
        log.warning(f"Feature Importance Fehler: {e}")


# ---------------------------------------------------------------------------
# 11. CSV-EXPORT
# ---------------------------------------------------------------------------

def save_all_csv(returns_df: pd.DataFrame,
                 metrics_df: pd.DataFrame,
                 weights_mvo: pd.DataFrame,
                 weights_rf: pd.DataFrame,
                 output_dir: str) -> None:
    log.info("Speichere CSV-Dateien …")
    cum = (1 + returns_df).cumprod()
    cum.to_csv(            os.path.join(output_dir, "cumulative_returns.csv"),  float_format="%.6f")
    returns_df.to_csv(     os.path.join(output_dir, "daily_returns.csv"),       float_format="%.6f")
    metrics_df.to_csv(     os.path.join(output_dir, "performance_metrics.csv"), float_format="%.4f")
    weights_mvo.T.to_csv(  os.path.join(output_dir, "weights_markowitz.csv"),   float_format="%.4f")
    weights_rf.T.to_csv(   os.path.join(output_dir, "weights_rf.csv"),          float_format="%.4f")

    # v3: Turnover-Tabelle aus Gewichts-Differenzen berechnen und speichern
    if not weights_mvo.empty and not weights_rf.empty:
        to_mvo = weights_mvo.diff(axis=1).abs().sum(axis=0) / 2
        to_rf  = weights_rf.diff(axis=1).abs().sum(axis=0) / 2
        turnover_df = pd.DataFrame({
            "Turnover_Markowitz" : to_mvo,
            "Turnover_RF"        : to_rf,
        })
        turnover_df.to_csv(os.path.join(output_dir, "turnover.csv"),
                           float_format="%.4f")
        log.info(f"  Durchschn. Turnover MVO: {to_mvo.mean()*100:.1f}% | "
                 f"RF: {to_rf.mean()*100:.1f}%")

    log.info("  -> 6 CSV-Dateien gespeichert.")


# ---------------------------------------------------------------------------
# 12. HAUPTPROGRAMM
# ---------------------------------------------------------------------------

def main():
    t0_main = time.perf_counter()
    log.info("=" * 68)
    log.info("  PORTFOLIO-OPTIMIERUNG V3 | MARKOWITZ (LW) vs. RANDOM FOREST")
    log.info("  Korrekturen: Look-Ahead-Bias | Max-Weight 20% | Transaktionskosten")
    log.info(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 68)

    if not YFINANCE_AVAILABLE:
        log.error("yfinance fehlt. Bitte 'pip install yfinance' ausfuehren.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info(f"Output-Ordner: {os.path.abspath(OUTPUT_DIR)}")

    # ------------------------------------------------------------------
    # A: Daten laden
    # ------------------------------------------------------------------
    asset_prices, asset_returns, spy_returns, tickers = download_data(
        TICKERS, SPY_TICKER, START_DATE, END_DATE
    )

    # ------------------------------------------------------------------
    # B: Technische Indikatoren berechnen
    # ------------------------------------------------------------------
    indicator_dict = build_all_indicators(
        asset_prices, asset_returns, spy_returns, tickers
    )

    # ------------------------------------------------------------------
    # C: Backtest durchfuehren
    # ------------------------------------------------------------------
    results = run_backtest(
        asset_prices, asset_returns, spy_returns, tickers, indicator_dict
    )

    returns_df  = results["returns"]
    weights_mvo = results["weights_mvo"]
    weights_rf  = results["weights_rf"]
    mu_hist     = results["mu_hist"]
    mu_rf       = results["mu_rf"]
    cov_ann     = results["cov_ann"]
    w_mvo_last  = results["w_mvo_last"]
    w_rf_last   = results["w_rf_last"]
    w_ew        = np.ones(len(tickers)) / len(tickers)

    # ------------------------------------------------------------------
    # D: Kennzahlen
    # ------------------------------------------------------------------
    metrics_df = compute_metrics(returns_df)
    print("\n" + "=" * 65)
    print("  PERFORMANCE-KENNZAHLEN (VOLLSTAENDIGE UEBERSICHT)")
    print("=" * 65)
    print(metrics_df.to_string())
    print("=" * 65 + "\n")

    # ------------------------------------------------------------------
    # E: Visualisierungen
    # ------------------------------------------------------------------
    plot_cumulative_returns(returns_df,
        os.path.join(OUTPUT_DIR, "01_kumulierte_renditen.png"))

    if not weights_mvo.empty:
        plot_weight_heatmap(weights_mvo, "Markowitz MVO (Ledoit-Wolf)",
            os.path.join(OUTPUT_DIR, "02_gewichte_markowitz.png"))

    if not weights_rf.empty:
        plot_weight_heatmap(weights_rf, "Random Forest MVO",
            os.path.join(OUTPUT_DIR, "03_gewichte_random_forest.png"))

    plot_efficient_frontier(
        mu_hist=mu_hist, cov_ann=cov_ann, tickers=tickers,
        w_mvo=w_mvo_last, w_rf=w_rf_last, w_ew=w_ew,
        mu_rf=mu_rf, rf=RISK_FREE_RATE,
        output_path=os.path.join(OUTPUT_DIR, "04_efficient_frontier.png"),
    )

    plot_performance_metrics(metrics_df,
        os.path.join(OUTPUT_DIR, "05_performance_kennzahlen.png"))

    plot_rolling_sharpe(returns_df,
        os.path.join(OUTPUT_DIR, "06_rollierender_sharpe.png"))

    # Feature Importance (RF-Optimizer aus letztem Backtest-Schritt)
    rfo_final = RFPortfolioOptimizer()
    # Letzter Trainingsschritt: kurz nachtrainieren fuer den Plot
    monthly_data = aggregate_to_monthly(indicator_dict, asset_returns, tickers)
    feat_rows, target_rows = [], []
    for ticker in tickers:
        t_rows = monthly_data[monthly_data["ticker"] == ticker]
        valid  = t_rows[FEATURE_COLS + ["target_next_month"]].dropna()
        if len(valid) >= 12:
            feat_rows.append(valid[FEATURE_COLS])
            target_rows.append(valid["target_next_month"])
    if feat_rows:
        rfo_final.fit_with_tuning(pd.concat(feat_rows), pd.concat(target_rows))
        plot_feature_importance(rfo_final,
            os.path.join(OUTPUT_DIR, "07_feature_importance.png"))

    # ------------------------------------------------------------------
    # F: CSV-Export
    # ------------------------------------------------------------------
    save_all_csv(returns_df, metrics_df, weights_mvo, weights_rf, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # G: Kompakte Zusammenfassung
    # ------------------------------------------------------------------
    total = time.perf_counter() - t0_main
    log.info("=" * 68)
    log.info(f"  FERTIG | Gesamtlaufzeit: {total/60:.1f} min ({total:.0f}s)")
    log.info(f"  Alle Dateien in: {os.path.abspath(OUTPUT_DIR)}/")
    log.info("=" * 68)

    print("\n" + "=" * 55)
    print("  KURZ-ZUSAMMENFASSUNG")
    print("=" * 55)
    for strat in metrics_df.index:
        m = metrics_df.loc[strat]
        print(f"\n  [{strat}]")
        print(f"    CAGR            : {m['CAGR (%)']:>7.2f} %")
        print(f"    Gesamtrendite   : {m['Gesamtrendite (%)']:>7.2f} %")
        print(f"    Sharpe Ratio    : {m['Sharpe Ratio']:>7.4f}")
        print(f"    Sortino Ratio   : {m['Sortino Ratio']:>7.4f}")
        print(f"    Max. Drawdown   : {m['Max. Drawdown (%)']:>7.2f} %")
        print(f"    Calmar Ratio    : {m['Calmar Ratio']:>7.4f}")
    print("=" * 55)
    print(f"\n  Ausgabe-Dateien: {os.path.abspath(OUTPUT_DIR)}/\n")


if __name__ == "__main__":
    main()