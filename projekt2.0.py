# =============================================================================
# PORTFOLIO-OPTIMIERUNG: MARKOWITZ MVO vs. RANDOM FOREST
# W-Seminararbeit | Bayerisches Gymnasium
# =============================================================================
# Vergleich zweier Portfoliooptimierungsstrategien:
#   1. Klassische Markowitz Mean-Variance Optimization (MVO)
#   2. Random-Forest-gestützte Portfoliooptimierung (RF-MVO)
#   3. Equal-Weight Benchmark (Blindprobe)
#
# Datenquelle : Yahoo Finance via yfinance
# Zeitraum    : 2015-01-01 bis 2024-12-31
# Backtest    : Rollierendes 3-Jahres-Trainingsfenster, vierteljährliches Rebalancing
# Outputs     : PNG-Grafiken + CSV-Dateien im Ordner ./output/
# =============================================================================

# ---------------------------------------------------------------------------
# 0. IMPORTS & KONFIGURATION
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

from scipy.optimize import minimize, OptimizeResult
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.covariance import LedoitWolf

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[WARNUNG] yfinance nicht installiert. Bitte 'pip install yfinance' ausführen.")

try:
    import ta  # technische Indikatoren
except ImportError:
    ta = None
    print("[WARNUNG] Paket 'ta' nicht installiert. Bitte 'pip install ta' ausführen.")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("portfolio_opt")

# ---------------------------------------------------------------------------
# GLOBALE PARAMETER
# ---------------------------------------------------------------------------
TICKERS = [
    # Technologie
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corp.
    "NVDA",  # NVIDIA Corp.
    # Gesundheitswesen
    "JNJ",   # Johnson & Johnson
    "UNH",   # UnitedHealth Group
    # Finanzsektor
    "JPM",   # JPMorgan Chase
    "GS",    # Goldman Sachs
    # Konsumgüter (nicht-zyklisch)
    "PG",    # Procter & Gamble
    "KO",    # Coca-Cola
    # Energie
    "XOM",   # Exxon Mobil
    # Industrie
    "CAT",   # Caterpillar
    "HON",   # Honeywell
    # Telekommunikation
    "VZ",    # Verizon
    # Immobilien (REIT)
    "PLD",   # Prologis
    # Rohstoffe / Grundstoffe
    "LIN",   # Linde plc
]

BENCHMARK_TICKER = "SPY"  # für Alpha-Berechnung

START_DATE       = "2015-01-01"
END_DATE         = "2024-12-31"
RISK_FREE_RATE   = 0.04          # Annualisierter risikoloser Zinssatz
TRAIN_YEARS      = 3             # Rollierendes Trainingsfenster (Jahre)
REBALANCE_FREQ   = "QE"          # Vierteljährliches Rebalancing (Quarter End)
N_FRONTIER       = 100           # Anzahl Punkte auf der Effizienzlinie
OUTPUT_DIR       = "./output2.0"

# RF-Parameter (Basis, werden via RandomizedSearchCV getuned)
RF_BASE_PARAMS = dict(
    random_state    = 42,
    n_jobs          = -1,
)

# Matplotlib-Stilkonfiguration (professionelles Layout)
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
    "legend.fontsize"  : 10,
})

COLORS = {
    "markowitz"    : "#1f77b4",   # Blau
    "rf"           : "#d62728",   # Rot
    "equal_weight" : "#2ca02c",   # Grün
    "frontier"     : "#7f7f7f",   # Grau
}


# ---------------------------------------------------------------------------
# 1. HILFSFUNKTIONEN
# ---------------------------------------------------------------------------

def timer(func):
    """Dekorator: Misst und protokolliert die Laufzeit einer Funktion."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.info(f"[TIMER] {func.__name__:<35} → {elapsed:.2f}s")
        return result
    return wrapper


def annualize(returns: pd.Series, freq: int = 252) -> tuple[float, float]:
    """Berechnet annualisierte Rendite und Volatilität aus täglichen Renditen."""
    ann_ret  = (1 + returns).prod() ** (freq / len(returns)) - 1
    ann_vol  = returns.std() * np.sqrt(freq)
    return ann_ret, ann_vol


def sharpe_ratio(returns: pd.Series,
                 rf: float = RISK_FREE_RATE,
                 freq: int = 252) -> float:
    """Sharpe Ratio nach Sharpe (1994)."""
    ann_ret, ann_vol = annualize(returns, freq)
    return (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0


def max_drawdown(cum_returns: pd.Series) -> float:
    """Maximaler Drawdown aus einer kumulierten Renditeserie."""
    rolling_max = cum_returns.cummax()
    drawdown    = (cum_returns - rolling_max) / rolling_max
    return drawdown.min()


def calmar_ratio(returns: pd.Series, freq: int = 252) -> float:
    """Calmar Ratio = CAGR / |Max Drawdown|."""
    cum = (1 + returns).cumprod()
    ann_ret, _ = annualize(returns, freq)
    mdd        = abs(max_drawdown(cum))
    return ann_ret / mdd if mdd > 0 else 0.0


def cagr(returns: pd.Series, freq: int = 252) -> float:
    """Compound Annual Growth Rate."""
    n_years = len(returns) / freq
    total   = (1 + returns).prod()
    return total ** (1 / n_years) - 1 if n_years > 0 else 0.0


def portfolio_performance(weights: np.ndarray,
                          mu: np.ndarray,
                          cov: np.ndarray,
                          rf: float = RISK_FREE_RATE) -> tuple[float, float, float]:
    """
    Berechnet erwartete Rendite, Volatilität und Sharpe Ratio eines Portfolios.
    Eingaben sind annualisiert.

    Returns
    -------
    (exp_return, volatility, sharpe)
    """
    ret = np.dot(weights, mu)
    vol = np.sqrt(weights @ cov @ weights)
    sr  = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sr


# ---------------------------------------------------------------------------
# 2. DATEN HERUNTERLADEN & VORBEREITEN (täglich + SPY)
# ---------------------------------------------------------------------------

@timer
def download_data(tickers: list[str],
                  benchmark: str,
                  start: str,
                  end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Lädt adjustierte Schlusskurse via yfinance herunter.
    Gibt tägliche Log-Renditen der Assets + SPY-Returns zurück.
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance ist nicht verfügbar. Bitte installieren und erneut ausführen.")

    all_tickers = tickers + [benchmark]
    log.info(f"Lade Daten für {len(all_tickers)} Assets (inkl. Benchmark {benchmark}): {start} → {end}")
    prices = yf.download(all_tickers, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]

    missing = [t for t in all_tickers if t not in prices.columns]
    if missing:
        log.warning(f"Fehlende Tickers (werden ignoriert): {missing}")
        all_tickers = [t for t in all_tickers if t in prices.columns]
        prices = prices[all_tickers]

    prices = prices.dropna(axis=0, how="any")
    log.info(f"Datenpunkte: {len(prices)} Handelstage | {prices.shape[1]} Assets")

    benchmark_prices = prices[benchmark]
    asset_prices     = prices.drop(columns=[benchmark])

    returns       = np.log(asset_prices / asset_prices.shift(1)).dropna()
    benchmark_ret = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()

    return asset_prices, returns, benchmark_ret


# ---------------------------------------------------------------------------
# 3. MARKOWITZ MVO (mit Ledoit-Wolf)
# ---------------------------------------------------------------------------

class MarkowitzOptimizer:
    """
    Klassische Mean-Variance Optimization nach Markowitz (1952).

    Minimiert die Portfoliovarianz für ein gegebenes Renditeziel oder
    maximiert die Sharpe Ratio (tangentiales Portfolio).
    """

    def __init__(self, rf: float = RISK_FREE_RATE):
        self.rf = rf

    @staticmethod
    def _neg_sharpe(weights, mu, cov, rf):
        ret = np.dot(weights, mu)
        vol = np.sqrt(weights @ cov @ weights)
        return -(ret - rf) / vol if vol > 0 else 0.0

    def max_sharpe(self,
                   mu: np.ndarray,
                   cov: np.ndarray,
                   allow_short: bool = False) -> np.ndarray:
        """
        Bestimmt das Tangential-Portfolio (maximale Sharpe Ratio).

        Parameters
        ----------
        mu          : Vektor erwarteter annualisierter Renditen
        cov         : Annualisierte Kovarianzmatrix
        allow_short : Erlaubt Leerverkäufe (hier: False für Long-Only)

        Returns
        -------
        Optimale Gewichtsvektor (np.ndarray)
        """
        n   = len(mu)
        w0  = np.ones(n) / n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds      = ((-1, 1) if allow_short else (0, 1),) * n

        result: OptimizeResult = minimize(
            fun         = self._neg_sharpe,
            x0          = w0,
            args        = (mu, cov, self.rf),
            method      = "SLSQP",
            bounds      = bounds,
            constraints = constraints,
            options     = {"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            log.warning(f"MVO Solver-Warnung: {result.message}")

        w = np.array(result.x)
        w = np.maximum(w, 0)                   # Numerische Kleinstresidualen → 0
        w /= w.sum()                           # Renormierung
        return w

    def efficient_frontier(self,
                           mu: np.ndarray,
                           cov: np.ndarray,
                           n_points: int = N_FRONTIER) -> pd.DataFrame:
        """
        Berechnet N Punkte auf der Effizienzlinie (Efficient Frontier).

        Für jedes Renditeziel γ zwischen minimalem und maximalem Asset-Return
        wird die minimale Varianz bestimmt.

        Returns
        -------
        DataFrame mit Spalten: ["ret", "vol", "sharpe"]
        """
        n      = len(mu)
        bounds = ((0.0, 1.0),) * n

        # Minimum-Varianz-Portfolio (untere Grenze der Frontier)
        res_min = minimize(
            fun         = lambda w: w @ cov @ w,
            x0          = np.ones(n) / n,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}],
        )
        ret_min = res_min.x @ mu
        ret_max = mu.max()

        target_returns = np.linspace(ret_min, ret_max, n_points)
        frontier = []

        for target in target_returns:
            res = minimize(
                fun         = lambda w: w @ cov @ w,
                x0          = np.ones(n) / n,
                method      = "SLSQP",
                bounds      = bounds,
                constraints = [
                    {"type": "eq", "fun": lambda w: w.sum() - 1},
                    {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
                ],
                options     = {"maxiter": 500, "ftol": 1e-12},
            )
            if res.success:
                vol = np.sqrt(res.x @ cov @ res.x)
                sr  = (target - self.rf) / vol if vol > 0 else 0.0
                frontier.append({"ret": target, "vol": vol, "sharpe": sr})

        return pd.DataFrame(frontier)


# ---------------------------------------------------------------------------
# 4. RANDOM FOREST PORTFOLIO OPTIMIZER (täglich, technische Indikatoren)
# ---------------------------------------------------------------------------

def build_rf_features_daily(daily_returns: pd.DataFrame,
                            benchmark_returns: pd.Series
                            ) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Erstellt tägliche Features je Asset auf Basis von Kursrenditen.

    Features:
      - ret_1d, ret_5d, ret_20d (Durchschnittsrenditen)
      - vol_20d (rollende Volatilität)
      - momentum_10d (Summe der letzten 10 Renditen)
      - RSI, MACD, Bollinger-Bänder (falls 'ta' verfügbar)
      - alpha_vs_benchmark (Asset-Return minus Benchmark-Return)
    Zielvariable:
      - nächster Tagesreturn (t+1)
    """
    features_list = []

    # Benchmark auf gleichen Index bringen
    bench_r = benchmark_returns.reindex(daily_returns.index).fillna(0)

    for col in daily_returns.columns:
        r = daily_returns[col].copy()

        df = pd.DataFrame(index=r.index)
        df["ret_1d"]   = r
        df["ret_5d"]   = r.rolling(5).mean()
        df["ret_20d"]  = r.rolling(20).mean()
        df["vol_20d"]  = r.rolling(20).std()
        df["mom_10d"]  = r.rolling(10).sum()
        df["alpha"]    = r - bench_r

        if ta is not None:
            r_pct = r * 100
            try:
                df["rsi"] = ta.momentum.RSIIndicator(r_pct, window=14).rsi()
                df["macd"] = ta.trend.MACD(r_pct).macd()
                bb = ta.volatility.BollingerBands(r_pct)
                df["bb_upper"] = bb.bollinger_hband()
                df["bb_lower"] = bb.bollinger_lband()
            except Exception:
                df["rsi"]      = 0.0
                df["macd"]     = 0.0
                df["bb_upper"] = 0.0
                df["bb_lower"] = 0.0
        else:
            df["rsi"]      = 0.0
            df["macd"]     = 0.0
            df["bb_upper"] = 0.0
            df["bb_lower"] = 0.0

        df["ticker"] = col
        df["target"] = r.shift(-1)  # nächster Tag

        features_list.append(df)

    combined = pd.concat(features_list).dropna()
    feature_cols = [
        "ret_1d", "ret_5d", "ret_20d",
        "vol_20d", "mom_10d",
        "alpha", "rsi", "macd", "bb_upper", "bb_lower"
    ]

    X = combined[feature_cols]
    y = combined["target"]
    tickers_col = combined["ticker"]

    return X, y, tickers_col, feature_cols


class RFPortfolioOptimizer:
    """
    Random-Forest-gestützte Portfoliooptimierung mit täglichen Daten.

    Methodik:
      1. Tägliche technische Features + Alpha vs. Benchmark.
      2. Rolling-Window-Training auf letztem Trainingsfenster.
      3. Hyperparameter-Tuning via RandomizedSearchCV (TimeSeriesSplit).
      4. Prognose des nächsten Tagesreturns je Asset.
      5. μ̂ (annualisiert) → Markowitz-Optimierung (max. Sharpe).
    """

    def __init__(self, rf: float = RISK_FREE_RATE):
        self.rf      = rf
        self.model   = None
        self.scaler  = StandardScaler()
        self.mvo     = MarkowitzOptimizer(rf=rf)
        self._fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Trainiert den Random Forest mit Hyperparameter-Tuning."""
        X_scaled = self.scaler.fit_transform(X_train)

        base_rf = RandomForestRegressor(**RF_BASE_PARAMS)
        param_dist = {
            "n_estimators":      [100, 200, 300],
            "max_depth":         [4, 6, 8, None],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf":  [2, 5, 10],
        }
        tscv = TimeSeriesSplit(n_splits=3)

        search = RandomizedSearchCV(
            estimator           = base_rf,
            param_distributions = param_dist,
            n_iter              = 10,
            cv                  = tscv,
            scoring             = "neg_mean_squared_error",
            random_state        = 42,
            n_jobs              = -1,
            verbose             = 0,
        )
        search.fit(X_scaled, y_train)
        self.model = search.best_estimator_
        self._fitted = True

    def predict_expected_returns(self,
                                 X_current: pd.DataFrame,
                                 tickers: list[str]) -> np.ndarray:
        """
        Prognostiziert erwartete tägliche Renditen für alle Assets
        und annualisiert sie (× 252).
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Modell muss zuerst mit fit() trainiert werden.")
        X_scaled = self.scaler.transform(X_current)
        preds_daily = self.model.predict(X_scaled)
        return preds_daily * 252  # tägliche → annualisierte Renditen

    def optimize(self,
                 mu_predicted: np.ndarray,
                 cov_ann: np.ndarray) -> np.ndarray:
        """Maximiert Sharpe Ratio mit RF-prognostizierten Renditen."""
        return self.mvo.max_sharpe(mu_predicted, cov_ann)


# ---------------------------------------------------------------------------
# 5. BACKTESTING ENGINE (täglich, Rolling-Window)
# ---------------------------------------------------------------------------

@timer
def run_backtest(daily_returns: pd.DataFrame,
                 benchmark_returns: pd.Series,
                 tickers: list[str]) -> dict:
    """
    Rollierendes Backtest-Framework.

    Ablauf je Rebalancing-Datum:
      1. Trainingsdaten der letzten TRAIN_YEARS Jahre extrahieren.
      2. Markowitz: μ und Σ (Ledoit-Wolf) aus Trainingsdaten schätzen → Gewichte.
      3. RF: tägliche Features bauen, Modell trainieren, μ̂ prognostizieren → Gewichte.
      4. Equal Weight: Gleichgewichtung aller Assets.
      5. Portfoliorenditen bis zum nächsten Rebalancing-Datum berechnen.

    Returns
    -------
    dict mit Keys:
      "returns"     : DataFrame (tägliche Portfoliorenditen je Strategie)
      "weights_mvo" : DataFrame (Gewichte über Zeit, Markowitz)
      "weights_rf"  : DataFrame (Gewichte über Zeit, RF)
    """
    log.info("Starte rollierendes Backtest-Framework …")
    t_total = time.perf_counter()

    # Rebalancing-Termine aus dem Testzeitraum ableiten
    test_start = daily_returns.index[0] + pd.DateOffset(years=TRAIN_YEARS)
    rebal_dates = pd.date_range(
        start=test_start, end=daily_returns.index[-1], freq=REBALANCE_FREQ
    )
    log.info(f"Rebalancing-Termine: {len(rebal_dates)} | "
             f"Test-Start: {test_start.date()}")

    # Ergebnis-Container
    returns_mvo = []
    returns_rf  = []
    returns_ew  = []
    weights_mvo_history = []
    weights_rf_history  = []

    mvo_optimizer = MarkowitzOptimizer()
    rf_optimizer  = RFPortfolioOptimizer()
    n             = len(tickers)

    mu_ann = None
    cov_ann = None
    mu_rf = None
    w_mvo = np.ones(n) / n
    w_rf = np.ones(n) / n

    for i, rebal_date in enumerate(rebal_dates):
        t_step = time.perf_counter()

        # Trainingsfenster: TRAIN_YEARS vor Rebalancing-Datum
        train_start = rebal_date - pd.DateOffset(years=TRAIN_YEARS)
        train_end   = rebal_date

        train_daily = daily_returns[(daily_returns.index >= train_start) &
                                    (daily_returns.index <  train_end)]

        if len(train_daily) < 252:
            log.warning(f"Zu wenige Daten für {rebal_date.date()}, überspringe.")
            continue

        # Nächster Rebalancing-Termin (für Hold-Periode)
        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) \
                    else daily_returns.index[-1]

        hold_returns = daily_returns[(daily_returns.index > rebal_date) &
                                     (daily_returns.index <= next_date)]

        if hold_returns.empty:
            continue

        # ---- Markowitz MVO: μ und Σ mit Ledoit-Wolf ----------------------
        mu_daily = train_daily.mean().values  # tägliche Mittelwerte

        try:
            lw = LedoitWolf().fit(train_daily.values)
            cov_daily = lw.covariance_
        except Exception as e:
            log.warning(f"Ledoit-Wolf fehlgeschlagen, nutze klassische Kovarianz: {e}")
            cov_daily = train_daily.cov().values

        mu_ann  = mu_daily * 252
        cov_ann = cov_daily * 252

        try:
            w_mvo = mvo_optimizer.max_sharpe(mu_ann, cov_ann)
        except Exception as e:
            log.warning(f"MVO-Fehler ({rebal_date.date()}): {e}")
            w_mvo = np.ones(n) / n

        # ---- Random Forest: tägliche Features + Rolling-Window -----------
        X_all, y_all, tickers_col, feat_cols = build_rf_features_daily(train_daily, benchmark_returns)

        last_train_day = train_daily.index[-1]

        # WICHTIG: konsistente Masken für X und y
        mask = X_all.index < last_train_day
        X_train_rf = X_all.loc[mask]
        y_train_rf = y_all.loc[mask]

        # Aktuelle Features: letzter verfügbarer Tag je Asset
        X_current_rows = []
        for ticker in tickers:
            ticker_rows = X_all[tickers_col == ticker]
            if len(ticker_rows) > 0:
                X_current_rows.append(ticker_rows.iloc[-1][feat_cols])
            else:
                X_current_rows.append(pd.Series(np.zeros(len(feat_cols)),
                                                index=feat_cols))
        X_current = pd.DataFrame(X_current_rows, columns=feat_cols)

        try:
            rf_optimizer.fit(X_train_rf[feat_cols], y_train_rf)
            mu_rf = rf_optimizer.predict_expected_returns(X_current, tickers)
            w_rf = rf_optimizer.optimize(mu_rf, cov_ann)
        except Exception as e:
            log.warning(f"RF-Fehler ({rebal_date.date()}): {e}")
            w_rf = np.ones(n) / n
            mu_rf = np.zeros(n)

        # ---- Equal Weight -------------------------------------------------
        w_ew = np.ones(n) / n

        # ---- Portfolio-Renditen für die Halteperiode berechnen ------------
        port_ret_mvo = (hold_returns * w_mvo).sum(axis=1)
        port_ret_rf  = (hold_returns * w_rf).sum(axis=1)
        port_ret_ew  = (hold_returns * w_ew).sum(axis=1)

        returns_mvo.append(port_ret_mvo)
        returns_rf.append(port_ret_rf)
        returns_ew.append(port_ret_ew)

        weights_mvo_history.append(
            pd.Series(w_mvo, index=tickers, name=rebal_date)
        )
        weights_rf_history.append(
            pd.Series(w_rf, index=tickers, name=rebal_date)
        )

        elapsed_step = time.perf_counter() - t_step
        try:
            sharpe_mvo = portfolio_performance(w_mvo, mu_ann, cov_ann)[2]
        except Exception:
            sharpe_mvo = np.nan
        try:
            sharpe_rf = portfolio_performance(w_rf, mu_rf, cov_ann)[2]
        except Exception:
            sharpe_rf = np.nan

        log.info(
            f"  Rebalancing {i+1:>3}/{len(rebal_dates)} | "
            f"{rebal_date.date()} | "
            f"MVO-Sharpe: {sharpe_mvo:.3f} | "
            f"RF-Sharpe: {sharpe_rf:.3f} | "
            f"Dauer: {elapsed_step:.2f}s"
        )

    # Ergebnisse zusammenführen
    all_returns = pd.DataFrame({
        "Markowitz MVO" : pd.concat(returns_mvo),
        "Random Forest" : pd.concat(returns_rf),
        "Equal Weight"  : pd.concat(returns_ew),
    }).sort_index()

    weights_mvo_df = pd.DataFrame(weights_mvo_history).T
    weights_rf_df  = pd.DataFrame(weights_rf_history).T

    log.info(f"Backtest abgeschlossen in {time.perf_counter() - t_total:.1f}s | "
             f"{len(all_returns)} Handelstage ausgewertet")

    return {
        "returns"     : all_returns,
        "weights_mvo" : weights_mvo_df,
        "weights_rf"  : weights_rf_df,
        "mu_ann"      : mu_ann,       # Aus letztem Trainingsfenster (für Frontier)
        "cov_ann"     : cov_ann,
        "mu_rf"       : mu_rf,        # Aus letzter RF-Prognose
        "w_mvo_last"  : w_mvo,
        "w_rf_last"   : w_rf,
    }


# ---------------------------------------------------------------------------
# 6. PERFORMANCE-METRIKEN
# ---------------------------------------------------------------------------

def compute_metrics(returns_df: pd.DataFrame,
                    rf: float = RISK_FREE_RATE) -> pd.DataFrame:
    """
    Berechnet umfassende Performance-Kennzahlen für alle Portfolios.

    Kennzahlen:
      - CAGR (Compound Annual Growth Rate)
      - Gesamtrendite
      - Annualisierte Volatilität
      - Sharpe Ratio
      - Maximaler Drawdown
      - Calmar Ratio
      - Value at Risk (95 %, historisch)
      - Positiver Handelstage-Anteil (Hit Rate)
    """
    metrics = {}
    for col in returns_df.columns:
        r   = returns_df[col].dropna()
        cum = (1 + r).cumprod()

        metrics[col] = {
            "CAGR (%)"                  : round(cagr(r) * 100, 2),
            "Gesamtrendite (%)"         : round((cum.iloc[-1] - 1) * 100, 2),
            "Annualisierte Vola. (%)"   : round(r.std() * np.sqrt(252) * 100, 2),
            "Sharpe Ratio"              : round(sharpe_ratio(r, rf), 4),
            "Max. Drawdown (%)"         : round(max_drawdown(cum) * 100, 2),
            "Calmar Ratio"              : round(calmar_ratio(r), 4),
            "VaR 95 % (tägl., %)"       : round(np.percentile(r, 5) * 100, 2),
            "Hit Rate (%)"              : round((r > 0).mean() * 100, 2),
        }

    return pd.DataFrame(metrics).T


# ---------------------------------------------------------------------------
# 7. VISUALISIERUNGEN
# ---------------------------------------------------------------------------

def plot_cumulative_returns(returns_df: pd.DataFrame,
                             output_path: str) -> None:
    """
    Abbildung 1: Kumulierte Portfoliorenditen über den Testzeitraum.

    Zeigt den Wachstumsverlauf von 1 € Anfangsinvestition je Strategie.
    """
    log.info("Erstelle Grafik: Kumulierte Renditen …")
    fig, ax = plt.subplots(figsize=(12, 6))

    for col, color in zip(returns_df.columns, COLORS.values()):
        cum = (1 + returns_df[col]).cumprod()
        ax.plot(cum.index, cum.values,
                label=col, color=color, linewidth=2)

        # Drawdown-Schattierung (je Strategie, halbtransparent)
        rolling_max = cum.cummax()
        ax.fill_between(cum.index, cum, rolling_max,
                         alpha=0.06, color=color)

    ax.set_title("Kumulierte Portfoliorenditen im Vergleich\n"
                 "(Testperiode: rollierendes Backtest, vierteljährliches Rebalancing)",
                 pad=14)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Kumulierter Wert (Startkapital = 1 €)")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.2f} €"))
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(returns_df.index[0], returns_df.index[-1])

    # Nulllinie
    ax.axhline(y=1, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → Gespeichert: {output_path}")


def plot_weight_heatmap(weights_df: pd.DataFrame,
                         title: str,
                         output_path: str) -> None:
    """
    Abbildung 2/3: Heatmap der Portfolio-Gewichtungen über die Zeit.

    Jede Zeile entspricht einem Asset, jede Spalte einem Rebalancing-Termin.
    Die Farbintensität zeigt das zugewiesene Gewicht (0 = 0 %, 1 = 100 %).
    """
    log.info(f"Erstelle Heatmap: {title} …")
    fig, ax = plt.subplots(figsize=(14, 7))

    # Spalten auf Prozent umrechnen
    data = weights_df.T * 100

    sns.heatmap(
        data.T,
        ax          = ax,
        cmap        = "YlOrRd",
        linewidths  = 0.4,
        linecolor   = "white",
        annot       = True,
        fmt         = ".1f",
        cbar_kws    = {"label": "Gewicht (%)", "shrink": 0.8},
        vmin        = 0,
        vmax        = 100,
    )

    # x-Achse: nur jedes zweite Datum anzeigen (Lesbarkeit)
    col_labels = [c.strftime("%b %Y") if hasattr(c, "strftime") else str(c)
                  for c in data.index]
    visible = [lbl if j % 2 == 0 else "" for j, lbl in enumerate(col_labels)]
    ax.set_xticklabels(visible, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    ax.set_title(f"Portfolio-Gewichtungen: {title}\n"
                 "(Werte in Prozent je Rebalancing-Termin)",
                 pad=14)
    ax.set_xlabel("Rebalancing-Datum")
    ax.set_ylabel("Asset")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → Gespeichert: {output_path}")


def plot_efficient_frontier(mu_ann: np.ndarray,
                             cov_ann: np.ndarray,
                             tickers: list[str],
                             w_mvo: np.ndarray,
                             w_rf: np.ndarray,
                             w_ew: np.ndarray,
                             mu_rf: np.ndarray,
                             rf: float,
                             output_path: str) -> None:
    """
    Abbildung: Vollständige Effizienzlinie (Efficient Frontier).

    Zeigt:
      - Effizienzkurve (Minimum-Varianz-Linie für Long-Only Portfolios)
      - Minimum-Varianz-Portfolio (MVP)
      - Tangential-Portfolio (Markowitz MVO, max. Sharpe)
      - RF-optimiertes Portfolio
      - Equal-Weight Portfolio
      - Capital Market Line (CML)
      - Einzelne Assets als Referenzpunkte
    """
    log.info("Erstelle Effizienzlinie …")
    mvo_opt  = MarkowitzOptimizer(rf=rf)
    frontier = mvo_opt.efficient_frontier(mu_ann, cov_ann, n_points=N_FRONTIER)

    fig, ax = plt.subplots(figsize=(11, 7))

    # Effizienzkurve
    ax.plot(frontier["vol"], frontier["ret"],
            color=COLORS["frontier"], linewidth=2.0,
            label="Effizienzlinie (Long-Only)")

    # Minimum-Varianz-Portfolio
    idx_mvp = frontier["vol"].idxmin()
    mvp_vol = frontier.loc[idx_mvp, "vol"]
    mvp_ret = frontier.loc[idx_mvp, "ret"]
    ax.scatter(mvp_vol, mvp_ret, color="black", s=60, zorder=5,
               label="Minimum-Varianz-Portfolio")

    # Tangential-Portfolio (Markowitz)
    ret_mvo, vol_mvo, _ = portfolio_performance(w_mvo, mu_ann, cov_ann, rf)
    ax.scatter(vol_mvo, ret_mvo, color=COLORS["markowitz"], s=70, zorder=6,
               label="Markowitz MVO (max. Sharpe)")

    # RF-Portfolio
    if mu_rf is not None:
        ret_rf, vol_rf, _ = portfolio_performance(w_rf, mu_rf, cov_ann, rf)
        ax.scatter(vol_rf, ret_rf, color=COLORS["rf"], s=70, zorder=6,
                   label="Random Forest Portfolio")

    # Equal-Weight Portfolio
    ret_ew, vol_ew, _ = portfolio_performance(w_ew, mu_ann, cov_ann, rf)
    ax.scatter(vol_ew, ret_ew, color=COLORS["equal_weight"], s=70, zorder=6,
               label="Equal Weight")

    # Einzelne Assets
    asset_vols = np.sqrt(np.diag(cov_ann))
    asset_rets = mu_ann
    ax.scatter(asset_vols, asset_rets, color="gray", s=40, alpha=0.7,
               label="Einzelne Assets")
    for v, r, name in zip(asset_vols, asset_rets, tickers):
        ax.annotate(name, (v, r), textcoords="offset points",
                    xytext=(4, 4), fontsize=8, alpha=0.8)

    # Capital Market Line (CML) durch risikolosen Zins und Tangential-Portfolio
    x_cml = np.linspace(0, frontier["vol"].max() * 1.1, 50)
    slope = (ret_mvo - rf) / vol_mvo if vol_mvo > 0 else 0.0
    y_cml = rf + slope * x_cml
    ax.plot(x_cml, y_cml, color="black", linestyle="--",
            linewidth=1.2, label="Capital Market Line (CML)")

    ax.set_title("Effizienzlinie und Portfolios im Risiko-Rendite-Raum", pad=14)
    ax.set_xlabel("Volatilität (σ, annualisiert)")
    ax.set_ylabel("Erwartete Rendite (μ, annualisiert)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → Gespeichert: {output_path}")


# ---------------------------------------------------------------------------
# 8. MAIN-FUNKTION
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("Starte vollständigen Workflow …")

    # Daten laden
    asset_prices, daily_returns, benchmark_returns = download_data(
        TICKERS, BENCHMARK_TICKER, START_DATE, END_DATE
    )

    # Backtest
    results = run_backtest(daily_returns, benchmark_returns, TICKERS)

    returns_df   = results["returns"]
    weights_mvo  = results["weights_mvo"]
    weights_rf   = results["weights_rf"]
    mu_ann       = results["mu_ann"]
    cov_ann      = results["cov_ann"]
    mu_rf        = results["mu_rf"]
    w_mvo_last   = results["w_mvo_last"]
    w_rf_last    = results["w_rf_last"]

    # Equal-Weight-Gewichte (für Frontier)
    w_ew = np.ones(len(TICKERS)) / len(TICKERS)

    # Performance-Metriken
    metrics_df = compute_metrics(returns_df, rf=RISK_FREE_RATE)
    metrics_path = os.path.join(OUTPUT_DIR, "performance_metrics.csv")
    metrics_df.to_csv(metrics_path, sep=";")
    log.info(f"Performance-Metriken gespeichert: {metrics_path}")

    # Kumulierte Renditen
    cum_plot_path = os.path.join(OUTPUT_DIR, "cumulative_returns.png")
    plot_cumulative_returns(returns_df, cum_plot_path)

    # Gewichtungs-Heatmaps
    mvo_heatmap_path = os.path.join(OUTPUT_DIR, "weights_markowitz_heatmap.png")
    rf_heatmap_path  = os.path.join(OUTPUT_DIR, "weights_rf_heatmap.png")
    if not weights_mvo.empty:
        plot_weight_heatmap(weights_mvo, "Markowitz MVO", mvo_heatmap_path)
    if not weights_rf.empty:
        plot_weight_heatmap(weights_rf, "Random Forest", rf_heatmap_path)

    # Effizienzlinie
    if (mu_ann is not None) and (cov_ann is not None):
        frontier_path = os.path.join(OUTPUT_DIR, "efficient_frontier.png")
        plot_efficient_frontier(
            mu_ann=mu_ann,
            cov_ann=cov_ann,
            tickers=TICKERS,
            w_mvo=w_mvo_last,
            w_rf=w_rf_last,
            w_ew=w_ew,
            mu_rf=mu_rf if mu_rf is not None else np.zeros_like(mu_ann),
            rf=RISK_FREE_RATE,
            output_path=frontier_path,
        )

    # Returns speichern
    returns_path = os.path.join(OUTPUT_DIR, "portfolio_returns.csv")
    returns_df.to_csv(returns_path, sep=";")
    log.info(f"Portfoliorenditen gespeichert: {returns_path}")

    log.info("Workflow abgeschlossen.")


# ---------------------------------------------------------------------------
# 9. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
