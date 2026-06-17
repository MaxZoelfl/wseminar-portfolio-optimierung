"""
=============================================================================
PORTFOLIO-OPTIMIERUNG: MARKOWITZ MVO vs. RANDOM FOREST
W-Seminararbeit | Bayerisches Gymnasium
=============================================================================
Vergleich zweier Portfoliooptimierungsstrategien:
  1. Klassische Markowitz Mean-Variance Optimization (MVO)
  2. Random-Forest-gestützte Portfoliooptimierung (RF-MVO)
  3. Equal-Weight Benchmark (Blindprobe)

Datenquelle : Yahoo Finance via yfinance
Zeitraum    : 2015-01-01 bis 2024-12-31
Backtest    : Rollierendes 3-Jahres-Trainingsfenster, vierteljährliches Rebalancing
Outputs     : PNG-Grafiken + CSV-Dateien im Ordner ./output/
=============================================================================
Quellen:
  - Markowitz, H. (1952): Portfolio Selection. Journal of Finance, 7(1), 77–91.
  - Breiman, L. (2001): Random Forests. Machine Learning, 45, 5–32.
  - Sharpe, W. F. (1966): Mutual Fund Performance. Journal of Business, 39(1).
  - DeMiguel et al. (2009): Optimal versus Naive Diversification. RFS, 22(5).
=============================================================================
"""

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

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[WARNUNG] yfinance nicht installiert. Bitte 'pip install yfinance' ausführen.")

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

START_DATE       = "2015-01-01"
END_DATE         = "2024-12-31"
RISK_FREE_RATE   = 0.04          # Annualisierter risikoloser Zinssatz (US-10J-Treasury, ca. 2024)
TRAIN_YEARS      = 3             # Rollierendes Trainingsfenster (Jahre)
REBALANCE_FREQ   = "QE"         # Vierteljährliches Rebalancing (Quarter End)
N_FRONTIER       = 100           # Anzahl Punkte auf der Effizienzlinie
OUTPUT_DIR       = "./output1.0"

# RF-Parameter
RF_PARAMS = dict(
    n_estimators    = 200,
    max_depth       = 5,
    min_samples_leaf= 5,
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
    Eingaben sind annualisiert (z. B. aus monatlichen Daten × 12).

    Returns
    -------
    (exp_return, volatility, sharpe)
    """
    ret = np.dot(weights, mu)
    vol = np.sqrt(weights @ cov @ weights)
    sr  = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sr


# ---------------------------------------------------------------------------
# 2. DATEN HERUNTERLADEN & VORBEREITEN
# ---------------------------------------------------------------------------

@timer
def download_data(tickers: list[str],
                  start: str,
                  end: str) -> pd.DataFrame:
    """
    Lädt adjustierte Schlusskurse via yfinance herunter.
    Gibt tägliche Log-Renditen zurück.
    """
    log.info(f"Lade Daten für {len(tickers)} Assets: {start} → {end}")
    prices = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]

    # Fehlende Tickers melden
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        log.warning(f"Fehlende Tickers (werden ignoriert): {missing}")
        tickers = [t for t in tickers if t in prices.columns]

    prices = prices[tickers].dropna(axis=0, how="any")
    log.info(f"Datenpunkte: {len(prices)} Handelstage | {prices.shape[1]} Assets")

    returns = np.log(prices / prices.shift(1)).dropna()
    return prices, returns


def build_monthly_data(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert tägliche zu monatlichen Renditen (für RF-Features)."""
    return daily_returns.resample("ME").sum()   # Summe der Log-Renditen ≈ Monatsrendite


# ---------------------------------------------------------------------------
# 3. MARKOWITZ MVO
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
# 4. RANDOM FOREST PORTFOLIO OPTIMIZER
# ---------------------------------------------------------------------------

def build_rf_features(monthly_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Erstellt ein Feature-Set für den Random Forest.

    Features je Asset (t):
      - Rendite der letzten 1, 3, 6, 12 Monate (Momentum)
      - Rollende 3-Monats-Volatilität (Risikofaktor)
      - Rendite im Vormonat (Mean Reversion Signal)

    Zielvariable:
      - Rendite im nächsten Monat (t+1)
    """
    features_list = []
    targets_list  = []

    for col in monthly_returns.columns:
        r = monthly_returns[col].copy()

        df = pd.DataFrame(index=r.index)
        df["ret_1m"]   = r.shift(1)
        df["ret_3m"]   = r.rolling(3).sum().shift(1)
        df["ret_6m"]   = r.rolling(6).sum().shift(1)
        df["ret_12m"]  = r.rolling(12).sum().shift(1)
        df["vol_3m"]   = r.rolling(3).std().shift(1)
        df["ticker"]   = col
        df["target"]   = r   # Zielvariable: nächster Monatsreturn

        features_list.append(df)

    combined   = pd.concat(features_list).dropna()
    feature_cols = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_3m"]

    X = combined[feature_cols]
    y = combined["target"]
    tickers_col = combined["ticker"]

    return X, y, tickers_col, feature_cols


class RFPortfolioOptimizer:
    """
    Random-Forest-gestützte Portfoliooptimierung.

    Methodik:
      1. Trainiere einen RF auf historischen Monatsrenditen + Momentum-Features.
      2. Nutze die vorhergesagten Renditen als Erwartungsvektor μ̂ in der MVO.
      3. Optimiere das Portfolio via Sharpe-Maximierung (wie Markowitz, aber mit
         ML-Prognosen statt historischen Mittelwerten).

    Referenz: Fischer & Krauss (2018) – Deep learning with long short-term memory
    networks for financial market predictions. European Journal of OR.
    """

    def __init__(self, rf: float = RISK_FREE_RATE):
        self.rf      = rf
        self.model   = RandomForestRegressor(**RF_PARAMS)
        self.scaler  = StandardScaler()
        self.mvo     = MarkowitzOptimizer(rf=rf)
        self._fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Trainiert den Random Forest auf dem Trainingsdatensatz."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self._fitted = True

    def predict_expected_returns(self,
                                  X_current: pd.DataFrame,
                                  tickers: list[str]) -> np.ndarray:
        """
        Prognostiziert die erwarteten Renditen für alle Assets.

        Parameters
        ----------
        X_current : Feature-Matrix für den aktuellen Zeitpunkt (1 Zeile je Asset)
        tickers   : Liste der Asset-Ticker (gleiche Reihenfolge wie X_current)

        Returns
        -------
        Vektor der vorhergesagten monatlichen Renditen (annualisiert × 12)
        """
        if not self._fitted:
            raise RuntimeError("Modell muss zuerst mit fit() trainiert werden.")
        X_scaled  = self.scaler.transform(X_current)
        preds     = self.model.predict(X_scaled)
        return preds * 12   # Monatliche → annualisierte Renditen

    def optimize(self,
                 mu_predicted: np.ndarray,
                 cov: np.ndarray) -> np.ndarray:
        """Wrapper: Maximiert Sharpe Ratio mit RF-prognostizierten Renditen."""
        return self.mvo.max_sharpe(mu_predicted, cov)


# ---------------------------------------------------------------------------
# 5. BACKTESTING ENGINE
# ---------------------------------------------------------------------------

@timer
def run_backtest(daily_returns: pd.DataFrame,
                 monthly_returns: pd.DataFrame,
                 tickers: list[str]) -> dict:
    """
    Rollierendes Backtest-Framework.

    Ablauf je Rebalancing-Datum:
      1. Trainingsdaten der letzten TRAIN_YEARS Jahre extrahieren.
      2. Markowitz: μ und Σ aus Trainingsdaten schätzen → Gewichte optimieren.
      3. RF: Features aufbauen, Modell trainieren, μ̂ prognostizieren → Gewichte.
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

    for i, rebal_date in enumerate(rebal_dates):
        t_step = time.perf_counter()

        # Trainingsfenster: TRAIN_YEARS vor Rebalancing-Datum
        train_start = rebal_date - pd.DateOffset(years=TRAIN_YEARS)
        train_end   = rebal_date

        train_daily   = daily_returns[(daily_returns.index >= train_start) &
                                      (daily_returns.index <  train_end)]
        train_monthly = monthly_returns[(monthly_returns.index >= train_start) &
                                        (monthly_returns.index <  train_end)]

        if len(train_daily) < 252 or len(train_monthly) < 24:
            log.warning(f"Zu wenige Daten für {rebal_date.date()}, überspringe.")
            continue

        # Nächster Rebalancing-Termin (für Hold-Periode)
        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) \
                    else daily_returns.index[-1]

        hold_returns = daily_returns[(daily_returns.index > rebal_date) &
                                     (daily_returns.index <= next_date)]

        if hold_returns.empty:
            continue

        # ---- Markowitz MVO ------------------------------------------------
        mu_daily  = train_daily.mean().values          # Tägliche Mittelwerte
        cov_daily = train_daily.cov().values           # Tägliche Kovarianzmatrix
        mu_ann    = mu_daily  * 252                    # Annualisiert
        cov_ann   = cov_daily * 252

        try:
            w_mvo = mvo_optimizer.max_sharpe(mu_ann, cov_ann)
        except Exception as e:
            log.warning(f"MVO-Fehler ({rebal_date.date()}): {e}")
            w_mvo = np.ones(n) / n

        # ---- Random Forest ------------------------------------------------
        X_all, y_all, tickers_col, feat_cols = build_rf_features(train_monthly)

        # Trainings-Features (alle bis auf letzten Monat)
        last_month  = train_monthly.index[-1]
        X_train_rf  = X_all[X_all.index < last_month]
        y_train_rf  = y_all[y_all.index < last_month]

        # Aktuelle Features (letzter verfügbarer Monat je Asset)
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

            # Kovarianzmatrix aus täglichen Daten (gleich wie bei MVO)
            w_rf = rf_optimizer.optimize(mu_rf, cov_ann)
        except Exception as e:
            log.warning(f"RF-Fehler ({rebal_date.date()}): {e}")
            w_rf = np.ones(n) / n

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
        log.info(f"  Rebalancing {i+1:>3}/{len(rebal_dates)} | "
                 f"{rebal_date.date()} | "
                 f"MVO-Sharpe: {portfolio_performance(w_mvo, mu_ann, cov_ann)[2]:.3f} | "
                 f"RF-Sharpe: {portfolio_performance(w_rf, mu_ann if len(mu_rf)==0 else mu_rf, cov_ann)[2]:.3f} | "
                 f"Dauer: {elapsed_step:.2f}s")

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
      - Sharpe Ratio (Sharpe, 1966/1994)
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
            "VaR 95 % (tägl., %)"      : round(np.percentile(r, 5) * 100, 2),
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
    Abbildung 4: Vollständige Effizienzlinie (Efficient Frontier).

    Zeigt:
      - Effizienzkurve (Minimum-Varianz-Linie für Long-Only Portfolios)
      - Minimum-Varianz-Portfolio (MVP)
      - Tangential-Portfolio (Markowitz MVO, max. Sharpe)
      - RF-optimiertes Portfolio
      - Equal-Weight Portfolio
      - Capital Market Line (CML) vom risikolosen Zinssatz durch das Tangential-Portfolio
      - Einzelne Assets als Referenzpunkte
    """
    log.info("Erstelle Effizienzlinie …")
    mvo_opt  = MarkowitzOptimizer(rf=rf)
    frontier = mvo_opt.efficient_frontier(mu_ann, cov_ann, n_points=N_FRONTIER)

    fig, ax = plt.subplots(figsize=(11, 7))

    # --- Effizienzkurve ---------------------------------------------------
    if not frontier.empty:
        ax.plot(
            frontier["vol"] * 100,
            frontier["ret"] * 100,
            color     = COLORS["frontier"],
            linewidth = 2.5,
            label     = "Efficient Frontier",
            zorder    = 2,
        )

    # --- Capital Market Line (CML) ----------------------------------------
    ret_tan, vol_tan, _ = portfolio_performance(w_mvo, mu_ann, cov_ann, rf)
    cml_vols = np.linspace(0, vol_tan * 1.5, 100)
    cml_rets = rf + (ret_tan - rf) / vol_tan * cml_vols
    ax.plot(
        cml_vols * 100, cml_rets * 100,
        color     = "#9467bd",
        linestyle = "--",
        linewidth = 1.5,
        label     = f"Capital Market Line (rf = {rf*100:.1f} %)",
        zorder    = 1,
        alpha     = 0.8,
    )
    ax.scatter([0], [rf * 100], marker="*", s=160, color="#9467bd",
               zorder=5, label=f"Risikoloser Zinssatz ({rf*100:.1f} %)")

    # --- Einzelne Assets --------------------------------------------------
    for i, t in enumerate(tickers):
        a_vol = np.sqrt(cov_ann[i, i]) * 100
        a_ret = mu_ann[i] * 100
        ax.scatter(a_vol, a_ret, color="lightsteelblue",
                   s=50, zorder=3, alpha=0.7)
        ax.annotate(t, (a_vol, a_ret),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=8, color="dimgrey")

    # --- Portfoliopunkte --------------------------------------------------
    def _add_portfolio(w, mu, label, color, marker="o", size=200):
        ret, vol, sr = portfolio_performance(w, mu, cov_ann, rf)
        ax.scatter(vol * 100, ret * 100,
                   color=color, marker=marker, s=size,
                   zorder=6, edgecolors="black", linewidths=0.8,
                   label=f"{label}\n(Rendite: {ret*100:.1f}%, Vola: {vol*100:.1f}%, Sharpe: {sr:.2f})")

    _add_portfolio(w_mvo, mu_ann, "Markowitz MVO (Tangential)", COLORS["markowitz"], "D", 220)
    _add_portfolio(w_rf,  mu_rf,  "Random Forest MVO",          COLORS["rf"],        "^", 220)
    _add_portfolio(w_ew,  mu_ann, "Equal Weight (Benchmark)",   COLORS["equal_weight"], "s", 180)

    # Minimum-Varianz-Portfolio markieren
    if not frontier.empty:
        mvp_row = frontier.loc[frontier["vol"].idxmin()]
        ax.scatter(mvp_row["vol"] * 100, mvp_row["ret"] * 100,
                   color="gold", marker="P", s=180, zorder=6,
                   edgecolors="black", linewidths=0.8,
                   label="Minimum-Varianz-Portfolio (MVP)")

    ax.set_title("Effizienzkurve (Efficient Frontier)\n"
                 "Mean-Variance Framework nach Markowitz (1952)",
                 pad=14)
    ax.set_xlabel("Annualisierte Volatilität (%)")
    ax.set_ylabel("Annualisierte Erwartungsrendite (%)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}%"))
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.1f}%"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → Gespeichert: {output_path}")


def plot_performance_metrics(metrics_df: pd.DataFrame,
                              output_path: str) -> None:
    """
    Abbildung 5: Horizontale Balkendiagramme der Performance-Kennzahlen.

    Ermöglicht einen direkten, visuellen Vergleich aller Strategien
    für die wichtigsten Kennzahlen.
    """
    log.info("Erstelle Performance-Vergleichs-Grafik …")

    display_metrics = [
        ("CAGR (%)",               "CAGR (%)"),
        ("Sharpe Ratio",           "Sharpe Ratio"),
        ("Max. Drawdown (%)",      "Max. Drawdown (%)"),
        ("Annualisierte Vola. (%)", "Vola. (% p.a.)"),
        ("Calmar Ratio",           "Calmar Ratio"),
        ("Hit Rate (%)",           "Hit Rate (%)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    portfolio_colors = [
        COLORS["markowitz"],
        COLORS["rf"],
        COLORS["equal_weight"],
    ]

    for ax, (metric_key, metric_label) in zip(axes, display_metrics):
        values = metrics_df[metric_key]
        bars   = ax.barh(
            values.index,
            values.values,
            color  = portfolio_colors[:len(values)],
            height = 0.5,
            edgecolor="white",
        )

        # Werte-Label auf den Balken
        for bar, val in zip(bars, values.values):
            sign = "+" if val > 0 else ""
            ax.text(
                val + (values.abs().max() * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.2f}",
                va="center", ha="left", fontsize=9, fontweight="bold",
            )

        ax.set_title(metric_label, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_xlim(
            values.min() - values.abs().max() * 0.25,
            values.max() + values.abs().max() * 0.35,
        )
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Performance-Kennzahlen im Vergleich\n"
        "(Testperiode: rollierendes Backtest)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → Gespeichert: {output_path}")


def plot_rolling_sharpe(returns_df: pd.DataFrame,
                        output_path: str,
                        window: int = 252) -> None:
    """
    Abbildung 6 (Bonus): Rollierender 1-Jahres-Sharpe Ratio.

    Zeigt die zeitliche Stabilität der Strategien.
    """
    log.info("Erstelle Grafik: Rollierender Sharpe Ratio …")
    fig, ax = plt.subplots(figsize=(12, 5))

    rf_daily = RISK_FREE_RATE / 252

    for col, color in zip(returns_df.columns, COLORS.values()):
        r = returns_df[col]
        rolling_sr = (
            (r.rolling(window).mean() - rf_daily)
            / r.rolling(window).std()
            * np.sqrt(252)
        )
        ax.plot(rolling_sr.index, rolling_sr.values,
                label=col, color=color, linewidth=1.8, alpha=0.9)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(f"Rollierender Sharpe Ratio ({window}-Tage-Fenster)",
                 pad=14)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Sharpe Ratio (annualisiert)")
    ax.legend(loc="upper left")
    ax.set_xlim(returns_df.index[0], returns_df.index[-1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  → Gespeichert: {output_path}")


# ---------------------------------------------------------------------------
# 8. CSV-EXPORT
# ---------------------------------------------------------------------------

def save_csv_results(returns_df: pd.DataFrame,
                     metrics_df: pd.DataFrame,
                     weights_mvo: pd.DataFrame,
                     weights_rf: pd.DataFrame,
                     output_dir: str) -> None:
    """Speichert alle Ergebnisse als CSV-Dateien."""
    log.info("Speichere CSV-Ergebnisse …")

    cum_returns = (1 + returns_df).cumprod()
    cum_returns.to_csv(os.path.join(output_dir, "cumulative_returns.csv"),
                       float_format="%.6f")

    returns_df.to_csv(os.path.join(output_dir, "daily_returns.csv"),
                      float_format="%.6f")

    metrics_df.to_csv(os.path.join(output_dir, "performance_metrics.csv"),
                      float_format="%.4f")

    weights_mvo.T.to_csv(os.path.join(output_dir, "weights_markowitz.csv"),
                          float_format="%.4f")

    weights_rf.T.to_csv(os.path.join(output_dir, "weights_rf.csv"),
                         float_format="%.4f")

    log.info("  → CSV-Dateien gespeichert.")


# ---------------------------------------------------------------------------
# 9. HAUPTPROGRAMM
# ---------------------------------------------------------------------------

def main():
    t_main = time.perf_counter()
    log.info("=" * 65)
    log.info("  PORTFOLIO-OPTIMIERUNG: MARKOWITZ vs. RANDOM FOREST")
    log.info(f"  Gestartet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 65)

    # Output-Ordner erstellen
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info(f"Output-Ordner: {os.path.abspath(OUTPUT_DIR)}")

    # ------------------------------------------------------------------
    # 1. Daten laden
    # ------------------------------------------------------------------
    if not YFINANCE_AVAILABLE:
        log.error("yfinance nicht verfügbar. Bitte installieren und erneut starten.")
        return

    prices, daily_returns = download_data(TICKERS, START_DATE, END_DATE)
    tickers = list(daily_returns.columns)
    monthly_returns = build_monthly_data(daily_returns)
    log.info(f"Assets im Portfolio: {', '.join(tickers)}")

    # ------------------------------------------------------------------
    # 2. Backtest durchführen
    # ------------------------------------------------------------------
    results = run_backtest(daily_returns, monthly_returns, tickers)

    returns_df  = results["returns"]
    weights_mvo = results["weights_mvo"]
    weights_rf  = results["weights_rf"]
    mu_ann      = results["mu_ann"]
    cov_ann     = results["cov_ann"]
    mu_rf       = results["mu_rf"]
    w_mvo_last  = results["w_mvo_last"]
    w_rf_last   = results["w_rf_last"]
    w_ew        = np.ones(len(tickers)) / len(tickers)

    # ------------------------------------------------------------------
    # 3. Performance-Kennzahlen
    # ------------------------------------------------------------------
    metrics_df = compute_metrics(returns_df)
    log.info("\n" + "=" * 55)
    log.info("  PERFORMANCE-KENNZAHLEN (ZUSAMMENFASSUNG)")
    log.info("=" * 55)
    print(metrics_df.to_string())
    log.info("=" * 55 + "\n")

    # ------------------------------------------------------------------
    # 4. Visualisierungen speichern
    # ------------------------------------------------------------------
    plot_cumulative_returns(
        returns_df,
        os.path.join(OUTPUT_DIR, "01_kumulierte_renditen.png")
    )

    if not weights_mvo.empty:
        plot_weight_heatmap(
            weights_mvo,
            title       = "Markowitz MVO",
            output_path = os.path.join(OUTPUT_DIR, "02_gewichte_markowitz.png"),
        )

    if not weights_rf.empty:
        plot_weight_heatmap(
            weights_rf,
            title       = "Random Forest MVO",
            output_path = os.path.join(OUTPUT_DIR, "03_gewichte_random_forest.png"),
        )

    plot_efficient_frontier(
        mu_ann      = mu_ann,
        cov_ann     = cov_ann,
        tickers     = tickers,
        w_mvo       = w_mvo_last,
        w_rf        = w_rf_last,
        w_ew        = w_ew,
        mu_rf       = mu_rf,
        rf          = RISK_FREE_RATE,
        output_path = os.path.join(OUTPUT_DIR, "04_efficient_frontier.png"),
    )

    plot_performance_metrics(
        metrics_df,
        os.path.join(OUTPUT_DIR, "05_performance_kennzahlen.png")
    )

    plot_rolling_sharpe(
        returns_df,
        os.path.join(OUTPUT_DIR, "06_rollierender_sharpe.png")
    )

    # ------------------------------------------------------------------
    # 5. CSV-Export
    # ------------------------------------------------------------------
    save_csv_results(
        returns_df, metrics_df, weights_mvo, weights_rf, OUTPUT_DIR
    )

    # ------------------------------------------------------------------
    # 6. Abschluss-Log
    # ------------------------------------------------------------------
    total_time = time.perf_counter() - t_main
    log.info("=" * 65)
    log.info(f"  FERTIG | Gesamtlaufzeit: {total_time:.1f}s")
    log.info(f"  Alle Dateien in: {os.path.abspath(OUTPUT_DIR)}/")
    log.info("=" * 65)

    # Kompakte Ergebnisübersicht ausgeben
    print("\n" + "─" * 55)
    print("  ERGEBNISÜBERSICHT")
    print("─" * 55)
    for strategy in metrics_df.index:
        m = metrics_df.loc[strategy]
        print(f"\n  [{strategy}]")
        print(f"    CAGR:           {m['CAGR (%)']:>7.2f} %")
        print(f"    Gesamtrendite:  {m['Gesamtrendite (%)']:>7.2f} %")
        print(f"    Sharpe Ratio:   {m['Sharpe Ratio']:>7.4f}")
        print(f"    Max. Drawdown:  {m['Max. Drawdown (%)']:>7.2f} %")
        print(f"    Calmar Ratio:   {m['Calmar Ratio']:>7.4f}")
    print("─" * 55)


if __name__ == "__main__":
    main()