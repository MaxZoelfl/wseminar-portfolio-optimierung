"""Technische Indikatoren, monatliche Aggregation und Feature-Spalten."""

import numpy as np
import pandas as pd
from .config import *
from .metrics import timer

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


FEATURE_COLS_BASE = [
    "rsi", "macd", "macd_sig", "macd_hist",
    "bb_pct_b", "bb_width",
    "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_21d", "vol_63d",
    "alpha_spy", "beta_spy",
    "monthly_ret",
]


FEATURE_COLS = FEATURE_COLS_BASE + [f"{c}_rank" for c in RANK_COLS]


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
