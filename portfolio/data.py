"""Marktdaten laden (Yahoo Finance) und einfache Renditen berechnen."""

import numpy as np
import pandas as pd
from .config import *

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
