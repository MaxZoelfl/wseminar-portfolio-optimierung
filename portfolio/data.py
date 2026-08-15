"""
Marktdaten laden (Yahoo Finance) und einfache Renditen berechnen.

FÜR EINSTEIGER — WAS MACHT DIESE DATEI?
Bevor irgendetwas verglichen werden kann, brauchen wir die Rohdaten: die
täglichen Schlusskurse der 15 Aktien und des Vergleichsindex (SPY). Diese
Datei lädt die Kurse kostenlos über die Bibliothek "yfinance" von Yahoo
Finance herunter und rechnet daraus die täglichen RENDITEN aus.

Was ist eine "Rendite"? Die prozentuale Wertänderung von einem Tag zum
nächsten. Beispiel: Kurs gestern 100 €, heute 102 € → Rendite = 102/100 − 1
= 0,02 = +2 %. Mit Renditen (statt mit Kursen) zu rechnen hat den Vorteil,
dass Aktien mit ganz unterschiedlichen Kursniveaus (z. B. 30 € vs. 500 €)
direkt vergleichbar werden.
"""

import os

import numpy as np    # numerische Grundbibliothek (schnelles Rechnen mit Zahlenreihen)
import pandas as pd   # Tabellen-Bibliothek (DataFrame = Tabelle, Series = Spalte)
from .config import *  # alle Einstellungen und das Logbuch aus config.py übernehmen

def download_data(tickers: list, spy_ticker: str, start: str, end: str):
    """Lädt Tages-Schlusskurse und berechnet daraus tägliche Renditen.

    Eingaben:
      tickers    — Liste der Aktienkürzel (z. B. ["AAPL", "MSFT", …])
      spy_ticker — Kürzel des Marktindex-Fonds (hier "SPY")
      start, end — gewünschter Zeitraum als Text "JJJJ-MM-TT"

    Rückgabe (vier Dinge auf einmal):
      asset_px   — Tabelle der täglichen Kurse aller verfügbaren Aktien
      asset_ret  — Tabelle der täglichen Renditen dieser Aktien
      spy_ret    — Spalte mit den täglichen Renditen des Marktindex
      available  — Liste der Aktien, für die tatsächlich Daten kamen
    """
    log.info(f"Lade Marktdaten: {len(tickers)} Assets + SPY | {start} → {end}")

    # ---- Kursspeicher --------------------------------------------------
    # Yahoo liefert bei zwei Abrufen NICHT bitgleiche Kurse: gemessen an zwei
    # Downloads im Abstand weniger Minuten wichen 36 553 von 45 285 Werten ab,
    # relativ bis 1,3e-6. Winzig — aber es ist die EINGANGSGRÖSSE, und über
    # Kovarianzschätzung und Optimierer schlägt sie bis in die Kennzahlen
    # durch (Markowitz-Gewichte im ersten Monat: Δ ≈ 1,4e-3).
    # Ein Backtest, dessen Rohdaten sich unter der Hand ändern, ist nicht
    # reproduzierbar. Deshalb wird der erste Download in PRICE_CACHE
    # abgelegt und danach von dort gelesen. Neu laden: Datei löschen.
    if PRICE_CACHE and os.path.exists(PRICE_CACHE):
        raw = pd.read_pickle(PRICE_CACHE)
        log.info(f"  Kurse aus dem Speicher: {PRICE_CACHE}")
    else:
        # Download aller Kürzel in einem Rutsch; wir behalten nur die Spalte
        # "Close" (Schlusskurs). auto_adjust=True rechnet Dividenden und
        # Aktiensplits in die Kurse ein — sonst sähe eine Dividendenzahlung
        # fälschlich wie ein Kursverlust aus.
        raw = yf.download(
            tickers + [spy_ticker], start=start, end=end,
            auto_adjust=True, progress=False,
        )["Close"]
        if PRICE_CACHE:
            os.makedirs(os.path.dirname(PRICE_CACHE) or ".", exist_ok=True)
            raw.to_pickle(PRICE_CACHE)
            log.info(f"  Kurse abgelegt unter: {PRICE_CACHE} "
                     f"(Abruf {pd.Timestamp.now():%d.%m.%Y %H:%M})")

    # Prüfen, ob Yahoo für alle Kürzel Daten geliefert hat; fehlende melden.
    missing   = [t for t in tickers + [spy_ticker] if t not in raw.columns]
    if missing:
        log.warning(f"Fehlende Tickers: {missing}")

    # Nur mit den tatsächlich verfügbaren Aktien weiterarbeiten.
    # dropna(how="any") wirft jeden Tag weg, an dem AUCH NUR EINE Aktie keinen
    # Kurs hat — so sind alle Zeitreihen exakt gleich lang und synchron.
    available = [t for t in tickers if t in raw.columns]
    prices    = raw[available + [spy_ticker]].dropna(how="any")
    log.info(f"  Verfügbare Assets: {len(available)} | Handelstage: {len(prices)}")

    # Einfache (arithmetische) Renditen: P_t / P_{t-1} - 1.
    # pct_change() macht genau diese Rechnung für jede Spalte automatisch;
    # der erste Tag hat keinen Vortag und wird per dropna() entfernt.
    # Bewusst KEINE Log-Renditen: Einfache Renditen sind über Assets additiv,
    # d.h. die Portfoliorendite ist exakt die gewichtete Summe der Asset-
    # Renditen ( sum_i w_i * r_i ). Log-Renditen sind das NICHT — ihre
    # gewichtete Summe über Assets ergibt keine gültige Portfoliorendite und
    # wäre inkonsistent mit der Kennzahl-Berechnung via (1 + r).cumprod().
    daily_returns = prices.pct_change(fill_method=None).dropna()
    spy_ret   = daily_returns[spy_ticker]   # Marktrendite separat herausziehen
    asset_ret = daily_returns[available]    # Renditen der 15 Aktien
    asset_px  = prices[available]           # Kurse der 15 Aktien
    return asset_px, asset_ret, spy_ret, available
