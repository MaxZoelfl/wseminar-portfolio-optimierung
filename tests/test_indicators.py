"""Indikatoren, Cross-Sectional-Ranks, Monats-Aufzinsung und Look-Ahead-Schutz.

FÜR EINSTEIGER: Automatische Prüfprogramme (Erklärung des Testprinzips im
Kopf von tests/test_metrics.py). Diese Datei prüft die Merkmals-Berechnung
aus portfolio/indicators.py — besonders wichtig sind die letzten beiden
Tests, die die WISSENSCHAFTLICHE Sauberkeit absichern: korrektes Aufzinsen
der Monatsrendite und der Schutz davor, dass das KI-Modell in die Zukunft
schauen könnte (Look-Ahead-Bias).
"""
import numpy as np
import pandas as pd

from portfolio.indicators import (
    compute_rsi, compute_momentum, add_cross_sectional_ranks, aggregate_to_monthly,
)


def test_rsi_within_bounds():
    # Der RSI ist per Konstruktion eine 0–100-Skala. Wir erzeugen 300 Tage
    # zufällige Kurse (fester Seed 0 → immer dieselben "Zufalls"-Zahlen,
    # damit der Test reproduzierbar ist) und prüfen, dass KEIN Wert die
    # Skala verlässt.
    rng = np.random.default_rng(0)
    prices = pd.Series(
        100 * np.cumprod(1 + rng.normal(0, 0.01, 300)),
        index=pd.bdate_range("2020-01-01", periods=300),
    )
    rsi = compute_rsi(prices).dropna()
    assert len(rsi) > 0
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_momentum_equals_rolling_sum():
    # Momentum über 3 Tage = Summe der letzten 3 Tagesrenditen. Bei den
    # Renditen 0,1,2,…,9 muss der letzte Wert also 7+8+9 = 24 sein.
    r = pd.Series(np.arange(10.0), index=pd.bdate_range("2020-01-01", periods=10))
    mom = compute_momentum(r, periods=[3])
    assert abs(mom["mom_3d"].iloc[-1] - (7 + 8 + 9)) < 1e-9


def test_cross_sectional_rank_top_is_one():
    # Drei Aktien im selben Monat mit RSI 10 < 20 < 30: Die beste (C) muss
    # den Perzentil-Rang 1.0 bekommen, die schlechteste (A) einen kleineren.
    idx = pd.to_datetime(["2020-01-31"] * 3)
    df = pd.DataFrame({"rsi": [10, 20, 30], "ticker": ["A", "B", "C"]}, index=idx)
    out = add_cross_sectional_ranks(df)
    assert out["rsi_rank"].iloc[2] == 1.0                 # höchster RSI -> Rang 1
    assert out["rsi_rank"].iloc[0] < out["rsi_rank"].iloc[2]


def _daily(ticker_ret, cols=("rsi",)):
    """Hilfsfunktion: minimale Dummy-Indikatortabelle (konstant 50), damit
    aggregate_to_monthly etwas zum Zusammenfassen hat — getestet werden hier
    ja nur die Rendite-Spalten, nicht die Indikatorwerte."""
    idx = ticker_ret.index
    ind = pd.DataFrame({c: 50.0 for c in cols}, index=idx)
    return ind


def test_monthly_return_is_compounded():
    # v4.1: Monatsrendite = prod(1+r) - 1 (nicht Summe)
    # Hintergrund: Renditen verketten sich multiplikativ (+10 % dann −10 %
    # ist NICHT 0 %). Der Test rechnet die Januar-Rendite von Hand nach.
    idx = pd.bdate_range("2020-01-01", "2020-02-28")
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.001, 0.01, len(idx)), index=idx)
    monthly = aggregate_to_monthly({"A": _daily(r)}, pd.DataFrame({"A": r}), ["A"])
    jan = r[r.index.month == 1]
    expected_jan = (1 + jan).prod() - 1
    got_jan = monthly[monthly["ticker"] == "A"]["monthly_ret"].iloc[0]
    assert abs(got_jan - expected_jan) < 1e-9


def test_target_is_next_month_and_last_is_nan():
    # Look-Ahead-Schutz: Ziel[t] = Monatsrendite[t+1]; letzter Monat -> NaN
    # Das Vorhersageziel eines Monats muss die Rendite des FOLGEmonats sein,
    # und der letzte Monat darf kein Ziel haben (seine Zukunft ist unbekannt).
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    r = pd.Series(0.001, index=idx)
    m = aggregate_to_monthly({"A": _daily(r)}, pd.DataFrame({"A": r}), ["A"])
    ma = m[m["ticker"] == "A"]
    assert abs(ma["target_next_month"].iloc[0] - ma["monthly_ret"].iloc[1]) < 1e-12
    assert pd.isna(ma["target_next_month"].iloc[-1])
