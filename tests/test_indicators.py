"""Indikatoren, Cross-Sectional-Ranks, Monats-Aufzinsung und Look-Ahead-Schutz."""
import numpy as np
import pandas as pd

from portfolio.indicators import (
    compute_rsi, compute_momentum, add_cross_sectional_ranks, aggregate_to_monthly,
)


def test_rsi_within_bounds():
    rng = np.random.default_rng(0)
    prices = pd.Series(
        100 * np.cumprod(1 + rng.normal(0, 0.01, 300)),
        index=pd.bdate_range("2020-01-01", periods=300),
    )
    rsi = compute_rsi(prices).dropna()
    assert len(rsi) > 0
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_momentum_equals_rolling_sum():
    r = pd.Series(np.arange(10.0), index=pd.bdate_range("2020-01-01", periods=10))
    mom = compute_momentum(r, periods=[3])
    assert abs(mom["mom_3d"].iloc[-1] - (7 + 8 + 9)) < 1e-9


def test_cross_sectional_rank_top_is_one():
    idx = pd.to_datetime(["2020-01-31"] * 3)
    df = pd.DataFrame({"rsi": [10, 20, 30], "ticker": ["A", "B", "C"]}, index=idx)
    out = add_cross_sectional_ranks(df)
    assert out["rsi_rank"].iloc[2] == 1.0                 # höchster RSI -> Rang 1
    assert out["rsi_rank"].iloc[0] < out["rsi_rank"].iloc[2]


def _daily(ticker_ret, cols=("rsi",)):
    idx = ticker_ret.index
    ind = pd.DataFrame({c: 50.0 for c in cols}, index=idx)
    return ind


def test_monthly_return_is_compounded():
    # v4.1: Monatsrendite = prod(1+r) - 1 (nicht Summe)
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
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    r = pd.Series(0.001, index=idx)
    m = aggregate_to_monthly({"A": _daily(r)}, pd.DataFrame({"A": r}), ["A"])
    ma = m[m["ticker"] == "A"]
    assert abs(ma["target_next_month"].iloc[0] - ma["monthly_ret"].iloc[1]) < 1e-12
    assert pd.isna(ma["target_next_month"].iloc[-1])
