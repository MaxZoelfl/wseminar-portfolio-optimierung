"""Tests für die Signifikanz-/Overfitting-Verfahren.

FÜR EINSTEIGER: Automatische Prüfprogramme (Erklärung des Testprinzips im
Kopf von tests/test_metrics.py). Diese Datei prüft die Statistik-Werkzeuge
aus portfolio/significance.py mit konstruierten Fällen, deren richtige
Antwort man KENNT: Wo es keinen echten Unterschied gibt, darf der Test
keinen finden — wo ein klarer eingebaut wurde, muss er anschlagen.
"""
import numpy as np
import pandas as pd

from portfolio.significance import (
    sharpe_difference_test, holm_bonferroni,
    expected_max_sharpe, deflated_sharpe_ratio,
)


# ---- Holm-Bonferroni ------------------------------------------------------
def test_holm_monotone_and_reject():
    # Handrechnung: Bei 3 Tests wird der kleinste p-Wert mit 3 multipliziert
    # (0.01·3 = 0.03 < 0.05 → abgelehnt). Die anderen landen nach Korrektur
    # über 0.05 und bleiben "nicht signifikant".
    res = holm_bonferroni([0.01, 0.04, 0.03], alpha=0.05)
    adj = res["p_adjusted"]
    # kleinster p: 0.01*3 = 0.03 (verworfen); übrige > 0.05 nach Korrektur
    assert abs(adj[0] - 0.03) < 1e-12
    assert res["reject"][0] and not res["reject"][1] and not res["reject"][2]


def test_holm_adjusted_geq_raw():
    # Grundeigenschaften der Korrektur: angepasste p-Werte dürfen nie KLEINER
    # als die rohen sein (Korrektur = verschärfen) und nie über 1.0 liegen.
    raw = [0.001, 0.02, 0.2, 0.5]
    adj = holm_bonferroni(raw)["p_adjusted"]
    assert (adj >= np.array(raw) - 1e-12).all()
    assert (adj <= 1.0 + 1e-12).all()


# ---- Sharpe-Differenz-Test (Ledoit-Wolf 2008) -----------------------------
def test_sharpe_test_no_real_difference_not_significant():
    # zwei unabhängige Reihen aus derselben Verteilung -> kein echter Unterschied
    # → der Test darf hier NICHT signifikant werden (sonst produziert er
    # Fehlalarme). n_boot ist im Test kleiner als im Ernstfall (Tempo).
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0.0005, 0.01, 1500))
    b = pd.Series(rng.normal(0.0005, 0.01, 1500))
    res = sharpe_difference_test(a, b, n_boot=399, seed=1)
    assert res["p_value"] > 0.10           # nicht signifikant


def test_sharpe_test_identical_series_degenerate():
    # identische Reihen: Differenz exakt 0, keine Unsicherheit -> p = 1.0
    # (prüft den Sonderfall-Zweig "se == 0" im Code, der den Bootstrap umgeht)
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    res = sharpe_difference_test(r, r.copy(), n_boot=199, seed=1)
    assert abs(res["diff_annual"]) < 1e-9
    assert res["p_value"] == 1.0


def test_sharpe_test_detects_clear_difference():
    # A = B + konstanter Alpha-Versatz (Rauschen hebt sich in der Differenz auf)
    # → hier GIBT es einen echten, klaren Unterschied; der Test MUSS
    # anschlagen (p < 0.05). Prüft die "Trennschärfe" des Verfahrens.
    rng = np.random.default_rng(2)
    base = rng.normal(0.0, 0.01, 2000)
    a = pd.Series(base + 0.0010)
    b = pd.Series(base)
    res = sharpe_difference_test(a, b, n_boot=499, seed=3)
    assert res["sharpe_a"] > res["sharpe_b"]
    assert res["p_value"] < 0.05           # klarer, robuster Unterschied


# ---- Deflated Sharpe Ratio ------------------------------------------------
def test_expected_max_sharpe_increases_with_trials():
    # Kernlogik der "Glücks-Hürde": Wer MEHR Strategien ausprobiert, dessen
    # zufällig beste sieht besser aus — die Hürde muss mit N wachsen.
    s1 = expected_max_sharpe(5, 0.1)
    s2 = expected_max_sharpe(50, 0.1)
    assert s2 > s1 > 0


def test_dsr_in_unit_interval_and_decreases_with_more_trials():
    # Die DSR ist eine Wahrscheinlichkeit (muss zwischen 0 und 1 liegen),
    # und: mehr getestete Strategien → strengere Bewertung → kleinere DSR.
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.0008, 0.01, 2000))
    sr = r.mean() / r.std(ddof=1)
    # gleiche Streuung der Trial-Sharpes, aber unterschiedlich viele Versuche:
    few_trials = [sr + 0.1, sr, sr - 0.1]            # N = 3
    many_trials = few_trials * 67                    # N = 201
    few = deflated_sharpe_ratio(r, few_trials, freq=252)
    many = deflated_sharpe_ratio(r, many_trials, freq=252)
    assert 0.0 <= few["deflated_sr"] <= 1.0
    assert 0.0 <= many["deflated_sr"] <= 1.0
    assert many["deflated_sr"] <= few["deflated_sr"]   # mehr Selektion -> strenger
