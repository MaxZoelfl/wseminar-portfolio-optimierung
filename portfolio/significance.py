"""
Wissenschaftlich fundierte Signifikanz- und Overfitting-Analysen.

Implementiert drei in der quantitativen Finanzliteratur etablierte Verfahren,
die den naiven i.i.d.-Bootstrap aus v4.1 ersetzen bzw. ergänzen:

  - sharpe_difference_test
        HAC-studentisierter Circular-Block-Bootstrap-Test für die Differenz
        zweier Sharpe Ratios (berücksichtigt Autokorrelation & Vol-Clustering).
        Ledoit, O. & Wolf, M. (2008): Robust performance hypothesis testing
        with the Sharpe ratio. Journal of Empirical Finance 15(5), 850-859.
        Block-Bootstrap: Politis & Romano (1994), JASA 89(428), 1303-1313.

  - holm_bonferroni
        Korrektur für multiples Testen (mehrere paarweise Vergleiche).
        Holm, S. (1979): A Simple Sequentially Rejective Multiple Test Procedure.
        Motivation: Harvey, Liu & Zhu (2016), Review of Financial Studies 29(1).

  - deflated_sharpe_ratio
        Wahrscheinlichkeit, dass die wahre Sharpe > 0 ist, nachdem für Selektion
        über N Versuche und Nicht-Normalität korrigiert wurde.
        Bailey, D. H. & López de Prado, M. (2014): The Deflated Sharpe Ratio.
        Journal of Portfolio Management 40(5), 94-107.

Konvention: Es wird auf PER-PERIODE-Renditen gerechnet (z. B. täglich) und mit
der ARITHMETISCHEN Sharpe (Standarddefinition nach Sharpe 1994). Für die Ausgabe
werden Sharpe-Werte mit sqrt(freq) annualisiert. Teststatistik und p-Wert des
studentisierten Tests sind gegenüber dieser Skalierung invariant.

Layer: nutzt nur numpy/pandas/scipy, keine projektinternen Module.
"""
import math
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _align(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    return df["a"].to_numpy(float), df["b"].to_numpy(float)


def _sharpe_per_period(x):
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else 0.0


def _nw_bandwidth(T):
    """Automatische Bandbreite (Newey & West 1994)."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def _hac_cov(U, L):
    """Newey-West/Bartlett-HAC-Schätzer der langfristigen Kovarianz von U (T×k)."""
    T = U.shape[0]
    Uc = U - U.mean(axis=0)
    S = (Uc.T @ Uc) / T
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        G = (Uc[lag:].T @ Uc[:-lag]) / T
        S = S + w * (G + G.T)
    return S


def _diff_and_se(a, b, L=None):
    """
    Sharpe-Differenz (per Periode) und HAC-Standardfehler via Delta-Methode.
    Momentvektor je t: u_t = (a, b, a^2, b^2); Sh = mu/sqrt(m - mu^2).
    """
    T = len(a)
    if L is None:
        L = _nw_bandwidth(T)
    mu_a, mu_b = a.mean(), b.mean()
    m_a, m_b = (a ** 2).mean(), (b ** 2).mean()
    s_a = math.sqrt(max(m_a - mu_a ** 2, 1e-18))
    s_b = math.sqrt(max(m_b - mu_b ** 2, 1e-18))
    diff = mu_a / s_a - mu_b / s_b
    grad = np.array([
        m_a / s_a ** 3,          # d Δ / d mu_a
        -m_b / s_b ** 3,         # d Δ / d mu_b
        -0.5 * mu_a / s_a ** 3,  # d Δ / d m_a
        0.5 * mu_b / s_b ** 3,   # d Δ / d m_b
    ])
    U = np.column_stack([a, b, a ** 2, b ** 2])
    Psi = _hac_cov(U, L)
    var = float(grad @ Psi @ grad / T)
    se = math.sqrt(var) if var > 0 else float("nan")
    return diff, se


# ---------------------------------------------------------------------------
# 1) Ledoit-Wolf (2008): robuster Sharpe-Differenz-Test
# ---------------------------------------------------------------------------
def sharpe_difference_test(returns_a, returns_b, rf=0.0, freq=252,
                           block_size=None, n_boot=4999, seed=42):
    """
    Zweiseitiger Test H0: Sharpe(A) = Sharpe(B) über studentisierten
    Circular-Block-Bootstrap (Ledoit & Wolf 2008).
    """
    a, b = _align(returns_a, returns_b)
    rf_p = rf / freq
    a = a - rf_p
    b = b - rf_p
    T = len(a)
    if T < 20:
        raise ValueError("Zu wenige Beobachtungen für den Block-Bootstrap.")
    if block_size is None:
        block_size = max(1, int(np.ceil(T ** (1.0 / 3.0))))

    diff, se = _diff_and_se(a, b)

    # Entarteter Fall: keine Schätzunsicherheit der Differenz (z. B. identische
    # oder perfekt deterministisch verknüpfte Reihen). Dann ist der Bootstrap
    # nicht definiert; das Ergebnis ist eindeutig.
    if not (se == se) or se <= 0:   # se NaN oder 0
        no_diff = abs(diff) < 1e-12
        sqf = math.sqrt(freq)
        return {
            "sharpe_a": _sharpe_per_period(a) * sqf,
            "sharpe_b": _sharpe_per_period(b) * sqf,
            "diff_annual": diff * sqf,
            "se_annual": 0.0,
            "statistic": 0.0 if no_diff else float("inf"),
            "p_value": 1.0 if no_diff else 0.0,
            "block_size": block_size,
            "n_boot": 0,
            "T": T,
        }

    stat = diff / se

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block_size))
    exceed = 0
    valid = 0
    for _ in range(n_boot):
        starts = rng.integers(0, T, n_blocks)
        idx = np.concatenate([(np.arange(s, s + block_size) % T) for s in starts])[:T]
        d_b, se_b = _diff_and_se(a[idx], b[idx])
        if se_b and se_b > 0 and not math.isnan(stat):
            valid += 1
            if abs((d_b - diff) / se_b) >= abs(stat):
                exceed += 1
    p_value = exceed / valid if valid else float("nan")

    sqf = math.sqrt(freq)
    return {
        "sharpe_a": _sharpe_per_period(a) * sqf,
        "sharpe_b": _sharpe_per_period(b) * sqf,
        "diff_annual": diff * sqf,
        "se_annual": (se * sqf) if se == se else float("nan"),
        "statistic": stat,
        "p_value": p_value,
        "block_size": block_size,
        "n_boot": n_boot,
        "T": T,
    }


# ---------------------------------------------------------------------------
# 2) Holm-Bonferroni: Korrektur für multiples Testen
# ---------------------------------------------------------------------------
def holm_bonferroni(pvalues, alpha=0.05):
    """
    Holm (1979) Step-down-Korrektur. Gibt angepasste p-Werte (monoton) und
    Ablehnungsentscheidungen zurück. Trennschärfer als reines Bonferroni.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return {"p_adjusted": adj, "reject": adj < alpha, "alpha": alpha, "n_tests": m}


# ---------------------------------------------------------------------------
# 3) Deflated Sharpe Ratio (Bailey & López de Prado 2014)
# ---------------------------------------------------------------------------
def expected_max_sharpe(n_trials, sr_std):
    """
    Erwartetes Maximum der Sharpe (per Periode) über N unabhängige Versuche
    unter H0 (wahre Sharpe = 0). Formel nach Bailey & López de Prado (2014).
    """
    N = int(n_trials)
    if N < 2 or sr_std <= 0:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / N)
    b = norm.ppf(1.0 - 1.0 / (N * math.e))
    return float(sr_std * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe_ratio(returns, trial_sharpes_per_period, rf=0.0, freq=252):
    """
    Deflated Sharpe Ratio der ausgewählten Strategie.
    trial_sharpes_per_period: per-Periode-Sharpe ALLER getesteten Strategien
    (zur Schätzung der Selektionsbreite N und der Streuung der Sharpe).
    """
    x = pd.Series(returns).dropna().to_numpy(float) - rf / freq
    T = len(x)
    sr = _sharpe_per_period(x)
    g3 = float(skew(x, bias=False))
    g4 = float(kurtosis(x, fisher=False, bias=False))   # normal = 3
    trials = np.asarray(trial_sharpes_per_period, dtype=float)
    N = len(trials)
    sr_std = float(trials.std(ddof=1)) if N > 1 else 0.0
    sr0 = expected_max_sharpe(N, sr_std)
    denom = math.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2, 1e-12))
    dsr = float(norm.cdf((sr - sr0) * math.sqrt(max(T - 1, 1)) / denom))
    sqf = math.sqrt(freq)
    return {
        "sharpe_annual": sr * sqf,
        "sr_per_period": sr,
        "sr0_per_period": sr0,
        "sr0_annual": sr0 * sqf,
        "n_trials": N,
        "skew": g3,
        "kurtosis": g4,
        "deflated_sr": dsr,
        "significant": dsr > 0.95,
        "T": T,
    }


def deflated_sharpe_from_strategies(returns_df, selected, rf=0.0, freq=252):
    """Bequemlichkeit: Trial-Sharpes aus allen Spalten von returns_df berechnen."""
    trials = [
        _sharpe_per_period(returns_df[c].dropna().to_numpy(float) - rf / freq)
        for c in returns_df.columns
    ]
    return deflated_sharpe_ratio(returns_df[selected], trials, rf=rf, freq=freq)
