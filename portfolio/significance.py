"""
Wissenschaftlich fundierte Signifikanz- und Overfitting-Analysen.

FÜR EINSTEIGER — WAS MACHT DIESE DATEI?
Angenommen, im Backtest hat Strategie A eine Sharpe Ratio von 0,95 und
Strategie B eine von 0,80. Ist A dann "besser"? Nicht unbedingt! Zehn Jahre
Börse sind statistisch gesehen eine kleine Stichprobe, und der Unterschied
könnte reiner Zufall sein — so wie eine Münze auch mal 6-mal hintereinander
Kopf zeigt. Diese Datei enthält die statistischen Werkzeuge, die genau das
prüfen. Das Kernkonzept ist der p-WERT: die Wahrscheinlichkeit, einen
mindestens so großen Unterschied zu beobachten, WENN in Wahrheit gar keiner
besteht. Üblich: p < 0,05 (unter 5 %) gilt als "statistisch signifikant".

Implementiert sind drei in der quantitativen Finanzliteratur etablierte
Verfahren, die den naiven i.i.d.-Bootstrap aus v4.1 ersetzen bzw. ergänzen:

  - sharpe_difference_test
        Prüft, ob sich zwei Sharpe Ratios ECHT unterscheiden. Kernidee
        "Bootstrap": Aus der beobachteten Renditereihe werden tausende
        künstliche Alternativ-Historien zusammengewürfelt; so sieht man, wie
        stark die Kennzahl allein durch Zufall streut. Gewürfelt wird in
        zusammenhängenden BLÖCKEN, weil Börsenrenditen zeitlich voneinander
        abhängen (ruhige und turbulente Phasen klumpen) — einzelne Tage zu
        mischen würde diese Struktur zerstören.
        HAC-studentisierter Circular-Block-Bootstrap-Test für die Differenz
        zweier Sharpe Ratios (berücksichtigt Autokorrelation & Vol-Clustering).
        Ledoit, O. & Wolf, M. (2008): Robust performance hypothesis testing
        with the Sharpe ratio. Journal of Empirical Finance 15(5), 850-859.
        Block-Bootstrap: Politis & Romano (1994), JASA 89(428), 1303-1313.

  - holm_bonferroni
        Korrektur für multiples Testen (mehrere paarweise Vergleiche).
        Problem: Wer 5 Vergleiche macht, hat 5 Chancen auf einen
        Zufallstreffer — die p-Werte müssen dafür verschärft werden.
        Holm, S. (1979): A Simple Sequentially Rejective Multiple Test Procedure.
        Motivation: Harvey, Liu & Zhu (2016), Review of Financial Studies 29(1).

  - deflated_sharpe_ratio
        Schutz vor "Strategie-Shopping": Wer N Strategien testet und die
        beste präsentiert, hat allein durch Auswahl eine geschönte Sharpe.
        Die DSR gibt die Wahrscheinlichkeit an, dass die wahre Sharpe > 0 ist,
        nachdem für Selektion über N Versuche und Nicht-Normalität (Schiefe,
        schwere Verteilungsränder) korrigiert wurde.
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
from scipy.stats import norm, skew, kurtosis   # Normalverteilung, Schiefe, Wölbung

# Euler-Mascheroni-Konstante γ ≈ 0.5772 — taucht in der Formel für das
# erwartete Maximum vieler Zufallsgrößen auf (expected_max_sharpe unten).
EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _align(a, b):
    """Bringt zwei Renditereihen auf gemeinsame Tage.

    Verglichen werden darf nur Tag für Tag paarweise; Tage, an denen eine
    der beiden Reihen fehlt, werden entfernt (dropna). Rückgabe als schnelle
    numpy-Zahlenfelder.
    """
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    return df["a"].to_numpy(float), df["b"].to_numpy(float)


def _sharpe_per_period(x):
    """Sharpe Ratio PRO PERIODE (z. B. pro Tag): Mittelwert / Standardabweichung.

    (ddof=1 = Stichproben-Standardabweichung; der if-Teil schützt wie üblich
    vor Division durch null.)
    """
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else 0.0


def _nw_bandwidth(T):
    """Automatische Bandbreite (Newey & West 1994).

    Legt fest, bis zu welchem zeitlichen Abstand ("Lag") Abhängigkeiten
    zwischen den Tagen im HAC-Schätzer (s. u.) berücksichtigt werden.
    Faustformel aus der Literatur, wächst langsam mit der Datenmenge T.
    """
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def _hac_cov(U, L):
    """Newey-West/Bartlett-HAC-Schätzer der langfristigen Kovarianz von U (T×k).

    Hintergrund: Die übliche Varianzformel unterstellt, dass die Beobachtungen
    voneinander unabhängig sind. Börsenrenditen sind das nicht (turbulente
    Tage folgen aufeinander). Der HAC-Schätzer ("Heteroskedasticity and
    Autocorrelation Consistent") korrigiert das: Er addiert zur normalen
    Kovarianz auch die Kovarianzen zwischen zeitversetzten Beobachtungen
    (Lag 1 bis L), mit linear abfallenden Bartlett-Gewichten w — nahe
    Nachbarn zählen voll, weiter entfernte immer weniger.
    """
    T = U.shape[0]
    Uc = U - U.mean(axis=0)          # Daten zentrieren (Mittelwert abziehen)
    S = (Uc.T @ Uc) / T              # gewöhnliche Kovarianz (Lag 0)
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)    # Bartlett-Gewicht: fällt linear auf 0
        G = (Uc[lag:].T @ Uc[:-lag]) / T   # Kovarianz mit um `lag` versetzten Werten
        S = S + w * (G + G.T)        # symmetrisch aufaddieren
    return S


def _diff_and_se(a, b, L=None):
    """
    Sharpe-Differenz (per Periode) und HAC-Standardfehler via Delta-Methode.
    Momentvektor je t: u_t = (a, b, a^2, b^2); Sh = mu/sqrt(m - mu^2).

    In Worten: Wir wollen zwei Dinge wissen —
      1. diff : Wie groß ist der Sharpe-Unterschied zwischen A und B?
      2. se   : Wie UNSICHER ist diese Zahl (ihr "Standardfehler")?
    Erst das Verhältnis diff/se sagt, ob der Unterschied groß RELATIV zu
    seiner Unsicherheit ist. Die "Delta-Methode" ist das Standardrezept der
    Statistik, um die Unsicherheit einer VERKETTETEN Größe (Sharpe = Funktion
    von Mittelwerten) aus der Unsicherheit ihrer Bausteine herzuleiten: Man
    braucht dafür die Ableitungen (den Gradienten) der Formel nach den
    Bausteinen — genau die vier grad-Einträge unten.
    """
    T = len(a)
    if L is None:
        L = _nw_bandwidth(T)
    # Die vier Bausteine ("Momente"): Mittelwerte und mittlere Quadrate.
    mu_a, mu_b = a.mean(), b.mean()
    m_a, m_b = (a ** 2).mean(), (b ** 2).mean()
    # Standardabweichungen daraus (max(...,1e-18) verhindert √negativ durch
    # Rundungsfehler). Es gilt: Varianz = E[x²] − (E[x])².
    s_a = math.sqrt(max(m_a - mu_a ** 2, 1e-18))
    s_b = math.sqrt(max(m_b - mu_b ** 2, 1e-18))
    diff = mu_a / s_a - mu_b / s_b          # Sharpe(A) − Sharpe(B), pro Periode
    # Gradient: Ableitungen der Differenz nach den vier Bausteinen.
    grad = np.array([
        m_a / s_a ** 3,          # d Δ / d mu_a
        -m_b / s_b ** 3,         # d Δ / d mu_b
        -0.5 * mu_a / s_a ** 3,  # d Δ / d m_a
        0.5 * mu_b / s_b ** 3,   # d Δ / d m_b
    ])
    # Langfrist-Kovarianz der Bausteine (HAC, s. o.), dann Delta-Formel:
    # Var(Δ) ≈ gradᵀ · Ψ · grad / T.
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

    Übersetzung der Fachbegriffe:
      - H0 ("Nullhypothese"): die Ausgangsannahme "beide Strategien sind in
        Wahrheit gleich gut". Der Test misst, wie unplausibel die Daten unter
        dieser Annahme wären (→ p-Wert).
      - "zweiseitig": Abweichungen in BEIDE Richtungen zählen (A besser oder
        B besser).
      - "Bootstrap": tausende künstliche Datensätze durch Neu-Zusammenwürfeln
        der echten Daten erzeugen, um die Zufallsstreuung zu ermitteln.
      - "Block": gewürfelt werden zusammenhängende Zeitblöcke (Länge ≈ ∛T),
        damit die zeitliche Abhängigkeit der Renditen erhalten bleibt;
        "circular": am Datenende wird nahtlos zum Anfang übergelaufen, damit
        jeder Tag gleich oft gezogen werden kann.
      - "studentisiert": verglichen wird nicht die rohe Differenz, sondern
        Differenz ÷ ihr Standardfehler — das macht den Test deutlich
        zuverlässiger (Kernpunkt von Ledoit & Wolf 2008).

    Rückgabe: ein dict u. a. mit den beiden (annualisierten) Sharpe Ratios,
    der Differenz, der Teststatistik und dem p-Wert.
    """
    # Reihen synchronisieren und den risikofreien Zins (auf Tagesbasis
    # heruntergebrochen) abziehen → "Überschussrenditen".
    a, b = _align(returns_a, returns_b)
    rf_p = rf / freq
    a = a - rf_p
    b = b - rf_p
    T = len(a)
    if T < 20:
        raise ValueError("Zu wenige Beobachtungen für den Block-Bootstrap.")
    if block_size is None:
        # Faustregel für die Blocklänge: Kubikwurzel der Datenlänge.
        block_size = max(1, int(np.ceil(T ** (1.0 / 3.0))))

    diff, se = _diff_and_se(a, b)

    # Entarteter Fall: keine Schätzunsicherheit der Differenz (z. B. identische
    # oder perfekt deterministisch verknüpfte Reihen). Dann ist der Bootstrap
    # nicht definiert; das Ergebnis ist eindeutig: identisch → p = 1 (sicher
    # kein Unterschied), sonst → p = 0.
    if not (se == se) or se <= 0:   # "se != se" ist der klassische NaN-Check
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

    # Teststatistik: Differenz in Einheiten ihrer eigenen Unsicherheit.
    stat = diff / se

    # ---- Der Bootstrap-Kern -------------------------------------------------
    # n_boot-mal (Standard 4999): künstliche Historie bauen und schauen, wie
    # oft dort eine mindestens so extreme Statistik auftritt wie die echte.
    # WICHTIG: In der Bootstrap-Welt ist "diff" der wahre Wert — deshalb wird
    # (d_b − diff) zentriert; das simuliert die Verteilung UNTER H0.
    rng = np.random.default_rng(seed)   # Zufallsgenerator mit festem Seed (reproduzierbar)
    n_blocks = int(np.ceil(T / block_size))
    exceed = 0   # Zähler: Bootstrap-Statistik ≥ echte Statistik
    valid = 0    # Zähler: verwertbare Bootstrap-Durchläufe
    for _ in range(n_boot):
        # Zufällige Blockanfänge ziehen und daraus die Indexfolge bauen;
        # "% T" setzt das Überlaufen am Datenende um (circular).
        starts = rng.integers(0, T, n_blocks)
        idx = np.concatenate([(np.arange(s, s + block_size) % T) for s in starts])[:T]
        d_b, se_b = _diff_and_se(a[idx], b[idx])
        if se_b and se_b > 0 and not math.isnan(stat):
            valid += 1
            if abs((d_b - diff) / se_b) >= abs(stat):
                exceed += 1
    # p-Wert = Anteil der Zufalls-Historien, die mindestens so extrem waren.
    p_value = exceed / valid if valid else float("nan")

    # Für die Ausgabe auf Jahresbasis skalieren (×√252 bei Tagesdaten).
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

    Warum überhaupt korrigieren? Bei 5 Tests mit je 5 % Irrtumsrisiko liegt
    die Chance auf MINDESTENS einen Zufallstreffer schon bei ~23 %. Die
    simple Bonferroni-Lösung multipliziert jeden p-Wert mit der Testanzahl m.
    Holm ist die klügere Variante ("step-down"): p-Werte aufsteigend sortieren
    und den kleinsten mit m, den zweitkleinsten mit m−1 usw. multiplizieren —
    gleicher Schutz vor Zufallstreffern, aber weniger übervorsichtig.
    Das "running maximum" unten stellt sicher, dass die korrigierten p-Werte
    in der Sortierreihenfolge nie wieder absinken (Monotonie-Eigenschaft
    des Verfahrens); min(…, 1.0) deckelt bei 1 (eine Wahrscheinlichkeit
    über 100 % gibt es nicht).
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)        # Reihenfolge der p-Werte (kleinster zuerst)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])   # Faktor m, m−1, m−2, …
        adj[i] = min(running, 1.0)
    return {"p_adjusted": adj, "reject": adj < alpha, "alpha": alpha, "n_tests": m}


# ---------------------------------------------------------------------------
# 3) Deflated Sharpe Ratio (Bailey & López de Prado 2014)
# ---------------------------------------------------------------------------
def expected_max_sharpe(n_trials, sr_std):
    """
    Erwartetes Maximum der Sharpe (per Periode) über N unabhängige Versuche
    unter H0 (wahre Sharpe = 0). Formel nach Bailey & López de Prado (2014).

    Anschaulich: Selbst wenn ALLE getesteten Strategien in Wahrheit wertlos
    wären (wahre Sharpe = 0), hätte die zufällig beste von N Strategien eine
    positive Schein-Sharpe — reines Auswahlglück. Diese Funktion berechnet,
    wie hoch dieses Glücks-Maximum im Erwartungswert ausfällt. Es wächst mit
    der Anzahl der Versuche N und mit der Streuung sr_std der Sharpe-Werte.
    Das Ergebnis dient als "Hürde" (SR0), die eine Strategie erst einmal
    überspringen muss, bevor man ihr etwas glaubt.
    (norm.ppf = Quantilfunktion der Standardnormalverteilung.)
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

    Ablauf in Worten:
      1. Sharpe der gewählten Strategie berechnen (sr).
      2. Die Glücks-Hürde SR0 bestimmen (expected_max_sharpe): so viel Sharpe
         wäre allein durchs Auswählen der besten aus N Strategien zu erwarten.
      3. Prüfen, wie sicher sr ÜBER dieser Hürde liegt — unter Berücksichtigung
         von Schiefe (g3: sind Verluste extremer als Gewinne?) und Kurtosis
         (g4: wie häufig sind Extremtage? Normalverteilung hätte g4 = 3),
         denn beide machen die Sharpe-Schätzung unsicherer.
    Ergebnis "deflated_sr": eine Wahrscheinlichkeit zwischen 0 und 1, dass
    die wahre Sharpe größer als die Zufalls-Hürde ist. Erst ab 0,95 gilt
    die Strategie hier als statistisch belastbar ("significant").
    """
    # Überschussrenditen der gewählten Strategie als Zahlenfeld:
    x = pd.Series(returns).dropna().to_numpy(float) - rf / freq
    T = len(x)
    sr = _sharpe_per_period(x)
    g3 = float(skew(x, bias=False))                     # Schiefe der Verteilung
    g4 = float(kurtosis(x, fisher=False, bias=False))   # Wölbung; normal = 3
    trials = np.asarray(trial_sharpes_per_period, dtype=float)
    N = len(trials)
    sr_std = float(trials.std(ddof=1)) if N > 1 else 0.0
    sr0 = expected_max_sharpe(N, sr_std)                # die "Glücks-Hürde"
    # Standardfehler-Korrekturterm nach Bailey/López de Prado: Schiefe und
    # dicke Verteilungsränder vergrößern die Unsicherheit der Sharpe.
    denom = math.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2, 1e-12))
    # norm.cdf übersetzt die standardisierte Überschreitung der Hürde in
    # eine Wahrscheinlichkeit (Fläche unter der Normalverteilungs-Glocke).
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
    """Bequemlichkeit: Trial-Sharpes aus allen Spalten von returns_df berechnen.

    Statt die Sharpe-Werte aller Strategien von Hand zu übergeben, reicht
    hier die Rendite-Tabelle (eine Spalte je Strategie) plus der Name der
    zu prüfenden Strategie — den Rest erledigt diese Funktion.
    """
    trials = [
        _sharpe_per_period(returns_df[c].dropna().to_numpy(float) - rf / freq)
        for c in returns_df.columns
    ]
    return deflated_sharpe_ratio(returns_df[selected], trials, rf=rf, freq=freq)
