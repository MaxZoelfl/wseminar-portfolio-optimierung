"""
Performance-Kennzahlen, Hilfsfunktionen (Timer) und Signifikanztests.

FÜR EINSTEIGER — WAS MACHT DIESE DATEI?
Wenn der Backtest gelaufen ist, liegt für jede Strategie eine lange Reihe
täglicher Renditen vor. Diese Datei verdichtet solche Reihen zu den üblichen
Finanz-Kennzahlen — den "Noten", mit denen man Anlagestrategien vergleicht:

  - CAGR           : durchschnittliches Wachstum PRO JAHR, inklusive
                     Zinseszinseffekt ("Compound Annual Growth Rate").
  - Volatilität    : wie stark die Renditen um ihren Mittelwert schwanken —
                     das gängigste Maß für RISIKO. "Annualisiert" heißt: auf
                     ein Jahr hochgerechnet, damit Werte vergleichbar sind.
  - Sharpe Ratio   : DIE zentrale Kennzahl der Arbeit. Mehrertrag über dem
                     sicheren Zins, GETEILT durch die Volatilität — also
                     "Wie viel Extra-Rendite bekomme ich pro Einheit Risiko?"
                     Höher = besser.
  - Max. Drawdown  : der schlimmste zwischenzeitliche Verlust vom Höchststand
                     bis zum folgenden Tiefpunkt (z. B. −30 % im Corona-Crash).
  - Sortino/Calmar : Varianten der Sharpe Ratio, die nur Verlustschwankungen
                     bzw. den Maximalverlust als Risikomaß verwenden.

Ein wiederkehrendes Muster unten: "if v > 0 … else 0.0" — das ist ein Schutz
gegen Division durch null (mathematisch verboten, würde das Programm stoppen).
"""

import time
import numpy as np
import pandas as pd
from .config import *

def timer(func):
    """Dekorator: misst und protokolliert Laufzeit.

    Ein "Dekorator" ist eine Funktion, die eine andere Funktion umhüllt und
    ihr eine Zusatzfähigkeit gibt — hier: eine Stoppuhr. Schreibt man
    ``@timer`` über eine Funktion, wird bei jedem Aufruf automatisch die
    Ausführungsdauer gemessen und ins Logbuch geschrieben. Am Ergebnis der
    Funktion ändert sich nichts.
    """
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()            # Stoppuhr starten
        result = func(*args, **kwargs)      # die eigentliche Funktion ausführen
        log.info(f"[TIMER] {func.__name__:<40} {time.perf_counter()-t0:>7.2f}s")
        return result
    return wrapper


def cagr(returns: pd.Series, freq: int = 252) -> float:
    """Compound Annual Growth Rate aus täglichen Renditen.

    In Worten: das konstante JÄHRLICHE Wachstum, das über den gesamten
    Zeitraum zum selben Endergebnis geführt hätte. Rechnung:
      1. (1 + r).prod()  → Gesamtwachstumsfaktor: alle Tagesfaktoren
         multipliziert (aus 1 € werden z. B. 1,85 €).
      2. hoch (1/n)      → auf EIN Jahr herunterbrechen; n = Anzahl Jahre.
         (freq = 252, weil ein Börsenjahr ca. 252 Handelstage hat.)
      3. minus 1         → vom Wachstumsfaktor zurück zur Prozentzahl.
    """
    n = len(returns) / freq          # Länge der Reihe in Jahren
    return (1 + returns).prod() ** (1 / n) - 1 if n > 0 else 0.0


def annualized_vol(returns: pd.Series, freq: int = 252) -> float:
    """Annualisierte Volatilität = Standardabweichung der Tagesrenditen × √252.

    Die Standardabweichung (.std()) misst die typische Schwankungsbreite der
    täglichen Renditen. Multiplikation mit der Wurzel aus 252 rechnet sie auf
    ein Jahr hoch (Wurzel — nicht mal 252 —, weil sich unabhängige Schwankungen
    teilweise gegenseitig aufheben; die Streuung wächst nur mit √Zeit).
    """
    return returns.std() * np.sqrt(freq)


def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE,
                 freq: int = 252) -> float:
    """Sharpe Ratio = (Jahresrendite − sicherer Zins) / Jahresvolatilität.

    Beispiel: 12 % Rendite bei 4 % sicherem Zins und 16 % Volatilität
    → Sharpe = (0,12 − 0,04) / 0,16 = 0,5. Werte um 1 gelten als sehr gut.
    """
    r = cagr(returns, freq)
    v = annualized_vol(returns, freq)
    return (r - rf) / v if v > 0 else 0.0    # Schutz: bei Vola 0 keine Division


def max_drawdown(returns: pd.Series) -> float:
    """Maximaler Drawdown: tiefster prozentualer Absturz vom bisherigen Hoch.

    Rechnung Schritt für Schritt:
      cum      — Depotwert-Verlauf: kumuliertes Produkt der (1+Rendite)-Faktoren
                 (wie viel aus 1 € im Zeitverlauf geworden wäre).
      roll_max — der bis dahin jeweils höchste erreichte Depotwert.
      dd       — aktueller Abstand zum bisherigen Hoch, in Prozent (≤ 0).
      .min()   — der schlimmste dieser Abstände, z. B. −0.30 = −30 %.
    """
    cum      = (1 + returns).cumprod()
    roll_max = cum.cummax()
    dd       = (cum - roll_max) / roll_max
    return dd.min()


def calmar_ratio(returns: pd.Series, freq: int = 252) -> float:
    """Calmar Ratio = Jahresrendite / maximaler Drawdown.

    Wie die Sharpe Ratio, aber als Risikomaß dient der schlimmste erlittene
    Absturz statt der alltäglichen Schwankung. Höher = besser.
    """
    mdd = abs(max_drawdown(returns))
    return cagr(returns, freq) / mdd if mdd > 0 else 0.0


def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE,
                  freq: int = 252) -> float:
    """Sortino Ratio: wie Sharpe, bestraft aber nur VERLUST-Schwankungen.

    Idee: Schwankung nach oben (unerwartete Gewinne) stört Anleger nicht —
    also wird die Volatilität nur aus den negativen Tagesrenditen berechnet
    ("Downside-Volatilität").
    """
    r        = cagr(returns, freq)
    down     = returns[returns < 0]              # nur die Verlusttage behalten
    down_vol = down.std() * np.sqrt(freq)        # deren Schwankung, annualisiert
    return (r - rf) / down_vol if down_vol > 0 else 0.0


def portfolio_perf(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                   rf: float = RISK_FREE_RATE):
    """Erwartete Rendite, Volatilität und Sharpe eines Portfolios — aus der Theorie.

    Anders als die Funktionen oben schaut diese NICHT auf realisierte
    Renditen, sondern rechnet mit den geschätzten Eingangsgrößen der
    Markowitz-Theorie:
      weights (w) — Anteil jeder Aktie am Depot (z. B. 0.10 = 10 %)
      mu          — erwartete Jahresrenditen der einzelnen Aktien
      cov         — Kovarianzmatrix: Tabelle, die für jedes Aktienpaar angibt,
                    wie stark sich die beiden gemeinsam bewegen (Diagonale =
                    Varianz der einzelnen Aktie).
    Formeln (das "@" ist die Matrixmultiplikation):
      Portfoliorendite  = w·mu          (gewichteter Durchschnitt)
      Portfoliovarianz  = wᵀ·cov·w      (hier stecken die Diversifikations-
                                         effekte: niedrige Kovarianzen
                                         drücken das Gesamtrisiko)
      Volatilität       = Wurzel daraus
    """
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sr  = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sr


def compute_metrics(returns_df: pd.DataFrame,
                    rf: float = RISK_FREE_RATE) -> pd.DataFrame:
    """Vollständige Performance-Analyse aller Portfoliostrategien.

    Nimmt die Tabelle der täglichen Renditen (eine Spalte je Strategie) und
    gibt eine übersichtliche Kennzahlen-Tabelle zurück: eine Zeile je
    Strategie, eine Spalte je Kennzahl. Zwei noch nicht erklärte Kennzahlen:
      - VaR 95 %  ("Value at Risk"): die Verlustschwelle, die nur an den
        schlechtesten 5 % aller Tage unterschritten wurde. Beispiel −1,8 %:
        an 95 % der Tage verlor die Strategie WENIGER als 1,8 %.
      - Hit Rate: Anteil der Tage mit positiver Rendite ("Gewinnquote").
    """
    metrics = {}
    for col in returns_df.columns:               # jede Strategie einzeln bewerten
        r   = returns_df[col].dropna()           # fehlende Werte entfernen
        cum = (1 + r).cumprod()                  # Depotwert-Verlauf (Start = 1)
        metrics[col] = {
            "CAGR (%)"                   : round(cagr(r) * 100, 2),
            "Gesamtrendite (%)"          : round((cum.iloc[-1] - 1) * 100, 2),
            "Annualisierte Vola. (%)"    : round(annualized_vol(r) * 100, 2),
            "Sharpe Ratio"               : round(sharpe_ratio(r, rf), 4),
            "Sortino Ratio"              : round(sortino_ratio(r, rf), 4),
            "Max. Drawdown (%)"          : round(max_drawdown(r) * 100, 2),
            "Calmar Ratio"               : round(calmar_ratio(r), 4),
            "VaR 95 % (tägl., %)"        : round(np.percentile(r, 5) * 100, 2),
            "Hit Rate (%)"               : round((r > 0).mean() * 100, 2),
        }
    # .T "transponiert" die Tabelle (Zeilen und Spalten tauschen), damit die
    # Strategien als Zeilen erscheinen.
    return pd.DataFrame(metrics).T
