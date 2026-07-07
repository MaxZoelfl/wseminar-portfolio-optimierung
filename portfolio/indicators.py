"""
Technische Indikatoren, monatliche Aggregation und Feature-Spalten.

FÜR EINSTEIGER — WAS MACHT DIESE DATEI?
Der Random Forest (unsere KI-Strategie) kann nicht direkt "auf Kurse schauen".
Er braucht ZAHLEN-MERKMALE, sogenannte "Features": für jede Aktie und jeden
Monat eine Reihe von Kennzahlen, die den aktuellen Zustand der Aktie
beschreiben. Aus diesen Merkmalen versucht das Modell dann, die Rendite des
FOLGEMONATS vorherzusagen.

Diese Datei berechnet solche Merkmale — die klassischen "technischen
Indikatoren" der Chartanalyse:

  - RSI            : "Relative Strength Index" — misst, ob eine Aktie zuletzt
                     ungewöhnlich stark gestiegen (überkauft, RSI nahe 100)
                     oder gefallen ist (überverkauft, RSI nahe 0).
  - MACD           : Differenz zweier gleitender Durchschnitte — zeigt, ob
                     gerade ein Aufwärts- oder Abwärtstrend läuft.
  - Bollinger      : Bänder um den Durchschnittskurs, ±2 Standardabweichungen —
                     zeigen, ob der Kurs relativ zu seiner üblichen Schwankung
                     gerade hoch oder tief steht.
  - Momentum       : schlicht die aufsummierte Rendite der letzten 1/3/6/12
                     Monate — "läuft die Aktie gerade gut?"
  - Volatilität    : Schwankungsstärke der letzten 1 bzw. 3 Monate.
  - Alpha/Beta     : Vergleich mit dem Gesamtmarkt (SPY). Beta ≈ "wie stark
                     bewegt sich die Aktie mit, wenn der Markt sich bewegt?";
                     Alpha ≈ "was verdient sie darüber hinaus aus eigener Kraft?"

Wichtig für die Sauberkeit des Experiments: Alle Merkmale eines Monats t
verwenden ausschließlich Daten BIS t; das Vorhersageziel ist die Rendite von
Monat t+1. So "sieht" das Modell niemals in die Zukunft (kein Look-Ahead-Bias).

Häufig benutzte pandas-Bausteine in dieser Datei:
  - .rolling(n)  : gleitendes Fenster — Berechnung jeweils über die letzten
                   n Tage (z. B. gleitender Durchschnitt).
  - .ewm(...)    : exponentiell gewichtetes Fenster — wie rolling, aber
                   jüngere Tage zählen stärker als ältere.
  - .resample("ME"): fasst tägliche Werte zu Monatswerten zusammen
                   ("ME" = Month End, Monatsende).
  - .shift(-1)   : verschiebt eine Spalte um eine Zeile nach OBEN — so wird
                   der Wert des Folgemonats zum Vorhersageziel des aktuellen.
"""

import numpy as np
import pandas as pd
from .config import *
from .metrics import timer

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI nach Wilder (1978): Überkauft/überverkauft Signal.

    Idee: Vergleiche die durchschnittlichen Kursgewinne der letzten 14 Tage
    mit den durchschnittlichen Kursverlusten. Ergebnis wird auf eine Skala
    von 0 bis 100 gebracht: Werte über ~70 gelten als "überkauft" (Rücksetzer
    wahrscheinlicher), unter ~30 als "überverkauft".
    """
    delta = prices.diff()               # tägliche Kursänderung (heute − gestern)
    gain  = delta.clip(lower=0)         # nur die Gewinne (Verluste → 0)
    loss  = -delta.clip(upper=0)        # nur die Verluste, als positive Zahl
    # Geglättete Durchschnitte nach Wilder (exponentielle Glättung):
    avg_g = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    # Verhältnis Gewinn/Verlust; replace(0, NaN) verhindert Division durch 0.
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))       # Wilders Formel: auf 0–100 skalieren


def compute_macd(prices: pd.Series, fast: int = 12,
                 slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD nach Appel (1979): Trendfolge-Indikator.

    Zwei exponentiell gewichtete Durchschnitte (EMA): ein "schneller" über
    12 Tage und ein "langsamer" über 26 Tage. Liegt der schnelle über dem
    langsamen (MACD > 0), ist der jüngere Trend aufwärts gerichtet.
    Zusätzlich: die "Signallinie" (9-Tage-Glättung des MACD) und das
    "Histogramm" (MACD − Signal) als Maß für die Trendbeschleunigung.
    """
    ema_f  = prices.ewm(span=fast,   adjust=False).mean()   # schneller Durchschnitt
    ema_s  = prices.ewm(span=slow,   adjust=False).mean()   # langsamer Durchschnitt
    macd   = ema_f - ema_s
    sig    = macd.ewm(span=signal,   adjust=False).mean()
    return pd.DataFrame({"macd": macd, "macd_sig": sig, "macd_hist": macd - sig})


def compute_bollinger(prices: pd.Series, period: int = 20,
                      n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bänder nach Bollinger (1983): Volatilitäts-Indikator.

    Um den 20-Tage-Durchschnittskurs (SMA) werden zwei Bänder im Abstand von
    ±2 Standardabweichungen gelegt. Zurückgegeben werden zwei Kennzahlen:
      %B        — wo steht der Kurs zwischen den Bändern? (0 = am unteren,
                  1 = am oberen Band; >1 = ungewöhnlich weit oben)
      Bandwidth — wie weit sind die Bänder gerade auseinander? (Maß dafür,
                  wie unruhig die Aktie aktuell ist)
    Das winzige "+ 1e-10" im Nenner ist ein üblicher Trick, um eine Division
    durch exakt null zu verhindern (es verfälscht das Ergebnis nicht messbar).
    """
    sma   = prices.rolling(period).mean()   # gleitender 20-Tage-Durchschnitt
    std   = prices.rolling(period).std()    # gleitende Standardabweichung
    upper = sma + n_std * std               # oberes Band
    lower = sma - n_std * std               # unteres Band
    pct_b = (prices - lower) / (upper - lower + 1e-10)
    bw    = (upper - lower) / (sma + 1e-10)
    return pd.DataFrame({"bb_pct_b": pct_b, "bb_width": bw})


def compute_momentum(returns: pd.Series,
                     periods: list = [21, 63, 126, 252]) -> pd.DataFrame:
    """Preismomentum über verschiedene Horizonte (Jegadeesh & Titman, 1993).

    "Momentum" = aufsummierte Rendite der letzten n Handelstage. Die Horizonte
    entsprechen ca. 1, 3, 6 und 12 Monaten (21 Handelstage ≈ 1 Monat).
    Klassischer empirischer Befund: Aktien, die zuletzt gut liefen, laufen
    im Schnitt noch eine Weile weiter gut.
    """
    return pd.DataFrame({f"mom_{p}d": returns.rolling(p).sum() for p in periods})


def compute_volatility_features(returns: pd.Series,
                                 periods: list = [21, 63]) -> pd.DataFrame:
    """Rollende historische Volatilität als Risikowarnsignal.

    Standardabweichung der Tagesrenditen über die letzten 21 bzw. 63 Tage,
    mit √252 auf Jahresniveau hochgerechnet (vgl. metrics.annualized_vol).
    """
    return pd.DataFrame({
        f"vol_{p}d": returns.rolling(p).std() * np.sqrt(252) for p in periods
    })


def compute_alpha_beta(asset_returns: pd.Series,
                       market_returns: pd.Series,
                       window: int = 63) -> pd.DataFrame:
    """
    Rollierendes Alpha & Beta vs. S&P 500 (SPY).
    OLS-Regression im rollierenden Fenster: r_i = α + β·r_m + ε

    In Worten: Für jeden Tag wird über die jeweils letzten 63 Handelstage
    (≈ 3 Monate) eine Gerade durch die Punktwolke (Marktrendite, Aktienrendite)
    gelegt. Deren Steigung ist das BETA (Marktabhängigkeit: Beta 1,5 heißt,
    die Aktie bewegt sich im Schnitt 1,5-mal so stark wie der Markt), der
    Achsenabschnitt das ALPHA (Zusatzrendite unabhängig vom Markt; wird mit
    ×252 auf ein Jahr hochgerechnet).
    """
    alphas, betas = [], []
    idx = asset_returns.index

    for i in range(len(idx)):           # für jeden Handelstag i …
        if i < window:
            # Noch keine 63 Vortage vorhanden → Kennzahl nicht berechenbar.
            # NaN ("Not a Number") ist der Standard-Platzhalter für "fehlt".
            alphas.append(np.nan)
            betas.append(np.nan)
            continue
        # Renditefenster der letzten 63 Tage für Aktie (ra) und Markt (rm):
        ra   = asset_returns.iloc[i-window:i].values
        rm   = market_returns.reindex(asset_returns.index[i-window:i]).values
        # Tage aussortieren, an denen einer der beiden Werte fehlt:
        mask = ~(np.isnan(ra) | np.isnan(rm))
        if mask.sum() < window // 2:
            # Weniger als die Hälfte der Tage brauchbar → zu unsicher, auslassen.
            alphas.append(np.nan)
            betas.append(np.nan)
            continue
        ra, rm = ra[mask], rm[mask]
        # Kleinste-Quadrate-Lösung in Kurzform:
        #   Beta  = Kovarianz(Aktie, Markt) / Varianz(Markt)
        #   Alpha = Mittelwert(Aktie) − Beta · Mittelwert(Markt)
        # (+1e-12 verhindert wieder eine Division durch null.)
        beta   = np.cov(ra, rm)[0, 1] / (np.var(rm) + 1e-12)
        alpha  = ra.mean() - beta * rm.mean()
        alphas.append(alpha * 252)      # Alpha auf Jahresbasis hochrechnen
        betas.append(beta)

    return pd.DataFrame({"alpha_spy": alphas, "beta_spy": betas}, index=idx)


def build_all_indicators(daily_prices: pd.DataFrame,
                          daily_returns: pd.DataFrame,
                          spy_returns: pd.Series,
                          tickers: list) -> dict:
    """Berechnet alle technischen Indikatoren für jedes Asset (täglich).

    Sammelfunktion: ruft für jede der 15 Aktien nacheinander alle obigen
    Indikator-Funktionen auf und klebt die Ergebnisse spaltenweise zu EINER
    Tabelle pro Aktie zusammen (pd.concat mit axis=1 = "nebeneinander").
    Rückgabe: ein "dict" (Nachschlage-Verzeichnis) Aktienkürzel → Tabelle.
    """
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

    Warum monatlich? Die Strategien schichten das Depot einmal pro Monat um
    ("Rebalancing") — also braucht das Vorhersagemodell auch einen Datensatz
    pro Aktie und Monat, nicht pro Tag.

    Aggregationslogik (wie wird aus ~21 Tageswerten EIN Monatswert?):
      - Endwert (last):  RSI, MACD, Bollinger, Alpha, Beta
        → Momentaufnahmen: Es zählt der Zustand am Monatsende.
      - Durchschnitt (mean): Volatilität, Momentum
        → Niveaugrößen: Der Monatsmittelwert ist stabiler als ein Stichtag.

    Zusätzlich entstehen hier zwei zentrale Spalten:
      monthly_ret       — die realisierte Rendite des Monats selbst,
      target_next_month — die Rendite des FOLGEmonats = das, was der
                          Random Forest vorhersagen soll (das "Label").
                          Durch shift(-1) rutscht sie eine Zeile nach oben;
                          der letzte Monat hat kein "nächstes" → NaN.
    """
    rows = []
    for ticker in tickers:
        ind_df     = indicator_dict[ticker].copy()
        ret_col    = daily_returns[ticker]
        # Monatsrendite korrekt aufzinsen: einfache Renditen sind über die
        # ZEIT multiplikativ ( prod(1+r) - 1 ), nicht additiv.
        # Beispiel: +10 % und −10 % ergeben nicht 0 %, sondern −1 %.
        monthly_r  = (1 + ret_col).resample("ME").prod() - 1

        feat = pd.DataFrame(index=monthly_r.index)
        # Momentaufnahme-Features: letzter Tageswert des Monats.
        for col in ["rsi", "macd", "macd_sig", "macd_hist",
                    "bb_pct_b", "bb_width", "alpha_spy", "beta_spy"]:
            if col in ind_df.columns:
                feat[col] = ind_df[col].resample("ME").last()
        # Niveau-Features: Monatsdurchschnitt.
        for col in ["mom_21d", "mom_63d", "mom_126d", "mom_252d",
                    "vol_21d", "vol_63d"]:
            if col in ind_df.columns:
                feat[col] = ind_df[col].resample("ME").mean()

        feat["ticker"]            = ticker
        feat["monthly_ret"]       = monthly_r
        feat["target_next_month"] = monthly_r.shift(-1)   # Ziel = Folgemonat
        rows.append(feat)

    # Alle Aktien untereinander stapeln → "Panel": je Monat 15 Zeilen.
    combined = pd.concat(rows).sort_index()
    # Unendlich-Werte (aus seltenen 0-Divisionen) als "fehlt" markieren.
    combined = combined.replace([np.inf, -np.inf], np.nan)
    return combined


def add_cross_sectional_ranks(monthly_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional Ranking Features (v4).
    Für jeden Monat t: jedes Asset erhält einen Perzentil-Rang (0–1)
    relativ zu ALLEN anderen Assets in diesem Monat.

    In Worten: Statt "Apple hat Momentum +8 %" lernt das Modell zusätzlich
    "Apple hat diesen Monat das drittbeste Momentum der 15 Aktien" (als Zahl
    zwischen 0 = schlechteste und 1 = beste). Solche RELATIVEN Ränge sind
    über die Jahre stabiler vergleichbar als absolute Werte, denn "+8 %" kann
    je nach Marktphase viel oder wenig sein.

    Wissenschaftliche Begründung (Jegadeesh & Titman, 1993):
      Das relative Momentum (Cross-sectional Momentum) erklärt Aktienrenditen
      besser als absolutes Momentum. Gewinner-Assets tendieren zur
      Outperformance gegenüber Verlierern über 3–12 Monate.
    """
    result = monthly_data.copy()
    for col in RANK_COLS:                      # RANK_COLS ist in config.py definiert
        if col not in monthly_data.columns:
            continue
        # groupby(level=0) = "gruppiere nach Datum": Der Rang wird also je
        # Monat NUR unter den Aktien dieses Monats vergeben. rank(pct=True)
        # liefert Perzentile 0–1 statt Plätzen 1–15.
        result[f"{col}_rank"] = (
            monthly_data.groupby(level=0)[col]
            .rank(pct=True, na_option="keep")
        )
    return result


# ---------------------------------------------------------------------------
# FESTE SPALTENLISTEN
# Diese Listen legen verbindlich fest, welche Spalten der Monatstabelle als
# Eingabe-Merkmale ("Features") für den Random Forest dienen.
# ---------------------------------------------------------------------------
FEATURE_COLS_BASE = [
    "rsi", "macd", "macd_sig", "macd_hist",
    "bb_pct_b", "bb_width",
    "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_21d", "vol_63d",
    "alpha_spy", "beta_spy",
    "monthly_ret",
]


# Gesamtliste = Basis-Features + die daraus abgeleiteten Rang-Features.
FEATURE_COLS = FEATURE_COLS_BASE + [f"{c}_rank" for c in RANK_COLS]


# Übersetzungstabelle: interner Spaltenname → lesbare Beschriftung in den
# Abbildungen ("CS" = cross-sectional, also der Quervergleich unter Aktien).
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
