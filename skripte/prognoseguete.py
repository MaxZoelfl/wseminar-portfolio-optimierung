"""
Prognosegüte des Random Forests (§ 6.5 der Arbeit).

FÜR EINSTEIGER — WAS MACHT DIESES SKRIPT?
Der Backtest verwendet die Prognosen des Waldes, um Portfolios zu bilden, speichert
sie aber nicht. Dieses Skript vergleicht die 119 Monatsprognosen je Titel mit den
Renditen, die im Folgemonat tatsächlich eingetreten sind:

  1. R² und RMSE gegenüber der Nullprognose, wie in § 3.1 (Maß von Gu, Kelly und Xiu),
  2. dasselbe für das historische Mittel, das der Wald in § 4.2 ersetzt,
  3. Wald gegen historisches Mittel: t-Wert der monatlichen Differenz der
     Fehlerquadrate und die Zahl der Monate, in denen der Wald besser ist,
  4. Rangkorrelation im Querschnitt: Trifft die Prognose die Reihenfolge der Titel?

Das 95-%-Intervall des R² stammt aus einem Bootstrap über die 119 Monate.

Vorsicht beim Lesen: In einem steigenden Markt schneidet gegen die Nullprognose
jede positive Prognose gut ab. Deshalb gibt das Skript auch die konstante Prognose
aus. Das historische Mittel profitiert zudem davon, dass die Titel nach Ablauf des
Zeitraums ausgewählt wurden (§ 4.1).

Die Prognosen liegen in skripte/daten/rf_prognosen.csv. Sie stammen aus einem Neulauf
des Backtests vom 05.11.2026 mit den Einstellungen der Arbeit. Er ist mit dem
maßgeblichen Lauf nicht bitgleich: Die Sharpe-Quotienten stimmen auf drei Stellen,
die Gewichte des Waldes weichen um höchstens 0,17 Prozentpunkte ab.

AUFRUF (virtuelle Umgebung aktiviert, aus einem beliebigen Ordner):
    python skripte/prognoseguete.py          # liest die Prognosen, wenige Sekunden
    python skripte/prognoseguete.py --neu    # erzeugt sie neu, rund 50 Minuten
"""

import json
import os
import sys
import tempfile
from pathlib import Path

NEU = "--neu" in sys.argv
if NEU:
    # Der Neulauf schreibt in einen Zwischenordner, ./output bleibt unberührt.
    # Die Einstellung muss stehen, bevor das Paket portfolio geladen wird.
    _ZWISCHEN = Path(tempfile.mkdtemp(prefix="prognoseguete_"))
    _KURSE = Path(__file__).resolve().parents[1] / "data" / "prices.pkl"
    (_ZWISCHEN / "config.json").write_text(json.dumps({
        "output_dir": str(_ZWISCHEN / "output"),
        "price_cache": str(_KURSE),
        "dashboard_update_every": 100000,
    }))
    os.environ["PORTFOLIO_CONFIG"] = str(_ZWISCHEN / "config.json")
    os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _werkzeuge import *                              # noqa: E402,F401,F403

from scipy.stats import spearmanr                     # noqa: E402

DATEI = Path(__file__).resolve().parent / "daten" / "rf_prognosen.csv"
KOPF = ("# Monatsprognosen des Random Forest je Termin und Titel, als einfache Monatsrendite.\n"
        "# Quelle: Neulauf des Backtests mit den Einstellungen der Arbeit, mitgeschrieben\n"
        "# mit prognoseguete.py --neu. Nicht bitgleich mit dem maßgeblichen Lauf, siehe dort.\n")
ZIEHUNGEN = 20000


def prognosen() -> pd.DataFrame:
    """Die gespeicherten Monatsprognosen: 119 Termine × 15 Titel."""
    return pd.read_csv(DATEI, comment="#", index_col=0, parse_dates=True)[TICKERS]


def folgemonat(termine: pd.DatetimeIndex) -> pd.DataFrame:
    """Rendite des Folgemonats je Termin, aufgezinst wie target_next_month im Backtest."""
    monat = (1 + titelrenditen(nur_auswertungszeitraum=False)).resample("ME").prod() - 1
    return monat.shift(-1).reindex(termine)


def historisches_mittel(termine: pd.DatetimeIndex) -> pd.DataFrame:
    """Das historische Mittel aus § 4.2 je Termin, umgerechnet auf einen Monat."""
    r = titelrenditen(nur_auswertungszeitraum=False)
    jahre = protokoll()["experiment_meta"]["train_years"]
    zeilen = [r[(r.index >= t - pd.DateOffset(years=jahre)) & (r.index <= t)].mean() * HANDELSTAGE / 12
              for t in termine]
    return pd.DataFrame(zeilen, index=termine)


def bestimmtheit(y: np.ndarray, f: np.ndarray) -> float:
    """R² gegenüber der Nullprognose (Gu/Kelly/Xiu, Gl. 19): Der Nenner ist nicht zentriert."""
    return 1 - np.sum((y - f) ** 2) / np.sum(y ** 2)


def rangkorrelation(y: np.ndarray, f: np.ndarray) -> tuple:
    """Mittlere Rangkorrelation je Monat und ihr t-Wert."""
    k = np.array([spearmanr(f[t], y[t]).statistic for t in range(len(y))])
    return k.mean(), k.mean() / (k.std(ddof=1) / np.sqrt(len(k)))


def kennzahlen() -> dict:
    """Alle Größen der Auswertung; R² und RMSE als Anteile, nicht in Prozent."""
    p = prognosen()
    y, f = folgemonat(p.index)[TICKERS].values, p.values
    h = historisches_mittel(p.index)[TICKERS].values
    rmse = lambda a: float(np.sqrt(np.mean(a ** 2)))
    zaehler, nenner = np.sum((y - f) ** 2, axis=1), np.sum(y ** 2, axis=1)
    zufall = np.random.default_rng(42)
    ziehungen = [1 - zaehler[i].sum() / nenner[i].sum()
                 for i in zufall.integers(0, len(y), (ZIEHUNGEN, len(y)))]
    differenz = np.sum((y - h) ** 2, axis=1) - zaehler      # positiv: Wald besser
    ohne = [i for i, t in enumerate(TICKERS) if t != "NVDA"]
    rang_wald, rang_wald_t = rangkorrelation(y, f)
    rang_mittel, rang_mittel_t = rangkorrelation(y, h)
    return {
        "monate": len(y),
        "rmse_null": rmse(y), "rmse_wald": rmse(y - f), "rmse_mittel": rmse(y - h),
        "r2_wald": bestimmtheit(y, f), "r2_mittel": bestimmtheit(y, h),
        "r2_konstant": bestimmtheit(y, np.full_like(y, y.mean())),
        "r2_unten": float(np.percentile(ziehungen, 2.5)),
        "r2_oben": float(np.percentile(ziehungen, 97.5)),
        "r2_ohne_nvidia": bestimmtheit(y[:, ohne], f[:, ohne]),
        "t_differenz": differenz.mean() / (differenz.std(ddof=1) / np.sqrt(len(y))),
        "monate_besser": int((differenz > 0).sum()),
        "rang_wald": rang_wald, "rang_wald_t": rang_wald_t,
        "rang_mittel": rang_mittel, "rang_mittel_t": rang_mittel_t,
    }


def neu_erzeugen() -> None:
    """Wiederholt den Backtest, schreibt die Prognosen mit und speichert sie."""
    from portfolio import optimizers
    from portfolio.run import main
    original = optimizers.RFPortfolioOptimizer.predict_monthly_returns
    mitschrift = []

    def mitschreiben(self, x):
        mu = original(self, x)
        mitschrift.append(np.asarray(mu) / 12)       # der Backtest rechnet auf ein Jahr hoch
        return mu

    optimizers.RFPortfolioOptimizer.predict_monthly_returns = mitschreiben
    main()
    termine = gewichte()["Random Forest"].index
    if len(mitschrift) != len(termine):
        sys.exit(f"{len(mitschrift)} Prognosen für {len(termine)} Termine, Datei bleibt unverändert")
    tabelle = pd.DataFrame(mitschrift, index=termine, columns=TICKERS)
    with open(DATEI, "w", encoding="utf-8") as datei:
        datei.write(KOPF)
        tabelle.to_csv(datei, float_format="%.8f", date_format="%Y-%m-%d", index_label="termin")
    print(f"{len(tabelle)} Termine gespeichert in {DATEI}")


if __name__ == "__main__":
    if NEU:
        neu_erzeugen()
    k = kennzahlen()
    print(f"Prognosegüte über {k['monate']} Monate und {len(TICKERS)} Titel\n")
    print(f"{'':26}{'R²':>9}{'RMSE':>9}")
    print(f"{'Nullprognose':26}{'0,00 %':>9}{de(k['rmse_null'] * 100, 2) + ' %':>9}")
    print(f"{'Random Forest':26}{de(k['r2_wald'] * 100, 2) + ' %':>9}{de(k['rmse_wald'] * 100, 2) + ' %':>9}")
    print(f"{'Historisches Mittel':26}{de(k['r2_mittel'] * 100, 2) + ' %':>9}{de(k['rmse_mittel'] * 100, 2) + ' %':>9}")
    print(f"{'Konstante im Nachhinein':26}{de(k['r2_konstant'] * 100, 2) + ' %':>9}")
    print(f"\n95-%-Intervall des R²:     {de(k['r2_unten'] * 100, 1)} bis {de(k['r2_oben'] * 100, 1)} %")
    print(f"R² ohne NVIDIA:            {de(k['r2_ohne_nvidia'] * 100, 2)} %")
    print(f"Wald besser als das Mittel: {k['monate_besser']} von {k['monate']} Monaten, "
          f"t = {de(k['t_differenz'], 2)}")
    print(f"Rangkorrelation Wald:      {de(k['rang_wald'], 2)} (t = {de(k['rang_wald_t'], 2)})")
    print(f"Rangkorrelation Mittel:    {de(k['rang_mittel'], 2)} (t = {de(k['rang_mittel_t'], 2)})")
