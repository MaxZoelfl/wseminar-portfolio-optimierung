"""
Zentrale Konfiguration & Umgebungs-Setup für das Portfolio-Optimierungsprojekt.

Enthält:
  - Matplotlib-Backend-Auswahl (interaktiv mit Agg-Fallback)
  - Logging, Warnungen, Plot-Design
  - optionale Abhängigkeiten mit graceful Fallback (shap, yfinance, animation)
  - die typisierte Konfiguration (dataclass ``Config``) als Single Source of Truth
  - rückwärtskompatible Modulkonstanten (TICKERS, RISK_FREE_RATE, …)

Konfiguration überschreiben (ohne Code-Änderung):
  - eine Datei ``config.json`` im Arbeitsverzeichnis ablegen, ODER
  - Umgebungsvariable ``PORTFOLIO_CONFIG=/pfad/zu/meiner.json`` setzen.
Beispiel siehe ``config.example.json``. Overrides werden beim Import dieses
Moduls eingelesen und wirken dadurch projektweit (vor allen ``from .config import *``).

Dieses Modul hat selbst KEINE projektinternen Abhängigkeiten (Layer 0).
"""

import os
import json
import warnings
import logging
from dataclasses import dataclass, field, fields, asdict

import matplotlib

# Matplotlib: interaktives Backend für Live-Dashboard versuchen,
# Fallback auf Agg (headless/serverless Umgebungen).
_BACKEND_TESTED = False
INTERACTIVE_DISPLAY = False
try:
    import matplotlib.pyplot as plt
    _test = plt.figure(figsize=(1, 1))
    plt.close(_test)
    INTERACTIVE_DISPLAY = True
    _BACKEND_TESTED = True
except Exception:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INTERACTIVE_DISPLAY = False
    _BACKEND_TESTED = True

# Optionale Abhängigkeiten mit graceful Fallback. Die Namen werden IMMER
# definiert (None bei fehlendem Paket), damit abhängiger Code keine
# NameErrors riskiert und sich allein auf die *_AVAILABLE-Flags stützen kann.
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

try:
    from matplotlib.animation import FuncAnimation, PillowWriter
    ANIMATION_AVAILABLE = True
except ImportError:
    FuncAnimation = None
    PillowWriter = None
    ANIMATION_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("portfolio_v4")


# ---------------------------------------------------------------------------
# TYPISIERTE KONFIGURATION
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Alle einstellbaren Parameter des Backtests an einem Ort."""
    # Universum & Zeitraum
    tickers: list = field(default_factory=lambda: [
        "AAPL",   # Apple            — Technologie
        "MSFT",   # Microsoft        — Technologie
        "NVDA",   # NVIDIA           — Halbleiter
        "JNJ",    # Johnson & Johnson — Gesundheit
        "UNH",    # UnitedHealth     — Gesundheit
        "JPM",    # JPMorgan         — Finanzen
        "GS",     # Goldman Sachs    — Finanzen
        "PG",     # Procter & Gamble — Konsum (nicht-zyklisch)
        "KO",     # Coca-Cola        — Konsum (nicht-zyklisch)
        "XOM",    # Exxon Mobil      — Energie
        "CAT",    # Caterpillar      — Industrie
        "HON",    # Honeywell        — Industrie
        "VZ",     # Verizon          — Telekommunikation
        "PLD",    # Prologis         — Immobilien (REIT)
        "LIN",    # Linde            — Grundstoffe
    ])
    spy_ticker: str       = "SPY"
    start_date: str       = "2013-01-01"   # Extra-Warmup für Indikatoren
    end_date: str         = "2024-12-31"
    backtest_start: str   = "2015-01-01"
    output_dir: str       = "./output1.6"

    # Modell- & Optimierungsparameter
    risk_free_rate: float    = 0.04        # annualisiert (~US-10J 2024)
    train_years: int         = 3
    n_frontier: int          = 120
    rf_n_iter: int           = 30          # RandomizedSearchCV-Iterationen
    rf_cv_splits: int        = 5           # TimeSeriesSplit-Folds
    max_weight: float        = 0.20        # Positionsobergrenze je Asset
    transaction_cost: float  = 0.0010      # 0.10 % auf Handelsumsatz
    rf_turnover_limit: float = 0.30        # max. einseitiger Turnover/Monat (RF)

    # Performance-Hebel (Default 1 = exakt bisheriges Verhalten):
    rf_retune_every: int        = 1  # RF-Hyperparametersuche nur alle k Monate;
                                     # dazwischen nur Refit auf neuem Fenster.
                                     # >1 beschleunigt deutlich, ändert Ergebnisse.
    dashboard_update_every: int = 1  # Live-Dashboard nur alle k Schritte rendern;
                                     # rein kosmetisch (kein Ergebniseinfluss).


def _load_config() -> Config:
    """Erzeugt die Konfiguration aus Defaults + optionalen JSON-Overrides."""
    cfg = Config()
    path = os.environ.get("PORTFOLIO_CONFIG", "config.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            valid = {f.name for f in fields(cfg)}
            applied, ignored = [], []
            for k, v in data.items():
                if k in valid:
                    setattr(cfg, k, v); applied.append(k)
                else:
                    ignored.append(k)
            log.info(f"Konfiguration aus '{path}' geladen — Overrides: {applied or 'keine'}"
                     + (f" | ignoriert: {ignored}" if ignored else ""))
        except Exception as e:
            log.warning(f"Konnte '{path}' nicht laden ({e}); nutze Defaults.")
    return cfg


CFG = _load_config()

# ---------------------------------------------------------------------------
# RÜCKWÄRTSKOMPATIBLE MODULKONSTANTEN (Single Source of Truth = CFG)
# Bestehender Code nutzt weiterhin die Großbuchstaben-Namen unverändert.
# ---------------------------------------------------------------------------
TICKERS           = CFG.tickers
SPY_TICKER        = CFG.spy_ticker
START_DATE        = CFG.start_date
END_DATE          = CFG.end_date
BACKTEST_START    = CFG.backtest_start
RISK_FREE_RATE    = CFG.risk_free_rate
TRAIN_YEARS       = CFG.train_years
N_FRONTIER        = CFG.n_frontier
OUTPUT_DIR        = CFG.output_dir
RF_N_ITER         = CFG.rf_n_iter
RF_CV_SPLITS      = CFG.rf_cv_splits
MAX_WEIGHT        = CFG.max_weight
TRANSACTION_COST  = CFG.transaction_cost
RF_TURNOVER_LIMIT = CFG.rf_turnover_limit
RF_RETUNE_EVERY        = CFG.rf_retune_every
DASHBOARD_UPDATE_EVERY = CFG.dashboard_update_every

# Cross-sectional Ranking: welche Features werden gerankt (strukturell, nicht
# über config.json einstellbar).
RANK_COLS = ["rsi", "mom_21d", "mom_63d", "mom_252d", "alpha_spy"]

# Matplotlib-Design
plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
    "axes.grid"         : True,
    "grid.alpha"        : 0.25,
    "grid.linewidth"    : 0.6,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "font.family"       : "serif",
    "font.size"         : 11,
    "axes.titlesize"    : 13,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 11,
    "legend.fontsize"   : 9,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
})

COLORS = {
    "Markowitz MVO" : "#1f77b4",
    "Random Forest" : "#d62728",
    "Equal Weight"  : "#2ca02c",
    "Risk Parity"   : "#ff7f0e",
    "frontier"      : "#7f7f7f",
    "cml"           : "#9467bd",
    "mvp"           : "#e7ba52",
}

STRATEGIES = ["Markowitz MVO", "Random Forest", "Equal Weight", "Risk Parity"]
