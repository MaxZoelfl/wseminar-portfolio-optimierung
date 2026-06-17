"""
Zentrale Konfiguration & Umgebungs-Setup für das Portfolio-Optimierungsprojekt.

Enthält:
  - Matplotlib-Backend-Auswahl (interaktiv mit Agg-Fallback)
  - Logging, Warnungen, Plot-Design
  - optionale Abhängigkeiten mit graceful Fallback (shap, yfinance, animation)
  - alle globalen Konstanten (Tickers, Zeitraum, Kosten, Modellparameter)

Jedes andere Modul des Pakets importiert hieraus via ``from .config import *``.
Dieses Modul hat selbst KEINE projektinternen Abhängigkeiten (Layer 0).
"""

import warnings
import logging

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
# GLOBALE KONFIGURATION
# ---------------------------------------------------------------------------

TICKERS = [
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
]

SPY_TICKER        = "SPY"
START_DATE        = "2013-01-01"   # Extra-Warmup für Indikatoren
END_DATE          = "2024-12-31"
BACKTEST_START    = "2015-01-01"
RISK_FREE_RATE    = 0.04           # Annualisiert (~US-10J 2024)
TRAIN_YEARS       = 3
N_FRONTIER        = 120
OUTPUT_DIR        = "./output1.6"

RF_N_ITER         = 30             # RandomizedSearchCV Iterationen
RF_CV_SPLITS      = 5              # TimeSeriesSplit Folds

MAX_WEIGHT        = 0.20           # Positionsobergrenze je Asset
TRANSACTION_COST  = 0.0010         # 0.10% auf Handelsumsatz
RF_TURNOVER_LIMIT = 0.30           # Max. 30% einseitiger Turnover/Monat (RF)

# Cross-sectional Ranking: welche Features werden gerankt (v4)
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
