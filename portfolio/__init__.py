"""W-Seminar Portfolio-Optimierung — modularisiertes Paket (v4.1).

FÜR EINSTEIGER: Eine Datei namens __init__.py macht einen Ordner für Python
zu einem "Paket" — erst dadurch funktionieren Importe wie
``from portfolio.metrics import sharpe_ratio``. Inhaltlich passiert hier
nichts weiter; nur die Versionsnummer des Pakets wird hinterlegt.

Wegweiser durch die Module (grob in Ausführungsreihenfolge):
  config.py           — alle Einstellungen ("Schaltzentrale")
  data.py             — Kursdaten laden, Renditen berechnen
  indicators.py       — technische Kennzahlen (Features) für die KI
  metrics.py          — Performance-Kennzahlen (Sharpe, Drawdown, …)
  cross_validation.py — leckagefreie Modellprüfung für Zeitreihen
  optimizers.py       — die drei Strategie-"Rezepte" (MVO, Risk Parity, RF)
  significance.py     — statistische Tests: echt oder Zufall?
  backtest.py         — die Zeitmaschinen-Simulation 2015–2024
  dashboard.py        — Live-Anzeigetafel während des Laufs
  plots.py            — Abbildungen und Daten-Export
  run.py              — der Dirigent: ruft alles der Reihe nach auf
  __main__.py         — Einstiegspunkt für ``python -m portfolio``
"""
__version__ = "4.2.0"
