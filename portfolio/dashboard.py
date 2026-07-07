"""
Live-Training-Dashboard (Echtzeit-Visualisierung während des Backtests).

FÜR EINSTEIGER — WAS MACHT DIESE DATEI?
Der komplette Backtest läuft je nach Rechner 30–90 Minuten. Damit man dabei
zuschauen kann (und früh merkt, wenn etwas schiefläuft), öffnet diese Datei
ein Fenster mit vier live mitwachsenden Diagrammen — wie eine Anzeigetafel
beim Sportereignis. Auf die RECHENERGEBNISSE hat das Dashboard keinerlei
Einfluss; es schaut nur zu. Läuft das Programm auf einem Rechner ohne
Bildschirm ("headless", z. B. ein Server), schaltet es sich selbst ab.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick        # Achsenformatierung (z. B. %-Zeichen)
import matplotlib.gridspec as gridspec   # flexibles Raster für mehrere Teilbilder
from matplotlib.colors import Normalize as MplNorm
import seaborn as sns                    # Zusatzbibliothek für schönere Statistik-Plots
from .config import *
from .metrics import *

class LiveDashboard:
    """
    Echtzeit-Visualisierung während des Backtests.

    Zeigt 4 Panels (Teilfenster):
      1. Kumulierte Renditen aller Strategien (wird mit jedem Step erweitert)
      2. Aktuelle MVO-Gewichtung (Balkendiagramm)
      3. Rollierender Sharpe Ratio (gleitend, letztes verfügbares Fenster)
      4. Fortschritt & aktuelle Kennzahlen (Texttafel)

    Hinweis: Funktioniert nur mit interaktivem Matplotlib-Backend.
    In headless-Umgebungen (z.B. Server ohne Display) wird das Dashboard
    automatisch deaktiviert ohne den Backtest zu unterbrechen.
    """

    def __init__(self, tickers: list, n_steps: int):
        # Grunddaten merken; die eigentliche Fenster-Einrichtung macht _setup().
        self.tickers  = tickers
        self.n_steps  = n_steps          # Gesamtzahl der Backtest-Schritte (für Fortschritt/ETA)
        self.fig      = None             # das Matplotlib-Fenster (Figure)
        self.axes     = {}               # die vier Teildiagramme, per Name auffindbar
        self.active   = False            # Ist das Dashboard einsatzbereit?
        self._init_time = datetime.now() # Startzeit (für Laufzeit- und ETA-Anzeige)
        self._setup()

    def _setup(self):
        """Baut das Fenster mit dem 4-Panel-Raster auf (einmalig beim Start)."""
        if not INTERACTIVE_DISPLAY:
            # Kein Bildschirm verfügbar → Dashboard bleibt aus, Backtest läuft trotzdem.
            log.info("Live-Dashboard: Kein interaktives Display verfügbar "
                     "(headless), überspringe.")
            return
        try:
            plt.ion()   # "interactive on": Fenster aktualisiert sich, ohne das Programm anzuhalten
            self.fig = plt.figure(figsize=(18, 10))
            self.fig.patch.set_facecolor("white")
            self.fig.suptitle(
                "Portfolio-Optimierung v4 | Live-Training-Dashboard",
                fontsize=14, fontweight="bold", y=0.98
            )
            # GridSpec teilt das Fenster in ein 2×3-Raster; die Panels dürfen
            # mehrere Zellen belegen (gs[0, :2] = Zeile 0, Spalten 0–1).
            gs = gridspec.GridSpec(
                2, 3, figure=self.fig,
                hspace=0.40, wspace=0.35,
                left=0.06, right=0.97, top=0.93, bottom=0.07,
            )
            self.axes["returns"]  = self.fig.add_subplot(gs[0, :2])   # oben links (breit)
            self.axes["weights"]  = self.fig.add_subplot(gs[0, 2])    # oben rechts
            self.axes["sharpe"]   = self.fig.add_subplot(gs[1, :2])   # unten links (breit)
            self.axes["progress"] = self.fig.add_subplot(gs[1, 2])    # unten rechts

            # Initiale Beschriftungen
            self.axes["returns"].set_title("Kumulierte Renditen", fontsize=11)
            self.axes["weights"].set_title("MVO-Gewichte (aktuell)", fontsize=11)
            self.axes["sharpe"].set_title("Rollierender Sharpe Ratio", fontsize=11)
            self.axes["progress"].axis("off")   # Texttafel braucht keine Achsen

            plt.pause(0.05)   # kurzer Moment, damit das Fenster gezeichnet wird
            self.active = True
            log.info("Live-Dashboard aktiviert.")
        except Exception as e:
            log.warning(f"Live-Dashboard Initialisierungsfehler: {e}")
            self.active = False

    def update(self, step: int, month, returns_so_far: pd.DataFrame,
               w_mvo: np.ndarray, w_rf: np.ndarray):
        """Aktualisiert alle 4 Dashboard-Panels nach jedem Rebalancing-Schritt.

        Bekommt vom Backtest den aktuellen Schrittzähler, das Datum, alle
        bisher aufgelaufenen Tagesrenditen sowie die frisch berechneten
        Gewichte von Markowitz (w_mvo) und Random Forest (w_rf).
        Jedes Panel wird geleert (ax.clear()) und neu gezeichnet.
        """
        if not self.active:
            return
        try:
            elapsed = (datetime.now() - self._init_time).seconds

            # ---- Panel 1: Kumulierte Renditen ----------------------------
            # Depotwert-Verlauf jeder Strategie: Produkt der (1+r)-Faktoren,
            # Start = 1 €. So sieht man live, wer vorne liegt.
            ax = self.axes["returns"]
            ax.clear()
            if not returns_so_far.empty:
                for col in STRATEGIES:
                    if col not in returns_so_far.columns:
                        continue
                    cum = (1 + returns_so_far[col].dropna()).cumprod()
                    ax.plot(cum.index, cum.values, label=col,
                            color=COLORS.get(col, "gray"), linewidth=1.8)
            # Gestrichelte Linie bei 1 € = die Startkapital-Marke.
            ax.axhline(1, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
            ax.set_title("Kumulierte Renditen (laufend)", fontsize=10, pad=6)
            ax.set_ylabel("Wert (Start = 1 €)")
            ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
            ax.grid(True, alpha=0.2)

            # ---- Panel 2: Aktuelle MVO-Gewichte -------------------------
            # Balkenpaare je Aktie: aktuelles Markowitz- vs. RF-Gewicht in %,
            # dazu die rote Linie der 20-%-Obergrenze.
            ax = self.axes["weights"]
            ax.clear()
            n  = len(self.tickers)
            x  = np.arange(n)             # x-Positionen 0..n-1 für die Balken
            ax.bar(x, w_mvo * 100, color=COLORS["Markowitz MVO"],
                   alpha=0.75, edgecolor="white", label="MVO", width=0.4)
            # Zweite Balkenreihe um 0.4 nach rechts versetzt (nebeneinander):
            ax.bar(x + 0.4, w_rf * 100, color=COLORS["Random Forest"],
                   alpha=0.75, edgecolor="white", label="RF", width=0.4)
            ax.axhline(MAX_WEIGHT * 100, color="red", linestyle="--",
                       linewidth=0.9, alpha=0.6, label=f"Max {MAX_WEIGHT*100:.0f}%")
            ax.set_xticks(x + 0.2)
            ax.set_xticklabels(self.tickers, rotation=45, ha="right", fontsize=6)
            ax.set_title("Aktuelle Gewichte: MVO vs. RF (%)", fontsize=10, pad=6)
            ax.set_ylabel("Gewicht (%)")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.2, axis="y")

            # ---- Panel 3: Rollierender Sharpe ----------------------------
            # Sharpe Ratio im gleitenden Fenster (bis zu 252 Tage ≈ 1 Jahr):
            # zeigt, WANN welche Strategie gut/schlecht lief — nicht nur im
            # Gesamtdurchschnitt.
            ax = self.axes["sharpe"]
            ax.clear()
            if not returns_so_far.empty and len(returns_so_far) > 60:
                rf_d   = RISK_FREE_RATE / 252   # Jahreszins auf Tagesbasis
                # Fensterbreite dynamisch: anfangs kürzer, später 1 Jahr.
                window = min(252, max(60, len(returns_so_far) // 3))
                for col in STRATEGIES:
                    if col not in returns_so_far.columns:
                        continue
                    r  = returns_so_far[col].dropna()
                    # (Mittelwert − Zins) / Streuung, annualisiert mit √252:
                    rs = (r.rolling(window).mean() - rf_d) / r.rolling(window).std() * np.sqrt(252)
                    ax.plot(rs.index, rs.values, label=col,
                            color=COLORS.get(col, "gray"), linewidth=1.4, alpha=0.9)
                # Orientierungslinien: 0 (kein Mehrwert) und 1 (sehr gut).
                ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
                ax.axhline(1, color="gray",  linewidth=0.5, linestyle=":",  alpha=0.4)
            ax.set_title(f"Rollierender Sharpe Ratio", fontsize=10, pad=6)
            ax.set_ylabel("Sharpe Ratio")
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.2)

            # ---- Panel 4: Fortschritt & Kennzahlen ----------------------
            # Reine Texttafel: Fortschritt, Laufzeit, Restzeit-Schätzung (ETA)
            # und die bisherigen Kennzahlen jeder Strategie.
            ax = self.axes["progress"]
            ax.clear()
            ax.axis("off")

            progress = step / max(self.n_steps, 1) * 100
            # ETA = bisherige Zeit pro Schritt × verbleibende Schritte:
            eta_s    = elapsed / max(step, 1) * (self.n_steps - step)
            eta_min  = eta_s / 60

            lines = [
                f"Fortschritt: {step}/{self.n_steps}  ({progress:.1f}%)",
                f"Datum:       {str(month)[:10]}",
                f"Laufzeit:    {elapsed//60:.0f}m {elapsed%60:.0f}s",
                f"ETA:         ~{eta_min:.1f} min",
                "",
            ]

            if not returns_so_far.empty:
                lines.append("Kennzahlen (bisher):")
                lines.append("─" * 30)
                for col in STRATEGIES:
                    if col not in returns_so_far.columns:
                        continue
                    r = returns_so_far[col].dropna()
                    if len(r) > 0:
                        sr = sharpe_ratio(r)
                        c  = cagr(r)
                        lines.append(f"{col[:14]:<14}")
                        lines.append(f"  CAGR: {c*100:>6.1f}%  SR: {sr:>5.2f}")

            # Alles als ein Textblock in einer hellblauen Box ausgeben:
            ax.text(0.04, 0.97, "\n".join(lines),
                    transform=ax.transAxes, fontsize=8.5,
                    va="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.5",
                              facecolor="#f0f4ff", alpha=0.9, edgecolor="#c0c8e0"))

            # Neuzeichnen anstoßen und dem Fenster kurz Zeit dafür geben:
            self.fig.canvas.draw_idle()
            plt.pause(0.03)

        except Exception:
            pass   # Dashboard-Fehler dürfen den Backtest nie unterbrechen

    def finalize(self, output_path: str):
        """Speichert den finalen Dashboard-Status als PNG."""
        if not self.active:
            return
        try:
            self.fig.savefig(output_path, dpi=150, bbox_inches="tight")
            log.info(f"  Live-Dashboard gespeichert: {output_path}")
            plt.ioff()   # interaktiven Modus wieder ausschalten
        except Exception as e:
            log.warning(f"Dashboard-Speicherung fehlgeschlagen: {e}")

    def close(self):
        """Schließt das Dashboard-Fenster und gibt die Ressourcen frei."""
        if self.active and self.fig is not None:
            try:
                plt.ioff()
                plt.close(self.fig)
            except Exception:
                pass
