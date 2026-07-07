"""
Portfolio-Optimierer: Markowitz (Ledoit-Wolf), Risk Parity, Random Forest.

FÜR EINSTEIGER — WAS MACHT DIESE DATEI?
Hier stecken die eigentlichen "Rezepte" der Anlagestrategien: drei Klassen,
die aus den Daten jeweils die PORTFOLIOGEWICHTE berechnen — also die Antwort
auf die Frage "Wie viel Prozent des Geldes stecke ich in welche Aktie?".

  1. MarkowitzLedoitWolf — die klassische Mean-Variance-Optimierung (MVO):
     sucht rechnerisch die Gewichte mit der höchsten Sharpe Ratio
     (bestes Verhältnis von erwartetem Mehrertrag zu Risiko).
  2. RiskParityPortfolio — verteilt nicht das GELD gleichmäßig, sondern das
     RISIKO: Jede Aktie soll gleich viel zur Gesamtschwankung beitragen.
  3. RFPortfolioOptimizer — die KI-Strategie: Ein Random Forest sagt die
     Renditen des nächsten Monats voraus; mit diesen Prognosen (statt der
     historischen Durchschnitte) wird dann wieder Markowitz-optimiert.

Zwei Grundbegriffe, die überall auftauchen:
  - GEWICHTSVEKTOR w: eine Liste von Anteilen, z. B. [0.10, 0.05, …] =
    10 % in Aktie 1, 5 % in Aktie 2, … Die Anteile summieren sich zu 1 (100 %).
  - KOVARIANZMATRIX Σ ("cov"): quadratische Tabelle (15×15), die für jedes
    Aktienpaar angibt, wie stark sich die beiden gemeinsam bewegen. Sie ist
    das Herz der Diversifikation: Aktien, die NICHT im Gleichschritt laufen,
    senken zusammen das Portfoliorisiko.

Was heißt hier "Optimierung"? Wir geben dem Computer eine Zielfunktion
("mache diese Zahl möglichst klein") plus Nebenbedingungen ("Summe der
Gewichte = 1", "kein Gewicht über 20 %") — und ein fertiger Such-Algorithmus
(scipy.optimize.minimize mit der Methode "SLSQP") tastet sich schrittweise
zur besten zulässigen Lösung vor. Da minimize nur MINIMIEREN kann, wird zum
Maximieren einfach das NEGATIV der Zielgröße minimiert (Minus-Trick).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize                  # der numerische Löser
from scipy.stats import randint, uniform             # Zufallsbereiche fürs RF-Tuning
from sklearn.covariance import LedoitWolf            # verbesserte Kovarianzschätzung
from sklearn.ensemble import RandomForestRegressor   # das KI-Modell
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler     # Merkmale auf gleiche Skala bringen
from sklearn.pipeline import Pipeline                # kettet Verarbeitungsschritte
from .config import *
from .metrics import timer
from .cross_validation import purged_kfold_splits

class MarkowitzLedoitWolf:
    """
    Klassische Mean-Variance Optimization mit Ledoit-Wolf Kovarianzschätzung.
    Referenz: Ledoit & Wolf (2004). Journal of Multivariate Analysis.

    Warum "Ledoit-Wolf" statt der gewöhnlichen Stichproben-Kovarianz?
    Mit nur ~3 Jahren Daten für 15 Aktien ist die roh geschätzte
    Kovarianzmatrix sehr verrauscht — und Markowitz reagiert bekanntermaßen
    extrem empfindlich auf solche Schätzfehler ("error maximizer").
    Ledoit-Wolf "schrumpft" die rohe Schätzung in Richtung einer einfachen,
    stabilen Zielmatrix (Mittelweg aus beiden). Das dämpft Ausreißer und
    liefert verlässlichere Gewichte.
    """

    def __init__(self, rf: float = RISK_FREE_RATE):
        # __init__ ist der "Konstruktor": Er läuft einmal beim Erzeugen des
        # Objekts und speichert hier nur den risikofreien Zins (self.rf).
        self.rf = rf

    def estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Schätzt die Kovarianzmatrix aus Tagesrenditen (Ledoit-Wolf).

        "* 252" rechnet die Tages-Kovarianzen auf Jahresbasis hoch, damit sie
        zur annualisierten Rendite und zum Jahreszins passen.
        """
        lw = LedoitWolf()
        lw.fit(returns.values)          # .fit() = "lerne/schätze aus den Daten"
        return lw.covariance_ * 252

    def _neg_sharpe(self, weights, mu, cov):
        """Hilfsfunktion für den Optimierer: die NEGATIVE Sharpe Ratio.

        (Der führende Unterstrich im Namen ist eine Python-Konvention für
        "nur für den internen Gebrauch dieser Klasse gedacht".)
        minimize() kann nur minimieren — das Minimum von −Sharpe ist aber
        genau das Maximum von +Sharpe (Minus-Trick, s. Datei-Kopf).
        """
        ret = float(weights @ mu)                        # erwartete Portfoliorendite
        vol = float(np.sqrt(weights @ cov @ weights))    # Portfoliovolatilität
        return -(ret - self.rf) / vol if vol > 0 else 0.0

    def max_sharpe(self, mu: np.ndarray, cov: np.ndarray,
                   w_prev: np.ndarray = None,
                   turnover_limit: float = None) -> np.ndarray:
        """
        Maximiert Sharpe Ratio (Tangential-Portfolio).
        Long-Only, vollständig investiert. Positionsobergrenze: MAX_WEIGHT.
        Optional: Turnover-Constraint nach Garleanu & Pedersen (2013).

        Begriffe:
          - "Tangentialportfolio": der Punkt auf der Effizienzlinie mit der
            steilsten Verbindungsgeraden zum risikofreien Zins — genau das
            Portfolio mit maximaler Sharpe Ratio.
          - "Long-Only": keine Leerverkäufe, alle Gewichte ≥ 0.
          - "Turnover-Constraint": Wenn Vorgängergewichte (w_prev) und ein
            Limit übergeben werden, darf die neue Lösung nur begrenzt vom
            alten Depot abweichen — das begrenzt Handelskosten.
        """
        n  = len(mu)
        # Startpunkt der Suche: die alten Gewichte (falls vorhanden), sonst 1/N.
        w0 = w_prev if w_prev is not None else np.ones(n) / n
        # Nebenbedingung "eq" (equality): Ausdruck muss exakt 0 sein
        # → w.sum() − 1 = 0 bedeutet "Gewichte summieren sich zu 100 %".
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        # "bounds": erlaubter Bereich je Gewicht → zwischen 0 % und 20 %.
        bounds      = [(0.0, MAX_WEIGHT)] * n

        if w_prev is not None and turnover_limit is not None:
            # Nebenbedingung "ineq" (inequality): Ausdruck muss ≥ 0 bleiben.
            # Turnover = 0.5·Σ|w_neu − w_alt| (halbe Summe, weil jeder Verkauf
            # zugleich ein Kauf ist). Bedingung: Limit − Turnover ≥ 0.
            constraints.append({
                "type": "ineq",
                "fun": lambda w, wp=w_prev, lim=turnover_limit:
                       lim - np.abs(w - wp).sum() / 2,
            })

        # Der eigentliche Optimierungslauf: SLSQP tastet sich von w0 aus zur
        # besten zulässigen Lösung vor (maxiter/ftol = Abbruchkriterien).
        result = minimize(
            fun=self._neg_sharpe, x0=w0, args=(mu, cov),
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        # Numerische Hygiene: Winzige negative Gewichte (Rechenungenauigkeit,
        # z. B. −1e-15) auf 0 setzen und exakt auf Summe 1 normieren.
        w = np.maximum(result.x, 0.0)
        w /= w.sum()
        return w

    def efficient_frontier(self, mu: np.ndarray, cov: np.ndarray,
                            n_points: int = N_FRONTIER,
                            max_weight: float = None) -> pd.DataFrame:
        """
        Berechnet N Punkte auf der Effizienzlinie (Long-Only).

        Die "Effizienzlinie" (Efficient Frontier) ist die Kurve aller
        bestmöglichen Portfolios: Für jede Zielrendite das Portfolio mit dem
        KLEINSTEN Risiko. Vorgehen: Zielrenditen zwischen Minimum und Maximum
        abschreiten und für jede das varianzminimale Portfolio suchen.
        Dient in dieser Arbeit nur der VISUALISIERUNG (Abbildungen 4, 8, 11).

        ``max_weight`` setzt – wie in der eigentlichen Optimierung
        (``max_sharpe``) – eine Positionsobergrenze je Titel. Ist sie gesetzt,
        liegt das Markowitz-Tangentialportfolio auf der gezeichneten Kurve, und
        die Kurve endet bei der unter dieser Obergrenze erreichbaren
        Maximalrendite statt bei einer 100-%-Einzelwette. ``None`` = keine
        Obergrenze (0 ≤ w ≤ 1), wie bisher.
        """
        n      = len(mu)
        upper  = 1.0 if max_weight is None else float(max_weight)
        bounds = [(0.0, upper)] * n

        # Schritt 1: das Portfolio mit der global KLEINSTEN Varianz finden
        # (Minimum-Variance-Portfolio, MVP) → linkes Ende der Kurve.
        res_mvp = minimize(
            lambda w: float(w @ cov @ w), np.ones(n) / n,
            method="SLSQP", bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        )
        ret_min = float(res_mvp.x @ mu)     # Rendite des MVP = kleinste Zielrendite

        # Schritt 2: die größte erreichbare Rendite bestimmen → rechtes Ende.
        if max_weight is None:
            ret_max = float(mu.max())       # ohne Cap: alles in die beste Aktie
        else:
            # maximal erreichbare Rendite unter der Obergrenze: Cap greedy auf
            # die renditestärksten Titel legen, bis das Budget (Σw = 1) erschöpft
            # ist (z. B. bei Cap 0,20 die fünf besten Titel zu je 20 %).
            w_hi, remaining = np.zeros(n), 1.0
            for idx in np.argsort(mu)[::-1]:      # Aktien absteigend nach Rendite
                take       = min(upper, remaining)
                w_hi[idx]  = take
                remaining -= take
                if remaining <= 1e-12:
                    break
            ret_max = float(w_hi @ mu)

        # Schritt 3: Zielrenditen gleichmäßig abschreiten; je Ziel das
        # varianzminimale Portfolio suchen und als Kurvenpunkt speichern.
        frontier = []
        for target in np.linspace(ret_min, ret_max, n_points):
            res = minimize(
                lambda w: float(w @ cov @ w), np.ones(n) / n,
                method="SLSQP", bounds=bounds,
                constraints=[
                    {"type": "eq", "fun": lambda w: w.sum() - 1},
                    # Zusatzbedingung: erwartete Rendite = aktuelle Zielrendite.
                    {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
                ],
                options={"maxiter": 500, "ftol": 1e-12},
            )
            if res.success:      # nur konvergierte Lösungen in die Kurve aufnehmen
                vol = float(np.sqrt(res.x @ cov @ res.x))
                sr  = (target - self.rf) / vol if vol > 0 else 0.0
                frontier.append({"ret": target, "vol": vol, "sr": sr, "w": res.x.copy()})

        return pd.DataFrame(frontier)


class RiskParityPortfolio:
    """
    Equal Risk Contribution (Risk Parity) Portfolio.

    Jedes Asset trägt gleich viel zum gesamten Portfoliorisiko bei.
    Formell: RC_i = w_i * (Sigma w)_i / sqrt(w' Sigma w) = 1/N für alle i

    Anschaulich: Bei Equal Weight bekommt jede Aktie gleich viel GELD — aber
    eine schwankungsstarke Aktie wie NVIDIA dominiert dann trotzdem das
    RISIKO des Depots. Risk Parity dreht das um: Wackelige Aktien bekommen
    weniger, ruhige mehr Gewicht, bis jede denselben "Risikobeitrag" (RC)
    leistet. Der Risikobeitrag misst, welchen Anteil der Gesamtschwankung
    eine Position verursacht (inklusive ihrer Kovarianzen zu den anderen).

    Vorteile gegenüber Equal Weight:
      - Berücksichtigt Asset-Volatilität und Korrelationen
      - Weniger Konzentration in risikoreichen Assets
      - Empirisch bessere risikobereinigte Renditen (Qian, 2005)

    Referenz: Qian, E. (2005). Risk Parity Portfolios: Efficient Portfolios
    through True Diversification. PanAgora Asset Management.
    """

    def optimize(self, cov: np.ndarray, max_weight: float = MAX_WEIGHT) -> np.ndarray:
        """Berechnet die Risk-Parity-Gewichte für eine gegebene Kovarianzmatrix.

        Beachte: Es wird NUR die Kovarianzmatrix gebraucht — keine Rendite-
        prognose. Das macht Risk Parity robust gegen die notorisch unsicheren
        Renditeschätzungen.
        """
        n = cov.shape[0]

        def _objective(w):
            """Minimiert die Streuung der Risikobeiträge.

            Perfekte Risk Parity = alle Risikobeiträge sind exakt gleich groß.
            Als Zielfunktion nehmen wir die Summe der quadrierten Abweichungen
            der Beiträge von ihrem Sollwert (Gesamtrisiko / n). Ist sie 0,
            trägt jede Aktie exakt gleich viel Risiko.
            """
            portfolio_vol = np.sqrt(max(w @ cov @ w, 1e-10))  # max(...) schützt vor √0
            rc = w * (cov @ w) / portfolio_vol   # Risikobeitrag jeder Aktie
            target_rc = portfolio_vol / n        # Soll: jeder trägt 1/n des Risikos
            return float(np.sum((rc - target_rc) ** 2))

        w0 = np.ones(n) / n                      # Start: Gleichgewichtung
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]   # voll investiert
        # Untergrenze 0.5 % statt 0: Jede Aktie bleibt minimal vertreten; das
        # verhindert auch, dass der Löser in der Ecke w=0 stecken bleibt.
        bounds = [(0.005, max_weight)] * n

        result = minimize(
            fun=_objective, x0=w0,
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 3000, "ftol": 1e-14},
        )

        # Numerische Hygiene wie bei Markowitz: keine Negativ-Reste, Summe = 1.
        w = np.maximum(result.x, 0.0)
        w /= w.sum()
        return w


class RFPortfolioOptimizer:
    """
    Random-Forest-gestützte Portfoliooptimierung mit monatlichem Rebalancing.
    Referenz: Fischer & Krauss (2018). European Journal of Operational Research.

    Was ist ein Random Forest? Ein KI-Modell aus dem maschinellen Lernen:
    ein "Wald" aus vielen (hier 100–500) Entscheidungsbäumen. Jeder Baum
    lernt einfache Wenn-dann-Regeln aus den Merkmalen (z. B. "WENN Momentum
    hoch UND Volatilität niedrig, DANN Rendite eher positiv"), und zwar auf
    einer jeweils zufällig gezogenen Teilmenge der Daten und Merkmale.
    Die Prognose des Waldes ist der DURCHSCHNITT aller Baum-Prognosen —
    durch das Mitteln vieler unterschiedlicher Bäume gleichen sich deren
    individuelle Fehler teilweise aus.

    Arbeitsteilung dieser Klasse:
      1. fit_with_tuning()        — Modell trainieren + beste Einstellungen suchen
      2. predict_monthly_returns()— Renditen des nächsten Monats vorhersagen
      3. optimize()               — mit den Prognosen ein Markowitz-Portfolio bauen
    Die KI ersetzt also nur die RENDITESCHÄTZUNG; die Portfoliokonstruktion
    bleibt dieselbe Sharpe-Maximierung wie bei der klassischen Strategie —
    so ist der Vergleich fair.
    """

    def __init__(self, rf: float = RISK_FREE_RATE,
                 n_iter: int = RF_N_ITER,
                 cv_splits: int = RF_CV_SPLITS):
        self.rf               = rf          # risikofreier Zins (für max_sharpe)
        self.n_iter           = n_iter      # Anzahl getesteter Einstellungs-Kombis
        self.cv_splits        = cv_splits   # Anzahl Validierungs-Zeitabschnitte
        self.best_estimator_  = None        # das fertig trainierte Modell (anfangs leer)
        self.best_params_     = {}          # die besten gefundenen Einstellungen
        self._mvo             = MarkowitzLedoitWolf(rf=rf)  # wiederverwendeter Optimierer

    def _build_pipeline(self) -> Pipeline:
        """Baut die Verarbeitungskette: erst skalieren, dann Random Forest.

        Eine "Pipeline" kettet Schritte, die immer zusammen laufen sollen:
          1. StandardScaler: bringt alle Merkmale auf vergleichbare Größen-
             ordnung (Mittelwert 0, Streuung 1) — sonst würde z. B. der RSI
             (0–100) ganz andere Zahlenbereiche haben als Renditen (±0,1).
          2. RandomForestRegressor: das eigentliche Vorhersagemodell.
        random_state=42 fixiert den Zufallsgenerator → jeder Lauf liefert
        exakt dieselben Ergebnisse (Reproduzierbarkeit!). n_jobs=-1 heißt:
        alle Prozessorkerne parallel nutzen.
        """
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestRegressor(random_state=42, n_jobs=-1)),
        ])

    def _param_grid(self) -> dict:
        """Suchraum der Hyperparameter (der "Stellschrauben" des Waldes).

        Für jede Stellschraube ein Zufallsbereich, aus dem die Suche zieht:
          n_estimators     — Anzahl Bäume im Wald (100–500)
          max_depth        — maximale Tiefe je Baum (3–15 Entscheidungsebenen);
                             flachere Bäume = einfachere Regeln = weniger Gefahr,
                             Zufallsrauschen "auswendig zu lernen" (Overfitting)
          min_samples_leaf — Mindestanzahl Datenpunkte pro Endknoten (3–20);
                             größer = glattere, vorsichtigere Prognosen
          max_features     — Anteil der Merkmale, die jeder Baum sehen darf
                             (30–90 %); erzwingt Vielfalt unter den Bäumen
          max_samples      — Anteil der Daten je Baum (60–95 %); ebenso
        (randint(a,b) zieht ganze Zahlen in [a,b); uniform(a,c) zieht
        Kommazahlen in [a, a+c].)
        """
        return {
            "rf__n_estimators"     : randint(100, 500),
            "rf__max_depth"        : randint(3, 15),
            "rf__min_samples_leaf" : randint(3, 20),
            "rf__max_features"     : uniform(0.3, 0.6),
            "rf__max_samples"      : uniform(0.6, 0.35),
        }

    @timer
    def fit_with_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                        sample_times=None) -> None:
        """
        Training mit RandomizedSearchCV.

        Ablauf in Worten: Es werden n_iter (=30) zufällige Einstellungs-
        Kombinationen aus _param_grid() gezogen. Jede Kombination wird per
        KREUZVALIDIERUNG bewertet: Die Trainingsdaten werden mehrfach in
        "Lernteil" und "Prüfteil" zerlegt, das Modell lernt auf dem einen und
        wird auf dem anderen getestet (Maß: mittlerer quadratischer
        Prognosefehler, MSE — "neg_" davor, weil sklearn Scores maximiert).
        Die beste Kombination gewinnt und wird abschließend auf ALLEN
        Trainingsdaten neu trainiert (refit=True).

        Standard: TimeSeriesSplit (Testdaten liegen zeitlich NACH dem Training) —
        bei Zeitreihen Pflicht, sonst würde das Modell an Daten geprüft, die
        zeitlich VOR seinem Lernstoff liegen (Blick in die Zukunft).
        Optional (config use_purged_cv=True): Purged & Embargoed CV nach
        López de Prado (2018), die überlappende Labels zwischen Train und Test
        entfernt — wissenschaftlich sauberer bei überlappenden Monatslabels.
        ``sample_times`` ist die Periode (Monatsende) je Zeile von X_train.

        (X_train = Tabelle der Merkmale, eine Zeile pro Aktie und Monat;
         y_train = zugehörige Zielwerte, die Rendite des jeweiligen Folgemonats.)
        """
        if USE_PURGED_CV and sample_times is not None:
            cv = purged_kfold_splits(
                sample_times, n_splits=self.cv_splits, embargo_pct=CV_EMBARGO,
            )
        else:
            cv = TimeSeriesSplit(n_splits=self.cv_splits)
        search = RandomizedSearchCV(
            estimator=self._build_pipeline(),
            param_distributions=self._param_grid(),
            n_iter=self.n_iter, scoring="neg_mean_squared_error",
            cv=cv, random_state=42, n_jobs=-1, refit=True,
        )
        search.fit(X_train.values, y_train.values)   # hier läuft die ganze Suche
        self.best_estimator_ = search.best_estimator_  # Sieger-Modell merken
        self.best_params_    = search.best_params_     # Sieger-Einstellungen merken
        log.info(
            f"    RF-Params: n_est={self.best_params_.get('rf__n_estimators','?')}, "
            f"depth={self.best_params_.get('rf__max_depth','?')}, "
            f"leaf={self.best_params_.get('rf__min_samples_leaf','?')}"
        )

    def refit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Schnelles Neutraining OHNE Hyperparametersuche: nutzt die zuletzt per
        fit_with_tuning() gefundenen Parameter und passt das Modell nur an das
        aktuelle (rollierende) Trainingsfenster an. Spart den teuren
        RandomizedSearchCV, wenn nicht in jedem Monat neu getunt werden soll
        (Performance-Hebel RF_RETUNE_EVERY). Fällt auf volle Suche zurück,
        falls noch keine Parameter bekannt sind.
        """
        if not self.best_params_:
            return self.fit_with_tuning(X_train, y_train)
        pipe = self._build_pipeline()
        pipe.set_params(**self.best_params_)   # gemerkte Einstellungen übernehmen
        pipe.fit(X_train.values, y_train.values)
        self.best_estimator_ = pipe

    def predict_monthly_returns(self, X_current: pd.DataFrame) -> np.ndarray:
        """Sagt für jede Aktie die Rendite des nächsten Monats voraus.

        X_current enthält je Aktie eine Zeile mit den AKTUELLEN Merkmalen.
        "* 12": Die Monatsprognose wird auf Jahresbasis hochgerechnet, weil
        der nachgelagerte Markowitz-Optimierer mit annualisierten Renditen
        und einer annualisierten Kovarianzmatrix rechnet.
        """
        if self.best_estimator_ is None:
            # Schutz vor Bedienfehler: Vorhersagen ohne Training ergäbe Unsinn.
            raise RuntimeError("Zuerst fit_with_tuning() aufrufen.")
        return self.best_estimator_.predict(X_current.values) * 12

    def optimize(self, mu_predicted: np.ndarray, cov: np.ndarray,
                 w_prev: np.ndarray = None) -> np.ndarray:
        """Baut aus den KI-Prognosen das Portfolio (Sharpe-Maximierung).

        Nutzt exakt denselben Markowitz-Löser wie die klassische Strategie —
        nur mit den PROGNOSTIZIERTEN statt den historischen Renditen, plus
        Turnover-Limit (max. 30 % Depotumbau pro Monat, gegen Kostenfraß).
        """
        return self._mvo.max_sharpe(
            mu_predicted, cov,
            w_prev=w_prev, turnover_limit=RF_TURNOVER_LIMIT,
        )
