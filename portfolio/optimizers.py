"""Portfolio-Optimierer: Markowitz (Ledoit-Wolf), Risk Parity, Random Forest."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import randint, uniform
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .config import *
from .metrics import timer
from .cross_validation import purged_kfold_splits

class MarkowitzLedoitWolf:
    """
    Klassische Mean-Variance Optimization mit Ledoit-Wolf Kovarianzschätzung.
    Referenz: Ledoit & Wolf (2004). Journal of Multivariate Analysis.
    """

    def __init__(self, rf: float = RISK_FREE_RATE):
        self.rf = rf

    def estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        lw = LedoitWolf()
        lw.fit(returns.values)
        return lw.covariance_ * 252

    def _neg_sharpe(self, weights, mu, cov):
        ret = float(weights @ mu)
        vol = float(np.sqrt(weights @ cov @ weights))
        return -(ret - self.rf) / vol if vol > 0 else 0.0

    def max_sharpe(self, mu: np.ndarray, cov: np.ndarray,
                   w_prev: np.ndarray = None,
                   turnover_limit: float = None) -> np.ndarray:
        """
        Maximiert Sharpe Ratio (Tangential-Portfolio).
        Long-Only, vollständig investiert. Positionsobergrenze: MAX_WEIGHT.
        Optional: Turnover-Constraint nach Garleanu & Pedersen (2013).
        """
        n  = len(mu)
        w0 = w_prev if w_prev is not None else np.ones(n) / n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds      = [(0.0, MAX_WEIGHT)] * n

        if w_prev is not None and turnover_limit is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, wp=w_prev, lim=turnover_limit:
                       lim - np.abs(w - wp).sum() / 2,
            })

        result = minimize(
            fun=self._neg_sharpe, x0=w0, args=(mu, cov),
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        w = np.maximum(result.x, 0.0)
        w /= w.sum()
        return w

    def efficient_frontier(self, mu: np.ndarray, cov: np.ndarray,
                            n_points: int = N_FRONTIER) -> pd.DataFrame:
        """Berechnet N Punkte auf der Effizienzlinie (Long-Only)."""
        n      = len(mu)
        bounds = [(0.0, 1.0)] * n

        res_mvp = minimize(
            lambda w: float(w @ cov @ w), np.ones(n) / n,
            method="SLSQP", bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        )
        ret_min = float(res_mvp.x @ mu)
        ret_max = float(mu.max())

        frontier = []
        for target in np.linspace(ret_min, ret_max, n_points):
            res = minimize(
                lambda w: float(w @ cov @ w), np.ones(n) / n,
                method="SLSQP", bounds=bounds,
                constraints=[
                    {"type": "eq", "fun": lambda w: w.sum() - 1},
                    {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
                ],
                options={"maxiter": 500, "ftol": 1e-12},
            )
            if res.success:
                vol = float(np.sqrt(res.x @ cov @ res.x))
                sr  = (target - self.rf) / vol if vol > 0 else 0.0
                frontier.append({"ret": target, "vol": vol, "sr": sr, "w": res.x.copy()})

        return pd.DataFrame(frontier)


class RiskParityPortfolio:
    """
    Equal Risk Contribution (Risk Parity) Portfolio.

    Jedes Asset trägt gleich viel zum gesamten Portfoliorisiko bei.
    Formell: RC_i = w_i * (Sigma w)_i / sqrt(w' Sigma w) = 1/N für alle i

    Vorteile gegenüber Equal Weight:
      - Berücksichtigt Asset-Volatilität und Korrelationen
      - Weniger Konzentration in risikoreichen Assets
      - Empirisch bessere risikobereinigte Renditen (Qian, 2005)

    Referenz: Qian, E. (2005). Risk Parity Portfolios: Efficient Portfolios
    through True Diversification. PanAgora Asset Management.
    """

    def optimize(self, cov: np.ndarray, max_weight: float = MAX_WEIGHT) -> np.ndarray:
        n = cov.shape[0]

        def _objective(w):
            """Minimiert die Streuung der Risikobeiträge."""
            portfolio_vol = np.sqrt(max(w @ cov @ w, 1e-10))
            rc = w * (cov @ w) / portfolio_vol
            target_rc = portfolio_vol / n
            return float(np.sum((rc - target_rc) ** 2))

        w0 = np.ones(n) / n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0.005, max_weight)] * n

        result = minimize(
            fun=_objective, x0=w0,
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 3000, "ftol": 1e-14},
        )

        w = np.maximum(result.x, 0.0)
        w /= w.sum()
        return w


class RFPortfolioOptimizer:
    """
    Random-Forest-gestützte Portfoliooptimierung mit monatlichem Rebalancing.
    Referenz: Fischer & Krauss (2018). European Journal of Operational Research.
    """

    def __init__(self, rf: float = RISK_FREE_RATE,
                 n_iter: int = RF_N_ITER,
                 cv_splits: int = RF_CV_SPLITS):
        self.rf               = rf
        self.n_iter           = n_iter
        self.cv_splits        = cv_splits
        self.best_estimator_  = None
        self.best_params_     = {}
        self._mvo             = MarkowitzLedoitWolf(rf=rf)

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestRegressor(random_state=42, n_jobs=-1)),
        ])

    def _param_grid(self) -> dict:
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

        Standard: TimeSeriesSplit (Testdaten liegen zeitlich NACH dem Training).
        Optional (config use_purged_cv=True): Purged & Embargoed CV nach
        López de Prado (2018), die überlappende Labels zwischen Train und Test
        entfernt — wissenschaftlich sauberer bei überlappenden Monatslabels.
        ``sample_times`` ist die Periode (Monatsende) je Zeile von X_train.
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
        search.fit(X_train.values, y_train.values)
        self.best_estimator_ = search.best_estimator_
        self.best_params_    = search.best_params_
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
        pipe.set_params(**self.best_params_)
        pipe.fit(X_train.values, y_train.values)
        self.best_estimator_ = pipe

    def predict_monthly_returns(self, X_current: pd.DataFrame) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Zuerst fit_with_tuning() aufrufen.")
        return self.best_estimator_.predict(X_current.values) * 12

    def optimize(self, mu_predicted: np.ndarray, cov: np.ndarray,
                 w_prev: np.ndarray = None) -> np.ndarray:
        return self._mvo.max_sharpe(
            mu_predicted, cov,
            w_prev=w_prev, turnover_limit=RF_TURNOVER_LIMIT,
        )
