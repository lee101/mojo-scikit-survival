"""Cox proportional-hazards regression."""

from __future__ import annotations

import numbers
import warnings

import numpy as np
from scipy.linalg import solve

from ._lib import addr, f64, lib
from .functions import StepFunction
from .metrics import concordance_index_censored
from .util import check_y_survival


class CoxPHSurvivalAnalysis:
    def __init__(self, alpha=0, *, ties="breslow", n_iter=100, tol=1e-9, verbose=0):
        self.alpha = alpha
        self.ties = ties
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

    def _evaluate(self, coefficients):
        gradient = np.empty(self.n_features_in_)
        hessian = np.empty((self.n_features_in_, self.n_features_in_))
        vector_work = np.empty(3 * self.n_features_in_)
        matrix_work = np.empty(2 * self.n_features_in_**2)
        exp_work = np.empty(self._fit_x.shape[0])
        loss = lib().mss_cox_evaluate(
            addr(self._fit_x),
            addr(self._fit_event),
            addr(self._fit_time),
            addr(coefficients),
            addr(self._alpha),
            addr(gradient),
            addr(hessian),
            addr(vector_work),
            addr(matrix_work),
            addr(exp_work),
            self._fit_x.shape[0],
            self.n_features_in_,
            int(self.ties == "efron"),
        )
        return loss, gradient, hessian

    def fit(self, X, y):
        X = f64(X)
        if (
            X.ndim != 2
            or X.shape[0] < 2
            or X.shape[1] < 1
            or not np.isfinite(X).all()
        ):
            raise ValueError(
                "X must be a finite 2D array with at least two rows and one feature"
            )
        event, time = check_y_survival(y)
        if len(event) != X.shape[0]:
            raise ValueError("X and y have inconsistent lengths")
        if self.ties not in {"breslow", "efron"}:
            raise ValueError("ties must be 'breslow' or 'efron'")
        if self.n_iter < 1 or self.tol < 0:
            raise ValueError("n_iter must be positive and tol must be non-negative")
        self.n_features_in_ = X.shape[1]
        if isinstance(self.alpha, numbers.Real | numbers.Integral):
            alpha = np.full(self.n_features_in_, float(self.alpha))
        else:
            alpha = f64(self.alpha)
        if (
            alpha.shape != (self.n_features_in_,)
            or not np.isfinite(alpha).all()
            or np.any(alpha < 0)
        ):
            raise ValueError(
                "alpha must be finite and non-negative with one value per feature"
            )
        order = np.argsort(-time, kind="mergesort")
        self._fit_x = f64(X[order])
        self._fit_event = f64(event[order])
        self._fit_time = f64(time[order])
        self._alpha = f64(alpha)
        coefficients = np.zeros(self.n_features_in_)
        previous = coefficients
        loss = float("inf")
        for iteration in range(self.n_iter):
            _, gradient, hessian = self._evaluate(coefficients)
            delta = solve(hessian, gradient, check_finite=False)
            if not np.all(np.isfinite(delta)):
                raise ValueError("search direction contains NaN or infinite values")
            candidate = coefficients - delta
            loss_new, _, _ = self._evaluate(candidate)
            if loss_new > loss:
                coefficients = (previous + coefficients) / 2.0
                loss, _, _ = self._evaluate(coefficients)
                continue
            previous = coefficients
            coefficients = candidate
            if abs(1.0 - loss_new / loss) < self.tol:
                break
            loss = loss_new
        else:
            warnings.warn(
                "optimization did not converge: maximum iterations exceeded",
                RuntimeWarning,
                stacklevel=2,
            )
        self.coef_ = coefficients
        self._fit_baseline(X, event, time)
        return self

    def _fit_baseline(self, X, event, time):
        linear = self._predict_array(X)
        order = np.argsort(time, kind="mergesort")
        sorted_event = f64(event[order])
        sorted_time = f64(time[order])
        risk = f64(np.exp(linear[order]))
        times = np.empty(len(time))
        hazard = np.empty(len(time))
        count = lib().mss_cox_baseline(
            addr(sorted_event),
            addr(sorted_time),
            addr(risk),
            addr(times),
            addr(hazard),
            len(time),
        )
        self.unique_times_ = times[:count].copy()
        hazard = hazard[:count].copy()
        self.cum_baseline_hazard_ = StepFunction(self.unique_times_, hazard)
        self.baseline_survival_ = StepFunction(
            self.unique_times_, np.exp(-hazard)
        )

    def _predict_array(self, X):
        X = f64(X)
        if (
            X.ndim != 2
            or X.shape[0] < 1
            or X.shape[1] != self.n_features_in_
            or not np.isfinite(X).all()
        ):
            raise ValueError("X has an inconsistent number of features")
        result = np.empty(X.shape[0])
        lib().mss_linear_predict(
            addr(X), addr(self.coef_), addr(result), X.shape[0], X.shape[1]
        )
        return result

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("estimator is not fitted")
        return self._predict_array(X)

    def predict_cumulative_hazard_function(self, X, return_array=False):
        risk = np.exp(self.predict(X))
        values = risk[:, None] * self.cum_baseline_hazard_.y
        if return_array:
            return values
        return np.asarray(
            [
                StepFunction(self.unique_times_, self.cum_baseline_hazard_.y, a=value)
                for value in risk
            ],
            dtype=object,
        )

    def predict_survival_function(self, X, return_array=False):
        risk = np.exp(self.predict(X))
        values = self.baseline_survival_.y[None, :] ** risk[:, None]
        if return_array:
            return values
        return np.asarray(
            [
                StepFunction(self.unique_times_, self.baseline_survival_.y**value)
                for value in risk
            ],
            dtype=object,
        )

    def score(self, X, y):
        event, time = check_y_survival(y)
        return concordance_index_censored(event, time, self.predict(X))[0]
