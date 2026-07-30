"""Metrics for right-censored survival outcomes."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib
from .exceptions import NoComparablePairException
from .nonparametric import CensoringDistributionEstimator, kaplan_meier_estimator
from .util import check_y_survival


def _one_dimensional(values, name):
    result = np.asarray(values)
    if np.iscomplexobj(result):
        raise ValueError(f"{name} must contain only real values")
    if result.ndim != 1 or not np.isfinite(result.astype(float)).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return result


def _concordance(event, time, estimate, weights, tied_tol):
    event = _one_dimensional(event, "event_indicator")
    time = _one_dimensional(time, "event_time")
    estimate = _one_dimensional(estimate, "estimate")
    if event.dtype != np.bool_:
        raise ValueError("only boolean arrays are supported as event indicators")
    if not (len(event) == len(time) == len(estimate)) or len(event) < 2:
        raise ValueError("inputs must have equal length of at least two")
    if not event.any():
        raise ValueError("all samples are censored")
    event_buffer = np.ascontiguousarray(event, dtype=np.bool_)
    time_buffer = f64(time)
    estimate_buffer = f64(estimate)
    weight_buffer = f64(weights)
    stats = np.empty(5)
    work = np.empty((len(event), 5))
    result = lib().mss_concordance(
        addr(event_buffer),
        addr(time_buffer),
        addr(estimate_buffer),
        addr(weight_buffer),
        addr(stats),
        addr(work),
        len(event),
        float(tied_tol),
    )
    if result < 0.0:
        raise NoComparablePairException(
            "data has no comparable pairs, cannot estimate concordance index"
        )
    return (result, *(int(value) for value in stats[:4]))


def concordance_index_censored(
    event_indicator, event_time, estimate, tied_tol=1e-8
):
    return _concordance(
        event_indicator,
        event_time,
        estimate,
        np.ones(len(event_time)),
        tied_tol,
    )


def concordance_index_ipcw(
    survival_train, survival_test, estimate, tau=None, tied_tol=1e-8
):
    event, time = check_y_survival(survival_test)
    estimate = _one_dimensional(estimate, "estimate")
    if len(estimate) != len(time):
        raise ValueError("estimate and survival_test must have equal length")
    censoring = CensoringDistributionEstimator().fit(survival_train)
    if tau is None:
        weights = censoring.predict_ipcw(survival_test)
    else:
        mask = time < tau
        weights = np.zeros(len(time))
        weights[mask] = censoring.predict_ipcw(survival_test[mask])
    return _concordance(event, time, estimate, weights * weights, tied_tol)


def _check_times(test_time, times):
    result = np.unique(np.atleast_1d(np.asarray(times, dtype=float)))
    if (
        result.size == 0
        or result.min() < test_time.min()
        or result.max() >= test_time.max()
    ):
        raise ValueError(
            "all times must be within the follow-up interval of the test data"
        )
    return result


def _estimate_matrix(estimate, n_samples, times, *, allow_1d=False):
    raw = np.asarray(estimate)
    if np.iscomplexobj(raw):
        raise ValueError("estimate must contain only real values")
    result = np.asarray(raw, dtype=float)
    if result.ndim == 1 and allow_1d:
        if len(result) != n_samples:
            raise ValueError("estimate has inconsistent number of samples")
    elif result.ndim == 1 and len(times) == 1:
        result = result.reshape(-1, 1)
    expected_shape = (n_samples,) if result.ndim == 1 and allow_1d else (
        n_samples,
        len(times),
    )
    if result.shape != expected_shape:
        raise ValueError(
            f"expected estimate with shape ({n_samples}, {len(times)}), "
            f"got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("estimate must contain only finite values")
    return f64(result)


def brier_score(survival_train, survival_test, estimate, times):
    event, time = check_y_survival(survival_test)
    times = _check_times(time, times)
    estimate = _estimate_matrix(estimate, len(time), times)
    censoring = CensoringDistributionEstimator().fit(survival_train)
    prob_t = f64(censoring.predict_proba(times))
    prob_y = f64(censoring.predict_proba(time))
    event_buffer = np.ascontiguousarray(event, dtype=np.bool_)
    time_buffer = f64(time)
    times_buffer = f64(times)
    scores = np.empty(len(times))
    lib().mss_brier_score(
        addr(event_buffer),
        addr(time_buffer),
        addr(estimate),
        addr(prob_y),
        addr(times_buffer),
        addr(prob_t),
        addr(scores),
        len(time),
        len(times),
    )
    return times, scores


def integrated_brier_score(survival_train, survival_test, estimate, times):
    times, scores = brier_score(survival_train, survival_test, estimate, times)
    if len(times) < 2:
        raise ValueError("at least two time points must be given")
    return float(np.trapezoid(scores, times) / (times[-1] - times[0]))


def cumulative_dynamic_auc(
    survival_train, survival_test, estimate, times, tied_tol=1e-8
):
    event, time = check_y_survival(survival_test)
    times = _check_times(time, times)
    estimate = _estimate_matrix(estimate, len(time), times, allow_1d=True)
    if estimate.ndim == 1:
        order = np.ascontiguousarray(np.argsort(estimate)[None, :], dtype=np.int64)
        estimate_times = 1
    else:
        order = np.ascontiguousarray(
            np.argsort(estimate.T, axis=1), dtype=np.int64
        )
        estimate_times = len(times)
    censoring = CensoringDistributionEstimator().fit(survival_train)
    ipcw = f64(censoring.predict_ipcw(survival_test))
    event_buffer = np.ascontiguousarray(event, dtype=np.bool_)
    time_buffer = f64(time)
    times_buffer = f64(times)
    scores = np.empty(len(times))
    lib().mss_dynamic_auc(
        addr(event_buffer),
        addr(time_buffer),
        addr(estimate),
        addr(ipcw),
        addr(times_buffer),
        addr(order),
        addr(scores),
        len(time),
        len(times),
        estimate_times,
        float(tied_tol),
    )
    if len(times) == 1:
        mean_auc = scores[0]
    else:
        km_time, km_prob = kaplan_meier_estimator(event, time)
        index = np.searchsorted(km_time, times, side="right") - 1
        survival = np.where(index >= 0, km_prob[np.maximum(index, 0)], 1.0)
        increments = -np.diff(np.r_[1.0, survival])
        mean_auc = np.sum(scores * increments) / (1.0 - survival[-1])
    return scores, float(mean_auc)
