"""Non-parametric survival estimators."""

from __future__ import annotations

import numbers

import numpy as np
from scipy import stats

from ._lib import addr, f64, lib


def _validate_event_time(event, time, *, allow_all_censored=False):
    event = np.asarray(event)
    raw_time = np.asarray(time)
    if np.iscomplexobj(raw_time):
        raise ValueError("time must contain only real values")
    time = np.asarray(raw_time, dtype=float)
    if event.ndim != 1 or time.ndim != 1 or len(event) != len(time):
        raise ValueError("event and time must be one-dimensional arrays of equal length")
    if len(event) == 0:
        raise ValueError("event and time must contain at least one sample")
    if event.dtype != np.bool_:
        raise ValueError("event indicator must be boolean")
    if not np.isfinite(time).all():
        raise ValueError("time must contain only finite values")
    if not allow_all_censored and not event.any():
        raise ValueError("all samples are censored")
    return event, time


def _loglog_ci(survival, ratio_var, conf_level):
    if not isinstance(conf_level, numbers.Real) or not 0.0 < conf_level < 1.0:
        raise ValueError("conf_level must be a float in the range (0.0, 1.0)")
    sigma = np.sqrt(np.cumsum(ratio_var))
    eps = np.finfo(float).eps
    mask = survival > eps
    log_p = np.zeros_like(survival)
    np.log(survival, where=mask, out=log_p)
    theta = np.zeros_like(survival)
    np.divide(sigma, log_p, where=log_p < -eps, out=theta)
    z = stats.norm.isf((1.0 - conf_level) / 2.0)
    ci = np.exp(np.exp(z * np.multiply.outer([-1, 1], theta)) * log_p)
    ci[:, ~mask] = 0.0
    return ci


def _left_truncated_counts(event, time_enter, time_exit):
    if np.any(time_enter > time_exit):
        raise ValueError("exit time must be larger start time for all samples")
    times = np.sort(np.unique(np.r_[time_enter, time_exit]), kind="mergesort")
    at_risk = np.empty(len(times), dtype=float)
    events = np.empty(len(times), dtype=float)
    for i, current in enumerate(times):
        if i == 0:
            risk = time_enter <= current
            events[i] = 0.0
        else:
            risk = (time_enter < current) & (time_exit >= current)
            events[i] = np.count_nonzero(event & (time_exit == current))
        at_risk[i] = np.count_nonzero(risk)
    return times, events, at_risk


def kaplan_meier_estimator(
    event,
    time_exit,
    time_enter=None,
    time_min=None,
    reverse=False,
    conf_level=0.95,
    conf_type=None,
):
    event, time_exit = _validate_event_time(
        event, time_exit, allow_all_censored=True
    )
    if conf_type is not None and reverse:
        raise NotImplementedError(
            "confidence intervals of the censoring distribution are not implemented"
        )
    if time_enter is not None:
        if reverse:
            raise ValueError(
                "the censoring distribution cannot be estimated from left truncated data"
            )
        raw_time_enter = np.asarray(time_enter)
        if np.iscomplexobj(raw_time_enter):
            raise ValueError("time_enter must contain only real values")
        time_enter = np.asarray(raw_time_enter, dtype=float)
        if time_enter.shape != time_exit.shape:
            raise ValueError("time_enter and time_exit must have equal shape")
        if not np.isfinite(time_enter).all():
            raise ValueError("time_enter must contain only finite values")
        times, n_events, n_at_risk = _left_truncated_counts(
            event, time_enter, time_exit
        )
        ratio = np.divide(
            n_events,
            n_at_risk,
            out=np.zeros_like(n_events),
            where=n_events != 0,
        )
        ratio_var = np.divide(
            n_events,
            n_at_risk * (n_at_risk - n_events),
            out=np.zeros_like(n_events),
            where=(n_events != 0) & (n_at_risk != n_events),
        )
        if time_min is not None:
            keep = times >= time_min
            times, ratio, ratio_var = times[keep], ratio[keep], ratio_var[keep]
        survival = np.cumprod(1.0 - ratio)
    else:
        order = np.argsort(time_exit, kind="mergesort")
        sorted_event = f64(event[order])
        sorted_time = f64(time_exit[order])
        times = np.empty(len(event))
        survival = np.empty(len(event))
        ratio_var = np.empty(len(event))
        count = lib().mss_product_limit(
            addr(sorted_event),
            addr(sorted_time),
            addr(times),
            addr(survival),
            addr(ratio_var),
            len(event),
            int(reverse),
            0.0 if time_min is None else float(time_min),
            int(time_min is not None),
        )
        times, survival, ratio_var = (
            values[:count].copy() for values in (times, survival, ratio_var)
        )
    if conf_type is None:
        return times, survival
    if conf_type != "log-log":
        raise ValueError("conf_type must be None or 'log-log'")
    return times, survival, _loglog_ci(survival, ratio_var, conf_level)


def nelson_aalen_estimator(event, time):
    event, time = _validate_event_time(event, time)
    order = np.argsort(time, kind="mergesort")
    sorted_event = f64(event[order])
    sorted_time = f64(time[order])
    times = np.empty(len(event))
    hazard = np.empty(len(event))
    count = lib().mss_nelson_aalen(
        addr(sorted_event), addr(sorted_time), addr(times), addr(hazard), len(event)
    )
    return times[:count].copy(), hazard[:count].copy()


def _step_predict(unique_time, probability, query):
    query = np.asarray(query, dtype=float)
    extends = query > unique_time[-1]
    if probability[-1] > 0 and extends.any():
        raise ValueError(
            f"time must be smaller than largest observed time point: {unique_time[-1]}"
        )
    result = np.empty(query.shape, dtype=float)
    result[extends] = 0.0
    valid = ~extends
    indices = np.searchsorted(unique_time, query[valid])
    exact = np.abs(unique_time[indices] - query[valid]) < np.finfo(float).eps
    indices[~exact] -= 1
    result[valid] = probability[indices]
    return result


class CensoringDistributionEstimator:
    def fit(self, y):
        from .util import check_y_survival

        event, time = check_y_survival(y)
        if event.all():
            self.unique_time_ = np.r_[-np.inf, np.unique(time)]
            self.prob_ = np.ones(len(self.unique_time_))
        else:
            unique_time, probability = kaplan_meier_estimator(
                event, time, reverse=True
            )
            self.unique_time_ = np.r_[-np.inf, unique_time]
            self.prob_ = np.r_[1.0, probability]
        return self

    def predict_proba(self, time):
        return _step_predict(self.unique_time_, self.prob_, time)

    def predict_ipcw(self, y):
        from .util import check_y_survival

        event, time = check_y_survival(y)
        probability = self.predict_proba(time[event])
        if np.any(probability == 0.0):
            raise ValueError(
                "censoring survival function is zero at one or more time points"
            )
        weights = np.zeros(len(time))
        weights[event] = 1.0 / probability
        return weights
