"""Survival target construction and validation."""

from __future__ import annotations

import numpy as np


class Surv:
    """Create the two-field structured target used by scikit-survival."""

    @staticmethod
    def from_arrays(event, time, name_event=None, name_time=None):
        event = np.asarray(event)
        time = np.asarray(time)
        if event.ndim != 1 or time.ndim != 1 or len(event) != len(time):
            raise ValueError("event and time must be one-dimensional arrays of equal length")
        if np.iscomplexobj(time):
            raise ValueError("time must contain only real values")
        if event.dtype != np.bool_:
            values = np.unique(event)
            if not np.all(np.isin(values, [0, 1])):
                raise ValueError("event indicator must be binary")
            event = event.astype(bool)
        if not np.isfinite(time.astype(float)).all():
            raise ValueError("time must contain only finite values")
        name_event = "event" if name_event is None else name_event
        name_time = "time" if name_time is None else name_time
        result = np.empty(len(event), dtype=[(name_event, "?"), (name_time, "<f8")])
        result[name_event] = event
        result[name_time] = time
        return result


def check_y_survival(y, *, allow_all_censored=False):
    y = np.asarray(y)
    if y.dtype.names is None or len(y.dtype.names) != 2:
        raise ValueError("y must be a structured array with two fields")
    event = np.asarray(y[y.dtype.names[0]])
    raw_time = np.asarray(y[y.dtype.names[1]])
    if np.iscomplexobj(raw_time):
        raise ValueError("survival times must be real")
    time = np.asarray(raw_time, dtype=float)
    if event.dtype != np.bool_:
        raise ValueError("first field of y must be boolean")
    if y.ndim != 1 or len(y) == 0 or not np.isfinite(time).all():
        raise ValueError("survival target must be non-empty, one-dimensional, and finite")
    if not allow_all_censored and not event.any():
        raise ValueError("all samples are censored")
    return event, time
