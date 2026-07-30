"""Step functions returned by survival estimators."""

from __future__ import annotations

import numpy as np


class StepFunction:
    def __init__(self, x, y, *, a=1.0, b=0.0, domain=(0, None)):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        if self.x.ndim != 1 or len(self.x) != len(self.y) or len(self.x) == 0:
            raise ValueError("x and y must be non-empty one-dimensional arrays of equal length")
        self.a = a
        self.b = b
        lower = self.x[0] if domain[0] is None else domain[0]
        upper = self.x[-1] if domain[1] is None else domain[1]
        self._domain = (float(lower), float(upper))

    @property
    def domain(self):
        return self._domain

    def __call__(self, x):
        values = np.atleast_1d(x)
        if not np.isfinite(values).all():
            raise ValueError("x must be finite")
        if values.min() < self.domain[0] or values.max() > self.domain[1]:
            raise ValueError(
                f"x must be within [{self.domain[0]:f}; {self.domain[1]:f}]"
            )
        values = np.clip(values, self.x[0], None)
        indices = np.searchsorted(self.x, values, side="left")
        exact = self.x[indices] == values
        indices[~exact] -= 1
        result = self.a * self.y[indices] + self.b
        return result[0] if result.size == 1 else result

    def __repr__(self):
        return (
            f"StepFunction(x={self.x!r}, y={self.y!r}, "
            f"a={self.a!r}, b={self.b!r})"
        )
