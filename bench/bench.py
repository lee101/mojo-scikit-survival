"""Benchmarks against scikit-survival on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

from mojo_sksurv import metrics as mojo_metrics  # noqa: E402
from mojo_sksurv import nonparametric as mojo_nonparametric  # noqa: E402
from mojo_sksurv.linear_model import CoxPHSurvivalAnalysis as MojoCoxPH  # noqa: E402
from mojo_sksurv.util import Surv  # noqa: E402
from sksurv import metrics as upstream_metrics  # noqa: E402
from sksurv import nonparametric as upstream_nonparametric  # noqa: E402
from sksurv.linear_model import CoxPHSurvivalAnalysis as UpstreamCoxPH  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def make_survival(n, seed=0):
    rng = np.random.default_rng(seed)
    time_ = np.round(rng.exponential(20.0, n), 3) + 0.01
    event = rng.random(n) < 0.7
    event[0] = True
    return rng, event, time_, Surv.from_arrays(event, time_)


def cases():
    rng, event, time_, y = make_survival(12_000)
    risk = rng.normal(size=len(y))
    yield (
        "concordance_index_censored (12k)",
        lambda: mojo_metrics.concordance_index_censored(event, time_, risk),
        lambda: upstream_metrics.concordance_index_censored(event, time_, risk),
    )

    rng, event, time_, y = make_survival(1_000_000, 1)
    yield (
        "kaplan_meier_estimator (1m)",
        lambda: mojo_nonparametric.kaplan_meier_estimator(event, time_),
        lambda: upstream_nonparametric.kaplan_meier_estimator(event, time_),
    )

    rng, event, time_, y = make_survival(300_000, 2)
    eval_times = np.linspace(np.quantile(time_, 0.1), np.quantile(time_, 0.9), 24)
    estimate = np.clip(
        rng.normal(0.6, 0.2, size=(len(y), len(eval_times))), 0.0, 1.0
    )
    yield (
        "brier_score (300k x 24)",
        lambda: mojo_metrics.brier_score(y, y, estimate, eval_times),
        lambda: upstream_metrics.brier_score(y, y, estimate, eval_times),
    )

    rng, event, time_, y = make_survival(2_000, 3)
    eval_times = np.linspace(np.quantile(time_, 0.1), np.quantile(time_, 0.9), 12)
    risk = rng.normal(size=len(y))
    yield (
        "cumulative_dynamic_auc (2k x 12)",
        lambda: mojo_metrics.cumulative_dynamic_auc(y, y, risk, eval_times),
        lambda: upstream_metrics.cumulative_dynamic_auc(y, y, risk, eval_times),
    )

    rng, event, time_, y = make_survival(20_000, 4)
    X = np.ascontiguousarray(rng.normal(size=(len(y), 10)))
    yield (
        "CoxPH.fit Breslow (20k x 10)",
        lambda: MojoCoxPH(n_iter=30).fit(X, y),
        lambda: UpstreamCoxPH(n_iter=30).fit(X, y),
    )


def machine():
    cpu = "unknown CPU"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{cpu}; {platform.system()} {platform.release()}"


def main():
    print(f"Machine: {machine()}")
    print()
    print("| case | mojo-scikit-survival | scikit-survival | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_function, upstream_function in cases():
        mojo_function()
        upstream_function()
        mojo_time = timeit(mojo_function, repeat=3)
        upstream_time = timeit(upstream_function, repeat=3)
        ratio = upstream_time / mojo_time
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1.0
            else f"{1.0 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
