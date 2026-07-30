# mojo-scikit-survival

`mojo-scikit-survival` is a standalone Mojo port of the compute-heavy core of
[scikit-survival](https://scikit-survival.readthedocs.io/). It provides a Python
package, `mojo_sksurv`, with the same names and call signatures for the covered
subset. Survival targets use the same two-field NumPy structured arrays, so existing
data can cross between the two libraries unchanged.

The project is useful today for non-parametric estimates, model evaluation, and
unregularized or ridge-regularized Cox proportional-hazards regression. It is not a
placeholder: every covered result is numerically compared with scikit-survival
0.25.0 in the test suite.

## Coverage

| Upstream module | Covered API |
| --- | --- |
| `sksurv.util` | `Surv.from_arrays` |
| `sksurv.functions` | `StepFunction` |
| `sksurv.nonparametric` | `kaplan_meier_estimator` including reverse KM, `time_min`, left truncation, and log-log confidence intervals; `nelson_aalen_estimator` |
| `sksurv.metrics` | `concordance_index_censored`, `concordance_index_ipcw`, `brier_score`, `integrated_brier_score`, `cumulative_dynamic_auc` |
| `sksurv.linear_model` | `CoxPHSurvivalAnalysis`, including Breslow and Efron ties, scalar or per-feature ridge penalties, risk prediction, baseline curves, survival curves, cumulative-hazard curves, and concordance scoring |

Not covered are Coxnet/LASSO, survival SVMs, trees, random survival forests,
gradient boosting, component-wise boosting, competing-risk estimators, dataset
loaders, preprocessing, and scikit-learn scorer adapters. Those APIs continue to
require upstream scikit-survival.

## Install

The Pixi environment contains the pinned Mojo nightly, Python dependencies, and
scikit-survival used for parity tests:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-scikit-survival.so`. Importing
`mojo_sksurv` also rebuilds a missing or stale library automatically.

## Usage

This example is complete and runs from the repository root with `pixi run python`:

```python
import numpy as np

from mojo_sksurv.linear_model import CoxPHSurvivalAnalysis
from mojo_sksurv.metrics import concordance_index_censored
from mojo_sksurv.util import Surv

X = np.array([
    [0.2, 1.1],
    [1.4, -0.3],
    [0.7, 0.5],
    [-0.2, 1.7],
    [1.1, 0.1],
    [0.0, 0.8],
])
y = Surv.from_arrays(
    event=[True, True, False, True, False, True],
    time=[5.0, 2.0, 8.0, 3.0, 7.0, 6.0],
)

model = CoxPHSurvivalAnalysis(alpha=0.1).fit(X, y)
risk = model.predict(X)
print(concordance_index_censored(y["event"], y["time"], risk)[0])
print(model.predict_survival_function(X[:1], return_array=True))
```

## Performance

Measured by `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic. Each implementation receives the same arrays, both are warmed
once, and the table reports the best of three runs.

| case | mojo-scikit-survival | scikit-survival | result |
| --- | ---: | ---: | ---: |
| `concordance_index_censored` (12k) | 9.05 ms | 850.19 ms | 93.90x faster |
| `kaplan_meier_estimator` (1m) | 230.55 ms | 1411.03 ms | 6.12x faster |
| `brier_score` (300k x 24) | 156.70 ms | 965.86 ms | 6.16x faster |
| `cumulative_dynamic_auc` (2k x 12) | 1.47 ms | 19.61 ms | 13.38x faster |
| `CoxPH.fit` Breslow (20k x 10) | 54.52 ms | 1004.14 ms | 18.42x faster |

Concordance classifies comparable pairs with host-width SIMD and distributes large
row sets across CPU workers. Brier-score time columns and sufficiently large dynamic
AUC scans are independent parallel tasks; smaller inputs remain serial. Dynamic AUC
uses a descending risk sort followed by a linear ROC scan, including upstream's
tolerance-based tie grouping.

No GPU path is provided; this port currently targets CPU execution only.

Run benchmarks only through the flocked task so concurrent jobs do not distort the
measurements:

```bash
pixi run bench
```

## How it works

All native routines live in `src/survival.mojo`, so one `mojo build --emit
shared-lib` invocation produces the library. Python calls exported C-ABI functions
through `ctypes`; a model fit or metric evaluation makes a small number of native
calls rather than crossing the boundary inside an inner loop.

NumPy owns inputs, outputs, and scratch memory. Numeric arrays are C-contiguous
`float64`, event indicators remain byte-sized NumPy booleans, and sorted AUC indices
are `int64`. Matrices are row-major and structured survival targets are split into
contiguous event and time buffers. Buffer addresses cross the ABI as 64-bit integers
and are rebuilt as matching typed pointers in Mojo. Mojo never retains a pointer or
owns an allocation, so Python controls every lifetime.

The Cox optimizer follows upstream's Newton-Raphson update and step-halving rule.
Mojo computes the partial negative log-likelihood, gradient, Hessian, and Breslow
baseline hazard. The Python layer performs the small dense linear solve and returns
upstream-compatible `StepFunction` objects.

## License

MIT. See `LICENSE`.
