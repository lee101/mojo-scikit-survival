from __future__ import annotations

import numpy as np
import pytest
from sksurv import metrics as upstream_metrics
from sksurv import nonparametric as upstream_nonparametric
from sksurv.linear_model import CoxPHSurvivalAnalysis as UpstreamCoxPH
from sksurv.util import Surv as UpstreamSurv

from mojo_sksurv._lib import addr, f64
from mojo_sksurv.functions import StepFunction
from mojo_sksurv.linear_model import CoxPHSurvivalAnalysis
from mojo_sksurv.metrics import (
    NoComparablePairException,
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from mojo_sksurv.nonparametric import (
    kaplan_meier_estimator,
    nelson_aalen_estimator,
)
from mojo_sksurv.util import Surv


@pytest.fixture
def survival_data():
    rng = np.random.default_rng(28)
    time = np.round(rng.exponential(12.0, 180), 1) + 0.1
    event = rng.random(180) < 0.68
    event[0] = True
    risk = rng.normal(size=180)
    y = Surv.from_arrays(event, time)
    return rng, event, time, risk, y


def assert_tuple_close(actual, expected):
    assert actual[1:] == tuple(int(value) for value in expected[1:])
    assert actual[0] == pytest.approx(expected[0], rel=2e-14, abs=2e-14)


def test_surv_from_arrays_matches_upstream():
    event = np.array([True, False, True])
    time = np.array([2.0, 3.5, 7.0])
    actual = Surv.from_arrays(event, time, "status", "days")
    expected = UpstreamSurv.from_arrays(event, time, "status", "days")
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_native_buffer_contract_rejects_unsafe_inputs():
    with pytest.raises(ValueError, match="complex"):
        f64(np.array([1.0 + 2.0j]))
    with pytest.raises(TypeError, match="C-contiguous"):
        addr(np.ones((3, 3))[:, 0])


def test_empty_and_complex_inputs_are_rejected_before_native_call():
    with pytest.raises(ValueError):
        kaplan_meier_estimator(np.array([], dtype=bool), np.array([]))
    with pytest.raises(ValueError, match="real"):
        concordance_index_censored(
            np.array([True, True]),
            np.array([1.0, 2.0]),
            np.array([0.0 + 1.0j, 1.0]),
        )


def test_concordance_published_style_vector():
    event = np.array([True, True, False, True, False])
    time = np.array([1.0, 2.0, 2.0, 4.0, 5.0])
    risk = np.array([0.9, 0.7, 0.7 + 5e-9, 0.2, 0.1])
    assert_tuple_close(
        concordance_index_censored(event, time, risk),
        upstream_metrics.concordance_index_censored(event, time, risk),
    )


def test_concordance_random_parity(survival_data):
    _, event, time, risk, _ = survival_data
    assert_tuple_close(
        concordance_index_censored(event, time, risk),
        upstream_metrics.concordance_index_censored(event, time, risk),
    )


def test_concordance_no_comparable_pairs():
    with pytest.raises(NoComparablePairException):
        concordance_index_censored(
            np.array([False, True]),
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
        )


def test_concordance_simd_tail_parallel_threshold():
    rng = np.random.default_rng(91)
    n = 2051
    time = np.round(rng.exponential(10.0, n), 2) + 0.01
    event = rng.random(n) < 0.7
    event[0] = True
    risk = rng.normal(size=n)
    assert_tuple_close(
        concordance_index_censored(event, time, risk),
        upstream_metrics.concordance_index_censored(event, time, risk),
    )


@pytest.mark.parametrize("tau_quantile", [None, 0.65, 0.85])
def test_ipcw_concordance_parity(survival_data, tau_quantile):
    _, _, time, risk, y = survival_data
    tau = None if tau_quantile is None else np.quantile(time, tau_quantile)
    assert_tuple_close(
        concordance_index_ipcw(y, y, risk, tau=tau),
        upstream_metrics.concordance_index_ipcw(y, y, risk, tau=tau),
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_kaplan_meier_parity(survival_data, reverse):
    _, event, time, _, _ = survival_data
    actual = kaplan_meier_estimator(event, time, reverse=reverse)
    expected = upstream_nonparametric.kaplan_meier_estimator(
        event, time, reverse=reverse
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-15, atol=2e-15)


def test_kaplan_meier_time_min_and_confidence_parity(survival_data):
    _, event, time, _, _ = survival_data
    time_min = np.quantile(time, 0.3)
    actual = kaplan_meier_estimator(
        event, time, time_min=time_min, conf_type="log-log", conf_level=0.9
    )
    expected = upstream_nonparametric.kaplan_meier_estimator(
        event, time, time_min=time_min, conf_type="log-log", conf_level=0.9
    )
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right, rtol=2e-14, atol=2e-14)


def test_left_truncated_kaplan_meier_parity():
    event = np.array([True, False, True, True, False, True])
    enter = np.array([0.0, 1.0, 2.0, 2.0, 4.0, 5.0])
    exit_ = np.array([3.0, 6.0, 7.0, 5.0, 9.0, 8.0])
    actual = kaplan_meier_estimator(event, exit_, time_enter=enter)
    expected = upstream_nonparametric.kaplan_meier_estimator(
        event, exit_, time_enter=enter
    )
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right, rtol=2e-15, atol=2e-15)


def test_nelson_aalen_parity(survival_data):
    _, event, time, _, _ = survival_data
    actual = nelson_aalen_estimator(event, time)
    expected = upstream_nonparametric.nelson_aalen_estimator(event, time)
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right, rtol=2e-15, atol=2e-15)


def metric_inputs(survival_data, n_times=12):
    rng, _, time, _, y = survival_data
    times = np.linspace(time.min(), np.nextafter(time.max(), -np.inf), n_times + 2)[
        1:-1
    ]
    estimate = np.clip(
        rng.normal(0.65, 0.18, size=(len(y), len(times))), 0.01, 0.99
    )
    return y, times, estimate


def test_brier_score_parity(survival_data):
    y, times, estimate = metric_inputs(survival_data, n_times=13)
    actual = brier_score(y, y, estimate, times)
    expected = upstream_metrics.brier_score(y, y, estimate, times)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-15, atol=2e-15)


def test_brier_single_time_1d_parity(survival_data):
    y, times, estimate = metric_inputs(survival_data, n_times=1)
    actual = brier_score(y, y, estimate[:, 0], times)
    expected = upstream_metrics.brier_score(y, y, estimate[:, 0], times)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-15, atol=2e-15)


def test_brier_parallel_threshold_parity():
    rng = np.random.default_rng(92)
    n = 5003
    time = np.round(rng.exponential(10.0, n), 3) + 0.01
    event = rng.random(n) < 0.7
    event[0] = True
    y = Surv.from_arrays(event, time)
    times = np.linspace(np.quantile(time, 0.1), np.quantile(time, 0.9), 200)
    estimate = rng.random((n, len(times)))
    actual = brier_score(y, y, estimate, times)
    expected = upstream_metrics.brier_score(y, y, estimate, times)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-15, atol=2e-15)


def test_integrated_brier_score_parity(survival_data):
    y, times, estimate = metric_inputs(survival_data)
    actual = integrated_brier_score(y, y, estimate, times)
    expected = upstream_metrics.integrated_brier_score(y, y, estimate, times)
    assert actual == pytest.approx(expected, rel=2e-15, abs=2e-15)


def test_dynamic_auc_1d_parity(survival_data):
    _, _, _, risk, y = survival_data
    _, times, _ = metric_inputs(survival_data)
    actual = cumulative_dynamic_auc(y, y, risk, times)
    expected = upstream_metrics.cumulative_dynamic_auc(y, y, risk, times)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-14, atol=3e-14)
    assert actual[1] == pytest.approx(expected[1], rel=3e-14, abs=3e-14)


def test_dynamic_auc_time_dependent_parity(survival_data):
    rng, _, _, _, y = survival_data
    _, times, _ = metric_inputs(survival_data)
    estimate = rng.normal(size=(len(y), len(times)))
    actual = cumulative_dynamic_auc(y, y, estimate, times)
    expected = upstream_metrics.cumulative_dynamic_auc(y, y, estimate, times)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-14, atol=3e-14)
    assert actual[1] == pytest.approx(expected[1], rel=3e-14, abs=3e-14)


def test_dynamic_auc_tolerance_tie_groups():
    event = np.array([True, True, True, False, False, True, False])
    time = np.array([1.0, 2.0, 3.0, 6.0, 7.0, 4.0, 8.0])
    y = Surv.from_arrays(event, time)
    risk = np.array([3.0, 2.0, 2.0 - 0.75e-8, 2.0 - 1.5e-8, 1.0, 0.5, 0.0])
    times = np.array([2.5, 5.0])
    actual = cumulative_dynamic_auc(y, y, risk, times)
    expected = upstream_metrics.cumulative_dynamic_auc(y, y, risk, times)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-14, atol=3e-14)
    assert actual[1] == pytest.approx(expected[1], rel=3e-14, abs=3e-14)


def test_dynamic_auc_parallel_threshold_parity():
    rng = np.random.default_rng(93)
    n = 12_503
    time = np.round(rng.exponential(10.0, n), 3) + 0.01
    event = rng.random(n) < 0.7
    event[0] = True
    y = Surv.from_arrays(event, time)
    times = np.linspace(np.quantile(time, 0.1), np.quantile(time, 0.9), 8)
    estimate = rng.normal(size=(n, len(times)))
    actual = cumulative_dynamic_auc(y, y, estimate, times)
    expected = upstream_metrics.cumulative_dynamic_auc(y, y, estimate, times)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-14, atol=3e-14)
    assert actual[1] == pytest.approx(expected[1], rel=3e-14, abs=3e-14)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_coefficients_and_baseline_parity(survival_data, ties):
    rng, _, _, _, y = survival_data
    X = rng.normal(size=(len(y), 6))
    actual = CoxPHSurvivalAnalysis(ties=ties, alpha=0.1).fit(X, y)
    expected = UpstreamCoxPH(ties=ties, alpha=0.1).fit(X, y)
    np.testing.assert_allclose(actual.coef_, expected.coef_, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(
        actual.unique_times_, expected.unique_times_, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        actual.cum_baseline_hazard_.y,
        expected.cum_baseline_hazard_.y,
        rtol=2e-11,
        atol=2e-12,
    )


def test_cox_prediction_apis_parity(survival_data):
    rng, _, _, _, y = survival_data
    X = rng.normal(size=(len(y), 4))
    query = X[:7]
    actual = CoxPHSurvivalAnalysis().fit(X, y)
    expected = UpstreamCoxPH().fit(X, y)
    np.testing.assert_allclose(
        actual.predict(query), expected.predict(query), rtol=2e-11, atol=2e-12
    )
    np.testing.assert_allclose(
        actual.predict_survival_function(query, return_array=True),
        expected.predict_survival_function(query, return_array=True),
        rtol=2e-11,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        actual.predict_cumulative_hazard_function(query, return_array=True),
        expected.predict_cumulative_hazard_function(query, return_array=True),
        rtol=2e-11,
        atol=2e-12,
    )
    assert actual.score(query, y[:7]) == pytest.approx(expected.score(query, y[:7]))


def test_cox_per_feature_penalty_and_function_objects(survival_data):
    rng, _, _, _, y = survival_data
    X = rng.normal(size=(len(y), 4))
    alpha = np.array([0.0, 0.05, 0.2, 0.8])
    actual = CoxPHSurvivalAnalysis(alpha=alpha).fit(X, y)
    expected = UpstreamCoxPH(alpha=alpha).fit(X, y)
    np.testing.assert_allclose(actual.coef_, expected.coef_, rtol=2e-11, atol=2e-12)
    actual_survival = actual.predict_survival_function(X[:2])
    expected_survival = expected.predict_survival_function(X[:2])
    actual_hazard = actual.predict_cumulative_hazard_function(X[:2])
    expected_hazard = expected.predict_cumulative_hazard_function(X[:2])
    query = actual.unique_times_[:: max(1, len(actual.unique_times_) // 7)]
    for left, right in zip(actual_survival, expected_survival):
        np.testing.assert_allclose(left(query), right(query), rtol=2e-11, atol=2e-12)
    for left, right in zip(actual_hazard, expected_hazard):
        np.testing.assert_allclose(left(query), right(query), rtol=2e-11, atol=2e-12)


def test_cox_rejects_inputs_that_cannot_safely_cross_ffi(survival_data):
    _, _, _, _, y = survival_data
    with pytest.raises(ValueError, match="one feature"):
        CoxPHSurvivalAnalysis().fit(np.empty((len(y), 0)), y)
    with pytest.raises(ValueError, match="finite"):
        CoxPHSurvivalAnalysis(alpha=np.array([np.nan])).fit(
            np.ones((len(y), 1)), y
        )


def test_step_function_matches_upstream_behavior():
    function = StepFunction(
        np.array([1.0, 3.0, 7.0]), np.array([0.9, 0.6, 0.2])
    )
    np.testing.assert_array_equal(function(np.array([0.0, 2.0, 3.0, 6.0])), [0.9, 0.9, 0.6, 0.6])
    assert function(7.0) == 0.2
