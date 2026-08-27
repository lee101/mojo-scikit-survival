"""Numerical kernels for right-censored survival analysis."""

from max.algorithm import parallelize
from std.math import exp, log
from std.runtime import initialize_runtime
from std.sys import simd_width_of as simdwidthof

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime EventPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IndexPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def p(address: Int) -> Ptr:
    return Ptr(unsafe_from_address=address)


def indices(address: Int) -> IndexPtr:
    return IndexPtr(unsafe_from_address=address)


def events(address: Int) -> EventPtr:
    return EventPtr(unsafe_from_address=address)


def dot(a: Ptr, b: Ptr, n: Int) -> Float64:
    comptime W = simdwidthof[DType.float64]()
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var value = acc.reduce_add()
    while i < n:
        value += a[i] * b[i]
        i += 1
    return value


def concordance_row(
    event: EventPtr,
    time: Ptr,
    estimate: Ptr,
    work: Ptr,
    i: Int,
    n: Int,
    tied_tol: Float64,
):
    var row = work + i * 5
    for q in range(5):
        row[q] = 0.0
    if event[i] == 0:
        return
    comptime W = simdwidthof[DType.float64]()
    var time_i = SIMD[DType.float64, W](time[i])
    var estimate_i = SIMD[DType.float64, W](estimate[i])
    var tolerance = SIMD[DType.float64, W](tied_tol)
    var zero = SIMD[DType.float64, W](0.0)
    var one = SIMD[DType.float64, W](1.0)
    var concordant = SIMD[DType.float64, W](0.0)
    var discordant = SIMD[DType.float64, W](0.0)
    var tied_risk = SIMD[DType.float64, W](0.0)
    var tied_time = SIMD[DType.float64, W](0.0)
    var comparable_count = SIMD[DType.float64, W](0.0)
    var j = 0
    while j + W <= n:
        var times = time.load[width=W](j)
        var event_values = event.load[width=W](j)
        var estimates = estimate.load[width=W](j)
        var same_time = times.eq(time_i)
        var comparable = times.gt(time_i) | (
            same_time & event_values.eq(SIMD[DType.uint8, W](0))
        )
        var ties = comparable & abs(estimates - estimate_i).le(tolerance)
        var concordance = comparable & ~ties & estimates.lt(estimate_i)
        comparable_count += comparable.select(one, zero)
        tied_time += (comparable & same_time).select(one, zero)
        tied_risk += ties.select(one, zero)
        concordant += concordance.select(one, zero)
        discordant += (comparable & ~ties & ~concordance).select(one, zero)
        j += W
    row[0] = concordant.reduce_add()
    row[1] = discordant.reduce_add()
    row[2] = tied_risk.reduce_add()
    row[3] = tied_time.reduce_add()
    row[4] = comparable_count.reduce_add()
    while j < n:
        var comparable = time[j] > time[i] or (
            time[j] == time[i] and event[j] == 0
        )
        if comparable:
            row[4] += 1.0
            if time[j] == time[i]:
                row[3] += 1.0
            var delta = abs(estimate[j] - estimate[i])
            if delta <= tied_tol:
                row[2] += 1.0
            elif estimate[j] < estimate[i]:
                row[0] += 1.0
            else:
                row[1] += 1.0
        j += 1


@export("mss_concordance")
def concordance(
    event_address: Int,
    time_address: Int,
    estimate_address: Int,
    weight_address: Int,
    stats_address: Int,
    work_address: Int,
    n: Int,
    tied_tol: Float64,
) abi("C") -> Float64:
    var event = events(event_address)
    var time = p(time_address)
    var estimate = p(estimate_address)
    var weight = p(weight_address)
    var stats = p(stats_address)
    var work = p(work_address)
    if n >= 2048:
        initialize_runtime()

        @parameter
        def compute_row(i: Int):
            concordance_row(event, time, estimate, work, i, n, tied_tol)

        parallelize[compute_row](n)
    else:
        for i in range(n):
            concordance_row(event, time, estimate, work, i, n, tied_tol)
    var concordant = 0.0
    var discordant = 0.0
    var tied_risk = 0.0
    var tied_time = 0.0
    var numerator = 0.0
    var denominator = 0.0
    for i in range(n):
        var row = work + i * 5
        concordant += row[0]
        discordant += row[1]
        tied_risk += row[2]
        tied_time += row[3]
        numerator += weight[i] * (row[0] + 0.5 * row[2])
        denominator += weight[i] * row[4]
    stats[0] = concordant
    stats[1] = discordant
    stats[2] = tied_risk
    stats[3] = tied_time
    stats[4] = denominator
    return numerator / denominator if denominator > 0.0 else -1.0


@export("mss_product_limit")
def product_limit(
    event_address: Int,
    time_address: Int,
    times_address: Int,
    values_address: Int,
    variances_address: Int,
    n: Int,
    reverse: Int,
    time_min: Float64,
    use_time_min: Int,
) abi("C") -> Int:
    var event = p(event_address)
    var time = p(time_address)
    var times = p(times_address)
    var values = p(values_address)
    var variances = p(variances_address)
    var survival = 1.0
    var count = 0
    var start = 0
    while start < n:
        var end = start + 1
        while end < n and time[end] == time[start]:
            end += 1
        var n_events = 0
        for i in range(start, end):
            if event[i] != 0.0:
                n_events += 1
        var n_at_risk = n - start
        var effective_events = n_events
        if reverse != 0:
            n_at_risk -= n_events
            effective_events = end - start - n_events
        if use_time_min == 0 or time[start] >= time_min:
            var ratio = 0.0
            if effective_events != 0:
                ratio = Float64(effective_events) / Float64(n_at_risk)
            survival *= 1.0 - ratio
            times[count] = time[start]
            values[count] = survival
            if effective_events != 0 and n_at_risk != effective_events:
                variances[count] = Float64(effective_events) / (
                    Float64(n_at_risk) * Float64(n_at_risk - effective_events)
                )
            else:
                variances[count] = 0.0
            count += 1
        start = end
    return count


@export("mss_nelson_aalen")
def nelson_aalen(
    event_address: Int,
    time_address: Int,
    times_address: Int,
    hazard_address: Int,
    n: Int,
) abi("C") -> Int:
    var event = p(event_address)
    var time = p(time_address)
    var times = p(times_address)
    var hazard = p(hazard_address)
    var cumulative = 0.0
    var count = 0
    var start = 0
    while start < n:
        var end = start + 1
        while end < n and time[end] == time[start]:
            end += 1
        var n_events = 0
        for i in range(start, end):
            if event[i] != 0.0:
                n_events += 1
        cumulative += Float64(n_events) / Float64(n - start)
        times[count] = time[start]
        hazard[count] = cumulative
        count += 1
        start = end
    return count


@export("mss_brier_score")
def brier_score(
    event_address: Int,
    time_address: Int,
    estimate_address: Int,
    prob_y_address: Int,
    eval_times_address: Int,
    prob_t_address: Int,
    scores_address: Int,
    n: Int,
    n_times: Int,
) abi("C"):
    var event = events(event_address)
    var time = p(time_address)
    var estimate = p(estimate_address)
    var prob_y = p(prob_y_address)
    var eval_times = p(eval_times_address)
    var prob_t = p(prob_t_address)
    var scores = p(scores_address)

    @parameter
    def compute_time(k: Int):
        var total = 0.0
        for i in range(n):
            var prediction = estimate[i * n_times + k]
            if time[i] <= eval_times[k] and event[i] != 0:
                if prob_y[i] > 0.0:
                    total += prediction * prediction / prob_y[i]
            elif time[i] > eval_times[k]:
                if prob_t[k] > 0.0:
                    var error = 1.0 - prediction
                    total += error * error / prob_t[k]
        scores[k] = total / Float64(n)

    if n * n_times >= 1000000 and n_times > 1:
        initialize_runtime()
        parallelize[compute_time](n_times)
    else:
        comptime W = simdwidthof[DType.float64]()
        var zero = SIMD[DType.float64, W](0.0)
        var k = 0
        while k + W <= n_times:
            scores.store(k, zero)
            k += W
        while k < n_times:
            scores[k] = 0.0
            k += 1
        for i in range(n):
            var time_i = SIMD[DType.float64, W](time[i])
            k = 0
            while k + W <= n_times:
                var predictions = estimate.load[width=W](i * n_times + k)
                var evaluation_times = eval_times.load[width=W](k)
                var probabilities = prob_t.load[width=W](k)
                var contribution = SIMD[DType.float64, W](0.0)
                if event[i] != 0 and prob_y[i] > 0.0:
                    contribution += evaluation_times.ge(time_i).select(
                        predictions * predictions / prob_y[i], zero
                    )
                var after = evaluation_times.lt(time_i) & probabilities.gt(zero)
                var error = 1.0 - predictions
                contribution += after.select(
                    error * error / probabilities, zero
                )
                scores.store(
                    k, scores.load[width=W](k) + contribution
                )
                k += W
            while k < n_times:
                var prediction = estimate[i * n_times + k]
                if time[i] <= eval_times[k] and event[i] != 0:
                    if prob_y[i] > 0.0:
                        scores[k] += prediction * prediction / prob_y[i]
                elif time[i] > eval_times[k]:
                    if prob_t[k] > 0.0:
                        var error = 1.0 - prediction
                        scores[k] += error * error / prob_t[k]
                k += 1
        k = 0
        var count = SIMD[DType.float64, W](Float64(n))
        while k + W <= n_times:
            scores.store(k, scores.load[width=W](k) / count)
            k += W
        while k < n_times:
            scores[k] /= Float64(n)
            k += 1


def dynamic_auc_at(
    event: EventPtr,
    time: Ptr,
    estimate: Ptr,
    ipcw: Ptr,
    eval_times: Ptr,
    order: IndexPtr,
    scores: Ptr,
    n: Int,
    n_times: Int,
    estimate_times: Int,
    k: Int,
    tied_tol: Float64,
):
    var case_weight = 0.0
    var n_controls = 0
    for i in range(n):
        if time[i] <= eval_times[k] and event[i] != 0:
            case_weight += ipcw[i]
        elif time[i] > eval_times[k]:
            n_controls += 1
    var cumulative_cases = 0.0
    var cumulative_controls = 0.0
    var previous_tp = 0.0
    var previous_fp = 0.0
    var area = 0.0
    var order_offset = 0 if estimate_times == 1 else k * n
    var estimate_k = 0 if estimate_times == 1 else k
    for r in range(n):
        var index = Int(order[order_offset + n - 1 - r])
        if time[index] <= eval_times[k] and event[index] != 0:
            cumulative_cases += ipcw[index]
        elif time[index] > eval_times[k]:
            cumulative_controls += 1.0
        var group_end = r + 1 == n
        if not group_end:
            var next_index = Int(order[order_offset + n - 2 - r])
            var score = estimate[index * estimate_times + estimate_k]
            var next_score = estimate[next_index * estimate_times + estimate_k]
            group_end = abs(score - next_score) > tied_tol
        if group_end:
            var true_positive = cumulative_cases / case_weight
            var false_positive = cumulative_controls / Float64(n_controls)
            area += (
                (false_positive - previous_fp)
                * (true_positive + previous_tp)
                * 0.5
            )
            previous_tp = true_positive
            previous_fp = false_positive
    scores[k] = area


@export("mss_dynamic_auc")
def dynamic_auc(
    event_address: Int,
    time_address: Int,
    estimate_address: Int,
    ipcw_address: Int,
    eval_times_address: Int,
    order_address: Int,
    scores_address: Int,
    n: Int,
    n_times: Int,
    estimate_times: Int,
    tied_tol: Float64,
) abi("C"):
    var event = events(event_address)
    var time = p(time_address)
    var estimate = p(estimate_address)
    var ipcw = p(ipcw_address)
    var eval_times = p(eval_times_address)
    var order = indices(order_address)
    var scores = p(scores_address)

    @parameter
    def compute_time(k: Int):
        dynamic_auc_at(
            event,
            time,
            estimate,
            ipcw,
            eval_times,
            order,
            scores,
            n,
            n_times,
            estimate_times,
            k,
            tied_tol,
        )

    if n * n_times >= 100000 and n_times > 1:
        initialize_runtime()
        parallelize[compute_time](n_times)
    else:
        for k in range(n_times):
            compute_time(k)


@export("mss_cox_evaluate")
def cox_evaluate(
    x_address: Int,
    event_address: Int,
    time_address: Int,
    coef_address: Int,
    alpha_address: Int,
    gradient_address: Int,
    hessian_address: Int,
    vector_work_address: Int,
    matrix_work_address: Int,
    exp_work_address: Int,
    n: Int,
    d: Int,
    ties: Int,
) abi("C") -> Float64:
    var x = p(x_address)
    var event = p(event_address)
    var time = p(time_address)
    var coef = p(coef_address)
    var alpha = p(alpha_address)
    var gradient = p(gradient_address)
    var hessian = p(hessian_address)
    var vector_work = p(vector_work_address)
    var matrix_work = p(matrix_work_address)
    var exp_work = p(exp_work_address)
    var risk_x = vector_work
    var event_x = vector_work + d
    var event_risk_x = vector_work + 2 * d
    var risk_xx = matrix_work
    var event_risk_xx = matrix_work + d * d
    for j in range(d):
        gradient[j] = 0.0
        risk_x[j] = 0.0
    for q in range(d * d):
        hessian[q] = 0.0
        risk_xx[q] = 0.0
    for i in range(n):
        exp_work[i] = exp(dot(x + i * d, coef, d))
    var risk_set = 0.0
    var loss = 0.0
    var start = 0
    while start < n:
        var end = start + 1
        while end < n and time[end] == time[start]:
            end += 1
        for j in range(d):
            event_x[j] = 0.0
            event_risk_x[j] = 0.0
        for q in range(d * d):
            event_risk_xx[q] = 0.0
        var event_risk = 0.0
        var event_linear = 0.0
        var n_events = 0
        for i in range(start, end):
            var score = exp_work[i]
            if event[i] != 0.0:
                n_events += 1
                event_risk += score
                event_linear += dot(x + i * d, coef, d)
                for a in range(d):
                    event_x[a] += x[i * d + a]
                    event_risk_x[a] += score * x[i * d + a]
                    for b in range(d):
                        event_risk_xx[a * d + b] += (
                            score * x[i * d + a] * x[i * d + b]
                        )
            else:
                risk_set += score
                for a in range(d):
                    risk_x[a] += score * x[i * d + a]
                    for b in range(d):
                        risk_xx[a * d + b] += (
                            score * x[i * d + a] * x[i * d + b]
                        )
        if n_events > 0:
            var repetitions = 1 if ties == 0 else n_events
            for repetition in range(repetitions):
                var fraction = 1.0 if ties == 0 else 1.0 / Float64(n_events)
                risk_set += fraction * event_risk
                for a in range(d):
                    risk_x[a] += fraction * event_risk_x[a]
                    for b in range(d):
                        risk_xx[a * d + b] += fraction * event_risk_xx[a * d + b]
                var multiplier = Float64(n_events) if ties == 0 else 1.0
                var numerator_scale = 1.0 if ties == 0 else 1.0 / Float64(n_events)
                loss -= (
                    numerator_scale * event_linear - multiplier * log(risk_set)
                ) / Float64(n)
                for a in range(d):
                    var mean_a = risk_x[a] / risk_set
                    gradient[a] -= (
                        numerator_scale * event_x[a] - multiplier * mean_a
                    ) / Float64(n)
                    for b in range(d):
                        var mean_b = risk_x[b] / risk_set
                        hessian[a * d + b] += multiplier * (
                            risk_xx[a * d + b] / risk_set - mean_a * mean_b
                        ) / Float64(n)
        start = end
    for j in range(d):
        loss += alpha[j] * coef[j] * coef[j] / (2.0 * Float64(n))
        gradient[j] += alpha[j] * coef[j] / Float64(n)
        hessian[j * d + j] += alpha[j] / Float64(n)
    return loss


@export("mss_cox_baseline")
def cox_baseline(
    event_address: Int,
    time_address: Int,
    risk_address: Int,
    times_address: Int,
    hazard_address: Int,
    n: Int,
) abi("C") -> Int:
    var event = p(event_address)
    var time = p(time_address)
    var risk = p(risk_address)
    var times = p(times_address)
    var hazard = p(hazard_address)
    var risk_set = 0.0
    for i in range(n):
        risk_set += risk[i]
    var cumulative = 0.0
    var count = 0
    var start = 0
    while start < n:
        var end = start + 1
        while end < n and time[end] == time[start]:
            end += 1
        var n_events = 0
        for i in range(start, end):
            if event[i] != 0.0:
                n_events += 1
        cumulative += Float64(n_events) / risk_set
        times[count] = time[start]
        hazard[count] = cumulative
        count += 1
        for i in range(start, end):
            risk_set -= risk[i]
        start = end
    return count


@export("mss_linear_predict")
def linear_predict(
    x_address: Int,
    coef_address: Int,
    result_address: Int,
    n: Int,
    d: Int,
) abi("C"):
    var x = p(x_address)
    var coef = p(coef_address)
    var result = p(result_address)
    for i in range(n):
        result[i] = dot(x + i * d, coef, d)
