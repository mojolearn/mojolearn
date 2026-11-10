# SPDX-License-Identifier: Apache-2.0
"""Native layout of the existing QN fitted coefficients and public views."""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk


def logistic_target_count(classes: Int) -> Int:
    return 1 if classes == 2 else classes


def logistic_fit_target_count(classes: Int) raises -> Int:
    if classes < 2:
        raise Error("mojolearn LogisticRegression: y has one class")
    return logistic_target_count(classes)


struct LogisticFitLayout(Copyable, Movable):
    var features: Int
    var targets: Int
    var intercept: Bool
    var parameters: Int
    var weights: Int
    var coefficients: Int

    def __init__(out self, features: Int, targets: Int, intercept: Bool) raises:
        # Every public word is float32. Validate before native products or
        # allocations, including zero-feature fits with an intercept.
        var maximum = 9223372036854775807 // 4
        if features < 0 or targets < 1 or features > maximum - Int(intercept):
            raise Error("ValueError: LogisticRegression coefficient layout exceeds native storage")
        self.features = features
        self.targets = targets
        self.intercept = intercept
        self.parameters = features + Int(intercept)
        if self.parameters != 0 and targets > maximum // self.parameters:
            raise Error("ValueError: LogisticRegression coefficient layout exceeds native storage")
        self.weights = self.parameters * targets
        self.coefficients = features * targets


def _logistic_state_span(address: Int, count: Int) raises:
    if count < 0 or count > 9223372036854775807 // 4:
        raise Error("ValueError: LogisticRegression coefficient span exceeds native storage")
    if count != 0 and (address <= 0 or address % 4 != 0 or address > 9223372036854775807 - count * 4):
        raise Error("ValueError: LogisticRegression coefficient storage must be an aligned float32 span")


def logistic_publish_coefficients(layout: LogisticFitLayout, weights: Int,
                                  coefficients: Int, intercepts: Int) raises:
    """Pack the already-computed QN host result into its public buffers.

    The QN kernel and its weight layout stay unchanged. Independent output
    words use the existing host work policy; no reduction or numerical
    arithmetic occurs. All owners remain with PACK until the join completes.
    """
    _logistic_state_span(weights, layout.weights)
    _logistic_state_span(coefficients, layout.coefficients)
    _logistic_state_span(intercepts, layout.targets)
    var count = layout.coefficients
    var targets = layout.targets
    var features = layout.features
    var intercept = layout.intercept
    var total = count + targets
    var tasks = host_predict_task_count_work(total, total)
    var chunk = host_predict_chunk(total, tasks)
    var source = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=weights)
    var coefficient_words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=coefficients)
    var intercept_words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=intercepts)

    def task(worker: Int) {imm count, imm targets, imm features, imm intercept,
            imm total, imm chunk, imm source, imm coefficient_words, imm intercept_words}:
        var begin = worker * chunk
        var end = min(begin + chunk, total)
        # A worker computes its first coordinates once, then advances through
        # each coefficient row. No per-cell division/modulo or staging copy.
        var at = begin
        if at < count:
            var target = at // features
            var feature = at - target * features
            while at < min(end, count):
                coefficient_words[at] = source[target + targets * feature]
                at += 1
                feature += 1
                if feature == features:
                    feature = 0
                    target += 1
        while at < end:
            var target = at - count
            intercept_words[target] = source[target + count] if intercept else UInt32(0)
            at += 1

    host_parallelize(task, tasks)
