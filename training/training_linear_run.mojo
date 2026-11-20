# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/training_linear_native.mojo` (wave 8 run split): the
linear layer's dimension checks, typed Mojo only. Imports no Python
(tools/python_free_dirs.sh); body moved unchanged (lane
rs-sequence-tokenizer, 2026-11-12). `shape_count` is the binding's alias of
training/optimizer_device_storage_plan.mojo `optimizer_device_count`."""
from training.optimizer_device_storage_plan import optimizer_device_count as shape_count


def linear_dimensions(a: List[Int], weight: List[Int], grad: List[Int], backward: Bool) raises -> List[Int]:
    if len(a) < 1 or len(weight) != 2:
        raise Error("ValueError: linear layer input must have a last feature axis and weight must be a matrix")
    for i in range(len(a)):  # driver-loop: over input shape words before indexing/allocation
        if a[i] < 0:
            raise Error("ValueError: linear layer dimensions must be nonnegative")
    if weight[0] < 0 or weight[1] < 0:
        raise Error("ValueError: linear layer dimensions must be nonnegative")
    var k = a[len(a) - 1]
    if k == 0:
        raise Error("ValueError: linear layer cannot infer rows with a zero feature dimension")
    if weight[1] != k:
        var name = String("linear_backward") if backward else String("linear_forward")
        raise Error("ValueError: mojolearn." + name + ": a has K=" + String(k) + ", weight has K=" + String(weight[1]))
    var m = shape_count(a) // k
    var n = weight[0]
    if m == 0 or n == 0:
        raise Error("ValueError: linear layer inputs must be nonempty")
    _ = shape_count(weight)
    if backward:
        if len(grad) < 1:
            raise Error("ValueError: mojolearn.linear_backward: dc must be (M, N)")
        for i in range(len(grad)):  # driver-loop: over gradient shape words before counting
            if grad[i] < 0:
                raise Error("ValueError: linear gradient dimensions must be nonnegative")
        if grad[len(grad) - 1] != n or shape_count(grad) != shape_count([m, n]):
            raise Error("ValueError: mojolearn.linear_backward: dc must be (M, N)")
    return [m, n, k]
