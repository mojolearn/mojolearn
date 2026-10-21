# SPDX-License-Identifier: Apache-2.0
"""Typed element loops behind the CausalLM construction shell (bindings/causal_lm_construct.mojo)."""
from std.math import isfinite


def causal_lm_gather_f32(
    source: MutPointer[Float32, MutUntrackedOrigin], target: MutPointer[Float32, MutUntrackedOrigin],
    size: Int, dims: List[Int], strides: List[Int],
):
    """Copy a strided float32 tensor into C order; strides are in bytes."""
    for i in range(size):
        var remaining = i
        var source_index = 0
        for j in range(len(dims)):
            var axis = len(dims) - 1 - j
            var coordinate = remaining % dims[axis]
            remaining //= dims[axis]
            source_index += coordinate * strides[axis] // 4
        target[i] = source[source_index]


def causal_lm_check_finite(ptr: MutPointer[Float32, MutUntrackedOrigin], n: Int) raises:
    for i in range(n):
        if not isfinite(ptr[i]):
            raise Error("CausalLM: top-level weight is not finite")
