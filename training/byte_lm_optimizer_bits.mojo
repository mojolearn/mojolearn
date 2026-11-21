# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The byte LM resident session's optimizer admission: every optimizer
scalar compared BY BITS, so a signed zero (`-0.0` against `0.0`) or a NaN
payload is a mismatch, never an equal value (lane fx-leftovers item 5,
2026-11-21). One spelling for the device session (`OptimizerConfig`,
training/byte_lm_binding_impl.mojo `_same_optimizer_bits` and
training/byte_lm_stateless_run.mojo, which repeated it inline) and the host
session (its `List[Float64]` words, training/byte_lm_host_session.mojo
`admit`, which compared with `!=` and so admitted `-0.0` for `0.0`).
Checked natively by training/checks/byte_lm_optimizer_bits_check.mojo."""
from std.memory import bitcast
from training.checks.optimizer_contract import OptimizerConfig


@always_inline
def _f32_bits_equal(a: Float32, b: Float32) -> Bool:
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


@always_inline
def optimizer_config_bits_equal(a: OptimizerConfig, b: OptimizerConfig) -> Bool:
    """Bitwise equality of every optimizer scalar, signed zero included."""
    return (a.kind == b.kind and a.nesterov == b.nesterov
        and _f32_bits_equal(a.lr, b.lr)
        and _f32_bits_equal(a.beta1, b.beta1)
        and _f32_bits_equal(a.beta2, b.beta2)
        and _f32_bits_equal(a.eps, b.eps)
        and _f32_bits_equal(a.weight_decay, b.weight_decay)
        and _f32_bits_equal(a.momentum, b.momentum)
        and _f32_bits_equal(a.dampening, b.dampening)
        and _f32_bits_equal(a.max_norm, b.max_norm))


def optimizer_words_bits_equal(a: List[Float64], b: List[Float64]) -> Bool:
    """The host session's optimizer words, compared by bits (signed zero and
    NaN payload included); a length difference is a mismatch."""
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(optimizer settings: a fixed handful of scalars)
        if bitcast[DType.uint64](a[i]) != bitcast[DType.uint64](b[i]):
            return False
    return True
