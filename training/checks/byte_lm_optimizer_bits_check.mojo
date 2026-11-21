# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Native check of the byte LM resident session's optimizer admission
(training/byte_lm_optimizer_bits.mojo), restoring the signed-zero edge that
tools/byte_lm_session_check.py lost when the trainer's state became a
native owner (lane byte-state; that edge wrote -0.0 into the Python-held
`_state['config']` under an open session, which no Python path can do now).
The admission must tell -0.0 from 0.0 and a NaN payload from another, for
every optimizer scalar, in both the device form (`OptimizerConfig`) and the
host session's words. Host only: no device, no model.

    pixi run mojo build -I . training/checks/byte_lm_optimizer_bits_check.mojo -o optimizer_bits_check && ./optimizer_bits_check
"""
from std.memory import bitcast
from training.checks.optimizer_contract import OptimizerConfig
from training.byte_lm_optimizer_bits import optimizer_config_bits_equal, optimizer_words_bits_equal


def _base() -> OptimizerConfig:
    return OptimizerConfig(2, Float32(0.001), Float32(0.0), Float32(0.999), Float32(1e-8), Float32(0.0),
                           Float32(0.0), Float32(0.0), False, Float32(0.0))


def _with(field: Int, value: Float32) -> OptimizerConfig:
    var c = _base()
    if field == 0:
        c.lr = value
    elif field == 1:
        c.beta1 = value
    elif field == 2:
        c.beta2 = value
    elif field == 3:
        c.eps = value
    elif field == 4:
        c.weight_decay = value
    elif field == 5:
        c.momentum = value
    elif field == 6:
        c.dampening = value
    else:
        c.max_norm = value
    return c^


def run_checks() raises:
    var a = _base()
    var b = _base()
    if not optimizer_config_bits_equal(a, b):
        raise Error("equal optimizer configurations compared unequal")
    var negative_zero = bitcast[DType.float32](UInt32(0x80000000))
    var quiet_nan = bitcast[DType.float32](UInt32(0x7FC00000))
    var other_nan = bitcast[DType.float32](UInt32(0x7FC00001))
    for field in range(8):  # small-loop(the eight optimizer scalars)
        var zero = _with(field, Float32(0.0))
        var signed = _with(field, negative_zero)
        if optimizer_config_bits_equal(zero, signed):
            raise Error("signed zero admitted as equal for optimizer scalar " + String(field))
        if not optimizer_config_bits_equal(signed, _with(field, negative_zero)):
            raise Error("-0.0 refused against itself for optimizer scalar " + String(field))
        if optimizer_config_bits_equal(_with(field, quiet_nan), _with(field, other_nan)):
            raise Error("NaN payloads admitted as equal for optimizer scalar " + String(field))
        if not optimizer_config_bits_equal(_with(field, quiet_nan), _with(field, quiet_nan)):
            raise Error("one NaN refused against itself for optimizer scalar " + String(field))
    var kind = _base()
    kind.kind = 1
    if optimizer_config_bits_equal(a, kind):
        raise Error("a different optimizer kind compared equal")
    var nesterov = _base()
    nesterov.nesterov = True
    if optimizer_config_bits_equal(a, nesterov):
        raise Error("a different nesterov flag compared equal")
    print("PASS device-session optimizer admission by bits (signed zero, NaN payload, kind, nesterov)")

    var words: List[Float64] = [0.001, 0.0, 0.999, 1e-8, 0.0]
    var same: List[Float64] = [0.001, 0.0, 0.999, 1e-8, 0.0]
    if not optimizer_words_bits_equal(words, same):
        raise Error("equal host optimizer words compared unequal")
    var negative_zero64 = bitcast[DType.float64](UInt64(0x8000000000000000))
    for i in range(len(words)):  # small-loop(the host optimizer words)
        if words[i] != 0.0:
            continue
        var signed = words.copy()
        signed[i] = negative_zero64
        if optimizer_words_bits_equal(words, signed):
            raise Error("host session admitted -0.0 for 0.0 at optimizer word " + String(i))
    var shorter: List[Float64] = [0.001, 0.0, 0.999, 1e-8]
    if optimizer_words_bits_equal(words, shorter):
        raise Error("host optimizer words of different lengths compared equal")
    print("PASS host-session optimizer admission by bits (signed zero, length)")


def main() raises:
    run_checks()
