# SPDX-License-Identifier: Apache-2.0
"""Typed optimizer configuration and checkpoint values for OpsSession.

Packed integer/float lists remain adapters for the existing native API. Public
Python boundaries decode directly into these values; no Python object or owner
crosses the native optimizer operation.
"""


comptime OPS_OPT_INTS = 4    # kind, nesterov, schedule, step size
comptime OPS_OPT_FLOATS = 7  # lr, momentum, beta1, beta2, eps, weight_decay, gamma


@fieldwise_init
struct OpsOptimizerRequest(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var nesterov: Bool
    var schedule: Int
    var step_size: Int
    var lr: Float32
    var momentum: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32
    var gamma: Float32
    var dampening: Float32


def ops_optimizer_codes(kind: String, schedule: String) raises -> Tuple[Int, Int]:
    var kind_code = 0 if kind == "sgd" else (1 if kind == "adam" else (2 if kind == "adamw" else -1))
    var schedule_code = 0 if schedule == "constant" else (1 if schedule == "step" else (2 if schedule == "exponential" else -1))
    if kind_code < 0 or schedule_code < 0:
        raise Error("ValueError: invalid optimizer kind or schedule")
    return (kind_code, schedule_code)


def ops_optimizer_request(ints: List[Int], fs: List[Float32]) raises -> OpsOptimizerRequest:
    if len(ints) != OPS_OPT_INTS or (len(fs) != OPS_OPT_FLOATS and len(fs) != OPS_OPT_FLOATS + 1):
        raise Error("ops: optimizer takes 4 integers and 7 floats (8 with SGD dampening)")
    return OpsOptimizerRequest(ints[0], ints[1] != 0, ints[2], ints[3],
        fs[0], fs[1], fs[2], fs[3], fs[4], fs[5], fs[6], fs[OPS_OPT_FLOATS] if len(fs) == OPS_OPT_FLOATS + 1 else Float32(0))


@fieldwise_init
struct OpsOptimizerStateValue(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var step: Int
    var beta1_power: Float32
    var beta2_power: Float32
    var mu_product: Float32
    var first: Int
    var second: Int
    var third: Int


@fieldwise_init
struct OpsOptimizerRestoreRequest(Copyable, ImplicitlyCopyable, Movable):
    var step: Int
    var first: Int
    var second: Int
    var third: Int
    var beta1_power: Float32
    var beta2_power: Float32
    var mu_product: Float32


def ops_optimizer_restore_request(ints: List[Int], fs: List[Float32]) raises -> OpsOptimizerRestoreRequest:
    if len(ints) != 4 or len(fs) != 3:
        raise Error("ops: load_optimizer_state takes 4 integers and 3 floats")
    return OpsOptimizerRestoreRequest(ints[0], ints[1], ints[2], ints[3], fs[0], fs[1], fs[2])
