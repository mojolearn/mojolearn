# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The contract half of profile `mojolearn.identical.optimizer.fp32.v1` that
the device optimizer and the host oracle share: the algorithm ids, the bit
constants, the refusals, the configuration, the step scalars and the
microbatch alignment predicate. Moved out of
`training/checks/optimizer_oracle.mojo` (which imports them back) so the
device training modules import the contract without linking the CPU oracle
(cpu-gpu-cleanup n-train-mamba, 2026-10-02). `refuse_nonfinite` here is the
plain ascending scan; the host-lane fast path it had in the oracle is gone
(a scan, same refusals, same messages)."""

from gemm.contract import OP_NT, contract_leaf_size
from gemm.experiments.neural_profile import NEURAL_PROFILE_CHANGED,NEURAL_LEAF,neural_partition
from checks.numerics import (
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)


comptime OPT_SGD = 0
comptime OPT_ADAM = 1
comptime OPT_ADAMW = 2

comptime CLIP_EPS_BITS: UInt32 = 0x358637BD

comptime ONE_BITS: UInt32 = 0x3F800000

comptime QNAN_BITS: UInt32 = 0x7FC00000

comptime INF_BITS: UInt32 = 0x7F800000


def clip_eps() -> Float32:
    """The profile's `CLIP_EPS`, from its bit pattern rather than from a decimal literal (`[[mojo-string-float-roundtrip]]`)."""
    from std.memory import bitcast

    return bitcast[DType.float32](CLIP_EPS_BITS)




def opt_nonfinite_message(name: String, index: Int, is_nan: Bool) -> String:
    """THE ONE SPELLING of this profile's non-finite refusal (DEVIATION
    2514 step 3). `refuse_nonfinite` below raises through it over a host
    List, and `training/checks/optimizer.mojo::opt_refuse_device_inputs`
    raises through it with the index `device_first_nonfinite` returned, so
    the host and device paths cannot produce two messages for one defect.
    The two strings are byte for byte the ones `refuse_nonfinite` built
    inline before this function existed; optimizer_check clause (f) asserts
    the device message EQUALS the host message on the same planted List."""
    if is_nan:
        return (
            String("optimizer: NaN in ")
            + name
            + String(" at flat index ")
            + String(index)
            + String(" REFUSED (row 39: NaN payloads are vendor-shaped")
            + String("; no stage may record one)")
        )
    return (
        String("optimizer: infinity in ")
        + name
        + String(" at flat index ")
        + String(index)
        + String(" REFUSED (row 39)")
    )


def refuse_nonfinite(name: String, values: List[Float32]) raises:
    """Row 39. A NaN or infinity in a gradient, a parameter or a state is REFUSED BY NAME before any recorded stage. The message is `opt_nonfinite_message`'s (DEVIATION 2514)."""
    from std.memory import bitcast

    for i in range(len(values)):
        var au = bitcast[DType.uint32](values[i]) & UInt32(0x7FFFFFFF)
        if au > INF_BITS:
            raise Error(opt_nonfinite_message(name, i, True))
        if au == INF_BITS:
            raise Error(opt_nonfinite_message(name, i, False))


def refuse_nonfinite_scalar(name: String, v: Float32) raises:
    """`refuse_nonfinite` for one value."""
    var one = List[Float32]()
    one.append(v)
    refuse_nonfinite(name, one)




def pow_int_f32(base: Float32, t: Int) -> Float32:
    """DEVIATION 1171. What it refuses is `b_t = b_{t-1} * beta`, which is what an implementation reaches for because it is one multiply per step."""
    from std.memory import bitcast

    var acc = bitcast[DType.float32](ONE_BITS)
    var b = base
    var e = t
    while e > 0:
        if (e & 1) != 0:
            acc = ftz(identical_mul(acc, b))
        b = ftz(identical_mul(b, b))
        e = e >> 1
    return acc




@fieldwise_init
struct OptimizerConfig(Copyable, Movable):
    """One parameter group's configuration. `max_norm <= 0.0` means the gradient-norm clip is OFF, and that is the configuration contract 11(c)'s parameter-count-invariance gate must run under -- with clipping ON, one parameter's update depends on every other parameter in the model, by the reference's own semantics and not by a defect (contract 3.5)."""

    var kind: Int
    var lr: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32
    var momentum: Float32
    var dampening: Float32
    var nesterov: Bool
    var max_norm: Float32


@fieldwise_init
struct StepScalars(Copyable, Movable):
    """The host scalars of contract 7.1, computed ONCE per step per parameter group."""

    var b1t: Float32
    var b2t: Float32
    var bc1: Float32
    var bc2: Float32
    var step_size: Float32
    var rt_bc2: Float32
    var c1: Float32
    var c2: Float32
    var decay_mul: Float32
    var neg_lr: Float32
    var c_damp: Float32


def step_scalars(cfg: OptimizerConfig, t: Int) -> StepScalars:
    """Contract 7.1, line for line."""
    from std.memory import bitcast

    var one = bitcast[DType.float32](ONE_BITS)

    var b1t = pow_int_f32(cfg.beta1, t)
    var b2t = pow_int_f32(cfg.beta2, t)
    var bc1 = ftz(one - b1t)
    var bc2 = ftz(one - b2t)
    var step_size = ftz(identical_div(cfg.lr, bc1))
    var rt_bc2 = ftz(identical_sqrt(bc2))
    var c1 = ftz(one - cfg.beta1)
    var c2 = ftz(one - cfg.beta2)
    var decay_mul = ftz(one - ftz(identical_mul(cfg.lr, cfg.weight_decay)))
    var neg_lr = -cfg.lr
    var c_damp = ftz(one - cfg.dampening)

    return StepScalars(
        b1t,
        b2t,
        bc1,
        bc2,
        step_size,
        rt_bc2,
        c1,
        c2,
        decay_mul,
        neg_lr,
        c_damp,
    )




def microbatch_split_is_identical(t_tokens: Int, a: Int) -> Bool:
    """Contract clause 9.2. The cross-microbatch combination must be the v1 BALANCED TREE over the `a` pieces in ascending microbatch index, `ftz(ftz(x) + ftz(y))` at every node -- **not a running serial sum.** Conditions 3 and 4 together are what make each microbatch's leaf range a COMPLETE SUBTREE."""
    if a <= 0 or t_tokens <= 0:
        return False
    if t_tokens % a != 0:
        return False
    var leaf_full = contract_leaf_size(t_tokens)
    var leaf_piece = contract_leaf_size(t_tokens // a)
    comptime if NEURAL_PROFILE_CHANGED:
        # Partition compatibility belongs to this version's neural profile;
        # the old 128-term graph cannot certify a new 64/256-term profile.
        leaf_full = neural_partition[NEURAL_LEAF](t_tokens)[0]
        leaf_piece = neural_partition[NEURAL_LEAF](t_tokens // a)[0]
    if leaf_full != leaf_piece:
        return False
    if leaf_full <= 0:
        return False
    if t_tokens % leaf_full != 0:
        return False
    var p_count = t_tokens // leaf_full
    if p_count % a != 0:
        return False
    var q = a
    while q > 1:
        if q % 2 != 0:
            return False
        q = q // 2
    return True



