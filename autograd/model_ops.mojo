# SPDX-License-Identifier: Apache-2.0
# Element bodies of the model-description trainer (autograd/model.mojo),
# shared by HostExec and DeviceExec through sequence/dispatch.mojo.
#
# Codes OP_AGM_FIRST..OP_AGM_LAST (1040..1042) are this family. They sit
# above the autograd tape range (autograd/ops.mojo, 1000..1010) with room for
# that family to grow; sequence/dispatch.mojo asserts the two ranges and the
# recurrent scan codes do not collide.
from sequence.ops import Args, FP, ld, st, mul
from core.philox import philox4x32_10

comptime OP_AGM_INIT_UNIFORM = 1040
comptime OP_AGM_ROW_ARGMAX = 1041
comptime OP_AGM_THRESHOLD = 1042
comptime OP_AGM_FIRST = OP_AGM_INIT_UNIFORM
comptime OP_AGM_LAST = OP_AGM_THRESHOLD

# Third counter word of every parameter-initialisation draw ("MODE"), so the
# initialisation stream never shares a counter with another Philox user that
# keys on the same seed (shuffle uses its own Feistel key, dropout its own
# counters).
comptime AGM_INIT_STREAM: UInt32 = 0x4D4F4445


@always_inline
def agm_seed_word(a: Args, w: Int) -> UInt32:
    """16-bit word w (0 = lowest) of the 64-bit seed, carried in i3..i6 so
    every integer slot fits Int32 on the device."""
    if w == 0:
        return UInt32(a.i3)
    if w == 1:
        return UInt32(a.i4)
    if w == 2:
        return UInt32(a.i5)
    return UInt32(a.i6)


@always_inline
def op_agm_init_uniform(t: Int, a: Args):
    """p0[t] = f0 * (2 u - 1), u = (word >> 8) / 2^24, word = Philox4x32-10
    word 0 at counter (i2 + t, 0, AGM_INIT_STREAM, 0) under the 64-bit seed
    (16-bit words i3..i6). i2 is the element's index in the flat parameter
    vector, so every parameter's draw depends on the seed and its position
    alone, never on the launch shape or the vendor. 2 u - 1 is exact in
    float32 (a 24-bit integer times 2^-24); the one rounding is the product
    with the bound. Thread t = element t."""
    var key = SIMD[DType.uint32, 2](
        agm_seed_word(a, 0) | (agm_seed_word(a, 1) << 16),
        agm_seed_word(a, 2) | (agm_seed_word(a, 3) << 16),
    )
    var ctr = SIMD[DType.uint32, 4](UInt32(a.i2 + t), 0, AGM_INIT_STREAM, 0)
    var word = philox4x32_10(ctr, key)[0]
    var k = Int(word >> 8)
    var centred = Float32(2 * k - 16777216) * Float32(5.9604644775390625e-08)
    st(a.p0, t, mul(centred, a.f0))


@always_inline
def op_agm_row_argmax(t: Int, a: Args):
    """p1[t] = the column of row t's largest score in p0 (rows of i0
    columns) as a float32 code: first max wins (strict >, from column 0),
    so ties keep the lowest column and a NaN never replaces. Thread t =
    row t, O(cols) each."""
    var cols = a.i0
    var base = t * cols
    var best = ld(a.p0, base)
    var at = 0
    for c in range(1, cols):
        var v = ld(a.p0, base + c)
        if v > best:
            best = v
            at = c
    st(a.p1, t, Float32(at))


@always_inline
def op_agm_threshold(t: Int, a: Args):
    """p1[t] = 1.0 when p0[t] > f0 else 0.0 (a NaN takes 0). Thread t =
    element t."""
    var v = Float32(0)
    if ld(a.p0, t) > a.f0:
        v = Float32(1)
    st(a.p1, t, v)


@always_inline
def apply_model_ops[OP: Int](t: Int, a: Args):
    comptime if OP == OP_AGM_INIT_UNIFORM:
        op_agm_init_uniform(t, a)
    elif OP == OP_AGM_ROW_ARGMAX:
        op_agm_row_argmax(t, a)
    elif OP == OP_AGM_THRESHOLD:
        op_agm_threshold(t, a)
