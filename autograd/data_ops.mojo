# SPDX-License-Identifier: Apache-2.0
# Element bodies of the data-side autograd family: dropout forward and VJP.
# Shared by HostExec and DeviceExec through sequence/dispatch.mojo.
#
# Randomness is the neural stream of core/philox_neural.mojo, reused as is:
# element e of a dropout at (seed, counter) reads the uniform word of
# element e of stream `counter` keyed by `seed`,
#     counter block = (e // 4 low, e // 4 high, counter, 0), key = (seed lo, seed hi),
# word e % 4, mapped to [0, 1) exactly as (x >> 8) * 2^-24. Keep the element
# when that coin is >= p, scaled by s = 1 / (1 - p); drop it to +0.0
# otherwise. Integers and one rounded multiply per element: the same mask
# and the same bits on Apple, NVIDIA, AMD and the host, and the same values
# as neural_rng's DROPOUT arms for the same (seed, stream, p, s).
#
# The VJP re-derives the mask from the same counter instead of storing it:
# no mask buffer lives across forward and backward, and nothing has to be
# kept in sync between them.
from sequence.ops import Args, ld, st, mul
from autograd.ops import AG_VEC, F4, ld4, st4
from core.philox_neural import neural_unit_at

# Codes OP_AGD_FIRST..OP_AGD_LAST are this family. They sit apart from the
# core tape family (autograd/ops.mojo, 1000..1010) so either range can grow
# without renumbering the other; sequence/dispatch.mojo asserts both ranges
# are disjoint from each other and from the recurrent scan codes.
comptime OP_AG_DROPOUT_FWD = 1100
comptime OP_AG_DROPOUT_VJP = 1101
comptime OP_AGD_FIRST = OP_AG_DROPOUT_FWD
comptime OP_AGD_LAST = OP_AG_DROPOUT_VJP

# Tape node kind of a dropout. Below CUSTOM (40, autograd/ops.mojo) and
# apart from the kinds there (0..30).
comptime DROPOUT = 35


def dropout_args(seed: UInt64, counter: Int, p: Float32, scale: Float32) -> Args:
    """The dropout bodies' scalar slots: i0..i3 the 64-bit seed as 16-bit
    words, low first; i4, i5 the 32-bit counter as 16-bit words; f0 p, f1
    the keep scale. Sixteen-bit words keep every integer slot inside the
    device executor's Int32 range (sequence/exec_device.mojo launch)."""
    var a = Args()
    a.i0 = Int(seed & UInt64(0xFFFF))
    a.i1 = Int((seed >> UInt64(16)) & UInt64(0xFFFF))
    a.i2 = Int((seed >> UInt64(32)) & UInt64(0xFFFF))
    a.i3 = Int((seed >> UInt64(48)) & UInt64(0xFFFF))
    a.i4 = counter & 0xFFFF
    a.i5 = (counter >> 16) & 0xFFFF
    a.f0 = p
    a.f1 = scale
    return a


@always_inline
def _keep(t: Int, a: Args) -> Bool:
    """Element t's coin against p (a.f0), from the slots of dropout_args."""
    var seed_lo = UInt32(a.i0) | (UInt32(a.i1) << UInt32(16))
    var seed_hi = UInt32(a.i2) | (UInt32(a.i3) << UInt32(16))
    var stream = UInt32(a.i4) | (UInt32(a.i5) << UInt32(16))
    return neural_unit_at(seed_lo, seed_hi, stream, t) >= a.f0


@always_inline
def op_ag_dropout_fwd(t: Int, a: Args):
    """p0 x, p1 y. Slots of dropout_args. Thread t = element t:
    y = x * s where the coin keeps t, else +0.0."""
    var y = Float32(0)
    if _keep(t, a):
        y = mul(ld(a.p0, t), a.f1)
    st(a.p1, t, y)


@always_inline
def op_ag_dropout_vjp(t: Int, a: Args):
    """p0 upstream, p1 destination. Slots of dropout_args (the forward's).
    Thread t = element t: dx = dy * s where the coin keeps t, else +0.0."""
    var g = Float32(0)
    if _keep(t, a):
        g = mul(ld(a.p0, t), a.f1)
    st(a.p1, t, g)


@always_inline
def _dropout4(t0: Int, a: Args):
    """op_ag_dropout_fwd / op_ag_dropout_vjp for elements t0 .. t0 + 3 (the
    same coin per element, the same multiply on the same flushed word; one
    16-byte load and store, lane ag-pass6-fold idea E). p0 the source, p1
    the destination in both bodies."""
    var x = ld4(a.p0, t0)
    var y = F4(0)
    comptime for k in range(AG_VEC):
        if _keep(t0 + k, a):
            y[k] = mul(x[k], a.f1)
    st4(a.p1, t0, y)


@always_inline
def ag_data_vec4_op[OP: Int]() -> Bool:
    """The OP_AGD_* codes with a four-element body."""
    return OP == OP_AG_DROPOUT_FWD or OP == OP_AG_DROPOUT_VJP


@always_inline
def apply_autograd_data_vec4[OP: Int](t0: Int, a: Args):
    """Elements t0 .. t0 + 3 of OP (ag_data_vec4_op[OP]())."""
    comptime if OP == OP_AG_DROPOUT_FWD or OP == OP_AG_DROPOUT_VJP:
        _dropout4(t0, a)


@always_inline
def apply_autograd_data[OP: Int](t: Int, a: Args):
    comptime if OP == OP_AG_DROPOUT_FWD:
        op_ag_dropout_fwd(t, a)
    elif OP == OP_AG_DROPOUT_VJP:
        op_ag_dropout_vjp(t, a)
