# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 612: the card records ONE NaN payload, not the vendor's.

NOT A PORT. cuML has no card.

==========================================================================
DEVIATION 612 -- NaN canonicalization at the solver's record sites
==========================================================================
WHAT THEIRS DOES: `cdFit` has no finiteness guard anywhere (`cd.cuh:115-
274`): a non-finite input, or an overflow of `x * residual` past FLT_MAX
(`inf - inf` in a dot, `(-inf) * x + inf` in the second axpy), produces a
NaN that propagates through `coef`, `residual` and `ConvState`, and the
soft threshold turns it back into a `0` coefficient (`coef > l1_alpha` and
`coef < -l1_alpha` are both false for NaN). cuML ships one backend, so it
never had to say WHICH NaN.

WHAT THIS SECTION DOES: every float stage of the card (`cd.input.x/y`,
`cd.mu_input/mu_labels`, `cd.colnorm/squared`, `cd.sweepNNN.coef/resid/
conv`, `cd.final.coef`, `cd.intercept`) is recorded through this file,
which copies the buffer, rewrites every NaN (either sign, any payload) to
the quiet NaN `0x7FC00000` -- the bit pattern `mojo_only/numerics.mojo`'s
`identical_log`/`identical_sqrt` already write -- and hashes the copy. The
LIVE buffers are untouched: the trace observes and never mutates.

WHY: IDENTITY_PATHS row 39, measured 2026-08-23 on three vendors: a
COMPUTED NaN carries the vendor's payload (Apple 0x7fc00000, NVIDIA
0x7fffffff, AMD 0xffc00000, the host's is its CPU's), so a raw-bit hash of
a stage holding one is a per-vendor fingerprint, not an identity, and a
NaN payload steers no downstream bit (every consumer of a NaN here is a
comparison, which is payload-blind, or an arithmetic op that yields a NaN
again). Host-side scalars (`cd.intercept`) go through `canon_nan_f32` for
the same reason; the oracle card in `solver/mojo_only/cd_check.mojo` goes
through `canon_nan_list`, so host-vs-device card agreement on a NaN stage
is a two-payload test on one machine whenever the host and the device
disagree on their default NaN.

`cd.l1_alpha` / `cd.l2_alpha` are NOT routed here: after DEVIATION 613
(`alpha` finite, `l1_ratio` and `tol` not NaN, `cd.mojo`) they are products
of finite operands and can overflow to +inf but never be NaN.

THE GATE: `check_cd_nan_payload_is_canonical` (cd_check) plants (A) a +inf
label, whose first coordinate update writes `inf - inf` into the residual
(a COMPUTED NaN, vendor payload), and (B) a payload-carrying quiet NaN
label `0x7FC0BEEF` (a PROPAGATED NaN), and requires the device card and
the oracle card to agree on every stage under IDENTICAL (REPORT under
FAST). It prints the raw payload the device and the host produced.

SABOTAGE `-D MOJOLEARN_CD_SABOTAGE_NO_NAN_CANON=1`: the copy is hashed
raw and `canon_nan_f32`/`canon_nan_list` are the identity, on BOTH sides,
so the cards carry the host's and the device's native payloads; it fails
exactly when those differ (the README records what this M4 and its host
produced, and what the NVIDIA and AMD legs are predicted to produce).
==========================================================================
"""

from std.memory import bitcast
from std.sys.compile import is_defined

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from core.identity_trace import IdentityTrace

#: SABOTAGE (a no-op in every build that does not name it): record raw
#: bits, canonicalize nothing, on the device copy AND in the host helpers.
comptime SAB_NO_NAN_CANON = is_defined["MOJOLEARN_CD_SABOTAGE_NO_NAN_CANON"]()

#: The one payload. The same bits numerics.mojo writes for log/sqrt NaN.
comptime CANON_NAN_BITS = UInt32(0x7FC00000)

comptime CANON_TPB = 256


def canon_nan_sabotage_name() -> String:
    comptime if SAB_NO_NAN_CANON:
        return String("NO_NAN_CANON")
    return String("none")


@always_inline
def canon_nan_f32(v: Float32) -> Float32:
    """`v`, with every NaN (either sign, any payload) replaced by the one
    quiet NaN; every non-NaN bit pattern is returned untouched (+-0.0,
    +-inf, denormals included)."""
    comptime if SAB_NO_NAN_CANON:
        return v
    if v != v:
        return bitcast[DType.float32](CANON_NAN_BITS)
    return v


def canon_nan_list(v: List[Float32]) -> List[Float32]:
    """`canon_nan_f32` over a host list (the oracle card's spelling)."""
    var out = List[Float32]()
    out.reserve(len(v))
    for e in v:
        out.append(canon_nan_f32(e))
    return out^


def canon_nan_kernel(v: MutPointer[Float32, MutAnyOrigin], n_in: Int32):
    """One thread, one cell: `v[i] != v[i]` -> the one payload. A bit
    write, not an arithmetic result, so the stored pattern is the same on
    every vendor (row 39's FACT 2)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var x = v.unsafe_load(i)
        if x != x:
            v.unsafe_store(i, bitcast[DType.float32](CANON_NAN_BITS))


def record_device_canon(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    tag: String,
    mut buf: DeviceBuffer[DType.float32],
    n: Int,
    mut scratch: DeviceBuffer[DType.float32],
) raises:
    """`trace.record_device(ctx, tag, buf, n)` through a canonicalized
    COPY: the first `n` floats of `buf` are copied into `scratch`, every
    NaN in the copy is rewritten to `CANON_NAN_BITS`, and the copy is
    hashed. `buf` is never written. No-op when the trace is disabled, so
    an untraced fit pays nothing. `scratch` must hold `n` floats."""
    if not trace.enabled:
        return
    if n > len(scratch):
        raise Error(
            "record_device_canon: tag '" + tag + "' asked for " + String(n)
            + " floats of a scratch holding " + String(len(scratch))
        )
    var src = buf.create_sub_buffer[DType.float32](0, n)
    var view = scratch.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_buf=view, src_buf=src)
    comptime if not SAB_NO_NAN_CANON:
        ctx.enqueue_function[canon_nan_kernel](
            view.unsafe_ptr(),
            Int32(n),
            grid_dim=((n + CANON_TPB - 1) // CANON_TPB, 1, 1),
            block_dim=(CANON_TPB, 1, 1),
        )
    trace.record_device[DType.float32](ctx, tag, view, n)
    _ = src^
    _ = view^


def record_scalar_f32_canon(
    mut trace: IdentityTrace, tag: String, v: Float32
) raises:
    """`trace.record_scalar_f32(tag, canon_nan_f32(v))`."""
    trace.record_scalar_f32(tag, canon_nan_f32(v))
