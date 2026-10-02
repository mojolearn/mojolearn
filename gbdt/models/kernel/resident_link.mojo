"""The resident GBDT predict's staging and link on the device (lane
hr2-gbdt-host, 2026-10-02).

`gbdt/resident_model.mojo` used to stage the input on host threads (NaN
substitution and the column-major transpose into a pinned buffer) and to
read the raw cursor back and apply the link (sigmoid, softmax, the Logloss
pair, the class codes) on host threads. Both are per element and per row,
so both are one GPU thread per element / row here:

  * `resident_stage_kernel`: one thread per (row, bordered column) of the
    raw input as the caller handed it (row- or column-major), the column's
    NaN treatment applied, written column-major for the binarize launches.
    A NaN on an `AsIs` column is recorded with an integer `Atomic.min` of
    the column number into one word, so the refusal names the LOWEST such
    column, as the host scan did.
  * `resident_link_kernel`: one thread per row, the mode's transform over
    the plane-major cursor, written in the caller's row-major layout.

The double-precision links are `checks/soft_f64.mojo` (integer arithmetic
on the binary64 encoding: the Apple GPU has no float64), so every vendor and
the host column (`bindings/_mojolearn_gbdt_host.mojo`) compute the same
words. A float32 output is flushed to its signed zero when subnormal and the
pair's `p` likewise, the FTZ+DAZ host pool's result, spelled explicitly.
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast

from checks.numerics import ftz
from checks.soft_f64 import (
    SF64_ONE,
    SF64_ZERO,
    sf64_add,
    sf64_div,
    sf64_exp,
    sf64_from_f32,
    sf64_ftz,
    sf64_gt,
    sf64_neg,
    sf64_sigmoid_f32,
    sf64_sub,
    sf64_to_f32,
)

comptime LINK_RAW = 0
comptime LINK_SOFTMAX = 1
comptime LINK_SIGMOID = 2
comptime LINK_SIGMOID_PAIR = 3
comptime LINK_CLASSES_BINARY = 4
comptime LINK_CLASSES_PINNED = 5
comptime LINK_CLASSES_OVA = 6

#: `treat[f]` for a column without borders: never read, never written
comptime STAGE_SKIP = -1
#: the NaN-refusal word's "no NaN" value
comptime STAGE_NO_BAD = Int32(0x7FFFFFFF)

comptime STAGE_BLOCK = 256
comptime LINK_BLOCK = 256


def resident_stage_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    treat: MutPointer[Int32, MutAnyOrigin],
    bad: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_cols: Int32,
    row_major: Int32,
):
    """One thread per element, grid-stride. `treat[f]` is the column's NaN
    treatment code, or `STAGE_SKIP`; codes 0/1/2 are `NAN_TREATMENT_AS_IS`,
    `AS_FALSE` (-inf) and `AS_TRUE` (+inf), `nan_substitution`'s values."""
    var n = Int(n_rows)
    var nc = Int(n_cols)
    var total = n * nc
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < total:
        var f: Int
        var r: Int
        if row_major != 0:
            r = i // nc
            f = i - r * nc
        else:
            f = i // n
            r = i - f * n
        var t = Int(treat[f])
        if t != STAGE_SKIP:
            var v = src[i]
            if v != v:
                if t == 0:
                    _ = Atomic.min(bad, Int32(f))
                elif t == 1:
                    v = bitcast[DType.float32](UInt32(0xFF800000))
                else:
                    v = bitcast[DType.float32](UInt32(0x7F800000))
            dst[f * n + r] = v
        i += stride


@always_inline
def _f32_out(w: UInt64) -> Float32:
    return ftz(sf64_to_f32(w))


def resident_link_kernel(
    cursor: MutPointer[Float32, MutAnyOrigin],
    out_f32: MutPointer[Float32, MutAnyOrigin],
    out_u64: MutPointer[UInt64, MutAnyOrigin],
    out_i64: MutPointer[Int64, MutAnyOrigin],
    n_rows: Int32,
    dim: Int32,
    mode: Int32,
    sabotage: Int32,
    pair_swap: Int32,
):
    """One thread per row, grid-stride. `cursor` is plane-major
    (`cursor[k * n_rows + r]`). `sabotage` adds 1.0 to row 0 of plane 0
    before the transform (`MOJOLEARN_GBDT_RESIDENT_SABOTAGE`); `pair_swap`
    writes the pair as `[p, 1 - p]` (`MOJOLEARN_FOREST_HOST_SABOTAGE`)."""
    var n = Int(n_rows)
    var d = Int(dim)
    var m = Int(mode)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while r < n:
        var raw0 = cursor[r]
        if sabotage != 0 and r == 0:
            raw0 = raw0 + Float32(1.0)
        if m == LINK_RAW:
            out_f32[r * d] = raw0
            for k in range(1, d):
                out_f32[r * d + k] = cursor[k * n + r]
        elif m == LINK_SIGMOID_PAIR:
            var p = sf64_ftz(sf64_sigmoid_f32(raw0))
            var q = sf64_sub(SF64_ONE, p)
            if pair_swap != 0:
                out_u64[2 * r] = p
                out_u64[2 * r + 1] = q
            else:
                out_u64[2 * r] = q
                out_u64[2 * r + 1] = p
        elif m == LINK_CLASSES_BINARY:
            out_i64[r] = Int64(1) if raw0 > Float32(0.0) else Int64(0)
        elif m == LINK_SOFTMAX or m == LINK_CLASSES_PINNED:
            # `multiclass_probabilities`' row: the max seeded at ZERO for
            # the pinned class, one double division per class
            var mx = SF64_ZERO
            for k in range(d):
                var v = sf64_from_f32(raw0 if k == 0 else cursor[k * n + r])
                if sf64_gt(v, mx):
                    mx = v
            var se = SF64_ZERO
            for k in range(d):
                var v = sf64_from_f32(raw0 if k == 0 else cursor[k * n + r])
                se = sf64_add(se, sf64_exp(sf64_sub(v, mx)))
            var e_pin = sf64_exp(sf64_neg(mx))
            se = sf64_add(se, e_pin)
            if m == LINK_SOFTMAX:
                var w = d + 1
                for k in range(d):
                    var v = sf64_from_f32(raw0 if k == 0 else cursor[k * n + r])
                    out_f32[r * w + k] = _f32_out(
                        sf64_div(sf64_exp(sf64_sub(v, mx)), se)
                    )
                out_f32[r * w + d] = _f32_out(sf64_div(e_pin, se))
            else:
                var best = 0
                var best_value = Float32(0.0)
                for k in range(d):
                    var v = sf64_from_f32(raw0 if k == 0 else cursor[k * n + r])
                    var value = _f32_out(sf64_div(sf64_exp(sf64_sub(v, mx)), se))
                    if k == 0 or value > best_value:
                        best = k
                        best_value = value
                var pinned = _f32_out(sf64_div(e_pin, se))
                if pinned > best_value:
                    best = d
                out_i64[r] = Int64(best)
        else:
            # LINK_SIGMOID and LINK_CLASSES_OVA: the one-vs-all element
            var best = 0
            var best_value = Float32(0.0)
            for k in range(d):
                var v = raw0 if k == 0 else cursor[k * n + r]
                var value = _f32_out(sf64_sigmoid_f32(v))
                if m == LINK_SIGMOID:
                    out_f32[r * d + k] = value
                elif k == 0 or value > best_value:
                    best = k
                    best_value = value
            if m == LINK_CLASSES_OVA:
                out_i64[r] = Int64(best)
        r += stride
