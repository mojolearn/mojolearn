# SPDX-License-Identifier: Apache-2.0
"""The typed scaler cores (lane scaler-typed-core, 2026-11-12).

StandardScaler and MinMaxScaler fit and transform on plain pointers, dims and
flags, shared by the public scaler doors (bindings/scaler_native_door.mojo,
bindings/scaler_lifecycle_native.mojo, bindings/minmax_native.mojo) and the
parallel shard worker (bindings/parallel_shard_native.mojo). The kernel and
x_prep program calls go through `ScalerCalls`; `ScalerTableCalls` is the
native one: the scaler kernel entry (`scaler_native_v1`, prep entry table
family PREP_TABLE_SCALER) and the x_prep transport entry (family
PREP_TABLE_X_PREP), each an `EntryRef` resolved once per operation. No
Python object is read, built or called here; the caller releases the GIL
around a core call made through `ScalerTableCalls`.

The NaN handling (missing-value fill, the per-column NaN restore), the
fitted-statistic checks, the Float32 overflow refusals and the extrema
publication program live here once. The stage words, the stage order and
the kernels are the ones the doors ran before: no bit moves.

Each core is one operation on one block of whole columns: serial by nature
at this level (the kernels it calls are the parallel part); the shard driver
splits independent column blocks across devices.
"""
from std.memory import unsafe_memcpy, bitcast
from core.entry_table import EntryRef, entry_ref_call
from core.scaler_native_api import (
    ScalerFitRequest, ScalerTransformRequest, MinMaxFitRequest, MinMaxTransformRequest,
    SCALER_OP_FIT, SCALER_OP_TRANSFORM, SCALER_OP_MINMAX_FIT, SCALER_OP_MINMAX_TRANSFORM,
    SCALER_OP_STANDARD_FIT_TRANSFORM, SCALER_OP_MINMAX_FIT_TRANSFORM,
)
from core.preprocessing_program import prep_program_stage as _stage
from core.scaler_fit_impl import (
    scaler_all_finite, scaler_check_stats, scaler_check_missing, scaler_refuse_infinite_columns,
    scaler_feature_indices, scaler_fill,
)
from x_prep.transport_abi import XP_RUN, XP_RUN_RANGES

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]

# How a fit publishes n_samples_seen_.
comptime SCALER_SEEN_INT = 0        # one native int (seen_scalar)
comptime SCALER_SEEN_FLOAT = 1      # one native float (seen_scalar)
comptime SCALER_SEEN_INT_ROW = 2    # an int64 row of d (seen, Float32 words narrowed at PACK)
comptime SCALER_SEEN_FLOAT_ROW = 3  # a float32 row of d (seen)


trait ScalerCalls:
    """The scaler kernel operations and the x_prep program executor a core
    calls. `direct` is the caller's route choice (the direct entries exist
    and MOJOLEARN_SCALER_DIRECT is not 0)."""

    def direct(self) -> Bool:
        ...

    def standard_fit(self, r: ScalerFitRequest) raises -> Int:
        ...

    def standard_transform(self, r: ScalerTransformRequest) raises -> Int:
        ...

    def minmax_fit(self, r: MinMaxFitRequest) raises -> Int:
        ...

    def minmax_transform(self, r: MinMaxTransformRequest) raises -> Int:
        ...

    def run(self, arena: Int, words: Int, program: Int, stages: Int) raises:
        ...

    def run_slot(self, arena: Int, words: Int, program: Int, stages: Int, x_slot: Int, x_words: Int,
                 host_hi: Int, out_lo: Int, out_hi: Int, keep_lo: Int, keep_hi: Int, keep_slot: Int) raises:
        """Lane vdf-scope-neural6: `run` with arena words [0, x_words) filled from x_prep store slot `x_slot`
        device to device, [x_words, host_hi) from the host arena (every later word starts zero on the device,
        as the host's do), and only [out_lo, out_hi) brought back (the x_prep ranges runner). Lane vdf-tail-a:
        with `keep_slot` >= 0 the words [keep_lo, keep_hi) inside it go device to device into that x_prep
        store slot instead (core/arena_io.mojo slot outputs) and never come back. A caller without a GPU
        x_prep entry refuses."""
        ...


@fieldwise_init
struct ScalerTableCalls(ScalerCalls, ImplicitlyCopyable, Movable):
    """The native calls: two resolved table entries and the route flag."""

    var scaler: EntryRef
    var xprep: EntryRef
    var use_direct: Bool

    def direct(self) -> Bool:
        return self.use_direct

    def standard_fit(self, r: ScalerFitRequest) raises -> Int:
        var a: Array[Int, 2] = [r.source, r.dst]
        var p: Array[Int, 5] = [r.rows, r.features, Int(r.with_mean), Int(r.with_std), Int(r.direct)]
        var f: Array[Float64, 0] = []
        return entry_ref_call(self.scaler, SCALER_OP_FIT, a, p, f)

    def standard_transform(self, r: ScalerTransformRequest) raises -> Int:
        var a: Array[Int, 4] = [r.source, r.mean, r.scale, r.dst]
        var p: Array[Int, 6] = [r.rows, r.features, Int(r.inverse), Int(r.with_mean), Int(r.with_std), Int(r.direct)]
        var f: Array[Float64, 0] = []
        return entry_ref_call(self.scaler, SCALER_OP_TRANSFORM, a, p, f)

    def minmax_fit(self, r: MinMaxFitRequest) raises -> Int:
        var a: Array[Int, 2] = [r.source, r.dst]
        var p: Array[Int, 3] = [r.rows, r.features, Int(r.direct)]
        var f: Array[Float64, 2] = [Float64(r.lower), Float64(r.upper)]
        return entry_ref_call(self.scaler, SCALER_OP_MINMAX_FIT, a, p, f)

    def minmax_transform(self, r: MinMaxTransformRequest) raises -> Int:
        var a: Array[Int, 4] = [r.source, r.scale, r.offset, r.dst]
        var p: Array[Int, 5] = [r.rows, r.features, Int(r.inverse), Int(r.clip), Int(r.direct)]
        var f: Array[Float64, 2] = [Float64(r.lower), Float64(r.upper)]
        return entry_ref_call(self.scaler, SCALER_OP_MINMAX_TRANSFORM, a, p, f)

    # The combined fit_transform (one upload of X), on this struct only: the
    # doors call the same exports through their Python entries. 1 both
    # written, 0 nothing written (the two-step route), -1 transform overflow.
    def standard_fit_transform(self, x: Int, stats: Int, dst: Int, n: Int, d: Int, with_mean: Bool,
                               with_std: Bool) raises -> Int:
        var a: Array[Int, 3] = [x, stats, dst]
        var p: Array[Int, 4] = [n, d, Int(with_mean), Int(with_std)]
        var f: Array[Float64, 0] = []
        return entry_ref_call(self.scaler, SCALER_OP_STANDARD_FIT_TRANSFORM, a, p, f)

    def minmax_fit_transform(self, x: Int, stats: Int, dst: Int, n: Int, d: Int, lower: Float32, upper: Float32,
                             clip: Bool) raises -> Int:
        var a: Array[Int, 3] = [x, stats, dst]
        var p: Array[Int, 3] = [n, d, Int(clip)]
        var f: Array[Float64, 2] = [Float64(lower), Float64(upper)]
        return entry_ref_call(self.scaler, SCALER_OP_MINMAX_FIT_TRANSFORM, a, p, f)

    def run(self, arena: Int, words: Int, program: Int, stages: Int) raises:
        var a: Array[Int, 2] = [arena, program]
        var p: Array[Int, 2] = [words, stages]
        var f: Array[Float64, 0] = []
        _ = entry_ref_call(self.xprep, XP_RUN, a, p, f)

    def run_slot(self, arena: Int, words: Int, program: Int, stages: Int, x_slot: Int, x_words: Int,
                 host_hi: Int, out_lo: Int, out_hi: Int, keep_lo: Int, keep_hi: Int, keep_slot: Int) raises:
        # Input triples (lo, hi, src: a slot id or -1 host) and output quads (lo, hi, -1 no count, 1; a slot
        # output's CNT is -slot - 2), core/arena_io.mojo; no scratch, no output region.
        var ins: List[Int32] = [Int32(0), Int32(x_words), Int32(x_slot)]
        if host_hi > x_words:
            ins.append(Int32(x_words))
            ins.append(Int32(host_hi))
            ins.append(Int32(-1))
        var outs: List[Int32] = [Int32(out_lo), Int32(out_hi), Int32(-1), Int32(1)]
        if keep_slot >= 0:
            outs = [Int32(out_lo), Int32(keep_lo), Int32(-1), Int32(1),
                    Int32(keep_lo), Int32(keep_hi), Int32(-keep_slot - 2), Int32(1),
                    Int32(keep_hi), Int32(out_hi), Int32(-1), Int32(1)]
        var a: Array[Int, 5] = [arena, program, 0, Int(ins.unsafe_ptr()), Int(outs.unsafe_ptr())]
        var p: Array[Int, 6] = [words, 0, 0, stages, len(ins) // 3, len(outs) // 4]
        var f: Array[Float64, 0] = []
        _ = entry_ref_call(self.xprep, XP_RUN_RANGES, a, p, f)
        _ = len(ins) + len(outs)  # the address words are not owners


def _run[C: ScalerCalls](calls: C, mut arena: List[Float32], program: List[Int32]) raises:
    calls.run(Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), len(program) // 16)
    # Integer addresses do not carry ownership: both stay alive to here.
    _ = arena
    _ = program


def _run_x[C: ScalerCalls](calls: C, mut arena: List[Float32], program: List[Int32], x_slot: Int, x_words: Int,
                           out_lo: Int) raises:
    """`_run`, or with `x_slot` >= 0 (lane vdf-scope-neural6) the ranges run: X's `x_words` from the x_prep slot,
    the arena's other words from the host, [out_lo, end) back. The host arena's X words are never read then."""
    if x_slot < 0:
        _run(calls, arena, program)
        return
    calls.run_slot(Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), len(program) // 16,
                   x_slot, x_words, len(arena), out_lo, len(arena), 0, 0, -1)
    _ = arena
    _ = program


def _run_x_keep[C: ScalerCalls](calls: C, mut arena: List[Float32], program: List[Int32], x_slot: Int,
                                x_words: Int, host_hi: Int, keep_lo: Int, keep_slot: Int) raises:
    """Lane vdf-tail-a: the ranges run of `_run_x` (X from x_prep slot `x_slot`, host words [x_words, host_hi),
    the rest zero on the device) whose block [keep_lo, keep_lo + x_words) goes device to device into x_prep
    slot `keep_slot`; every other word from x_words on comes back. Same stages on the same words."""
    calls.run_slot(Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), len(program) // 16,
                   x_slot, x_words, host_hi, x_words, len(arena), keep_lo, keep_lo + x_words, keep_slot)
    _ = arena
    _ = program


def _arena_after_x(size: Int, x_words: Int, x: F32P, x_slot: Int, zero_hi: Int = -1) -> List[Float32]:
    """A zeroed `size`-word host arena with X's `x_words` at 0 (copied from `x`), or, for a slot (lane
    vdf-scope-neural6), those words left unset: the device fills them from the slot and they never come back.
    `zero_hi` >= 0 (a slot, lane vdf-tail-a): only [x_words, zero_hi) is zeroed (the host input range of
    `_run_x_keep`); the later words start zero on the device and are either brought back or never read."""
    if x_slot < 0:
        var arena = List[Float32](length=size, fill=0)
        unsafe_memcpy(dest=arena.unsafe_ptr(), src=x, count=x_words)
        return arena^
    var arena = List[Float32](unsafe_uninit_length=size)
    var ap = arena.unsafe_ptr()
    var hi = size if zero_hi < 0 else min(zero_hi, size)
    for i in range(x_words, hi):  # small-loop(host arena words after X: the zero fill the host path's List does)
        ap[i] = Float32(0)
    return arena^


def _addr(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


# ---------------------------------------------------------------- StandardScaler fit

struct ScalerStandardFit(Movable):
    """A fit's statistics (mean, var, scale rows of d) and n_samples_seen_."""

    var stats: List[Float32]
    var seen_kind: Int
    var seen_scalar: Float64
    var seen: List[Float32]

    def __init__(out self, d: Int):
        self.stats = List[Float32](length=3 * d, fill=0)
        self.seen_kind = SCALER_SEEN_INT
        self.seen_scalar = 0
        self.seen = List[Float32]()


def scaler_standard_fit_core[C: ScalerCalls](calls: C, x: F32P, n: Int, d: Int, with_mean: Bool, with_std: Bool,
                                             weight: Int, x_slot: Int = -1) raises -> ScalerStandardFit:
    """StandardScaler.fit of the C-ordered (n, d) Float32 block at `x`.
    `weight` is the address of n Float32 sample weights (a scalar already
    broadcast), or 0. Finite unweighted X takes the kernel; NaN or weights
    take the weighted-statistics x_prep program (stages 1, 101, 102 and the
    weight check 270, 248), whose missing-value and infinity checks run here.
    `x_slot` (lane vdf-scope-neural6): X is that x_prep store slot (a member's fold; `x` unused) and the caller
    already took the program route (weights, or its direct slot fit found X non-finite): the program reads X
    there (`_run_x`), the same stages on the same words."""
    var result = ScalerStandardFit(d)
    var weighted = weight != 0
    var stats = F32P(unsafe_from_address=_addr(result.stats))
    var finite = x_slot < 0
    if x_slot >= 0:
        pass
    elif not weighted and calls.direct():
        finite = calls.standard_fit(ScalerFitRequest(Int(x), Int(stats), n, d, with_mean, with_std, True)) != 0
    else:
        finite = scaler_all_finite(x, n * d)
        if finite and not weighted:
            _ = calls.standard_fit(ScalerFitRequest(Int(x), Int(stats), n, d, with_mean, with_std, False))
    if finite and not weighted:
        scaler_check_stats(stats, stats + d, stats + 2 * d, d)
        result.seen_kind = SCALER_SEEN_INT
        result.seen_scalar = Float64(n)
        return result^
    var wo = n * d if weighted else -1
    var st = n * d + (n if weighted else 0)
    var out = st + 6 * d
    var sc = out + 3 * d
    var ch = max(4096, (n + 4095) // 4096)
    var nch = (n + ch - 1) // ch
    var wc = sc + d
    var size = wc + (nch + 1 if weighted else 0)
    if size > 2147483647:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var arena = _arena_after_x(size, n * d, x, x_slot)
    var ap = F32P(unsafe_from_address=_addr(arena))
    if weighted:
        unsafe_memcpy(dest=ap + wo, src=F32P(unsafe_from_address=weight), count=n)
    var program = List[Int32]()
    _stage(program, 1, d, 0, n, d, st)
    _stage(program, 101, d, 0, n, d, wo, out)
    _stage(program, 102, d, out + 2 * d, sc)
    if weighted:
        _stage(program, 270, nch, wo, n, ch, wc)
        _stage(program, 248, 1, wc, nch, wc + nch)
    _run_x(calls, arena, program, x_slot, n * d, n * d)
    if weighted and bitcast[DType.int32](arena[wc + nch]) != 0:
        raise Error("sample_weight must be finite")
    var all_same = scaler_check_missing(ap, st, out, sc, d)
    unsafe_memcpy(dest=stats, src=ap + out + d, count=d)
    unsafe_memcpy(dest=stats + d, src=ap + out + 2 * d, count=d)
    unsafe_memcpy(dest=stats + 2 * d, src=ap + sc, count=d)
    if all_same:
        result.seen_kind = SCALER_SEEN_FLOAT if weighted else SCALER_SEEN_INT
        result.seen_scalar = Float64(arena[out]) if weighted else Float64(Int(arena[out]))
    else:
        result.seen_kind = SCALER_SEEN_FLOAT_ROW if weighted else SCALER_SEEN_INT_ROW
        result.seen = List[Float32](length=d, fill=0)
        unsafe_memcpy(dest=result.seen.unsafe_ptr(), src=ap + out, count=d)
    _ = arena^
    return result^


# ---------------------------------------------------------------- MinMaxScaler fit

def scaler_minmax_keep_extrema[C: ScalerCalls](calls: C, stats: F32P, colnan: Int, d: Int) raises -> List[Float32]:
    """The published MinMax rows (data_min_, data_max_, data_range_, scale_,
    min_; 5*d) from a kernel's five rows at `stats` and the per-column NaN
    flags at `colnan` (0: none), by the c2_mm_keep program (267, 248)."""
    var arena = List[Float32](length=12 * d + 1, fill=0)
    var ap = F32P(unsafe_from_address=_addr(arena))
    if colnan != 0:
        unsafe_memcpy(dest=ap, src=F32P(unsafe_from_address=colnan), count=d)
    unsafe_memcpy(dest=ap + d, src=stats, count=5 * d)
    var program = List[Int32]()
    _stage(program, 267, d, d, d, 0, 6 * d, 11 * d)
    _stage(program, 248, 1, 11 * d, d, 12 * d)
    _run(calls, arena, program)
    if bitcast[DType.int32](arena[12 * d]) != 0:
        raise Error("MinMaxScaler fitted statistics overflowed or scale is not positive in Float32")
    var rows = List[Float32](unsafe_uninit_length=5 * d)
    unsafe_memcpy(dest=rows.unsafe_ptr(), src=ap + 6 * d, count=5 * d)
    _ = arena^
    return rows^


def scaler_minmax_fit_core[C: ScalerCalls](calls: C, x: F32P, n: Int, d: Int, lower: Float32,
                                           upper: Float32) raises -> List[Float32]:
    """MinMaxScaler.fit of the C-ordered (n, d) Float32 block at `x`: the
    published rows (5*d, `scaler_minmax_keep_extrema`). NaN columns take the
    fill program (1, 24, 266) and the refusing kernel on the filled block.
    A member's fold slot takes `scaler_minmax_fill_slot` (lane vdf-tail-a)."""
    var output = List[Float32](length=5 * d, fill=0)
    var op = F32P(unsafe_from_address=_addr(output))
    var finite: Bool
    if calls.direct():
        finite = calls.minmax_fit(MinMaxFitRequest(Int(x), Int(op), n, d, lower, upper, True)) != 0
    else:
        finite = scaler_all_finite(x, n * d)
        if finite:
            _ = calls.minmax_fit(MinMaxFitRequest(Int(x), Int(op), n, d, lower, upper, False))
    if finite:
        var rows = scaler_minmax_keep_extrema(calls, op, 0, d)
        _ = output^
        return rows^
    var st = n * d
    var keep = st + 6 * d
    var dest = keep + d
    var cn = dest + n * d
    var arena = _arena_after_x(cn + d, n * d, x, -1)
    var ap = F32P(unsafe_from_address=_addr(arena))
    scaler_feature_indices(ap + keep, d)
    var program = _minmax_fill_program(n, d)
    _run(calls, arena, program)
    scaler_refuse_infinite_columns(ap, st, d)
    _ = calls.minmax_fit(MinMaxFitRequest(Int(ap + dest), Int(op), n, d, lower, upper, False))
    var rows = scaler_minmax_keep_extrema(calls, op, Int(ap + cn), d)
    _ = arena^
    _ = output^
    return rows^


def _minmax_fill_program(n: Int, d: Int) -> List[Int32]:
    """The MinMax NaN fill program (1, 24, 266) over the arena of `scaler_minmax_fit_core`: X at 0, the column
    statistics at n*d (6d), the feature indices (d), the filled block (n*d), the column-NaN flags (d)."""
    var st = n * d
    var keep = st + 6 * d
    var dest = keep + d
    var cn = dest + n * d
    var program = List[Int32]()
    _stage(program, 1, d, 0, n, d, st)
    _stage(program, 24, n * d, 0, n, d, st + 3 * d, dest, keep, d)
    _stage(program, 266, d, st, cn)
    return program^


def scaler_minmax_fill_colnan(n: Int, d: Int) -> Int:
    """Where `scaler_minmax_fill_slot`'s arena holds the d column-NaN flags."""
    return n * d + 6 * d + d + n * d


def scaler_minmax_fill_slot[C: ScalerCalls](calls: C, n: Int, d: Int, x_slot: Int, fill_slot: Int) raises -> List[Float32]:
    """MinMaxScaler.fit's NaN fill (`scaler_minmax_fit_core`'s program, the same stages on the same words) of X
    in x_prep store slot `x_slot` (lane vdf-tail-a: a member's fold whose direct slot fit found a NaN or an
    infinity), the filled (n, d) block written device to device into x_prep slot `fill_slot` (n*d words): it
    never comes to the host. An infinite column is refused. Returns the arena (the column statistics and the
    column-NaN flags at `scaler_minmax_fill_colnan`); the caller fits the filled slot with the direct kernel
    (the refusing kernel's body) and publishes with `scaler_minmax_keep_extrema`."""
    var st = n * d
    var keep = st + 6 * d
    var dest = keep + d
    var cn = dest + n * d
    var arena = _arena_after_x(cn + d, n * d, F32P(unsafe_from_address=0), x_slot, dest)
    var ap = F32P(unsafe_from_address=_addr(arena))
    scaler_feature_indices(ap + keep, d)
    var program = _minmax_fill_program(n, d)
    _run_x_keep(calls, arena, program, x_slot, n * d, dest, dest, fill_slot)
    scaler_refuse_infinite_columns(ap, st, d)
    return arena^


def scaler_minmax_partial_fit_run[C: ScalerCalls](calls: C, old_min: F32P, old_max: F32P, fitted_d: Int, x: F32P,
                                                  n: Int, d: Int, lower: Float32, upper: Float32) raises -> List[Float32]:
    """MinMaxScaler.partial_fit of a fitted state on typed values (lane
    bm-trees, 2026-11-22; the composition bindings/minmax_native_door.mojo
    `minmax_partial_fit` ran over Python objects): the batch (C-ordered
    (n, d) Float32 at `x`) fitted by `scaler_minmax_fit_core`, the width
    refusal against the fitted `fitted_d`, then the merge: stage 268 over
    the old and the batch's extrema (NaN-aware min / max and the per-column
    NaN flags), the kernel fit of the 4 x d extrema block, and the published
    rows by `scaler_minmax_keep_extrema` with those flags. Returns the five
    published rows (5*d). The door and a native check both call it."""
    var fresh = scaler_minmax_fit_core(calls, x, n, d, lower, upper)
    if d != fitted_d:
        raise Error("MinMaxScaler input feature count differs from fit")
    var arena = List[Float32](length=9 * d, fill=0)
    var ap = F32P(unsafe_from_address=_addr(arena))
    var fp = F32P(unsafe_from_address=_addr(fresh))
    unsafe_memcpy(dest=ap, src=old_min, count=d)
    unsafe_memcpy(dest=ap + d, src=old_max, count=d)
    unsafe_memcpy(dest=ap + 2 * d, src=fp, count=d)
    unsafe_memcpy(dest=ap + 3 * d, src=fp + d, count=d)
    var program = List[Int32]()
    _stage(program, 268, d, 0, d, 2 * d, 3 * d, d, 4 * d, 8 * d)
    _run(calls, arena, program)
    var output = List[Float32](length=5 * d, fill=0)
    var op = F32P(unsafe_from_address=_addr(output))
    _ = calls.minmax_fit(MinMaxFitRequest(Int(ap + 4 * d), Int(op), 4, d, lower, upper, False))
    var rows = scaler_minmax_keep_extrema(calls, op, Int(ap + 8 * d), d)
    _ = arena^
    _ = output^
    _ = fresh^
    return rows^


# ---------------------------------------------------------------- transforms (both scalers)

def scaler_transform_rows[C: ScalerCalls](calls: C, center: Int, scale: Int, d: Int, with_mean: Bool,
                                          with_std: Bool) raises -> List[Float32]:
    """The derived and validated rows (7*d+5 words) of a fitted pair: the
    centre row at `center` (StandardScaler mean_, MinMaxScaler min_; 0 for
    none: zeros) and `scale` (0 for none: ones), by the stage 264 + 248
    program. Words 7d, 7d+1, 7d+2 are the column-NaN, nonfinite and
    nonpositive flags `scaler_transform_core` reads."""
    var arena = List[Float32](length=7 * d + 5, fill=0)
    var ap = F32P(unsafe_from_address=_addr(arena))
    if center != 0:
        unsafe_memcpy(dest=ap, src=F32P(unsafe_from_address=center), count=d)
    if scale != 0:
        unsafe_memcpy(dest=ap + d, src=F32P(unsafe_from_address=scale), count=d)
    else:
        scaler_fill(ap + d, d, 1)
    arena[7 * d + 4] = 1
    var program = List[Int32]()
    _stage(program, 264, d, 0, d, Int(with_mean), Int(with_std), 7 * d + 3, 7 * d + 4, 2 * d, 3 * d, 4 * d, 5 * d, 6 * d)
    _stage(program, 248, 1, 4 * d, d, 7 * d)
    _stage(program, 248, 1, 5 * d, d, 7 * d + 1)
    _stage(program, 248, 1, 6 * d, d, 7 * d + 2)
    _run(calls, arena, program)
    return arena^


def scaler_transform_rows_valid(rows: F32P, d: Int) -> Bool:
    """True when the rows passed both statistic checks (cacheable)."""
    return bitcast[DType.int32](rows[7 * d + 1]) == 0 and bitcast[DType.int32](rows[7 * d + 2]) == 0


def scaler_nan_fill_offset(n: Int, d: Int) -> Int:
    """Where `scaler_nan_fill`'s filled block starts in its arena."""
    return n * d + 8 * d


def _nan_fill_program(n: Int, d: Int) -> List[Int32]:
    """The transform's NaN fill program (stages 1, 24) over `scaler_nan_fill`'s arena."""
    var st = n * d
    var keep = st + 6 * d
    var stat = keep + d
    var dest = stat + d
    var program = List[Int32]()
    _stage(program, 1, d, 0, n, d, st)
    _stage(program, 24, n * d, 0, n, d, stat, dest, keep, d)
    return program^


def scaler_nan_fill[C: ScalerCalls](calls: C, x: F32P, n: Int, d: Int) raises -> List[Float32]:
    """The (n, d) block with each NaN replaced by its column statistic (the
    stage 1 + 24 program), at `scaler_nan_fill_offset` of the returned
    arena; an infinite column is refused."""
    var st = n * d
    var keep = st + 6 * d
    var dest = keep + 2 * d
    var arena = _arena_after_x(dest + n * d, n * d, x, -1)
    var ap = F32P(unsafe_from_address=_addr(arena))
    scaler_feature_indices(ap + keep, d)
    var program = _nan_fill_program(n, d)
    _run(calls, arena, program)
    scaler_refuse_infinite_columns(ap, st, d)
    return arena^


def scaler_nan_fill_slot[C: ScalerCalls](calls: C, n: Int, d: Int, x_slot: Int, fill_slot: Int) raises:
    """`scaler_nan_fill` of X in x_prep store slot `x_slot` (lane vdf-tail-a: a member's fold), the filled
    (n, d) block written device to device into x_prep slot `fill_slot` (n*d words) and never brought to the
    host: only the column statistics come back (the infinite-column refusal). The same stages on the same words."""
    var st = n * d
    var keep = st + 6 * d
    var dest = keep + 2 * d
    var arena = _arena_after_x(dest + n * d, n * d, F32P(unsafe_from_address=0), x_slot, dest)
    var ap = F32P(unsafe_from_address=_addr(arena))
    scaler_feature_indices(ap + keep, d)
    var program = _nan_fill_program(n, d)
    _run_x_keep(calls, arena, program, x_slot, n * d, dest, dest, fill_slot)
    scaler_refuse_infinite_columns(ap, st, d)
    _ = arena^


@fieldwise_init
struct ScalerTransformSpec(Copyable, Movable):
    """The transform's flags: StandardScaler reads with_mean/with_std,
    MinMaxScaler reads clip and the Float32 feature range."""

    var minmax: Bool
    var inverse: Bool
    var with_mean: Bool
    var with_std: Bool
    var clip: Bool
    var lower: Float32
    var upper: Float32


def _transform_call[C: ScalerCalls](calls: C, spec: ScalerTransformSpec, direct: Bool, src: F32P, rows: F32P,
                                    n: Int, d: Int, dst: F32P) raises -> Int:
    if spec.minmax:
        return calls.minmax_transform(MinMaxTransformRequest(Int(src), Int(rows + 3 * d), Int(rows + 2 * d), Int(dst),
                                                             n, d, spec.inverse, spec.clip, spec.lower, spec.upper, direct))
    return calls.standard_transform(ScalerTransformRequest(Int(src), Int(rows + 2 * d), Int(rows + 3 * d), Int(dst),
                                                           n, d, spec.inverse, spec.with_mean, spec.with_std, direct))


def scaler_transform_core[C: ScalerCalls](calls: C, spec: ScalerTransformSpec, rows: F32P, x: F32P, n: Int, d: Int,
                                          dst: F32P) raises:
    """transform / inverse_transform of the (n, d) Float32 block at `x` into
    `dst` with the derived rows of `scaler_transform_rows`: the statistic
    refusals, the kernel (NaN rows filled first), the Float32 overflow
    refusal, then the per-column NaN restore (stages 10, 103)."""
    var name = String("MinMaxScaler") if spec.minmax else String("StandardScaler")
    if bitcast[DType.int32](rows[7 * d + 1]) != 0:
        raise Error(name + (String(" min_") if spec.minmax else String(" mean_")) + " and scale_ must remain finite (NaN: a feature never seen)")
    if bitcast[DType.int32](rows[7 * d + 2]) != 0:
        raise Error(name + " scale_ must remain positive")
    var colnan = bitcast[DType.int32](rows[7 * d]) != 0
    var finite = True
    if calls.direct():
        var got = _transform_call(calls, spec, True, x, rows, n, d, dst)
        if got == 0:
            raise Error(name + " transform overflowed in Float32")
        finite = got == 1
        if not finite:
            var filled = scaler_nan_fill(calls, x, n, d)
            var fp = F32P(unsafe_from_address=_addr(filled)) + scaler_nan_fill_offset(n, d)
            if _transform_call(calls, spec, True, fp, rows, n, d, dst) != 1:
                raise Error(name + " transform overflowed in Float32")
            _ = filled^
    else:
        finite = scaler_all_finite(x, n * d)
        if finite:
            _ = _transform_call(calls, spec, False, x, rows, n, d, dst)
        else:
            var filled = scaler_nan_fill(calls, x, n, d)
            var fp = F32P(unsafe_from_address=_addr(filled)) + scaler_nan_fill_offset(n, d)
            _ = _transform_call(calls, spec, False, fp, rows, n, d, dst)
            _ = filled^
        if not scaler_all_finite(dst, n * d):
            raise Error(name + " transform overflowed in Float32")
    if not finite or colnan:
        _transform_restore(calls, rows, x, -1, n, d, dst)


def scaler_transform_slot_restore[C: ScalerCalls](calls: C, rows: F32P, x_slot: Int, n: Int, d: Int,
                                                  dst: F32P) raises:
    """`scaler_transform_core`'s per-column NaN restore of `dst` with X in x_prep store slot `x_slot` (lane
    vdf-scope-neural6; lane vdf-tail-a: the member runs the NaN fill into a slot (`scaler_nan_fill_slot`) and the
    direct slot transform itself, then calls this). The same program on the same words."""
    _transform_restore(calls, rows, F32P(unsafe_from_address=0), x_slot, n, d, dst)


def _transform_restore[C: ScalerCalls](calls: C, rows: F32P, x: F32P, x_slot: Int, n: Int, d: Int,
                                       dst: F32P) raises:
    """The per-column NaN restore (stages 10, 103) of `dst` from X (host words at `x`, or x_prep store slot
    `x_slot`, lane vdf-scope-neural6)."""
    var keep = _arena_after_x(3 * n * d + 2 * d, n * d, x, x_slot)
    var kp = F32P(unsafe_from_address=_addr(keep))
    unsafe_memcpy(dest=kp + n * d, src=dst, count=n * d)
    unsafe_memcpy(dest=kp + 2 * n * d, src=rows + 4 * d, count=d)
    var restore = List[Int32]()
    # c2_nan_sub publishes int32 words, while nan_keep reads numeric
    # Float32 flags. Preserve the old _Prog.put_codes i2f stage: raw
    # integer 1 is a subnormal float that Metal can flush to zero.
    _stage(restore, 10, d, 2 * n * d, 2 * n * d + d)
    _stage(restore, 103, n * d, 0, d, n * d, 2 * n * d + d, 2 * n * d + 2 * d)
    _run_x(calls, keep, restore, x_slot, n * d, 2 * n * d + 2 * d)
    unsafe_memcpy(dest=dst, src=kp + 2 * n * d + 2 * d, count=n * d)
    _ = keep^
