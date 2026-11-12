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
)
from core.preprocessing_program import prep_program_stage as _stage
from core.scaler_fit_impl import (
    scaler_all_finite, scaler_check_stats, scaler_check_missing, scaler_refuse_infinite_columns,
    scaler_feature_indices, scaler_fill,
)
from x_prep.transport_abi import XP_RUN

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

    def run(self, arena: Int, words: Int, program: Int, stages: Int) raises:
        var a: Array[Int, 2] = [arena, program]
        var p: Array[Int, 2] = [words, stages]
        var f: Array[Float64, 0] = []
        _ = entry_ref_call(self.xprep, XP_RUN, a, p, f)


def _run[C: ScalerCalls](calls: C, mut arena: List[Float32], program: List[Int32]) raises:
    calls.run(Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), len(program) // 16)
    # Integer addresses do not carry ownership: both stay alive to here.
    _ = arena
    _ = program


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
                                             weight: Int) raises -> ScalerStandardFit:
    """StandardScaler.fit of the C-ordered (n, d) Float32 block at `x`.
    `weight` is the address of n Float32 sample weights (a scalar already
    broadcast), or 0. Finite unweighted X takes the kernel; NaN or weights
    take the weighted-statistics x_prep program (stages 1, 101, 102 and the
    weight check 270, 248), whose missing-value and infinity checks run here."""
    var result = ScalerStandardFit(d)
    var weighted = weight != 0
    var stats = F32P(unsafe_from_address=_addr(result.stats))
    var finite = True
    if not weighted and calls.direct():
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
    var arena = List[Float32](length=size, fill=0)
    var ap = F32P(unsafe_from_address=_addr(arena))
    unsafe_memcpy(dest=ap, src=x, count=n * d)
    if weighted:
        unsafe_memcpy(dest=ap + wo, src=F32P(unsafe_from_address=weight), count=n)
    var program = List[Int32]()
    _stage(program, 1, d, 0, n, d, st)
    _stage(program, 101, d, 0, n, d, wo, out)
    _stage(program, 102, d, out + 2 * d, sc)
    if weighted:
        _stage(program, 270, nch, wo, n, ch, wc)
        _stage(program, 248, 1, wc, nch, wc + nch)
    _run(calls, arena, program)
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
    var rows = List[Float32](length=5 * d, fill=0)
    unsafe_memcpy(dest=rows.unsafe_ptr(), src=ap + 6 * d, count=5 * d)
    _ = arena^
    return rows^


def scaler_minmax_fit_core[C: ScalerCalls](calls: C, x: F32P, n: Int, d: Int, lower: Float32,
                                           upper: Float32) raises -> List[Float32]:
    """MinMaxScaler.fit of the C-ordered (n, d) Float32 block at `x`: the
    published rows (5*d, `scaler_minmax_keep_extrema`). NaN columns take the
    fill program (1, 24, 266) and the refusing kernel on the filled block."""
    var output = List[Float32](length=5 * d, fill=0)
    var op = F32P(unsafe_from_address=_addr(output))
    var finite = True
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
    var arena = List[Float32](length=cn + d, fill=0)
    var ap = F32P(unsafe_from_address=_addr(arena))
    unsafe_memcpy(dest=ap, src=x, count=n * d)
    scaler_feature_indices(ap + keep, d)
    var program = List[Int32]()
    _stage(program, 1, d, 0, n, d, st)
    _stage(program, 24, n * d, 0, n, d, st + 3 * d, dest, keep, d)
    _stage(program, 266, d, st, cn)
    _run(calls, arena, program)
    scaler_refuse_infinite_columns(ap, st, d)
    _ = calls.minmax_fit(MinMaxFitRequest(Int(ap + dest), Int(op), n, d, lower, upper, False))
    var rows = scaler_minmax_keep_extrema(calls, op, Int(ap + cn), d)
    _ = arena^
    _ = output^
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


def scaler_nan_fill[C: ScalerCalls](calls: C, x: F32P, n: Int, d: Int) raises -> List[Float32]:
    """The (n, d) block with each NaN replaced by its column statistic (the
    stage 1 + 24 program), at `scaler_nan_fill_offset` of the returned
    arena; an infinite column is refused."""
    var st = n * d
    var keep = st + 6 * d
    var stat = keep + d
    var dest = stat + d
    var arena = List[Float32](length=dest + n * d, fill=0)
    var ap = F32P(unsafe_from_address=_addr(arena))
    unsafe_memcpy(dest=ap, src=x, count=n * d)
    scaler_feature_indices(ap + keep, d)
    var program = List[Int32]()
    _stage(program, 1, d, 0, n, d, st)
    _stage(program, 24, n * d, 0, n, d, stat, dest, keep, d)
    _run(calls, arena, program)
    scaler_refuse_infinite_columns(ap, st, d)
    return arena^


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
        var keep = List[Float32](length=3 * n * d + 2 * d, fill=0)
        var kp = F32P(unsafe_from_address=_addr(keep))
        unsafe_memcpy(dest=kp, src=x, count=n * d)
        unsafe_memcpy(dest=kp + n * d, src=dst, count=n * d)
        unsafe_memcpy(dest=kp + 2 * n * d, src=rows + 4 * d, count=d)
        var restore = List[Int32]()
        # c2_nan_sub publishes int32 words, while nan_keep reads numeric
        # Float32 flags. Preserve the old _Prog.put_codes i2f stage: raw
        # integer 1 is a subnormal float that Metal can flush to zero.
        _stage(restore, 10, d, 2 * n * d, 2 * n * d + d)
        _stage(restore, 103, n * d, 0, d, n * d, 2 * n * d + d, 2 * n * d + 2 * d)
        _run(calls, keep, restore)
        unsafe_memcpy(dest=dst, src=kp + 2 * n * d + 2 * d, count=n * d)
        _ = keep^
