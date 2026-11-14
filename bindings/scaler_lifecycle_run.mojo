# SPDX-License-Identifier: Apache-2.0
"""RUN half of StandardScaler.partial_fit (lane py-msel-prep-cnn, 2026-11-14).

The merge of a fitted state with one more batch, on typed values only: the
batch fit is the typed core (core/scaler_core.mojo `scaler_standard_fit_core`:
NaN as missing, sample weights, the weighted-statistics program), the merge is
the x_prep program the door composed before (stage 91 Chan's pairwise rule +
stage 102 the scale, stage 269 the exact int64 count addition) and the
Float64 count-only addition when neither mean nor variance is kept. The
stage words, the stage order, the count kinds and the narrowing are the ones
bindings/scaler_lifecycle_native_door.mojo ran: no bit moves.

Callers:
- the door (UNPACK the fitted fields into `ScalerSeenView` and row
  addresses, call `scaler_standard_fit_core` then `scaler_standard_merge_run`
  so its refusals keep their order, PACK the merged rows);
- a native check, through `scaler_standard_partial_fit_run` with
  `ScalerDriverCalls` (the typed `ScalerDriver` and `PrepProgramDriver` of
  core/scaler_native_api.mojo as the `ScalerCalls` of the core).

No Python object is read, built or called here. One operation on one block
of whole columns: serial at this level (the kernels and programs it calls
are the parallel part).
"""
from std.memory import unsafe_memcpy
from core.preprocessing_program import prep_program_stage as _stage
from core.scaler_core import (
    ScalerCalls, ScalerStandardFit, scaler_standard_fit_core,
    SCALER_SEEN_INT, SCALER_SEEN_FLOAT, SCALER_SEEN_INT_ROW, SCALER_SEEN_FLOAT_ROW,
)
from core.scaler_fit_impl import (
    scaler_counts, scaler_counts_f32, scaler_counts_i64, scaler_counts_f64, scaler_equal_f32, scaler_equal_i64,
    scaler_merge_count_only, scaler_narrow_counts,
)
from core.scaler_native_api import (
    ScalerDriver, PrepProgramDriver, ScalerFitRequest, ScalerTransformRequest, MinMaxFitRequest,
    MinMaxTransformRequest, PrepProgramRequest, SCALER_OP_DIRECT,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]

# `ScalerSeenView.source`: the source kinds of core/scaler_fit_impl.mojo's
# `scaler_counts_*` (0 a scalar, 1 an int64 row, 2 a float32 row).
comptime SCALER_SEEN_SOURCE_SCALAR = 0
comptime SCALER_SEEN_SOURCE_I64 = 1
comptime SCALER_SEEN_SOURCE_F32 = 2


@fieldwise_init
struct ScalerSeenView(ImplicitlyCopyable, Movable):
    """A borrowed n_samples_seen_: a scalar (`integral`: a native int count,
    else a float weight sum) or the address of a row of d counts."""

    var source: Int
    var scalar: Float64
    var integer: Int64
    var integral: Bool
    var row: Int


struct ScalerStandardMerged(Movable):
    """The merged fitted state: `stats` holds mean, var and scale rows of d
    when `kept` (the count-only merge leaves them untouched), and the merged
    n_samples_seen_ as `seen_kind` (core/scaler_core.mojo SCALER_SEEN_*):
    `seen_integer` / `seen_scalar` for a scalar, `seen_i64` / `seen_f32` for
    a row."""

    var kept: Bool
    var stats: List[Float32]
    var seen_kind: Int
    var seen_scalar: Float64
    var seen_integer: Int64
    var seen_i64: List[Int64]
    var seen_f32: List[Float32]

    def __init__(out self, d: Int, kept: Bool):
        self.kept = kept
        self.stats = List[Float32](length=3 * d if kept else 0, fill=0)
        self.seen_kind = SCALER_SEEN_INT
        self.seen_scalar = 0
        self.seen_integer = 0
        self.seen_i64 = List[Int64]()
        self.seen_f32 = List[Float32]()


def scaler_seen_of_fit(fit: ScalerStandardFit, mut scratch: List[Int64], d: Int) -> ScalerSeenView:
    """The batch fit's n_samples_seen_ as the merge reads it: an int row is
    the int64 row the fit published (`scaler_counts`, the Float32 words
    narrowed), kept in `scratch`; a float row is borrowed from `fit.seen`
    (`fit` and `scratch` stay alive through the merge)."""
    if fit.seen_kind == SCALER_SEEN_INT:
        return ScalerSeenView(SCALER_SEEN_SOURCE_SCALAR, fit.seen_scalar, Int64(Int(fit.seen_scalar)), True, 0)
    if fit.seen_kind == SCALER_SEEN_FLOAT:
        return ScalerSeenView(SCALER_SEEN_SOURCE_SCALAR, fit.seen_scalar, Int64(0), False, 0)
    if fit.seen_kind == SCALER_SEEN_FLOAT_ROW:
        return ScalerSeenView(SCALER_SEEN_SOURCE_F32, 0, Int64(0), False, Int(fit.seen.unsafe_ptr()))
    scratch = List[Int64](length=d, fill=0)
    scaler_counts(_F32P(unsafe_from_address=Int(fit.seen.unsafe_ptr())), _I64P(unsafe_from_address=Int(scratch.unsafe_ptr())), d)
    return ScalerSeenView(SCALER_SEEN_SOURCE_I64, 0, Int64(0), False, Int(scratch.unsafe_ptr()))


def _run_program[C: ScalerCalls](calls: C, mut arena: List[Float32], program: List[Int32]) raises:
    calls.run(Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), len(program) // 16)
    # Integer addresses do not carry ownership: both stay alive to here.
    _ = arena
    _ = program


def _integer_counts(seen: ScalerSeenView, dest: _I64P, d: Int) raises:
    """`_integer_counts`: a native int scalar or an int64 row."""
    if seen.source == SCALER_SEEN_SOURCE_SCALAR and seen.integral:
        scaler_counts_i64(0, False, seen.integer, dest, d)
    elif seen.source == SCALER_SEEN_SOURCE_I64:
        scaler_counts_i64(seen.row, True, Int64(0), dest, d)
    else:
        raise Error("StandardScaler n_samples_seen_ must remain int64 counts")


def scaler_standard_merge_run[C: ScalerCalls](calls: C, d: Int, weights_given: Bool, old: ScalerSeenView,
                                              new: ScalerSeenView, kept: Bool, old_mean: Int, new_mean: Int,
                                              old_var: Int, new_var: Int) raises -> ScalerStandardMerged:
    """The merge of the fitted state (`old`, `old_mean`, `old_var`) with a
    batch's (`new`, `new_mean`, `new_var`); a row address is 0 when that row
    is None. `kept`: the fitted mean_ is not None (the batch's is then not
    None either). The result is weighted when sample weights were given or
    the fitted counts are weight sums."""
    var weighted = (weights_given or old.source == SCALER_SEEN_SOURCE_F32
                    or (old.source == SCALER_SEEN_SOURCE_SCALAR and not old.integral))
    var result = ScalerStandardMerged(d, kept)
    if not kept:
        # The Float64 addition and the publication of the count-only merge.
        var wide = List[Float64](length=2 * d, fill=0)
        var wp = _F64P(unsafe_from_address=Int(wide.unsafe_ptr()))
        scaler_counts_f64(old.row, old.source, old.scalar, wp, d)
        scaler_counts_f64(new.row, new.source, new.scalar, wp + d, d)
        if scaler_merge_count_only(wp, wp + d, d):
            if weighted:
                result.seen_kind = SCALER_SEEN_FLOAT
                result.seen_scalar = wp[0]
            else:
                result.seen_kind = SCALER_SEEN_INT
                result.seen_integer = Int64(Int(wp[0]))
        elif weighted:
            result.seen_kind = SCALER_SEEN_FLOAT_ROW
            result.seen_f32 = List[Float32](length=d, fill=0)
            scaler_narrow_counts(wp, Int(result.seen_f32.unsafe_ptr()), True, d)
        else:
            result.seen_kind = SCALER_SEEN_INT_ROW
            result.seen_i64 = List[Int64](length=d, fill=0)
            scaler_narrow_counts(wp, Int(result.seen_i64.unsafe_ptr()), False, d)
        _ = wide^
        return result^
    var arena = List[Float32](length=10 * d, fill=0)
    var ap = _F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    scaler_counts_f32(old.row, old.source, old.scalar, ap, d)
    scaler_counts_f32(new.row, new.source, new.scalar, ap + d, d)
    unsafe_memcpy(dest=ap + 2 * d, src=_F32P(unsafe_from_address=old_mean), count=d)
    unsafe_memcpy(dest=ap + 3 * d, src=_F32P(unsafe_from_address=new_mean), count=d)
    if old_var != 0:
        unsafe_memcpy(dest=ap + 4 * d, src=_F32P(unsafe_from_address=old_var), count=d)
    if new_var != 0:
        unsafe_memcpy(dest=ap + 5 * d, src=_F32P(unsafe_from_address=new_var), count=d)
    var program = List[Int32]()
    _stage(program, 91, d, 0, 2 * d, 4 * d, d, 3 * d, 5 * d, d, 1, 6 * d, 7 * d, 8 * d, 1)
    _stage(program, 102, d, 8 * d, 9 * d)
    _run_program(calls, arena, program)
    ap = _F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    unsafe_memcpy(dest=result.stats.unsafe_ptr(), src=ap + 7 * d, count=3 * d)
    if weighted:
        if scaler_equal_f32(ap + 6 * d, d):
            result.seen_kind = SCALER_SEEN_FLOAT
            result.seen_scalar = Float64(ap[6 * d])
        else:
            result.seen_kind = SCALER_SEEN_FLOAT_ROW
            result.seen_f32 = List[Float32](length=d, fill=0)
            unsafe_memcpy(dest=result.seen_f32.unsafe_ptr(), src=ap + 6 * d, count=d)
    else:
        # The exact int64 count addition (stage 269), not the float32 arena.
        var counts = List[Float32](length=6 * d, fill=0)
        var ip = _I64P(unsafe_from_address=Int(counts.unsafe_ptr()))
        _integer_counts(old, ip, d)
        _integer_counts(new, ip + d, d)
        var count_program = List[Int32]()
        _stage(count_program, 269, d, 0, 2 * d, 4 * d)
        _run_program(calls, counts, count_program)
        ip = _I64P(unsafe_from_address=Int(counts.unsafe_ptr()))
        if scaler_equal_i64(ip + 2 * d, d):
            result.seen_kind = SCALER_SEEN_INT
            result.seen_integer = ip[2 * d]
        else:
            result.seen_kind = SCALER_SEEN_INT_ROW
            result.seen_i64 = List[Int64](length=d, fill=0)
            unsafe_memcpy(dest=result.seen_i64.unsafe_ptr(), src=ip + 2 * d, count=d)
        _ = counts^
    _ = arena^
    return result^


def scaler_standard_partial_fit_run[C: ScalerCalls](calls: C, x: Int, n: Int, batch_d: Int, d: Int,
                                                    with_mean: Bool, with_std: Bool, weight: Int,
                                                    old: ScalerSeenView, old_mean: Int,
                                                    old_var: Int) raises -> ScalerStandardMerged:
    """StandardScaler.partial_fit of a fitted state on typed values: the
    batch (C-ordered (n, batch_d) Float32 at `x`; `weight` the address of n
    Float32 weights or 0) fitted by the typed core, the width refusal, then
    the merge. `old_mean` 0 means the fitted mean_ is None (count-only)."""
    var fit = scaler_standard_fit_core(calls, _F32P(unsafe_from_address=x), n, batch_d, with_mean, with_std, weight)
    if batch_d != d:
        raise Error("StandardScaler input feature count differs from fit")
    var scratch = List[Int64]()
    var new = scaler_seen_of_fit(fit, scratch, d)
    var stats = Int(fit.stats.unsafe_ptr())
    var merged = scaler_standard_merge_run(calls, d, weight != 0, old, new, old_mean != 0, old_mean, stats,
                                           old_var, stats + 4 * d if with_std else 0)
    _ = scratch^
    _ = fit^
    return merged^


# ---------------------------------------------------------------- the typed drivers as the core's calls

struct ScalerDriverCalls(ScalerCalls, Movable):
    """The typed scaler kernel driver and x_prep program driver
    (core/scaler_native_api.mojo) as the `ScalerCalls` of the scaler cores,
    for a caller holding the descriptor entries instead of the prep entry
    table (the native checks). `allow_direct` is the caller's
    MOJOLEARN_SCALER_DIRECT reading; the route is direct when the kernel
    entry also reports it (SCALER_OP_DIRECT)."""

    var scaler: ScalerDriver
    var program: PrepProgramDriver
    var use_direct: Bool

    def __init__(out self, scaler: ScalerDriver, program: PrepProgramDriver, allow_direct: Bool) raises:
        self.scaler = scaler.copy()
        self.program = program.copy()
        self.use_direct = allow_direct and scaler.metadata(SCALER_OP_DIRECT) == 1

    def direct(self) -> Bool:
        return self.use_direct

    def standard_fit(self, r: ScalerFitRequest) raises -> Int:
        return self.scaler.fit(r)

    def standard_transform(self, r: ScalerTransformRequest) raises -> Int:
        return self.scaler.transform(r)

    def minmax_fit(self, r: MinMaxFitRequest) raises -> Int:
        return self.scaler.minmax_fit(r)

    def minmax_transform(self, r: MinMaxTransformRequest) raises -> Int:
        return self.scaler.minmax_transform(r)

    def run(self, arena: Int, words: Int, program: Int, stages: Int) raises:
        self.program.run(PrepProgramRequest(arena, words, program, stages))
