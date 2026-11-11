# SPDX-License-Identifier: Apache-2.0
"""GP target normalization using the existing StandardScaler stages.

The provider and x_prep descriptors belong to their original images. This
plan replaces the temporary Python StandardScaler, its fitted fields and
prepared-row cache. The direct switch, missing-value programs and separate
fit/transform kernels are unchanged. Callers own the input/output spans.
"""
from std.memory import unsafe_memcpy, bitcast
from core.scaler_native_api import ScalerDriver, PrepProgramDriver, ScalerFitRequest, ScalerTransformRequest, PrepProgramRequest
from core.preprocessing_program import prep_program_stage as _stage
from core.scaler_fit_impl import scaler_all_finite, scaler_check_stats, scaler_check_missing, scaler_feature_indices, scaler_refuse_infinite_columns

alias GPNormF32P = MutPointer[Float32, MutUntrackedOrigin]


@fieldwise_init
struct GPRNormalizeRequest(Copyable, Movable):
    var source: Int
    var dst: Int
    var rows: Int
    var direct: Bool


@fieldwise_init
struct GPRNormalizeResult(Copyable, Movable):
    var mean: Float64
    var stddev: Float64


def _gp_prep_run(driver: PrepProgramDriver, mut arena: List[Float32], mut program: List[Int32]) raises:
    driver.run(PrepProgramRequest(Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), len(program) // 16))
    _ = arena
    _ = program


def _gp_nan_fill(source: GPNormF32P, n: Int, program_driver: PrepProgramDriver) raises -> List[Float32]:
    # The original scaler _nan_fill offsets and exact program, d=1.
    var st = n
    var keep = st + 6
    var stat = keep + 1
    var dest = stat + 1
    var arena = List[Float32](length=dest+n, fill=0)
    var ap = GPNormF32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    unsafe_memcpy(dest=ap, src=source, count=n)
    scaler_feature_indices(ap+keep, 1)
    var program = List[Int32]()
    _stage(program, 1, 1, 0, n, 1, st)
    _stage(program, 24, n, 0, n, 1, stat, dest, keep, 1)
    _gp_prep_run(program_driver, arena, program)
    scaler_refuse_infinite_columns(ap, st, 1)
    var values = List[Float32](length=n, fill=0)
    unsafe_memcpy(dest=values.unsafe_ptr(), src=ap+dest, count=n)
    _ = arena
    return values^


def gp_normalize_targets(scaler: ScalerDriver, program_driver: PrepProgramDriver,
                         r: GPRNormalizeRequest) raises -> GPRNormalizeResult:
    if r.rows < 1:
        raise Error("Scaler input must contain at least one sample and feature")
    if r.source == 0 or r.dst == 0:
        raise Error("GP normalization: nonempty target spans require storage")
    var n = r.rows
    # Preserve targets -> column and transformed column -> flat copies.
    var column = List[Float32](length=n, fill=0)
    var xp = GPNormF32P(unsafe_from_address=Int(column.unsafe_ptr()))
    unsafe_memcpy(dest=xp, src=GPNormF32P(unsafe_from_address=r.source), count=n)
    var stats = List[Float32](length=3, fill=0)
    var sp = GPNormF32P(unsafe_from_address=Int(stats.unsafe_ptr()))
    var finite = True
    if r.direct:
        finite = scaler.fit(ScalerFitRequest(Int(xp), Int(sp), n, 1, True, True, True)) != 0
    else:
        finite = scaler_all_finite(xp, n)
        if finite:
            _ = scaler.fit(ScalerFitRequest(Int(xp), Int(sp), n, 1, True, True, False))
    var fitted = List[Float32](length=3, fill=0)
    var fp = GPNormF32P(unsafe_from_address=Int(fitted.unsafe_ptr()))
    if finite:
        scaler_check_stats(sp, sp+1, sp+2, 1)
        unsafe_memcpy(dest=fp, src=sp, count=1)
        unsafe_memcpy(dest=fp+1, src=sp+1, count=1)
        unsafe_memcpy(dest=fp+2, src=sp+2, count=1)
    else:
        # Original unweighted missing-value fit, d=1; no weighted branch
        # exists for the private GP StandardScaler.
        var st = n
        var dst = st + 6
        var sc = dst + 3
        var size = sc + 1
        if size > 2147483647:
            raise Error("x_prep: the program exceeds the native Int32 indexing bound")
        var arena = List[Float32](length=size, fill=0)
        var ap = GPNormF32P(unsafe_from_address=Int(arena.unsafe_ptr()))
        unsafe_memcpy(dest=ap, src=xp, count=n)
        var program = List[Int32]()
        _stage(program, 1, 1, 0, n, 1, st)
        _stage(program, 101, 1, 0, n, 1, -1, dst)
        _stage(program, 102, 1, dst+2, sc)
        _gp_prep_run(program_driver, arena, program)
        _ = scaler_check_missing(ap, st, dst, sc, 1)
        unsafe_memcpy(dest=fp, src=ap+dst+1, count=1)
        unsafe_memcpy(dest=fp+1, src=ap+dst+2, count=1)
        unsafe_memcpy(dest=fp+2, src=ap+sc, count=1)
        _ = arena
    var mean = Float64(fp[0])
    var stddev = Float64(fp[2])
    # Same transform-statistic preparation and validation program. The old
    # one-use prepared Python cache has no consumer after this private call.
    var rows = List[Float32](length=12, fill=0)
    var rp = GPNormF32P(unsafe_from_address=Int(rows.unsafe_ptr()))
    unsafe_memcpy(dest=rp, src=fp, count=1)
    unsafe_memcpy(dest=rp+1, src=fp+2, count=1)
    rows[11] = 1
    var row_program = List[Int32]()
    _stage(row_program, 264, 1, 0, 1, 1, 1, 10, 11, 2, 3, 4, 5, 6)
    _stage(row_program, 248, 1, 4, 1, 7)
    _stage(row_program, 248, 1, 5, 1, 8)
    _stage(row_program, 248, 1, 6, 1, 9)
    _gp_prep_run(program_driver, rows, row_program)
    if bitcast[DType.int32](rows[8]) != 0:
        raise Error("StandardScaler mean_ and scale_ must remain finite (NaN: a feature never seen)")
    if bitcast[DType.int32](rows[9]) != 0:
        raise Error("StandardScaler scale_ must remain positive")
    var colnan = bitcast[DType.int32](rows[7]) != 0
    var output = List[Float32](length=n, fill=0)
    var op = GPNormF32P(unsafe_from_address=Int(output.unsafe_ptr()))
    finite = True
    if r.direct:
        var got = scaler.transform(ScalerTransformRequest(Int(xp), Int(rp+2), Int(rp+3), Int(op), n, 1, False, True, True, True))
        if got == 0:
            raise Error("StandardScaler transform overflowed in Float32")
        finite = got == 1
        if not finite:
            var filled = _gp_nan_fill(xp, n, program_driver)
            if scaler.transform(ScalerTransformRequest(Int(filled.unsafe_ptr()), Int(rp+2), Int(rp+3), Int(op), n, 1, False, True, True, True)) != 1:
                raise Error("StandardScaler transform overflowed in Float32")
            _ = filled
    else:
        finite = scaler_all_finite(xp, n)
        var filled = List[Float32]()
        var source = Int(xp)
        if not finite:
            filled = _gp_nan_fill(xp, n, program_driver)
            source = Int(filled.unsafe_ptr())
        _ = scaler.transform(ScalerTransformRequest(source, Int(rp+2), Int(rp+3), Int(op), n, 1, False, True, True, False))
        if not scaler_all_finite(op, n):
            raise Error("StandardScaler transform overflowed in Float32")
        _ = filled
    if not finite or colnan:
        var keep = List[Float32](length=3*n+2, fill=0)
        var kp = GPNormF32P(unsafe_from_address=Int(keep.unsafe_ptr()))
        unsafe_memcpy(dest=kp, src=xp, count=n)
        unsafe_memcpy(dest=kp+n, src=op, count=n)
        unsafe_memcpy(dest=kp+2*n, src=rp+4, count=1)
        var restore = List[Int32]()
        _stage(restore, 10, 1, 2*n, 2*n+1)
        _stage(restore, 103, n, 0, 1, n, 2*n+1, 2*n+2)
        _gp_prep_run(program_driver, keep, restore)
        unsafe_memcpy(dest=op, src=kp+2*n+2, count=n)
        _ = keep
    unsafe_memcpy(dest=GPNormF32P(unsafe_from_address=r.dst), src=op, count=n)
    _ = output
    _ = rows
    _ = fitted
    _ = stats
    _ = column
    return GPRNormalizeResult(mean, stddev)
