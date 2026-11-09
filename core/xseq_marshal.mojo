# SPDX-License-Identifier: Apache-2.0
"""Host input marshaling and result publication of the sequence-expansion
native owner (bindings/xseq_native_common.mojo and its siblings): the
label-code conversion, constant learning-rate tables and the column / block
copies the fitted attributes are made of. The element casts of inputs into
float32 and int32 are the one admission path's (core/admitted_input_nd.mojo). Typed pointers in, no Python; the words are the ones the
NumPy expressions they replace produced."""

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]


# TOMBSTONE: xseq_cast_nd (admission helper, merged into the one path) deleted 2026-11-08 by lane a4-admit-sequence-tsa-neural; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A4_ADMISSION_HELPERS.patch. It is
# `admitted_cast_nd_f32` (core/admitted_input_nd.mojo): the same bits per
# item, row tasks across the host's cores instead of one serial odometer.


def xseq_fill_f32(dst: _F32P, count: Int, value: Float32):
    """`np.full(count, value, dtype=float32)` into `dst`."""
    for i in range(count):  # native input marshaling: one word per entry
        dst[i] = value


def xseq_codes_f32(src: _I32P, dst: _F32P, count: Int):
    """int32 class codes as the float32 target a recurrent loss reads."""
    for i in range(count):  # native input marshaling: one class code per row
        dst[i] = Float32(src[i])


def xseq_column_copy_f32(src: _F32P, dst: _F32P, rows: Int, cols: Int, col: Int):
    """`a[:, col]` of a C-ordered (rows, cols) float32 block."""
    for r in range(rows):  # native result publication: one value per row
        dst[r] = src[r * cols + col]


def xseq_column_copy_i64(src: _F32P, dst: _I64P, rows: Int, cols: Int, col: Int):
    """`a[:, col].astype(np.int64)` (truncation toward zero, as astype)."""
    for r in range(rows):  # native result publication: one value per row
        dst[r] = Int64(src[r * cols + col])


def xseq_var_blocks(params: _F32P, endog: _F32P, resid: _F32P, intercept: _F32P, coefs: _F32P,
                    fitted: _F32P, lags: Int, trend: Int, neqs: Int, rows: Int):
    """VARResults' derived arrays from the fitted (trend + K p, K) params:
    the intercept row, `coefs[i, a, c] = params[trend + i K + c, a]` (the lag
    blocks transposed) and `fittedvalues = endog[p:] - resid` in float32."""
    if trend:
        for j in range(neqs):  # native result publication: the intercept row
            intercept[j] = params[j]
    for i in range(lags):  # native result publication: one lag coefficient block per lag
        for a in range(neqs):  # native result publication: the block's rows
            for c in range(neqs):  # native result publication: the block's columns
                coefs[(i * neqs + a) * neqs + c] = params[(trend + i * neqs + c) * neqs + a]
    for r in range(rows):  # native result publication: one fitted row per observation
        for c in range(neqs):  # native result publication: one fitted value per variable
            fitted[r * neqs + c] = endog[(lags + r) * neqs + c] - resid[r * neqs + c]


# TOMBSTONE: neural_cast_i32_2d (admission helper, merged into the one path) deleted 2026-11-08 by lane a4-admit-sequence-tsa-neural; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A4_ADMISSION_HELPERS.patch. It is
# `admitted_ids_i32` (core/admitted_input_nd.mojo), row tasks across cores.


def neural_first_bad_id(ids: _I32P, rows: Int, cols: Int, vocab: Int, target_column: Int, ignore: Int) -> Int:
    """The flat index of the first id outside [0, vocab), or -1. Column
    `target_column` (when >= 0) feeds only the loss and may also hold
    `ignore`."""
    for r in range(rows):  # native input marshaling: input rows, never a model loop
        for c in range(cols):  # native input marshaling: one id per position
            var v = Int(ids[r * cols + c])
            if v >= 0 and v < vocab:
                continue
            if target_column >= 0 and c == target_column and v == ignore:
                continue
            return r * cols + c
    return -1


def tsa_flags_u8(src: _I32P, dst: _U8P, count: Int):
    """`flags.astype("<u1")` of the stationarity flags (each 0 or 1)."""
    for i in range(count):  # native result publication: one flag per series
        dst[i] = UInt8(src[i])


# TOMBSTONE: xseq_all_finite_f32 (admission helper, a second name of host_all_finite_f32) deleted 2026-11-08 by lane a4-admit-sequence-tsa-neural; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A4_ADMISSION_HELPERS.patch. Callers read
# `host_all_finite_f32` (core/host_finite_scan.mojo) directly.


def xseq_min_max_f32(src: _F32P, count: Int) -> Tuple[Float32, Float32]:
    """`(a.min(), a.max())` of `count` (finite, count >= 1) float32 words."""
    var lo = src[0]
    var hi = src[0]
    for i in range(1, count):
        var v = src[i]
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    return (lo, hi)


def xseq_widen_f32_f64(src: _F32P, dst: MutPointer[Float64, MutUntrackedOrigin], count: Int):
    """`a.astype("<f8")` of `count` float32 words: each widened exactly."""
    for i in range(count):
        dst[i] = Float64(src[i])


def xseq_take_native(base: Int, dims: List[Int], strides: List[Int], item: Int,
                      codes: _I64P, rows: Int, output: _U8P) raises:
    # ndarray.take(axis=None) for native numeric class storage: flatten in C
    # index order, preserve raw dtype bytes (including nonnative byte order).
    # Shape/strides are captured once; no provider lookups inside either loop.
    var total = 1
    for size in dims:  # driver-loop: admitted class-array rank metadata
        total *= size
    for row in range(rows):  # native result publication: validate all class codes
        var index = Int(codes[row])
        if index < -total or index >= total:
            raise Error("IndexError: class index is out of bounds")
    for row in range(rows):  # native result publication: copy selected label bytes
        var index = Int(codes[row])
        if index < 0:
            index += total
        var address = base
        for reverse in range(len(dims)):
            var axis = len(dims) - reverse - 1
            address += (index % dims[axis]) * strides[axis]
            index //= dims[axis]
        var source = _U8P(unsafe_from_address=address)
        for byte in range(item):
            output[row * item + byte] = source[byte]
