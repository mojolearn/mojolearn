# SPDX-License-Identifier: Apache-2.0
"""Host input marshaling and result publication of the sequence-expansion
native owner (bindings/xseq_native_common.mojo and its siblings): the
element casts into C-ordered float32, the label-code conversion, constant
learning-rate tables and the column / block copies the fitted attributes
are made of. Typed pointers in, no Python; the words are the ones the
NumPy expressions they replace produced."""
from std.memory import bitcast
from core.classical_cast import classical_f32_bits
from core.cast_in import cast_f32_bits, cast_i32_bits

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]


def xseq_cast_nd(base: Int, dims: List[Int], steps: List[Int], item: Int, kind: Int, double_ints: Bool,
                 dst: _F32P):
    """`np.array(a, dtype=float32, order="C")` of a strided array: element i
    (C order) of the array at `base` with byte `steps` per axis, cast by the
    shared integer-defined conversion (`cast_f32_bits`, NumPy's astype bits;
    `double_ints` the integer -> double -> float32 route of `Array.astype`)."""
    var rank = len(dims)
    var total = 1
    for k in range(rank):  # native input marshaling: the array's rank metadata
        total *= dims[k]
    var counter = List[Int](length=rank, fill=0)
    var offset = 0
    for i in range(total):  # native input marshaling: numeric buffer items, exact cast bits
        var bytes = _U8P(unsafe_from_address=base + offset)
        var raw = UInt64(0)
        for j in range(item):  # native input marshaling: the bytes of one item
            raw |= UInt64(bytes[j]) << UInt64(j * 8)
        var bits = classical_f32_bits(raw, kind, item) if double_ints else cast_f32_bits(raw, kind, item)
        dst[i] = bitcast[DType.float32](bits)
        var axis = rank - 1
        while axis >= 0:  # native input marshaling: the next element's byte offset
            counter[axis] += 1
            offset += steps[axis]
            if counter[axis] < dims[axis]:
                break
            offset -= steps[axis] * dims[axis]
            counter[axis] = 0
            axis -= 1


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


def neural_cast_i32_2d(addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, kind: Int,
                       dst: _I32P):
    """A strided (rows, cols) integer or bool array as C-ordered int32 words
    (`cast_i32_bits`: exact, saturated outside int32)."""
    for r in range(rows):  # native input marshaling: input rows, never a model loop
        for c in range(cols):  # native input marshaling: one id per position
            var bytes = _U8P(unsafe_from_address=addr + r * s0 + c * s1)
            var raw = UInt64(0)
            for j in range(item):  # native input marshaling: the bytes of one item
                raw |= UInt64(bytes[j]) << UInt64(j * 8)
            dst[r * cols + c] = bitcast[DType.int32](cast_i32_bits(raw, kind, item))


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


def xseq_all_finite_f32(src: _F32P, count: Int) -> Bool:
    """`all_finite(a)` of `count` float32 words: no NaN and no infinity."""
    for i in range(count):
        var bits = bitcast[DType.uint32](src[i])
        if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
            return False
    return True


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
