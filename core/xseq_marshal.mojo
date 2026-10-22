# SPDX-License-Identifier: Apache-2.0
"""Host input marshaling and result publication of the sequence-expansion
native owner (bindings/xseq_native_common.mojo and its siblings): the
element casts into C-ordered float32, the label-code conversion, constant
learning-rate tables and the column / block copies the fitted attributes
are made of. Typed pointers in, no Python; the words are the ones the
NumPy expressions they replace produced."""
from std.memory import bitcast
from bindings.classical_cast import classical_f32_bits
from core.cast_in import cast_f32_bits

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
