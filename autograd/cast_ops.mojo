# SPDX-License-Identifier: Apache-2.0
# THE INPUT CAST AND THE LABEL GATHER (lane ag-thin-glue, 2026-10-19): the
# caller's array goes to the executor as it is (its bytes, one upload) and
# one launch writes its float32 row-major copy there; predicted class codes
# become the caller's labels by one gather over classes_ on the executor.
# Python passes buffers only (AGENTS.md "No Python in the runtime").
#
# Shared by HostExec and DeviceExec through sequence/dispatch.mojo, so the
# host column and every GPU run the same bodies; the conversions are
# core/cast_in.mojo's integer-only statements, so the words are the same
# on Apple, NVIDIA, AMD and the host. A float32 C-order input takes no
# launch at all: its words are uploaded as before (no bit moves).
#
# Codes OP_CAST_FIRST..OP_CAST_LAST (3500..3502) are this family;
# sequence/dispatch.mojo routes and fences them.
from std.memory import bitcast, memcpy
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from core.cast_in import (
    CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CastIn, cast_f32_bits, cast_i32_bits, cast_raw_bits,
    cast_w64_bits, label_kind_of,
)
from core.label_encode import label_sort_key, label_is_nan
from autograd.embedding_ops import embedding_check_words, embedding_refuse_ids

comptime OP_CAST_F32 = 3500
comptime OP_CAST_GATHER = 3501
# lane ag-thin-glue2: integer ids (any integer dtype, any strides) to the
# int32 words the embedding op reads
comptime OP_CAST_I32 = 3502
comptime OP_CAST_LABEL_CODES = 3503
comptime OP_CAST_FIRST = OP_CAST_F32
comptime OP_CAST_LAST = OP_CAST_LABEL_CODES

comptime _BP = MutPointer[UInt8, MutUntrackedOrigin]


# ---------------------------------------------------------------- bodies
@always_inline
def op_cast_f32(t: Int, a: Args):
    """dst[t] = float32 of the caller's row-major element t. p0 the
    caller's bytes in words, p1 dst; i0 kind, i1 item size, i2 cols, i3 the
    row stride and i4 the column stride in bytes (signed), i5 element
    (0, 0)'s byte offset in the words (CastIn.origin)."""
    var r = t // a.i2
    var c = t - r * a.i2
    var raw = cast_raw_bits(a.p0, a.i5 + r * a.i3 + c * a.i4, a.i1)
    a.p1.unsafe_store(t, bitcast[DType.float32](cast_f32_bits(raw, a.i0, a.i1)))


@always_inline
def op_cast_i32(t: Int, a: Args):
    """dst[t] = the int32 bits (core/cast_in.mojo cast_i32_bits, saturated)
    of the caller's row-major integer element t, in one float32 word. The
    slots are op_cast_f32's."""
    var r = t // a.i2
    var c = t - r * a.i2
    var raw = cast_raw_bits(a.p0, a.i5 + r * a.i3 + c * a.i4, a.i1)
    a.p1.unsafe_store(t, bitcast[DType.float32](cast_i32_bits(raw, a.i0, a.i1)))


@always_inline
def _code(a: Args, i: Int) -> Int:
    """Row i's class code, clamped into [0, k): the codes come from the
    model's own argmax or threshold, so the clamp only fences the read."""
    var c = Int(a.p0.unsafe_load(i))
    if c < 0:
        return 0
    if c >= a.i2:
        return a.i2 - 1
    return c


@always_inline
def op_cast_gather(t: Int, a: Args):
    """Word t of out = the labels classes[codes[i]] it holds. p0 the n codes
    (float32 values), p1 the k classes' bytes in words, p2 out; i0 n, i1
    item size, i2 k. One element per output word, so a 1- or 2-byte label
    never shares a store with another thread."""
    var size = a.i1
    if size == 8:
        var i = t >> 1
        a.p2.unsafe_store(t, a.p1.unsafe_load(_code(a, i) * 2 + (t & 1)))
        return
    var per = 4 // size
    var acc = UInt64(0)
    for j in range(per):  # the 1, 2 or 4 labels of this word
        var i = t * per + j
        if i < a.i0:
            var v = cast_raw_bits(a.p1, _code(a, i) * size, size)
            acc |= v << UInt64(j * size * 8)
    a.p2.unsafe_store(t, bitcast[DType.float32](UInt32(acc & UInt64(0xFFFFFFFF))))


@always_inline
def op_cast_label_codes(t: Int, a: Args):
    # Each validation label searches the sorted training vocabulary. Integer
    # keys preserve int64 labels and float64 labels exactly on every vendor.
    var value = cast_w64_bits(cast_raw_bits(a.p0, a.i4 + t * a.i3, a.i1), a.i0, a.i1)
    # Mixed signed/unsigned validation labels compare in the training
    # vocabulary's integer domain, refusing only values outside that domain.
    if a.i2 == 2 and a.i0 == 1 and (value >> 63) != 0:
        a.p2.unsafe_store(t, bitcast[DType.float32](Int32(-1)))
        return
    if a.i2 == 1 and a.i0 == 0 and a.i1 == 8 and (value >> 63) != 0:
        a.p2.unsafe_store(t, bitcast[DType.float32](Int32(-1)))
        return
    var key = label_sort_key(value, a.i2)
    var lo = 0
    var hi = a.i8
    while lo < hi:
        var mid = (lo + hi) // 2
        var bits = cast_w64_bits(cast_raw_bits(a.p1, a.i9 + mid * a.i7, a.i6), a.i5, a.i6)
        if label_sort_key(bits, a.i2) < key:
            lo = mid + 1
        else:
            hi = mid
    var code = -1
    if lo < a.i8 and not label_is_nan(value, a.i2):
        var bits = cast_w64_bits(cast_raw_bits(a.p1, a.i9 + lo * a.i7, a.i6), a.i5, a.i6)
        if label_sort_key(bits, a.i2) == key:
            code = lo
    a.p2.unsafe_store(t, bitcast[DType.float32](Int32(code)))


@always_inline
def apply_cast[OP: Int](t: Int, a: Args):
    comptime if OP == OP_CAST_F32:
        op_cast_f32(t, a)
    elif OP == OP_CAST_GATHER:
        op_cast_gather(t, a)
    elif OP == OP_CAST_I32:
        op_cast_i32(t, a)
    elif OP == OP_CAST_LABEL_CODES:
        op_cast_label_codes(t, a)


# ---------------------------------------------------------------- drivers
def validation_codes[E: Exec](mut ex: E, src: CastIn, classes: CastIn, dst: FP) raises:
    src.check("validation labels")
    classes.check("training classes")
    if src.cols != 1 or classes.cols != 1:
        raise Error("validation labels and training classes must be vectors")
    var src_kind = label_kind_of(src.kind, src.itemsize, "validation labels")
    var kind = label_kind_of(classes.kind, classes.itemsize, "training classes")
    if (kind == 0) != (src_kind == 0):
        raise Error("validation labels must use the training labels' integer or floating kind")
    var raw = ex.alloc(src.words())
    var cls = ex.alloc(classes.words())
    upload_bytes(ex, raw, src.start(), src.nbytes())
    upload_bytes(ex, cls, classes.start(), classes.nbytes())
    var a = Args()
    a.p0 = raw
    a.p1 = cls
    a.p2 = dst
    a.i0 = src.kind
    a.i1 = src.itemsize
    a.i2 = kind
    a.i3 = src.s0
    a.i4 = src.origin()
    a.i5 = classes.kind
    a.i6 = classes.itemsize
    a.i7 = classes.s0
    a.i8 = classes.rows
    a.i9 = classes.origin()
    ex.launch[OP_CAST_LABEL_CODES](a, src.rows)
    var sa = ex.alloc(embedding_check_words(src.rows))
    var sb = ex.alloc(embedding_check_words(src.rows))
    try:
        embedding_refuse_ids(ex, dst, src.rows, classes.rows, sa, sb)
    except:
        raise Error("validation labels contain an unknown class or NaN")
    ex.free(raw)
    ex.free(cls)
    ex.free(sa)
    ex.free(sb)


def upload_bytes[E: Exec](mut ex: E, dst: FP, addr: Int, nbytes: Int) raises:
    """The nbytes at host `addr` into the executor words at dst; a 1..3-byte
    tail travels zero-padded in one more word (no read past the caller's
    buffer)."""
    var full = nbytes // 4
    var rem = nbytes - full * 4
    if full > 0:
        ex.upload(dst, FP(unsafe_from_address=addr), full)
    if rem > 0:
        var tail = List[Float32](length=1, fill=Float32(0))
        memcpy(dest=_BP(unsafe_from_address=Int(tail.unsafe_ptr())),
               src=_BP(unsafe_from_address=addr + full * 4), count=rem)
        ex.upload(dst + full, FP(unsafe_from_address=Int(tail.unsafe_ptr())), 1)
        _ = tail^


def download_bytes[E: Exec](mut ex: E, addr: Int, src: FP, nbytes: Int) raises:
    """The first nbytes of the executor words at src into host `addr` (the
    tail word's spare bytes are not written)."""
    var full = nbytes // 4
    var rem = nbytes - full * 4
    if full > 0:
        ex.download(FP(unsafe_from_address=addr), src, full)
    if rem > 0:
        var tail = List[Float32](length=1, fill=Float32(0))
        ex.download(FP(unsafe_from_address=Int(tail.unsafe_ptr())), src + full, 1)
        memcpy(dest=_BP(unsafe_from_address=addr + full * 4),
               src=_BP(unsafe_from_address=Int(tail.unsafe_ptr())), count=rem)
        _ = tail^


def upload_cast[E: Exec](mut ex: E, dst: FP, src: CastIn) raises:
    """dst (src.count() floats, row-major (rows, cols)) = the caller's array
    as float32. float32 in C order is one plain upload (the words as they
    are); every other accepted array is one upload of its bytes and one
    OP_CAST_F32 launch. `src` must have passed CastIn.check."""
    if src.is_f32_c():
        ex.upload(dst, FP(unsafe_from_address=src.addr), src.count())
        return
    var raw = ex.alloc(src.words())
    try:
        upload_bytes(ex, raw, src.start(), src.nbytes())
        var a = Args()
        a.p0 = raw
        a.p1 = dst
        a.i0 = src.kind
        a.i1 = src.itemsize
        a.i2 = src.cols
        a.i3 = src.s0
        a.i4 = src.s1
        a.i5 = src.origin()
        ex.launch[OP_CAST_F32](a, src.count())
    except error:
        ex.free(raw)
        raise error
    ex.free(raw)


def upload_cast_i32[E: Exec](mut ex: E, dst: FP, src: CastIn, what: String) raises:
    """dst (src.count() words, row-major) = the caller's integer array as
    int32 bits (values outside int32 saturate; see cast_i32_bits). int32 in
    C order is one plain upload (the words as they are); every other integer
    array is one upload of its bytes and one OP_CAST_I32 launch. `src` must
    have passed CastIn.check."""
    if src.kind != CAST_KIND_SIGNED and src.kind != CAST_KIND_UNSIGNED:
        raise Error(what + " must be integers")
    if src.kind == CAST_KIND_SIGNED and src.itemsize == 4 and src.is_c_order():
        ex.upload(dst, FP(unsafe_from_address=src.addr), src.count())
        return
    var raw = ex.alloc(src.words())
    try:
        upload_bytes(ex, raw, src.start(), src.nbytes())
        var a = Args()
        a.p0 = raw
        a.p1 = dst
        a.i0 = src.kind
        a.i1 = src.itemsize
        a.i2 = src.cols
        a.i3 = src.s0
        a.i4 = src.s1
        a.i5 = src.origin()
        ex.launch[OP_CAST_I32](a, src.count())
    except error:
        ex.free(raw)
        raise error
    ex.free(raw)


def gather_labels[E: Exec](
    mut ex: E, codes_addr: Int, n: Int, classes_addr: Int, k: Int, itemsize: Int, out_addr: Int,
) raises:
    """out[i] = classes[codes[i]] for n float32 codes at host codes_addr,
    k classes of `itemsize` bytes at host classes_addr, n items into host
    out_addr: one upload each, one OP_CAST_GATHER launch, one download."""
    if n < 1 or k < 1:
        raise Error("model_gather_labels: n and k must be positive")
    if itemsize != 1 and itemsize != 2 and itemsize != 4 and itemsize != 8:
        raise Error("model_gather_labels: item size must be 1, 2, 4 or 8")
    if codes_addr == 0 or classes_addr == 0 or out_addr == 0:
        raise Error("model_gather_labels: null buffer address")
    var dcodes = ex.alloc(n)
    ex.upload(dcodes, FP(unsafe_from_address=codes_addr), n)
    var dcls = ex.alloc((k * itemsize + 3) // 4)
    upload_bytes(ex, dcls, classes_addr, k * itemsize)
    var out_words = (n * itemsize + 3) // 4
    var dout = ex.alloc(out_words)
    var a = Args()
    a.p0 = dcodes
    a.p1 = dcls
    a.p2 = dout
    a.i0 = n
    a.i1 = itemsize
    a.i2 = k
    ex.launch[OP_CAST_GATHER](a, out_words)
    download_bytes(ex, out_addr, dout, n * itemsize)
    ex.free(dcodes)
    ex.free(dcls)
    ex.free(dout)
