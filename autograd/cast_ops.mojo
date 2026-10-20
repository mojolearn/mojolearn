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
# Codes OP_CAST_FIRST..OP_CAST_LAST (3500..3501) are this family;
# sequence/dispatch.mojo routes and fences them.
from std.memory import bitcast, memcpy
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from core.cast_in import CastIn, cast_f32_bits, cast_raw_bits

comptime OP_CAST_F32 = 3500
comptime OP_CAST_GATHER = 3501
comptime OP_CAST_FIRST = OP_CAST_F32
comptime OP_CAST_LAST = OP_CAST_GATHER

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
def apply_cast[OP: Int](t: Int, a: Args):
    comptime if OP == OP_CAST_F32:
        op_cast_f32(t, a)
    elif OP == OP_CAST_GATHER:
        op_cast_gather(t, a)


# ---------------------------------------------------------------- drivers
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
