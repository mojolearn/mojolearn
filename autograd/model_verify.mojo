# SPDX-License-Identifier: Apache-2.0
"""The numbers of a model verification report (python/mojolearn/_verify_model.py,
`python -m mojolearn verify-model`).

A report says which model ran (the file's SHA-256 and its description hash),
on which inputs (each input's digest) and what came out (each output's
digest), so a report written on one install can be compared with one written
on another. Python formats the report; every number in it comes from here:

  verify_digest_py           FNV-1a64 over a host buffer, the identity digest
                             (core/identity_trace.mojo fnv1a64_bytes, called,
                             not restated), as sixteen hex digits
  verify_file_sha256_py      SHA-256 of a whole file (FIPS 180-4), 64 hex digits
  verify_model_info_py       the kind of a model file and its description hash
  verify_fill_py             the seeded inputs of a report, Philox4x32-10
                             (core/philox.mojo) per element

All of it is host side: a report is computed over files and over the host
arrays a run returned, after the run (cpu-route: verification digests over
host files and returned arrays, outside any fit or predict).
"""
from std.python import Python, PythonObject
from core.identity_trace import FNV_OFFSET, _hex16, fnv1a64_bytes
from core.philox import philox4x32_10
from sequence.ops import FP
from autograd.model_checkpoint import _seq_record
from training.checkpoint import (
    CKPT_SECTION_PROGRAM, CKPT_SECTION_SEQUENTIAL, read_checkpoint_unchecked,
)

comptime _BP = UnsafePointer[UInt8, MutUntrackedOrigin]

# verify_model_info_py's kind names
comptime VERIFY_KIND_PROGRAM = "program"
comptime VERIFY_KIND_SEQUENTIAL = "sequential"
comptime VERIFY_KIND_SEQUENTIAL_EXPORT = "sequential-export"


# ---------------------------------------------------------------- FNV-1a64

def _fold_addr(seed: UInt64, addr: Int, nbytes: Int) -> UInt64:
    if nbytes <= 0:
        return seed
    return fnv1a64_bytes(seed, _BP(unsafe_from_address=addr), nbytes)


def _fold_i64(seed: UInt64, xs: List[Int64]) -> UInt64:
    # the words' little-endian bytes, in order
    var h = _fold_addr(seed, Int(xs.unsafe_ptr()), 8 * len(xs))
    _ = len(xs)  # hold the list past the fold (Mojo frees a local at its last use)
    return h


def _fold_f32(seed: UInt64, xs: List[Float32]) -> UInt64:
    # the floats' bits, in order
    var h = _fold_addr(seed, Int(xs.unsafe_ptr()), 4 * len(xs))
    _ = len(xs)
    return h


def verify_digest_py(addr: PythonObject, nbytes: PythonObject) raises -> PythonObject:
    """FNV-1a64 over `nbytes` bytes at host address `addr`, as sixteen hex
    digits. The caller keeps the buffer alive across the call."""
    var a = Int(py=addr)
    var n = Int(py=nbytes)
    if n < 0:
        raise Error("mojolearn: verify_digest takes a byte count >= 0")
    if n > 0 and a == 0:
        raise Error("mojolearn: verify_digest was given a null buffer")
    # cpu-route: verification digest over a returned host array
    return PythonObject(_hex16(_fold_addr(FNV_OFFSET, a, n)))


# ---------------------------------------------------------------- SHA-256

def _sha256_k() -> List[UInt32]:
    var k: List[UInt32] = [
        0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
        0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
        0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
        0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
        0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
        0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
        0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
        0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
    ]
    return k^


@always_inline
def _rotr(x: UInt32, n: UInt32) -> UInt32:
    return (x >> n) | (x << (UInt32(32) - n))


def _sha256_block(mut h: List[UInt32], k: List[UInt32], mut w: List[UInt32], src: List[UInt8], off: Int):
    """One 64-byte block of `src` at `off` into the state `h`. UInt32
    arithmetic wraps, as SHA-256 requires."""
    for t in range(16):  # the block's message words
        var o = off + 4 * t
        w[t] = (
            (src[o].cast[DType.uint32]() << 24)
            | (src[o + 1].cast[DType.uint32]() << 16)
            | (src[o + 2].cast[DType.uint32]() << 8)
            | src[o + 3].cast[DType.uint32]()
        )
    for t in range(16, 64):  # the message schedule
        var s0 = _rotr(w[t - 15], 7) ^ _rotr(w[t - 15], 18) ^ (w[t - 15] >> 3)
        var s1 = _rotr(w[t - 2], 17) ^ _rotr(w[t - 2], 19) ^ (w[t - 2] >> 10)
        w[t] = w[t - 16] + s0 + w[t - 7] + s1
    var a = h[0]
    var b = h[1]
    var c = h[2]
    var d = h[3]
    var e = h[4]
    var f = h[5]
    var g = h[6]
    var hh = h[7]
    for t in range(64):  # the compression rounds
        var big1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        var ch = (e & f) ^ ((~e) & g)
        var t1 = hh + big1 + ch + k[t] + w[t]
        var big0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        var maj = (a & b) ^ (a & c) ^ (b & c)
        var t2 = big0 + maj
        hh = g
        g = f
        f = e
        e = d + t1
        d = c
        c = b
        b = a
        a = t1 + t2
    h[0] = h[0] + a
    h[1] = h[1] + b
    h[2] = h[2] + c
    h[3] = h[3] + d
    h[4] = h[4] + e
    h[5] = h[5] + f
    h[6] = h[6] + g
    h[7] = h[7] + hh


def _hex32(v: UInt32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var out = String("")
    for i in range(8):  # the word's hex digits
        var nib = Int((v >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def sha256_hex(data: List[UInt8]) -> String:
    """SHA-256 of `data`, 64 lowercase hex digits."""
    var h: List[UInt32] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    ]
    var k = _sha256_k()
    var w = List[UInt32](length=64, fill=UInt32(0))
    var n = len(data)
    var full = n // 64
    for blk in range(full):  # cpu-route: the model file hashed on the host, block by block
        _sha256_block(h, k, w, data, 64 * blk)
    # the tail, the 0x80 marker, zeros and the message length in bits (big-endian)
    var tail = List[UInt8]()
    for i in range(64 * full, n):  # the last partial block
        tail.append(data[i])
    tail.append(UInt8(0x80))
    while len(tail) % 64 != 56:  # padding bytes
        tail.append(UInt8(0))
    var bits = UInt64(n) * UInt64(8)
    for i in range(8):  # the length bytes
        tail.append(((bits >> UInt64(56 - 8 * i)) & UInt64(0xFF)).cast[DType.uint8]())
    for blk in range(len(tail) // 64):  # the padded tail blocks
        _sha256_block(h, k, w, tail, 64 * blk)
    var out = String("")
    for i in range(8):  # the state words
        out += _hex32(h[i])
    return out


def _read_file(path: String) raises -> List[UInt8]:
    # training/checkpoint.mojo's binary read idiom
    var raw = List[UInt8]()
    with open(path, "r") as fh:
        var body = fh.read_bytes()
        for i in range(len(body)):  # cpu-route: the model file read on the host
            raw.append(body[i])
    return raw^


def verify_file_sha256_py(path: PythonObject) raises -> PythonObject:
    """[the SHA-256 of the file at `path` (64 hex digits), its byte count]."""
    var data = _read_file(String(py=path))
    var out = Python.list()
    out.append(PythonObject(sha256_hex(data)))
    out.append(PythonObject(len(data)))
    return out


# ---------------------------------------------------------------- model files

def verify_model_info_py(path: PythonObject) raises -> PythonObject:
    """[kind, description hash] of a saved model file.

    kind is "program" (Program.save), "sequential" (Sequential.save_checkpoint)
    or "sequential-export" (Sequential.export). The description hash is
    FNV-1a64 over what the model computes, without its parameter values or
    training state: a program's description words (autograd/program.mojo
    program_words: inputs, parameter shapes, every node and its arguments,
    outputs), or a Sequential model's description words and layer table. So
    a checkpoint and an export of one model share it, and a model whose
    weights changed keeps it while its file hash changes."""
    var file = String(py=path)
    var ck = read_checkpoint_unchecked(file)
    var kind: String
    var h = FNV_OFFSET
    if ck.section_kind == CKPT_SECTION_PROGRAM:
        kind = String(VERIFY_KIND_PROGRAM)
        h = _fold_i64(h, ck.ext_ints)
        h = _fold_f32(h, ck.ext_floats)
    elif ck.section_kind == CKPT_SECTION_SEQUENTIAL:
        var r = _seq_record(ck, file)
        kind = String(VERIFY_KIND_SEQUENTIAL_EXPORT) if r.is_export else String(VERIFY_KIND_SEQUENTIAL)
        var words = List[Int64]()
        for i in range(len(r.prog)):  # driver-loop: over description words
            words.append(Int64(r.prog[i]))
        h = _fold_i64(h, words)
        h = _fold_f32(h, r.table)
    else:
        raise Error("mojolearn: '" + file + "' is not a saved model (a Program, Sequential checkpoint "
                    + "or Sequential export)")
    var out = Python.list()
    out.append(PythonObject(kind))
    out.append(PythonObject(_hex16(h)))
    return out


# ---------------------------------------------------------------- seeded inputs

def verify_fill_py(addr: PythonObject, n: PythonObject, seed: PythonObject, stream: PythonObject) raises -> PythonObject:
    """Fill `n` float32 values at `addr` with the report's seeded inputs:
    value i of input `stream` is Philox4x32-10 at counter (i low, i high,
    stream, 0) under key `seed`, its first word's top 24 bits u, as
    u * 2^-23 - 1 in [-1, 1). Every step is exact in float32, so the values
    are the same bits on every install."""
    var count = Int(py=n)
    var s = Int(py=seed)
    var st = Int(py=stream)
    if count < 0 or s < 0 or st < 0 or st > 0xFFFFFFFF:
        raise Error("mojolearn: verify_fill takes n >= 0, seed >= 0 and a stream below 2^32")
    if count == 0:
        return PythonObject(0)
    var p = FP(unsafe_from_address=Int(py=addr))
    var su = UInt64(s)
    var key = SIMD[DType.uint32, 2](UInt32(su & 0xFFFFFFFF), UInt32((su >> 32) & 0xFFFFFFFF))
    var step = Float32(1.1920928955078125e-07)  # 2^-23
    for i in range(count):  # cpu-route: verification inputs made on the host before the run
        var iu = UInt64(i)
        var ctr = SIMD[DType.uint32, 4](
            UInt32(iu & 0xFFFFFFFF), UInt32((iu >> 32) & 0xFFFFFFFF), UInt32(st), UInt32(0)
        )
        var r = philox4x32_10(ctr, key)
        var u = r[0] >> 8
        p[i] = Float32(Int(u)) * step - Float32(1.0)
    return PythonObject(count)
