# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CROSS-CHECK AGAINST THE SIMULATION: the fifteen-bit profile's host
oracle, and the device, against the quality lane's PyTorch arithmetic, bit
for bit.

    pixi run check-gemm-int15-sim                  every gate passes
    pixi run check-gemm-int15-sim-host-sabotage    MUST FAIL the host product gate
    pixi run check-gemm-int15-sim-convert-sabotage MUST FAIL the host codes gate

Contract `gemm/IDENTICAL_LOWBIT_CONTRACT.md` section 6.6. Lane
lane/lowbit-int15, 2026-09-29.

WHY. The quality lane measured held-out perplexity under the arithmetic of
`bench/lowbit_quality/arith.py`, a simulation in PyTorch with float64
integers. What ships is `gemm/host/gemm_int15_oracle.mojo` and the kernels
held to it. This check is what makes the two ONE arithmetic: on the
exported operands the host quantizer's codes and exponents are the
simulation's, and the oracle's product is the simulation's product, every
bit, NaN cells as NaN.

THE VECTORS. `bench/lowbit_quality/int15_export.py` writes them on a box
that has PyTorch; its header says the format. The file is read from
`MOJOLEARN_INT15_SIM_VECTORS`, default
`gemm/checks/vectors/int15_sim_vectors.q15`. The header carries the
git blob hash of the `arith.py` that produced it, and the check prints it,
so a record says which arithmetic it was held to.

A file that is missing, short or of another format is a FAILURE, never a
skip: a cross-check that did not run has checked nothing.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.os import getenv
from std.sys import has_accelerator

from checks.kernel_matrix import TARGET_COLUMN, column_name
from checks.numerics import numeric_mode_name
from gemm.checks.gemm_int15 import (
    identical_gemm_int15_from_f32,
    int15_plan_dispatch_name,
    int15_sabotage_name,
)
from gemm.host.gemm_int15_oracle import (
    INT15_CONVERT_SABOTAGE,
    gemm_int15_oracle,
    quantize_rows_int15,
)
from gemm.host.gemm_oracle import GEMM_ORACLE_HOST_SABOTAGE

comptime SIM_MAGIC = 0x35314951
comptime SIM_VERSION = 1
comptime POISON = Float32(-987654.0)


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _show(x: Float32) -> String:
    return String(x) + "/0x" + hex(_bits(x))


def _same(a: Float32, b: Float32) -> Bool:
    if a != a and b != b:
        return True
    return _bits(a) == _bits(b)


def _digest(v: List[Float32]) -> String:
    """`gemm_int15_check.mojo::_digest`: FNV-1a over the words, every NaN
    as one word."""
    var h = UInt64(0xCBF29CE484222325)
    for i in range(len(v)):
        var w = _bits(v[i])
        if v[i] != v[i]:
            w = UInt32(0x7FC00000)
        for b in range(4):
            var byte = UInt64((w >> UInt32(8 * b)) & UInt32(0xFF))
            h = (h ^ byte) * UInt64(0x100000001B3)
    return hex(h)


def _hex8(w: UInt32) -> String:
    """Eight hexadecimal digits, leading zeros kept: one word of a hash."""
    var digits = String("0123456789abcdef")
    var out = String("")
    for i in range(8):
        var nib = Int((w >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(digits[byte=nib])
    return out


struct SimCase(Copyable, Movable):
    """One exported product: the operands and everything the simulation
    computed from them."""

    var m: Int
    var n: Int
    var k: Int
    var a: List[Float32]
    var b: List[Float32]
    var qa: List[Int16]
    var ea: List[Int32]
    var qb: List[Int16]
    var eb: List[Int32]
    var c: List[Float32]

    def __init__(
        out self,
        m: Int,
        n: Int,
        k: Int,
        var a: List[Float32],
        var b: List[Float32],
        var qa: List[Int16],
        var ea: List[Int32],
        var qb: List[Int16],
        var eb: List[Int32],
        var c: List[Float32],
    ):
        self.m = m
        self.n = n
        self.k = k
        self.a = a^
        self.b = b^
        self.qa = qa^
        self.ea = ea^
        self.qb = qb^
        self.eb = eb^
        self.c = c^


struct WordReader(Movable):
    """The file as little-endian 32-bit words, read in order."""

    var bytes: List[UInt8]
    var at: Int

    def __init__(out self, var bytes: List[UInt8]):
        self.bytes = bytes^
        self.at = 0

    def words_left(self) -> Int:
        return (len(self.bytes) - self.at) // 4

    def word(mut self) raises -> UInt32:
        if self.at + 4 > len(self.bytes):
            raise Error("the vector file ends at byte " + String(len(self.bytes)) + ", inside a case")
        var w = UInt32(0)
        for i in range(4):
            w = w | (UInt32(self.bytes[self.at + i]) << UInt32(8 * i))
        self.at += 4
        return w

    def floats(mut self, count: Int) raises -> List[Float32]:
        var out = List[Float32]()
        for _ in range(count):
            out.append(bitcast[DType.float32](self.word()))
        return out^

    def codes(mut self, count: Int) raises -> List[Int16]:
        var out = List[Int16]()
        for _ in range(count):
            var v = Int(bitcast[DType.int32](self.word()))
            if v < -16383 or v > 16383:
                raise Error("the vector file holds the code " + String(v) + ", outside [-16383, 16383]")
            out.append(Int16(v))
        return out^

    def ints(mut self, count: Int) raises -> List[Int32]:
        var out = List[Int32]()
        for _ in range(count):
            out.append(bitcast[DType.int32](self.word()))
        return out^


def _load(path: String) raises -> List[SimCase]:
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    if len(bytes) < 32 or len(bytes) % 4 != 0:
        raise Error(path + " holds " + String(len(bytes)) + " bytes: not a vector file")
    var r = WordReader(bytes^)
    if r.word() != UInt32(SIM_MAGIC):
        raise Error(path + " does not start with the magic word")
    var version = Int(r.word())
    if version != SIM_VERSION:
        raise Error(path + " is format version " + String(version) + ", this check reads " + String(SIM_VERSION))
    var count = Int(r.word())
    var blob = String("")
    for _ in range(5):
        blob += _hex8(r.word())
    print("   vectors: " + path + ", " + String(count) + " cases, arith.py blob " + blob)
    var cases = List[SimCase]()
    for _ in range(count):
        var m = Int(r.word())
        var n = Int(r.word())
        var k = Int(r.word())
        if m <= 0 or n <= 0 or k <= 0 or m * k > 1 << 26 or n * k > 1 << 26:
            raise Error("a case of the vector file has the shape " + String(m) + " x " + String(n) + " x " + String(k))
        var a = r.floats(m * k)
        var b = r.floats(n * k)
        var qa = r.codes(m * k)
        var ea = r.ints(m)
        var qb = r.codes(n * k)
        var eb = r.ints(n)
        var c = r.floats(m * n)
        cases.append(SimCase(m, n, k, a^, b^, qa^, ea^, qb^, eb^, c^))
    if r.words_left() != 0:
        raise Error(path + " holds " + String(r.words_left()) + " words after its last case")
    return cases^


def _tag(c: SimCase) -> String:
    return String(c.m) + "x" + String(c.n) + "x" + String(c.k)


def _diff_cells(got: List[Float32], want: List[Float32], tag: String) raises:
    var bad = 0
    var first = -1
    for i in range(len(want)):
        if not _same(got[i], want[i]):
            bad += 1
            if first < 0:
                first = i
    if bad > 0:
        raise Error(
            tag + ": " + String(bad) + " of " + String(len(want))
            + " cells differ; first at " + String(first) + " got "
            + _show(got[first]) + ", the simulation has " + _show(want[first])
        )


def check_sim_host_codes(cases: List[SimCase]) raises:
    """GATE: the host quantizer's codes and exponents are the simulation's
    on every exported operand."""
    var codes = 0
    var clamped = 0
    for ci in range(len(cases)):
        ref c = cases[ci]
        var qa = quantize_rows_int15(c.a, c.m, c.k)
        var qb = quantize_rows_int15(c.b, c.n, c.k)
        for r in range(c.m):
            if qa.e[r] != c.ea[r]:
                raise Error(_tag(c) + ": exponent of row " + String(r) + " of A is " + String(qa.e[r]) + ", the simulation has " + String(c.ea[r]))
        for r in range(c.n):
            if qb.e[r] != c.eb[r]:
                raise Error(_tag(c) + ": exponent of row " + String(r) + " of B is " + String(qb.e[r]) + ", the simulation has " + String(c.eb[r]))
        for i in range(c.m * c.k):
            if qa.q[i] != c.qa[i]:
                raise Error(_tag(c) + ": code " + String(i) + " of A (row " + String(i // c.k) + ", value " + _show(c.a[i]) + ") is " + String(Int(qa.q[i])) + ", the simulation has " + String(Int(c.qa[i])))
            if Int(c.qa[i]) == 16383 or Int(c.qa[i]) == -16383:
                clamped += 1
        for i in range(c.n * c.k):
            if qb.q[i] != c.qb[i]:
                raise Error(_tag(c) + ": code " + String(i) + " of B (row " + String(i // c.k) + ", value " + _show(c.b[i]) + ") is " + String(Int(qb.q[i])) + ", the simulation has " + String(Int(c.qb[i])))
        codes += (c.m + c.n) * c.k
    print("   ok " + String(codes) + " codes and their exponents are the simulation's; " + String(clamped) + " codes of A sit at the clamp")


def check_sim_host_product(cases: List[SimCase]) raises:
    """GATE: `gemm_int15_oracle` on the SIMULATION'S codes is the
    simulation's product. The codes are the file's and not this host's, so
    this gate and the one above fail apart."""
    var cells = 0
    var nans = 0
    var infs = 0
    for ci in range(len(cases)):
        ref c = cases[ci]
        var got = gemm_int15_oracle(c.qa, c.ea, c.qb, c.eb, c.m, c.n, c.k)
        print("   DIGEST sim-" + _tag(c) + " simulation " + _digest(c.c))
        print("   DIGEST sim-" + _tag(c) + " oracle " + _digest(got))
        _diff_cells(got, c.c, "host product " + _tag(c))
        for i in range(len(got)):
            if got[i] != got[i]:
                nans += 1
            elif got[i] - got[i] != Float32(0.0):
                infs += 1
        cells += c.m * c.n
    if nans == 0 or infs == 0:
        raise Error("the vectors hold " + String(nans) + " NaN cells and " + String(infs) + " infinite cells; the ends of the scale are not exercised")
    print("   ok " + String(cells) + " cells are the simulation's (" + String(nans) + " NaN, " + String(infs) + " infinite)")


def check_sim_device_product(ctx: DeviceContext, cases: List[SimCase]) raises:
    """GATE: from the exported float32 operands, the device quantizer and
    the dispatched device product are the simulation's product. The path a
    caller takes, held to the arithmetic that was measured."""
    var cells = 0
    for ci in range(len(cases)):
        ref c = cases[ci]
        var da = _upload(ctx, c.a)
        var db = _upload(ctx, c.b)
        var poison = List[Float32]()
        for _ in range(c.m * c.n):
            poison.append(POISON)
        var dc = _upload(ctx, poison)
        identical_gemm_int15_from_f32(ctx, dc, da, db, c.m, c.n, c.k)
        var hb = ctx.enqueue_create_host_buffer[DType.float32](c.m * c.n)
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=dc)
        ctx.synchronize()
        var got = List[Float32]()
        for i in range(c.m * c.n):
            var v = hb.unsafe_ptr().unsafe_load(i)
            if _bits(v) == _bits(POISON):
                raise Error("POISON SURVIVED at cell " + String(i) + " of " + _tag(c))
            got.append(v)
        _ = hb
        _ = da
        _ = db
        _ = dc
        print("   DIGEST sim-" + _tag(c) + " device " + _digest(got))
        _diff_cells(got, c.c, "device product " + _tag(c))
        cells += c.m * c.n
    print("   ok " + String(cells) + " device cells are the simulation's")


def _upload(ctx: DeviceContext, h: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var d = ctx.enqueue_create_buffer[DType.float32](len(h))
    var hb = ctx.enqueue_create_host_buffer[DType.float32](len(h))
    ctx.synchronize()
    for i in range(len(h)):
        hb.unsafe_ptr().unsafe_store(i, h[i])
    ctx.enqueue_copy(dst_buf=d, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    _ = hb
    return d^


def _gate(name: String, mut ran: Int, mut failed: Int, e: String):
    ran += 1
    if e.byte_length() > 0:
        failed += 1
        print("!! GATE FAILED: " + name)
        print("   " + e)
    else:
        print("ok " + name)


def main() raises:
    print(
        "== gemm/checks/gemm_int15_sim_check.mojo [" + numeric_mode_name()
        + "]  sabotage: " + int15_sabotage_name()
        + "  host sabotage: " + String(GEMM_ORACLE_HOST_SABOTAGE)
        + "  convert sabotage: " + String(INT15_CONVERT_SABOTAGE) + " =="
    )
    print("   profile: mojolearn.identical.gemm.int15i64.v1 against bench/lowbit_quality/arith.py, kind int15")
    print("   column: " + column_name(TARGET_COLUMN) + "  int15 dispatch: " + int15_plan_dispatch_name())
    var path = String(getenv("MOJOLEARN_INT15_SIM_VECTORS"))
    if path.byte_length() == 0:
        path = String("gemm/checks/vectors/int15_sim_vectors.q15")
    var ran = 0
    var failed = 0
    var cases = List[SimCase]()
    try:
        cases = _load(path)
        _gate(String("check_sim_vectors_load"), ran, failed, String(""))
    except e:
        _gate(String("check_sim_vectors_load"), ran, failed, String(e))
    if len(cases) > 0:
        try:
            check_sim_host_codes(cases)
            _gate(String("check_sim_host_codes"), ran, failed, String(""))
        except e:
            _gate(String("check_sim_host_codes"), ran, failed, String(e))
        try:
            check_sim_host_product(cases)
            _gate(String("check_sim_host_product"), ran, failed, String(""))
        except e:
            _gate(String("check_sim_host_product"), ran, failed, String(e))
        comptime if not has_accelerator():
            print("   no accelerator: the device gate did not run")
        else:
            var ctx = DeviceContext()
            try:
                check_sim_device_product(ctx, cases)
                _gate(String("check_sim_device_product"), ran, failed, String(""))
            except e:
                _gate(String("check_sim_device_product"), ran, failed, String(e))
            _ = ctx^
    print("== " + String(ran) + " gates, " + String(failed) + " failed ==")
    if failed > 0:
        raise Error(String(failed) + " gate(s) failed")
