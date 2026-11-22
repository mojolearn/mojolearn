# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""A computed NaN is ONE word on every column (lane fx-nan-canon, 2026-11-19).

IEEE 754 fixes when a NaN appears and not its word: NVIDIA writes 0x7fffffff,
Apple and an ARM host 0x7fc00000, AMD and an x86 host 0xffc00000. By
default (NAN_CANON_ON, since 2026-11-20) a float32 result that leaves the device
(and the CPU column's finished result) carries the canonical quiet NaN
0x7fc00000 instead (core/nan_canon.mojo). This check:

- cases 0-2, every build, arm on or off: `nan_canon_f32` on planted words
  (every NaN class, both infinities, signed zeros, subnormals, +-FLT_MAX and
  ordinary values) at a length below one vector, one below the task grain
  and one split across the cores. Every word must equal the scalar rule
  "a NaN becomes 0x7fc00000, nothing else moves", and the returned count
  must be the number of NaN words planted.
- cases 3-5: the fp32 GEMM through the entry a caller reaches
  (`gemm/host_entry.mojo` `identical_gemm_host` on a device column,
  `gemm/host/linalg_native.mojo` `gemm_native` on the CPU column) on
  operands holding +-inf and +-FLT_MAX, against the host oracle. The shapes
  take the raw download, one staged chunk and two staged chunks. ARM ON:
  every output word must equal the oracle's word with its NaNs made
  canonical, exactly (a vendor NaN word fails here). ARM OFF: non-NaN cells
  must match bit for bit and a NaN must face a NaN (today's rule); the
  device's NaN word is printed. Each fixture must hold NaN, infinite and
  finite cells or the case fails as vacuous.
- cases 6-7, device columns: planted words up through a device buffer and
  down through `core/staged_download.mojo` `download_f32_into_canon` (raw
  and staged). ARM ON: the canonical rule. ARM OFF: the words unchanged.

Cases are independent: the CPU column runs them across the host's cores; a
device column submits them in order on one context and the word fills and
compares inside each case run across the cores.

    pixi run check-native-cpu nan_canon
    pixi run check-native-gpu nan_canon
and both again from a build with -D MOJOLEARN_IDENTITY_FAST_NAN_CANON_OFF
(the arm-off clauses; `pixi run check-nan-canon-off-cpu` / `-gpu`).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, require
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from core.nan_canon import NAN_CANON_BITS, NAN_CANON_ON, nan_canon_f32
from core.staged_download import download_f32_into_canon
from gemm.contract import OP_NN
from gemm.host.gemm_oracle import gemm_oracle
from gemm.host.linalg_native import gemm_native as gemm_native_host
from gemm.host_entry import identical_gemm_host
from gemm.checks.gemm_lowbit_check import _upload_f32
from max.gpu.host import DeviceContext
from checks.fixture_rng import splitmix64_of_sum
from std.sys.compile import is_defined


comptime NAN_CANON_UNIT_CASES = 3
comptime NAN_CANON_GEMM_CASES = 3
comptime NAN_CANON_DOWN_CASES = 2
#: A finite word no kernel here produces: an output cell still holding it
#: was never written. Not a NaN, so the canonical pass cannot hide it.
comptime _POISON: UInt32 = 0x4B4B4B4B


@always_inline
def _is_nan_word(w: UInt32) -> Bool:
    return (w & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000)


@always_inline
def _canon_word(w: UInt32) -> UInt32:
    """The rule, written a second time on purpose (the check does not call
    the code it judges): a NaN word becomes the constant, nothing else."""
    if _is_nan_word(w):
        return UInt32(0x7FC00000)
    return w


@always_inline
def _planted_word(i: Int, salt: Int) -> UInt32:
    """Word `i` of the planted buffer: ten special classes, one in sixteen
    each, and ordinary values in [0.5, 1) of either sign for the rest."""
    var h = splitmix64_of_sum(i, salt)
    var r = Int(h & UInt64(15))
    if r == 0:
        return UInt32(0x7FFFFFFF)  # NVIDIA's computed NaN
    if r == 1:
        return UInt32(0xFFC00000)  # AMD's and an x86 host's
    if r == 2:
        return UInt32(0x7FC00000)  # the canonical word itself
    if r == 3:
        return UInt32(0x7F800001)  # a signalling NaN, smallest payload
    if r == 4:
        return UInt32(0xFFFFFFFF)  # every bit set
    if r == 5:
        return UInt32(0x7F800000)  # +inf
    if r == 6:
        return UInt32(0xFF800000)  # -inf
    if r == 7:
        return UInt32(0x80000000)  # -0.0
    if r == 8:
        return UInt32(0x00000001)  # the smallest subnormal
    if r == 9:
        return UInt32(0xFF7FFFFF)  # -FLT_MAX
    return UInt32(0x3F000000) | UInt32((h >> UInt64(16)) & UInt64(0x807FFFFF))


def _plant(n: Int, salt: Int, mut words: List[Float32], mut want: List[Float32]) -> Int:
    """Fill `words` with the planted buffer and `want` with the rule applied
    to it; returns the number of NaN words planted. Units of 1024 words
    across the host's cores (each unit writes only its own words)."""
    var units = (n + 1023) // 1024
    var counts = List[Int](length=max(units, 1), fill=0)
    var cp = counts.unsafe_ptr()
    var wp = words.unsafe_ptr()
    var ep = want.unsafe_ptr()
    def task(unit: Int) {imm cp, imm wp, imm ep, imm n, imm salt}:
        var start = unit * 1024
        var nans = 0
        for i in range(start, min(start + 1024, n)):
            var w = _planted_word(i, salt)
            if _is_nan_word(w):
                nans += 1
            wp.unsafe_store(i, bitcast[DType.float32](w))
            ep.unsafe_store(i, bitcast[DType.float32](_canon_word(w)))
        cp.unsafe_store(unit, nans)
    host_parallelize(task, units)
    var total = 0
    for unit in range(units):  # driver-loop: over the fill units' counts
        total += counts[unit]
    _ = counts^
    return total


def _require_words(actual: List[Float32], expected: List[Float32], label: String) raises:
    """Every word of `actual` equals `expected`, bit for bit. Units of 1024
    words across the host's cores; the first differing word is named."""
    require(len(actual) == len(expected), label + ": element count")
    var count = len(actual)
    var units = (count + 1023) // 1024
    var first = List[Int](length=max(units, 1), fill=-1)
    var fp = first.unsafe_ptr()
    var ap = actual.unsafe_ptr()
    var ep = expected.unsafe_ptr()
    def task(unit: Int) {imm fp, imm ap, imm ep, imm count}:
        var start = unit * 1024
        for i in range(start, min(start + 1024, count)):
            if bitcast[DType.uint32](ap.unsafe_load(i)) != bitcast[DType.uint32](ep.unsafe_load(i)):
                fp.unsafe_store(unit, i)
                return
    host_parallelize(task, units)
    for unit in range(units):  # driver-loop: over the compare units' verdicts, lowest first
        if first[unit] >= 0:
            var at = first[unit]
            raise Error(
                label + ": word " + String(at) + " is " + hex(Int(bitcast[DType.uint32](actual[at])))
                + ", want " + hex(Int(bitcast[DType.uint32](expected[at])))
            )
    _ = first^
    _ = actual
    _ = expected


def _unit_length(which: Int) -> Int:
    if which == 0:
        return 13  # a vector and a scalar tail
    if which == 1:
        return 70001  # one task (below twice the grain floor), ragged tail
    return (1 << 22) + 13  # split across the cores, ragged last span


def unit_case(which: Int) raises:
    """Cases 0-2: `nan_canon_f32` against the rule on planted words."""
    var n = _unit_length(which)
    var words = List[Float32](length=n, fill=Float32(0.0))
    var want = List[Float32](length=n, fill=Float32(0.0))
    var planted = _plant(n, 1000 + which, words, want)
    require(planted > 0 and planted < n, "nan_canon unit: VACUOUS, the planted buffer has no NaN or only NaN")
    var hits = nan_canon_f32(f32_ptr(Int(words.unsafe_ptr())), n)
    _require_words(words, want, "nan_canon_f32 n=" + String(n))
    require(hits == planted, "nan_canon_f32 n=" + String(n) + ": counted " + String(hits) + " NaN words, planted "
            + String(planted))
    # a second pass finds every NaN again (they are still NaN) and moves nothing
    var again = nan_canon_f32(f32_ptr(Int(words.unsafe_ptr())), n)
    _require_words(words, want, "nan_canon_f32 second pass n=" + String(n))
    require(again == planted, "nan_canon_f32 second pass n=" + String(n) + ": count moved")
    _ = words
    _ = want


def _gemm_shape(which: Int) -> Tuple[Int, Int, Int]:
    """(m, n, k). Sizes chosen by the transport's own thresholds
    (gemm/host_transport.mojo: raw below 2^18 cells, 2^22 cells a chunk)."""
    if which == 0:
        return (64, 64, 300)  # 4,096 cells: the raw download; many leaves
    if which == 1:
        return (640, 512, 40)  # 327,680 cells: one staged chunk
    return (2112, 2048, 2)  # 4,325,376 cells: two staged chunks


def _ordinary(i: Int, salt: Int) -> Float32:
    return bitcast[DType.float32](
        UInt32(0x3F000000) | UInt32(splitmix64_of_sum(i, salt * 7919) & UInt64(0x807FFFFF))
    )


def _gemm_operands(m: Int, n: Int, k: Int, salt: Int) -> Tuple[List[Float32], List[Float32]]:
    """A (m x k) and B (k x n), row-major, for OP_NN. Rows of A by r % 4:
    ordinary; +FLT_MAX then -FLT_MAX in its first two columns; +inf first;
    +inf then -inf. Columns of B by c % 3: ordinary; FLT_MAX in its first
    two rows (so +-FLT_MAX rows overflow to +inf and -inf and sum to NaN);
    a zero first (inf * 0). The rest ordinary, in [0.5, 1) of either sign.
    Small loops: m * k and k * n operand words, not the m * n cells."""
    var flt_max = bitcast[DType.float32](UInt32(0x7F7FFFFF))
    var neg_max = bitcast[DType.float32](UInt32(0xFF7FFFFF))
    var pos_inf = bitcast[DType.float32](UInt32(0x7F800000))
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var a = List[Float32](length=m * k, fill=Float32(0.0))
    var b = List[Float32](length=k * n, fill=Float32(0.0))
    for r in range(m):  # driver-loop: over the operand rows of a small fixture
        var kind = r % 4
        for p in range(k):
            var v = _ordinary(r * k + p, salt)
            if kind == 1 and p == 0:
                v = flt_max
            elif kind == 1 and p == 1:
                v = neg_max
            elif kind == 2 and p == 0:
                v = pos_inf
            elif kind == 3 and p == 0:
                v = pos_inf
            elif kind == 3 and p == 1:
                v = neg_inf
            a[r * k + p] = v
    for p in range(k):  # driver-loop: over the operand rows of a small fixture
        for c in range(n):
            var v = _ordinary(p * n + c, salt + 31)
            var ckind = c % 3
            if ckind == 1 and p < 2:
                v = flt_max
            elif ckind == 2 and p == 0:
                v = Float32(0.0)
            b[p * n + c] = v
    return (a^, b^)


def _judge_gemm(got: List[Float32], oracle: List[Float32], label: String) raises:
    """The verdict of one GEMM case (see the module docstring)."""
    var cells = len(oracle)
    require(len(got) == cells, label + ": element count")
    var units = (cells + 1023) // 1024
    # per unit: NaN cells, infinite cells, finite cells of the oracle
    var tally = List[Int](length=3 * max(units, 1), fill=0)
    var canon = List[Float32](length=cells, fill=Float32(0.0))
    var tp = tally.unsafe_ptr()
    var orp = oracle.unsafe_ptr()
    var kp = canon.unsafe_ptr()
    def task(unit: Int) {imm tp, imm orp, imm kp, imm cells}:
        var start = unit * 1024
        var nans = 0
        var infs = 0
        var fins = 0
        for i in range(start, min(start + 1024, cells)):
            var w = bitcast[DType.uint32](orp.unsafe_load(i))
            if _is_nan_word(w):
                nans += 1
            elif (w & UInt32(0x7FFFFFFF)) == UInt32(0x7F800000):
                infs += 1
            else:
                fins += 1
            kp.unsafe_store(i, bitcast[DType.float32](_canon_word(w)))
        tp.unsafe_store(3 * unit, nans)
        tp.unsafe_store(3 * unit + 1, infs)
        tp.unsafe_store(3 * unit + 2, fins)
    host_parallelize(task, units)
    var nans = 0
    var infs = 0
    var fins = 0
    for unit in range(units):  # driver-loop: over the tally units
        nans += tally[3 * unit]
        infs += tally[3 * unit + 1]
        fins += tally[3 * unit + 2]
    require(nans > 0 and infs > 0 and fins > 0,
            label + ": VACUOUS fixture, oracle cells nan=" + String(nans) + " inf=" + String(infs) + " finite="
            + String(fins))
    var first_nan_word = UInt32(0)
    for i in range(cells):  # driver-loop: stops at the first NaN cell (the fixture's second row holds one)
        if _is_nan_word(bitcast[DType.uint32](got[i])):
            first_nan_word = bitcast[DType.uint32](got[i])
            break
    print("NANCANON " + label + " arm=" + ("on" if NAN_CANON_ON else "off") + " cells=" + String(cells)
          + " oracle_nan=" + String(nans) + " oracle_inf=" + String(infs)
          + " first_nan_word=" + hex(Int(first_nan_word)))
    comptime if NAN_CANON_ON:
        _require_words(got, canon, label + " (arm on: exact words, NaN canonical)")
    else:
        # today's rule: a NaN faces a NaN, every other cell bit for bit.
        var got_canon = List[Float32](length=cells, fill=Float32(0.0))
        var gp = got.unsafe_ptr()
        var gk = got_canon.unsafe_ptr()
        def fold(unit: Int) {imm gp, imm gk, imm cells}:
            var start = unit * 1024
            for i in range(start, min(start + 1024, cells)):
                gk.unsafe_store(i, bitcast[DType.float32](_canon_word(bitcast[DType.uint32](gp.unsafe_load(i)))))
        host_parallelize(fold, units)
        _require_words(got_canon, canon, label + " (arm off: NaN faces NaN, other cells exact)")
    _ = tally^
    _ = canon
    _ = got
    _ = oracle


def gemm_case_cpu(which: Int) raises:
    """Cases 3-5 on the CPU column: the host GEMM entry against the oracle."""
    var shape = _gemm_shape(which)
    var m = shape[0]
    var n = shape[1]
    var k = shape[2]
    var operands = _gemm_operands(m, n, k, 50 + which)
    var oracle = gemm_oracle(operands[0], operands[1], OP_NN, m, n, k)
    var got = List[Float32](length=m * n, fill=bitcast[DType.float32](_POISON))
    var params: List[Int] = [m, n, k, OP_NN]
    var wrote = gemm_native_host(
        Int(got.unsafe_ptr()), Int(operands[0].unsafe_ptr()), Int(operands[1].unsafe_ptr()), params,
    )
    require(wrote == m * n, "nan_canon host gemm: written count")
    _judge_gemm(got, oracle, "host-gemm " + String(m) + "x" + String(n) + "x" + String(k))
    _ = operands


def gemm_case_gpu(ctx: DeviceContext, which: Int) raises:
    """Cases 3-5 on a device column: `identical_gemm_host` (upload, the
    dispatcher's GEMM, `gemm_down_f32`) against the host oracle."""
    var shape = _gemm_shape(which)
    var m = shape[0]
    var n = shape[1]
    var k = shape[2]
    var operands = _gemm_operands(m, n, k, 50 + which)
    var oracle = gemm_oracle(operands[0], operands[1], OP_NN, m, n, k)
    var got = List[Float32](length=m * n, fill=bitcast[DType.float32](_POISON))
    identical_gemm_host(
        ctx, f32_ptr(Int(got.unsafe_ptr())), f32_ptr(Int(operands[0].unsafe_ptr())),
        f32_ptr(Int(operands[1].unsafe_ptr())), m, n, k, OP_NN,
    )
    ctx.synchronize()
    _judge_gemm(got, oracle, "device-gemm " + String(m) + "x" + String(n) + "x" + String(k))
    _ = operands


def download_case_gpu(ctx: DeviceContext, which: Int) raises:
    """Cases 6-7: planted words through a device buffer and
    `download_f32_into_canon` (raw below 2^20 floats, staged above)."""
    var n = 70001 if which == 0 else (1 << 22) + 13
    var words = List[Float32](length=n, fill=Float32(0.0))
    var want = List[Float32](length=n, fill=Float32(0.0))
    var planted = _plant(n, 2000 + which, words, want)
    require(planted > 0, "nan_canon download: VACUOUS, no NaN planted")
    var dev = _upload_f32(ctx, words)
    var got = List[Float32](length=n, fill=bitcast[DType.float32](_POISON))
    download_f32_into_canon(ctx, dev, n, f32_ptr(Int(got.unsafe_ptr())))
    comptime if NAN_CANON_ON:
        _require_words(got, want, "download_f32_into_canon n=" + String(n) + " (arm on)")
    else:
        _require_words(got, words, "download_f32_into_canon n=" + String(n) + " (arm off: a plain copy)")
    _ = dev
    _ = words
    _ = want


def _host_case(unit: Int) raises:
    if unit < NAN_CANON_UNIT_CASES:
        unit_case(unit)
    else:
        gemm_case_cpu(unit - NAN_CANON_UNIT_CASES)


def test_host_fixtures() raises:
    comptime assert NAN_CANON_BITS == 0x7FC00000, "nan_canon: the canonical word moved; update _canon_word"
    comptime if CPU_COLUMN_BUILD:
        check_cases("nan_canon host", NAN_CANON_UNIT_CASES + NAN_CANON_GEMM_CASES, _host_case)


def _device_case(ctx: DeviceContext, unit: Int) raises:
    if unit < NAN_CANON_UNIT_CASES:
        unit_case(unit)
    elif unit < NAN_CANON_UNIT_CASES + NAN_CANON_GEMM_CASES:
        gemm_case_gpu(ctx, unit - NAN_CANON_UNIT_CASES)
    else:
        download_case_gpu(ctx, unit - NAN_CANON_UNIT_CASES - NAN_CANON_GEMM_CASES)


def test_device_fixtures() raises:
    comptime assert NAN_CANON_BITS == 0x7FC00000, "nan_canon: the canonical word moved; update _canon_word"
    comptime if not CPU_COLUMN_BUILD:
        var ctx = DeviceContext()
        var failed = String("")
        for unit in range(NAN_CANON_UNIT_CASES + NAN_CANON_GEMM_CASES + NAN_CANON_DOWN_CASES):  # driver-loop: over the cases, in order on the one device context
            try:
                _device_case(ctx, unit)
            except problem:
                failed += "\n      nan_canon device row " + String(unit) + ": " + String(problem)
        if failed.byte_length() > 0:
            raise Error("nan_canon device: cases failed" + failed)


def main() raises:
    # Each column registers its own tests by hand. Discovery (`__functions_in_module()`) reflects over every
    # function of the module, which elaborates the device helpers in the host column's build ("target 'unknown'
    # is not supported"), so a two-column file names its tests.
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
    else:
        suite.test[test_device_fixtures]()
    suite^.run()
