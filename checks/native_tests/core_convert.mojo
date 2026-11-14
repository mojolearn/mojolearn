# SPDX-License-Identifier: Apache-2.0
"""Float64 -> float32 narrowing and float32 transposes, moved from
test_native_convert.py (the native functions of that file).

Every comparison is on raw float32 words, never a tolerance (the narrowing is
one round-to-nearest-even per element). The oracle is `rne_word` below: an
integer rounding of the float64 significand written from the IEEE 754 rules,
independent of the product loops and of the compiler's own cast. NaN payloads
are not promised: a NaN answer accepts any NaN. A transpose is a pure move, so
it is compared word for word including NaN payloads and signed zeros.

Fixtures (CONVERT_FIXTURES):
  0..12   flat cast over each of the 13 shapes (`cast_f64_to_f32`)
  13..25  fused cast and column-major transpose over each shape
          (`cast_colmajor_f64_to_f32`)
  26..38  float32 transpose over each shape, both directions (`transpose_f32`)
  39      every edge value alone, through the flat and the fused helper
  40      the ties: round to nearest EVEN, never half away from zero
  41      every tail length 1..39 with a tie in the head and one in the tail

The matrices are seeded float64 data (a SplitMix64 stream regenerated here; the
former NumPy PCG64 stream is not needed because the oracle is computed in this
file) scaled by 1e-30, 1e-3, 1, 1e3 or 1e30, with the edge values and ties
planted at seeded positions so the rounding cases land in every tile position
across the shapes, as before. The binding's refusals (negative counts, null
addresses) and the through-`_buffer` arms stay in the Python test (boundary).
"""
from std.math import cos, log, sqrt
from std.memory import bitcast
from core.core_host_impl import transpose_to_f32
from core.host_helpers_impl import cast_f64_to_f32
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, f64_ptr
from core.mojolearn_shell_impl import cast_f64_to_f32_host, tiled_transpose_to_f32
from core.hotpath_device_run import hpdev_try_cast_f64_to_f32, hpdev_try_transpose_to_f32
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime CONVERT_SHAPES = 13
comptime CONVERT_FIXTURES = 3 * CONVERT_SHAPES + 3
comptime CONVERT_EDGE_FIXTURE = 3 * CONVERT_SHAPES
comptime CONVERT_TIES_FIXTURE = 3 * CONVERT_SHAPES + 1
comptime CONVERT_TAIL_FIXTURE = 3 * CONVERT_SHAPES + 2


trait ConvertCalls:
    """The three product operations, on raw addresses."""
    @staticmethod
    def cast(src: Int, dst: Int, n: Int) raises: ...
    @staticmethod
    def cast_colmajor(src: Int, dst: Int, rows: Int, cols: Int) raises: ...
    @staticmethod
    def transpose(src: Int, dst: Int, rows: Int, cols: Int) raises: ...


# ------------------------------------------------------------------ the oracle

def rne_word(v: Float64, ties_even: Bool = True) -> UInt32:
    """The float32 word of `v` rounded to nearest, ties to even (or, with
    `ties_even=False`, ties away from zero), from the float64 bits."""
    var b = bitcast[DType.uint64](v)
    var sign = ((b >> 63) & 1).cast[DType.uint32]() << 31
    var e = Int((b >> 52) & 0x7FF)
    var m = b & 0xFFFFFFFFFFFFF
    if e == 0x7FF:
        if m != 0:
            return sign | 0x7FC00000
        return sign | 0x7F800000
    if e == 0:
        # A float64 subnormal is below 2^-1022, far under half of 2^-149.
        return sign
    var sig = (UInt64(1) << 52) | m
    var ex = e - 1023
    var normal = ex >= -126
    # Bits of the 53-bit significand below the float32 quantum.
    var shift = 29 if normal else 29 + (-126 - ex)
    if shift > 60:
        return sign
    var k = sig >> UInt64(shift)
    var rem = sig & ((UInt64(1) << UInt64(shift)) - 1)
    var half = UInt64(1) << UInt64(shift - 1)
    if rem > half or (rem == half and (not ties_even or (k & 1) == 1)):
        k += 1
    if not normal:
        # k <= 2^23; k == 2^23 is the smallest normal word, as written.
        return sign | k.cast[DType.uint32]()
    if k == (UInt64(1) << 24):
        k = UInt64(1) << 23
        ex += 1
    var biased = ex + 127
    if biased >= 255:
        return sign | 0x7F800000
    return sign | (UInt32(biased) << 23) | (k - (UInt64(1) << 23)).cast[DType.uint32]()


def _exact_half(v: Float64) -> Bool:
    var b = bitcast[DType.uint64](v)
    var e = Int((b >> 52) & 0x7FF)
    if e == 0 or e == 0x7FF:
        return False
    var sig = (UInt64(1) << 52) | (b & 0xFFFFFFFFFFFFF)
    var ex = e - 1023
    var shift = 29 if ex >= -126 else 29 + (-126 - ex)
    if shift > 60:
        return False
    return (sig & ((UInt64(1) << UInt64(shift)) - 1)) == (UInt64(1) << UInt64(shift - 1))


def _is_nan_word(w: UInt32) -> Bool:
    return (w & 0x7F800000) == 0x7F800000 and (w & 0x007FFFFF) != 0


def require_word(got: Float32, v: Float64, label: String) raises:
    var g = bitcast[DType.uint32](got)
    var want = rne_word(v)
    if _is_nan_word(want):
        require(_is_nan_word(g), label + ": NaN in, got word " + String(g))
        return
    require(g == want, label + ": value " + String(v) + " got word " + String(g) + ", RNE word " + String(want))


# ------------------------------------------------------------------ the inputs

def pow2(k: Int) -> Float64:
    """2^k exactly, for k in [-1022, 1023]."""
    return bitcast[DType.float64](UInt64(1023 + k) << 52)


def _f32_max() -> Float64:
    return Float64(bitcast[DType.float32](UInt32(0x7F7FFFFF)))


def tie_values() -> List[Float64]:
    """float64 values exactly halfway between two adjacent float32s, chosen so
    round-to-nearest-EVEN and round-half-away-from-zero DISAGREE (and one
    pair where RNE rounds UP), in two binades and the subnormal range."""
    var ulp = pow2(-23)
    var half = ulp / 2
    var s = pow2(-149)
    var q = pow2(-15)
    var values: List[Float64] = [
        1.0 + half, 1.0 + 3 * half, -(1.0 + half), -(1.0 + 3 * half),
        1000.0 + q, 1000.0 + 3 * q,
        s * 2.5, s * 3.5, -(s * 2.5),
    ]
    return values^


def edge_values() -> List[Float64]:
    var fmax = _f32_max()
    var tiny = Float64(bitcast[DType.float32](UInt32(0x00800000)))
    var sub = Float64(Float32(1e-40))
    var values: List[Float64] = [
        0.0, -0.0, 1.0, -1.0, 0.1, 1.0 / 3.0, 3.141592653589793, -2.718281828459045,
        fmax, -fmax,
        fmax * (1 + pow2(-25)),
        fmax * (1 + pow2(-24)),
        fmax * 2, -fmax * 2, 1e39, -1e39, 1e308,
        tiny, -tiny, sub, -sub,
        bitcast[DType.float64](UInt64(1)), bitcast[DType.float64](UInt64(0x8000000000000001)),
        1e-46, pow2(-150),
        bitcast[DType.float64](UInt64(0x7FF0000000000000)), bitcast[DType.float64](UInt64(0xFFF0000000000000)),
        bitcast[DType.float64](UInt64(0x7FF8000000000000)), bitcast[DType.float64](UInt64(0xFFF8000000000000)),
        bitcast[DType.float64](UInt64(0x7FF8DEADBEEF0001)),
    ]
    var ties = tie_values()
    for t in ties:
        values.append(t)
    return values^


def convert_shape(index: Int) -> Tuple[Int, Int]:
    var rows: List[Int] = [1, 1, 7, 3, 5, 2, 64, 128, 129, 130, 1000, 257, 3]
    var cols: List[Int] = [1, 7, 1, 5, 3, 64, 2, 64, 65, 20, 10, 3, 257]
    return (rows[index], cols[index])


struct SplitMix(Movable):
    var state: UInt64

    def __init__(out self, seed: Int):
        self.state = UInt64(seed)

    def next(mut self) -> UInt64:
        self.state += 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def uniform(mut self) -> Float64:
        # (0, 1): 53 random bits, never zero.
        return ((self.next() >> 11).cast[DType.float64]() + 0.5) * pow2(-53)

    def normal(mut self) -> Float64:
        var u = self.uniform()
        var w = self.uniform()
        return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * w)

    def below(mut self, n: Int) -> Int:
        return Int(self.next() % UInt64(n))


def seeded_matrix(rows: Int, cols: Int, seed: Int) -> List[Float64]:
    """C-order float64 [rows, cols]: scaled normals with the edge values and
    ties planted at the first positions of a seeded permutation."""
    var rng = SplitMix(seed)
    var n = rows * cols
    var scales: List[Float64] = [1e-30, 1e-3, 1.0, 1e3, 1e30]
    var flat = List[Float64](length=n, fill=0)
    for i in range(n):
        flat[i] = rng.normal() * scales[rng.below(5)]
    var edges = edge_values()
    var order = List[Int](length=n, fill=0)
    for i in range(n):
        order[i] = i
    for i in range(n - 1, 0, -1):
        var j = rng.below(i + 1)
        var t = order[i]
        order[i] = order[j]
        order[j] = t
    for i in range(min(n, len(edges))):
        flat[order[i]] = edges[i]
    return flat^


def _addr64(values: List[Float64]) -> Int:
    return Int(values.unsafe_ptr())


def _addr32(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


# ------------------------------------------------------------------ the fixtures

def _flat[C: ConvertCalls](index: Int) raises:
    var shape = convert_shape(index)
    var x = seeded_matrix(shape[0], shape[1], shape[0] * 1000 + shape[1])
    var n = len(x)
    var dst = List[Float32](length=n, fill=0)
    C.cast(_addr64(x), _addr32(dst), n)
    for i in range(n):
        require_word(dst[i], x[i], "flat cast " + String(shape[0]) + "x" + String(shape[1]) + " at " + String(i))
    _ = x^


def _colmajor[C: ConvertCalls](index: Int) raises:
    var shape = convert_shape(index)
    var rows = shape[0]
    var cols = shape[1]
    var x = seeded_matrix(rows, cols, rows * 7 + cols * 13)
    var dst = List[Float32](length=rows * cols, fill=0)
    C.cast_colmajor(_addr64(x), _addr32(dst), rows, cols)
    for r in range(rows):
        for c in range(cols):
            require_word(dst[c * rows + r], x[r * cols + c],
                         "colmajor cast " + String(rows) + "x" + String(cols) + " at " + String(r) + "," + String(c))
    _ = x^


def _transpose[C: ConvertCalls](index: Int) raises:
    var shape = convert_shape(index)
    var rows = shape[0]
    var cols = shape[1]
    var wide = seeded_matrix(rows, cols, rows * 3 + cols * 31)
    var n = rows * cols
    var x = List[Float32](length=n, fill=0)
    for i in range(n):
        x[i] = bitcast[DType.float32](rne_word(wide[i]))
    # NaN payloads and both signed zeros: a pure move keeps every bit.
    if n >= 3:
        x[0] = bitcast[DType.float32](UInt32(0x7FC0BEEF))
        x[1] = bitcast[DType.float32](UInt32(0x80000000))
        x[2] = bitcast[DType.float32](UInt32(0xFFA00001))
    var dst = List[Float32](length=n, fill=0)
    C.transpose(_addr32(x), _addr32(dst), rows, cols)
    for r in range(rows):
        for c in range(cols):
            require(bitcast[DType.uint32](dst[c * rows + r]) == bitcast[DType.uint32](x[r * cols + c]),
                    "transpose " + String(rows) + "x" + String(cols) + " at " + String(r) + "," + String(c))
    # The other direction: the column-major block as a C [cols, rows] source.
    var back = List[Float32](length=n, fill=0)
    C.transpose(_addr32(dst), _addr32(back), cols, rows)
    for i in range(n):
        require(bitcast[DType.uint32](back[i]) == bitcast[DType.uint32](x[i]),
                "transpose back " + String(rows) + "x" + String(cols) + " at " + String(i))


def _edges_alone[C: ConvertCalls]() raises:
    """Each edge value as a one-element buffer, through both helpers, so a
    failure names the value rather than a position."""
    var edges = edge_values()
    for v in edges:
        var src: List[Float64] = [v]
        var d1: List[Float32] = [0.0]
        var d2: List[Float32] = [0.0]
        C.cast(_addr64(src), _addr32(d1), 1)
        C.cast_colmajor(_addr64(src), _addr32(d2), 1, 1)
        require_word(d1[0], v, "edge value alone, flat")
        require_word(d2[0], v, "edge value alone, colmajor")


def _ties[C: ConvertCalls]() raises:
    """The tie values come out as RNE says, and NOT as round-half-away says;
    the two answers differ by one ulp on at least four of them."""
    var disagreements = 0
    var ties = tie_values()
    for v in ties:
        require(_exact_half(v), "not a tie: " + String(v))
        var src: List[Float64] = [v]
        var dst: List[Float32] = [0.0]
        C.cast(_addr64(src), _addr32(dst), 1)
        var got = bitcast[DType.uint32](dst[0])
        var even = rne_word(v, True)
        var away = rne_word(v, False)
        require(got == even, "tie " + String(v) + ": got " + String(got) + ", RNE " + String(even))
        if away != even:
            disagreements += 1
            require(got != away, "tie " + String(v) + ": helper rounded half away from zero")
    require(disagreements >= 4, "tie set no longer separates the two rules")


def _tails[C: ConvertCalls]() raises:
    """Every remainder modulo the SIMD width, so the scalar tail is exercised
    at each length it can have."""
    var rng = SplitMix(2470)
    for n in range(1, 40):
        var x = List[Float64](length=n, fill=0)
        for i in range(n):
            x[i] = rng.normal()
        x[0] = 1.0 + pow2(-24)
        x[n - 1] = 1000.0 + 3 * pow2(-15)
        var dst = List[Float32](length=n, fill=0)
        C.cast(_addr64(x), _addr32(dst), n)
        for i in range(n):
            require_word(dst[i], x[i], "tail length " + String(n) + " at " + String(i))


def run_convert_fixture[C: ConvertCalls](fixture_id: Int) raises:
    require(fixture_id >= 0 and fixture_id < CONVERT_FIXTURES, "unknown convert fixture")
    if fixture_id < CONVERT_SHAPES:
        _flat[C](fixture_id)
    elif fixture_id < 2 * CONVERT_SHAPES:
        _colmajor[C](fixture_id - CONVERT_SHAPES)
    elif fixture_id < 3 * CONVERT_SHAPES:
        _transpose[C](fixture_id - 2 * CONVERT_SHAPES)
    elif fixture_id == CONVERT_EDGE_FIXTURE:
        _edges_alone[C]()
    elif fixture_id == CONVERT_TIES_FIXTURE:
        _ties[C]()
    else:
        _tails[C]()


# ---- The host column (was core_convert_cpu_suite.mojo) ----
# The host narrowing and transposes, every fixture through both host routes,
# run across host cores.
#
# Route 0 is the host bindings' column (core/host_helpers_impl.mojo
# `cast_f64_to_f32`, core/core_host_impl.mojo `transpose_to_f32`); route 1 is
# the base binding's host loop (core/mojolearn_shell_impl.mojo
# `cast_f64_to_f32_host`, `tiled_transpose_to_f32`).
struct HostHelpersConvert(ConvertCalls):
    @staticmethod
    def cast(src: Int, dst: Int, n: Int) raises:
        cast_f64_to_f32(f64_ptr(src), f32_ptr(dst), n)

    @staticmethod
    def cast_colmajor(src: Int, dst: Int, rows: Int, cols: Int) raises:
        transpose_to_f32[DType.float64](f64_ptr(src), f32_ptr(dst), rows, cols)

    @staticmethod
    def transpose(src: Int, dst: Int, rows: Int, cols: Int) raises:
        transpose_to_f32[DType.float32](f32_ptr(src), f32_ptr(dst), rows, cols)


struct HostShellConvert(ConvertCalls):
    @staticmethod
    def cast(src: Int, dst: Int, n: Int) raises:
        cast_f64_to_f32_host(f64_ptr(src), f32_ptr(dst), n)

    @staticmethod
    def cast_colmajor(src: Int, dst: Int, rows: Int, cols: Int) raises:
        tiled_transpose_to_f32[DType.float64](f64_ptr(src), f32_ptr(dst), rows, cols)

    @staticmethod
    def transpose(src: Int, dst: Int, rows: Int, cols: Int) raises:
        tiled_transpose_to_f32[DType.float32](f32_ptr(src), f32_ptr(dst), rows, cols)


def _run_cpu(mut report: Report) raises:
    comptime UNITS = 2 * CONVERT_FIXTURES
    var cases = report.cases(UNITS)
    def task(unit: Int) {imm cases}:
        var fixture = unit % CONVERT_FIXTURES
        try:
            if unit < CONVERT_FIXTURES:
                run_convert_fixture[HostHelpersConvert](fixture)
            else:
                run_convert_fixture[HostShellConvert](fixture)
        except error:
            cases.fail(unit, error)
    report.parallel(task, UNITS)


# ---- The device column (was core_convert_gpu_suite.mojo) ----
# The device narrowing and transposes the GPU base binding takes
# (core/hotpath_device_run.mojo `hpdev_try_cast_f64_to_f32`,
# `hpdev_try_transpose_to_f32`), every fixture in order.
#
# Both entries use the one process-wide device context of the base binding,
# so fixtures are submitted in order; each operation is one thread per element.
# A False return (a sabotage build, or a size outside the device twin) fails the
# fixture: the default GPU build takes the device route for every size here.
struct DeviceConvert(ConvertCalls):
    @staticmethod
    def cast(src: Int, dst: Int, n: Int) raises:
        require(hpdev_try_cast_f64_to_f32(src, dst, n), "device cast_f64_to_f32 not taken for n=" + String(n))

    @staticmethod
    def cast_colmajor(src: Int, dst: Int, rows: Int, cols: Int) raises:
        require(hpdev_try_transpose_to_f32(src, dst, rows, cols, True), "device cast_colmajor not taken")

    @staticmethod
    def transpose(src: Int, dst: Int, rows: Int, cols: Int) raises:
        require(hpdev_try_transpose_to_f32(src, dst, rows, cols, False), "device transpose_f32 not taken")


def _run_gpu(mut report: Report) raises:
    var cases = report.cases(CONVERT_FIXTURES)
    for fixture in range(CONVERT_FIXTURES):
        try:
            run_convert_fixture[DeviceConvert](fixture)
        except error:
            cases.fail(fixture, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("core_convert", cpu=True, gpu=True, serial="", run=run_cases)
