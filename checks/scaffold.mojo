# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Check scaffolding: the helpers nearly every check driver re-declares.

Each function here is the MOST COMMON existing version of a helper that
dozens of drivers define privately (`_grid`, `_bits`, `_hex32`, the
`_rows` task-range arithmetic, `_upload`, `_download`, `card_path`,
`_mode_name`, the `run_case` tally, the `_fixture` matrix). The names drop
the leading underscore because these are shared. None of it is numeric
contract: a scaffold helper moves, prints or counts bits and never
computes a value a card hashes. Fixture VALUES come from
`checks/fixture_rng.mojo`.

New check code imports from here (CURRENT DIRECTIVES). Existing drivers
keep their private copies until the consolidation pass migrates them.
Tested by `checks/scaffold_check.mojo` (pixi run check-scaffold).
"""

from std.memory import bitcast, unsafe_memcpy
from std.os import getenv

from max.gpu.host import DeviceBuffer, DeviceContext

from checks.fixture_rng import u16_row_f32
from checks.numerics import numeric_mode_name


# --- launch geometry ------------------------------------------------------


def grid(n: Int, tpb: Int) -> Int:
    """Blocks to cover `n` items at `tpb` threads per block: `ceil(n / tpb)`.
    The common `_grid(n)` is this with the module's own block size."""
    return (n + tpb - 1) // tpb


@fieldwise_init
struct RowRange(Copyable, ImplicitlyCopyable, Movable):
    """Rows `[lo, hi)` of one host task: the arithmetic every `_rows(task)`
    closure opens with (`lo = task * chunk`, `hi = min(lo + chunk, n)`)."""

    var lo: Int
    var hi: Int


def rows(task: Int, chunk: Int, n: Int) -> RowRange:
    var lo = task * chunk
    var hi = lo + chunk
    if hi > n:
        hi = n
    if lo > n:
        lo = n
    return RowRange(lo, hi)


def task_count(n: Int, chunk: Int) -> Int:
    """Tasks for `n` rows at `chunk` rows each (at least one)."""
    if n <= 0:
        return 1
    return (n + chunk - 1) // chunk


# --- bits and printing ----------------------------------------------------


def bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def bits(x: Float64) -> UInt64:
    return bitcast[DType.uint64](x)


def same_bits(a: Float32, b: Float32) -> Bool:
    """Bitwise equality: tells -0.0 from +0.0 and compares NaN payloads."""
    return bits(a) == bits(b)


def hex32(v: Float32) -> String:
    """`0x` and eight lowercase hex digits of the Float32's bits."""
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def hex64(u: UInt64) -> String:
    """`0x` and sixteen lowercase hex digits."""
    comptime DIGITS = "0123456789abcdef"
    var out = String("0x")
    for i in range(16):
        var nib = Int((u >> UInt64(60 - 4 * i)) & UInt64(0xF))
        out += String(DIGITS[byte=nib])
    return out


def mode_name() -> String:
    """The build's numeric tier for a banner (`_mode_name`)."""
    return numeric_mode_name()


# --- host <-> device ------------------------------------------------------


def upload(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    """A device copy of `values`, synchronized before it returns. An empty
    list gets a one-element buffer, since a zero-size allocation is refused
    on some backends."""
    var n = len(values)
    var n_buf = n if n > 0 else 1
    var buf = ctx.enqueue_create_buffer[DType.float32](n_buf)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n_buf)
    ctx.synchronize()
    if n > 0:
        unsafe_memcpy(dest=host.unsafe_ptr(), src=values.unsafe_ptr(), count=n)
    else:
        host.unsafe_ptr().unsafe_store(0, Float32(0.0))
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def download(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    """The first `n` values of `buf`, synchronized."""
    var out = List[Float32]()
    if n <= 0:
        return out^
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    var view = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = view^
    return out^


# --- identity cards and cases ---------------------------------------------


def card_path(default: String) -> String:
    """`MOJOLEARN_IDENTITY_TRACE` when the caller set it, else `default`."""
    var p = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if p.byte_length() > 0:
        return p^
    return default


def fixture_matrix(n: Int, d: Int, salt: Int) -> List[Float32]:
    """The `_fixture(n, d, salt)` shape: an `n x d` row-major matrix of
    hashed cells, cell `(i, f)` = `u16_row_f32(i, f, salt)` (16-bit values
    in `[0, 1)`, exact in Float32, distinct per cell so a permutation
    shows)."""
    var out = List[Float32](capacity=n * d)
    for i in range(n):
        for f in range(d):
            out.append(u16_row_f32(i, f, salt))
    return out^


struct CaseTally(Movable):
    """The `run_case` shape: a production case must return 0 bad cells and
    a sabotage case must MOVE something; `finish` raises on any failure."""

    var failures: Int
    var cases: Int

    def __init__(out self):
        self.failures = 0
        self.cases = 0

    def production(mut self, name: String, bad: Int):
        self.cases += 1
        if bad != 0:
            self.failures += 1
            print("  FAIL", name, "bad", bad)
        else:
            print("  ok  ", name)

    def sabotage(mut self, name: String, moved: Int):
        self.cases += 1
        if moved == 0:
            self.failures += 1
            print("  FAIL", name, "moved nothing")
        else:
            print("  ok  ", name, "->", moved)

    def finish(self, check: String) raises:
        if self.cases == 0:
            raise Error(check + ": VACUOUS, no case ran")
        if self.failures != 0:
            raise Error(check + ": " + String(self.failures) + " failures")
        print(check + ": PASS (" + String(self.cases) + " cases)")
