# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 2514 step 1: `core/device_scan.mojo` reports the SAME first
index a host loop reports, for every predicate, every plant site and
every length shape the grid-stride kernel has an arm for.

Plants: a quiet NaN, `+inf`, `-inf` and a negative value, each at index
0, at the last index, and at two places at once (the smaller index must
win, whichever block or thread finds the larger one first). Lengths: one
element, one block exactly, one below and one above a block, the grid cap
exactly, one below and one above it (so every thread strides), all of
them checked against `host_first_nonfinite` / `host_first_negative`, which
are the loops the scans replace, spelled BY BITS. The all-clean buffer
returns -1 (the public form of `NONFINITE_NONE`), the negative scan does
not report `-0.0`, and `device_classify_nonfinite` says NaN for the NaN
plant and not-NaN for both infinities. The `DeviceScanScratch` form is
run beside the free functions on every case and must agree. Run:

    pixi run mojo run -I . core/device_scan_check.mojo
"""
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from core.device_scan import (
    DeviceScanScratch,
    NONFINITE_NONE,
    SCAN_BLOCKS,
    SCAN_TPB,
    device_classify_nonfinite,
    device_first_negative,
    device_first_nonfinite,
)

comptime BITS_QNAN: UInt32 = 0x7FC00000
comptime BITS_POS_INF: UInt32 = 0x7F800000
comptime BITS_NEG_INF: UInt32 = 0xFF800000
comptime BITS_NEG_ONE: UInt32 = 0xBF800000
comptime BITS_NEG_ZERO: UInt32 = 0x80000000


def f32_from_bits(b: UInt32) -> Float32:
    return bitcast[DType.float32](b)


def host_first_nonfinite(values: List[Float32]) -> Int:
    """The loop the scan replaces, by bits."""
    for i in range(len(values)):
        var au = bitcast[DType.uint32](values[i]) & UInt32(0x7FFFFFFF)
        if au >= UInt32(0x7F800000):
            return i
    return -1


def host_first_negative(values: List[Float32]) -> Int:
    """`x < 0` for non-NaN floats, by bits: sign set and magnitude nonzero."""
    for i in range(len(values)):
        var bits = bitcast[DType.uint32](values[i])
        if (bits & UInt32(0x80000000)) != UInt32(0) and (
            bits & UInt32(0x7FFFFFFF)
        ) != UInt32(0):
            return i
    return -1


def clean_values(n: Int) -> List[Float32]:
    """Finite, positive, no zeros, no subnormals, and not all equal, so a
    scan that read the wrong element or the wrong predicate has something
    to trip over."""
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32(1.0) + Float32(i % 251) * Float32(0.03125))
    return out^


def upload(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var dev = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


def check_case(
    ctx: DeviceContext,
    mut scratch: DeviceScanScratch,
    label: String,
    values: List[Float32],
    expect_nan: Int,
) raises -> Bool:
    """Both scans, both forms, against both host loops. `expect_nan` is
    -1 (no non-finite planted), 0 (the first non-finite is not a NaN) or
    1 (it is), for the classifier."""
    var n = len(values)
    var want_nf = host_first_nonfinite(values)
    var want_ng = host_first_negative(values)
    var d = upload(ctx, values)
    var got_nf = device_first_nonfinite(ctx, d, n)
    var got_ng = device_first_negative(ctx, d, n)
    var got_nf_s = scratch.first_nonfinite(ctx, d, n)
    var got_ng_s = scratch.first_negative(ctx, d, n)
    var ok = True
    if got_nf != want_nf or got_nf_s != want_nf:
        print(
            "FAIL " + label + " n=" + String(n) + " nonfinite: host "
            + String(want_nf) + " device " + String(got_nf) + " scratch "
            + String(got_nf_s)
        )
        ok = False
    if got_ng != want_ng or got_ng_s != want_ng:
        print(
            "FAIL " + label + " n=" + String(n) + " negative: host "
            + String(want_ng) + " device " + String(got_ng) + " scratch "
            + String(got_ng_s)
        )
        ok = False
    if want_nf >= 0 and expect_nan >= 0:
        var is_nan = device_classify_nonfinite(ctx, d, want_nf)
        if is_nan != (expect_nan == 1):
            print(
                "FAIL " + label + " n=" + String(n) + " classify at "
                + String(want_nf) + ": is_nan " + String(is_nan)
            )
            ok = False
    _ = d^
    return ok


def main() raises:
    var ctx = DeviceContext()
    var scratch = DeviceScanScratch(ctx)
    var ok = True
    var n_cases = 0

    # -1 is the public spelling of "none"; the constant itself must stay
    # above every index an Int32-indexed buffer can hold.
    if Int(NONFINITE_NONE) != 2147483647:
        print("FAIL NONFINITE_NONE is", NONFINITE_NONE)
        ok = False

    var block = SCAN_TPB
    var cap = SCAN_TPB * SCAN_BLOCKS
    var lengths = [
        1, block - 1, block, block + 1, 2 * block + 3,
        cap - 1, cap, cap + 1, 3 * cap + 7,
    ]
    var patterns = [BITS_QNAN, BITS_POS_INF, BITS_NEG_INF, BITS_NEG_ONE]
    var pat_names = [
        String("nan"), String("+inf"), String("-inf"), String("negative"),
    ]
    # classifier expectation per pattern: 1 NaN, 0 infinity, -1 not
    # non-finite (the negative plant is finite)
    var pat_nan = [1, 0, 0, -1]

    for li in range(len(lengths)):
        var n = lengths[li]
        var last = n - 1
        var mid = n // 2

        # all clean: both scans say -1
        ok = check_case(ctx, scratch, "clean", clean_values(n), -1) and ok
        n_cases += 1

        # -0.0 is not negative and not non-finite
        var nz = clean_values(n)
        nz[mid] = f32_from_bits(BITS_NEG_ZERO)
        ok = check_case(ctx, scratch, "neg_zero", nz, -1) and ok
        n_cases += 1

        for pk in range(len(patterns)):
            var val = f32_from_bits(patterns[pk])
            var name = pat_names[pk]

            var at0 = clean_values(n)
            at0[0] = val
            ok = check_case(ctx, scratch, name + "@0", at0, pat_nan[pk]) and ok
            n_cases += 1

            var atl = clean_values(n)
            atl[last] = val
            ok = check_case(
                ctx, scratch, name + "@last", atl, pat_nan[pk]
            ) and ok
            n_cases += 1

            if n >= 2:
                # two plants, the smaller index must win; the larger one
                # is at the last element so it sits in the highest block
                # and the highest thread of it
                var two = clean_values(n)
                two[mid] = val
                two[last] = val
                ok = check_case(
                    ctx, scratch, name + "@mid+last", two, pat_nan[pk]
                ) and ok
                n_cases += 1
                # and the reverse: first at index 0, second in the middle,
                # so a kernel that took the LAST hit per thread is caught
                var two_b = clean_values(n)
                two_b[0] = val
                two_b[mid] = val
                ok = check_case(
                    ctx, scratch, name + "@0+mid", two_b, pat_nan[pk]
                ) and ok
                n_cases += 1
            if n > 2 * block:
                # two plants ONE STRIDE apart in the same thread's walk
                # (index i and i + grid stride), so the per-thread break
                # is what decides, not the tree
                var blocks = (n + SCAN_TPB - 1) // SCAN_TPB
                if blocks > SCAN_BLOCKS:
                    blocks = SCAN_BLOCKS
                var stride = SCAN_TPB * blocks
                if 5 + stride < n:
                    var two_c = clean_values(n)
                    two_c[5] = val
                    two_c[5 + stride] = val
                    ok = check_case(
                        ctx, scratch, name + "@5+stride", two_c, pat_nan[pk]
                    ) and ok
                    n_cases += 1

        # a NaN with the sign bit set is non-finite first; the negative
        # scan alone would also report it, which is why callers run the
        # finite scan first (module docstring)
        var sn = clean_values(n)
        sn[mid] = f32_from_bits(UInt32(0xFFC00000))
        ok = check_case(ctx, scratch, "signed_nan", sn, 1) and ok
        n_cases += 1

        # a negative BEFORE a NaN: the two scans disagree on purpose, each
        # against its own host loop
        if n >= 3:
            var mixed = clean_values(n)
            mixed[1] = f32_from_bits(BITS_NEG_ONE)
            mixed[last] = f32_from_bits(BITS_QNAN)
            ok = check_case(ctx, scratch, "neg_then_nan", mixed, 1) and ok
            n_cases += 1

    # the empty span: -1 from both, no launch
    var empty = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()
    if device_first_nonfinite(ctx, empty, 0) != -1:
        print("FAIL empty nonfinite")
        ok = False
    if device_first_negative(ctx, empty, 0) != -1:
        print("FAIL empty negative")
        ok = False
    if scratch.first_nonfinite(ctx, empty, 0) != -1:
        print("FAIL empty scratch nonfinite")
        ok = False
    _ = empty^
    n_cases += 1

    if not ok:
        raise Error("device_scan_check FAILED")
    print("device_scan_check PASS,", n_cases, "cases")
    _ = scratch^
