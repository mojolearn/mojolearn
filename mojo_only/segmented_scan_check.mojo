"""Gate the segmented scan against a host tally, per cell, on a hostile shape.

`gbdt/gpu_util/kernel/segmented_scan.mojo` is a device-wide scan whose carry
RESETS at a segment start. Everything that can go wrong with it is a
boundary: a start on the first element, a start exactly on a block boundary,
a start one element past one, two starts in a row (a segment of length 1),
and a carry that crosses several empty blocks. So the fixture plants all of
those on purpose rather than hoping a random pattern contains them.

Three rules from `PORTING_RULES.md:7` shape this file, and each of them
already caught something real in this repository:

  * COMPARE PER CELL, against a tally computed independently on the host.
    A total is not evidence: a scan that ignored every flag would produce
    the correct grand total at the last element and be wrong everywhere
    else.
  * PLANT SCATTERED VALUES. Values here are `hash(i) % 13`, so consecutive
    elements almost never match and an off-by-one inside a segment moves a
    number. Uniform values would hide it. They are also small integers, so
    every partial sum is exact in float32 and the comparison can be `!=`
    rather than a tolerance -- a tolerance is where a real drift hides.
  * EXERCISE BOTH SIDES OF EVERY SWITCH (`PORTING_RULES.md:8`). `inclusive`
    is a switch and so is the flag source, so there are four named checks:
    {vector, scatter} x {inclusive, exclusive}.

The permutation the scatter form writes through is an affine map mod a
PRIME `n`, so it is a bijection and every output slot is accounted for.
"""

from max.gpu.host import DeviceContext

from gbdt.gpu_util.kernel.segmented_scan import (
    SEG_EMIT_BLOCK,
    SEG_SCAN_BLOCK,
    SEGMENT_START_MASK,
    launch_segmented_scan_and_scatter_non_negative,
    launch_segmented_scan_vector,
)

#: Prime, so the affine permutation below is a bijection, and not a multiple
#: of either launch geometry (768 scan blocks, 256 emit blocks) so both
#: kernels run a ragged tail.
comptime N_ROWS = 4001

#: Written into every output slot before the launch. Any slot the device
#: leaves alone shows up as this rather than as a plausible zero.
comptime SENTINEL = Float32(-987654.0)


def hashed(i: Int) -> UInt32:
    """The xorshift `mojo_only/reorder_check.mojo` plants its flags with."""
    var x = UInt32(i * 2654435761 + 0x9E3779B9)
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x


def build_flags(n: Int) -> List[Int]:
    """Segment starts: hashed, then the boundary cases forced on top."""
    var f = List[Int]()
    for i in range(n):
        f.append(1 if (hashed(i) % 23) == 0 else 0)

    # A segment at index 0. Their `UpdateBordersMaskImpl` flags `i == 0`
    # unconditionally (`ctrs/kernel/ctr_calcers.cu:139`), and the exclusive
    # vector form writes slot 0 ONLY through the zeroing kernel, so this is
    # load-bearing rather than decorative.
    f[0] = 1

    # Exactly on a scan-block boundary, and one past it: the carry from the
    # previous block must die at the first and must not reach the second.
    f[SEG_SCAN_BLOCK] = 1
    f[SEG_SCAN_BLOCK + 1] = 1
    f[2 * SEG_SCAN_BLOCK] = 1
    f[2 * SEG_SCAN_BLOCK - 1] = 1
    f[3 * SEG_SCAN_BLOCK] = 1

    # Exactly on an emit-block boundary. The emit kernels run at a
    # DIFFERENT block size (256) from the scan (768), so their boundaries
    # are not the scan's and the shift-by-one has its own off-by-one to
    # get wrong.
    f[SEG_EMIT_BLOCK] = 1
    f[2 * SEG_EMIT_BLOCK] = 1
    f[5 * SEG_EMIT_BLOCK] = 1
    f[5 * SEG_EMIT_BLOCK - 1] = 1

    # Three starts in a row: two segments of length exactly 1.
    f[n // 2] = 1
    f[n // 2 + 1] = 1
    f[n // 2 + 2] = 1

    # The last element as a start, so the final segment is length 1 and the
    # exclusive shift has nothing to write past it.
    f[n - 1] = 1
    return f^


def build_values(n: Int, flags: List[Int]) -> List[Float32]:
    """Scattered small integers, exact in float32."""
    var v = List[Float32]()
    for i in range(n):
        v.append(Float32(Int(hashed(i + 7) % 13)))
    # A ZERO value sitting on a segment start. In the sign-bit encoding that
    # is `-0.0f`, whose sign bit is set and which compares equal to zero.
    # `raw < 0` would call this row an interior element; the bit test their
    # `ExtractSignBit` does calls it a start. This cell is the difference.
    v[SEG_SCAN_BLOCK] = Float32(0.0)
    _ = flags
    return v^


def build_perm(n: Int) -> List[Int]:
    """`i -> (a*i + b) mod n`, a bijection because `n` is prime."""
    var p = List[Int]()
    for i in range(n):
        p.append((i * 1103515245 + 12345) % n)
    return p^


def host_segmented_inclusive(
    values: List[Float32], flags: List[Int]
) -> List[Float32]:
    """The tally. One pass, on the host, from the definition."""
    var out = List[Float32]()
    var running = Float32(0.0)
    for i in range(len(values)):
        if flags[i] == 1:
            running = values[i]
        else:
            running = running + values[i]
        out.append(running)
    return out^


def describe_fixture(flags: List[Int]) raises:
    var n = len(flags)
    var segs = 0
    var ones = 0
    var longest = 0
    var run = 0
    for i in range(n):
        if flags[i] == 1:
            if run == 1:
                ones += 1
            if run > longest:
                longest = run
            segs += 1
            run = 1
        else:
            run += 1
    if run == 1:
        ones += 1
    if run > longest:
        longest = run
    print(
        "    fixture:", n, "rows,", segs, "segments, longest", longest,
        ", length-1 segments", ones,
    )
    if segs < 100:
        raise Error("fixture has too few segments to be adversarial")
    if ones < 3:
        raise Error("fixture has no length-1 segments")


def check_segmented_scan_vector(inclusive: Bool) raises:
    """`SegmentedScanVector` (`cuda_util/segmented_scan.h:8`), both arms.

    Flags come from bit 31 of a separate `ui32` word whose low bits carry a
    row index, which is exactly the shape `ctr_calcers.h:182` passes: the
    permutation IS the flag array. Planting a nonzero payload under the mask
    is deliberate -- a scan that read the whole word instead of masking it
    would pass on a fixture whose low bits were zero.
    """
    var n = N_ROWS
    var ctx = DeviceContext()
    var flags = build_flags(n)
    var values = build_values(n, flags)
    var perm = build_perm(n)
    describe_fixture(flags)

    var h_vals = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_flags = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var h_out = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        h_vals.unsafe_ptr().unsafe_store(i, values[i])
        var word = UInt32(perm[i])
        if flags[i] == 1:
            word |= UInt32(SEGMENT_START_MASK)
        h_flags.unsafe_ptr().unsafe_store(i, word)
        h_out.unsafe_ptr().unsafe_store(i, SENTINEL)

    var d_vals = ctx.enqueue_create_buffer[DType.float32](n)
    var d_flags = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_out = ctx.enqueue_create_buffer[DType.float32](n)
    var d_scanned = ctx.enqueue_create_buffer[DType.float32](n)
    var d_hasflag = ctx.enqueue_create_buffer[DType.uint8](n)
    var n_blocks = (n + SEG_SCAN_BLOCK - 1) // SEG_SCAN_BLOCK
    var d_bsum = ctx.enqueue_create_buffer[DType.float32](n_blocks)
    var d_bflag = ctx.enqueue_create_buffer[DType.uint8](n_blocks)
    ctx.enqueue_copy(dst_buf=d_vals, src_ptr=h_vals.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_flags, src_ptr=h_flags.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_out, src_ptr=h_out.unsafe_ptr())
    ctx.synchronize()

    launch_segmented_scan_vector(
        ctx, n, inclusive, SEGMENT_START_MASK, d_vals, d_flags, d_out,
        d_scanned, d_hasflag, d_bsum, d_bflag,
    )
    ctx.synchronize()

    var got = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=d_out)
    ctx.synchronize()

    var incl = host_segmented_inclusive(values, flags)
    var want = List[Float32]()
    if inclusive:
        for i in range(n):
            want.append(incl[i])
    else:
        # Their exclusive answer is the inclusive one written one slot to
        # the right (`segmented_scan_helpers.cuh:206-209`), then zeroed at
        # every segment start (`ZeroSegmentStartsImpl`,
        # `segmented_scan.cu:11`). Slot 0 is reached only by the zeroing,
        # which is why the fixture flags row 0.
        want.append(Float32(0.0))
        for i in range(1, n):
            if flags[i] == 1:
                want.append(Float32(0.0))
            else:
                want.append(incl[i - 1])

    var wrong = 0
    var first_bad = -1
    for i in range(n):
        if got.unsafe_ptr().unsafe_load(i) != want[i]:
            wrong += 1
            if first_bad < 0:
                first_bad = i
    var arm = "inclusive" if inclusive else "exclusive"
    print("    vector", arm, "-- cells wrong:", wrong, "of", n)
    if wrong != 0:
        print(
            "      first at", first_bad,
            "got", got.unsafe_ptr().unsafe_load(first_bad),
            "want", want[first_bad],
            "flag", flags[first_bad],
        )
        raise Error(
            String("SegmentedScanVector ") + arm + ": " + String(wrong)
            + " of " + String(n) + " cells disagree with the host tally"
        )


def check_segmented_scan_scatter(inclusive: Bool) raises:
    """`SegmentedScanAndScatterNonNegativeVector` (`scan.cu:47`), both arms.

    Flag from the SIGN BIT of the value, answer scattered through the
    permutation. Two independent things can go wrong -- reading the flag and
    choosing the destination -- and a scattered permutation is what
    separates them: with the identity map a destination bug is invisible.
    """
    var n = N_ROWS
    var ctx = DeviceContext()
    var flags = build_flags(n)
    var values = build_values(n, flags)
    var perm = build_perm(n)

    var h_vals = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_idx = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var h_out = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        # `WriteMaskImpl`: `dst[i] = isSegmentStart ? -val : val`
        # (`ctrs/kernel/ctr_calcers.cu:40`).
        var v = values[i]
        if flags[i] == 1:
            v = -v
        h_vals.unsafe_ptr().unsafe_store(i, v)
        var word = UInt32(perm[i])
        if flags[i] == 1:
            word |= UInt32(SEGMENT_START_MASK)
        h_idx.unsafe_ptr().unsafe_store(i, word)
        h_out.unsafe_ptr().unsafe_store(i, SENTINEL)

    var d_vals = ctx.enqueue_create_buffer[DType.float32](n)
    var d_idx = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_out = ctx.enqueue_create_buffer[DType.float32](n)
    var d_scanned = ctx.enqueue_create_buffer[DType.float32](n)
    var d_hasflag = ctx.enqueue_create_buffer[DType.uint8](n)
    var n_blocks = (n + SEG_SCAN_BLOCK - 1) // SEG_SCAN_BLOCK
    var d_bsum = ctx.enqueue_create_buffer[DType.float32](n_blocks)
    var d_bflag = ctx.enqueue_create_buffer[DType.uint8](n_blocks)
    ctx.enqueue_copy(dst_buf=d_vals, src_ptr=h_vals.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_idx, src_ptr=h_idx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_out, src_ptr=h_out.unsafe_ptr())
    ctx.synchronize()

    launch_segmented_scan_and_scatter_non_negative(
        ctx, n, inclusive, d_vals, d_idx, d_out,
        d_scanned, d_hasflag, d_bsum, d_bflag,
    )
    ctx.synchronize()

    var got = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=d_out)
    ctx.synchronize()

    var incl = host_segmented_inclusive(values, flags)
    var want = List[Float32]()
    if inclusive:
        for _ in range(n):
            want.append(SENTINEL)
        # `Ptr[TIndexWrapper(Index[0]).Index()] = abs(val)` (`:66`). No fill
        # on this arm, and the permutation is a bijection, so every slot is
        # written and the sentinel must be gone everywhere.
        for i in range(n):
            want[perm[i]] = incl[i]
    else:
        # `FillBuffer<T>(output, 0, size)` (`scan.cu:59`), then
        # `Ptr[w.Index()] = w.IsSegmentStart() ? 0 : abs(val)` reading
        # `Index[1]` (`:70-72`). The row named by `perm[0]` is never
        # written and keeps the fill.
        for _ in range(n):
            want.append(Float32(0.0))
        for i in range(n - 1):
            if flags[i + 1] == 1:
                want[perm[i + 1]] = Float32(0.0)
            else:
                want[perm[i + 1]] = incl[i]

    var wrong = 0
    var first_bad = -1
    for i in range(n):
        if got.unsafe_ptr().unsafe_load(i) != want[i]:
            wrong += 1
            if first_bad < 0:
                first_bad = i
    var arm = "inclusive" if inclusive else "exclusive"
    print("    scatter", arm, "-- cells wrong:", wrong, "of", n)
    if wrong != 0:
        print(
            "      first at", first_bad,
            "got", got.unsafe_ptr().unsafe_load(first_bad),
            "want", want[first_bad],
        )
        raise Error(
            String("SegmentedScanAndScatterNonNegativeVector ") + arm + ": "
            + String(wrong) + " of " + String(n)
            + " cells disagree with the host tally"
        )


def main() raises:
    print("segmented scan, scan block", SEG_SCAN_BLOCK,
          "emit block", SEG_EMIT_BLOCK)
    check_segmented_scan_vector(False)
    check_segmented_scan_vector(True)
    check_segmented_scan_scatter(False)
    check_segmented_scan_scatter(True)
    print("  segmented scan matches the host tally in every cell, 4 arms")
