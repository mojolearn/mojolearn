"""Gate the radix sort on STABILITY, not on sortedness.

`gbdt/gpu_util/kernel/radix_sort.mojo` exists to put rows in
`(bin, permutation position)` order for the CTR block, and the second half
of that key is never written down: it is the INPUT ORDER, preserved by a
stable sort. So a sort that is perfectly sorted and reorders ties is not a
slow version of the right answer, it is the wrong answer, and it is the
failure this file is built to see.

This is the same trap `RESUME.md` records for the histogram: a check whose
expected value is the same in every cell verifies the total and nothing
about placement. Sortedness is exactly such a check -- it compares each
output slot only with its neighbour, so any permutation within an equal-key
run satisfies it. Both numbers are therefore printed side by side on every
run, and the report is expected to show them disagreeing under sabotage:

    keys out of order:      0        <- sortedness alone, still green
    stable pairing wrong: 3856       <- what the tally sees

The fixture: keys are `hash(i) % 37` or `% 97`, so 40 to 110 rows share
every key and there is a large tie population to permute. Values are an
affine permutation mod a prime, so they are all distinct, scattered, and a
mis-paired value is visible in the cell it lands in rather than only in a
count.
"""

from max.gpu.host import DeviceContext

from gbdt.gpu_util.kernel.radix_sort import launch_radix_sort_bins
from gbdt.gpu_util.kernel.reorder_one_bit import REORDER_BLOCK

#: Prime, so the value permutation is a bijection, and not a multiple of
#: the 512-wide reorder block so the tail is ragged.
comptime N_ROWS = 4001

#: Distinct key values, chosen so a pass count of EITHER parity exists with
#: every pass doing real work: 37 keys fill 6 bits, 97 keys fill 7.
#:
#: THE FIRST VERSION OF THIS FILE GOT THAT WRONG AND THE SABOTAGE FOUND IT.
#: It ran the odd arm as 37 keys over 7 bits, where bit 6 is zero in every
#: key, so the seventh pass was the identity: the sorted answer already sat
#: in the caller's buffer after six passes and DELETING THE COPY-BACK
#: CHANGED NOTHING. The check was green on a path it never reached.
#: `check_radix_sort` now refuses any arm whose top bit is constant.
comptime N_BINS_6 = 37
comptime N_BINS_7 = 97


def hashed(i: Int) -> UInt32:
    var x = UInt32(i * 2654435761 + 0x9E3779B9)
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x


def build_keys(n: Int, n_bins: Int) -> List[Int]:
    var k = List[Int]()
    for i in range(n):
        k.append(Int(hashed(i) % UInt32(n_bins)))
    return k^


def build_values(n: Int) -> List[Int]:
    """`Indices`, in learn-permutation order. Scattered and all distinct."""
    var p = List[Int]()
    for i in range(n):
        p.append((i * 1103515245 + 12345) % n)
    return p^


def sort_key(key: Int, first_bit: Int, last_bit: Int) -> Int:
    """What the sort is allowed to see: bits `[first_bit, last_bit)`.

    `ReorderBinsImpl` passes `false, offset, offset + bits`
    (`cuda_util/sort.cpp:558`) straight into `TRadixSortContext`, so bits
    outside the window do not participate and rows equal inside it are
    ties, whatever their full keys are.
    """
    return (key >> first_bit) & ((1 << (last_bit - first_bit)) - 1)


def check_radix_sort(n_bins: Int, first_bit: Int, last_bit: Int) raises:
    """`ReorderBins(bins, indices, offset, bits, tmp, tmp)`
    (`cuda_util/sort.cpp:544`), against a host STABLE counting sort.

    The host reference is built by walking the key space in order and, for
    each key, appending the input positions in increasing order. That is
    the definition of a stable sort by key, written so it cannot share a
    bug with the device implementation -- it does no bit arithmetic and has
    no notion of passes.
    """
    var n = N_ROWS
    var ctx = DeviceContext()
    var keys = build_keys(n, n_bins)
    var values = build_values(n)
    var passes = last_bit - first_bit

    # REACH, per branch (`PORTING_RULES.md:8`). The ping-pong parity is a
    # switch, and it is only OBSERVABLE when the last pass actually moves
    # something: if the top bit is zero in every key that pass is the
    # identity, the answer is already in the caller's buffer, and removing
    # the copy-back is invisible. Measured, not assumed -- this exact hole
    # let a sabotage of the copy-back run green.
    var top = 0
    for i in range(n):
        if ((keys[i] >> (last_bit - 1)) & 1) == 1:
            top += 1
    if top == 0 or top == n:
        raise Error(
            String("arm [") + String(first_bit) + ", " + String(last_bit)
            + ") cannot reach the final pass: bit " + String(last_bit - 1)
            + " is constant across all " + String(n) + " keys"
        )

    var h_keys = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var h_vals = ctx.enqueue_create_host_buffer[DType.uint32](n)
    for i in range(n):
        h_keys.unsafe_ptr().unsafe_store(i, UInt32(keys[i]))
        h_vals.unsafe_ptr().unsafe_store(i, UInt32(values[i]))

    var d_keys = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_vals = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_tkeys = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_tvals = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_off = ctx.enqueue_create_buffer[DType.int32](n)
    var n_blocks = (n + REORDER_BLOCK - 1) // REORDER_BLOCK
    var d_bsum = ctx.enqueue_create_buffer[DType.int32](n_blocks)
    ctx.enqueue_copy(dst_buf=d_keys, src_ptr=h_keys.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_vals, src_ptr=h_vals.unsafe_ptr())
    ctx.synchronize()

    launch_radix_sort_bins(
        ctx, n, first_bit, last_bit, d_keys, d_vals, d_tkeys, d_tvals,
        d_off, d_bsum,
    )
    ctx.synchronize()

    var got_k = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var got_v = ctx.enqueue_create_host_buffer[DType.uint32](n)
    ctx.enqueue_copy(dst_ptr=got_k.unsafe_ptr(), src_buf=d_keys)
    ctx.enqueue_copy(dst_ptr=got_v.unsafe_ptr(), src_buf=d_vals)
    ctx.synchronize()

    # The host tally: stable order = key ascending, then input position.
    var order = List[Int]()
    for k in range(1 << passes):
        for i in range(n):
            if sort_key(keys[i], first_bit, last_bit) == k:
                order.append(i)

    var keys_wrong = 0
    var pairing_wrong = 0
    var first_bad = -1
    for j in range(n):
        var src = order[j]
        if Int(got_k.unsafe_ptr().unsafe_load(j)) != keys[src]:
            keys_wrong += 1
        if Int(got_v.unsafe_ptr().unsafe_load(j)) != values[src]:
            pairing_wrong += 1
            if first_bad < 0:
                first_bad = j

    # SORTEDNESS ALONE, computed separately and printed beside the tally so
    # the difference between the two is visible rather than argued.
    var unsorted = 0
    for j in range(n - 1):
        var a = sort_key(Int(got_k.unsafe_ptr().unsafe_load(j)),
                         first_bit, last_bit)
        var b = sort_key(Int(got_k.unsafe_ptr().unsafe_load(j + 1)),
                         first_bit, last_bit)
        if a > b:
            unsorted += 1

    # Conservation: every input value lands exactly once. A reorder that
    # dropped or duplicated a row would otherwise have to be caught by the
    # per-cell comparison alone.
    var seen = List[Int]()
    for _ in range(n):
        seen.append(0)
    var out_of_range = 0
    for j in range(n):
        var v = Int(got_v.unsafe_ptr().unsafe_load(j))
        if v < 0 or v >= n:
            out_of_range += 1
        else:
            seen[v] += 1
    var not_once = 0
    for i in range(n):
        if seen[i] != 1:
            not_once += 1

    print(
        "    bits [", first_bit, ",", last_bit, ") --",
        passes, "passes,", "copy-back" if (passes & 1) == 1 else "in place",
    )
    print("      keys out of order:   ", unsorted)
    print("      keys wrong:          ", keys_wrong, "of", n)
    print("      stable pairing wrong:", pairing_wrong, "of", n)
    print("      values not seen once:", not_once, "out of range", out_of_range)

    if unsorted != 0:
        raise Error(
            String("radix sort left ") + String(unsorted)
            + " descending steps in the key column"
        )
    if keys_wrong != 0:
        raise Error(
            String("radix sort key column disagrees with the host tally in ")
            + String(keys_wrong) + " cells"
        )
    if not_once != 0 or out_of_range != 0:
        raise Error("radix sort did not conserve the value column")
    if pairing_wrong != 0:
        print(
            "      first at", first_bad,
            "got", got_v.unsafe_ptr().unsafe_load(first_bad),
            "want", values[order[first_bad]],
            "key", got_k.unsafe_ptr().unsafe_load(first_bad),
        )
        raise Error(
            String("radix sort is NOT STABLE: ") + String(pairing_wrong)
            + " of " + String(n)
            + " ties are in the wrong order (the key column is sorted, so"
            + " a sortedness check would have passed this)"
        )


def main() raises:
    print("radix sort over (bin, permutation position), reorder block",
          REORDER_BLOCK)
    # Six bits covers every key, and the pass count is EVEN, so the answer
    # ends in the caller's own buffers and the copy-back is skipped.
    check_radix_sort(N_BINS_6, 0, 6)
    # Seven bits is the same answer through an ODD pass count, which is the
    # other side of the ping-pong parity switch -- their
    # `if (doubleBufferKeys.Current() != keys)` (`sort_templ.cuh:53`).
    check_radix_sort(N_BINS_7, 0, 7)
    # A window that does not start at bit 0, their `offset` argument. Rows
    # equal in bits [2, 7) are ties even when their full keys differ, so
    # this arm has a much larger tie population than the other two.
    check_radix_sort(N_BINS_7, 2, 7)
    print("  radix sort is stable and matches the host tally, 3 arms")
