# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Native category encoding: arbitrary category values to dense codes
(lane cat-ctr, 2026-11-21; docs/plans/FUTURE_WORK_2026-11-21.md section 1,
step 1).

WHAT IT REPLACES. `GradientBoosting(cat_features=...)` took dense codes
`0..k-1` with every code present in the training rows, and refused anything
else by name (`gbdt/models/ctr_value_table.dense_category_code`). The caller
had to build that encoding. It is built here, at the fit's UNPACK, and the
fit and the apply still read dense codes: nothing downstream changed its
contract, the codes are now always dense by construction.

THE KEY. Every category value becomes one 64-bit key, whatever container it
arrived in:

  * a whole number `v` with `-2^61 <= v < 2^61` (a Python int, or a float
    whose value is whole): `v + 2^61`, tag bits 00. The int 5 and the float
    5.0 are the same category;
  * any other real: the bit pattern of its Float32, tag bits 01. Every NaN
    (and None) is ONE key, `CATEGORY_KEY_MISSING`: a missing value is a
    category of its own, as in the reference, where it is the text "nan";
  * text: the CityHash64 of its UTF-8 bytes with the top bit set (tag 1).
    The reference keeps 32 bits of the same hash
    (`CalcCatFeatureHash`); 63 are kept here, so two texts colliding is not
    a practical event.

The three ranges are disjoint, so a column may mix numbers and text.

THE CODE of a value is the rank of its key among the column's distinct
fitted keys in ascending key order. It does not depend on the row order or
on the task count (the distinct set is a set; its ascending order is
unique), so every vendor column and the host column, which all receive the
codes this file wrote, start from the same integers. A value the fit never
saw takes code `k` (one past the largest fitted code): the CTR tables read
an out-of-range code as their unseen-category value, the prior alone
(`gbdt/models/ctr_value_table.mojo`), and a one-hot column compares bins by
equality, so `k` equals no fitted category.

THE MAP is one block of 64-bit words, searched in place at predict (no
parse, no per-call build):

    word 0            CATEGORY_MAP_VERSION
    word 1            C, the number of categorical columns
    word 2 + 2c       the raw feature index of column c (ascending)
    word 3 + 2c       k_c, its number of distinct fitted keys
    then              the keys of column 0 ascending, column 1, ...

PARALLEL PLAN. Rows are independent in the key and code passes: both split
across the host's workers (`core/host_parallel.mojo`, the task count from
the cores and the work). The distinct set is a chunked sort (one chunk per
task, sorted and deduplicated in its own span) followed by pairwise merge
rounds, every pair of runs one task; only the last round (two runs) is one
task wide. The columns of one X run one after another, each using every
worker.

GPU-free: imports no device API, so the doors and the host column import it.
"""

from std.builtin.sort import sort
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from gbdt.digest.city import city_hash_64

comptime CATEGORY_MAP_VERSION = 1
#: a dense code travels as a Float32 column value, exact up to 2^24
comptime CATEGORY_MAX_VALUES = 16777216
comptime CATEGORY_INT_BIAS = 2305843009213693952  # 2^61
comptime CATEGORY_TAG_REAL = UInt64(0x4000000000000000)
comptime CATEGORY_TAG_TEXT = UInt64(0x8000000000000000)
#: every NaN and None: the Float32 quiet-NaN pattern under the real tag
comptime CATEGORY_KEY_MISSING = UInt64(0x400000007FC00000)

comptime CategoryKeyPtr = MutPointer[UInt64, MutUntrackedOrigin]
comptime CategoryF32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime _WordPtr = MutPointer[Int, MutUntrackedOrigin]


@always_inline
def category_integer_in_range(v: Int) -> Bool:
    return v >= -CATEGORY_INT_BIAS and v < CATEGORY_INT_BIAS


@always_inline
def category_key_integer(v: Int) -> UInt64:
    """The key of a whole number in range (`category_integer_in_range`)."""
    return UInt64(v + CATEGORY_INT_BIAS)


@always_inline
def category_key_f32(v: Float32) -> UInt64:
    """The key of one Float32 cell of a categorical column."""
    if v != v:
        return CATEGORY_KEY_MISSING
    # 2.0e18 < 2^61: inside it the Int conversion is exact and in range
    if v > Float32(-2.0e18) and v < Float32(2.0e18):
        var whole = Int(v)
        if Float32(whole) == v:
            return category_key_integer(whole)
    return CATEGORY_TAG_REAL | UInt64(bitcast[DType.uint32](v))


@always_inline
def category_key_f64(v: Float64) -> UInt64:
    """The key of one Python float cell: a whole value is its integer (so it
    agrees with the same value passed as an int), anything else is keyed as
    the Float32 a float matrix would have held."""
    if v != v:
        return CATEGORY_KEY_MISSING
    if v > Float64(-2.0e18) and v < Float64(2.0e18):
        var whole = Int(v)
        if Float64(whole) == v:
            return category_key_integer(whole)
    return category_key_f32(Float32(v))


def category_key_text(address: Int, nbytes: Int) -> UInt64:
    """The key of a text category: `nbytes` UTF-8 bytes at `address`."""
    var view = Span(
        unsafe_ptr=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=address),
        length=nbytes,
    )
    return city_hash_64(view) | CATEGORY_TAG_TEXT


def category_keys_from_f32(src: CategoryF32Ptr, n: Int, keys: CategoryKeyPtr):
    """`keys[r] = category_key_f32(src[r])`, rows split across the workers."""
    if n <= 0:
        return
    var units = max(1, host_predict_task_count_work(n, n * 4))
    var chunk = host_predict_chunk(n, units)

    def fill(unit: Int) {imm chunk, imm n, imm src, imm keys}:
        for r in range(unit * chunk, min((unit + 1) * chunk, n)):
            keys[r] = category_key_f32(src[r])

    host_parallelize(fill, units)


def _unique_in_place(p: CategoryKeyPtr, n: Int) -> Int:
    """Deduplicate an ascending run in place; the number kept."""
    if n <= 0:
        return 0
    var kept = 1
    for i in range(1, n):
        if p[i] != p[kept - 1]:
            p[kept] = p[i]
            kept += 1
    return kept


def _merge_unique(
    a: CategoryKeyPtr, na: Int, b: CategoryKeyPtr, nb: Int, dst: CategoryKeyPtr
) -> Int:
    """Merge two ascending distinct runs into `dst`, a key held by both
    written once; the number written (at most `na + nb`)."""
    var i = 0
    var j = 0
    var w = 0
    while i < na and j < nb:
        var x = a[i]
        var y = b[j]
        if x < y:
            dst[w] = x
            i += 1
        elif y < x:
            dst[w] = y
            j += 1
        else:
            dst[w] = x
            i += 1
            j += 1
        w += 1
    while i < na:
        dst[w] = a[i]
        i += 1
        w += 1
    while j < nb:
        dst[w] = b[j]
        j += 1
        w += 1
    return w


def category_distinct_sorted(keys: CategoryKeyPtr, n: Int) -> List[UInt64]:
    """The distinct values of `keys[0..n)` in ascending order. `keys` is
    read only. See the module docstring for the parallel plan; the result
    is a set in its unique order, so the task count moves nothing."""
    if n <= 0:
        return List[UInt64]()
    var levels = 1
    var remaining = n
    while remaining > 1:
        levels += 1
        remaining = (remaining + 1) // 2
    var units = max(1, host_predict_task_count_work(n, n * levels))
    var chunk = host_predict_chunk(n, units)
    var first = List[UInt64](unsafe_uninit_length=n)
    var second = List[UInt64](unsafe_uninit_length=n)
    var run_lo = List[Int](length=units, fill=0)
    var run_len = List[Int](length=units, fill=0)
    var next_lo = List[Int](length=units, fill=0)
    var next_len = List[Int](length=units, fill=0)
    var ap = CategoryKeyPtr(unsafe_from_address=Int(first.unsafe_ptr()))
    var bp = CategoryKeyPtr(unsafe_from_address=Int(second.unsafe_ptr()))
    var lo_p = _WordPtr(unsafe_from_address=Int(run_lo.unsafe_ptr()))
    var len_p = _WordPtr(unsafe_from_address=Int(run_len.unsafe_ptr()))
    var nlo_p = _WordPtr(unsafe_from_address=Int(next_lo.unsafe_ptr()))
    var nlen_p = _WordPtr(unsafe_from_address=Int(next_len.unsafe_ptr()))

    def initial(unit: Int) {imm chunk, imm n, imm keys, imm ap, imm lo_p, imm len_p}:
        var lo = min(unit * chunk, n)
        var hi = min(lo + chunk, n)
        lo_p[unit] = lo
        if lo >= hi:
            len_p[unit] = 0
            return
        for i in range(lo, hi):
            ap[i] = keys[i]
        var part = Span(unsafe_ptr=ap + lo, length=hi - lo)
        sort(part)
        len_p[unit] = _unique_in_place(ap + lo, hi - lo)

    host_parallelize(initial, units)

    var runs = units
    while runs > 1:
        var pairs = (runs + 1) // 2
        var source = ap
        var target = bp
        var live = runs

        def merge(pair: Int) {imm source, imm target, imm live, imm lo_p, imm len_p, imm nlo_p, imm nlen_p}:
            # runs 2*pair and 2*pair+1 are adjacent spans; the merged run
            # starts where the left one did and cannot outgrow the two
            var left = 2 * pair
            var at = lo_p[left]
            var count = len_p[left]
            if left + 1 < live:
                count = _merge_unique(
                    source + at, len_p[left],
                    source + lo_p[left + 1], len_p[left + 1],
                    target + at,
                )
            else:
                for i in range(count):
                    target[at + i] = source[at + i]
            nlo_p[pair] = at
            nlen_p[pair] = count

        host_parallelize(merge, pairs)
        # driver-loop: over the runs of this round (at most the task count)
        for p in range(pairs):
            lo_p[p] = nlo_p[p]
            len_p[p] = nlen_p[p]
        var swap = ap
        ap = bp
        bp = swap
        runs = pairs

    var count = len_p[0]
    var result = List[UInt64](unsafe_uninit_length=count)
    var rp = CategoryKeyPtr(unsafe_from_address=Int(result.unsafe_ptr()))
    var copy_units = max(1, host_predict_task_count_work(count, count))
    var copy_chunk = host_predict_chunk(count, copy_units)
    var from_p = ap

    def copy_out(unit: Int) {imm copy_chunk, imm count, imm rp, imm from_p}:
        for i in range(unit * copy_chunk, min((unit + 1) * copy_chunk, count)):
            rp[i] = from_p[i]

    host_parallelize(copy_out, copy_units)
    # the scratch runs and their tables are read through raw pointers above
    _ = first^
    _ = second^
    _ = run_lo^
    _ = run_len^
    _ = next_lo^
    _ = next_len^
    return result^


@always_inline
def category_code_of(sorted_keys: CategoryKeyPtr, count: Int, key: UInt64) -> Int:
    """The rank of `key` in an ascending distinct run, or `count` (the
    unseen code) when the run does not hold it."""
    var lo = 0
    var hi = count
    while lo < hi:
        var mid = (lo + hi) // 2
        if sorted_keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    if lo < count and sorted_keys[lo] == key:
        return lo
    return count


def category_codes_into(
    sorted_keys: CategoryKeyPtr, count: Int, keys: CategoryKeyPtr, n: Int, dst: CategoryF32Ptr
):
    """`dst[r] = Float32(code of keys[r])`, rows split across the workers."""
    if n <= 0:
        return
    var levels = 1
    var remaining = count
    while remaining > 1:
        levels += 1
        remaining = (remaining + 1) // 2
    var units = max(1, host_predict_task_count_work(n, n * levels))
    var chunk = host_predict_chunk(n, units)

    def encode(unit: Int) {imm chunk, imm n, imm sorted_keys, imm count, imm keys, imm dst}:
        for r in range(unit * chunk, min((unit + 1) * chunk, n)):
            dst[r] = Float32(category_code_of(sorted_keys, count, keys[r]))

    host_parallelize(encode, units)


struct CategoryMaps(Movable):
    """The maps of one fit while it is being built: one ascending distinct
    key run per categorical column, in ascending feature order."""

    var features: List[Int]
    #: `offsets[c]..offsets[c + 1]` is column c's run in `keys`
    var offsets: List[Int]
    var keys: List[UInt64]

    def __init__(out self):
        self.features = List[Int]()
        self.offsets = List[Int](length=1, fill=0)
        self.keys = List[UInt64]()

    def columns(self) -> Int:
        return len(self.features)

    def count(self, column: Int) -> Int:
        return self.offsets[column + 1] - self.offsets[column]

    def keys_ptr(self, column: Int) -> CategoryKeyPtr:
        return CategoryKeyPtr(
            unsafe_from_address=Int(self.keys.unsafe_ptr()) + 8 * self.offsets[column]
        )

    def fit_column(
        mut self, feature: Int, keys: CategoryKeyPtr, n: Int, dst: CategoryF32Ptr
    ) raises:
        """Fit categorical column `feature` from its `n` row keys and write
        its dense codes to `dst` (which may be the Float32 column the keys
        were read from: the keys are a separate block). Columns are added
        in ascending feature order."""
        if len(self.features) != 0 and feature <= self.features[len(self.features) - 1]:
            raise Error("category map: columns must be added in ascending feature order")
        var distinct = category_distinct_sorted(keys, n)
        var k = len(distinct)
        if k > CATEGORY_MAX_VALUES:
            raise Error(
                "mojolearn: categorical feature " + String(feature) + " holds "
                + String(k) + " distinct values; at most "
                + String(CATEGORY_MAX_VALUES) + " are carried (a dense code"
                + " travels as a float32 column value)"
            )
        var base = len(self.keys)
        self.keys.resize(base + k, UInt64(0))
        var kp = CategoryKeyPtr(unsafe_from_address=Int(self.keys.unsafe_ptr()) + 8 * base)
        var dp = CategoryKeyPtr(unsafe_from_address=Int(distinct.unsafe_ptr()))
        # driver-loop: one copy of the column's distinct keys into the fit's block
        for i in range(k):
            kp[i] = dp[i]
        _ = distinct^
        self.features.append(feature)
        self.offsets.append(base + k)
        category_codes_into(kp, k, keys, n, dst)
        _ = self.keys

    def block_words(self) -> Int:
        """The 64-bit words of the stored map (module docstring layout)."""
        return 2 + 2 * len(self.features) + len(self.keys)

    def write_block(self, dst: CategoryKeyPtr):
        """Write the stored map to `dst[0..block_words())`."""
        var c = len(self.features)
        dst[0] = UInt64(CATEGORY_MAP_VERSION)
        dst[1] = UInt64(c)
        # driver-loop: over the categorical columns
        for i in range(c):
            dst[2 + 2 * i] = UInt64(self.features[i])
            dst[3 + 2 * i] = UInt64(self.count(i))
        var base = 2 + 2 * c
        var kp = CategoryKeyPtr(unsafe_from_address=Int(self.keys.unsafe_ptr()))
        var total = len(self.keys)
        var units = max(1, host_predict_task_count_work(total, total))
        var chunk = host_predict_chunk(total, units)

        def copy_keys(unit: Int) {imm chunk, imm total, imm dst, imm kp, imm base}:
            for i in range(unit * chunk, min((unit + 1) * chunk, total)):
                dst[base + i] = kp[i]

        host_parallelize(copy_keys, units)
        _ = self.keys


def category_block_columns(block: CategoryKeyPtr, words: Int, n_features: Int) raises -> Int:
    """Check a stored map against its word count and the model's feature
    count (a few words per column, no key is read) and return its number of
    categorical columns. A loaded file is refused here, by name, before any
    key is searched."""
    if words < 2 or Int(block[0]) != CATEGORY_MAP_VERSION:
        raise Error("mojolearn: the model's category map is not a version "
                    + String(CATEGORY_MAP_VERSION) + " map; the file is corrupt")
    var c = Int(block[1])
    if c < 0 or 2 + 2 * c > words:
        raise Error("mojolearn: the model's category map is truncated; the file is corrupt")
    var total = 2 + 2 * c
    var previous = -1
    # driver-loop: over the categorical columns
    for i in range(c):
        var feature = Int(block[2 + 2 * i])
        var k = Int(block[3 + 2 * i])
        if feature <= previous or feature >= n_features or k < 0 or k > CATEGORY_MAX_VALUES:
            raise Error("mojolearn: the model's category map names a column or a size the"
                        + " model does not have; the file is corrupt")
        previous = feature
        total += k
    if total != words:
        raise Error("mojolearn: the model's category map holds " + String(words)
                    + " words where its header describes " + String(total)
                    + "; the file is corrupt")
    return c


@always_inline
def category_block_feature(block: CategoryKeyPtr, column: Int) -> Int:
    return Int(block[2 + 2 * column])


@always_inline
def category_block_count(block: CategoryKeyPtr, column: Int) -> Int:
    return Int(block[3 + 2 * column])


def category_block_keys(block: CategoryKeyPtr, column: Int) -> CategoryKeyPtr:
    """Column `column`'s ascending keys inside a checked stored map."""
    var c = Int(block[1])
    var at = 2 + 2 * c
    # driver-loop: over the categorical columns before this one
    for i in range(column):
        at += Int(block[3 + 2 * i])
    return block + at


def category_apply_column(
    block: CategoryKeyPtr, column: Int, keys: CategoryKeyPtr, n: Int, dst: CategoryF32Ptr
):
    """Write the dense codes of `n` row keys of stored column `column` to
    `dst`; a key the fit did not see takes the unseen code `k`."""
    category_codes_into(
        category_block_keys(block, column), category_block_count(block, column), keys, n, dst
    )
