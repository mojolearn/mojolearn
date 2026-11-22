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
from std.collections.string import atof
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


comptime _CodeI8 = MutPointer[Int8, MutUntrackedOrigin]
comptime _CodeI16 = MutPointer[Int16, MutUntrackedOrigin]
comptime _CodeI32 = MutPointer[Int32, MutUntrackedOrigin]
comptime _CodeI64 = MutPointer[Int64, MutUntrackedOrigin]
comptime _FlagPtr = MutPointer[UInt8, MutUntrackedOrigin]


@always_inline
def category_code_at(address: Int, itemsize: Int, stride: Int, r: Int) -> Int:
    """Row `r` of a native-order signed integer codes array (a pandas
    Categorical's codes: int8, int16, int32 or int64; -1 is missing)."""
    var at = address + r * stride
    if itemsize == 1:
        return Int(_CodeI8(unsafe_from_address=at)[0])
    if itemsize == 2:
        return Int(_CodeI16(unsafe_from_address=at)[0])
    if itemsize == 4:
        return Int(_CodeI32(unsafe_from_address=at)[0])
    return Int(_CodeI64(unsafe_from_address=at)[0])


def _codes_refusal(flags: List[UInt8], units: Int) raises:
    var bad = False
    for u in range(units):  # driver-loop: over the tasks' flags
        if flags[u] != 0:
            bad = True
    if bad:
        raise Error("mojolearn: a pandas Categorical column holds a code outside its categories;"
                    + " the column is corrupt")


def category_keys_from_codes(
    codes: Int, itemsize: Int, stride: Int, n: Int, table: CategoryKeyPtr, k: Int, dst: CategoryKeyPtr
) raises:
    """`dst[r] = table[code r]` for the `n` codes of a categorical column
    whose `k` categories are keyed in `table`; code -1 is the missing key.
    Rows split across the workers; a code outside -1 .. k-1 is refused."""
    if itemsize != 1 and itemsize != 2 and itemsize != 4 and itemsize != 8:
        raise Error("mojolearn: a pandas Categorical's codes are not 1, 2, 4 or 8-byte integers")
    if n <= 0:
        return
    var units = max(1, host_predict_task_count_work(n, n * 2))
    var chunk = host_predict_chunk(n, units)
    var flags = List[UInt8](length=units, fill=UInt8(0))
    var fp = _FlagPtr(unsafe_from_address=Int(flags.unsafe_ptr()))

    def gather(unit: Int) {imm chunk, imm n, imm codes, imm itemsize, imm stride, imm table, imm k, imm dst, imm fp}:
        var bad = False
        for r in range(unit * chunk, min((unit + 1) * chunk, n)):
            var code = category_code_at(codes, itemsize, stride, r)
            if code >= 0 and code < k:
                dst[r] = table[code]
            else:
                dst[r] = CATEGORY_KEY_MISSING
                if code != -1:
                    bad = True
        if bad:
            fp[unit] = UInt8(1)

    host_parallelize(gather, units)
    _codes_refusal(flags, units)
    _ = flags^


def category_values_from_codes(
    codes: Int, itemsize: Int, stride: Int, n: Int, table: CategoryF32Ptr, k: Int, dst: CategoryF32Ptr
) raises:
    """`dst[r] = table[code r]` for a categorical column given to a numeric
    feature (its `k` categories are numbers in `table`); code -1 is NaN.
    Rows split across the workers; a code outside -1 .. k-1 is refused."""
    if itemsize != 1 and itemsize != 2 and itemsize != 4 and itemsize != 8:
        raise Error("mojolearn: a pandas Categorical's codes are not 1, 2, 4 or 8-byte integers")
    if n <= 0:
        return
    var units = max(1, host_predict_task_count_work(n, n * 2))
    var chunk = host_predict_chunk(n, units)
    var flags = List[UInt8](length=units, fill=UInt8(0))
    var fp = _FlagPtr(unsafe_from_address=Int(flags.unsafe_ptr()))
    var missing = bitcast[DType.float32](UInt32(0x7FC00000))

    def gather(unit: Int) {imm chunk, imm n, imm codes, imm itemsize, imm stride, imm table, imm k, imm dst, imm fp, imm missing}:
        var bad = False
        for r in range(unit * chunk, min((unit + 1) * chunk, n)):
            var code = category_code_at(codes, itemsize, stride, r)
            if code >= 0 and code < k:
                dst[r] = table[code]
            else:
                dst[r] = missing
                if code != -1:
                    bad = True
        if bad:
            fp[unit] = UInt8(1)

    host_parallelize(gather, units)
    _codes_refusal(flags, units)
    _ = flags^


comptime _ByteP = MutPointer[UInt8, MutUntrackedOrigin]
comptime _Ucs4P = MutPointer[UInt32, MutUntrackedOrigin]

#: why `category_fixed_text` refused a cell (0: it did not)
comptime CATEGORY_TEXT_OK = 0
comptime CATEGORY_TEXT_INVALID = 1
comptime CATEGORY_TEXT_NOT_A_NUMBER = 2


@always_inline
def _is_space(c: UInt8) -> Bool:
    return c == 32 or (c >= 9 and c <= 13)


@always_inline
def _lower(c: UInt8) -> UInt8:
    return c + 32 if c >= 65 and c <= 90 else c


def _word_is(p: _ByteP, lo: Int, hi: Int, word: StaticString) -> Bool:
    """Whether bytes `lo..hi` are `word` (lower case ASCII), any case."""
    var n = word.byte_length()
    if hi - lo != n:
        return False
    var wp = word.unsafe_ptr()
    for i in range(n):
        if _lower(p[lo + i]) != wp[i]:
            return False
    return True


def category_text_number(address: Int, nbytes: Int, mut ok: Bool) -> Float64:
    """The number a numeric column's text cell spells: surrounding ASCII
    whitespace ignored, an optional sign, then a decimal or exponent number
    (the stdlib's correctly rounded `atof`), or `nan`, `inf` or `infinity`
    in any case. `ok` is False (and the result 0) for anything else,
    non-ASCII text included."""
    ok = False
    var p = _ByteP(unsafe_from_address=address)
    var lo = 0
    var hi = nbytes
    while lo < hi and _is_space(p[lo]):
        lo += 1
    while hi > lo and _is_space(p[hi - 1]):
        hi -= 1
    if lo == hi:
        return 0.0
    for i in range(lo, hi):
        if p[i] >= 128:
            return 0.0
    var negative = p[lo] == 45
    var body = lo + 1 if (p[lo] == 45 or p[lo] == 43) else lo
    if _word_is(p, body, hi, "nan"):
        ok = True
        return Float64(bitcast[DType.float32](UInt32(0x7FC00000)))
    if _word_is(p, body, hi, "inf") or _word_is(p, body, hi, "infinity"):
        ok = True
        var big = Float64(bitcast[DType.float32](UInt32(0x7F800000)))
        return -big if negative else big
    var view = Span(
        unsafe_ptr=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=address + lo),
        length=hi - lo,
    )
    try:
        var value = atof(StringSlice(unsafe_from_utf8=view))
        ok = True
        return value
    except:
        return 0.0


def _ucs4_to_utf8(src: Int, chars: Int, dst: _ByteP, mut ok: Bool) -> Int:
    """UTF-8 of `chars` native-order UCS4 code points at `src` into `dst`
    (room for 4 bytes a code point); the byte count. `ok` is False for a
    surrogate or a value past U+10FFFF (text CPython cannot encode)."""
    ok = True
    var p = _Ucs4P(unsafe_from_address=src)
    var w = 0
    for i in range(chars):
        var c = p[i]
        if c < 0x80:
            dst[w] = c.cast[DType.uint8]()
            w += 1
        elif c < 0x800:
            dst[w] = (0xC0 | (c >> 6)).cast[DType.uint8]()
            dst[w + 1] = (0x80 | (c & 0x3F)).cast[DType.uint8]()
            w += 2
        elif c < 0x10000:
            if c >= 0xD800 and c <= 0xDFFF:
                ok = False
                return 0
            dst[w] = (0xE0 | (c >> 12)).cast[DType.uint8]()
            dst[w + 1] = (0x80 | ((c >> 6) & 0x3F)).cast[DType.uint8]()
            dst[w + 2] = (0x80 | (c & 0x3F)).cast[DType.uint8]()
            w += 3
        elif c <= 0x10FFFF:
            dst[w] = (0xF0 | (c >> 18)).cast[DType.uint8]()
            dst[w + 1] = (0x80 | ((c >> 12) & 0x3F)).cast[DType.uint8]()
            dst[w + 2] = (0x80 | ((c >> 6) & 0x3F)).cast[DType.uint8]()
            dst[w + 3] = (0x80 | (c & 0x3F)).cast[DType.uint8]()
            w += 4
        else:
            ok = False
            return 0
    return w


def category_fixed_text(
    base: Int, itemsize: Int, wide: Bool, row_stride: Int, col_stride: Int, n_rows: Int, n_features: Int,
    slots: Int, keys: CategoryKeyPtr, values: CategoryF32Ptr,
) -> Tuple[Int, Int, Int]:
    """Every cell of a fixed-width NumPy text array, natively: a unicode
    (`wide`: UCS4, native order) or bytes array of `itemsize`-byte cells,
    trailing NULs dropped as NumPy drops them. Column j is categorical when
    `slots[j] >= 0` (its cells keyed as text into `keys[slot * n_rows + r]`,
    the key a str or bytes cell of the same UTF-8 bytes gets) and numeric
    otherwise (its text parsed by `category_text_number` into
    `values[j * n_rows + r]`). Rows split across the workers, each with
    its own UTF-8 scratch. Returns (row, column, why) of the first refused
    cell in row order, or (-1, -1, CATEGORY_TEXT_OK)."""
    if n_rows <= 0 or n_features <= 0:
        return (Int(-1), Int(-1), Int(CATEGORY_TEXT_OK))
    var units = max(1, host_predict_task_count_work(n_rows, n_rows * n_features * max(itemsize, 1)))
    var chunk = host_predict_chunk(n_rows, units)
    var room = max(itemsize, 1)
    var scratch = List[UInt8](length=units * room, fill=UInt8(0))
    var fail_row = List[Int](length=units, fill=-1)
    var fail_col = List[Int](length=units, fill=-1)
    var fail_why = List[Int](length=units, fill=CATEGORY_TEXT_OK)
    var scratch_p = _ByteP(unsafe_from_address=Int(scratch.unsafe_ptr()))
    var row_p = _WordPtr(unsafe_from_address=Int(fail_row.unsafe_ptr()))
    var col_p = _WordPtr(unsafe_from_address=Int(fail_col.unsafe_ptr()))
    var why_p = _WordPtr(unsafe_from_address=Int(fail_why.unsafe_ptr()))
    var slot_p = _WordPtr(unsafe_from_address=slots)
    var step = 4 if wide else 1
    var width = itemsize // step

    def cells(unit: Int) {imm chunk, imm n_rows, imm n_features, imm base, imm row_stride, imm col_stride, imm step, imm width, imm wide, imm room, imm scratch_p, imm row_p, imm col_p, imm why_p, imm slot_p, imm keys, imm values}:
        var own = scratch_p + unit * room
        for r in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
            for j in range(n_features):
                var at = base + r * row_stride + j * col_stride
                # the cell's length without NumPy's trailing NUL padding
                var length = width
                if wide:
                    var cp = _Ucs4P(unsafe_from_address=at)
                    while length > 0 and cp[length - 1] == 0:
                        length -= 1
                else:
                    var bp = _ByteP(unsafe_from_address=at)
                    while length > 0 and bp[length - 1] == 0:
                        length -= 1
                var text = at
                var nbytes = length
                if wide:
                    var valid = True
                    nbytes = _ucs4_to_utf8(at, length, own, valid)
                    text = Int(own)
                    if not valid:
                        row_p[unit] = r
                        col_p[unit] = j
                        why_p[unit] = CATEGORY_TEXT_INVALID
                        return
                var slot = slot_p[j]
                if slot >= 0:
                    keys[slot * n_rows + r] = category_key_text(text, nbytes)
                else:
                    var parsed = False
                    var number = category_text_number(text, nbytes, parsed)
                    if not parsed:
                        row_p[unit] = r
                        col_p[unit] = j
                        why_p[unit] = CATEGORY_TEXT_NOT_A_NUMBER
                        return
                    values[j * n_rows + r] = Float32(number)

    host_parallelize(cells, units)
    var result: Tuple[Int, Int, Int] = (Int(-1), Int(-1), Int(CATEGORY_TEXT_OK))
    for u in range(units):  # driver-loop: over the tasks, in row order; the first refusal wins
        if fail_why[u] != CATEGORY_TEXT_OK:
            result = (fail_row[u], fail_col[u], fail_why[u])
            break
    _ = scratch^
    _ = fail_row^
    _ = fail_col^
    _ = fail_why^
    return result


def category_apply_column(
    block: CategoryKeyPtr, column: Int, keys: CategoryKeyPtr, n: Int, dst: CategoryF32Ptr
):
    """Write the dense codes of `n` row keys of stored column `column` to
    `dst`; a key the fit did not see takes the unseen code `k`."""
    category_codes_into(
        category_block_keys(block, column), category_block_count(block, column), keys, n, dst
    )


# The Arrow C data interface columns (lane arrow-strings, 2026-11-21): an
# Arrow-backed pandas string column hands its buffers over at UNPACK
# (`bindings/arrow_c_data_unpack.mojo`); these read them natively. Arrow
# layout (columnar format specification): element i of an array with
# `offset` o sits at slot o + i of every buffer; validity is a bitmap, bit
# (o + i) of byte (o + i) / 8, least significant bit first, 1 = valid, the
# buffer absent (address 0) = every element valid. A string / large_string
# array keeps int32 / int64 offsets (slot i .. i + 1 bound the UTF-8 bytes
# of element i) and one data buffer.

comptime _CodeU8 = MutPointer[UInt8, MutUntrackedOrigin]
comptime _CodeU16 = MutPointer[UInt16, MutUntrackedOrigin]
comptime _CodeU32 = MutPointer[UInt32, MutUntrackedOrigin]
comptime _CodeU64 = MutPointer[UInt64, MutUntrackedOrigin]


@always_inline
def _arrow_valid(validity: Int, i: Int) -> Bool:
    """Bit `i` of an Arrow validity bitmap (address 0: every bit set)."""
    if validity == 0:
        return True
    var byte = _CodeU8(unsafe_from_address=validity + (i >> 3))[0]
    return ((byte >> UInt8(i & 7)) & UInt8(1)) != UInt8(0)


def _arrow_refusal(flags: List[UInt8], units: Int, message: String) raises:
    var bad = False
    for u in range(units):  # driver-loop: over the tasks' flags
        if flags[u] != 0:
            bad = True
    if bad:
        raise Error(message)


def category_keys_from_arrow_text(
    validity: Int, offsets: Int, data: Int, wide: Bool, offset: Int, n: Int, dst: CategoryKeyPtr
) raises:
    """`dst[r]` = the key of element r of an Arrow string (`wide` False:
    int32 offsets) or large_string (int64 offsets) array: the text key of
    its UTF-8 bytes (the key a str cell with the same text gets), a null
    element the missing key. Rows split across the workers; the GIL is not
    needed (plain buffers). Offsets that run backwards are refused."""
    if n <= 0:
        return
    var units = max(1, host_predict_task_count_work(n, n * 16))
    var chunk = host_predict_chunk(n, units)
    var flags = List[UInt8](length=units, fill=UInt8(0))
    var fp = _FlagPtr(unsafe_from_address=Int(flags.unsafe_ptr()))

    def fill(unit: Int) {imm chunk, imm n, imm validity, imm offsets, imm data, imm wide, imm offset, imm dst, imm fp}:
        var bad = False
        for r in range(unit * chunk, min((unit + 1) * chunk, n)):
            var i = offset + r
            if not _arrow_valid(validity, i):
                dst[r] = CATEGORY_KEY_MISSING
                continue
            var lo = 0
            var hi = 0
            if wide:
                lo = Int(_CodeI64(unsafe_from_address=offsets + 8 * i)[0])
                hi = Int(_CodeI64(unsafe_from_address=offsets + 8 * (i + 1))[0])
            else:
                lo = Int(_CodeI32(unsafe_from_address=offsets + 4 * i)[0])
                hi = Int(_CodeI32(unsafe_from_address=offsets + 4 * (i + 1))[0])
            if lo < 0 or hi < lo:
                dst[r] = CATEGORY_KEY_MISSING
                bad = True
                continue
            dst[r] = category_key_text(data + lo, hi - lo)
        if bad:
            fp[unit] = UInt8(1)

    host_parallelize(fill, units)
    _arrow_refusal(flags, units, "mojolearn: an Arrow string array's offsets run backwards; the array is corrupt")
    _ = flags^


@always_inline
def _arrow_index_at(address: Int, itemsize: Int, signed: Bool, i: Int) -> Int:
    """Element `i` of an Arrow dictionary's index buffer (1, 2, 4 or 8-byte
    integers, signed or not); an unsigned 64-bit index past 2^63 - 1 reads
    as -2 (outside every dictionary)."""
    if signed:
        if itemsize == 1:
            return Int(_CodeI8(unsafe_from_address=address + i)[0])
        if itemsize == 2:
            return Int(_CodeI16(unsafe_from_address=address + 2 * i)[0])
        if itemsize == 4:
            return Int(_CodeI32(unsafe_from_address=address + 4 * i)[0])
        return Int(_CodeI64(unsafe_from_address=address + 8 * i)[0])
    if itemsize == 1:
        return Int(_CodeU8(unsafe_from_address=address + i)[0])
    if itemsize == 2:
        return Int(_CodeU16(unsafe_from_address=address + 2 * i)[0])
    if itemsize == 4:
        return Int(_CodeU32(unsafe_from_address=address + 4 * i)[0])
    var wide = _CodeU64(unsafe_from_address=address + 8 * i)[0]
    if wide >= (UInt64(1) << UInt64(63)):
        return -2
    return Int(wide)


def category_keys_from_arrow_indices(
    validity: Int, indices: Int, itemsize: Int, signed: Bool, offset: Int, n: Int,
    table: CategoryKeyPtr, k: Int, dst: CategoryKeyPtr,
) raises:
    """`dst[r] = table[index r]` for a dictionary-encoded Arrow array whose
    `k` dictionary values are keyed in `table` (a null dictionary value is
    already the missing key there); a null index is the missing key. Rows
    split across the workers; an index outside 0 .. k-1 is refused."""
    if itemsize != 1 and itemsize != 2 and itemsize != 4 and itemsize != 8:
        raise Error("mojolearn: an Arrow dictionary's indices are not 1, 2, 4 or 8-byte integers")
    if n <= 0:
        return
    var units = max(1, host_predict_task_count_work(n, n * 2))
    var chunk = host_predict_chunk(n, units)
    var flags = List[UInt8](length=units, fill=UInt8(0))
    var fp = _FlagPtr(unsafe_from_address=Int(flags.unsafe_ptr()))

    def gather(unit: Int) {imm chunk, imm n, imm validity, imm indices, imm itemsize, imm signed, imm offset, imm table, imm k, imm dst, imm fp}:
        var bad = False
        for r in range(unit * chunk, min((unit + 1) * chunk, n)):
            var i = offset + r
            if not _arrow_valid(validity, i):
                dst[r] = CATEGORY_KEY_MISSING
                continue
            var code = _arrow_index_at(indices, itemsize, signed, i)
            if code >= 0 and code < k:
                dst[r] = table[code]
            else:
                dst[r] = CATEGORY_KEY_MISSING
                bad = True
        if bad:
            fp[unit] = UInt8(1)

    host_parallelize(gather, units)
    _arrow_refusal(flags, units, "mojolearn: an Arrow dictionary array holds an index outside its dictionary;"
                   + " the array is corrupt")
    _ = flags^
