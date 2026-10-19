# SPDX-License-Identifier: Apache-2.0
# THE EMBEDDING TAPE OP (lane ag-embedding-op, 2026-10-19; TAPE_EVERYWHERE
# path 4): the element bodies and the Exec drivers of Tape.embedding, a
# row gather whose vector-Jacobian product is the run-sorted scatter-add of
# the embedding profile (embedding/IDENTICAL_EMBEDDING_CONTRACT.md 5.1).
# Shared by HostExec and DeviceExec through sequence/dispatch.mojo, so the
# host column and every GPU run the same bodies.
#
# Words. Forward: Y[t, j] = ftz(ftz(W[ids[t], j])) (emb_gather_kernel's
# store). Backward, per (v, j) cell: acc = +0.0 (or, when the gradient
# already holds a value, that value: the contract's 7.4 carry), then for
# every position t with ids[t] == v in ASCENDING t, acc = ftz(ftz(acc) +
# ftz(dY[t, j])), stored ftz(acc); a row no position names keeps +0.0 (or
# its carried word, untouched); the padding row stores +0.0. These are
# emb_backward_kernel's words (and emb_backward_oracle's at every touched
# cell), so a tape backward gives Embedding.backward's bits on every
# vendor: same inputs, same chain per word, any thread order.
#
# Runs. Both plans produce, for each row v, its positions in ascending t at
# perm[run_begin[v] .. run_begin[v + 1]):
#   scan (OP_EMB_SCAN_BEGIN, OP_EMB_SCAN_PERM): one element per row walks
#     the ids once each (2 V T id reads), as emb_counts_kernel and
#     emb_perm_kernel do; run_begin[v] is the count of non-padding ids
#     below v (the exclusive prefix of the counts, exact in integers).
#   sort (OP_EMB_SORT_PACK, OP_EMB_SORT_STAGE, OP_EMB_SORT_BEGIN): the
#     (id, position) pairs, unique by position, sorted by a bitonic network
#     over the padded power of two (padding and padding_idx positions take
#     the key V and sort last), then one lower-bound search per row.
# The pick moves no bit (the fold reads the same positions in the same
# order); AG_EMB_PLAN_AUTO takes sort from V * T >= EMB_AUTO_SORT_MIN_CELLS
# (the cost rule of embedding/checks/embedding_identical.mojo, the same
# define and default: scan reads 2 V T ids, sort ~T log2(T)^2 / 4
# exchanges; a work-size rule, not a shape).
#
# Ids are int32 row numbers held as their bit patterns in 4-byte executor
# words (Tape.ids), read with bitcast and never through ld (ftz would zero
# a small id's subnormal pattern). The sort moves them in the integer
# domain. Refusal: Tape.embedding checks every id against the table's rows
# on the executor before the gather (OP_EMB_IDS_CHECK, a blocked first-bad
# search, levels of EMB_CHECK_BLOCK, then one two-word read-back), as
# emb_refuse_device_ids does by reading the ids back to the host.
from std.memory import bitcast
from std.sys.defines import get_defined_int
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from checks.numerics import ftz
from embedding.checks.embedding_oracle import EMB_MAX_POSITIONS

# Codes OP_EMB_FIRST..OP_EMB_LAST are this family (the 2700 range of the
# tape-everywhere plan); sequence/dispatch.mojo routes and fences them.
comptime OP_EMB_GATHER = 2700
comptime OP_EMB_IDS_CHECK = 2701
comptime OP_EMB_IDS_VALUE = 2702
comptime OP_EMB_SCAN_BEGIN = 2703
comptime OP_EMB_SCAN_PERM = 2704
comptime OP_EMB_SORT_PACK = 2705
comptime OP_EMB_SORT_STAGE = 2706
comptime OP_EMB_SORT_BEGIN = 2707
comptime OP_EMB_FOLD = 2708
comptime OP_EMB_FIRST = OP_EMB_GATHER
comptime OP_EMB_LAST = OP_EMB_FOLD

# Tape node kind of an embedding: below CUSTOM (40, autograd/ops.mojo) and
# apart from the kinds there (0..31) and DROPOUT (35).
comptime EMBEDDING = 36

# Run plans; the codes are embedding/checks/embedding_sort.mojo's PLAN_SCAN
# and PLAN_SORT and embedding_identical.mojo's PLAN_AUTO.
comptime AG_EMB_PLAN_SCAN = 0
comptime AG_EMB_PLAN_SORT = 1
comptime AG_EMB_PLAN_AUTO = 2
comptime EMB_AUTO_SORT_MIN_CELLS = get_defined_int["MOJOLEARN_EMB_AUTO_SORT_MIN_CELLS", 1 << 26]()

# Ids a first-bad search element walks: a fixed block, so the levels are a
# function of the id count alone (256: one level covers 256 ids, two cover
# 65,536, three 16.7 M, each element a short contiguous walk).
comptime EMB_CHECK_BLOCK = 256

# Every launch count and Args integer stays inside the device executor's
# Int32 range (sequence/exec_device.mojo _args_i32_out).
comptime _I32_MAX = 2147483647


# ---------------------------------------------------------------- words
@always_inline
def _ld_i(p: FP, i: Int) -> Int:
    """The int32 held as bits in word i of p."""
    return Int(bitcast[DType.int32](p.unsafe_load(i)))


@always_inline
def _st_i(p: FP, i: Int, v: Int):
    """Word i of p holds int32 v as its bit pattern."""
    p.unsafe_store(i, bitcast[DType.float32](Int32(v)))


# ---------------------------------------------------------------- bodies
@always_inline
def op_emb_gather(t: Int, a: Args):
    """Y[pos, j] = ftz(ftz(W[ids[pos], j])). p0 W (V, d), p1 ids (T
    words), p2 Y (T, d); i0 d. Thread t = output cell t."""
    var width = a.i0
    var pos = t // width
    var j = t - pos * width
    var v = _ld_i(a.p1, pos)
    a.p2.unsafe_store(t, ftz(ftz(a.p0.unsafe_load(v * width + j))))


@always_inline
def op_emb_ids_check(t: Int, a: Args):
    """One level of the first-bad search. p0 the source words, p1 the
    destination; i0 the source word count, i1 V, i2 the level. Element t
    walks source words [t B, min((t + 1) B, i0)) in order (B =
    EMB_CHECK_BLOCK) and stores the first hit, or -1: at level 0 the first
    position whose id is outside [0, V); at a later level the first
    non-negative word (blocks are in position order, so it is the least
    bad position)."""
    var lo = t * EMB_CHECK_BLOCK
    var hi = min(lo + EMB_CHECK_BLOCK, a.i0)
    var first = -1
    for i in range(lo, hi):
        var w = _ld_i(a.p0, i)
        if a.i2 == 0:
            if w < 0 or w >= a.i1:
                first = i
                break
        elif w >= 0:
            first = w
            break
    _st_i(a.p1, t, first)


@always_inline
def op_emb_ids_value(t: Int, a: Args):
    """p0 ids, p1 the search result: p1[1] = ids[p1[0]] when p1[0] >= 0,
    else 0. One element: the refused id's value for the message."""
    var at = _ld_i(a.p1, 0)
    var v = 0
    if at >= 0:
        v = _ld_i(a.p0, at)
    _st_i(a.p1, 1, v)


@always_inline
def op_emb_scan_begin(t: Int, a: Args):
    """run_begin[v] = the number of positions whose id is below v and is
    not padding_idx, for v = t in [0, V]. p0 ids, p1 run_begin; i0 T, i1
    padding_idx (-1 none). Exact integer counts: the exclusive prefix of
    the per-row counts, in any thread order."""
    var c = 0
    for i in range(a.i0):
        var w = _ld_i(a.p0, i)
        if w < t and w != a.i1:
            c += 1
    _st_i(a.p1, t, c)


@always_inline
def op_emb_scan_perm(t: Int, a: Args):
    """Row v = t's positions in ascending order at perm[run_begin[v] ..].
    p0 ids, p1 run_begin, p2 perm; i0 T, i1 padding_idx. Each row owns its
    region and appends in position order (emb_perm_kernel's walk)."""
    if t == a.i1:
        return
    var w = _ld_i(a.p1, t)
    var hi = _ld_i(a.p1, t + 1)
    if w >= hi:
        return
    for i in range(a.i0):
        if _ld_i(a.p0, i) == t:
            _st_i(a.p2, w, i)
            w += 1


@always_inline
def op_emb_sort_pack(t: Int, a: Args):
    """Sort input, element t of the padded length P: key = ids[t] (V for a
    padding_idx position or t >= T), value = t. p0 ids, p1 keys, p2
    values; i0 T, i1 padding_idx, i2 V."""
    var key = a.i2
    if t < a.i0:
        var w = _ld_i(a.p0, t)
        if w != a.i1:
            key = w
    _st_i(a.p1, t, key)
    _st_i(a.p2, t, t)


@always_inline
def op_emb_sort_stage(t: Int, a: Args):
    """One compare-exchange stage of the bitonic network over (key, value)
    pairs, ordered by key then value (unique: the values are positions).
    p0 keys, p1 values; i0 the merge size k, i1 the partner distance j.
    Element t owns the pair (i, i + j), i = 2 j (t // j) + t % j,
    ascending when i & k == 0; the pairs of a stage are disjoint."""
    var j = a.i1
    var i = 2 * j * (t // j) + t % j
    var l = i + j
    var ka = _ld_i(a.p0, i)
    var va = _ld_i(a.p1, i)
    var kb = _ld_i(a.p0, l)
    var vb = _ld_i(a.p1, l)
    var a_after = ka > kb or (ka == kb and va > vb)
    var swap = a_after if (i & a.i0) == 0 else not a_after
    if swap:
        _st_i(a.p0, i, kb)
        _st_i(a.p1, i, vb)
        _st_i(a.p0, l, ka)
        _st_i(a.p1, l, va)


@always_inline
def op_emb_sort_begin(t: Int, a: Args):
    """run_begin[v] = the first sorted slot whose key is >= v, for v = t in
    [0, V]. p0 the sorted keys, p1 run_begin; i0 P."""
    var lo = 0
    var hi = a.i0
    while lo < hi:
        var mid = (lo + hi) // 2
        if _ld_i(a.p0, mid) < t:
            lo = mid + 1
        else:
            hi = mid
    _st_i(a.p1, t, lo)


@always_inline
def op_emb_fold(t: Int, a: Args):
    """THE CONTRACT'S FOLD (emb_backward_kernel's words, the seed and the
    padding row in the same body). p0 dY (T, d), p1 perm, p2 run_begin
    (V + 1), p3 dW (V, d); i0 d, i1 padding_idx (-1 none), i2 1 when dW
    holds a carried gradient. Thread t = cell (v, j)."""
    var width = a.i0
    var v = t // width
    var j = t - v * width
    if v == a.i1:
        a.p3.unsafe_store(t, Float32(0.0))
        return
    var lo = _ld_i(a.p2, v)
    var hi = _ld_i(a.p2, v + 1)
    if lo >= hi:
        if a.i2 == 0:
            a.p3.unsafe_store(t, Float32(0.0))
        return
    var acc = Float32(0.0)
    if a.i2 != 0:
        acc = a.p3.unsafe_load(t)
    for r in range(lo, hi):
        var pos = _ld_i(a.p1, r)
        acc = ftz(ftz(acc) + ftz(a.p0.unsafe_load(pos * width + j)))
    a.p3.unsafe_store(t, ftz(acc))


@always_inline
def apply_embedding[OP: Int](t: Int, a: Args):
    comptime if OP == OP_EMB_GATHER:
        op_emb_gather(t, a)
    elif OP == OP_EMB_IDS_CHECK:
        op_emb_ids_check(t, a)
    elif OP == OP_EMB_IDS_VALUE:
        op_emb_ids_value(t, a)
    elif OP == OP_EMB_SCAN_BEGIN:
        op_emb_scan_begin(t, a)
    elif OP == OP_EMB_SCAN_PERM:
        op_emb_scan_perm(t, a)
    elif OP == OP_EMB_SORT_PACK:
        op_emb_sort_pack(t, a)
    elif OP == OP_EMB_SORT_STAGE:
        op_emb_sort_stage(t, a)
    elif OP == OP_EMB_SORT_BEGIN:
        op_emb_sort_begin(t, a)
    elif OP == OP_EMB_FOLD:
        op_emb_fold(t, a)


# ---------------------------------------------------------------- drivers
def embedding_refuse_shape(n_positions: Int, vocab: Int, width: Int, padding_idx: Int) raises:
    """The embedding op's shape refusals (Tape.embedding and every direct
    caller of the drivers below)."""
    if n_positions < 1 or n_positions > EMB_MAX_POSITIONS:
        raise Error("embedding: the id count must be in [1, " + String(EMB_MAX_POSITIONS) + "]")
    if vocab < 1 or width < 1:
        raise Error("embedding: the table must have at least one row and one column")
    if vocab > _I32_MAX // width or n_positions > _I32_MAX // width:
        raise Error("embedding: the table and the output must each hold fewer than 2^31 values")
    if padding_idx < -1 or padding_idx >= vocab:
        raise Error("embedding: padding_idx must be -1 (none) or a row of the table")


def embedding_check_words(n_positions: Int) -> Int:
    """Words of each of the two buffers embedding_refuse_ids takes."""
    return max(2, (n_positions + EMB_CHECK_BLOCK - 1) // EMB_CHECK_BLOCK)


def embedding_refuse_ids[E: Exec](
    mut ex: E, ids: FP, n_positions: Int, vocab: Int, scratch_a: FP, scratch_b: FP,
) raises:
    """Refuse an id outside [0, vocab), by position and value, before any
    gather reads it. Each scratch buffer holds embedding_check_words(n)
    words of `ex`. One two-word read-back (a wait)."""
    var a = Args()
    a.p0 = ids
    a.p1 = scratch_a
    a.i0 = n_positions
    a.i1 = vocab
    a.i2 = 0
    var count = (n_positions + EMB_CHECK_BLOCK - 1) // EMB_CHECK_BLOCK
    ex.launch[OP_EMB_IDS_CHECK](a, count)
    var result = scratch_a
    var other = scratch_b
    var level = 1
    while count > 1:  # driver-loop: over first-bad search levels
        var s = Args()
        s.p0 = result
        s.p1 = other
        s.i0 = count
        s.i1 = vocab
        s.i2 = level
        count = (count + EMB_CHECK_BLOCK - 1) // EMB_CHECK_BLOCK
        ex.launch[OP_EMB_IDS_CHECK](s, count)
        var was = result
        result = other
        other = was
        level += 1
    var v = Args()
    v.p0 = ids
    v.p1 = result
    ex.launch[OP_EMB_IDS_VALUE](v, 1)
    var out = List[Float32](length=2, fill=Float32(0))
    ex.download(FP(unsafe_from_address=Int(out.unsafe_ptr())), result, 2)
    var at = Int(bitcast[DType.int32](out[0]))
    if at >= 0:
        raise Error(
            "embedding: id " + String(Int(bitcast[DType.int32](out[1]))) + " at position " + String(at)
            + " is outside [0, " + String(vocab) + ")"
        )


def embedding_forward_into[E: Exec](mut ex: E, table: FP, ids: FP, output: FP, n_positions: Int, width: Int) raises:
    """output (n_positions, width) = table[ids] (ids refused by the caller)."""
    var a = Args()
    a.p0 = table
    a.p1 = ids
    a.p2 = output
    a.i0 = width
    ex.launch[OP_EMB_GATHER](a, n_positions * width)


def embedding_plan(plan: Int, vocab: Int, n_positions: Int) raises -> Int:
    """AG_EMB_PLAN_AUTO resolved by work (sort from V * T >=
    EMB_AUTO_SORT_MIN_CELLS); an explicit scan or sort unchanged."""
    if plan == AG_EMB_PLAN_SCAN or plan == AG_EMB_PLAN_SORT:
        return plan
    if plan != AG_EMB_PLAN_AUTO:
        raise Error("embedding: plan must be 0 (scan), 1 (sort) or 2 (auto)")
    if vocab * n_positions >= EMB_AUTO_SORT_MIN_CELLS:
        return AG_EMB_PLAN_SORT
    return AG_EMB_PLAN_SCAN


def embedding_sort_length(n_positions: Int) -> Int:
    """The sort's padded length: the least power of two >= n (at least 2)."""
    var p = 2
    while p < n_positions:  # driver-loop: over powers of two (at most 30)
        p *= 2
    return p


def embedding_perm_words(plan: Int, n_positions: Int) -> Int:
    """Words of the perm buffer (and, for the sort, of the keys buffer)."""
    if plan == AG_EMB_PLAN_SORT:
        return embedding_sort_length(n_positions)
    return n_positions


def embedding_backward_into[E: Exec](
    mut ex: E, ids: FP, dy: FP, dw: FP, n_positions: Int, vocab: Int, width: Int,
    padding_idx: Int, carry: Bool, plan: Int, run_begin: FP, perm: FP, keys: FP,
) raises:
    """dw (vocab, width) = the run-sorted scatter-add of dy (n_positions,
    width) over ids, the profile's words: continued from dw's own values
    when `carry`, else from +0.0; the padding row +0.0. `plan` is resolved
    (embedding_plan). Scratch: run_begin vocab + 1 words, perm
    embedding_perm_words(plan, n) words, keys the same for the sort (unused
    by the scan). The ids are refused by the caller."""
    var a = Args()
    a.p0 = ids
    a.p1 = run_begin
    a.i0 = n_positions
    a.i1 = padding_idx
    if plan == AG_EMB_PLAN_SORT:
        var padded = embedding_sort_length(n_positions)
        var pk = Args()
        pk.p0 = ids
        pk.p1 = keys
        pk.p2 = perm
        pk.i0 = n_positions
        pk.i1 = padding_idx
        pk.i2 = vocab
        ex.launch[OP_EMB_SORT_PACK](pk, padded)
        var k = 2
        while k <= padded:  # driver-loop: over sort network merge sizes
            var j = k // 2
            while j > 0:  # driver-loop: over sort network stages
                var s = Args()
                s.p0 = keys
                s.p1 = perm
                s.i0 = k
                s.i1 = j
                ex.launch[OP_EMB_SORT_STAGE](s, padded // 2)
                j //= 2
            k *= 2
        var b = Args()
        b.p0 = keys
        b.p1 = run_begin
        b.i0 = padded
        ex.launch[OP_EMB_SORT_BEGIN](b, vocab + 1)
    else:
        ex.launch[OP_EMB_SCAN_BEGIN](a, vocab + 1)
        var pm = Args()
        pm.p0 = ids
        pm.p1 = run_begin
        pm.p2 = perm
        pm.i0 = n_positions
        pm.i1 = padding_idx
        ex.launch[OP_EMB_SCAN_PERM](pm, vocab)
    var f = Args()
    f.p0 = dy
    f.p1 = perm
    f.p2 = run_begin
    f.p3 = dw
    f.i0 = width
    f.i1 = padding_idx
    f.i2 = 1 if carry else 0
    ex.launch[OP_EMB_FOLD](f, vocab * width)
