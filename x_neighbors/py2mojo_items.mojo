# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Items for the neighbors lane's Python data loops moved into Mojo (lane
apple-fast-py2mojo-neighbors, 2026-10-03), in the `items.mojo` contract: one
item is one device thread's work and the host driver runs the same items, so
the CPU column and every GPU column store the same bits. Every item here is a
copy, a compare or integer work: no float arithmetic, so no bit can move.

  p2m_mask_value     KNNImputer: cells equal to a numeric missing_values -> NaN
  p2m_zero_cols      KNNImputer keep_empty_features: flagged columns -> +0.0
  p2m_nan_indicator  KNNImputer add_indicator: [out | isnan(src[:, cols])]
  p2m_sign_label     LOF fit_predict / predict: the +-1 labels
  p2m_relabel        connected_components / Louvain: labels renumbered by
                     first occurrence (an atomic min per label, a blocked
                     flag count, scan and emit)
  p2m_class_counts   NearestCentroid's class counts (atomic adds per row)
  p2m_const_cols     NearestCentroid's all-features-constant test
  p2m_fill, p2m_iota PageRank's uniform start vectors, the cc start labels
  p2m_negate         kneighbors' inner-product distances (exact negation)
  p2m_transpose(_i)  the k-NN multi-output label / target transposes
  p2m_row_sort       RadiusNeighbors sort_results (and LOF's percentile): a
                     segmented bitonic sort by (row, value, position), so the
                     order is Python's stable sort by value within each row

Nothing here imports a GPU module, so the CPU-only host binding compiles it.
"""
from std.atomic import Atomic
from std.memory import bitcast
from x_neighbors.items import XN_FOLD_BLOCK

comptime FP = MutPointer[Float32, MutAnyOrigin]
comptime IP = MutPointer[Int32, MutAnyOrigin]


# ------------------------------------------------------------------ KNNImputer
def p2m_mask_value_item(t: Int, x: FP, res: FP, count: Int, want: Float32):
    """res = NaN where x == want (a float compare, as Python's), else x."""
    var v = x.unsafe_load(t)
    if v == want:
        res.unsafe_store(t, bitcast[DType.float32](UInt32(0x7FC00000)))
    else:
        res.unsafe_store(t, v)


def p2m_zero_cols_item(t: Int, x: FP, flags: IP, res: FP, n: Int, d: Int):
    """res[i, f] = +0.0 where flags[f] != 0, else x[i, f]; t = i*d + f."""
    if flags.unsafe_load(t % d) != 0:
        res.unsafe_store(t, Float32(0))
    else:
        res.unsafe_store(t, x.unsafe_load(t))


def p2m_nan_indicator_item(t: Int, src: FP, cur: FP, cols: IP, res: FP, n: Int, d: Int, c: Int, q: Int):
    """res row i = cur row i (c values) then 1.0 / 0.0 for src[i, cols[j]]
    being NaN (j < q); t = i*(c + q) + k."""
    var w = c + q
    var i = t // w
    var k = t - i * w
    if k < c:
        res.unsafe_store(t, cur.unsafe_load(i * c + k))
    else:
        var v = src.unsafe_load(i * d + Int(cols.unsafe_load(k - c)))
        res.unsafe_store(t, Float32(1) if v != v else Float32(0))


# ------------------------------------------------------------------ LOF labels
def p2m_sign_label_item(t: Int, x: FP, res: IP, count: Int, mode: Int, thr: Float32):
    """mode 0: -1 if x < thr else 1 (fit_predict); mode 1: 1 if x >= thr
    else -1 (predict); mode 2: 1 if x > thr else -1 (OneClassSVM predict,
    lane pyglue-numeric). Spelled as the Python they replace, so NaN agrees."""
    var v = x.unsafe_load(t)
    if mode == 0:
        res.unsafe_store(t, Int32(-1) if v < thr else Int32(1))
    elif mode == 1:
        res.unsafe_store(t, Int32(1) if v >= thr else Int32(-1))
    else:
        res.unsafe_store(t, Int32(1) if v > thr else Int32(-1))


# ------------------------------------------------------------------ relabel
def p2m_relabel_init_item(t: Int, lab: IP, res: IP, info: IP, first: IP, rk: IP, part: IP, n: Int):
    """Stage 1: first[t] = n (no occurrence yet); item 0 clears info."""
    first.unsafe_store(t, Int32(n))
    if t == 0:
        info.unsafe_store(0, Int32(0))
        info.unsafe_store(1, Int32(0))


def p2m_relabel_first_item(t: Int, lab: IP, res: IP, info: IP, first: IP, rk: IP, part: IP, n: Int):
    """Stage 2: first[v] = the lowest t with lab[t] == v (an atomic min: the
    result is the same in any order). A label outside [0, n) sets info[1]."""
    var v = Int(lab.unsafe_load(t))
    if v < 0 or v >= n:
        info.unsafe_store(1, Int32(1))
        return
    _ = Atomic[DType.int32].min(first + v, Int32(t))


@always_inline
def _is_first(i: Int, lab: IP, first: IP, n: Int) -> Bool:
    var v = Int(lab.unsafe_load(i))
    if v < 0 or v >= n:
        return False
    return Int(first.unsafe_load(v)) == i


def p2m_relabel_count_item(t: Int, lab: IP, res: IP, info: IP, first: IP, rk: IP, part: IP, n: Int):
    """Stage 3, one item per XN_FOLD_BLOCK block: its count of first
    occurrences."""
    var lo = t * XN_FOLD_BLOCK
    var hi = min(lo + XN_FOLD_BLOCK, n)
    var c = Int32(0)
    for i in range(lo, hi):
        if _is_first(i, lab, first, n):
            c += 1
    part.unsafe_store(t, c)


def p2m_relabel_scan_item(t: Int, lab: IP, res: IP, info: IP, first: IP, rk: IP, part: IP, n: Int):
    """Stage 4, one item: the exclusive scan of the block counts; the number
    of distinct labels to info[0]."""
    var nb = (n + XN_FOLD_BLOCK - 1) // XN_FOLD_BLOCK
    var run = Int32(0)
    for b in range(nb):
        var c = part.unsafe_load(b)
        part.unsafe_store(b, run)
        run += c
    info.unsafe_store(0, run)


def p2m_relabel_emit_item(t: Int, lab: IP, res: IP, info: IP, first: IP, rk: IP, part: IP, n: Int):
    """Stage 5, one item per block: rk[i] = the rank of first occurrence i."""
    var lo = t * XN_FOLD_BLOCK
    var hi = min(lo + XN_FOLD_BLOCK, n)
    var at = part.unsafe_load(t)
    for i in range(lo, hi):
        if _is_first(i, lab, first, n):
            rk.unsafe_store(i, at)
            at += 1


def p2m_relabel_map_item(t: Int, lab: IP, res: IP, info: IP, first: IP, rk: IP, part: IP, n: Int):
    """Stage 6: res[t] = the rank of lab[t]'s first occurrence (-1 for a
    label outside [0, n); the caller refuses that call)."""
    var v = Int(lab.unsafe_load(t))
    if v < 0 or v >= n:
        res.unsafe_store(t, Int32(-1))
        return
    res.unsafe_store(t, rk.unsafe_load(Int(first.unsafe_load(v))))


# ------------------------------------------------------------------ fills, copies
# ------------------------------------------------------------------ NearestCentroid
def p2m_ccount_zero_item(t: Int, lab: IP, nk: FP, info: IP, cnt: IP, n: Int, n_classes: Int):
    """Stage 1: cnt[c] = 0; item 0 clears info."""
    cnt.unsafe_store(t, Int32(0))
    if t == 0:
        info.unsafe_store(0, Int32(0))


def p2m_ccount_add_item(t: Int, lab: IP, nk: FP, info: IP, cnt: IP, n: Int, n_classes: Int):
    """Stage 2: one atomic add per row (integer: the counts are the same in
    any order). A label outside [0, n_classes) sets info[0]."""
    var v = Int(lab.unsafe_load(t))
    if v < 0 or v >= n_classes:
        info.unsafe_store(0, Int32(1))
        return
    _ = Atomic[DType.int32].fetch_add(cnt + v, Int32(1))


def p2m_ccount_emit_item(t: Int, lab: IP, nk: FP, info: IP, cnt: IP, n: Int, n_classes: Int):
    """Stage 3: nk[c] = cnt[c] as float32 (exact below 2**24 rows a class)."""
    nk.unsafe_store(t, Float32(cnt.unsafe_load(t)))


def p2m_const_init_item(t: Int, x: FP, flag: IP, n: Int, d: Int):
    """Stage 1, one item: flag = 0."""
    flag.unsafe_store(0, Int32(0))


def p2m_const_cmp_item(t: Int, x: FP, flag: IP, n: Int, d: Int):
    """Stage 2: flag = 1 when cell t differs from row 0 of its column (a NaN
    differs from everything), so flag 0 means every column is constant
    (sklearn's ptp == 0 over all features). Every writer stores the same 1."""
    if x.unsafe_load(t) != x.unsafe_load(t % d):
        flag.unsafe_store(0, Int32(1))


def p2m_fill_item(t: Int, res: FP, count: Int, value: Float32):
    res.unsafe_store(t, value)


def p2m_iota_item(t: Int, res: IP, count: Int):
    res.unsafe_store(t, Int32(t))


def p2m_negate_item(t: Int, x: FP, res: FP, count: Int):
    """res = -x, a sign flip (exact, keeps subnormals and the zero sign)."""
    res.unsafe_store(t, -x.unsafe_load(t))


def p2m_transpose_item(t: Int, src: FP, res: FP, r: Int, c: Int):
    """res[j, i] = src[i, j] for the r x c src; t = j*r + i."""
    var j = t // r
    var i = t - j * r
    res.unsafe_store(t, src.unsafe_load(i * c + j))


def p2m_transpose_i_item(t: Int, src: IP, res: IP, r: Int, c: Int):
    var j = t // r
    var i = t - j * r
    res.unsafe_store(t, src.unsafe_load(i * c + j))


# ------------------------------------------------------------------ segmented sort
@always_inline
def _row_less(a: Int, b: Int, dists: FP, rowid: IP, nnz: Int) -> Bool:
    """Key order (row, value, position); the padding position nnz sorts
    last. Positions are distinct, so the order is total and unique."""
    if a == nnz:
        return False
    if b == nnz:
        return True
    var ra = rowid.unsafe_load(a)
    var rb = rowid.unsafe_load(b)
    if ra != rb:
        return ra < rb
    var va = dists.unsafe_load(a)
    var vb = dists.unsafe_load(b)
    if va < vb:
        return True
    if vb < va:
        return False
    return a < b


def p2m_row_sort_init_item(
    t: Int, indptr: IP, cols: IP, dists: FP, out_cols: IP, out_d: FP, perm: IP, rowid: IP,
    nq: Int, nnz: Int, p: Int, n_steps: Int,
):
    """Stage 1, t < p: perm[t] = t (padding slots nnz); for t < nnz, rowid[t]
    = the row r with indptr[r] <= t < indptr[r + 1] (a binary search)."""
    perm.unsafe_store(t, Int32(t if t < nnz else nnz))
    if t < nnz:
        var lo = 0
        var hi = nq
        while hi - lo > 1:
            var mid = (lo + hi) // 2
            if Int(indptr.unsafe_load(mid)) <= t:
                lo = mid
            else:
                hi = mid
        rowid.unsafe_store(t, Int32(lo))


def p2m_row_sort_step_item(
    t: Int, j: Int, indptr: IP, cols: IP, dists: FP, out_cols: IP, out_d: FP, perm: IP, rowid: IP,
    nq: Int, nnz: Int, p: Int, n_steps: Int,
):
    """Stage 2, network step j (as nc_median_step_item): one compare pair
    per item."""
    var kk = 1
    var rem = j
    while rem >= kk:
        rem -= kk
        kk += 1
    var jj = kk - 1 - rem
    var stride = 1 << jj
    var lo = ((t >> jj) << (jj + 1)) | (t & (stride - 1))
    var hi = lo + stride
    var up = (lo & (1 << kk)) == 0
    var a = Int(perm.unsafe_load(lo))
    var b = Int(perm.unsafe_load(hi))
    var swap = _row_less(b, a, dists, rowid, nnz) if up else _row_less(a, b, dists, rowid, nnz)
    if swap:
        perm.unsafe_store(lo, Int32(b))
        perm.unsafe_store(hi, Int32(a))


def p2m_row_sort_emit_item(
    t: Int, indptr: IP, cols: IP, dists: FP, out_cols: IP, out_d: FP, perm: IP, rowid: IP,
    nq: Int, nnz: Int, p: Int, n_steps: Int,
):
    """Stage 3, t < nnz: the sorted entries."""
    var s = Int(perm.unsafe_load(t))
    out_cols.unsafe_store(t, cols.unsafe_load(s))
    out_d.unsafe_store(t, dists.unsafe_load(s))
