"""The GreedyLogSum float-column border build, one definition for the device
and the host column (lane hr2-gbdt-host, 2026-10-02).

WHAT MOVED. `train._quantize_training_columns` drew the border subsample on
the host (a serial `TRandom` Fisher-Yates / rejection draw), gathered every
float column through it on host threads, and ran `calc_quantization` per
column on host threads. This file is the whole build on the device:

  1. THE SUBSAMPLE, PARALLEL. Row `i < sn` of the sample is `perm(i)` for a
     keyed Feistel bijection on `[0, 2^bits)` cycle-walked into `[0, n)`
     (`border_sample_row`). A bijection draws WITHOUT REPETITION, the
     property their `SampleIndices` provides; every index is computed
     independently, so the draw is one GPU thread per sample row. (The set
     drawn differs from the old `TRandom` draw: same-bits-within-a-version
     is the contract, not old bits.)
  2. THE KEYS. `border_keys_kernel`: per (column, sample row) the value,
     flushed (`ftz`, DEVIATION 5900's flushed search), as the monotone
     sortable twiddle; a NaN becomes `0xFFFFFFFF`, which no value maps to,
     so NaNs sort last and are counted per column (their `filterNans`).
  3. THE SORT. `launch_segmented_radix_sort`, every column of the chunk at
     once (one segment per column).
  4. THE SEARCH. `gls_column`: CatBoost's GreedyLogSum (`binarization.mojo`
     `best_split`, libc++ heap semantics included) over the sorted keys,
     one GPU thread per column, its heap in global scratch. The split score
     `log(l + 1e-8) + log(r + 1e-8) - log(n + 1e-8)` reads a table of
     `log(w + 1e-8)` built by one thread per `w` in soft binary64
     (`checks/soft_f64.mojo`), so every vendor and the host column score
     the same words; the sums are soft binary64 too.
  5. The midpoints, flushed and with a pinned product (no contraction),
     sorted by the twiddled key and de-duplicated.

The host column (`gbdt/host/gbdt_oracle.mojo::gbdt_host_grid`) calls the
same `border_sample_row`, `border_key`, `gls_log_table_entry` and
`gls_column` on host memory, so the grids agree by construction.
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import bitcast

from checks.numerics import ftz, pinned_mul_f32
from checks.soft_f64 import (
    sf64_add,
    sf64_from_int,
    sf64_log,
    sf64_lt,
    sf64_sub,
)

from gbdt.grid_creator.gls_borders_cells import (
    GLS_EPS8,
    GLS_NO_SCORE,
    GLS_NAN_KEY,
    GLS_BLOCK,
    _mix64,
    border_sample_row,
    border_key,
    key_value,
    gls_log_table_entry,
    _score,
    _best,
    gls_column,
)


# ---- the kernels -------------------------------------------------------------


def gls_log_table_kernel(logtab: MutPointer[UInt64, MutAnyOrigin], n: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < Int(n):
        logtab[i] = gls_log_table_entry(i)
        i += stride


def border_sample_kernel(
    idx: MutPointer[UInt32, MutAnyOrigin], sn: Int32, n: Int32, key: UInt64
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < Int(sn):
        idx[i] = UInt32(border_sample_row(i, Int(n), key))
        i += stride


def border_keys_kernel(
    cols: MutPointer[Float32, MutAnyOrigin],
    idx: MutPointer[UInt32, MutAnyOrigin],
    keys: MutPointer[UInt32, MutAnyOrigin],
    vals: MutPointer[UInt32, MutAnyOrigin],
    nan_in_sample: MutPointer[Int32, MutAnyOrigin],
    nan_in_column: MutPointer[Int32, MutAnyOrigin],
    n_cols: Int32,
    n_rows: Int32,
    sn: Int32,
    sampled: Int32,
):
    """Grid-stride over `n_cols * max(sn, n_rows)`: the sample's keys
    (`keys[c * sn + i]`, NaN counted per column) and, on the sampled path,
    the full column's NaN flag."""
    var nc = Int(n_cols)
    var n = Int(n_rows)
    var s = Int(sn)
    var span = s
    if sampled != 0 and n > s:
        span = n
    var total = nc * span
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < total:
        var c = i // span
        var j = i - c * span
        if j < s:
            var row = Int(idx[j]) if sampled != 0 else j
            var v = cols[c * n + row]
            var k = border_key(v)
            keys[c * s + j] = k
            vals[c * s + j] = UInt32(j)
            if k == GLS_NAN_KEY:
                _ = Atomic.fetch_add(nan_in_sample + c, Int32(1))
        if sampled != 0 and j < n:
            var v2 = cols[c * n + j]
            if v2 != v2:
                _ = Atomic.fetch_add(nan_in_column + c, Int32(1))
        i += stride


def gls_columns_kernel(
    keys: MutPointer[UInt32, MutAnyOrigin],
    sn: Int32,
    n_cols: Int32,
    valid: MutPointer[Int32, MutAnyOrigin],
    budget: MutPointer[Int32, MutAnyOrigin],
    logtab: MutPointer[UInt64, MutAnyOrigin],
    h_start: MutPointer[Int32, MutAnyOrigin],
    h_end: MutPointer[Int32, MutAnyOrigin],
    h_split: MutPointer[Int32, MutAnyOrigin],
    h_score: MutPointer[UInt64, MutAnyOrigin],
    heap_cap: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
    out_cap: Int32,
    counts: MutPointer[Int32, MutAnyOrigin],
):
    """One thread per column of the chunk: `gls_column` over its sorted
    keys' first `valid[c]` entries with `budget[c]` borders."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= Int(n_cols):
        return
    var hc = Int(heap_cap)
    var oc = Int(out_cap)
    counts[c] = Int32(
        gls_column(
            keys + c * Int(sn), Int(valid[c]), Int(budget[c]), logtab,
            h_start + c * hc, h_end + c * hc, h_split + c * hc,
            h_score + c * hc, dst + c * oc,
        )
    )


#: `gls_budget_kernel`'s mode word for a NaN under `nan_mode=Forbidden`
comptime GLS_MODE_REFUSED = Int32(-1)


def gls_budget_kernel(
    nan_in_sample: MutPointer[Int32, MutAnyOrigin],
    nan_in_column: MutPointer[Int32, MutAnyOrigin],
    sn: Int32,
    sampled: Int32,
    n_cols: Int32,
    border_count: Int32,
    nan_mode_option: Int32,
    forbidden: Int32,
    valid: MutPointer[Int32, MutAnyOrigin],
    budget: MutPointer[Int32, MutAnyOrigin],
    mode: MutPointer[Int32, MutAnyOrigin],
):
    """Per column: `ComputeNanMode` over the column's NaN flag (the full
    column on the sampled path), the non-NaN border budget, and the count
    of non-NaN sorted keys."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= Int(n_cols):
        return
    var ns = nan_in_sample[c]
    var has_nan = ns > 0
    if sampled != 0:
        has_nan = nan_in_column[c] > 0
    var m = forbidden
    if has_nan:
        if nan_mode_option == forbidden:
            m = GLS_MODE_REFUSED
        else:
            m = nan_mode_option
    var b = border_count
    if m != forbidden:
        b -= 1
    valid[c] = sn - ns
    budget[c] = b
    mode[c] = m
