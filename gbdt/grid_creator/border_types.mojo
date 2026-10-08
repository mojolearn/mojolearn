"""The six non-default numeric border types on the device, one definition for
the device and the host column (cpu-gpu-cleanup w2-trees, 2026-10-02).

WHAT MOVED. `train._quantize_training_columns` built the Median, Uniform,
UniformAndQuantiles, MaxLogSum, MinEntropy and GreedyMinEntropy grids on
the host: a serial `TRandom` subsample, a host-thread gather (`_draw_task`)
and `select_borders` per column on host threads (`_dp_task`). They now ride
GreedyLogSum's device pipeline (`gls_borders.mojo`: the Feistel subsample,
the flushed twiddled keys, the segmented sort, `ComputeNanMode`) and differ
only in the per-column search, which runs here, one GPU thread per column
of a chunk, on the sorted keys:

  * Median / Uniform / UniformAndQuantiles: `simple_column`, their
    `GenerateMedianBorders`, `RegularBorder` and uniform values
    (`binarization.mojo` `_median_borders`, `_regular_border`,
    `_uniform_value`) over the sorted keys, every float32 product pinned and
    the uniform step through `portable_divf` (one correctly rounded
    division on every column);
  * GreedyMinEntropy: `gls_column` itself over a score table of
    `-(w * log(w + 1e-8))` (`TGreedyBinarizer<MinEntropy>` is
    `TGreedyBinarizer<MaxSumLog>` with the other penalty, the same heap);
  * MaxLogSum / MinEntropy: `exact_column`, `_exact_best_split`'s E_RLM2
    dynamic program statement for statement, every double operation in soft
    binary64 (`checks/soft_f64.mojo`; the Apple GPU has no float64) and the
    penalty read from a table (`border_table_entry`): the DP's penalty
    arguments are differences of integer count sums, so `Penalty(w)` for
    `w` in `[0, sample]` is every value it can ask for.

Every result then goes through `finish_borders` (`SetQuantization`: -0.0
written as +0.0, sorted, de-duplicated; the sabotage arm as
`select_borders`). The host column (`gbdt/host/gbdt_oracle.mojo`) calls the
same functions on host memory, so the grids agree by construction.

BITS. The subsample is GreedyLogSum's (the old `TRandom` draw and its NaN
seed into the sample are gone), so the six types' grids move once on the
sampled path; the searches themselves are the old host arithmetic
(`portable_log64` is `sf64_log` statement for statement, and IEEE
add/mul/div are what soft binary64 returns).
"""
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from checks.numerics import ftz, pinned_mul_f32, portable_divf
from checks.soft_f64 import (
    sf64_add,
    sf64_from_int,
    sf64_gt,
    sf64_lt,
    sf64_mul,
    sf64_neg,
)
from gbdt.grid_creator.binarization import (
    BORDER_TYPE_GREEDY_LOG_SUM,
    BORDER_TYPE_GREEDY_MIN_ENTROPY,
    BORDER_TYPE_MAX_LOG_SUM,
    BORDER_TYPE_MEDIAN,
    BORDER_TYPE_MIN_ENTROPY,
    BORDER_TYPE_UNIFORM,
    BORDER_TYPE_UNIFORM_AND_QUANTILES,
    BORDER_TYPES_SABOTAGE,
)
from gbdt.grid_creator.gls_borders_cells import (
    border_key,
    gls_column,
    gls_log_table_entry,
    key_value,
)

from gbdt.grid_creator.border_types_cells import (
    BT_EPS12,
    border_table_entry,
    finish_borders,
    _lower_bound,
    _regular_border,
    _median_borders,
    _uniform_value,
    simple_column,
    _pen,
    exact_column,
    _thresholds_to_borders,
    border_type_column,
    exact_i32_words,
)


def border_table_kernel(tab: MutPointer[UInt64, MutAnyOrigin], n: Int32, border_type: Int32):
    """One thread per entry, grid-stride."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < Int(n):
        tab[i] = border_table_entry(i, Int(border_type))
        i += stride


def border_types_columns_kernel(
    border_type: Int32,
    keys: MutPointer[UInt32, MutAnyOrigin],
    sn: Int32,
    n_cols: Int32,
    valid: MutPointer[Int32, MutAnyOrigin],
    budget: MutPointer[Int32, MutAnyOrigin],
    tab: MutPointer[UInt64, MutAnyOrigin],
    h_start: MutPointer[Int32, MutAnyOrigin],
    h_end: MutPointer[Int32, MutAnyOrigin],
    h_split: MutPointer[Int32, MutAnyOrigin],
    h_score: MutPointer[UInt64, MutAnyOrigin],
    heap_cap: Int32,
    f64s: MutPointer[UInt64, MutAnyOrigin],
    f64_stride: Int64,
    i32s: MutPointer[Int32, MutAnyOrigin],
    i32_stride: Int64,
    uniq: MutPointer[Float32, MutAnyOrigin],
    uniq_stride: Int64,
    dst: MutPointer[Float32, MutAnyOrigin],
    out_cap: Int32,
    counts: MutPointer[Int32, MutAnyOrigin],
):
    """One thread per column of the chunk: `border_type_column` over its
    sorted keys' first `valid[c]` entries with `budget[c]` borders. A
    scratch plane the type does not read may be one word (stride 0)."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= Int(n_cols):
        return
    var hc = Int(heap_cap)
    counts[c] = Int32(
        border_type_column(
            Int(border_type), keys + c * Int(sn), Int(valid[c]), Int(budget[c]), tab,
            h_start + c * hc, h_end + c * hc, h_split + c * hc, h_score + c * hc,
            f64s + c * Int(f64_stride), i32s + c * Int(i32_stride),
            uniq + c * Int(uniq_stride), dst + c * Int(out_cap),
        )
    )
