# SPDX-License-Identifier: Apache-2.0
"""Opt-in Apple FAST row gather, recovered independently of old resample bundle."""
from max.gpu import block_idx, block_dim, thread_idx
from resample.checks.index_map import permutation_merge_place


def gather_rows_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin], count_in: Int32, d_in: Int32,
):
    var count = Int(count_in)
    var d = Int(d_in)
    var o = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if o < count * d:
        var r = o // d
        dst.unsafe_store(o, src.unsafe_load(Int(rows.unsafe_load(r)) * d + o % d))

# A 32-column by 8-row output tile reads each selected row index once.
# Adjacent lanes read contiguous features; width tails are masked. This is
# a bandwidth/reuse experiment, never an input-width dispatch shortcut.
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

# AFCL-P10: NEVER RUN — PENDING MEASUREMENT. Uncompiled/unverified.
# Keep 256 threads but exchange feature reuse for more rows per block. This
# covers every width with masked tails; it is not an input-width dispatch rule.
comptime AFCL_P10 = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU and is_defined["MOJOLEARN_AFCL_P10"]()
comptime GATHER_COLS = 16 if AFCL_P10 else 32
# TOMBSTONE: MOJOLEARN_AFCL_P10 (never-run Apple FAST programme card: 16-column gather tiles) deleted 2026-11-15 by lane/expdel2-37-series; code recoverable at 72ae22cf1f.
# Restore: git apply experiments/removed/MOJOLEARN_AFCL_P10.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-classical: Apple fast is paused, not abandoned (Andrew); opt-in, default off, Apple fast only.
comptime GATHER_ROWS = 256 // GATHER_COLS


def gather_rows_tiled_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin], count_in: Int32, d_in: Int32,
):
    var tid = Int(thread_idx.x)
    var local_row = tid // GATHER_COLS
    var row = Int(block_idx.y) * GATHER_ROWS + local_row
    var col = Int(block_idx.x) * GATHER_COLS + tid % GATHER_COLS
    var selected = stack_allocation[GATHER_ROWS, Int32, address_space=AddressSpace.SHARED]()
    if tid < GATHER_ROWS:
        var output_row = Int(block_idx.y) * GATHER_ROWS + tid
        selected[tid] = rows.unsafe_load(output_row) if output_row < Int(count_in) else Int32(0)
    barrier()
    if row < Int(count_in) and col < Int(d_in):
        dst.unsafe_store(row * Int(d_in) + col, src.unsafe_load(Int(selected[local_row]) * Int(d_in) + col))


# Stable total-order merge passes over device-generated uint64 draw keys.
# Every position scatters to its unique rank in one merged pair; kernel
# boundaries preserve visibility. No host key/index materialization. The
# per-slot body is `permutation_merge_place` (resample/checks/index_map.mojo),
# shared with the host column's parallel passes (lane vdf-misc3).
def permutation_positions_kernel(rows: MutPointer[Int32, MutAnyOrigin], n_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        rows[i] = Int32(i)

def permutation_merge_kernel(
    dst: MutPointer[Int32, MutAnyOrigin], src: MutPointer[Int32, MutAnyOrigin],
    keys: MutPointer[UInt64, MutAnyOrigin], n_in: Int32, width_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    permutation_merge_place(dst, src, keys, Int(n_in), Int(width_in), i)


# TOMBSTONE: MOJOLEARN_CLASSICAL_C11_DRAW_GATHER (opt-in IDENTICAL arm, no A/B recorded; non-owner site, owner expdel2-07-base): classical_draw_gather_kernel, launched only by that arm, deleted 2026-11-15 by lane/expdel2-37-series; code recoverable at 72ae22cf1f.
# Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C11_DRAW_GATHER.expdel2-37-series.patch; record in docs/TOMBSTONES.md.
