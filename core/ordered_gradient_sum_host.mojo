# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU side of `mojolearn.parallel_training.ordered_sum_gradients` (lane
ogs-cpu, 2026-11-24): the ordered left fold of logical gradient shards on
the host, the reference column the GPU columns are checked against.

THE FOLD ORDER, and where the GPU fixes it. The door
(bindings/parallel_mlp_native_door.mojo `neural_ordered_sum_on`) copies the
shards once into one float32 block of `shards` slots of `count` words (slot k
is shard k, the tensors concatenated in the caller's order). The GPU side
(bindings/parallel_mlp_native_run.mojo `mlp_ordered_fold`) folds it in
ascending shard order, step k = 1 .. shards-1 one `accumulate` launch over the
[2, count] pair (slot k-1, slot k), whose one tree level
(training/samba_ops.mojo `samba_tree_level_kernel`) stores, per element e,

    slot_k[e] = ftz(ftz(slot_{k-1}[e]) + ftz(slot_k[e]))      (one rounding)

so the total of element e is s0, then folded with s1, s2, ... in that order.
Here the same recurrence runs per element through the one host ordered fold
of vd-functions (core/hotpath_impl_rows.mojo `ordered_fold_f32`, step
`ordered_fold_step`: the same ftz on both operands and on the sum, one float32
rounding; its NaN canonicalization never fires, because every operand is
refused unless finite and a sum of two finite floats is finite or an
infinity), with the running total in slot 0. Same operands, same order, same
rounding per step: the same bytes by construction.

THE REFUSALS are the GPU side's. Each GPU step uploads its pair through the
non-finite scan (`_upload_checked`, words "non-finite accumulate parts at
flat index i" of the [2, count] pair): every input slot, and the running
totals after steps 1 .. shards-2, are refused when not finite; the final
total is not scanned. An infinite running total stays infinite under further
finite additions (no NaN can form from finite shards), so scanning the
running total once, after step shards-2, refuses exactly the inputs the GPU
refuses.

PARALLEL: the elements are independent (the shard order inside each element
is fixed), so `ordered_fold_f32` splits them into tasks across the host's
cores (worker count from the allocation, core/host_predict_threads.mojo);
the fold across shards is serial by definition, step k reads step k-1's
total. The finite scans are the parallel scan of core/host_finite_scan.mojo.

Pure Mojo, no Python: checked by checks/native_tests/ordered_gradient_sum_cpu.mojo.
"""
from core.host_finite_scan import host_all_finite_f32, host_first_nonfinite_f32
from core.hotpath_impl_rows import ordered_fold_f32


def _ogs_refuse_nonfinite(addr: Int, words: Int, pair_count: Int, first_slot_input: Bool) raises:
    """The GPU side's refusal words for `words` float32 at `addr`. The index
    named is the one the GPU's [2, count] pair scan names: an input in slot
    0 or 1 at its flat index, a later input in the pair's second half, a
    running total (`first_slot_input` False) in the pair's first half."""
    var p = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)
    if host_all_finite_f32(p, words):
        return
    var flat = host_first_nonfinite_f32(p, words)
    var shown = flat
    if first_slot_input and flat >= 2 * pair_count:
        shown = pair_count + flat % pair_count
    raise Error("mojolearn samba ops: non-finite accumulate parts at flat index " + String(shown))


def ordered_gradient_sum_cpu(base: Int, count: Int, shards: Int) raises -> Int:
    """Fold the `shards` slots of `count` float32 words at byte address
    `base` in place, in ascending shard order; return the slot that holds
    the total (always 0). `shards >= 2` and `count >= 1` (the door returns a
    single shard as its copy and refuses empty tensors before this)."""
    if shards < 2 or count < 1:
        raise Error("ordered reduction: the CPU fold needs at least two shards of nonempty tensors")
    _ogs_refuse_nonfinite(base, shards * count, count, True)
    # The logical order of the folded shards: slots 1 .. shards-1, ascending.
    var table = List[Int64](capacity=shards - 1)
    for shard in range(1, shards):  # driver-loop: over logical gradient shard addresses, ascending
        table.append(Int64(base + 4 * shard * count))
    var table_addr = Int(table.unsafe_ptr())
    if shards > 2:
        _ = ordered_fold_f32(base, table_addr, shards - 2, count)
        # the running total the GPU's last step uploads and scans
        _ogs_refuse_nonfinite(base, count, count, False)
    _ = ordered_fold_f32(base, table_addr + 8 * (shards - 2), 1, count)
    _ = table^  # the shard address table is read by the fold above: keep it alive through it
    return 0
