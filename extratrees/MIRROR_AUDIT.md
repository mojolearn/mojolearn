# Are we doing what cuML does? An operation-by-operation audit

Written 2026-08-21, because "we mirror cuML" is a claim and a claim needs a
table. `PORTING_RULES.md` rule 2 is the standard being audited against:

> If they do something on the GPU in the control plane, we do it on the GPU. If
> they keep a decision on the device so the host never learns it, we keep it on
> the device. If they pass a value as a kernel argument, we pass it as a kernel
> argument. **The host/device split is part of the algorithm, not an
> implementation detail we get to re-decide.**

cuML pinned at `00094f7`. Every row below was read out of their source this
session, not remembered.

## `Builder::train` — the outer loop, `builder.cuh:344-359`

| step | cuML | ours | verdict |
|---|---|---|---|
| the node queue | HOST (`std::deque` + `std::vector`) | HOST | MIRRORED |
| `queue.Pop()` batch | HOST | HOST | MIRRORED |
| `queue.Push()` after each batch | HOST | HOST | MIRRORED |
| `SetLeafPredictions` at the end | device kernel | device kernel | MIRRORED |

Their queue being on the host is not a concession, it is their design: `doSplit`
ends with `raft::update_host(h_splits, splits, work_items.size())` and a
`sync_stream` (`:492-494`), so the batch's chosen splits cross to the host every
level BY CONSTRUCTION. Ours crosses exactly the same thing.

## `Builder::doSplit` — the per-batch work, `builder.cuh:386-494`

| step | their line | cuML | ours | verdict |
|---|---|---|---|---|
| `memset(n_nodes, 0)` | `:390` | device | **absent** | THEIRS IS DEAD CODE — `n_nodes` is allocated (`:173`, `:280`, `:314`), memset here, and **read by nothing**. Grepped this session. Not porting a dead counter. |
| `initSplit(splits, ...)` | `:391` | device kernel | device kernel (`split_reduce_init_kernel`) | MIRRORED |
| `update_device(d_work_items, ...)` | `:394` | H2D copy | H2D copy | MIRRORED |
| `updateWorkloadInfo` | `:365-385` | **HOST**, then one H2D copy | HOST, then one H2D copy | MIRRORED — theirs builds `h_workload_info` on the host too |
| feature sampling | `:398-471` | **device kernels** | device kernels (algo-L arm host-side, deviation 201) | MIRRORED |
| the split search | `:475-477` | device kernel | device kernels | MIRRORED in placement, DEVIATION 137 in content |
| the column-block loop `c += n_blks_for_cols` | `:475` | 10 columns per launch | all columns in one launch | EXPLAINED BELOW |
| `launchNodeSplitKernel` (partition) | `:478-491` | device kernel | device kernel | MIRRORED |
| `update_host(h_splits, ...)` | `:492-494` | D2H copy | D2H copy | MIRRORED |

## The column-block loop, and why not mirroring it is correct

`n_blks_for_cols = 10` (`:193`) exists for ONE reason, and their own code says
so: it sizes the histogram workspace,
`max_batch * max_n_bins * n_blks_for_cols * num_outputs` (`:278`, `:312`), and
the loop at `:475` exists so that workspace stays bounded — their comment at
`:474` is "iterate through a batch of columns (to reduce the memory pressure)".

**DEVIATION 137 deletes the histogram.** There is no `max_n_bins`, no bin
dimension, and the per-cell state is a handful of accumulators. The quantity
their loop bounds does not exist here, so the loop has nothing to bound. Our
`gridDim.y` is the full sampled-column count, which is what theirs would be if
`n_blks_for_cols` were `n_sampled_cols`.

That is a structural difference from their kernel strategy and it is recorded
as one rather than left to be noticed. It is NOT a licence to differ elsewhere:
the grid SHAPE is theirs (`gridDim.x` indexes `WorkloadInfo`, `gridDim.y` is
the feature), and every kernel in this lane keeps it.

## `RandomForest::fit` — the forest, `randomforest.cuh:150-195`

| step | their line | cuML | ours | verdict |
|---|---|---|---|---|
| the tree loop | `:165` | HOST, `#pragma omp parallel for` over `n_streams` | HOST, serial | **Metal has no streams** — traps register. Trees are independent, so the answer does not depend on order. |
| `get_row_sample`, `bootstrap=false` | `:68-70` | `thrust::sequence` on DEVICE | host `List`, uploaded once | MINOR DRIFT, noted below |
| the dataset | `dataset.h:22-38` | resident for the whole fit | resident for the whole fit | MIRRORED (deviation 184, closed) |
| `predict` | `:229-242` | HOST loop, `+=` per tree | HOST loop, `+=` per tree | MIRRORED (deviation 147) |

**The `row_sample` drift is FIXED** (deviation 200): `row_ids_sequence_kernel`
fills it on the device, which is what `thrust::sequence` does. What remains is
a LIFETIME difference — their buffer is owned by `fit`, ours by the trainer —
not a host/device one.

## What was WRONG and is now fixed

1. **The gain was computed on the host.** cuML computes `GainPerSplit` inside
   `computeSplitKernel` (`objectives.cuh:52-83`) and the host never sees a
   per-candidate gain. Deviation 183's first closure copied `status`,
   `n_total` and `acc_total` back per level to form it host-side. It was
   correct and it was the wrong shape. **Fixed**: the gain is computed on the
   device from their expression, the three readbacks are gone, and
   `best_metric_val` is now bit-identical between the two paths.
2. **The feature sampler ran on the host.** FIXED: `sample_features_device`
   enqueues their kernel, bit-identical to the host oracle over 23,462 asserted
   slots, with both `MAX_SAMPLES_PER_THREAD` instantiations and the
   all-features arm named by the DEVICE's own report.

   **One arm of it cannot run on Metal and is named rather than hidden.**
   cuML's algorithm L is a `double` algorithm in four places
   (`builder_kernels.cuh:291`, `:306` twice, `:313`) and Metal rejects `double`
   at COMPILE time. That arm takes the host transcription — the same
   algorithm, the checked oracle — and the placement is reported in the
   returned plan. A `Float32` substitute was written and REJECTED: at the `k/n`
   their dispatch routes there, `W ~ 1 - 1e-4`, so forming `1 - W` in `Float32`
   leaves about 13 bits. Deviations 199 and 201.

3. **`row_ids` was built on the host.** FIXED, deviation 200.

## What is deliberately NOT mirrored, with the reason

Every one of these is a numbered deviation with a measurement behind it, not a
preference:

| not mirrored | why | entry |
|---|---|---|
| the histogram in `computeSplitKernel` | this formulation exists to delete it | 137 |
| `quantiles.cuh` | same | `UNPORTED.tsv` |
| `signalDone` / last-block election | `threadfence` is NVIDIA-only in Mojo | 170 |
| `atomicAdd` on a float RANGE | no portable float `atomicMin`/`Max` | 161 |
| `cub::BlockRadixSort` | no MAX counterpart; hand-written portably per 0b-i | 157 |
| `Int128` in a kernel | does not compile; `XPC_ERROR_CONNECTION_INTERRUPTED` | 167 |
| whole-struct kernel arguments | kills the Metal compiler | 162 |
| dynamic shared memory | `stack_allocation` is comptime-sized | 172, 178 |
| their `mask[0]` tile predecessor | **it is a BUG**: column 0 never drawn | 164 |
| their `n-1` filler | **it is a BUG**: column `n-1` over-drawn 662 vs 512 | 165 |
| `Split::update`'s metric arm on an exact tie | ordering by float noise; 145's argument, measured twice | 194 |
| `adaptive_sample_kernel` | dead code in cuML | `UNPORTED.tsv` |
| `n_nodes` | dead code in cuML | this file |

## How to re-run this audit

It is a grep, not a memory:

    cd ~/CascadeProjects/upstream/cuml/cpp/src/decisiontree/batched-levelalgo
    sed -n '386,494p' builder.cuh        # doSplit, every step in order
    grep -n 'n_blks_for_cols' builder.cuh
    grep -rn 'n_nodes' builder.cuh kernels/*.cuh

Then read `extratrees/ported/decisiontree/batched_levelalgo/builder.mojo`'s
`train_classification_device_resident` beside it. The two should read as the
same list of steps in the same order. **Where they do not, either there is a
numbered deviation or there is a defect, and there is no third possibility.**
