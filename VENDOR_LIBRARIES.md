# Vendor primitives, mojolearn sections (cluster, neighbors, decomposition, dbscan, glm)

> **RULE CHANGE, Andrew, 2026-08-19 (evening). READ THIS BEFORE THE REST OF
> THIS FILE.** The rule this document was written to serve — "where the
> incumbent calls a vendor primitive, call OURS" — is **DELETED**. It is
> replaced by one rule: **FOLLOW THEIR DISPATCH.** Port the path their library
> actually takes for the parameters in question.
>
> All that survives is a narrow exception: where the path their dispatch
> actually takes calls a **CLOSED** library we cannot read or port (cuBLAS,
> cuSOLVER), call the MAX equivalent, because there is nothing to port. **CUB
> and Thrust are OPEN.** Their kernels are readable, so they are PORT
> candidates, not substitution candidates.
>
> **What killed it: a device-wide vendor call cannot be fused.** It reads its
> input from memory and writes its output to memory, by construction.
> Substituting one for a step that belongs INSIDE a kernel freezes the unfused
> structure permanently. k-NN substituted `linalg.matmul` for its distance step
> and `nn.topk` for its selection under the old rule, so the distance matrix had
> to be materialized, so a 13 ms job takes 306 ms — while cuVS's default path
> for those same parameters (`knn_brute_force.cuh:443` -> `fusedL2Knn`) calls no
> vendor primitive at all. It is their own tile kernel with a register-resident
> `faiss_select::WarpSelect` queue. **When they fuse, they hand-write.**
>
> The probe results below are still VALID and still useful — what MAX ships and
> what it does not is a fact. What is no longer valid is treating an available
> primitive as a reason to substitute. Before citing any row of this file,
> answer in writing: does their dispatch take this path for these parameters,
> and is this step standalone in THEIR code too? If either answer is no, port
> their kernel.

**Companion to `VENDOR_LIBS.md`, which covers the CatBoost boosting port.**
Same rule, different upstreams. Where the two disagree, the probe result in
whichever file was written later wins, and both record the exact path tried.

**The rule, from Andrew, 2026-08-19: where the incumbent calls a vendor
primitive, call OURS. Hand-write only where no equivalent exists, and CHECK
that rather than assume it.**

Every one of these libraries sits on cuBLAS, cuSOLVER, CUB and Thrust. Those
are where a large share of their performance lives, and reimplementing one by
hand is almost always worse than calling the tuned equivalent. This file
exists because I assumed twice that no equivalent existed and was wrong both
times, at a measured cost.

Everything below is **probed against this toolchain**, not read from
documentation. `AVAILABLE` means the import compiled. `NOT FOUND` means it
did not, and the exact path tried is recorded so nobody assumes the search
was never done.

---

## THE CORRECTION THIS FILE OPENS WITH

**Mojo HAS warp primitives.** This repository has claimed since its first
commit that it does not, in `PORTING.md 2`, `PORTING.md 8`, in the reasoning
that made RAFT's `select_warpsort.cuh` "untranslatable", and in the memory
notes. That claim was wrong. The namespace is `std.gpu.primitives.warp`, and
the earlier searches looked under `std.gpu`, `max.gpu`, `std.gpu.block` and
`max.gpu.block`, all of which miss the `primitives` level.

    AVAILABLE  std.gpu.primitives.warp   shuffle_down, shuffle_idx,
                                         shuffle_xor, lane_id, prefix_sum,
                                         reduce, sum, max, broadcast
    AVAILABLE  max.gpu.sync              syncwarp
    AVAILABLE  max.gpu.primitives.block  sum, max, min, broadcast, prefix_sum

What follows from it has to be re-derived rather than assumed:

- `cub::WarpScan` and `cub::ShuffleIndex` in `histogram_utils.cu`, which
  `PORTING.md 8` replaced with a block scan and called a fidelity loss, may
  now be portable directly.
- RAFT's `select_warpsort.cuh` was ruled out at 14 warp intrinsics on the
  same false premise, and under the rule change at the top of this file it is
  now the file that MATTERS. cuVS's dispatch for a normal k does not call
  `select_k` standalone at all: `knn_brute_force.cuh:443` goes to
  `fusedL2Knn`, whose selection is a register-resident
  `faiss_select::WarpSelect` queue inside the distance kernel, and
  `select_warpsort.cuh` is the port of that family. Finishing it is the work;
  the three `set_k_th_` queues in `neighbors/UNPORTED.tsv` are what is left.

**A float `atomicAdd` DOES work on Metal.** Probed on this M4: 1024 threads
each adding 1.0 return exactly 1024.0. This file previously called it a
hardware limit; that was wrong, and every design in this tree that cited it
has to be re-argued on its own merits. `mojo_only/fixed_point.mojo` survives
the re-argument, but on a different ground: an integer accumulator is exactly
order-independent and a float atomic is not, so fixed point is what makes a
fit reproducible run to run and device to device. It is a CHOICE about
determinism, not a workaround for missing hardware.

---

## Device-wide calls: real substitution candidates

| their file | vendor call | Mojo equivalent | ours today |
|---|---|---|---|
| `split_points.cu` | `cub::DeviceRadixSort::SortPairs` | `nn.argsort.argsort` AVAILABLE | hand-written stable partition |
| `split_points.cu` | `cub::DeviceScan::ExclusiveSum` | **`nn.cumsum` IS CPU-ONLY** and is not a swap candidate: its signature carries neither `ctx: DeviceContext` nor `target`, unlike `argsort`, `top_k` and `gather` which carry both. Corrected here after the first version of this table listed it as available. | hand-written chunk offsets |
| `cuda_util/reduce.cu` | `cub::DeviceReduce::Reduce` | **FOUND, and this table said NOT FOUND.** `algorithm.reductions.reduce_{sum,max,min,mean,product,argmax,argmin}` take `target`, an optional `DeviceContext`, an `InputFn` and an `OutputFn`, and a `reduce_dim`. They run on CPU and GPU through the `algorithm.rowwise` scaffolder. The earlier search tried `algorithm.reduce` and `algorithm.reduction`, both singular, and missed the plural module. | not used, and the `algorithm.reductions` section below says why that is now the expected answer rather than a gap |
| `cuda_util/reduce.cu` | `cub::DeviceSegmentedReduce` | **NOT FOUND.** `algorithm.reductions` reduces a DENSE axis of a rectangular shape. Our segments are leaf partitions, `[part_offset[leaf], + part_size[leaf])`, ragged and different every level, which no dense-axis reduction expresses. | hand-written `partitions_reduce`, whose per-block reduction is now `block.sum` |
| `split_points.cu` | `cub::DeviceSegmentedRadixSort` | **NOT FOUND** | segmented stable partition |
| cuVS/cuML | `cublasGemmEx` | `linalg.matmul.matmul` AVAILABLE | **WIRED**, N-T shape only |
| RAFT `lstsq.cuh` | `raft::linalg::gemv` | **`linalg.gemv.gemv` IS HOST-ONLY** (no `ctx`, no `target`; its docstring opens "Computes a CPU matrix-vector product"). The GPU counterpart is **`linalg.gemv.gemv_gpu`**, `gemv_gpu[transpose_b](c, a, b, ctx)`. | **WIRED and now the only path.** `glm/.../lstsq.mojo` step 6. The `use_vendor_gemv=False` arm and the ported contraction behind it are deleted. |
| RAFT `pca.cuh` | cuSOLVER `syevj` | **NOT FOUND** as a dense eigensolver; `linalg.qr_factorization` AVAILABLE | `jacobi_eigh_device.mojo` |
| CatBoost multiclass | cuSOLVER dense Newton solve | same gap | not ported |
| RAFT distance | `raft::stats::cov` (`OP_T, OP_N`) | **NOT BLOCKED, AND IT IS ON THE TUNED MATMUL NOW.** `transpose_a` is still refused by `linalg.matmul`, but `transpose(X) . transpose(X)^T` is the same matrix in the N-T shape it does support, and a transpose is two passes against an `O(rows * cols^2)` product. | **`core/gemm.mojo::gemm_tn`, on `linalg.matmul`.** The ported column-major contraction (`covariance_kernel`) and its split-K reduction are DELETED: about 250 lines that nothing called once `gemm_tn` took this route. |

## Block and warp scope: ordinary kernel code, not swap candidates

These are things a kernel does inside itself. They are not device-wide calls
and substituting them is writing normal kernel code, not plugging in a
library. Listed so the distinction is explicit.

| vendor call | Mojo equivalent | ours today |
|---|---|---|
| `cub::BlockReduce` | `max.gpu.primitives.block.sum[block_size=N](val)` AVAILABLE | **SUBSTITUTED** in `core/row_norms.mojo`, `core/column_stats.mojo` (2 kernels), `dbscan/vertexdeg`, `cluster/plus_plus`, `jacobi_eigh_device`, and now `ported/gpu_util/partitions_reduce.mojo`, which had been faking a reduce with a `prefix_sum` plus a `broadcast` on the belief that no block reduce shipped |
| `cub::BlockScan` | `max.gpu.primitives.block.prefix_sum` AVAILABLE | **SUBSTITUTED** in `dbscan/adjgraph`, the k-means++ device scan, and now `ported/gpu_util/kernel/reorder_one_bit.mojo` (a Hillis-Steele loop and its shared page, 16 barriers per block, down to one call). It is also substituted in `select_radix`, which stays |
| `cub::WarpScan` | `std.gpu.primitives.warp.prefix_sum` AVAILABLE | hand-written serial scan (boosting side, untouched) |
| `cub::ShuffleIndex` / `raft::shfl` | `std.gpu.primitives.warp.shuffle_xor` AVAILABLE | **SUBSTITUTED** in `unfused_distance_nn` and the fused SIMT kernel. `shuffle_xor` not `shuffle_idx`: theirs is a rotate relying on CUDA's `width` modulo, Mojo's `shuffle_idx` has no width parameter, and XOR over the same aligned group returns the identical pair because the reducer is an idempotent min over a total order. |
| `cub::BlockReduce<KeyValuePair>` | none directly; built from `warp.shuffle_xor` + a 4-way block merge | **SUBSTITUTED**. `block.min` reduces VALUES only and cannot carry the key, which is why the plain block collectives never solved this. CUB's own default is `BLOCK_REDUCE_WARP_REDUCTIONS`, a two-stage warp-then-propagate shape, so the two-stage port is closer to theirs than the old whole-block tree was. |
| `cub::BlockRadixSort` | **NOT FOUND** | n/a |
| `ThreadLoad` / `ThreadStore` cache hints | **NOT FOUND** | plain loads |
| `LoadDirectWarpStriped` | **NOT FOUND** | plain strided loads |

## Also confirmed available, unused so far

`nn.gather_scatter.gather`, `nn.topk.top_k`, `nn.softmax.softmax`,
`nn.concat.concat`, `linalg.transpose.transpose`,
`linalg.qr_factorization.qr_factorization`.

**`nn.topk.top_k` IS wired in `neighbors/`, behind `use_vendor_topk`, and the
ported RAFT radix select is still the default arm.**

This entry said, for about an hour, that `top_k` was the ONLY selection and
that both hand-written selectors were deleted. That change was MADE and then
REVERTED, and the reason is the rule change at the top of this file rather
than anything wrong with the call itself. `top_k` and the ported radix select
agree on all 512 neighbours of the fixture, which is a real result and stands.
What does not stand is the inference from "cuVS calls a vendor primitive
somewhere" to "our selection should be that primitive": for the parameters a
k-NN user actually passes, their dispatch fuses the selection INTO the
distance kernel and hand-writes it, and a device-wide `top_k` cannot be fused
into anything.

So the state is: `top_k` reachable and checked, the ported selectors kept,
and the fused path (`fused_l2_knn.mojo`) is where the work is.

One behaviour to know: `top_k` prints

    Warning: Unsorted top-k is not supported on GPU. Falling back to sorted
    top-k.

so `sorted=False` is ignored on the GPU path and you pay for a sort you did
not ask for. For k-NN that is harmless, and for anything that only needs the
SET it is wasted work. Not yet measured against the ported kernel; per
Andrew, no timing runs until the substitution passes are finished.

## Probed and NOT FOUND, recorded so the search is not repeated blindly

    algorithm.reduce, algorithm.reduction.sum, nn.reduction.reduce
        -> WRONG CONCLUSION, and this entry is what a NOT FOUND that names
           only the paths it tried is for. The module is `algorithm.
           reductions`, PLURAL, and it ships reduce_sum, reduce_max,
           reduce_min, reduce_mean, reduce_product, reduce_argmax and
           reduce_argmin, each taking an InputFn, an OutputFn, a reduce_dim,
           a `target` and an optional DeviceContext, over the
           `algorithm.rowwise` scaffolder. The InputFn IS the custom
           operator this entry says does not exist. Re-checked against the
           docs 2026-08-19.
    nn.cumsum.cumsum_exclusive
        -> cumsum is inclusive only
    nn.sort.sort
        -> no free-standing sort; argsort exists
    std.gpu.primitives.block.*
        -> block primitives are under MAX, warp primitives under STD.
           They are NOT in the same package, which is what made the first
           search miss both.

## Vendor calls that EXIST, COMPILE, and DO NOT WORK HERE

Three so far, and the pattern is the same each time: the symbol resolves, the
call typechecks, and it fails at a level the signature does not advertise.
Recorded in full because "AVAILABLE" in the table above means *the import
compiled* and nothing more.

| call | how it fails | what we do instead |
|---|---|---|
| `linalg.matmul` with `transpose_a=True` | `constraint failed: transpose_a not yet supported` at compile time | `covariance_kernel`, the hand-ported contraction |
| `linalg.matmul` with `n = 1` | returns zeros for some outputs, no error | ported contraction; RAFT calls `gemv` here anyway |
| `linalg.transpose` on device buffers | **SIGNALS at runtime** inside `linalg::transpose::_copy_with_strides[...] rank=2, dtype=f32`, a HOST strided-copy path handed DEVICE pointers | nothing, and it no longer matters: the transpose route was ABANDONED, not deferred. See the `raft::stats::cov` row. |

That third one killed the planned unblock for PCA and OLS. The identity was
right (`Xt . Xt^T == X^T X` turns the unsupported T-N shape into the
supported N-T one), it compiled, and it died on execution.
`linalg.transpose` takes an `Optional[DeviceContext]`, and **accepting one is
not the same as dispatching on it.**

**RESOLVED, and by neither of those routes.** The framing was wrong in the
same way the CUB framing was wrong: I was shopping for a vendor call to reach
a shape their own source already handles. RAFT ships `ColKernelPolicy`
alongside `KernelPolicy` (`raft/linalg/contractions.cuh:96`), and a row-major
`X` viewed column-major turns `X^T X` into the N-T shape. `covariance_kernel`
is now a port of that policy. No transpose, no `transpose_a`, no vendor call.

That is three times in one session that the answer was in their source and I
was looking for a library call instead.

## Limits of the ones we do use, both compiler-verified

`linalg.matmul`:

    constraint failed: transpose_a not yet supported

so the N-T shape (every distance) works and the T-N shape (every covariance:
`raft::stats::cov`, `lstsqEig` step 1, `tsvd_fit`) does not. **That single
limit is why PCA and OLS did not move across six benchmark rounds.** The
answer is a transpose followed by an N-T matmul, and it IS wired now:
`core/gemm.mojo::gemm_tn` transposes with `column_stats.mojo::
transpose_kernel` (`linalg.transpose` itself signals on device buffers, see
above) and calls `gemm_nt`. PCA, OLS and truncated SVD all reach it.

`linalg.matmul` with `n = 1` returns zeros for some outputs. RAFT does not
call gemm there either; it calls `gemv`, and so do we
(`core/gemm.mojo::gemv_n`).

## `algorithm.reductions`: shipped, unused, and now mostly NOT wanted

Three kernels in this tree are a dense-axis reduction wearing a hand-written
block reduce, and once the plural module turned up they looked like swaps:

| kernel | what it is | shipped counterpart |
|---|---|---|
| `core/row_norms.mojo::row_norm_kernel` | sum of squares along the feature axis, one output per row | `reduce_sum`, `reduce_dim=1`, an `InputFn` that squares |
| `core/column_stats.mojo::column_mean_kernel` | mean down the ROW axis, one output per column | `reduce_mean`, but the reduced axis is the OUTER one and the scaffolder is row-based |
| `core/column_stats.mojo::xty_kernel` | `A^T b`, a weighted sum down the row axis | `reduce_sum` with an `InputFn` reading both operands |
| `cluster/ported/distance/unfused_distance_nn.mojo::reduce_min_kernel` | argmin over the candidate axis | `reduce_argmin` plus `reduce_min` |

**NOT DONE, and after the rule change at the top of this file, mostly not to
be done.** Their upstreams are `raft::linalg::norm`, `raft::stats::mean`,
`raft::linalg::gemv(trans=true)` and RAFT's KVP argmin. Every one of those is
RAFT's OWN kernel, open source and readable, so the rule makes them PORT
candidates and not substitution candidates. The last row is worse than
neutral: cuVS's dispatch fuses that argmin into the distance kernel
(`fusedDistanceNN`), which is precisely the fusion a device-wide reduction
cannot be part of.

What survives as a real question is narrower: each of these reductions already
calls `block.sum`, which is the collective CUB would use inside the same
kernel, so the remaining difference is launch shape, not arithmetic. Any swap
here also changes the ORDER of a float summation, and `row_norms.mojo` records
that its reduction is in the `IDENTICAL` column's scope, so it moves last bits
and needs a run. Not a doc question.

## Built out of a block primitive because the device-wide one is missing

`cub::DeviceScan::InclusiveSum` has no shipped GPU counterpart. `nn.cumsum`
carries neither `ctx` nor `target` and is CPU-only.

So `cluster/mojo_only/plus_plus.mojo` builds one from
`max.gpu.primitives.block.prefix_sum` in three stages: chunk sums, an
exclusive scan of the chunk totals, then an inclusive scan within each chunk
plus its offset. `check_device_inclusive_scan` verifies it against a host
scan at 20,000 elements and it is EXACT.

That is the pattern for every missing device-wide primitive on this list:
the block-scope one ships, and two extra launches turn it into the
device-scope one. It is worth writing down because the alternative that keeps
suggesting itself, a fixed per-thread slice, silently caps the kernel at
`threads * slice` and truncates without raising. That bug shipped once in
DBSCAN's CSR and was found by audit, not by a test.

## How to add a row

Probe the import, record AVAILABLE or NOT FOUND with the exact path tried,
and say what we do instead. A row asserting an equivalent does not exist is
only worth having if it names what was searched.


---

## The correction this file has to carry, from the boosting side

`VENDOR_LIBS.md` section 3c establishes something that reframes this whole
document: **`split_points.cu` is mostly NOT a CUB call site.** Their
`ReorderOneBitImpl` is an ordinary portable kernel with a rank-based scatter:

    totalOnes    = offsets[size-1] + ((tempKeys[size-1] >> bit) & 1)
    totalZeros   = size - totalOnes
    onesBefore   = offsets[idx]
    zeroesBefore = idx - onesBefore
    offset       = isZero ? zeroesBefore : (totalZeros + onesBefore)

Their scan is over ELEMENTS and the scatter reads a per-element rank. Ours
counts zeros per 256-row chunk, scans the chunk counts, and then scatters.
Same partition, different decomposition, and ours has never been diffed
against a reference.

**So "swap in a vendor primitive" was partly the wrong frame.** The answer
there is the same one the fused-distance round produced: PORT THEIR KERNEL,
which was portable all along. Only the device-wide exclusive scan under it
has no shipped GPU primitive.

That is now twice in one session that the useful move was reading their
kernel rather than shopping for a library call, and both times I had a
plausible reason for not reading it.

---

# APPENDIX A — DOCUMENTATION-DERIVED INVENTORY, 2026-08-19. NOT COMPILE-PROBED.

**READ THIS BEFORE USING ANYTHING BELOW.** Everything above this line was
probed against this toolchain: `AVAILABLE` there means an import compiled.
**Nothing in Appendix A was compiled.** It is read off signatures published at
`https://max.modular.com/api/mojo/...` for the *nightly* docs (pages fetched
2026-08-19, each row carries its URL) and, for `std.*`, off
`https://mojolang.org/docs/std/...`. A signature is evidence that a symbol is
*declared*; it is not evidence that it *runs*. This file already records three
calls that resolved, typechecked, and failed anyway — `matmul` with
`transpose_a`, `matmul` at `n=1`, and `linalg.transpose` on device buffers,
which takes an `Optional[DeviceContext]` and does not dispatch on it.
**Accepting a `DeviceContext` is not the same as dispatching on it**, and that
sentence applies to every "GPU" verdict below that rests on an
`Optional[DeviceContext]` rather than a required one.

Treat Appendix A as a *list of things to probe*, ranked. Promote a row above
this line only after it has compiled AND run AND been checked against a
reference here.

## A0. What was actually searched, so the search is not repeated blindly

- The toolchain packages at
  `~/CascadeProjects/mojotrees/.pixi/envs/default/lib/mojo/*.mojoc` are
  **compressed binaries with no readable symbol table** (`strings` over
  `linalg.mojoc` returns nothing but a leading SHA and entropy). There is no
  Mojo source for the kernel libraries on disk in this env; the only `.mojo`
  files anywhere under the pixi env are 17 files under
  `lib/python3.14/site-packages/max/`, all HAL / distributed-op glue. **The
  docs are the only source available without compiling.**
- 524 package and module index pages were crawled breadth-first from
  `https://max.modular.com/api/mojo/{algorithm, linalg, nn, layout, max,
  structured_kernels, quantization, comm, extensibility, pipeline,
  state_space, kv_cache, builtin_kernels, shmem, profiling_range}` to
  exhaustion (depth 7, 0 new pages at the last level), yielding 3,194 distinct
  symbol paths. Every claim of NOT FOUND below is against that set.
- Packages Modular ships that the docs site does **not** publish an index for,
  and which were therefore NOT searched: `builtin_primitives`, `machine`,
  `matmul_rs`, `mega_ffn`, `msa`, `weights_registry`, `internal_utils`, and the
  vendor shims `_cublas`, `_cudnn`, `_cufft`, `_hal`, `_miopen`, `_rocblas`.
  `std` is documented separately at mojolang.org and was searched only for the
  `gpu` subtree. **Those are a genuine blind spot, not an absence.**

## A1. The GPU-form column

The tell, as this file established the hard way: a real device entry point
carries `ctx: DeviceContext` and/or `target: StringSpan`. A **required** `ctx`
is the strongest signal (the function cannot run without a device). An
**`Optional[DeviceContext]`** is the weakest — that is exactly the shape
`linalg.transpose` has, and it signals on device pointers.

Legend: **GPU** = required `ctx` and/or a `target` parameter that selects a GPU
backend. **GPU?** = only an `Optional[DeviceContext]`, same shape as the known
`linalg.transpose` failure — verify before trusting. **CPU-ONLY** = neither.
**UNKNOWN** = could not be determined from the published page.

| symbol | `ctx` | `target` | verdict | provenance |
|---|---|---|---|---|
| `nn.argsort.argsort` | required | yes (`"cpu"`/`"gpu"`) | **GPU** | [/api/mojo/nn/argsort/argsort](https://max.modular.com/api/mojo/nn/argsort/argsort.md) — a second overload with neither is the documented CPU-only form |
| `nn.toppminp_gpu.run_radix_sort_pairs_gpu` | required | n/a | **GPU** | [/api/mojo/nn/toppminp_gpu/run_radix_sort_pairs_gpu](https://max.modular.com/api/mojo/nn/toppminp_gpu/run_radix_sort_pairs_gpu.md) |
| `nn.topk.top_k` | required | yes | **GPU** | [/api/mojo/nn/topk/top_k](https://max.modular.com/api/mojo/nn/topk/top_k.md) |
| `nn.topk.topk_gpu` | required | n/a | **GPU** | [/api/mojo/nn/topk/topk_gpu](https://max.modular.com/api/mojo/nn/topk/topk_gpu.md) |
| `nn.argmaxmin_gpu.argmaxmin_gpu` (+ `argmax_gpu`, `argmin_gpu`) | required | n/a | **GPU** | [/api/mojo/nn/argmaxmin_gpu/argmaxmin_gpu](https://max.modular.com/api/mojo/nn/argmaxmin_gpu/argmaxmin_gpu.md) |
| `nn.argmaxmin.argmin` / `argmax` | Optional | no | **GPU?** | [/api/mojo/nn/argmaxmin/argmin](https://max.modular.com/api/mojo/nn/argmaxmin/argmin.md) — module blurb says "CPU and GPU"; the `_gpu` module above is the unambiguous route |
| `nn.gather_scatter.gather` | required | yes | **GPU** | [/api/mojo/nn/gather_scatter/gather](https://max.modular.com/api/mojo/nn/gather_scatter/gather.md) |
| `nn.gather_scatter.gather_reduce` | Optional | no | **GPU?** | [/api/mojo/nn/gather_scatter/gather_reduce](https://max.modular.com/api/mojo/nn/gather_scatter/gather_reduce.md) — takes a custom `reduce_fn` |
| `nn.moe.moe_create_indices` | required | yes | **GPU** | [/api/mojo/nn/moe/moe_create_indices](https://max.modular.com/api/mojo/nn/moe/moe_create_indices.md) |
| `nn.softmax.softmax` | Optional | yes (required param) | **GPU** | [/api/mojo/nn/softmax/softmax](https://max.modular.com/api/mojo/nn/softmax/softmax.md) |
| `nn.cumsum.cumsum` | none | none | **CPU-ONLY** | [/api/mojo/nn/cumsum/cumsum](https://max.modular.com/api/mojo/nn/cumsum/cumsum.md) — confirms the correction above |
| `nn.arg_nonzero.arg_nonzero` | none | none | **CPU-ONLY** | [/api/mojo/nn/arg_nonzero/arg_nonzero](https://max.modular.com/api/mojo/nn/arg_nonzero/arg_nonzero.md) |
| `algorithm.reductions.reduce_sum` / `_max` / `_min` / `_mean` / `_product` / `_argmax` / `_argmin` / `_min_and_max` | Optional | **yes, mandatory** | **GPU** | [/api/mojo/algorithm/reductions](https://max.modular.com/api/mojo/algorithm/reductions.md), sig at [.../reduce_sum](https://max.modular.com/api/mojo/algorithm/reductions/reduce_sum.md) |
| `algorithm.rowwise.launch` | Optional (docs: "required for `target="gpu"`") | **yes, mandatory** | **GPU** | [/api/mojo/algorithm/rowwise/launch](https://max.modular.com/api/mojo/algorithm/rowwise/launch.md) |
| `algorithm.gpu.rowwise.launch` / `reduce` / `pjoin` / `once` | GPU backend by construction | n/a | **GPU** | [/api/mojo/algorithm/gpu/rowwise](https://max.modular.com/api/mojo/algorithm/gpu/rowwise.md) |
| `max.algorithm.reduction.sum` / `max` / `min` / `mean` / `product` (the shape-taking overload) | Optional | yes | **GPU** | [/api/mojo/max/algorithm/reduction/sum](https://max.modular.com/api/mojo/max/algorithm/reduction/sum.md) |
| `max.algorithm.reduction.reduce` (arbitrary lambda) | none | none | **CPU-ONLY** | [/api/mojo/max/algorithm/reduction/reduce](https://max.modular.com/api/mojo/max/algorithm/reduction/reduce.md) — `Span`-based |
| `max.algorithm.reduction.map_reduce` | none | none | **CPU-ONLY** | [/api/mojo/max/algorithm/reduction/map_reduce](https://max.modular.com/api/mojo/max/algorithm/reduction/map_reduce.md) |
| `max.algorithm.reduction.cumsum` | none | none | **CPU-ONLY** | [/api/mojo/max/algorithm/reduction/cumsum](https://max.modular.com/api/mojo/max/algorithm/reduction/cumsum.md) — a *second* CPU-only cumsum |
| `max.algorithm.reduction.variance` | none | none | **CPU-ONLY** | [/api/mojo/max/algorithm/reduction/variance](https://max.modular.com/api/mojo/max/algorithm/reduction/variance.md) |
| `max.algorithm.functional.elementwise` | required | yes | **GPU** | [/api/mojo/max/algorithm/functional/elementwise](https://max.modular.com/api/mojo/max/algorithm/functional/elementwise.md) |
| `max.algorithm.backend.gpu.reduction.reduce_launch` (+ `row_reduce`, `twophase_reduce_kernel`, `small_reduce_kernel`, `saturated_reduce_kernel`) | GPU backend by construction | n/a | **GPU** | [/api/mojo/max/algorithm/backend/gpu/reduction](https://max.modular.com/api/mojo/max/algorithm/backend/gpu/reduction.md) |
| `linalg.matmul.matmul` | Optional (docs: "required for GPU targets") | yes | **GPU** | [/api/mojo/linalg/matmul/matmul](https://max.modular.com/api/mojo/linalg/matmul/matmul.md) |
| `linalg.bmm.batched_matmul` | Optional | yes | **GPU** | [/api/mojo/linalg/bmm/batched_matmul](https://max.modular.com/api/mojo/linalg/bmm/batched_matmul.md) |
| `linalg.gemv.gemv_gpu` | required | n/a | **GPU** | [/api/mojo/linalg/gemv/gemv_gpu](https://max.modular.com/api/mojo/linalg/gemv/gemv_gpu.md) — confirms the correction above |
| `linalg.gemv.gemv` | none | none | **CPU-ONLY** | [/api/mojo/linalg/gemv](https://max.modular.com/api/mojo/linalg/gemv.md) — confirms the correction above |
| `linalg.gemv.gemv_split_k` | (kernel, launched by `gemv_gpu`) | n/a | **GPU** | [/api/mojo/linalg/gemv](https://max.modular.com/api/mojo/linalg/gemv.md) |
| `linalg.transpose.transpose` | Optional | no | **GPU? — KNOWN TO SIGNAL**, see the failure table above | [/api/mojo/linalg/transpose/transpose](https://max.modular.com/api/mojo/linalg/transpose/transpose.md) |
| `linalg.transpose.transpose_2d` | Optional | no | **GPU?** | [/api/mojo/linalg/transpose/transpose_2d](https://max.modular.com/api/mojo/linalg/transpose/transpose_2d.md) — same shape as the one that signalled; docs say it "selects a serial tiled or a parallel tiled implementation", both of which read as host paths |
| `linalg.qr_factorization.qr_factorization` / `form_q` / `apply_q` | **none** | **none** | **CPU-ONLY** | [/api/mojo/linalg/qr_factorization/qr_factorization](https://max.modular.com/api/mojo/linalg/qr_factorization/qr_factorization.md), [/form_q](https://max.modular.com/api/mojo/linalg/qr_factorization/form_q.md) — **CORRECTION, see A3** |
| `linalg.matmul.gpu.apple.matmul_kernel.enqueue_apple_matmul` | required | n/a | **GPU, but RAISES on non-M5** | [.../enqueue_apple_matmul](https://max.modular.com/api/mojo/linalg/matmul/gpu/apple/matmul_kernel/enqueue_apple_matmul.md) — "Raises: If the attached GPU is not Apple M5 (`compute_capability != 5`). M1-M4 lack GPU `neural accelerator`." **Unusable on the M4 this repo runs on.** |
| `linalg.matmul.gpu.apple.matmul_8x8.gemm_kernel_apple_8x8` | (kernel body) | n/a | **GPU** | [.../gemm_kernel_apple_8x8](https://max.modular.com/api/mojo/linalg/matmul/gpu/apple/matmul_8x8/gemm_kernel_apple_8x8.md) — the 8x8 simdgroup path, which the `mma_apple` module documents as "all Apple GPU generations (M1-M5)" |
| `max.gpu.compute.arch.mma_apple.apple_mma_load_8x8` / `apple_mma_store_8x8` | (in-kernel) | n/a | **GPU, M1-M5** | [/api/mojo/max/gpu/compute/arch/mma_apple](https://max.modular.com/api/mojo/max/gpu/compute/arch/mma_apple.md) |
| `max.gpu.memory.memory.load[read_only, prefetch_size, cache_policy, eviction_policy]` | (in-kernel) | n/a | **GPU** | [/api/mojo/max/gpu/memory/memory/load](https://max.modular.com/api/mojo/max/gpu/memory/memory/load.md) |
| `std.gpu.intrinsics.ldg` | (in-kernel) | n/a | **GPU** | [mojolang.org/docs/std/gpu/intrinsics](https://mojolang.org/docs/std/gpu/intrinsics.md) |
| `std.gpu.primitives.warp.lane_group_reduce` / `lane_group_sum` / `lane_group_min` / `lane_group_max` `[num_lanes, stride]` | (in-kernel) | n/a | **GPU** | [mojolang.org/docs/std/gpu/primitives/warp/lane_group_min](https://mojolang.org/docs/std/gpu/primitives/warp/lane_group_min.md) |
| `std.gpu.primitives.warp.shuffle_up`, `vote`, `match_any`, `match_all`, `reduce` | (in-kernel) | n/a | **GPU** | [mojolang.org/docs/std/gpu/primitives/warp](https://mojolang.org/docs/std/gpu/primitives/warp.md) |

## A2. What the file was MISSING, ranked by what it unblocks

**1. A GPU key/value radix sort exists, under a name nobody would search for.**
`nn.toppminp_gpu.run_radix_sort_pairs_gpu[dtype, out_idx_type, ascending,
BLOCK_SIZE=256, NUM_BITS_PER_PASS=4](ctx, keys: DoubleBuffer[dtype], key_ids:
DoubleBuffer[out_idx_type], skip_sort, in_shape: [batch, n])`
([docs](https://max.modular.com/api/mojo/nn/toppminp_gpu/run_radix_sort_pairs_gpu.md)).
That is `cub::DeviceRadixSort::SortPairs` — multi-pass, double-buffered, keys
plus payload indices — and because `in_shape` is `[batch_size, vocab_size]` and
it sorts **each batch row independently**, it is also
`cub::DeviceSegmentedRadixSort` for **equal-length segments**. Both of those
sit in the table above as `argsort` (values only, no payload) and
**NOT FOUND** respectively. It lives in the sampling module because that is who
needed it; nothing about it is sampling-specific except `skip_sort`. Caveats to
probe: `DoubleBuffer` is a `nn.toppminp_gpu` struct, so the caller owns the
ping-pong allocation and must read the result out of whichever half the last
pass wrote; `normalize` / `normalize_u32` in the same module are the
float-to-sortable-unsigned bijections it expects for float keys.

**2. There IS a device-wide reduce with a custom operator. The old probe
searched three paths that do not exist.** The `NOT FOUND` line above reads
`algorithm.reduce, algorithm.reduction.sum, nn.reduction.reduce`. The real
shape is a whole library:

- `algorithm.reduce_op` ([docs](https://max.modular.com/api/mojo/algorithm/reduce_op.md))
  defines the `ReduceOp` trait — identity `__init__`, `accumulate[w]` for a
  SIMD tile, `join`, optional `join_parallel[R: Reducer]` — plus shipped
  monoids `ReduceSum`, `ReduceMax`, `ReduceMin`, `ReduceProduct`, `ArgMax`,
  `ArgMin`, `MinMax`, `Welford`, `OnlineLogSumExp`.
- `algorithm.rowwise.launch[body, axis, simd_width, target, num_phases](body,
  shape, ctx)` ([docs](https://max.modular.com/api/mojo/algorithm/rowwise/launch.md))
  runs **your own** `ReduceOp` struct on CPU or GPU from one body, with
  `target` mandatory and `ctx` required when `target="gpu"`.
- `algorithm.gpu.rowwise` ([docs](https://max.modular.com/api/mojo/algorithm/gpu/rowwise.md))
  is the GPU backend, and it ships `WarpReducer` and `BlockReducer` structs and
  a tier dispatch (warp-per-row / block-per-row / two-phase split) — that is
  the same three-tier shape `max.algorithm.backend.gpu.reduction` publishes as
  `small_reduce_kernel` / `reduce_kernel` / `twophase_reduce_kernel`
  ([docs](https://max.modular.com/api/mojo/max/algorithm/backend/gpu/reduction.md)).

`ArgMax` / `ArgMin` are the direct answer to the
`cub::BlockReduce<KeyValuePair>` row above, which this file records as having
no equivalent and being hand-built from `shuffle_xor` plus a 4-way merge. **The
key-carrying reduce monoid ships.** Whether the hand-built one is faster is a
measurement, not a fact — but "none directly" is wrong.

**3. `cub::DeviceSegmentedReduce` is half-found.** For **equal-length**
segments it is `algorithm.reductions.reduce_*[..., target, reduce_dim=k]` or
`max.algorithm.reduction.sum[..., target, reduce_dim=k]`, both with a
`DeviceContext`. For the **ragged** segments `cuda_util/reduce.cu` actually
uses (and that `partitions_reduce` hand-writes), still **NOT FOUND** — nothing
in the 3,194 symbols takes a segment-offsets array. `nn.gather_scatter.gather_reduce`
([docs](https://max.modular.com/api/mojo/nn/gather_scatter/gather_reduce.md))
is the closest: it reduces `input[indices[i,j], k]` over `j` with a caller-supplied
`reduce_fn` and an `Optional[DeviceContext]` — an indexed, fixed-multi-hot
segmented reduce. Ragged-by-offsets is still ours to write.

**4. Apple simdgroup MMA is reachable on M4, and the packaged Apple matmul is not.**
`max.gpu.compute.arch.mma_apple` documents two shapes: 16x16x16 `_mma_apple`
is **M5 only**, and 8x8 `_mma_apple_8x8` is **"all Apple GPU generations
(M1-M5), float-only"**, F16/BF16/F32 in, F32 accumulate
([docs](https://max.modular.com/api/mojo/max/gpu/compute/arch/mma_apple.md)).
`apple_mma_load_8x8(ptr, row_stride, col_stride)`
([docs](https://max.modular.com/api/mojo/max/gpu/compute/arch/mma_apple/apple_mma_load_8x8.md))
takes **both strides**, so a transposed 8x8 fragment is a stride swap and
costs nothing — which is a route to the T-N Gram shape that does not go
through `linalg.transpose` at all. Meanwhile
`linalg.matmul.gpu.apple.matmul_kernel.enqueue_apple_matmul` **raises** unless
`compute_capability == 5`, so the tuned Apple GEMM in the box is M5-only and
this repo's M4 cannot call it. Both halves of that are new information here.

**5. `cub::ThreadLoad` / cache hints are NOT missing.**
`max.gpu.memory.memory.load[width, read_only, prefetch_size, cache_policy:
CacheOperation, eviction_policy: CacheEviction](ptr[, offset])`
([docs](https://max.modular.com/api/mojo/max/gpu/memory/memory/load.md)) is
exactly the modified load, and `std.gpu.intrinsics.ldg` is the non-coherent
read-only load ([docs](https://mojolang.org/docs/std/gpu/intrinsics.md)).
`CacheOperation` is a `std.gpu.intrinsics` struct; `CacheEviction`,
`Consistency` and `Fill` are `max.gpu.memory.memory` structs. The `NOT FOUND`
row for `ThreadLoad`/`ThreadStore` is wrong and is corrected in A3.

**6. A GPU argmin over the inner axis ships.** `nn.argmaxmin_gpu.argmaxmin_gpu[dtype,
output_type, largest](ctx, input, output)`
([docs](https://max.modular.com/api/mojo/nn/argmaxmin_gpu/argmaxmin_gpu.md)),
"single streaming pass" over the inner-most dimension, plus
`algorithm.reductions.reduce_argmin` with `target` and a `DeviceContext`. That
is the k-means assignment step and the reduction half of `unfused_distance_nn`,
both of which this repo hand-writes.

**7. A GPU bucket sort that emits CSR offsets ships.**
`nn.moe.moe_create_indices[..., target](token_expert_order,
expert_start_indices, restore_token_order, expert_ids, expert_usage_stats,
topk_ids, context)`
([docs](https://max.modular.com/api/mojo/nn/moe/moe_create_indices.md)):
"Groups tokens by their assigned expert using a bucket sort algorithm", output
`expert_start_indices` is "CSR-style start offsets", and `restore_token_order`
is the inverse permutation. That is the DBSCAN CSR build and the small-key
partition, with the inverse permutation handed to you. It is written for one
block per bucket with a shared-memory cache that spills at `expected_count`, so
it will be shaped for few-buckets/many-items; probe before assuming it scales
to k-means-sized cluster counts.

**8. `nn.cumsum` supports exclusive and reverse — and is still CPU-only.**
`cumsum[dtype, exclusive: Bool, reverse: Bool, *, axis](output, input)`
([docs](https://max.modular.com/api/mojo/nn/cumsum/cumsum.md)). The old probe
searched for a symbol named `cumsum_exclusive`, which does not exist, and
concluded "cumsum is inclusive only". That conclusion is wrong; the outcome is
unchanged, because there is still no `ctx` and no `target`.

**9. Batched GPU matmul, unlisted.** `linalg.bmm.batched_matmul[transpose_a,
transpose_b, ..., target](c, a, b, context)`
([docs](https://max.modular.com/api/mojo/linalg/bmm/batched_matmul.md)).
`transpose_b` works; `transpose_a` is documented "not yet supported", the same
limit as `matmul`. Relevant wherever a per-cluster or per-segment Gram or
distance block is wanted in one launch.

**10. Warp primitives the headline correction did not list.**
`shuffle_up`, `reduce`, `vote`, `match_any`, `match_all`, and the width-limited
family `lane_group_reduce` / `lane_group_sum` / `lane_group_min` /
`lane_group_max`, each parameterized `[num_lanes: Int, stride: Int = 1]`
([docs](https://mojolang.org/docs/std/gpu/primitives/warp.md)). `match_any`
returns, per lane, the mask of warp lanes whose value bits match it — a
warp-scope equality-class primitive with no entry anywhere in this file.

## A3. Corrections to rows already in this file

**C1. `linalg.qr_factorization` is HOST-ONLY, and the "eigensolver" row and the
"Also confirmed available" list both imply otherwise.** The row for
`cuSOLVER syevj` reads "**NOT FOUND** as a dense eigensolver;
`linalg.qr_factorization` AVAILABLE", which sets QR up as the device-side
consolation prize. It is not one.
`qr_factorization[dtype, element_layout](sigma, A)` takes **no `ctx` and no
`target`**, and neither do `form_q` or `apply_q`
([docs](https://max.modular.com/api/mojo/linalg/qr_factorization/qr_factorization.md),
[form_q](https://max.modular.com/api/mojo/linalg/qr_factorization/form_q.md)).
It is a host LAPACK-style in-place Householder routine over `LayoutTensor`.
This is the *same failure mode* as `linalg.gemv` and `nn.cumsum`: the import
compiles, the symbol is real, and the GPU is not involved. The conclusion —
that `jacobi_eigh_device.mojo` has to exist — is unchanged and now better
supported.

**C2. There is no dense eigensolver, SVD, Cholesky, LU, triangular solve, or
least-squares anywhere in the shipped kernel libraries, and this is now an
exhaustive statement, not an assumption.** Across all 524 crawled module index
pages, a case-insensitive search for `eigen`, `svd`, `singular value`,
`cholesky`, `lu decomp`, `linear system`, `least squares`, `lstsq`,
`triangular solve`, and `covarian` returns **zero pages**. The single hit for
`householder` is `linalg.qr_factorization`, which is C1. The `linalg` package
index lists no factorization module other than `qr_factorization`
([docs](https://max.modular.com/api/mojo/linalg.md)). Subject to the blind
spot in A0 (`matmul_rs`, `machine`, `builtin_primitives`, the vendor shims),
**the hand-written Jacobi is not a failure to shop; there is nothing to buy.**

**C3. "no device-wide reduce with a custom operator" is wrong — the paths were
wrong.** See A2 item 2. The correct paths are `algorithm.reduce_op` +
`algorithm.rowwise.launch` (arbitrary monoid, `target` mandatory,
`DeviceContext` required for GPU) and `algorithm.reductions.*` /
`max.algorithm.reduction.{sum,max,min,mean,product}` (closed set, `target` +
`Optional[DeviceContext]`). What genuinely *is* CPU-only is the
**arbitrary-lambda, flat-buffer** family: `max.algorithm.reduction.reduce`,
`map_reduce`, `variance`, and `cumsum`, all of which take a `Span` and no
context. The old line conflated the two.

**C4. "`ThreadLoad` / `ThreadStore` cache hints -> NOT FOUND" is wrong.** See
A2 item 5. `max.gpu.memory.memory.load` carries `read_only`, `prefetch_size`,
`cache_policy: CacheOperation` and `eviction_policy: CacheEviction`;
`std.gpu.intrinsics.ldg` is the read-only non-coherent load. No `store` with
symmetric cache-policy parameters was found, so the `ThreadStore` half of that
row stands as NOT FOUND.

**C5. "`nn.cumsum.cumsum_exclusive` -> cumsum is inclusive only" is wrong.**
See A2 item 8. `exclusive` and `reverse` are comptime parameters of
`nn.cumsum.cumsum`. The probe failed because it searched for a symbol name
that does not exist rather than reading the signature of the one that does —
which is the same mistake, in miniature, that produced the two failures this
file opens with.

**C6. `lane_id` is not in `std.gpu.primitives.warp`.** The headline correction
lists `lane_id` among the warp module's exports. The published warp module
lists `broadcast`, `lane_group_max`, `lane_group_min`, `lane_group_reduce`,
`lane_group_sum`, `match_all`, `match_any`, `max`, `min`, `prefix_sum`,
`reduce`, `shuffle_down`, `shuffle_idx`, `shuffle_up`, `shuffle_xor`, `sum`,
`vote` — and no `lane_id`
([docs](https://mojolang.org/docs/std/gpu/primitives/warp.md)). `lane_id`,
`warp_id` and `sm_id` are in **`std.gpu.primitives.id`**
([docs](https://mojolang.org/docs/std/gpu/primitives/id.md)). Everything else
in that list checks out.

**C7. The `shuffle_xor` note is right about `shuffle_idx` and incomplete about
the alternative.** `shuffle_idx(val, offset: UInt32)` genuinely has no `width`
parameter ([docs](https://mojolang.org/docs/std/gpu/primitives/warp/shuffle_idx.md)),
so the reasoning that chose `shuffle_xor` stands. But the CUDA `width`-modulo
behaviour the note works around **does** exist in Mojo, as
`lane_group_min[num_lanes, stride]` / `lane_group_max` / `lane_group_sum` /
`lane_group_reduce`, whose docs state that lanes with `lane_id >= num_lanes`
retain their original values
([docs](https://mojolang.org/docs/std/gpu/primitives/warp/lane_group_min.md)).
Those reduce **values only**, so they still do not solve the key-value case —
`algorithm.reduce_op.ArgMin` (A2 item 2) is the candidate for that.

**C8. `matmul`'s `transpose_a` limit is confirmed in the published docs, not
only by the compiler, and it is family-wide.** The parameter's own description
reads "Transpose `a` before the matmul (defaults to `False`); **currently
unsupported**"
([docs](https://max.modular.com/api/mojo/linalg/matmul/matmul.md)), and
`linalg.bmm.batched_matmul` carries the identical "not yet supported"
([docs](https://max.modular.com/api/mojo/linalg/bmm/batched_matmul.md)). So it
is a documented product limit across the whole matmul family, not a local
quirk — and `transpose_b` is supported everywhere, including `gemv_gpu` and
`enqueue_apple_matmul`. Note that a peer resolved the T-N Gram shape while
this appendix was being written, via RAFT's `ColKernelPolicy` and no transpose
at all; **the "Limits of the ones we do use" section above is stale where it
still says "The answer is `linalg.transpose` followed by an N-T matmul, and it
is not yet wired"** — that sentence is contradicted by the RESOLVED note a few
paragraphs earlier in the same file, and whoever owns that section should
delete it. What C8 adds is only that `transpose_a` will not arrive by upgrade
either, so no future round should re-plan around it.

**C9. `linalg.transpose` has siblings, and they have the same suspect shape.**
The module also ships `transpose_2d`, `transpose_3d_swap_inner`,
`transpose_3d_swap_outer`, `transpose_4d_swap_middle`, `transpose_inplace`,
`transpose_strided`, and `transpose_trivial_memcpy`
([docs](https://max.modular.com/api/mojo/linalg/transpose.md)).
`transpose_2d` takes the same `ctx: Optional[DeviceContext] = None`
([docs](https://max.modular.com/api/mojo/linalg/transpose/transpose_2d.md))
and its own description — "selects a serial tiled implementation or a parallel
tiled implementation based on the problem size and available parallelism" —
describes host paths only. Given that `linalg.transpose` signalled inside
`_copy_with_strides`, **assume the whole module is host-side until one of them
is proven otherwise**, and note that `transpose_strided` is named after the
very function that signalled.

## A4. Still NOT FOUND after an exhaustive symbol sweep

Searched over all 3,194 symbol paths and all 524 module index page bodies from
the crawl in A0.

    device-wide (grid-level) scan of any kind
        -> the only prefix-sum symbols in the entire corpus are BLOCK scope:
           max.gpu.primitives.block.prefix_sum, std.gpu.primitives.warp.prefix_sum,
           shmem.ep_comm.block_prefix_sum, and one tile-scheduler internal.
           A full-text search for "prefix sum" / "exclusive sum" / "cumulative
           sum" across every crawled page returns six files, all of those.
           The three-stage build in cluster/mojo_only/plus_plus.mojo remains
           the answer and is now known to be the ONLY answer.
    cub::DeviceSegmentedReduce over RAGGED segments (an offsets array)
        -> nothing takes segment offsets. Equal-length is covered (A2 item 3).
    cub::DeviceSegmentedRadixSort over RAGGED segments
        -> run_radix_sort_pairs_gpu is equal-length batches only.
    cub::DeviceSelect / stream compaction on device
        -> nn.arg_nonzero.arg_nonzero is the shape, and it is CPU-ONLY
           (no ctx, no target).
    any dense eigensolver / SVD / Cholesky / LU / triangular solve / lstsq
        -> see C2. Zero hits, exhaustively.
    cub::BlockRadixSort
        -> unchanged, NOT FOUND. nn.topk_bitonic.persistent_topk_block is a
           block-wide bitonic top-k, not a general block sort.
    ThreadStore with cache-policy parameters
        -> the load half exists (C4); no symmetric store was found.
    LoadDirectWarpStriped
        -> unchanged, NOT FOUND.

## A5. What could NOT be determined, and why

- **Whether any `Optional[DeviceContext]` row actually dispatches on it.**
  This is the central unknown and it is not answerable from documentation. It
  covers `linalg.transpose*` (one member already known to fail),
  `nn.argmaxmin.argmin`/`argmax`, `nn.gather_scatter.gather_reduce`,
  `linalg.matmul.matmul`, `linalg.bmm.batched_matmul`, `algorithm.rowwise.launch`
  and the `algorithm.reductions.*` family. For `matmul` we already know it
  dispatches, because this repo has it WIRED; for the rest, unknown.
- **Whether `run_radix_sort_pairs_gpu` is usable outside sampling.** `skip_sort`
  is a per-batch `Pointer[Scalar[bool]]` with no documented "always sort"
  value, `DoubleBuffer` has no published field list on its page, and there is
  no documented way to learn which half of the double buffer holds the final
  result. Probing is the only way.
- **Whether `moe_create_indices` scales past a few buckets.** `expected_count`
  is documented as a shared-memory cache size that spills to global memory, and
  it launches one block per expert; the performance shape at k-means-sized
  cluster counts is unknown.
- **Whether the 8x8 Apple MMA path is reachable from a normal kernel on M4.**
  `mma_apple` documents M1-M5 support for the 8x8 shape, but
  `gemm_kernel_apple_8x8` is described in its own docs as a "launchable wrapper
  ... (bench/test)", and the underlying `_mma_apple_8x8` is underscore-prefixed
  and has no public page. Whether the public `apple_mma_load_8x8` /
  `apple_mma_store_8x8` pair plus an mma call is a supported combination on M4
  is unknown.
- **Everything in `builtin_primitives`, `machine`, `matmul_rs`, `mega_ffn`,
  `msa`, `weights_registry`, `internal_utils`, `_cublas`, `_cudnn`, `_cufft`,
  `_hal`, `_miopen`, `_rocblas`.** These ship in the toolchain and the docs
  site publishes no index for them (A0). `matmul_rs` in particular is named
  like a matmul backend and was not searched at all. **A NOT FOUND anywhere in
  this file is scoped to the packages that are documented.**
- **Whether these nightly docs match the pinned toolchain.** The crawl targets
  the docs site's `nightly` version. The packages in this env are dated
  2026-08-09. Symbols may have been added or renamed since. Any row here that
  fails to import may be a version skew rather than a documentation error, and
  the `stable` docs should be checked before concluding the symbol is fictional.

## A6. The method note this appendix exists to record

Both prior errors — `nn.cumsum` and `linalg.gemv` — came from the same move:
**probing whether an import resolves, and stopping there.** The import
resolving tells you a name exists in a package. It tells you nothing about
where the code runs. The cheap discriminator is the signature, and it is
published: **look for a required `ctx: DeviceContext`, then a `target:
StringSpan`, then an `Optional[DeviceContext]` (weak), then neither
(CPU-only)**. Applying that discriminator once, on paper, would have caught
both. It also caught `qr_factorization`, `arg_nonzero`,
`max.algorithm.reduction.cumsum` and `max.algorithm.reduction.reduce` here,
before any of them cost anything.

The second move that paid: **when a probe returns NOT FOUND, read the package
index rather than trying another guessed symbol name.** Three of the six
corrections above (C3, C5, and the segmented-reduce half of A2) are cases where
the capability was present and the guessed path was not.


---

# Appendix A, COMPILE-PROBED. 2026-08-19.

Appendix A was crawled from documentation and flagged as never compiled.
**That flag was right to insist on and here is the follow-through.** Every
row below was put through `mojo build` in this toolchain. This is what the
rest of the file means by AVAILABLE.

    AVAILABLE  nn.toppminp_gpu.run_radix_sort_pairs_gpu
    AVAILABLE  algorithm.reduce_op.ReduceOp
    AVAILABLE  algorithm.reduce_op.ArgMin
    AVAILABLE  algorithm.reduce_op.ArgMax
    AVAILABLE  algorithm.rowwise.launch
    AVAILABLE  nn.argmaxmin_gpu.argmaxmin_gpu
    AVAILABLE  linalg.bmm.batched_matmul
    AVAILABLE  std.gpu.primitives.id.lane_id
    AVAILABLE  max.gpu.memory.memory.load
    AVAILABLE  nn.moe.moe_create_indices
    MISSING    algorithm.reductions.reduce_sum

So the sweep was right on ten of eleven, and the eleventh is a wrong symbol
name rather than a wrong package.

## What this changes, in order

**`algorithm.reduce_op.ArgMin` and `ArgMax` exist and compile.** This file
has said in two places that there is "no block reduce over a custom operator"
and "none directly" for `cub::BlockReduce<KeyValuePair>`. Both statements
were produced by searching three paths that do not exist. The real thing is a
monoid trait plus `algorithm.rowwise.launch`, and ArgMin is exactly the
key-carrying reduction k-means and k-NN need.

That does NOT automatically mean we should switch. `cluster/ported/distance/`
now runs a warp-shuffle butterfly, which is what CUB's own default
(`BLOCK_REDUCE_WARP_REDUCTIONS`) does underneath, so we are already close to
their implementation rather than merely their call. Whether `rowwise.launch`
beats it is a MEASUREMENT nobody has taken, and taking it needs the
benchmark, which is on hold.

**A GPU key-value radix sort exists**, `run_radix_sort_pairs_gpu`. This file
lists `cub::DeviceRadixSort::SortPairs` and `cub::DeviceSegmentedRadixSort`
as NOT FOUND. Both rows are wrong. It sorts each batch row independently, so
for equal-length segments it is the segmented variant too. Caveats the sweep
raised and probing cannot settle: `DoubleBuffer` has no published field list,
there is no documented way to learn which half holds the result, and
`skip_sort` has no documented always-sort value.

**`linalg.qr_factorization` is HOST-ONLY.** Third instance of the same trap
after `nn.cumsum` and `linalg.gemv`. This file had it as the device-side
consolation prize for the missing eigensolver; it is not one.

**No eigensolver, SVD, Cholesky, LU, triangular solve or lstsq exists
anywhere**, now established by exhaustion over 524 documented pages rather
than assumed. `jacobi_eigh_device.mojo` is not a failure to shop.

**`lane_id` is in `std.gpu.primitives.id`, not `.warp`.** Corrected above.

## The remaining unknown, and it is the important one

**Whether a signature carrying `Optional[DeviceContext]` actually DISPATCHES
on it.** That is unanswerable from documentation and it is not answerable by
compiling either, because it is a RUNTIME property. `linalg.transpose` has
exactly that shape, compiles, and signals on device pointers.

So the file's three tiers are now four:

    NOT FOUND        the symbol does not exist
    CPU-ONLY         no ctx and no target in the signature
    COMPILES         the import and the call typecheck
    RUNS ON DEVICE   only provable by executing it

Nothing may be called a substitution until it reaches the fourth tier, and
`linalg.transpose` is the standing reminder of why.

## Not searched at all

`builtin_primitives`, `machine`, `matmul_rs`, `mega_ffn`, `msa`,
`weights_registry`, `internal_utils`, and the `_cublas` / `_cudnn` / `_cufft`
/ `_hal` / `_miopen` / `_rocblas` shims ship in the toolchain and publish no
documentation index. `matmul_rs` is named like a matmul backend. Every NOT
FOUND in this file is scoped to documented packages only.
