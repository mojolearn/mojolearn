# Upstream design audit

Read-only audit, 2026-08-24. No code was run, built or tested. Nothing in the
repository was edited except this file.

The question asked was Andrew's, in his words: "try to improve paths by seeing
what cuml or pytorch or some other library does in cuda and mirror it in mojo."
This file is the answer as a prioritized findings list. It proposes no
optimization and changes no default. Optimization stays lane-owned and
serialized, so what follows is material for a lane owner, not a patch.

## Scope, stated honestly

**Lanes audited with depth** (upstream file opened, ours read against it):

| lane | what was read |
|---|---|
| `kde/` | full lane against cuVS `v26.08.00` `distance/kde.cu` and cuML `v26.08.00` `kde/kde.cu` |
| `svm/` | dispatch and cache scope against cuML `v26.08.00` `svm/kernelcache.cuh`, `svc.cu`, `svm_base.pyx` |
| `cluster/` | `kmeans_fit_main` end to end against cuVS `94c2819` and the full `94c2819` to `6ba2ce2` redesign |
| `neighbors/` | `fused_l2_knn.mojo` deviation blocks against cuVS `fused_l2_knn.cuh` on both pins |
| `dbscan/` | `adjgraph/algo.mojo` against cuML `adjgraph/algo.cuh` and RAFT `adj_to_csr.cuh` |
| `hierarchy/` | `sparse/op/sort.mojo` against RAFT `sparse/op/detail/sort.h` |
| cross-cutting | all 18 `DERIVATION_MAP.tsv` files, `PORTING_RULES.md`, `VENDOR_LIBS.md`, `IDENTITY_PATHS.md`, the upstream checkout revisions |

**Lanes NOT audited.** `gbdt/` (see finding 6, its upstream is absent from this
machine), `ensemble/`, `extratrees/`, `isolation_forest/`, `decomposition/`,
`glm/`, `solver/`, `spectral/`, `metrics/`, `gemm/`, `arima/`, `tsa/`,
`holtwinters/`, `transformer/`, `mamba/`, `bindings/`, `python/`, and the
internals of `core/` beyond `row_norms.mojo` and `kernel_matrix.mojo`.

**Not available.** There is no PyTorch checkout under
`/Users/andrewhendel/CascadeProjects/upstream/`. No finding below cites
PyTorch, and none should until a checkout exists.

**The checkouts, verified by `git log -1` on 2026-08-24.** Every finding names
which one it read.

| checkout | revision | what it is |
|---|---|---|
| `upstream/cuml` | `00094f7`, `branch-25.08` | cuML 25.08 |
| `upstream/cuml-v26.08.00` | `265b9da`, tag `v26.08.00` | cuML 26.08 |
| `upstream/cuvs` | `94c2819`, `branch-25.08` | cuVS 25.08 |
| `upstream/cuvs-v26.08.00` | `6ba2ce2`, tag `v26.08.00` | cuVS 26.08 |
| `upstream/raft` | `661a3b8`, `branch-25.08` | RAFT 25.08 |
| `upstream/raft-v26.08.00` | `ebf9268`, tag `v26.08.00` | RAFT 26.08 |

## The current position on vendor libraries, restated accurately

`VENDOR_LIBS.md:3-30` carries a rule change banner. The old rule, "where the
incumbent calls a vendor primitive, call OURS", is **deleted**. The rule in
force is **FOLLOW THEIR DISPATCH**, and it has exactly one narrow exception:
where the path their dispatch actually takes calls a **closed** library
(cuBLAS, cuSOLVER), call the MAX equivalent, because there is nothing to port.
**CUB and Thrust are open and are port candidates, not substitution
candidates.** Block and warp collectives are a third case the banner does not
reach (`VENDOR_LIBS.md:44-51`); they run inside one kernel, cannot break a
fusion, and are free to use.

Findings 1 and 2 below are both applications of that rule, and both are the
same failure shape the rule was written for.

## Prioritized findings

| # | finding | our file and line | their file and line, checkout named | class | identity impact | size |
|---|---|---|---|---|---|---|
| 1 | KDE materializes two `n_query * n_train` matrices; cuVS 26.08 ships a fused tiled kernel that writes neither. The lane's own docstring states the fused entry does not exist. It does. | `kde/derived/neighbors/kernel_density.mojo:884-935`; false claim at `kde/derived/kde/kde.mojo:9-11` | `cuvs-v26.08.00/cpp/src/distance/kde.cu:334-473` (kernel), `:530-657` (launcher) | 1 (missed design) + 3 (false doc) | identity-safe on the single-pass arm; multi-pass needs one PIN | L |
| 2 | SVM runs cuML's `cache_size == 0` path. Their default is 200 MiB in C++ and 1024 MiB in Python, so the kernel row cache is on for every real call. | `svm/derived/svm/kernelcache.mojo:1-22`, refusal at `svm/derived/svm/svm_parameter.mojo:145` | `cuml-v26.08.00/cpp/src/svm/kernelcache.cuh:381`, `cpp/include/cuml/svm/svc.hpp:227`, `python/cuml/cuml/svm/svm_base.pyx:274` | 1 (dispatch, rule 0b-i) | identity-safe; the lane already wrote the determinism argument | L |
| 3 | Fused k-NN is single-buffered. RAFT/cuVS `Contractions_NT` is double-buffered, and the matrix accessor that would grant the second page already exists and is unused by this kernel. | `neighbors/derived/neighbors/detail/fused_l2_knn.mojo:153-161` | `cuvs/cpp/src/neighbors/detail/fused_l2_knn.cuh` via `raft/.../Contractions.cuh` `SmemSize = 2 * SmemPage`, both pins | 1 (missed design, open item) | identity-safe, no summation order changes | M |
| 4 | DBSCAN's CSR compaction uses one atomic per set bit. RAFT uses a warp-aggregated atomic and prices it at up to 32x less atomic traffic. The stated blocker ("no Mojo counterpart") appears expired. | `dbscan/derived/dbscan/adjgraph/algo.mojo:334-348` | `raft-v26.08.00/cpp/include/raft/sparse/convert/detail/adj_to_csr.cuh:77-99`; `atomicIncWarp` in `raft/util/device_atomics.cuh` | 2 (hand-written weaker) | identity-neutral; the output is unordered on both sides already | S |
| 5 | The MST edge sort runs on the host, with a full round trip of `n-1` edges. Upstream sorts on the device, and this repo already owns a deterministic device radix sort. | `hierarchy/derived/sparse/op/sort.mojo:121-160` | `raft-v26.08.00/cpp/include/raft/sparse/op/detail/sort.h:94-102` | 2 (hand-written weaker) | identity-safe if the packed total-order key is preserved | M |
| 6 | `PORTING_RULES.md` section 0a, the binding "where their source is" table, is stale. It names only the 25.08 checkouts, roughly half the lanes are pinned to `v26.08.00`, and the CatBoost checkout it names is absent from this machine. | `PORTING_RULES.md:9-16` | n/a (repository documentation) | 3 (undocumented divergence) | none | S |
| 7 | `cluster/derived/cluster/detail/kmeans.mojo:35` asserts `check_convergence` "does not exist anywhere in cuVS, cuML or RAFT". It exists in both cuML pins and, in cuVS 26.08, does precisely the lagged device-side test this lane deleted as an invention. | `cluster/derived/cluster/detail/kmeans.mojo:29-38` | `cuvs-v26.08.00/cpp/src/cluster/detail/kmeans_common.cuh:637-661`, used at `kmeans.cuh` loop top | 3 (false doc) + 1 (upstream adopted it) | the lagged flag is iteration-count equivalent, so identity-safe | S |
| 8 | The SVM row norm is a per-row serial chain with an uncoalesced access pattern. `core/row_norms.mojo` is the pinned block fold for the same operation, and the "no fold shape to pin" reason given is contradicted by `K_LIB_ROW_NORM`. | `svm/derived/distance/kernel_matrices.mojo:77-92`; claim in `svm/NOT_IMPLEMENTED.tsv` | `raft-v26.08.00` `linalg::norm<L2Norm, Apply::ALONG_ROWS>` via `coalescedReduction` | 2 (hand-written weaker) | changes the pinned bits, so the SVM card needs re-baselining | S |
| 9 | cuVS 26.08 rewrote the Lloyd loop. Four separable pieces our 25.08-pinned port predates, of which two are cheap and one is a behavior fix. | `cluster/derived/cluster/detail/kmeans.mojo:1114-1300` | `cuvs-v26.08.00/cpp/src/cluster/detail/kmeans.cuh:678-980`, `kmeans_common.cuh:500-745` | 1 (missed design, mixed value) | mixed, per piece; see the section | M |
| 10 | Confirmation, not a defect. DEVIATION 602 (the cosine kernel norm is wrong upstream for even `d`) is the identical fix cuVS 26.08 shipped, by the identical recurrence. | `kde/derived/neighbors/kernel_density.mojo:398-420` | `cuvs-v26.08.00/cpp/src/distance/kde.cu:295-317` | validation | none | none |

## Finding 1. KDE materializes what cuVS fuses, and the lane believes the fused kernel does not exist

**Their code, cuVS `v26.08.00` (`6ba2ce2`).** `cpp/src/distance/kde.cu` is 685
lines and contains a complete fused implementation. cuML 26.08's
`cpp/src/kde/kde.cu` is pure delegation to it (`kde.cu:43-52`).

`kde_tiled_kernel` (`cuvs-v26.08.00/cpp/src/distance/kde.cu:334-440`) is one
thread per query point. Train vectors are cooperatively staged in shared memory
as `[feat_tile][CELL_TILE]`. Each thread holds `CELL_TILE * N_ACC` distance
accumulators **in registers**, tiles over features, finalizes each cell's
distance, applies the log kernel, and folds it into a **streaming logsumexp**
in the same registers:

    if (log_k > running_max) { running_sum = running_sum * exp(running_max - log_k) + 1; running_max = log_k; }
    else                     { running_sum += exp(log_k - running_max); }

No distance and no log-kernel value is ever written to global memory. The
kernel writes exactly `n_query` floats (single pass) or `n_query *
n_train_blocks` pairs (multi pass, `:507-517` of the reduce kernel merges
them).

Two more pieces of their design are worth taking with it. The metric is
decomposed into `DistOp<T, Metric>` with `init` / `accumulate` / `finalize` and
a compile-time `N_ACC` (`:40-207`), which is what lets a register-resident tile
serve thirteen metrics; and `CELL_TILE` is chosen at compile time from register
pressure (`:583-591`), `feat_tile = min(64, d)` (`:568`).

**Our code.** `kde/derived/neighbors/kernel_density.mojo:884-935` allocates
`dist` and `logk`, both `n_query * n_train` float32, runs
`pairwise_distance` into the first, an elementwise `log_kernel_matrix_kernel`
into the second, and then a per-row serial `logsumexp_kernel`. That is 16 bytes
of traffic per (query, train) pair against cuVS's zero, and 8 bytes of
allocation per pair. At `n_query = n_train = 100,000` the two buffers are 80 GB,
so the current path does not merely run slowly at the shipped sizes required by
`large-data-runs-default`, it cannot allocate.

**The false sentence.** `kde/derived/kde/kde.mojo:9-11` reads

> That cuVS entry is in cuVS 26.08, which this tree's pinned checkout (25.08,
> `94c2819`) does not have, so the delegate here is
> `kde/derived/neighbors/kernel_density.mojo` -- the 25.08 Python-layer algorithm
> the fused kernel reproduces.

`cuvs-v26.08.00/cpp/src/distance/kde.cu` and
`cuvs-v26.08.00/cpp/include/cuvs/distance/kde.hpp` both exist on this machine
and have since the tag was cloned. The lane read the `cuvs` checkout, where KDE
genuinely is absent, while the same lane's own `DERIVATION_MAP.tsv` row cites cuML
at `265b9da` (26.08). This is the exact trap the pinned-tree rule exists to
prevent, and it is the third recorded instance. Per rule 1, the sentence should
be deleted, not annotated.

**What it is worth.** This is the same shape as the measured k-NN case that
`PORTING_RULES.md:0b-i` records: we ported a reference implementation that
materializes, while their dispatch takes a fused kernel that materializes
nothing. That case was 306 ms against a 13 ms compute floor. The KDE case is
larger, because it also removes an `O(n_query * n_train)` allocation.

**Identity.** The single-pass arm (`n_train_blocks == 1`) is identity-friendly.
The feature accumulation is ascending, and the streaming logsumexp folds train
points in ascending `j` within one thread, so the fold order is a pure function
of `(n_train, CELL_TILE, feat_tile)`, all of which are host or compile-time
constants. The multi-pass arm derives `n_train_blocks` from
`cudaDevAttrMultiProcessorCount * 4` (`:600-604`), which is a core count, and
its `kde_reduce_kernel` merge is therefore a device-dependent summation
partition. That is `IDENTITY_PATHS` row 3's class exactly. The clean move is
**PIN `n_train_blocks = 1` under `IDENTICAL`** (identical to the `grid_x = 1`
pin row 23 already applies to fused k-NN) and allow multi-pass under `FAST`.

Bits will move relative to the current KDE card, because a streaming logsumexp
is a different arithmetic from a two-pass max-then-sum. That is a
re-baselining, not a defect.

**Unexamined.** I did not read cuVS's `kde` tests
(`cuvs-v26.08.00/cpp/tests/distance/kde.cu`) or their golden generator, which
would be the natural oracle for a port.

## Finding 2. SVM runs the non-default half of cuML's cache dispatch

**Their dispatch.** `SvmParameter::cache_size` is in MiB. The C++ `SVC`
constructor defaults it to 200 (`cuml-v26.08.00/cpp/include/cuml/svm/svc.hpp:227`),
`KernelCache`'s own default is 200
(`cuml-v26.08.00/cpp/src/svm/kernelcache.cuh:381`), and the Python surface every
real user reaches defaults it to 1024
(`cuml-v26.08.00/python/cuml/cuml/svm/svm_base.pyx:274`). There is no default
path with the cache off.

With the cache on, `InitWorkingSet` (`kernelcache.cuh:485-511`) calls
`PreparePartitionedIdxOrder`, which partitions the working set into cached and
uncached keys. `getNextBatchKernel` (`:689-700`) then fills the cached columns
from `batch_cache.GetVecs` and computes only the uncached remainder. Since
cuML's FIFO working-set selection retains a substantial fraction of the previous
working set between outer iterations, the kernel-tile GEMM, which is the
dominant cost of SMO, shrinks accordingly.

**Our code.** `svm/derived/svm/kernelcache.mojo:1-22` states plainly that the
port takes the `cache_size == 0` path, and
`svm/derived/svm/svm_parameter.mojo:145` refuses `cache_size != 0` by name. This
is documented, in `NOT_IMPLEMENTED.tsv` and in `svm/README.md:184-205`, and it is not a
silent substitution. It is still a rule 0b-i miss, because the ported path is
not the path their dispatch takes.

**Identity.** The lane has already done the hard part of this argument
(`svm/README.md:191-197`). Eviction order and the working-set permutation are
pure functions of the working-set sequence, `cache_size` and `n_rows`; every
input is a counter, a modulus, or a stable sort, and none is a block count, a
clock, or a vendor. Under DEVIATIONS 633 and 634 neither the permutation nor
which columns come from the cache can reach a bit, because a cached row equals
a recomputed row bit for bit when every cell is a pure function of its pair. So
the port is bit-inert on landing, and the gate the lane already names (cache
size 0 against 200 giving the same bytes) is the right check.

**Size.** Large. `raft::cache::Cache` is a 32-way set-associative LRU with
`get_cache_idx`, `assign_cache_idx`, `rank_set_entries` (a `cub::BlockRadixSort`
over time stamps), `get_vecs` and `store_vecs`, plus
`PreparePartitionedIdxOrder` and `GetCacheIdxPartitionedStable` on top. CUB is
open, so all of it is a port candidate rather than a substitution candidate.

## Finding 3. Fused k-NN is single-buffered where their contraction is double-buffered

**Their code.** `Contractions_NT`, which `fusedL2kNN` inherits its staging from,
sets `Policy::SmemSize = 2 * SmemPage` because the k-tile pipeline ping-pongs
between two shared pages while the next tile's global loads are in flight. Same
on both cuVS pins; `fused_l2_knn.cuh` is unchanged between `94c2819` and
`6ba2ce2` apart from an include path and a namespace.

**Our code.** `neighbors/derived/neighbors/detail/fused_l2_knn.mojo:153-161`
records DEVIATION BLOCK 3. Two pages at Policy2x8 is 36,992 bytes against
Metal's 32 KB threadgroup limit, so the port is single-buffered. The block
already says what should happen next:

> the ceiling that forces this is Apple's alone and the double buffer would fit
> on NVIDIA's 48 KB and AMD's 64 KB. That is a `lib_smem_pages_for` row, not a
> constant, and it is left OPEN.

**Why this is the most actionable of the three big ones.** The accessor exists
already. `original/kernel_matrix.mojo:1803-1813` defines `lib_smem_pages`,
returning 2 when `2 * page_bytes <= column_shared_limit(column)` and 1
otherwise, and `lib_smem_pages_for[column, page_bytes]` at `:1995` is its
comptime form. At an 18,496 byte page it already yields 1 on Apple, 2 on NVIDIA
(48 KB) and 2 on AMD (64 KB). `probe_main.mojo:427` and `matrix_main.mojo:76-79`
consume it. The fused k-NN kernel does not. So this is not a new mechanism, it
is wiring one kernel to a matrix row that is already computed correctly.

**Identity.** Safe. Double buffering changes when a k-tile is staged, not the
order in which its contributions accumulate. The `identical_mul_add` pinning and
the `grid_x = 1` pin are untouched.

**Size.** Medium. The ping-pong staging is real kernel surgery, and the vendor
matrix means Apple and the other two columns compile different staging, which
is the intended shape (`always-gpu-agnostic`, a kernel matrix row rather than an
inline vendor branch).

## Finding 4. DBSCAN's CSR compaction takes one atomic per set bit

**Their code.** `raft-v26.08.00/cpp/include/raft/sparse/convert/detail/adj_to_csr.cuh:77-99`.
The row is scanned in 16-byte `TxN_t<bool,16>` chunks with an alignment peel and
a remainder, and every hit takes its output slot from `atomicIncWarp(row_count)`.
`atomicIncWarp` (`raft/util/device_atomics.cuh`) forms
`cg::coalesced_threads()`, has lane 0 issue one `atomicAdd` of the group size,
and broadcasts the base with `g.shfl(warp_res, 0)` plus `g.thread_rank()`. Their
own comment prices it: "It can reduce the amount of atomic memory traffic by a
factor of 32."

**Our code.** `dbscan/derived/dbscan/adjgraph/algo.mojo:334-348`. The 16-bool
chunked load **is** ported. The atomic is not aggregated, one `Atomic.fetch_add`
per hit, and the deviation block calls it "the largest remaining gap in this
file". The reason given is that
`cooperative_groups::coalesced_threads()` has no Mojo counterpart, "not
something `std.gpu.primitives` exposes".

**Why that reason looks expired.** `std.gpu.primitives.warp` exposes a ballot.
`upstream/modular/max/kernels/src/nn/attention/gpu/amd_structured/mha_softmax.mojo:37`
imports it as

    from std.gpu.primitives.warp import vote as warp_vote

and uses it at `:649` as `warp_vote[DType.uint64](lane_ok)`. With a ballot plus
the `shuffle_idx` this repository already probed sound, `atomicIncWarp` is
expressible: ballot the predicate, take the popcount of the mask below your
lane as your rank, have the lowest set lane issue one `atomicAdd` of the full
popcount, broadcast the base.

`VENDOR_LIBS.md:113-116` states the rule that applies here, after two false
negatives of exactly this kind: "a NOT FOUND is only worth writing down if it
names the exact paths tried, and **it expires**. Re-probe before hand-writing
anything on the strength of one."

**What I did not verify, and it matters.** I read `warp.vote` in an
AMD-specific MAX kernel. I did not confirm it compiles on the Metal target or
what it returns for an inactive lane. Per rule 7 that is a probe, not an
inference, and the probe comes before the port.

**Identity.** Neutral. The output column order inside a row is already unordered
on both sides, the deviation block records that nothing reads the CSR in order,
and `weak_cc` converges to the same fixed point on any edge order.

**Size.** Small, once the probe lands. One kernel, one helper.

## Finding 5. The MST edge sort round-trips to the host

**Their code.** `raft-v26.08.00/cpp/include/raft/sparse/op/detail/sort.h:94-102`,
a device `thrust::sort_by_key` over `nnz` weights with the `(row, col)` pair as
payload.

**Our code.** `hierarchy/derived/sparse/op/sort.mojo:121-160` copies all three
device arrays to host buffers, sorts on the host by a packed
`(weight_order_key, min(u,v), max(u,v))` key, and writes back.

The *key* is DEVIATION 621 and is correct and well argued. `thrust::sort_by_key`
is not stable, equal-weight MST edges therefore come out in an
implementation-chosen order, and `build_dendrogram_host` walks the list in
order, so a swap across the `n_clusters` cut is a different partition. Making the
key a total order over the edge set is the right fix and the sabotage that
proves it is in the lane. **None of that requires the sort to be on the host.**

`nnz` is `n - 1`, one edge per point. At a million points that is a million-element
host merge sort plus a six-array round trip, on the critical path of a fit whose
every other stage is on the device.

**What is available in-repo.** `gbdt/gpu_util/kernel/radix_sort.mojo` is a
stable LSD radix sort built from CatBoost's own `ReorderOneBit`, and
`svm/original/device_select.mojo` already establishes the precedent of a lane
importing a `gbdt/gpu_util/` primitive. Sorting a packed UInt64 total-order key
with a stable radix preserves DEVIATION 621 exactly, because the key is a pure
function of the edge and no two distinct edges compare equal, so stability is
moot and the result is the same permutation the host merge sort produces.

**Identity.** Safe, provided the packed key is preserved bit for bit. The
existing `LINK_SAB_SORT_WEIGHT_ONLY` sabotage is the right gate to keep.

**Size.** Medium. A 64-bit key radix sort with a payload, plus retiring the host
path behind a check that the two agree.

## Finding 6. The binding "where their source is" table is stale, and one checkout is missing

**The rule.** `PORTING_RULES.md:9-16` is the table every other rule points at,
under a heading that says "Every rule below says 'read their file'. These are
the files. Clone them if the directory is missing; a session without them is a
session guessing."

**What it says.** CatBoost at `/private/tmp/catboost-src` `54a8143a`; cuVS at
`upstream/cuvs` `94c2819`; cuML at `upstream/cuml` `00094f7`; RAFT at
`upstream/raft` `661a3b8`. It names no `v26.08.00` tree at all.

**What the repository actually does.** Counting revision and version strings in
the 18 `DERIVATION_MAP.tsv` files:

| pinned to 25.08 (`00094f7` / `94c2819` / `661a3b8`) | pinned to v26.08.00 (`265b9da` / `6ba2ce2` / `ebf9268`) |
|---|---|
| `cluster`, `dbscan`, `decomposition`, `extratrees`, `glm`, `neighbors`, `gemm` | `arima`, `holtwinters`, `isolation_forest`, `metrics`, `solver`, `spectral`, `svm`, `tsa` |
| | mixed: `hierarchy`, `kde` |

Roughly half the repository reads a tree the binding table does not mention.
Finding 1 is the cost of that gap being implicit rather than written down, and
the auto-memory note "mojolearn's pinned upstream trees" records two earlier
lanes that audited the wrong tree and produced confident nonsense. The table
should carry both pins with an explicit per-section assignment, and any lane
whose upstream gained an algorithm after 25.08 should be listed as
`v26.08.00`-only.

**And the CatBoost checkout is absent.** `/private/tmp/catboost-src` does not
exist on this machine, and there is no `catboost` directory under
`upstream/`. `gbdt/` is the largest ported tree in the repository, 78 rows in
the root `DERIVATION_MAP.tsv`, and by the repository's own rule it currently cannot
be audited or extended against its upstream at all. A `/private/tmp` path also
does not survive a reboot, which is probably how it went. Re-clone it under
`upstream/` alongside the others and update the table.

## Finding 7. `check_convergence` does exist, and cuVS 26.08 adopted the design this lane deleted

**The claim.** `cluster/derived/cluster/detail/kmeans.mojo:29-38`:

> An earlier version of this port ran the test in a one-thread kernel, read the
> flag one iteration late, and attributed both to cuVS. Neither is in their
> source: `check_convergence` does not exist anywhere in cuVS, cuML or RAFT.

**What a grep of the checkouts finds.** `check_convergence` exists in
`cuml/cpp/src/glm/qn/qn_util.cuh`, `glm_base.cuh`, `qn_solvers.cuh`,
`tsne/exact_tsne.cuh` and `tsne/exact_kernels.cuh` on **both** cuML pins, and
in `cuvs-v26.08.00/cpp/src/cluster/detail/kmeans_common.cuh:637`,
`kmeans.cuh` and `kmeans_mg.cuh`.

The cuVS one is not a coincidence of naming. It is a `__device__` function that
takes the clustering cost, the prior cost, the squared norm error and `tol`,
applies both stopping criteria, advances the prior cost and writes a flag
(`kmeans_common.cuh:637-661`). `kmeans_fit` launches it inside a
`raft::linalg::map_offset`, copies the flag to a **pinned** host scalar, and
reads it at the **top of the next iteration**, decrementing the counter before
breaking. That is the one-thread convergence kernel plus the one-iteration-late
flag read, both of them, in their source.

**What this changes.** The lane was right that neither was in cuVS `94c2819`,
and right to delete an invention attributed to a citation that did not support
it. That is the discipline working. But the sentence as written is false against
three of the six checkouts on this machine, and per rule 1 it should be deleted
in the same commit as whatever touches this file next.

**Whether to re-adopt it.** Weak recommendation, and only as part of finding 9.
The lag does not change the iteration count (the decrement makes it equivalent),
so it is identity-safe, but the sync count per iteration is the same and the
saving is one float readback replaced by one int flag readback. It is a small
win, not the reason the lane should look at 26.08.

## Finding 8. The SVM row norm is a serial chain where the pinned block fold already exists

**Our code.** `svm/derived/distance/kernel_matrices.mojo:77-92`. One thread per
row, a serial loop over `n_cols`, reading `x[i * k + c]`. Threads in a warp
therefore read addresses `k` apart, so the access is fully uncoalesced as well
as serial along the reduction axis.

**Their code.** `matrixRowNormL2` calls
`raft::linalg::norm<L2Norm, Apply::ALONG_ROWS>`, which dispatches to
`coalescedReduction`, one warp or block per row with lanes striding along the
row so consecutive lanes read consecutive addresses.

**The reason given, and why it does not hold.** `svm/NOT_IMPLEMENTED.tsv` records the
substitution as "MIRRORED BY A PER-ROW SERIAL CHAIN ... no fold shape to pin".
There is one. `core/row_norms.mojo` is exactly this operation as a pinned block
fold: `NORM_TPB` is `lib_block_size_for[K_LIB_ROW_NORM]` (a pinned width,
DEVIATION 508), the lane fold is replaced by
`core/pinned_reduce.pinned_block_sum` (a halving tree with no lane primitive),
the contraction is `identical_mul_add`, and the accumulator is `ftz`-flushed.
`IDENTITY_PATHS` row 21 names `K_LIB_ROW_NORM` as one of exactly three library
rows whose block size is a summation order, and pins it. `cluster/` and `kde/`
both already call it.

**Weight.** Small. `row_norms_l2sq` is called once per fit
(`svm/derived/svm/kernelcache.mojo:173`) and twice at predict
(`svm/derived/svm/svc_impl.mojo:246-247`), so the win is one `O(n_rows *
n_cols)` pass, not a per-iteration one. I am rating it low deliberately; it
matters mainly because the *stated reason* is wrong and will otherwise be
copied forward.

**Identity.** The pinned block fold is cross-vendor identical, but it is a
different summation order from the serial chain, so the recorded bits move. The
SVM card needs re-baselining, and the `smo_oracle.mojo` float32 spelling has to
follow the same tree (it already replays `row_norm_kernel`'s halving tree for
kde's sqeuclidean norms, so the pattern exists).

## Finding 9. What cuVS 26.08 changed in the Lloyd loop

Our `cluster/` is pinned to cuVS `94c2819`. Between that and `6ba2ce2`,
`detail/kmeans.cuh` changed by 1,332 diff lines and `kmeans_common.cuh` by 772.
Most of it is RAFT mdspan API modernization with no algorithmic content. Four
pieces are not.

| piece | theirs, `cuvs-v26.08.00` | ours | worth |
|---|---|---|---|
| (a) centroid buffers are swapped, not copied | `kmeans.cuh`, `std::swap(cur_centroids_ptr, new_centroids_ptr)` after the shift is measured | `kmeans.mojo:1266-1275`, a `copy_f32_kernel` of `n_clusters * n_features` every iteration | small but free. One launch and one `cd`-element device copy per Lloyd iteration, deleted. Identity-safe, a swap is not arithmetic |
| (b) the fit streams the data in batches | `kmeans.cuh:678-980`, `batch_load_iterator` over `device_buffer_samples`, per-sample buffers sized to the batch not to `n_samples`, host-resident `X` supported | whole-dataset buffers | large, and it is the piece that lifts a memory ceiling rather than a time one. Identity needs the batch count PINNED, since it decomposes every fold |
| (c) weights are renormalized instead of rejected | `kmeans.cuh`, `weightSum` then a device `map` of `w * n_samples / sum(w)` | `checkWeight`'s throw, mirrored | a behavior fix, small. Note it changes what a fit accepts |
| (d) the cost criterion is always on | `check_convergence` reads `clustering_cost` unconditionally; `process_batch` accumulates it from the `minClusterAndDistance` already in hand | `kmeans.mojo:1277-1296`, gated on `params.inertia_check`, which is false by default | this changes iteration counts, so it is a **model change**, not an optimization. Do not take it without a decision |

(a) is the one to take first, and it is nearly free.

I did not evaluate whether (b) is worth the disruption to the identity trace
tags, which are keyed by algorithm position and would gain a batch axis.

## Finding 10. DEVIATION 602 is confirmed by upstream's own fix

Not a defect, recorded because it is a positive result and because it should
change how the deviation is described.

DEVIATION 602 (`kde/derived/neighbors/kernel_density.mojo:398-420`) found that
the cosine kernel's normalization loop in cuML's
`kernel_density.py:131-137`, inherited from scikit-learn's
`_binary_tree.pxi:465-470`, is wrong for even `d`: the loop unrolls the
recurrence `I_n = 2/pi - n(n-1)(2/pi)^2 I_{n-2}` and stops at the `I_0 = 2/pi`
base, which is correct for odd `d` only. For even `d` the chain should end at
`I_1 = 2/pi - (2/pi)^2` and the loop never adds the second term, giving a NaN at
`d = 4`.

cuVS 26.08 fixed the same bug, by the same recurrence, with the same two base
cases (`cuvs-v26.08.00/cpp/src/distance/kde.cu:295-317`), and its comment says
so: "the old loop-based formula missed the cos terms at t=0 for even d".

So DEVIATION 602 is no longer "we fixed their bug", it is "we match their
current source, independently derived". That is a stronger claim and it should
be written that way when the file is next touched, with the cuVS citation
beside the scikit-learn one.

## Stated gaps

Things that look important and that I did not check. A stated gap is useful; a
silent one is not.

1. **`gbdt/`, entirely.** Its upstream is absent from this machine (finding 6).
   Seventy-eight mapped files, the largest tree here, unaudited.
2. **`ensemble/` and `extratrees/`.** cuML's RF is a large, heavily tuned lane
   and I did not open `cpp/src/decisiontree/batched-levelalgo/` on either pin.
   The 25.08 to 26.08 delta there is unmeasured by this audit.
3. **`isolation_forest/`.** Exists only in `cuml-v26.08.00`, so it is the one
   lane where the pin is forced and where a 25.08 read would be empty. I
   confirmed the lane cites `265b9da` and went no further.
4. **`metrics/`.** Their trustworthiness and silhouette paths use
   `cub::DeviceSegmentedRadixSort` and tiled `pairwise_distance`, both of which
   materialize upstream too, so I expect our ports to be faithful. Not verified.
5. **`decomposition/`, `glm/`, `solver/`, `gemm/`, `core/`.** `IDENTITY_PATHS`
   rows 27 through 41 cover these in far more detail than a design audit could
   add in one pass, and the linear-algebra section reports all six of rows 27 to
   32 closed. I read the ledger, not the code.
6. **`spectral/`.** Its map claims "RAFT-26.08 VERIFIED" on most rows and it
   substitutes a host cyclic Jacobi for `raft::linalg::eig_dc` (cuSOLVER,
   closed, so the substitution is rule-legal). Whether a host eigensolver on the
   critical path is the same problem as finding 5 is unchecked.
7. **`arima/`, `tsa/`, `holtwinters/`, `transformer/`, `mamba/`.** Untouched.
8. **`warp.vote` on Metal.** Finding 4 rests on it and it is unprobed.
9. **cuVS's KDE tests and golden generator** as an oracle for finding 1.

## Honest assessment of how much is available

**The ports are faithful, and the documentation discipline is unusually good.**
That is the main result, and it deserves to be stated before the findings are
weighed. Every lane I opened had deviation blocks with numbers, upstream line
citations, a stated reason, and in several cases a priced cost for the thing
that was *not* adopted. `dbscan`'s DEVIATION 34 names its own gap as "the
largest remaining gap in this file" and quotes RAFT's own 32x figure. The fused
k-NN file leaves the double buffer OPEN and names the matrix row that would
close it. `svm/README.md` wrote the determinism argument for a cache it has not
yet ported. This is not a codebase where a reader finds silent divergence.

So the value here is not "we missed a dozen design ideas". It is concentrated:

- **One large genuine miss, KDE**, and it is large because the lane looked in
  the wrong checkout and concluded a fused kernel did not exist. That single
  finding is probably worth more than everything else in this file combined,
  both in speed and because the current path cannot allocate at shipped sizes.
- **One large known deferral, the SVM kernel cache**, already scoped, already
  argued for determinism, waiting on a lane owner rather than on analysis.
- **One medium known deferral, the fused k-NN double buffer**, where the matrix
  accessor is already built and unused, so the remaining work is one kernel.
- **A handful of small items** where a stated blocker looks expired (findings 4
  and 8) or where an upstream refactor landed after our pin (finding 9a).
- **Three documentation corrections** that matter because the repository's own
  rule is that a false sentence gets deleted rather than annotated, and because
  the stale pin table (finding 6) is the mechanism that produced finding 1.

Two structural observations for the orchestrator. First, **the pin split is the
single highest-leverage process fix available**, cheaper than any of the
findings and the direct cause of the largest one. Second, **the failure mode
this repository actually has is not sloppiness, it is a lane reading the tree it
was told about rather than the tree its algorithm lives in**, which is exactly
what rule 0b-i and the pinned-tree memory already say and exactly what happened
again in `kde/`.
