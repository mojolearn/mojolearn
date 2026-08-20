# LANE vendor-correctness, 2026-08-19

Every vendor primitive this repo uses or recommends now has a CORRECTNESS
verdict backed by a run against an independent host oracle. `vendor_main.mojo`
prints the table and exits non-zero when a WIRED primitive tests WRONG.

**IT EXITS NON-ZERO TODAY. Read section 0 first.**

---

## 0. THE WIRED DEFECT. LOUD, AND NOT MINE TO FIX.

**`core/gemm.mojo::gemm_nt` at `n = 1` does not write its output.** Measured
through the real wrapper, not a copy: m=64, n=1, k=32, output buffer poisoned
with `-987654.0` before the call, **63 of 64 rows still held the poison
afterwards**. Zero rows were written wrong; 63 were not written at all.

`gemm_nt` is `linalg.matmul.matmul[transpose_b=True]` and nothing else
(`core/gemm.mojo:88`). The failure belongs to `transpose_b=True`, not to
`n = 1`: the same product with `transpose_b=False` and `y` laid out `k x n` is
CORRECT at the identical shape.

**Why it matters: `n` is a user-facing count at every call site, and `n = 1`
arrives from ordinary parameters.**

| file:line | call | how `n` reaches 1 |
|---|---|---|
| `cluster/gbdt/cluster/detail/min_cluster_distance_compute.mojo:197` | `gemm_nt(ctx, dist_buf, x_tile, c_tile, ns, nc, n_features)` | `nc = min(centroid_batch, n_clusters - c_idx)` (`:183`). n=1 whenever `n_clusters == 1` **or `n_clusters % centroid_batch == 1`**. That is a remainder, not an exotic input. |
| `glm/gbdt/linalg/detail/lstsq.mojo:120` | `gemm_tn(ctx, cov_a, a, a_alias, a_alias2, n_cols, n_cols, n_rows)`, and `gemm_tn` ends in `gemm_nt` (`core/gemm.mojo:148`) | simple linear regression on ONE predictor. No guard on `n_cols` anywhere in `lstsq.mojo`. |
| `glm/gbdt/linalg/detail/lstsq.mojo:193` | `gemm_nt(ctx, inv, qs, q, n_cols, n_cols, n_cols)` | same |
| `decomposition/gbdt/linalg/detail/pca.mojo:202` | `gemm_tn(ctx, cov, x, x_alias, x_alias2, n_cols, n_cols, n_rows)` | PCA on a one-column matrix |
| `decomposition/gbdt/linalg/detail/tsvd.mojo:89` | `gemm_tn(ctx, gram, x, x_alias, x_alias2, n_cols, n_cols, n_rows)` | same |
| `neighbors/gbdt/neighbors/detail/knn_brute_force.mojo:154` | `gemm_nt(ctx, dist_tile, q_tile, index, rows, n_index, n_features)` | one index point |

**Not fixed here.** `core/`, `cluster/`, `glm/`, `decomposition/` and
`neighbors/` all belong to other lanes this round and my brief forbids
touching a call site. The fix is the one RAFT itself uses at `n = 1`: `gemv`.
`core/gemm.mojo::gemv_n` already wraps `linalg.gemv.gemv_gpu`, and that call
tests **GPU, CORRECT** at every m from 1 to 100003. So the change is a size
guard inside `gemm_nt`, not new code.

`VENDOR_LIBRARIES.md` has recorded "`linalg.matmul` with `n = 1` returns zeros
for some outputs" since before this round. **That sentence is wrong in the way
that matters** and I deleted it: it does not return zeros, it does not write,
and a caller reusing a buffer therefore reads stale data rather than an
obviously-wrong zero. It is also not a property of `n = 1`.

---

## 1. THE TABLE

Regenerate, do not trust:

    tools/with_build_lock.sh pixi run \
      --manifest-path /Users/andrewhendel/CascadeProjects/mojotrees/pixi.toml \
      mojo build -I . vendor_main.mojo -o /tmp/vendor_probe
    /tmp/vendor_probe            # exit 1 today, because of section 0
    /tmp/vendor_probe --transpose   # the one probe that ABORTS

Device Apple M4, `WARP_SIZE = 32`, mojo 1.0.0 / max 26.5.0, 2026-08-19.

| symbol | CUDA counterpart | verdict | wired | sizes tested / evidence |
|---|---|---|---|---|
| `core/gemm.mojo::gemm_nt` at `n = 1` | `cublasGemmEx` degenerate | **GPU, WRONG** | **YES** | m=64 n=1 k=32 through the real wrapper: 63 of 64 outputs NEVER WRITTEN |
| `nn.argsort.argsort` | `cub::DeviceRadixSort::SortKeys` | **GPU, WRONG** | no | correct at 1, 2, 255, 256; NOT MONOTONE at 257 and every larger size, first inversion always at output index 256 |
| `linalg.matmul.matmul[transpose_b]` at `n = 1` | `cublasGemmEx` degenerate | **GPU, WRONG** | via `gemm_nt` | 64x1x32; the same product with `transpose_b=False` is CORRECT |
| `linalg.transpose.transpose` | `raft::linalg::transpose` | **GPU, WRONG** | no | 257x129: **ABORTS the process**, not a catchable raise |
| `linalg.matmul.matmul[transpose_b]` | `cublasGemmEx` (N-T) | **GPU, CORRECT** | YES | 1x1x1, 1x8x3, 255x255x33, 256x256x64, 257x257x65, 513x129x127, 1024x512x32, 100003x4x8 vs Float64 host |
| `linalg.gemv.gemv_gpu[transpose_b=False]` | `raft::linalg::gemv` | **GPU, CORRECT** | YES | m in {1,2,255,256,257,512,513,1023,1024,1025,4096,100003}, k in {3..9}; plus m=1000 k=1025 |
| `nn.topk.top_k[largest=False]` | `raft::select_k` | **GPU, CORRECT** | YES | batch=3, n in {1..100003}, k in {1,4,32}; values strict, indices up to ties |
| `max.gpu.primitives.block.sum` | `cub::BlockReduce` | **GPU, CORRECT** | YES | TPB 32/64/128/256/512/1024, 37 blocks, exact integer fixture; broadcast to ALL threads |
| `max.gpu.primitives.block.max` | `cub::BlockReduce(Max)` | **GPU, CORRECT** | YES | same |
| `max.gpu.primitives.block.min` | `cub::BlockReduce(Min)` | **GPU, CORRECT** | YES | same |
| `max.gpu.primitives.block.prefix_sum` | `cub::BlockScan` | **GPU, CORRECT** | YES | same, inclusive AND exclusive, per lane |
| `std.gpu.primitives.id.lane_id` | `raft::laneId` | **GPU, CORRECT** | YES | 32 lanes, 104 warps |
| `std.gpu.primitives.warp.sum` | `cub::WarpReduce` | **GPU, CORRECT** | YES | 32 lanes, 104 warps |
| `std.gpu.primitives.warp.prefix_sum` | `cub::WarpScan` | **GPU, CORRECT** | YES | 32 lanes, 104 warps, **INCLUSIVE** |
| `std.gpu.primitives.warp.shuffle_xor` | `raft::shfl_xor` | **GPU, CORRECT** | YES | butterfly min fold |
| `std.gpu.primitives.warp.shuffle_idx` | `raft::shfl` | **GPU, CORRECT** | YES | lane-0 broadcast |
| `std.gpu.primitives.warp.vote` | `__ballot_sync` | **GPU, CORRECT** | YES | ballot mask per lane |
| `std.gpu.primitives.warp.lane_group_min[8]` | `__shfl_xor_sync(width=8)` | **GPU, CORRECT** | YES | folds within aligned 8-lane groups as documented |
| `nn.toppminp_gpu.run_radix_sort_pairs_gpu` @ `BLOCK_SIZE=256` (**default**) | `cub::DeviceRadixSort::SortPairs` | **GPU, UNCHECKED** | no | **RAISES on Apple at every size incl. n=1**: threadgroup 32900 > 32768 |
| `nn.toppminp_gpu.run_radix_sort_pairs_gpu` @ `BLOCK_SIZE=128` or `64` | same | **GPU, CORRECT** | no | batch=3, n in {1..100003}: keys ascending, multiset exact, payload a permutation pointing at the right key |
| `nn.argmaxmin_gpu.argmaxmin_gpu[largest=False]` | `cub::ArgMin` | **GPU, CORRECT** | no | batch=5, n in {1..100003} |
| `nn.gather_scatter.gather[axis=0]` | `thrust::gather` | **GPU, CORRECT** | no | rows {1..100003} x 3 cols, hashed non-monotone indices |
| `linalg.bmm.batched_matmul[transpose_b]` | `cublasGemmStridedBatchedEx` | **GPU, CORRECT** | no | batch 1..7, m/n/k straddling 255/256/257 and 1025, per-batch Float64 oracle |
| `nn.cumsum.cumsum` | `cub::DeviceScan` | **CPU ONLY** | no | no ctx, no target; `target="gpu"` is "invalid call, unexpected argument". Correct on HOST at n=4099, inclusive and exclusive |
| `linalg.qr_factorization.qr_factorization` | cuSOLVER `geqrf` | **CPU ONLY** | no | passing a `DeviceContext`: invalid call |
| `nn.arg_nonzero.arg_nonzero` | `cub::DeviceSelect::Flagged` | **CPU ONLY** | no | passing a `DeviceContext`: invalid call |
| `algorithm.reductions.reduce_*` | `cub::DeviceReduce` | **NOT FOUND** | no | "unable to locate module 'reductions'". **The module does not exist.** |
| `linalg.matmul.matmul[transpose_a=True]` | `cublasGemmEx` (T-N) | **NOT FOUND** | no | compile-time: `linalg/matmul/__init__.mojo:110:9` constraint failed: transpose_a not yet supported |
| `nn.argsort.argsort` on rank 2 | `cub::DeviceSegmentedRadixSort` | **NOT FOUND** | no | compile-time: `nn/argsort.mojo:547` constraint failed |
| `nn.softmax.softmax` | — | **GPU, UNCHECKED** | no | imports; takes an `InputFn` closure. No call site here or in cuVS/cuML's dispatch for our algorithms |
| `nn.concat.concat` | — | **GPU, UNCHECKED** | no | imports; same |
| `nn.moe.moe_create_indices` | CUB bucket sort + CSR | **GPU, UNCHECKED** | no | imports; needs an MoE-shaped fixture |
| `algorithm.rowwise.launch` + `algorithm.reduce_op.ArgMin` | `cub::BlockReduce<KeyValuePair>` | **GPU, UNCHECKED** | no | both import; needs a user `ReduceOp` struct and a body closure. **The row worth promoting next.** |
| `max.algorithm.reduction.sum` | `cub::DeviceReduce` | **GPU, UNCHECKED** | no | imports; not run |

---

## 2. DIVERGENCES FOUND

"Upstream" here is the MAX kernel library and `VENDOR_LIBRARIES.md`'s account
of it, not cuVS/cuML — this lane audits the vendor layer.

| what the source/doc says (file:line) | what is actually true | fixed? |
|---|---|---|
| `VENDOR_LIBRARIES.md`: "`linalg.matmul` with `n = 1` returns zeros for some outputs, no error" | It does **not write the output at all**. 63 of 64 rows kept their poison value. And it is `transpose_b=True` that fails, not `n = 1`: the N-N form at the identical shape is CORRECT. | Sentence DELETED and replaced in `VENDOR_LIBRARIES.md`. The CODE defect is **NOT fixed** — `core/gemm.mojo` belongs to another lane. Section 0. |
| `VENDOR_LIBRARIES.md` A1: `nn.argsort.argsort` -> **GPU**, on a signature | **GPU, WRONG** above 256. Independently reproduced this round with a fresh splitmix64 fixture and a host radix-sort oracle: correct at 1/2/255/256, first inversion at output index 256 for 257/512/513/1023/1024/1025/4096/100003. | Row marked WRONG; the standing rule at the top of the file now forbids citing availability. |
| `VENDOR_LIBRARIES.md` A1/A2 item 2/A3 C3: `algorithm.reductions.reduce_{sum,max,min,mean,product,argmax,argmin}` -> **GPU**, "the plural module", with docs URLs | **The module does not exist.** `from algorithm.reductions import reduce_sum` -> "unable to locate module 'reductions'". This was a documentation crawl that was never compiled, and it produced three separate "correction" paragraphs asserting a capability that is not there. `algorithm.reduce_op` and `algorithm.rowwise.launch` DO compile. | All four sites corrected; the whole `## algorithm.reductions` section rewritten. |
| `VENDOR_LIBRARIES.md` A2 item 1: `run_radix_sort_pairs_gpu` recommended as the missing `SortPairs`/`SegmentedRadixSort`, on a signature | Correct at `BLOCK_SIZE` 128 and 64. **Its DEFAULT `BLOCK_SIZE=256` cannot run on Apple at all** — "Threadgroup memory size (32900) exceeds the maximum threadgroup memory allowed (32768)", at every size including n=1. Anyone following the recommendation as written would have hit a hard failure on the first call. | Rewritten with the measured facts and the three previously-unsettled caveats resolved. |
| `VENDOR_LIBRARIES.md`: "`linalg.transpose` **SIGNALS at runtime**" | It **ABORTS the process**. Not a catchable raise; a `try` around it does not help. `_copy_with_strides rank=2 dtype=f32` -> "enqueue_cpu_range is only supported on CPU DeviceContexts". "Signals" reads like something a caller can fall back from. | Corrected. The probe is gated behind `vendor_main --transpose` precisely because it would take the table down. |
| `VENDOR_LIBRARIES.md`: "## Also confirmed available, unused so far: `gather`, `top_k`, `softmax`, `concat`, `linalg.transpose`, `qr_factorization`" | Two of the six are unusable and one of those ABORTS. Presenting them as a menu is the failure mode this lane exists to close. | Section replaced with a verdict table. |
| `VENDOR_LIBRARIES.md`: `cuSOLVER syevj` row -> "`linalg.qr_factorization` AVAILABLE", as the device-side consolation prize | **CPU ONLY**, confirmed by compile: passing a `DeviceContext` is an invalid call. (C1 later in the same file said so; the headline row still said otherwise.) | Row corrected. |
| `decomposition/gbdt/linalg/detail/pca.mojo:299` comment: "Only thread 0 is promised the reduction's result by CUB's contract, so it is published through shared memory rather than assumed broadcast." | On this backend `block.sum` / `max` / `min` **do broadcast to all threads**, verified at TPB 32..1024. The comment states CUB's contract, which is a defensible reason to keep the shared-memory publish, but the empirical claim it implies about MAX is not what happens. | **Not edited** — `decomposition/` is another lane's. Reported here. |
| `VENDOR_LIBRARIES.md` A3 C6: "`lane_id` is not in `std.gpu.primitives.warp`" | `neighbors/.../ball_cover/registers.mojo:78` imports `lane_id` from `std.gpu.primitives.warp` and it compiles, so it is re-exported there. `std.gpu.primitives.id.lane_id` also compiles. Both work; C6 is right about the canonical home and wrong that the other path fails. | Left alone — harmless, and C6's advice (use `.id`) is what this lane's check does. |

---

## 3. WHAT I CHANGED

### `mojo_only/vendor_correctness_check.mojo` (NEW, ~1900 lines)

The whole harness. Design constraints, each written into the file's docstring:

- **oracles computed independently on the host** — an LSD radix sort over IEEE
  bit patterns for every sort/selection check (four 8-bit counting passes,
  written here and shared with nothing in the tree), a Float64 triple loop
  with the direct formula for every matmul/gemv check;
- **splitmix64 fixtures over `(index, salt)`**, with a distinct salt per check
  so two checks cannot silently become the same fixture;
- **integer-valued Float32 keys under 2^23** for the sort and reduction
  checks, so host expectations are EXACT and a mismatch cannot be argued away
  as rounding;
- **poisoned outputs** (`-987654.0`) everywhere a call might not write. That
  is what turned "matmul returns zeros" into "matmul does not write";
- **sizes 1, 2, 255, 256, 257, 512, 513, 1023, 1024, 1025, 4096, 100003** on
  every size-parameterized check;
- **ties handled explicitly** for `top_k`, `argmaxmin_gpu` and
  `run_radix_sort_pairs_gpu`: values strict, indices only up to ties.

Checks: `check_argsort`, `check_matmul`, `check_wired_gemm_at_n1`,
`check_gemv`, `check_topk`, `check_block`, `check_warp`,
`check_radix_sort_pairs`, `check_argmaxmin`, `check_gather`, `check_bmm`,
`check_cumsum`, `check_compile_established`, and `check_transpose_aborts`
(not in the table run — it aborts).

`WARP_LANES` is read from `mojo_only/kernel_matrix.mojo::lib_lane_width_for`,
not typed, so the warp check tests 64 lanes on AMD without an edit.

### `vendor_main.mojo` (NEW, repo root)

Prints the table; exits 1 if a WIRED primitive tested WRONG; `--transpose`
runs the aborting probe alone. The table is sorted so WIRED-and-WRONG is the
first row.

### `VENDOR_LIBRARIES.md` (REWRITTEN around a correctness column)

- New top section: **the standing rule** ("a signature proves reach, a run
  proves the answer"), the five tiers, and what a `GPU, CORRECT` verdict has
  to survive to be earned.
- New **correctness table** — symbol | CUDA counterpart | verdict | wired |
  sizes tested — with the date and toolchain on it, and the regeneration
  command.
- New section explaining the wired `n = 1` defect with its five call sites.
- Every sentence that recommended a primitive on signature evidence alone was
  **deleted or replaced with a verdict**, not annotated. The reachability
  probe results are all kept: what MAX ships is still a fact worth having.
- `## How to add a row` now has three steps and the third is the run.
- Appendix A carries a SUPERSEDED banner naming the three verdicts it got
  wrong.

---

## 4. PROPOSED `PORTED_MAP.tsv` / `UNPORTED.tsv` ROWS

None. This lane ported nothing from cuVS/cuML/RAFT; it audits the vendor
layer. The one thing worth recording in `UNPORTED.tsv` if the orchestrator
wants it:

```
cub	DeviceRadixSort	cub/device/dispatch/dispatch_radix_sort.cuh	NO LONGER NEEDED as a port: nn.toppminp_gpu.run_radix_sort_pairs_gpu is a working device key/value radix sort, GPU CORRECT at BLOCK_SIZE 128 or 64 across n in 1..100003, and it segments by batch row. Its default BLOCK_SIZE=256 raises on Apple (threadgroup 32900 > 32768), which is why nobody had reached it. nn.argsort remains WRONG above 256 and is not a substitute for either.
```

---

## 5. PROPOSED `PORTING.md` DEVIATION ENTRIES (numbered from 30)

**30. `nn.argsort[target="gpu"]` IS WRONG ABOVE 256 ELEMENTS. DO NOT USE IT.**
Correct at n <= 256; non-monotone at 257 and every larger size tried, with the
first inversion always at output position 256. It raises nothing and returns a
well-formed permutation. Rank-2 input is refused at compile time
(`nn/argsort.mojo:547`), so a `(1, n)` batched shape is not a workaround.
Reproduce: `/tmp/vendor_probe`, section `check_argsort`.

**31. `linalg.matmul[transpose_b=True]` DOES NOT WRITE ITS OUTPUT AT `n = 1`.**
Not zeros — untouched. `transpose_b=False` at the same shape is correct, so
the bug belongs to `transpose_b`. `core/gemm.mojo::gemm_nt` is this call and
its `n` reaches 1 from ordinary parameters (k-means `n_clusters % batch == 1`,
OLS/PCA/TSVD with one feature, k-NN with one index point). Guard `gemm_nt` to
`gemv_n` at `n == 1`, which is what RAFT does and which tests CORRECT.

**32. `linalg.transpose` ABORTS THE PROCESS on device buffers.** Not a
catchable raise — a `try` does not help. `_copy_with_strides rank=2 dtype=f32`
-> "enqueue_cpu_range is only supported on CPU DeviceContexts". Assume the
whole `linalg.transpose` module is host-side. `core/column_stats.mojo::
transpose_kernel` is the device transpose.

**33. A MAX PRIMITIVE'S DOCUMENTED DEFAULT MAY NOT FIT APPLE'S THREADGROUP.**
`nn.toppminp_gpu.run_radix_sort_pairs_gpu` at its default `BLOCK_SIZE=256`
asks for 32900 bytes of threadgroup memory against Apple's 32768 and fails
pipeline creation at every input size. At `BLOCK_SIZE=128` or `64` it is
correct. When a vendor call takes a block-size parameter, the default is an
NVIDIA default.

**34. `algorithm.reductions` DOES NOT EXIST.** The documentation site
publishes it; the toolchain does not ship it. `algorithm.reduce_op` (with
`ReduceSum`, `ArgMin`, `ArgMax`, `Welford`, ...) and `algorithm.rowwise.launch`
are the real symbols. A documentation crawl is not a probe.

**35. BLOCK COLLECTIVES BROADCAST HERE, BUT CUB DOES NOT PROMISE IT.**
`max.gpu.primitives.block.{sum,max,min}` return the reduced value in EVERY
thread on this backend, verified TPB 32..1024. Kernels in this tree publish
through shared memory anyway, which is the portable choice; do not remove that
on the strength of this measurement alone.

---

## 6. FALSE DOC SENTENCES I FOUND IN FILES I MAY NOT EDIT

- `decomposition/gbdt/linalg/detail/pca.mojo:299-300` — "Only thread 0 is
  promised the reduction's result by CUB's contract, so it is published
  through shared memory rather than assumed broadcast." True about CUB's
  contract; on this backend MAX's `block_max`/`block_min` do broadcast. Not a
  bug, but the comment reads as a statement about MAX.
- `neighbors/gbdt/neighbors/detail/knn_brute_force.mojo:50` — "`nn.topk.top_k`
  stays reachable behind [the flag]". Fine as written, and now backed:
  `top_k` is **GPU, CORRECT** at batch=3 and n up to 100003, independently of
  the agreement-with-our-selector check that already existed.
- `dbscan/gbdt/dbscan/adjgraph/algo.mojo:86-90` — "`nn.cumsum` is NOT the
  answer and was never available: it carries neither ... unlike
  `nn.argsort`/`nn.topk`/`nn.gather`, which take both". The `nn.cumsum` half
  is correct and re-confirmed. The clause treating `nn.argsort` as a
  functioning member of that list is now false; `argsort` takes both and is
  wrong.
- `neighbors/gbdt/neighbors/ball_cover/scan.mojo:16` — "There would in any
  case be nothing to substitute: `nn.cumsum.cumsum` and ..." — correct for
  cumsum. But the tree's broader claim that no device sort exists is now
  false: `run_radix_sort_pairs_gpu` is one, at `BLOCK_SIZE <= 128`.

---

## 7. BUILD AND CHECK EVIDENCE

Build (warm, ~2 s):

```
tools/with_build_lock.sh pixi run \
  --manifest-path /Users/andrewhendel/CascadeProjects/mojotrees/pixi.toml \
  mojo build -I . vendor_main.mojo -o /tmp/vendor_probe
```

clean. `/tmp/vendor_probe` prints the table in section 1 and exits **1**
(section 0). `/tmp/vendor_probe --transpose` aborts with the stack frame
`linalg::transpose::_copy_with_strides[...]rank=2,dtype=f32`.

### SABOTAGE — three at once, three predicted movements, all observed

The harness is checks-that-pass, so it must be shown able to fail. Three
independent corruptions applied in one build, then reverted from a scratchpad
copy:

| sabotage | prediction | observed |
|---|---|---|
| swap output positions `0` and `n-1` of the `argsort` result on the host before checking | the sizes that PASSED (255, 256) must start failing, and the first inversion must move from 256 to 1 | `argsort n = 255 FAIL: NOT MONOTONE, first inversion at output index 1`; same at 256; smallest failing size moved 257 -> 255 |
| `block_sum[block_size=TPB](v)` -> `block_sum[block_size=TPB](v + 1.0)` | every block sum must be exactly `TPB` too large | `TPB=32 device 16676 host 16644` (+32), `64` +64, `128` +128, `256` +256, `512` +512, `1024` +1024 |
| `matmul[transpose_b=True]` -> `[transpose_b=False]` with `y` re-laid-out | every non-square shape must mismatch | all 7 shapes FAIL; and `64 x 1 x 32` **started passing**, which is how the `n = 1` bug got pinned to `transpose_b` rather than to `n = 1` |

The third sabotage produced a finding, which is the argument for running them.

A first attempt at sabotage 2 (`block_size = TPB // 2`) was rejected by the
compiler: `max/mojo/max/gpu/primitives/block.mojo:186 constraint failed: Block
size must be a greater than warp size`. Recorded because it is a real limit on
`block.sum`: **block sizes below the warp width are refused.**

An earlier rejected sabotage worth naming: rotating each thread's input index
by one. That leaves the block SUM unchanged and would have "passed" — the
uniform-total trap this repo has been bitten by twice.

### Compile-established rows

Each is its own throwaway file under the scratchpad, built with `mojo build`:

```
p_transpose_a     FAILS  linalg/matmul/__init__.mojo:110:9 constraint failed: transpose_a not yet supported
p_argsort_rank2   FAILS  nn/argsort.mojo:547 constraint failed
p_reductions      FAILS  unable to locate module 'reductions'
p_cumsum_gpu      FAILS  invalid call to 'cumsum': unexpected argument   (the DeviceContext)
p_qr_gpu          FAILS  invalid call to 'qr_factorization': unexpected argument
p_argnonzero_gpu  FAILS  invalid call to 'arg_nonzero': unexpected argument
p_moe             COMPILES
p_rowwise         COMPILES  (algorithm.rowwise.launch + algorithm.reduce_op.{ArgMin,ArgMax,ReduceSum})
p_maxreduction    COMPILES  (max.algorithm.reduction.sum)
p_gpu_rowwise     COMPILES
p_maxgpu_backend  COMPILES  (max.algorithm.backend.gpu.reduction.reduce_launch)
p_apple_mma       COMPILES  (apple_mma_load_8x8 / apple_mma_store_8x8)
p_concat_softmax  COMPILES
p_topk_gpu        COMPILES  (nn.topk.topk_gpu)
```

### Mechanics learned, for the next person

- `DoubleBuffer[dtype](current, alternate, size)` needs
  `rebind[Pointer[Scalar[dtype], MutUntrackedOrigin]](buf.unsafe_ptr())`.
  `unsafe_ptr()` takes its origin from `self` and there is no cast method on
  the returned pointer, so `rebind` is the route.
- `linalg.transpose.transpose(output, input, perms: Pointer[Int], ctx)` — the
  `perms` argument is not in the summary docs.
- `IndexList` is `std.utils`, not `std.index`.
- `nn.topk.top_k` prints "Unsorted top-k is not supported on GPU. Falling back
  to sorted top-k." on every call; `sorted=False` is ignored on the device.
- `max.gpu.primitives.block.sum` refuses `block_size < WARP_SIZE`.

---

## 8. WHAT I DID NOT DO, AND WHY

- **Did not fix the wired `n = 1` defect.** `core/gemm.mojo` and all five call
  sites belong to other lanes this round; my brief says to report it loudly
  and not touch the call site. Section 0 is that report. The fix is a size
  guard routing `n == 1` to `gemv_n`, which already exists and tests CORRECT.
- **Did not run `algorithm.rowwise.launch` + `ArgMin`.** It needs a user
  `ReduceOp` struct and a body closure, which is a lane's worth of work
  against a Mojo closure API this repo has a migration note about. It imports;
  it is `GPU, UNCHECKED`; it is the single row most worth promoting next,
  because it is the only shipped key-carrying reduction and this file twice
  claimed none existed.
- **Did not run `nn.moe.moe_create_indices`.** Needs an MoE-shaped fixture
  (`topk_ids`, `expert_usage_stats`, `expected_count`). Its correctness at 8
  buckets would not answer the question anyone here has, which is scaling to
  k-means-sized cluster counts.
- **Did not run `nn.softmax` / `nn.concat`.** `softmax` takes an `InputFn`
  closure rather than a tensor, and neither has a call site here or in
  cuVS/cuML's dispatch for k-means, DBSCAN, PCA, k-NN or OLS. They are in the
  table as UNCHECKED with that reason, and the sentence that recommended them
  on availability is deleted.
- **Did not probe `enqueue_apple_matmul` or the 8x8 Apple MMA path.** The
  imports compile; `enqueue_apple_matmul` documents a raise unless
  `compute_capability == 5` and this box is an M4. Not this lane's question.
- **Did not time anything.** Forbidden, and this box drifts 2-3x across
  thermal windows. Where a correct primitive might be slower than what it
  replaced, that is noted and left.
- **Did not edit `VENDOR_LIBS.md`, `PORTING.md`, `UNWIRED.md`, `PORTED_MAP.tsv`,
  or any kernel** in `cluster/ dbscan/ decomposition/ glm/ neighbors/ core/
  gbdt/`. `mojo_only/vendor_correctness_check.mojo` IMPORTS
  `core.gemm.gemm_nt` and `mojo_only.kernel_matrix`; it modifies neither.
- **Did not `git` anything.**
