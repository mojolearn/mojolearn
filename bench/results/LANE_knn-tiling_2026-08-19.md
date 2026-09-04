# LANE knn-tiling — 2026-08-19

**Verdict in one line.** We had ported cuVS's k-NN *fallback* and benchmarked it
as if it were their algorithm; their dispatch sends every parameter set we have
ever measured to `fusedL2Knn` instead, which never materializes the distance
matrix. The fused kernel is now written, wired through their dispatch, reached
by both benchmarks, and exact against a host Float64 oracle at k = 1, 8, 32, 64
with a partial tile on all three axes. The strongest remaining gap is that the
selector inside it is a placeholder, not their `faiss_select::WarpSelect`.

**Second finding, and it invalidates every k-NN check taken before today:** the
check fixture was not random. It was affine in the row index, i.e. a lattice.

---

## 1. DIVERGENCES FOUND

| # | What upstream does (file:line) | What ours did | Fixed? |
|---|---|---|---|
| **D0** | `knn_brute_force.cuh:443-447` — `brute_force_knn_impl` dispatches to **`fusedL2Knn`** when `k <= 64 && rowMajorQuery == rowMajorIndex && rowMajorQuery == true && metric ∈ {L2Unexpanded, L2SqrtUnexpanded, L2Expanded, L2SqrtExpanded}`; `tiled_brute_force_knn` is the **else** branch, reached only past that gate (and only after a `Haversine` case at `:481-487`). | This repo had ported **only the else branch**, and `bench/scaling_main.mojo` / `bench/bench_main.mojo` called it directly at k=10, row-major, L2, d=32 — all four conditions true, so cuVS would have run the fused kernel. Every published k-NN number compared our port of their fallback against scikit-learn. | **FIXED.** `fused_l2_knn.mojo` written; `brute_force_knn_impl` added with their four conditions in their order; both bench call sites now go through the dispatch. |
| **D1** | `:145-166`, `:265-320` — they tile BOTH axes: `num_col_tiles = ceildiv(n, tile_cols)`, `temp_out_cols = k * num_col_tiles` shortened by `k - last_col_tile_size` when `last_col_tile_size && last_col_tile_size < k` (`:152-153`), per-tile `select_k` at `:265`, a column-id fixup at `:278-300` whose own comment says `select_k` writes row-major so tiles cannot be concatenated, and a FINAL `select_k` over the concatenation at `:305-320`. | Tiles queries only. Docstring admitted it. | **NOT FIXED** — deliberately deprioritized by the orchestrator's redirect once D0 landed, since this is their fallback. Full execution notes in §7 so the next lane does not re-read their file. |
| **D2** | `:157-166` — `raft::matrix::fill(distances, lowest())` and, guarded by `std::is_signed_v<IndexType>`, `fill(indices, -1)` when `n < k`. | Not ported. Worse: the selector this reaches, `select_radix.mojo:329`, looks for the bucket where `prev_count < current_k <= cur_count`; with fewer than `k` elements in the whole row no bucket ever satisfies it, `ctr` is never updated, every later pass drops every element, and `last_filter` reads a buffer nothing wrote. Silent garbage. | **PARTIALLY FIXED**: `brute_force_knn_impl` now **raises** on `k > n_index` instead of returning garbage. The fill itself is still unported; it needs the same packed-output scatter D1 needs. Read from their file and ours; **not measured on hardware.** |
| **D3** | `distance_ops/l2_exp.cuh:120-129` — the L2-expanded epilogue is `val * (val > 0) * !((val*val < get_clamp_precision<float,float>()) && (regxn[i] == regyn[j]))`, i.e. **two** clamp clauses, the second zeroing a *small positive* whose two row norms are equal (their fix for a point finding itself at round-off distance rather than 0). `get_clamp_precision<float,float>() == 1e-6` (`:31-39`). | `core/expand_distances.mojo:40-41` has only the first clause (`if d <= 0: d = 0`). A self-match at ~1e-4 stays 1e-4 instead of becoming 0, so it can rank behind a genuine zero-distance duplicate. | **Fixed in the fused kernel** (both clauses ported). **NOT fixed in `core/`** — not my file. Proposed patch in §5. |
| **D4** | `:110-146` — `rowNorm<L2Norm, true>` with `raft::sqrt_op{}` for Cosine and without it for L2 (squared), hoisted out of the tile loop. | `compute_norms` + `core/row_norms.mojo`, same flag, same hoist. | **No divergence.** Confirmed for both metrics. |
| **D5a** | `raft/linalg/detail/contractions.cuh:282-313` — `ldsXY` gives thread `accrowid` the rows `accrowid + i*AccThRows` and columns `acccolid + j*AccThCols`: **strided**. | `core/gemm.mojo:172-173` uses **blocked** (`tr * AccRowsPerTh`, `tc * AccColsPerTh`), and `simt_kernel.mojo:198-199` inherits it. Same arithmetic, different thread→output map, different shared-memory bank pattern, and it would silently break any epilogue that recomputes their index expressions. | **Fused kernel uses THEIRS (strided).** `core/` not fixed — not my file. |
| **D5b** | `fused_l2_knn.cuh:1017-1019` — `constexpr bool sqrt = false`, the root taken afterwards over `n*k` outputs by `raft::linalg::unaryOp` at `knn_brute_force.cuh:463-472` (`powf(fabsf(x), 0.5)`). | The tiled path takes the root inside `expand_distances_kernel` over all `m*n` cells. | Copied correctly in the fused path (`sqrt_postprocess_kernel`). The tiled path's choice is their own on that path, so it is not a divergence there. |
| **D5c** | `select_k-inl.cuh:47-72` — `choose_select_k_algorithm` sends `2 < k <= 256` to WARPSORT and only `k > 256` to radix. | Radix is our only ported selector for the fallback. | Not fixable here; `select_warpsort.mojo` and `faiss_select/` are other lanes'. Recorded. |
| **D5d** | `fused_l2_knn.cuh:251-279`, `:313-337` — cross-column-block merge with a mutex array, `atomicCAS`/`atomicExch`, `__threadfence`. | Not ported; we launch `gridDim.x == 1`. | **Not a behavioral divergence**: their own `rowEpilog_lambda` opens `if (gridDim.x == 1) { return; }` (`:226`) and their final store is gated on `gridDim.x == 1` (`:474`), so at that configuration their kernel *is* ours. We lose their `grid.x > 1` configuration, which `launchConfigGenerator` picks when `m` is small. Same deviation, same reason, as `simt_kernel.mojo`. |
| **D5e** | `fused_l2_knn.cuh:221-222` — the selector is `faiss_select::WarpSelect<AccT, uint32_t, false, Comparator<AccT>, NumWarpQ, NumThreadQ, 32>`, register-resident, merged by `updateSortedWarpQ` (`:148-179`) with `__ballot_sync` / `__shfl_up_sync` / `__ffs`. | Placeholder: shared-memory sorted list, threshold reject (theirs, `:383`), staging buffer (theirs, `:424`), serial merge (**not** theirs). | **NOT FIXED BY DESIGN** — `faiss_select/` is the warpsort lane's. Marked `SELECTOR SLOT` in the file. |
| **D5f** | `Policy::SmemSize = 2 * SmemPage` (`contractions.cuh:100`) — their contraction is double-buffered. | Single page, as `core/gemm.mojo` already is. | Deviation, with a number: two pages at `Policy2x8` is 36,992 bytes against Metal's 32 KB (`archive/reference/PORTING.md 1`). Forced. It is also why we cannot alias their 256-wide `allWarpTopKs` over the GEMM pages, which is what drove DEVIATION BLOCK 2 in the file. |
| **D5g** | `fused_l2_knn.cuh:405-412` — the staging slot comes from a warp prefix sum over `numValsWarpTopK`. | Shared-memory integer `Atomic.fetch_add`. | Different ORDER of staged candidates, same SET; stage 3 sorts them, so it cannot change the result. Documented in the file. |
| **D5h** | `faiss_distance_utils.h::chooseTileSize` — `preferredTileRows = 512`, `1024` if `dim <= 32`; `tileCols = numCentroids` if `tileRows * n * elemSize * 2 <= 512 MB`, else `min(targetUsage / (2*elemSize*tileRows), n)` with `targetUsage` 768 MB / 1 GB by device memory; then `tile_cols = max(tile_cols, k)` (`:107`). | Not ported at all; `query_tile` is a caller constant (256 in the benches) and there is no column tile. | **NOT FIXED** — goes with D1. Note their heuristic would pick `tile_rows = 1024`, `tile_cols = 131072` at the scaling harness' 400k × 2000 × 32, i.e. 4 column tiles, not one. |
| **D5i** | `knn_brute_force.cuh:229-256` — bitmap/bitset filter, masking rejected candidates to ±inf before `select_k`. | Not ported. | Recorded in §3 as an UNPORTED row. |
| **DX** | *(not upstream — ours)* `neighbors/mojo_only/knn_check.mojo::_coord` claimed "hashed so it is reproducible and not structured" but computed `(row*2654435761 + feature*40503 + salt*2246822519) % 1000003`, which is **affine in `row`**: point `j+1` is point `j` plus a fixed offset mod 1 in every feature at once. A lattice, not a sample. | At 4,093 points in 20 dimensions the true nearest-neighbour distance on that lattice is **4e-6** while row norms are ~7, whose float32 ulp is ~5e-7 — the expanded identity cannot rank them, exactly the failure this file's own header describes for collinear points. | **FIXED**: `_coord` is now splitmix64. `bench/*_main.mojo::_u01` was already a splitmix mixer, so **no measured number was taken on the lattice — only the checks were.** |

---

## 2. What I changed, file by file

**`neighbors/gbdt/neighbors/detail/fused_l2_knn.mojo`** — NEW, ~460 lines.
Port of `cuvs/cpp/src/neighbors/detail/fused_l2_knn.cuh::fusedL2kNN` plus its
`fusedL2ExpKnn` driver.
- Policy constants from `raft/linalg/contractions.cuh:203-206`
  (`Policy2x8<float,1>` = `KernelPolicy<float, 1, 16, 2, 8, 8, 32>`: 256
  threads, `Mblk=16`, `Nblk=256`, `Kblk=16`). Deliberately **not**
  `core/gemm.mojo`'s Policy4x4 — `fusedL2ExpKnnImpl:722` selects Policy2x8 for
  the k-NN shape and 16 rows per block is what lets a per-row top-k fit shared
  memory.
- Contraction body from `raft/linalg/detail/contractions.cuh:176-313`
  (`ldgXY`/`stsXY`/`ldsXY`), **strided** thread→output mapping (D5a).
- Epilogue from `distance_ops/l2_exp.cuh:120-129`, both clamp clauses (D3).
- Selection structure from `fused_l2_knn.cuh:378-386` (threshold reject),
  `:424` (staging), `:433` (`updateSortedWarpQ`), with the merge itself a
  placeholder (D5e) behind a marked `SELECTOR SLOT`.
- Store from `:93-115` (`storeWarpQGmem`), reached through their
  `gridDim.x == 1` guard at `:474`.
- `sqrt_postprocess_kernel` from `knn_brute_force.cuh:463-472`.
- Host driver `fused_l2_knn` with their validation block `:1000-1015`.
- Shared memory: 30,848 bytes of Metal's 32,768 (`sx` 1,088 + `sy` 17,408 +
  top-k lists 8,192 + staging 4,096 + counters 64). Sized for `FKNN_MAX_NN=64`.

**`neighbors/gbdt/neighbors/detail/knn_brute_force.mojo`**
- Docstring: the "SUBSTITUTION AUDIT, 2026-08-19: NOTHING LEFT HERE" block is
  **deleted**, not annotated. Its conclusion was false — the thing left was the
  entire fused path — and it reasoned from the retired rule 2. Replaced with
  what their dispatch actually does.
- `brute_force_knn_impl` ADDED: their `:443-447` conditions, in their order.
- `k > n_index` now raises (D2).
- `tiled_brute_force_knn` restored to its two-selector form. It had been
  reduced to `nn.topk.top_k`-only in the working tree; see §7 on the race.

**`neighbors/mojo_only/knn_check.mojo`**
- `_coord` replaced with splitmix64 (DX), and the module docstring's claim
  about the fixture corrected in place.
- `check_fused_l2_knn`, `check_fused_reach_by_sabotage`,
  `check_dispatch_takes_fused` ADDED.
- Fused fixture is 53 queries × 4,093 index × 20 features — a partial tile on
  **all three** of `Mblk=16`, `Nblk=256`, `Kblk=16` simultaneously. The old
  4096/64/16 fixture crossed no tile edge at all.

**`neighbors/knn_main.mojo`** — the three new checks wired in.

**`neighbors/gbdt/matrix/detail/select_radix.mojo`** — RESTORED from HEAD
(byte-identical); it had been deleted in the working tree. Not otherwise
touched; it is the warpsort lane's directory.

**`bench/scaling_main.mojo:78`, `bench/bench_main.mojo:240`** — call
`brute_force_knn_impl` instead of `tiled_brute_force_knn`. **Flagging loudly:**
this is the only reason the next measurement will show anything. The argument
list is unchanged (the dispatch's positional prefix is identical), so nothing
else in those files moved.

---

## 3. Proposed rows

`neighbors/PORTED_MAP.tsv`:

```
cuvs/cpp/src/neighbors/detail/fused_l2_knn.cuh	neighbors/gbdt/neighbors/detail/fused_l2_knn.mojo	partial	fusedL2kNN + fusedL2ExpKnn driver; selector is a placeholder, not faiss_select::WarpSelect
cuvs/cpp/src/neighbors/detail/knn_brute_force.cuh	neighbors/gbdt/neighbors/detail/knn_brute_force.mojo	partial	brute_force_knn_impl dispatch (:443) + tiled_brute_force_knn (:69-340), query-axis tiling only
raft/cpp/include/raft/linalg/contractions.cuh	neighbors/gbdt/neighbors/detail/fused_l2_knn.mojo	partial	Policy2x8 + Contractions_NT, single-buffered (Metal 32KB)
cuvs/cpp/src/distance/detail/distance_ops/l2_exp.cuh	neighbors/gbdt/neighbors/detail/fused_l2_knn.mojo	partial	l2_exp epilogue, both clamp clauses
```

`neighbors/UNPORTED.tsv`:

```
raft/cpp/include/raft/neighbors/detail/faiss_select/Select.cuh	WarpSelect register queue; the fused kernel's real selector	warpsort lane
cuvs/cpp/src/neighbors/detail/fused_l2_knn.cuh:251-337	cross-column-block mutex merge (atomicCAS/atomicExch)	we launch gridDim.x==1, which is their own gridDim.x==1 path
cuvs/cpp/src/neighbors/detail/fused_l2_knn.cuh:620-696	fusedL2UnexpKnn (L2Unexpanded / L2SqrtUnexpanded arm)	only the expanded arm is ported
cuvs/cpp/src/neighbors/detail/knn_brute_force.cuh:145-166,278-320	two-axis tiling: num_col_tiles / temp_out_cols / column-id fixup / final merge select_k	see lane file section 7
cuvs/cpp/src/neighbors/detail/knn_brute_force.cuh:157-166	n < k short-fill (lowest() / -1)	brute_force_knn_impl raises instead
cuvs/cpp/src/neighbors/detail/knn_brute_force.cuh:229-256	bitmap/bitset filter masking before select_k	no filter type is ported
cuvs/cpp/src/neighbors/detail/faiss_distance_utils.h	chooseTileSize heuristic	needs the two-axis tiling first
cuvs/cpp/src/neighbors/detail/haversine_distance.cuh	haversine_knn, the other arm of their metric switch	
```

---

## 4. Proposed archive/reference/PORTING.md deviation entries (numbered from 30)

**30. A device-wide vendor call cannot be fused, and that is what made k-NN
slow.** `linalg.matmul` must write its output to memory, so substituting it for
a step that upstream fuses freezes the unfused structure permanently: the
distances got materialized, so the top-k had to read them back. cuVS's own
default for those parameters calls no vendor primitive at all. Rule: port the
path their dispatch takes; substitute a MAX equivalent only where that path
itself calls something closed.

**31. Their dispatch is part of the algorithm.** Porting a function upstream
reaches only as a fallback, and then benchmarking it, measures nothing about
them. Check the caller's branch conditions against the benchmark's parameters
before claiming a port is comparable.

**32. `Policy::SmemSize` is TWO pages.** RAFT's contraction policies are
double-buffered, so a faithful `SmemSize` at `Policy2x8` is 36,992 bytes and
does not fit Metal's 32 KB. Single-buffer and say so; and do not assume any
buffer they alias over the GEMM pages will fit in one page.

**33. A hash that is affine in its input is a lattice.**
`(row*A + feature*B + salt*C) % M` gives point `j+1` = point `j` + constant in
every coordinate. It looks random per-cell and is perfectly structured
per-row, and it defeats the expanded identity in float32 for the same reason
collinear points do. Mix (splitmix64) rather than combine linearly.

---

## 5. False doc sentences found in files I may not edit

1. **`core/expand_distances.mojo:9-12`** — "The arithmetic is the same expanded
   identity and the same clamp (`unfused_distance_nn.cuh:80-81`)". For the
   *brute-force k-NN* call site the epilogue upstream is
   `l2_exp_cutlass_op` (`distance_ops/l2_exp.cuh:120-129`), which has **two**
   clamp clauses, and this file implements one. Proposed patch, after the
   existing `if d <= 0` clamp:
   ```
       elif d * d < Float32(1.0e-6) and x_norm.unsafe_load(row) == y_norm.unsafe_load(col):
           d = Float32(0.0)
   ```
   (`get_clamp_precision<float, float>() == 1e-6`, `l2_exp.cuh:34`.) This
   changes arithmetic, so it re-anchors.

2. **`core/gemm.mojo:170-173`** — "THEIR POLICY, COPIED" and the `ldsXY`
   comment, next to a **blocked** `xb = tr * AccRowsPerTh` / `yb = tc *
   AccColsPerTh`. RAFT's `ldsXY` (`detail/contractions.cuh:282-313`) is
   **strided**: `accrowid + i*AccThRows`, `acccolid + j*AccThCols`. The policy
   constants are copied; the index expressions are not.

3. **`neighbors/README.md`** — I did not edit it, but if it repeats this tree's
   claim that the k-NN port is a port of cuVS's brute force, that sentence is
   false as written: it was a port of their fallback. It should name both paths
   and say which one their dispatch takes.

4. **`archive/reference/VENDOR_LIBRARIES.md`** — every entry justified by the retired rule 2
   needs re-reading against "follow their dispatch". At minimum the k-NN
   `select_k -> nn.topk.top_k` entry must be scoped to the fallback path only.

---

## 6. Build / check evidence

```
cd /Users/andrewhendel/CascadeProjects/mojolearn
tools/with_build_lock.sh pixi run --manifest-path .../mojotrees/pixi.toml \
  mojo build -I . neighbors/knn_main.mojo -o /tmp/knn_probe
/tmp/knn_probe
```

```
check_knn OK: 64 queries x k=8 over 4096 index points, every returned neighbor is in the exact true set
check_knn_reach_by_sabotage OK: index_norm moved 512/512 neighbors; query_norm offset moved 0 sets, which is the predicted shape
check_vendor_topk_matches_ported OK: nn.topk.top_k and the ported RAFT radix select agree on all 512 neighbours
check_fused_l2_knn OK: k = 1, 8, 32, 64 over 53 queries x 4093 index x 20 features, every slot matches the host Float64 oracle in order, and is_sqrt is exactly one square root away
check_fused_reach_by_sabotage OK: index_norm ramp moved 424/424 neighbours and 424 distances; query_norm offset moved 0, which is the predicted shape
check_dispatch_takes_fused OK: k=8 wrote out_idx and left out_idx32 at its sentinel; k=65 did the opposite, so the `k <= 64` branch at knn_brute_force.cuh:443 is wired both ways
```

`bench/scaling_main.mojo` and `bench/bench_main.mojo` both build clean. No
timing was run.

### Sabotage results

1. **The fixture sabotage that was not planned and is the strongest evidence
   here.** `check_fused_l2_knn` failed on first run: 19 of 53 slots at k=1,
   with the diagnostic printing `got=3013 d=4.88e-06 want=3567 d=3.68e-06
   gpu_d=3.81e-06`. Distances of 4e-6 are impossible for random points in
   `[0,1)^20`; they are the signature of a lattice. The kernel was right and
   the fixture was wrong (DX). This is the third time in this file's history
   that a structured fixture has accused a correct kernel.

2. **`index_norm` ramped to `1000 * (n - j)`** → 424/424 neighbours moved AND
   424/424 distances moved. Predicted shape: a per-index-point quantity enters
   the distance per column and must reorder. The distance assertion is the
   sharper half — it would catch an epilogue that read the norms while the
   queue's threshold compared a stale value.

3. **`query_norm` offset by `+0.1*(i+1)`** → 0/424 moved. Predicted shape: a
   per-query constant added identically to every candidate of that row cannot
   change a ranking. The *offset*, not a replacement, and *small*: both windows
   were paid for by earlier runs on the tiled path (5000 pushes distances to
   where float32's ulp is 0.03 and the ranking dissolves; replacement drives
   the expanded distance negative into the clamp).

4. **Dispatch reach, two-sided, by which BUFFER comes back written.** Both arms
   return the same neighbours, so equality proves nothing about which ran. The
   fused arm writes `out_idx` (uint32); the fallback under
   `use_vendor_topk=True` writes `out_idx32` (int32) and never touches
   `out_idx`. Seeded with `0xDEADBEEF` / `-777`: at k=8 `out_idx` came back
   correct and all 424 `out_idx32` slots still held `-777`; at k=65 exactly the
   opposite. A dispatch wired to a constant fails one direction or the other.

5. **`is_sqrt` reach.** Two runs at k=8, `False` then `True`; every distance
   matched `sqrt` of the squared run to 1e-5 relative, and the check *also*
   fails if the two runs are bit-identical, which is what would happen if
   `sqrt_postprocess_kernel` were never enqueued.

---

## 7. What I did NOT do, and why

**a) Two-axis tiling of `tiled_brute_force_knn` (D1).** Deprioritized by the
orchestrator's redirect once D0 was found; it is their fallback, now reached
only at k > 64. The reading is done, so here is the whole of it, ready to
execute:

- `tile_rows`, `tile_cols` from `faiss_distance_utils.h::chooseTileSize`, then
  `max_row_tile_size` / `max_col_tile_size` caps, then `tile_cols =
  max(tile_cols, k)` (`:93-107`). **`query_tile` must become
  `max_row_tile_size`, not `tile_rows`**, because `buf_val`/`buf_idx` are
  caller-allocated at `query_tile * 2 * buf_len` and the radix kernel indexes
  them by `batch * 2 * buf_len`; `tile_rows > query_tile` overruns them.
- `num_col_tiles = ceildiv(n, tile_cols)`; `temp_out_cols = k * num_col_tiles`,
  minus `k - last_col_tile_size` when `last_col_tile_size = n % tile_cols` is
  nonzero and `< k` (`:145-153`). `temp_out_distances` / `temp_out_indices` are
  `tile_rows * temp_out_cols` and are allocated INSIDE the function upstream
  (`:167-168`), so allocating them inside is the faithful choice and keeps the
  public signature intact.
- Per column tile: `current_centroid_size = min(tile_cols, n - j)`,
  `current_k = min(current_centroid_size, k)`. The GEMM and epilogue take an
  index sub-buffer at `j * n_features` and `index_norm` offset by `j`.
- Per-tile select writes PACKED at width `current_k`. Our radix kernel's output
  stride is its `k` argument, so passing `current_k` gives exactly the packed
  layout upstream relies on at `:270-274`.
- Fixup (`:278-300`): `out_index = row * temp_out_cols + j*k/tile_cols + col`;
  `temp_out_indices[out_index] = in_indices[i] + j`. `j` is a multiple of
  `tile_cols`, so `j*k/tile_cols` is exact integer arithmetic and equals
  `tile_index * k`.
- Final merge (`:305-320`): `select_k` over `(rows, temp_out_cols)`. **Ours
  cannot take their `in_idx` argument** — `select_radix.mojo` has no such
  parameter and belongs to another lane — so it needs a gather kernel after the
  select: `final_idx[r][s] = temp_out_indices[r * temp_out_cols + pos]`. That
  is exactly equivalent (their `in_idx` just substitutes the payload) and it
  works identically for `nn.topk.top_k`, which also returns positions.
- **The trap to test first** is a last column tile SHORTER than `k`, because
  that is the only case where `temp_out_cols` is not `k * num_col_tiles` and
  the last tile's fixup writes must land exactly flush against the end of the
  row. Force `max_col_tile_size` small, and assert the two-level result is
  **exactly equal** to the single-tile result, not merely a correct set.

**b) The `n < k` fill (D2).** Needs the same packed-output scatter as (a);
doing one without the other duplicates the work. Replaced with a raise so the
current behavior is defined rather than silently wrong.

**c) `faiss_select::WarpSelect`, `select_warpsort.mojo`, `ball_cover/`,
`core/**`, `PORTED_MAP.tsv`, `UNPORTED.tsv`, `neighbors/README.md`.** Other
lanes'. To swap the warpsort lane's queue into the fused kernel, the
orchestrator changes exactly two marked regions in `fused_l2_knn.mojo`: the
`SELECTOR SLOT` shared arrays (become register queues) and the
`updateSortedWarpQ` block (becomes their warp-parallel merge). The staging
buffer and the threshold reject are already theirs and stay.

**d) No timing runs.** Per the standing rule.

**e) A CPU/host fallback arm.** Explicitly out of scope. The host Float64
oracle in the check is an oracle, not a path.

---

## 8. A WARNING THE ORCHESTRATOR NEEDS

**Another agent was writing my files while I worked.** At 18:03, 18:04 and
18:09 today, `knn_brute_force.mojo`, `knn_check.mojo` and `knn_main.mojo` were
rewritten under me, and `select_radix.mojo` and `select_warpsort.mojo` were
deleted and later restored. Twice I restored a file from HEAD and read back
different content seconds later. The version of "delete the hand-written
selectors" that the redirect attributes to me was **not written by this agent**
— I had made no edits at the time it appeared, only reads.

The tree is green as of my last build and run. If any of the six checks is
missing or `select_radix.mojo` is absent when you look, that is the other
writer, not a revert by me. Worth confirming there is only one knn-tiling lane
running before the next round.
