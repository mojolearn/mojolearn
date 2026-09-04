# LANE ball-cover — random ball cover index for the eps query, 2026-08-19

**Verdict.** The index cuML's DBSCAN uses is ported and passes exact set
equality against a host brute force at five configurations, with two
independent sabotages proving both prunes are reached. Nothing was wired into
`dbscan/` — the orchestrator's wiring instructions are in section 8.

**The strongest remaining gap** is that the port is not yet reached by
DBSCAN, so the 37x asymptotic loss it exists to fix is still unfixed in the
shipped path. The second gap is the in-group ordering step, which is
`O(sum |group|^2) = O(m^1.5)` where theirs is a radix sort; that matches the
ball cover's own query bound so it does not change the asymptotics, but it
will dominate the BUILD at large m.

**The find that matters beyond this lane: `nn.argsort[target="gpu"]` is
silently wrong above 256 elements.** See section 1, row D3, and section 6.

---

## 1. DIVERGENCES FOUND

Our code did not previously exist here, so "what ours did" is read as "what
the port does" and every row is a decision that was made against their file.

| # | what upstream does (file:line) | what ours does | fixed, or why not |
|---|---|---|---|
| D0 | `ball_cover.hpp:62` `n_landmarks = raft::sqrt(m)`, truncated into `int64_t` | `rbc_n_landmarks` = `floor(sqrt(m))` with an integer correction loop | SAME. Confirmed against their file; the brief said "do not assume" and it was checked. |
| D1 | `ball_cover.cuh:89-97` `sampleWithoutReplacement(rng 12345, weights all 1)`, which is `rng_impl.cuh:292-325`: random key per element, FULL device sort of all m, take first sqrt(m) | Floyd's algorithm on the host, splitmix64 keyed on the same seed | DIVERGES DELIBERATELY. With equal weights their three steps *are* "a uniform subset of size sqrt(m)"; Floyd's draws that directly in O(sqrt(m)). The subset differs; **the answer cannot**, because the triangle-inequality prune is exact for any landmark set. Their version also needs a device sort, which does not work here (D3). |
| D2 | `ball_cover.cuh:188-200` `k_closest_landmarks` calls `brute_force::build`+`search` at k=1 (GEMM + `select_k`) | `rbc_landmark_1nn_kernel`, a fused argmin, no distance matrix | DIVERGES DELIBERATELY. Their path materializes `m x sqrt(m)` floats — 357 MB at m=200k — for an argmin that keeps two values. RAFT's own k=1 form is `fusedDistanceNN`, already ported in this repo. This is the exact shape the k-NN lane measured a 20x loss on. |
| D2b | their build metric is `L2SqrtExpanded` (`cuml runner.cuh:152`), i.e. the EXPANDED identity, while their query uses `EuclideanSqFunc`, unexpanded | ours is unexpanded in both | DIVERGES, AND OURS IS SAFER. `archive/reference/PORTING.md 21` already records what the expanded identity costs in float32, and here the two formulas are being compared to EACH OTHER — `R_radius` from one against `cur_R_dist` from the other. Their pairing permits a point inside eps by one formula and outside by the other, exactly at the boundary the radius test straddles. |
| D3 | `ball_cover.cuh:148-152` `thrust::sort_by_key` with `NNComp` over `(landmark, distance)`, then `sorted_coo_to_csr` at `:155` | counting sort by landmark (`rbc_count_landmarks_kernel` + exclusive scan, which IS `sorted_coo_to_csr`), then `rbc_scatter_kernel`, then exact rank-by-counting per group in `rbc_rank_kernel` | DIVERGES. **`nn.argsort[target="gpu"]` is measurably WRONG above 256 elements** — see section 6 — so there was never a substitution available, and CUB's radix sort is a port that does not fit this lane. Ours produces the same order theirs does, with ties broken deterministically by point index where `thrust::sort_by_key` leaves them arbitrary. |
| D4 | `ball_cover.cuh:224` `R_radius[r] = R_1nn_dists[R_indptr[r+1] - 1]` with no emptiness guard | radius 0 when the group is empty | FIXED, THEIRS IS A LATENT OOB. An empty landmark 0 reads `R_1nn_dists[-1]` in their code. It is unreachable in practice (a landmark is its own nearest landmark except against a duplicate point with a lower index), but 0 is also the exact answer: an empty ball can hold no neighbor, so pruning it always is correct. |
| D5 | `registers.cuh:498,536` sentinel is `std::numeric_limits<value_idx>::max()` — int64 max, ~9.2e18, used as a FLOAT distance | `Float32` max | SAME BEHAVIOR, cleaner type. Both are larger than any real squared distance in both comparisons that consume them. Theirs is an accident of their template parameter. |
| D6 | `registers.cuh:629-636` `raft::ballot` then `__brev` then `__clz`; bit cleared with `mask &= (0x7fffffff >> k_offset)` | `std.gpu.primitives.warp.vote` then `std.bit.count_trailing_zeros`; bit cleared with `mask &= mask - 1` | SAME ORDER, different instructions. `__brev` exists only so their loop can use `__clz` instead of `__ffs`. Both walk set landmarks ascending, so the column order in `adj_ja` is theirs too. `vote` was probed working on this M4 (section 6). |
| D7 | `registers.cuh:1332-1335` launches `tpb = 64`, two warps per block, `ceildiv(n_queries, 2)` blocks | `tpb = 32`, one query per block, `n_queries` blocks | DIVERGES, FOR CORRECTNESS ON NON-32-LANE HARDWARE. `vote` returns a mask over the whole warp; two 32-lane query groups inside one 64-lane AMD wavefront would merge their ballots and corrupt both rows. One query per block makes the lane group and the warp the same set. Their `query_id >= n_queries` early-out is kept and is dead at this shape. |
| D8 | `registers.cuh:1376,1470` `thrust::exclusive_scan`; `:1453` `thrust::reduce(maximum)` | `rbc_exclusive_scan_kernel`, `rbc_max_reduce_kernel` in `ball_cover/scan.mojo` | PORTED, not substituted. Both are standalone device-wide steps between separate launches in their code too. There is also nothing to substitute: `nn.cumsum` and `max.algorithm.reduction.cumsum` are both host-only (`archive/reference/VENDOR_LIBRARIES.md`). |
| D9 | `registers.cuh:714` `block_rbc_kernel_eps_csr_pass_xd`, taken when `index.n == 2 \|\| index.n == 3` (`:1331`) | not ported; the generic kernel runs at every dimension | NOT FIXED, and it is not a semantic difference: it is the same kernel with the query row copied into a `local_x_ptr[MAX_COL_Q]` register array, which needs the dimension at compile time. Proposed `UNPORTED.tsv` row in section 3. |
| D10 | `registers.cuh:305,64,153` the k-NN query passes | not ported | OUT OF SCOPE (brief priority 3, reached only if 1 and 2 landed; they landed with no room left). Proposed `UNPORTED.tsv` row in section 3. |

### Is the port exact?

**Yes, and their header says so:** "Performs a faster **exact** knn in metric
spaces using the triangle inequality"
(`cuvs/include/cuvs/neighbors/ball_cover.hpp:191`). Neither prune can drop a
point within eps, for any landmark set:

1. every `y` in landmark `r`'s ball has `d(r,y) <= radius(r)`, so
   `d(q,y) <= eps` forces `d(q,r) <= eps + radius(r)` and the ball is not
   skipped;
2. inside a ball the points are ascending in `d(r,y)`, so once
   `d(q,r) - d(r,y) > eps` every remaining point has a smaller `d(r,y)` and a
   larger lower bound.

The port preserves both, and the check asserts set equality with **no
tolerance** rather than a recall threshold. cuVS's own test
(`cpp/tests/neighbors/ball_cover.cu`) uses a recall tolerance; ours does not
need one.

### At what n should it beat brute force?

Structurally, the index touches `O(sqrt(m))` landmark distances plus the
points of the balls that survive, against `O(m)` for the brute-force arm, so
the crossover is where `sqrt(m) * d + (surviving points)` falls below `m * d`.
**This lane ran no timing and is not entitled to a number.** What is measured
here is the prune's strength on the check fixture, which is a proxy the
orchestrator can use to decide where to measure: at n=1200, d=3, eps=2.5 the
honest run visits few enough balls that shrinking every radius by 0.3x still
leaves 86% of the edges, i.e. most balls are already being skipped. The
DBSCAN measurement to run is the interleaved one at 10k / 50k / 200k against
the brute-force arm, since `FIRST_RUN_2026-08-19.md` put the loss at 10k and
the 37x at 200k.

### Which of their eps kernels does cuML's dispatch actually take?

Asked mid-lane and answered from their file rather than assumed. **Both, in a
fixed order — and I ported both plus the dense one.**

`vertexdeg/algo.cuh:119-136` branches on `data.max_k`:

```cpp
int64_t spare_elemets_per_row =
  data.max_k > 0 ? (batch_size * data.N - data.ja->capacity()) / n : 0;
if (data.max_k > 0 && data.max_k < spare_elemets_per_row) { /* one-pass max_k */ }
else                                                      { /* two-pass CSR   */ }
```

and `dbscan/runner.cuh` calls `VertexDeg::run` twice with different `max_k`:

| call site | `max_k` argument | branch taken | what it does |
|---|---|---|---|
| `runner.cuh:258-275`, the degree/core-point loop over all batches | literal `0` (`:262`) | **two-pass CSR** | `ja` is passed only when `need_ja_compute` (`:257`: batch 0, or `sample_weight != nullptr`), so for most batches only the COUNT pass runs |
| `runner.cuh:331-341`, the labelling loop, batches > 0 | `maxklen.at(i)` (`:335`), the longest row measured in the first loop at `:289-293` | **one-pass `max_k`** | fills `ja` in a single walk, using the bound the first loop established |

So their dispatch is: count everything with the two-pass form, learn the
longest row, then re-walk with the one-pass form under that bound. That is
why `rbc_eps_pass` has two overloads and why `algo.cuh:135` ASSERTS that the
bound held rather than retrying — the bound came from a real measurement one
loop earlier.

`rbc_eps_pass_count`, `rbc_eps_pass_fill` and `rbc_eps_pass_max_k` are all
three, and all three are checked against the host oracle. The DENSE kernel
(`registers.cuh:458`) is NOT on cuML's DBSCAN path at all — the dense
`eps_nn` overload (`ball_cover.cuh:533`) has no caller in cuML, whose
`rbc_index != nullptr` branch only ever calls the CSR one. It is ported
anyway because it is a second, independent bookkeeping over the same walk and
so is worth something as a cross-check; that is the only reason, and it
should not be wired.

---

---

## 2. WHAT I CHANGED, FILE BY FILE

All four paths are new; no shared file was touched. `git status` confirms the
only entries under my name are these.

**`neighbors/gbdt/neighbors/ball_cover/common.mojo`** (new)
- `eps_dist_sq` — `registers_types.cuh:62-75` `EuclideanSqFunc`. The only
  functor their codegen instantiates for the eps path
  (`detail/ball_cover/registers_00_generate.py:116,148` → exactly one
  combination, `(int64_t, float, EuclideanSqFunc)`).
- `RBC_FLT_MAX` — `registers.cuh:498,536`, see D5.
- The `NNComp` order (`common.cuh:26-37`) is documented here and produced in
  `ball_cover.mojo`.

**`neighbors/gbdt/neighbors/ball_cover/scan.mojo`** (new)
- `rbc_exclusive_scan_kernel` — `registers.cuh:1376`, `:1470`, and the scan
  inside `sorted_coo_to_csr`.
- `rbc_max_reduce_kernel` — `registers.cuh:1453`.
- `rbc_clamp_kernel` — `registers.cuh:1461-1467`.

**`neighbors/gbdt/neighbors/ball_cover/registers.mojo`** (new)
- `block_rbc_kernel_eps_csr_pass` — `registers.cuh:576-708`.
- `block_rbc_kernel_eps_dense` — `registers.cuh:458-570`.
- `block_rbc_kernel_eps_max_k` — `registers.cuh:859-980`.
- `block_rbc_kernel_eps_max_k_copy` — `registers.cuh:983-1000`.
- `rbc_eps_pass_count` / `_fill` — `registers.cuh:1314-1426` plus the
  `vd[n] = total` copy at `:1484-1490`.
- `rbc_eps_pass_dense` — `registers.cuh:1271-1311` plus the `adj` memset from
  `ball_cover.cuh:288-289`.
- `rbc_eps_pass_max_k` — `registers.cuh:1427-1482`, their exact order: run,
  max of the UNCLAMPED `vd`, clamp only on overflow, scan, compact.

**`neighbors/gbdt/neighbors/ball_cover/ball_cover.mojo`** (new)
- `rbc_n_landmarks` — `ball_cover.hpp:62`.
- `_floyd_sample` — `ball_cover.cuh:62-108`, see D1.
- `rbc_copy_rows_kernel` — `raft::matrix::copy_rows`, used at
  `ball_cover.cuh:102` and `:162`.
- `rbc_landmark_1nn_kernel` — `ball_cover.cuh:180-201` at k=1, see D2.
- `rbc_scatter_kernel` + `rbc_rank_kernel` + `rbc_count_landmarks_kernel` —
  `ball_cover.cuh:121-164`, see D3.
- `rbc_landmark_radii_kernel` — `ball_cover.cuh:212-227`, see D4.
- `rbc_build_index` — `ball_cover.cuh:330-378`, their four steps in order.
- `rbc_eps_nn_query_count` / `_fill` / `_max_k` / `_dense` —
  `ball_cover.cuh:533-575` in the shapes `vertexdeg/algo.cuh:107-164`
  consumes.

**`neighbors/mojo_only/ball_cover_check.mojo`** (new) and
**`neighbors/ball_cover_main.mojo`** (new) — five checks, section 6.

---

## 3. PROPOSED ROWS

For `neighbors/PORTED_MAP.tsv` (tab separated, columns guessed from the
existing file — re-tab if that file's schema differs):

```
cuvs/cpp/src/neighbors/ball_cover/ball_cover.cuh	sample_landmarks, k_closest_landmarks, construct_landmark_1nn, compute_landmark_radii, rbc_build_index, rbc_eps_nn_query	neighbors/gbdt/neighbors/ball_cover/ball_cover.mojo	partial	94c2819	landmark draw is Floyd's on the host; 1-nn is fused instead of brute_force; in-group order is a counting sort plus rank instead of thrust::sort_by_key
cuvs/cpp/src/neighbors/ball_cover/registers.cuh	block_rbc_kernel_eps_csr_pass, block_rbc_kernel_eps_dense, block_rbc_kernel_eps_max_k, block_rbc_kernel_eps_max_k_copy, rbc_eps_pass (both overloads)	neighbors/gbdt/neighbors/ball_cover/registers.mojo	partial	94c2819	ballot is warp.vote + count_trailing_zeros; 32 threads per block, one query per block
cuvs/cpp/src/neighbors/ball_cover/registers_types.cuh	EuclideanSqFunc	neighbors/gbdt/neighbors/ball_cover/common.mojo	partial	94c2819	the only functor their codegen instantiates for the eps path
cuvs/cpp/src/neighbors/ball_cover/common.cuh	NNComp	neighbors/gbdt/neighbors/ball_cover/common.mojo	partial	94c2819	the ORDER is ported, the comparator-taking sort is not
```

For `neighbors/UNPORTED.tsv`:

```
cuvs/cpp/src/neighbors/ball_cover/registers.cuh	block_rbc_kernel_eps_csr_pass_xd	:714	same kernel as the generic CSR pass with the query row held in a local_x_ptr[MAX_COL_Q] register array, selected at :1331 when index.n is 2 or 3; needs the dimension at compile time; computes identical numbers
cuvs/cpp/src/neighbors/ball_cover/registers.cuh	block_rbc_kernel_registers, perform_post_filter_registers, compute_final_dists_registers	:305, :64, :153	the k-NN query passes; the eps query is what DBSCAN needs and is what landed. Also gated upstream by ASSERT(index.n <= 3) at ball_cover.cuh:393,457
cuvs/cpp/src/neighbors/ball_cover/ball_cover.cuh	rbc_all_knn_query, rbc_knn_query, perform_rbc_query, compute_landmark_dists	:384, :446, :240, :501	the k-NN entry points; blocked on the three k-NN kernels above
cuvs/cpp/src/neighbors/ball_cover/registers_types.cuh	HaversineFunc, EuclideanFunc, compute_distance_by_metric	:37, :47, :79	never instantiated for the eps path; registers_00_generate.py:148 ships EuclideanSqFunc only
cub	DeviceRadixSort	cub/device/dispatch/dispatch_radix_sort.cuh	the device-wide sort thrust::sort_by_key and raft's sampleWithoutReplacement both call. Open and portable; the digit-histogram machinery is the same shape as the RAFT radix SELECT already at neighbors/gbdt/matrix/detail/select_radix.mojo. Not needed by ball_cover as built (a counting sort by landmark plus per-group rank gives the same order), but it is the general primitive this repo lacks and nn.argsort cannot stand in for it
```

---

## 4. PROPOSED `archive/reference/PORTING.md` DEVIATION ENTRIES (numbered from 30)

**30. `nn.argsort[target="gpu"]` IS WRONG ABOVE 256 ELEMENTS. DO NOT USE IT.**
Measured 2026-08-19 on the M4 with the pinned toolchain: correct at n=256,
non-monotone at n=257 and at every larger size tried (512, 1024, 1025, 1200,
4096), for `uint64`, `uint32` and `float32` keys and for both `int32` and
`int64` index outputs. The first inversion is always at output position 256,
which reads as a single-block sort with no merge across blocks. It raises
nothing and returns a well-formed permutation, so it fails silently. It is
listed as **GPU** in `archive/reference/VENDOR_LIBRARIES.md`'s availability table on the
strength of its signature (`ctx` + `target`), which is a test of whether the
GPU is *reachable*, not of whether the answer is *right*. That table needs a
correctness column. The probe is `bench/results/` section 6 of this file.
Also: `argsort` rejects a rank-2 `TileTensor` (`nn/argsort.mojo:547`
constraint), so it is 1-D only.

**31. `std.gpu.primitives.warp.vote` WORKS ON APPLE SILICON.** `archive/reference/PORTING.md 2`
corrected the claim that Mojo has no warp primitives and listed the ones
probed; `vote` was not among them. It is real and it works: probed on the M4
returning a `uint32` mask, and used in production in
`ball_cover/registers.mojo` as the port of `raft::ballot`. With
`std.bit.pop_count` and `std.bit.count_trailing_zeros` it covers CUDA's
`__ballot_sync` / `__popc` / `__ffs` triple, so **the whole ballot-and-walk
idiom that RAFT and CUB use everywhere is portable.** `__brev` + `__clz` has
no counterpart and needs none: it is an instruction-selection trick for
finding the lowest set bit, which `count_trailing_zeros` does directly.

**32. `vote` MAKES `block_dim` A CORRECTNESS PARAMETER, NOT A TUNING ONE.**
It returns a mask over the entire warp. A kernel that treats a 32-lane group
as its unit of work and is launched with a block larger than one warp will,
on 64-lane hardware, merge two groups' ballots into one mask and corrupt
both. `ball_cover/registers.mojo` launches 32 threads and one query per block
for exactly this reason, diverging from cuVS's `tpb = 64`. Any future port of
a `raft::ballot` kernel has to make the same decision explicitly.

---

## 5. FALSE DOC SENTENCES FOUND (in files I may not edit)

1. **`neighbors/UNPORTED.tsv`** says of `ball_cover`: *"an index that prunes
   the neighbourhood search. Brute force first, measure where it stops
   winning."* The measurement has been taken
   (`bench/results/FIRST_RUN_2026-08-19.md`: 37x behind scikit-learn at
   200k, exactly quadratic) and the index is now ported. **Delete the row**
   and replace it with the `PORTED_MAP.tsv` rows in section 3.

2. **`archive/reference/VENDOR_LIBRARIES.md`** lists `nn.argsort.argsort` as **GPU** with the
   verdict derived from its signature, and item **1** of A2 recommends
   `nn.toppminp_gpu.run_radix_sort_pairs_gpu` as "a GPU key/value radix sort
   [that] exists". The `argsort` row is not false about availability but is
   materially misleading about usability: **it is wrong above 256 elements**
   (deviation 30). Neither that table nor the A2 item distinguishes "the
   symbol exists and takes a `DeviceContext`" from "the answer is correct",
   and this lane is the second time in this repo that distinction has cost
   something. Recommended edit: add a `checked?` column, mark `argsort`
   **BROKEN >256, measured 2026-08-19**, and mark
   `run_radix_sort_pairs_gpu` **UNVERIFIED** since it is recommended on the
   same evidence (a signature) and has never been run.

3. **`cuvs/include/cuvs/neighbors/ball_cover.hpp:57-60`** (upstream, not ours,
   recorded because the port relies on it): their footprint comment
   `(2 * sqrt(m)) + (n * sqrt(m)) + (2 * m)` undercounts by `X_reordered`,
   another `n * m`, allocated at `:68`, six lines below. The real footprint
   is dominated by a second full copy of the dataset. Anyone sizing a device
   for this index from their comment will be short by `n * m` floats.

---

## 6. BUILD AND CHECK EVIDENCE

### Build

```
cd /Users/andrewhendel/CascadeProjects/mojolearn
tools/with_build_lock.sh pixi run \
  --manifest-path /Users/andrewhendel/CascadeProjects/mojotrees/pixi.toml \
  mojo build -I . neighbors/ball_cover_main.mojo -o /tmp/ball_cover_probe
```
Exit 0, no output, no warnings.

### Checks

```
$ /tmp/ball_cover_probe
ball_cover: exact set match at eps 0.9 / 2.5 / 8.0, edges 5132 73050 1014290
ball_cover: dense and max_k both match brute force at eps 1.6, longest row 39
ball_cover: sabotage reached; radii x0.3 took edges from 73050 to 62612 and every survivor is still a true neighbor
ball_cover: the in-group order is load bearing; reversing it took edges from 73050 to 69189
ball_cover: n=8000 d=5, 200 sampled rows match brute force exactly; 130546 edges over 89 landmarks
0.65s user 0.02s system 68% cpu 0.976 total
```

What each line asserts:

1. **n=1200, d=3, three radii, exact set equality per row** against a host
   brute force that repeats `eps_dist_sq` in Float32 in the same accumulation
   order over the same operand bits. Not a count — the check marks every
   returned column and then walks all 1200 candidates per row, failing on the
   first extra AND on the first missing one. It also asserts `adj_ia` is
   monotone, `adj_ia[n] == nnz`, `vd[i]` equals each row length, `vd[n]`
   holds the total (which is the value `vertexdeg/algo.cuh:150` reads back),
   and that no column repeats in a row. The fixture is splitmix64-hashed, so
   every row's neighbor SET is different and a misplaced row cannot pass on
   totals. Before any query it also asserts the index itself: `R_indptr[L]==m`,
   every point in exactly one group, each group ascending in `R_1nn_dists`,
   `R_radius[k]` equal to its group's last distance, and `X_reordered[j]`
   byte-equal to `X[R_1nn_cols[j]]`.
   The three radii give 5132 / 73050 / 1,014,290 edges out of 1.44M ordered
   pairs, i.e. 0.36% / 5% / 70% density, which is the small/medium/near-total
   spread the brief asked for.
2. **The other two output shapes**, `block_rbc_kernel_eps_dense` and
   `block_rbc_kernel_eps_max_k`, against the same host oracle, plus that the
   returned `actual_max` equals the CSR's longest row.
3. **Sabotage 1 — the landmark prune.** Every `R_radius` scaled by 0.3. Edges
   73050 → 62612, and the surviving rows are still asserted to be a SUBSET of
   the true answer (an extra edge would still fail). That is the predicted
   shape: a smaller radius can only make `d(q,r) <= eps + radius(r)` reject
   balls it should accept, never accept ones it should reject. A no-op change
   or an unread `R_radius` would leave the count identical.
4. **Sabotage 2 — the backward walk.** Each group's `R_1nn_dists` reversed in
   place, `R_1nn_cols` and `X_reordered` untouched, so the group is described
   as descending while it is physically ascending. Edges 73050 → 69189, again
   a strict subset. This is a *different* branch and a *different* input from
   sabotage 1: passing sabotage 1 says nothing about whether the early stop
   is reached, and vice versa.
5. **n=8000, d=5**, 200 strided rows against the full oracle. This exists
   because the `argsort` bug below was invisible at every size below 257, and
   this port has three size-dependent pieces: the scan's dynamic chunk, the
   per-landmark rank kernel, and the per-warp CSR offsets.

No timing was run. This lane took no benchmark.

### The `nn.argsort` finding, with its probe

The first version of `construct_landmark_1nn` used `nn.argsort[target="gpu"]`
on a 64-bit composite key. The index-invariant check failed with *"landmark 0
group is NOT sorted ascending at position 18"*. The counting histogram and the
scan were both correct; the sort was not. Isolated:

```mojo
keys[i] = (UInt64(i % 34) << 32) | UInt64((i * 977) % 100000)
argsort[target="gpu"](TileTensor(perm, row_major(m)),
                      TileTensor(keys, row_major(m)), ctx)
# then assert keys[perm[j]] is non-decreasing in j
```

```
uint64  m =   64  monotone: True
uint64  m =  128  monotone: True
uint64  m =  256  monotone: True
uint64  m =  512  monotone: False   first inversion at output index 256
uint64  m = 1024  monotone: False   first inversion at output index 256
uint64  m = 1025  monotone: False   first inversion at output index 256
uint64  m = 1200  monotone: False   first inversion at output index 256
uint64  m = 4096  monotone: False   first inversion at output index 256
uint32  m = 1200  monotone: False
uint32  m = 4096  monotone: False
float32 m = 1200  monotone: False
float32 m = 4096  monotone: False
int64 index output, float32 keys, m = 256  monotone: True
int64 index output, float32 keys, m = 257  monotone: False
```

Rank-2 input is rejected outright (`nn/argsort.mojo:547` constraint failed),
so a `(1, m)` batched shape is not a workaround.

**Two things follow.** First, `archive/reference/PORTING.md 30` above. Second, and this is the
part worth arguing about: substituting this call would have shipped an index
whose groups are unsorted, which makes the query kernel's early stop drop
real neighbors, which makes DBSCAN produce a plausible and wrong clustering.
It was caught only because the check asserts an INDEX INVARIANT — group
ordering — and not merely the final answer. A check that only compared
neighborhoods at the fixture's eps would likely have passed: the early stop
fires on chunk boundaries, so at a small eps a partially-sorted group often
still returns everything.

### Warp primitive probe

`vote`, `lane_id`, `shuffle_idx`, `warp.sum`, `pop_count`,
`count_trailing_zeros` and `WARP_SIZE` were probed together in one kernel
before any of them was used. `WARP_SIZE == 32` on this M4. `vote` returns a
correct `uint32` ballot mask. This is `archive/reference/PORTING.md 31` above.

---

## 7. WHAT I DID NOT DO, AND WHY

- **Did not wire it into DBSCAN.** Forbidden by the brief; another lane owns
  `dbscan/`. Instructions in section 8.
- **Did not port the k-NN query passes** (`block_rbc_kernel_registers`,
  `perform_post_filter_registers`, `compute_final_dists_registers`). Brief
  priority 3, contingent on 1 and 2 landing with room left. They landed; the
  room went into finding and working around the `argsort` bug, into the
  second sabotage, and into the scale check. Recorded as unported with
  reasons in section 3. Note that upstream gates the k-NN entry points on
  `ASSERT(index.n <= 3)` (`ball_cover.cuh:393,457`), which the eps path does
  not have, so the k-NN passes are a narrower feature than they look.
- **Did not port `block_rbc_kernel_eps_csr_pass_xd`** — D9, an optimization
  with identical arithmetic.
- **Did not port CUB's `DeviceRadixSort`.** It is the right general answer to
  the missing device sort and it is a lane of its own. Recorded.
- **Did not run any timing.** Forbidden; and this box drifts 2-3x across
  thermal windows so only interleaved arms in one window compare.
- **Did not touch `core/**` — not `gemm.mojo`, not `column_stats.mojo`, not
  anything else in it.** Asked directly by the orchestrator on 2026-08-19:
  **no.** This lane imports nothing from `core/`; `grep -rn 'core\.' ` over
  all four of my files returns nothing, and the fused argmin in D2 exists
  precisely so that `k_closest_landmarks` does NOT go through a GEMM. The
  full check suite was rebuilt and re-run after that question and is green,
  so whatever happened to those kernels did not break this lane and this lane
  did not cause it.
- **Did not edit any shared file.** `archive/reference/PORTING.md`, `archive/reference/VENDOR_LIBRARIES.md`,
  `neighbors/PORTED_MAP.tsv`, `neighbors/UNPORTED.tsv`,
  `neighbors/README.md`, `core/**`, `dbscan/**` are all untouched;
  everything that would have gone in them is in sections 3, 4 and 5.
- **Did not build a CPU path.** Per the mid-lane correction, there is none.
  The host brute force in the check is an ORACLE — it is only ever used to
  decide whether the device answer is right, and nothing in the port calls
  it.

---

## 8. WHAT THE ORCHESTRATOR HAS TO CALL TO WIRE THIS INTO DBSCAN

The insertion point is `cuml/cpp/src/dbscan/vertexdeg/algo.cuh:226-231`: the
`if (data.rbc_index != nullptr)` branch. The other lane's
`dbscan/gbdt/neighbors/epsilon_neighborhood.mojo` is the `else` branch of
that same `if`, so the two are complementary and neither replaces the other.
`runner.cuh:234-242` builds the index ONCE, before the batch loop.

Build once per fit, mirroring `runner.cuh:237-239`:

```mojo
from neighbors.ported.neighbors.ball_cover.ball_cover import (
    rbc_build_index, rbc_eps_nn_query_count, rbc_eps_nn_query_fill,
    rbc_n_landmarks,
)

var L = rbc_n_landmarks(n_rows)
# device buffers: r[L*d], x_reordered[n*d], landmark_ids[L], slot_cols[n],
# slot_dists[n], nearest[n], nearest_dist[n], r_indptr[L+1], r_1nn_cols[n],
# r_1nn_dists[n], r_radius[L], counts[L]
rbc_build_index(ctx, x, r, x_reordered, landmark_ids, slot_cols, slot_dists,
                nearest, nearest_dist, r_indptr, r_1nn_cols, r_1nn_dists,
                r_radius, counts, n_rows, n_features, L)
```

Then per batch, mirroring `algo.cuh:137-163` exactly — count, read the total
back, size `ja`, fill:

```mojo
var nnz = rbc_eps_nn_query_count(
    ctx, x_reordered, query_rows, r, r_indptr, r_1nn_cols, r_1nn_dists,
    r_radius, ex_scan, vd, n_batch, n_features, L, Float32(eps))
# allocate col_ind with nnz entries (cuml resizes data.ja here)
rbc_eps_nn_query_fill(
    ctx, x_reordered, query_rows, r, r_indptr, r_1nn_cols, r_1nn_dists,
    r_radius, ex_scan, col_ind, n_batch, n_features, L, Float32(eps))
```

Four things the wiring must get right, all of them theirs:

1. **`eps` is the RADIUS, not its square.** `algo.cuh:227` passes `data.eps`
   into `eps_nn` while the brute-force arm on the next line gets `eps2`. The
   query kernel squares it internally, once. The existing
   `dbscan/gbdt/dbscan/runner.mojo` squares `eps` on the host before
   calling `vertexdeg`; that value must NOT be the one passed here.
2. **`vd` must be `n_batch + 1` long.** `vd[n_batch]` receives the edge total
   and is what `algo.cuh:150` reads back with `update_host`.
3. **`ex_scan` is `adj_ia` directly.** The count pass writes offsets into it,
   so DBSCAN's separate exclusive-scan step
   (`dbscan/gbdt/dbscan/adjgraph/algo.mojo`) is NOT run on this path —
   `algo.cuh` has no `adj_to_csr` in the rbc branch because the query emits
   CSR itself. Running both would double-scan.
4. **RBC is only valid for L2.** `runner.cuh:152-153` disables it for any
   metric other than `L2SqrtExpanded` / `L2SqrtUnexpanded`. Cosine goes
   through `algo.cuh:186-207`, which normalizes the rows in place first and
   then uses the same eps query, so it is reachable but needs that
   normalization step, which is not in this lane.

The `max_k` form (`rbc_eps_nn_query_max_k`) is the one-pass alternative
`algo.cuh:122-135` takes when the caller can bound the row length; it needs
an `n_batch * max_k` scratch buffer and returns the longest row found, which
the caller must check against the bound it passed (theirs asserts at `:135`).
It is ported and checked but is the second thing to wire, not the first.
