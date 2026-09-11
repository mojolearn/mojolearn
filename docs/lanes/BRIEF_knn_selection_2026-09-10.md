# BRIEF: kNN selection (top-k) under IDENTICAL, DEVIATION 2496

Source-analysis pass, September 10, 2026. Nothing here was built or run on
the Mac; no kernel was modified. Two new tools and this brief are the
output. The qualified state this brief starts from: bare kNN search,
400,000 index rows x 32 features, 4,000 queries, k 10 and 15, IDENTICAL,
NVIDIA H100, request 26.66 ms (k10) and 31.13 ms (k15), 2.61x and 2.88x the
cached cuML rows (`bench/OPPONENT_REFERENCE.md`, "Sep10 NVIDIA aligned kNN
loads"); aligned loads admitted as the default on that scope.

Standing rules applied: every candidate below preserves the composite
(distance bits, index) total order, the pop sequence and the output bits;
the gate for a candidate is bitwise equality plus reach by sabotage, then
an ordinary-request price on the large shape; cuML is not rerun (the tuple
is cached); a tile or kernel win alone cannot promote a request default.

## 1. Kernel map: the bare search request, IDENTICAL, NVIDIA column

Entry chain (host):

| step | where | what it decides |
|---|---|---|
| Python | `python/mojolearn/neighbors.py:475` `NearestNeighbors.kneighbors`, `:546` calls `_mojolearn.knn_search` with `[n_index, n_queries, d, k, return_sqrt=1, query_tile]` and `[metric, metric_arg, weights]` | `as_f32_c` on the queries (index converted once at `fit`); outputs `dist <f4`, `ind <u4` then `ind.astype('<i8')` (a C-level `array.array` widening of `n_queries*k` values, inside the timed public boundary) |
| binding | `bindings/_mojolearn.mojo:182` `knn_search_binding` -> `:230` `knn_search(ctx, ..., KNN_METHOD_AUTO, metric, arg)` under `GILReleased` | one `DeviceContext()` per call |
| estimator | `neighbors/estimator.mojo:300` `knn_search` -> `:373` `_knn_search_traced_retaining`; `:252` `plan_query_tile` (NVIDIA IDENTICAL default 512, `:231-236`; budget 768 MiB on `query_tile x identical_index_tile`); `:477` `buf_len = n_index // 8`; `:481-501` allocations (index, queries, norms, `dist_tile = query_tile x 65536`, radix scratch `2 x query_tile x buf_len` floats and uint32 even though the smallk selector never reads it, outputs); `:503-505` two uploads; `:522-525` norms; `:557` `brute_force_knn_impl` | 400k/4k: query batches 512 (8 batches, last 416 rows), index partitions 65,536 (7, last 6,784 columns) |
| dispatch | `neighbors/impl/detail/knn_brute_force.mojo:1122` `brute_force_knn_impl`; `:1233` AUTO pinned to TILED under IDENTICAL (DEVIATION 509, every column) | the fused FAISS-queue arm is never entered under IDENTICAL |
| layout | `:393` `tiled_brute_force_knn`: `EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL` (row `knn_transposed_index_for`, on for NVIDIA/AMD/AMD_RDNA/Apple since 2026-09-09, `checks/kernel_matrix.mojo:834-864`) allocates `n_index x d` and launches `transpose_kernel` `:431` (grid `(d/32, min(n_index/32, 65535))`, block 32x32) | 0.075 ms per request (phase logs) |
| loop | `:470` `_tiled_brute_force_knn_impl`; `:543` `index_tile = identical_index_tile(n_index)` (row `knn_index_tile_columns_for` = 65,536); `:560` `use_vector` (NVIDIA and EXACTLY `n_index == 400000 and n_queries == 4000 and d == 32 and k in {10, 15}` and L2SqrtExpanded); `:601` query loop, `:604` partition loop, `:607` remainder-shorter-than-k carve; `:1026` one `synchronize` at the end | 56 distance launches, 56 selections, 48 merges per request |

Device kernels on the path, in launch order per (query batch, index partition):

| kernel | file:line | launch shape | shared memory | per-thread state | what selects it |
|---|---|---|---|---|---|
| `row_norm_kernel` | `core/row_norms.mojo:70`, launched `knn_brute_force.mojo:253` | grid `(n_rows)`, block `NORM_TPB` (`lib_block_size_for[K_LIB_ROW_NORM]`) | the pinned block sum's scratch (`core/pinned_reduce.mojo`) | one Float32 accumulator, serial `ftz(fma)` over its column stride, then a halving tree | every L2 request, once for index and once for queries (0.25 ms) |
| `transpose_kernel` | `core/column_stats.mojo`, launched `:431` | `((d+31)/32, min((n_index+31)/32, 65535))`, block `(32, 32)` | 32x32 tile | none | transposed row on |
| `pinned_distance_register_tile_kernel[METADATA=False, VECTOR=True]` | `neighbors/checks/pinned_distance_tile.mojo:281`, launched `knn_brute_force.mojo:755` | grid `((cols+511)/512, (rows+7)/8, 1)`, block `RT_TPB=128`; each thread owns `RT_ROWS=8` query rows (`knn_distance_rows_for`, NVIDIA) x `RT_COLS=4` index columns | none | `acc[32]` Float32, `rows_idx[8]`, `cols_idx[4]`; per feature: one 16-byte vector load of `yt`, 8 scalar loads of `q`, 32 `fma.rn` + `mul.rn.ftz` (`_rt_step`, `knn_distance_hardware_flush_for` = NVIDIA) | `use_vector` and `n_index % 4 == 0 and c % 4 == 0 and cols % 4 == 0`; otherwise `[False, False]` (`:773`) with 4 scalar `yt` loads; Apple takes `[True]` metadata or the repair chain |
| `smallk_bucket_kernel[CAP=16, K=10 or 15]` | `neighbors/checks/select_smallk_identical_candidate.mojo:254`, launched `:369` / `:376` from `smallk_select_launch` (`:348`), called at `knn_brute_force.mojo:913` | grid `(rows, 1, 1)` = one block per query row of the tile, block `SMALLK_BLOCK=256` | `heads`: 256 x UInt64 = 2 KiB (the butterfly form uses 2 x 8 slots of it) | `local_keys` SIMD[uint64, 16] (32 registers), `threshold` UInt64, `batch` SIMD[float32, 8], `col`, `base`; scan reads `values[row*length + tid + u*256]` (coalesced 128 B per warp per `u`) | rows `knn_smallk_select_for` (on, all four columns) and `knn_selector_specialize_common_for` (NVIDIA only: comptime K); `knn_selector_shuffle_for` (fixed-lane-width columns) picks the butterfly rank loop `:299-321` over the shared tree `:323-346`; k > 64 or `-D MOJOLEARN_KNN_IDENTICAL_LEGACY_SELECT` falls to `radix_topk_identical_kernel[256]` (`select_radix_identical.mojo:136`, launched `:923`) |
| `partial_topk_merge_kernel` | `select_smallk_identical_candidate.mojo:418`, launched `:956` for every partition after the first | grid `(rows)`, block `MERGE_BLOCK=256` | STATIC 2 x 1024 UInt64 keys + 2 x 1024 Float32 vals = 24 KiB per block regardless of k (`MERGE_MAX_K`) | rank = own position + count of the other list's keys below it; O(k^2) per row, in place into `out_dist/out_idx` | tiled index axis (always at 400k) |
| `radix_topk_identical_kernel` / `warpsort_topk_block_kernel` / `radix_topk_one_block_kernel` / `nn.topk.top_k` | `select_radix_identical.mojo:136`; `neighbors/impl/matrix/detail/select_warpsort.mojo:792`; `select_radix.mojo:173`; vendor | | | | NOT on this path: radix only for k > 64 or the legacy define; warpsort and one-block radix are the FAST arms (`PIN_DETERMINISM` false); `use_vendor_topk` is a check-only second opinion. `neighbors/warpsort_probe_main.mojo` is a probe for the FAST warpsort and says so in its docstring |
| `fused_l2_knn` (FAISS warp queue, `neighbors/impl/detail/fused_l2_knn.mojo`, `neighbors/impl/topk/warp_topk.mojo:100` `WarpSelect`, `bitonic.mojo`) | | | | | NOT reachable under IDENTICAL (DEVIATION 509); its tie rule differs (distance only, arrival order) |

Reach and arm switches that exist today are ALL compile-time defines
(`-D`), enumerated at `checks/kernel_matrix.mojo:844-987` and
`neighbors/estimator.mojo:231-236`: `MOJOLEARN_KNN_IDENTICAL_LEGACY_SELECT`,
`_LEGACY_LAYOUT`, `_SCALAR_TILE`, `_NO_INDEX_TILE`, `_GENERIC_K`,
`_SPECIALIZE_COMMON`, `_TREE_SELECT`, `_ROWS4`, `_SOFTWARE_FLUSH`,
`_NO_ZERO_FMA_REPAIR`, `_NO_PREFLIGHT`, `_NO_METADATA`, `_QUERY_TILE_512`,
`MOJOLEARN_KNN_LEGACY_QUERY_TILE`, `MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL`,
`MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL`,
`MOJOLEARN_EXPERIMENTAL_KNN_PREFLIGHT_METADATA`, `MOJOLEARN_KNN_PHASE_TIMERS`.
The only RUNTIME switch is `MOJOLEARN_KNN_VECTOR_TRIAL` (`knn_brute_force.mojo:561-565`),
and it is read only in a build with `-D MOJOLEARN_KNN_VECTOR_REQUEST_CHECK=1`.
That build-define-plus-runtime-env pattern is what the selection trial
hook below copies, because the protocol wants both arms in ONE process.

## 2. Where the selection's time goes (k10 and k15, 400k x 4k, d32)

Measured split, H100, phase timers (serializing; the split is the
measurement), `bench/results/knn_loads_2026-09-10/raw/knn-loads/phase-k10.log`
and `phase-k15.log`, 56 distance and selection launches, 48 merges,
batch 512, partition 65,536:

| class | k10 ms | k15 ms | per launch |
|---|---:|---:|---:|
| distance (register tile) | 16.21 | 16.23 | 290 us |
| selection (`smallk_bucket_kernel[16, K]`) | 10.30 | 14.61 | 184 us / 261 us |
| merge | 0.45 | 0.46 | 9 us |
| transpose + norms | 0.33 | 0.33 | |

Selection is 38% of the k10 request and 47% at k15, and the k15 selection
is 42% dearer than k10 with IDENTICAL loads, tile and merge. The k-slope is
(14.61 - 10.30) / 56 / 5 = 15.4 us per launch per unit of k; the
extrapolated k -> 0 intercept is about 30 us per launch, of the order of
the 128 MiB tile read (45 us at 3 TB/s; the tile is not L2-resident). So
roughly 84% (k10) to 89% (k15) of the selection is k-proportional work,
and that work is in the scan, not the rank phase. Reading the kernel:

The scan (`:285-297`). Each thread walks 256 elements (65,536 / 256) as 32
unrolled batches of 8 loads. Per element: one load, `composite_key`
(`twiddle_in`, about 5 instructions), a UInt64 compare against
`threshold`, and, when it passes, `_smallk_insert` (`:234`): a K-step
carry chain of UInt64 compare-and-swap (about 6 instructions per step,
every step executed because the tail shifts) plus the threshold refresh.
A lane inserts its i-th element with probability about k/i on unordered
data (expected 42 inserts per lane at k10, 58 at k15). BUT THE WARP PAYS
PER STEP, NOT PER LANE: the insert is a divergent branch, and the chance
that at least one of 32 lanes inserts at step i is 1 - (1 - k/i)^32, which
is 0.996 at i = 64, 0.92 at i = 128 and 0.72 at i = 256 for k10. So a warp
executes the K-step chain on about 230 of its 256 steps at k10 and about
240 at k15. Estimated instructions per lane: 256 x 10 (loads, keys,
compares) + 230 x 60 = about 16k at k10; + 240 x 90 = about 24k at k15.
Ratio 1.47 against the measured 1.42, and the non-k share (2.5k / 16k =
16%) against the measured 16%. The model is consistent with the split; it
is still a model, and the profile phase in section 5 (select_ms at k = 1,
2, 5, 10, 15 on one generic bucket) is the measurement that confirms or
kills it before any kernel moves.

The rank phase (`:299-321`, butterfly form). k rounds of: `shuffle_min_u64`
(5 xor steps x 2 halves = 10 shuffles), one shared store, ONE barrier, 8
shared loads and 7 compares, thread 0's two output stores, and a
predicated 15-slot shift on the winning lane. About 90 instructions and
one barrier per round: 900 to 1,350 per lane, 3 to 6% of the scan. Not
the driver.

Memory traffic per launch: one read of the 512 x 65,536 x 4 = 128 MiB
tile (coalesced), 512 x k x 8 bytes out, k gathers of `values[selected]`
per row. The distance tile writes those 128 MiB just before. No candidate
buffer is materialized; the per-lane list lives in registers (every index
comptime, so no local-memory spill in the specialized form, which is what
the Sep 9 "20.3 -> 9.1 ms" specialization bought).

What is order-sensitive (must not move), and what is pure data movement:

- ORDER-SENSITIVE: the composite key (`twiddle_in(distance) << 32 | index`,
  `select_radix_identical.mojo:102`), the fact that every comparison is an
  unsigned 64-bit compare on that key (this is what orders signed zeros,
  subnormals, infinities and NaN payloads the same on every vendor), the
  pop rule (the block's exact minimum key each rank), the output value
  read back from the original tile cell (`values[row*length + selected]`,
  bits untouched), and the merge's rank rule over global indices.
- PURE DATA MOVEMENT: which lane holds which candidate, how many
  candidates a lane holds, when a candidate is dropped PROVIDED at least k
  keys smaller than it are known to exist somewhere in the block, the
  order in which a lane's list is maintained (sorted or not), the shape of
  the block minimum (tree or butterfly), the merge cadence and the barrier
  count. Any change confined to this list yields the same k keys in the
  same order, because the rank phase pops exact minima of a union that
  still contains the true top-k.

## 3. Candidates, one at a time, ranked

Estimated effects use the section 2 model and are to be replaced by the
gate's numbers. "Events" means warp-level executions of the insertion chain.

### C1 (top): block head-bound rejection, `headbound`

Mechanism. At a fixed cadence inside the unrolled batch loop (after batch
1, then after batches 4, 12 and 28; four refreshes), every thread publishes
its current head `local_keys[0]` (sentinel if empty) into the existing
`heads` shared array; the block computes the k-th smallest of those 256
heads with the SAME machinery as the rank phase (k rounds of butterfly
warp-minimum, 8 slots, one barrier per round, the winning lane's
contribution replaced by the sentinel), and every thread sets
`gate = min(threshold, bound)`. The scan's reject test becomes
`pending < gate` instead of `pending < threshold`; on an insert the gate
is refreshed as `min(new threshold, bound)`.

Why the bits are unchanged. `bound` is the k-th smallest of a SUBSET of the
union of lists, so at least k keys of the union are at most `bound`; a
pending key at or above it cannot be among the block's k smallest and is
dropped exactly as a key at or above the lane's own k-th is dropped today.
The union therefore still contains the true top-k, the rank phase pops
exact minima, the output cell is still read from the tile. When fewer than
k real heads exist (rows shorter than one batch: the carved k-wide tail
partition never enters the batch loop) `bound` is the sentinel and the
kernel is the baseline kernel.

Why it pays. The 256 heads are 256 per-lane minima, so their k-th smallest
sits near the k/(256 i) quantile, close to the union's true k-th, whereas
the lane's own threshold sits at k/i: the per-lane pass rate drops by two
orders of magnitude and the warp's event count from about 230 to about
18 (8 in batch 1 before the first bound, then 1 - (1 - 0.0049)^32 = 0.145
per step until the second refresh, then about 0.02). Cost: four refreshes
of k rounds at about 45 instructions plus one barrier each (1.8k at k10,
2.7k at k15). Model: 16k -> 5.4k instructions per lane at k10 (selection
10.3 -> about 3.5 ms), 24k -> 6.8k at k15 (14.6 -> about 4.2 ms); request
26.7 -> about 20 ms and 31.1 -> about 21 ms, that is about 2.0x the cached
cuML rows instead of 2.6x/2.9x. Larger gain at k15.

Risk. Barriers inside the batch loop require a BLOCK-UNIFORM trip count;
the current loop condition `col + 7*256 < length` (`:285`) is per-thread and
diverges for partition lengths with (length - 1792) mod 2048 in 1..255
(the `divergent_tail` fixture, 3,940-column tail, is built for exactly
this); C4 below is the prerequisite. The refresh's rounds reuse the rank
phase's parity double-buffering; the rank phase after the scan must start
from a known parity. If the section 2 model is wrong (the scan is
load-latency-bound as the 2026-09-09 comment claims), the gain shrinks to
the refresh cost; the profile phase decides first.

### C2: warp head-bound, `warpbound`

Same idea at warp scope: the k-th smallest of the 32 lane heads via k
rounds of `shuffle_min_u64` with removal, no barrier, no shared memory,
refreshed every 4 batches. Bound near k/(32 i): events about 42 at k10
(model: 16k -> about 8.5k). Order argument identical to C1 (subset of the
union). Lower gain, no barrier in the loop, still needs C4 (shuffles must
be convergent). The fallback if C1's barriers cost more than modeled.

### C3: cross-batch compaction of passing candidates, `compact`

Per unrolled batch, lanes ballot their passing elements per `u`, write
them compacted (prefix popcount) into a warp-private shared staging area
(256 x UInt64 per warp, 16 KiB per block), and the warp inserts them in
ceil(P/32) rounds of one element per lane rather than in one chain per
`u` with any passer. Rounds over the scan about 55 (k10) and 73 (k15)
against 230 events; overhead about 64 instructions per batch. Model: 2x
at both k. Pure data movement (an element inserted into ANOTHER lane's
list is dropped there only if that lane holds k smaller keys, which is
the same guarantee). Orthogonal to C1; combine after C1 lands. More
machinery (warp-synchronous shared staging needs the warp sync primitive
on Volta+ and its Apple/AMD spellings), so third.

### C4 (prerequisite, zero-effect): block-uniform batch trip count

Change the unrolled loop's condition to `batch_base + 2048 <= length` with
`col = batch_base + tid`, and let the per-lane tail loop (`:294`) take the
rest. Every lane still visits the same columns in the same ascending
order; only the split between batched and tail processing moves for
lengths with (length - 1792) mod 2048 in 1..255. No performance claim; it
must land and be gated ALONE (arm equality on `divergent_tail`) before C1
or C2, because a warp collective or barrier inside a divergent-trip-count
loop hangs or garbles.

### C5 (deferred): warp bitonic merge for the rank phase

Replace k block-wide pops with per-warp bitonic merges of the lanes'
sorted lists followed by an 8-list rank merge. Exact integer compares, so
order-preserving, but the rank phase is 3 to 6% of the selection; not
worth a gate before C1 is measured.

### C6 (deferred, large): fused distance + select tile

Select directly from the register tile's `acc` values inside the distance
kernel (per-block partial top-k, then the existing merge), removing the
128 MiB tile write and read per launch (about 85 us of 474 us). The
distance bits are the same cells; the merge is the existing rank merge;
order-preserving in principle, but it restructures both kernels and the
merge count (one per distance block rather than one per partition). Not a
one-at-a-time candidate; recorded so it is not rediscovered.

### Not candidates / REJECTED

- Per-thread register queue size (CAP): K is folded at compile time on
  NVIDIA, CAP=16 is already the smallest bucket; changing CAP changes no
  work. Not a lever.
- k padding to a lane multiple: k is a comptime constant in the
  specialized kernel; nothing to pad.
- Deeper load unroll (16, 32): tried and rejected by the 2026-09-09 lane
  (`docs/lanes/HANDOFF_knn_selector.md`).
- Fewer threads per row alone (128 per row): the warp still pays the
  chain on nearly every step (P(any lane) stays near 1), so per-row work
  is unchanged while parallelism halves. Only meaningful after C1.
- REJECTED (order-changing): comparing Float32 distances with `<` and
  breaking ties by arrival (changes NaN/signed-zero/tie order); keeping
  only the twiddled high half in the block minimum (drops the index
  tie-break); a value-only radix with atomically ordered index emission
  (the DEVIATION 500/501 tie class returns); any bound that is not an
  upper bound on the union's k-th smallest (for example a median of lane
  thresholds), which would drop true neighbors.

## 4. The trial hook the follow-on lane must wire (not done in this pass)

Build define `-D MOJOLEARN_KNN_SELECT_TRIAL=1` (never on a shipped build).
Under it, `_tiled_brute_force_knn_impl` reads `MOJOLEARN_KNN_SELECT` once
per request (the `MOJOLEARN_KNN_VECTOR_TRIAL` pattern at
`knn_brute_force.mojo:561`): `baseline` forces the current kernel,
`headbound` (and later `warpbound`, `compact`) forces the candidate, unset
takes the kernel-matrix default (a new SCHEDULING row
`knn_selector_head_bound_for[column, identical]`, off until the gate
passes; flipped in the same session as the measured win), any other
value RAISES. `MOJOLEARN_KNN_SELECT_SABOTAGE=1` instantiates the chosen
kernel with `SABOTAGE=True`: XOR bit 0 of the composite key's index half
for `u == 0` of every batch. That corrupts about one candidate column in
eight on both arms, so the selected index and the gathered distance both
move for thousands of winners on the large shape; the gate requires the
flip on every arm and on the default, and requires the next clean call to
restore the reference bits. The kernel signature becomes
`smallk_bucket_kernel[CAP, K, BOUND: Bool = False, SABOTAGE: Bool = False]`;
without the define only the two existing instantiations exist.

Native check owed with the kernel change: extend
`neighbors/checks/knn_selector_long_rows_check.mojo` (exhaustive host
order statistics over 8192/8193/65537-long rows, all buckets, duplicates,
signed zeros, subnormals, infinities, NaN payloads) with the `BOUND=True`
instantiation, plus lengths 3,940 and 65,536 + 3,940 for C4.

## 5. The gate: `tools/knn_selection_gate.py` and `tools/knn_selection_gate.sh`

`tools/knn_selection_gate.py` (Python, one process, public API only:
`NearestNeighbors(n_neighbors=k).fit(X).kneighbors(Q)`, outputs read via
`__array_interface__`):

- fixtures from `--seed` (sha256 of every array in the JSON): `large`
  (400k x 32 / 4k, splitmix64-hashed non-uniform cells, per-feature
  scales, clustered offsets, distinct rows asserted, 16 exact-match queries
  at partition boundaries), `dyadic` (the opponent's dyadic-v1 bytes,
  imported from `tools/knn_cuml_reference.py` so they cannot drift),
  `ties` (131,079 = 2 x 65,536 + 7 rows, the remainder-shorter-than-k
  carve; 40 anchors planted three times each including a 65,535/65,536
  straddle and the carved tail; 40 exact-match queries, 40 offset queries;
  700 queries so the second batch is ragged), `divergent_tail`
  (397,156 = 6 x 65,536 + 3,940 rows; the per-thread trip-count shape).
- correctness per fixture and k: bitwise equality of distances and
  indices across `baseline`, every candidate and the DEFAULT (env unset);
  per-row pinned order (non-decreasing distance, equal distances ascending
  by index, no repeats, in range, no NaN); planted expectations (exact
  matches at rank 0 by index and equal bits across a duplicate group;
  whether exact-match bits are exactly zero is REPORTED, not asserted,
  because the expanded form sums the norm through a block tree and the
  product through a serial chain, so an exact match is not guaranteed a
  zero on non-dyadic data); an independent float64 oracle (complete on
  `ties`, sampled on the others) with float32-resolution ambiguity counted
  separately from hard mismatches.
- reach per arm and default: sabotage must flip at least one cell and the
  next clean call must restore the reference bits; a build without the
  hook fails here by design.
- timing on `dyadic` and `large` at k10 and k15: one warmup per arm,
  `--pairs` pairs in order (A, B) then `--pairs` in order (B, A); every
  sample kept, medians and minima per arm per order and pooled, paired
  ratios, pairs favoring B, spread; every timed output compared to the
  clean reference outside the timed region; `--cached-opponent` reports
  the cached-reference ratio on the `dyadic` rows only, labeled as such.
- hard deadline: `--deadline 300` via SIGALRM plus a backstop timer;
  partial JSON on expiry, exit 3. Exit 1 on any failure, 0 on pass.
- `--selftest --quick` exercises the checker with a numpy stand-in and no
  GPU; run on the Mac at tiny sizes on 2026-09-10: all checks pass, and
  with an unwired sabotage switch the gate fails with "REACH NOT PROVEN",
  as it must.

`tools/knn_selection_gate.sh` (POSIX sh, the leg's EXTRA hook): phase
`profile` builds `bench/knn_reference_price_main.mojo` under IDENTICAL with
phase timers, once with `-D MOJOLEARN_KNN_IDENTICAL_GENERIC_K=1` and runs
k = 1, 2, 5, 10, 15 (one bucket, so select_ms against k is one kernel's
slope) and once with the shipped specialization at k10/k15; phase `build`
compiles `python/mojolearn/identical/_mojolearn.so` with
`-D MOJOLEARN_KNN_SELECT_TRIAL=1` (mirrors `bindings/build.sh`'s Linux
command; that script has no extra-define hook); phase `gate` runs the
harness under `timeout 420`. Outputs under `/root/gemm_leg_out/knn-selection/`
(`status.tsv`, `profile_summary.tsv`, `profile-*.log`, `binding.sha256`,
`knn_selection_gate.json`, `summary.txt`, `gpu_before/after.csv`). Binaries
are hashed, never copied home.

## 6. RUN OWED (orchestrator; nothing here ran on a GPU)

The leg ships `git archive` of the COMMITTED tree: commit the two tools and
this brief first. The H100 is named so the cached cuML tuple applies; the
gemm payload's own step 9 makes the leg exit 1 (structural card diff, see
`docs/lanes/HANDOFF_gemm_splitk.md`), so the verdict is the fetched
`remote/knn-selection/summary.txt` and `status.tsv`.

```
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
MOJOLEARN_GEMM_LEG_EXTRA=tools/knn_selection_gate.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-knn-selection \
sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" \
    --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
cat <leg>/remote/knn-selection/profile_summary.tsv
cat <leg>/remote/knn-selection/summary.txt
cat <leg>/remote/knn-selection/status.tsv
```

Run order and what each run means:

1. NOW, on the current source: the `profile` phase is the number this lane
   needs (select_ms at k = 1, 2, 5, 10, 15 on one bucket). A slope near
   15 us per launch per unit of k confirms section 2 and C1's premise; a
   flat slope says the scan is load-bound and C1 is not worth a kernel.
   The `gate` phase will FAIL at reach on this source (no hook); that
   failure is expected and is recorded. `MOJOLEARN_KNN_SELECTION_SKIP_GATE=1`
   skips it.
2. After the follow-on lane lands C4 (uniform trip count) with the hook:
   the same leg with `MOJOLEARN_KNN_SELECTION_ARMS=baseline,uniform`; the
   verdict is equality on every fixture (including `divergent_tail`),
   reach on both arms, and no price change claimed.
3. After C1: `MOJOLEARN_KNN_SELECTION_ARMS=baseline,headbound`. Promotion
   requires all correctness checks green, reach on both arms and default,
   and both orders' medians favoring `headbound` on `dyadic` AND `large`
   at k10 and k15; then flip the kernel-matrix row in the same session,
   rebuild without the trial define, rerun the gate with
   `--arms baseline` plus default to show the default equals the explicit
   candidate, and record the new `dyadic` medians against the cached rows
   in `bench/OPPONENT_REFERENCE.md` as cached-reference ratios.
4. Apple and AMD columns: the same harness on each box (the shell script's
   build line is Linux; on the Mac use `bindings/build.sh` after adding an
   extra-define hook, or a copied build line with the Metal flags), before
   any column other than NVIDIA takes the row. RUN OWED, not assumed.

Opponent rule: `bench/OPPONENT_REFERENCE.md` holds the cuML rows for
400k x 32 / 4k / k10 (10.225 ms) and k15 (10.817 ms) on H100 80GB HBM3,
driver 580.126.09, dyadic-v1. Do not rerun cuML unless the leg lands on a
different GPU model or driver, in which case the tuple is absent and
`tools/knn_cuml_reference.py` measures it once with provenance.

## 7. Contradictions between source and handoffs, and things to fix as we go

1. `docs/lanes/HANDOFF_knn.md` "Flags and rows now in force" says the Apple
   column has the small-k selector, transposed index and register tile OFF.
   `checks/kernel_matrix.mojo:834-842` (`_knn_identical_round_column`) has
   included `COLUMN_APPLE` since the Sep 9 afternoon flip recorded in its
   docstring. The older handoff is stale; `HANDOFF_knn_selector.md` is
   consistent with the source.
2. `neighbors/impl/detail/knn_brute_force.mojo` module docstring ("`select_radix.mojo`
   is implemented and is the selector here") predates the small-k
   selector: under IDENTICAL on all four columns the default is
   `smallk_bucket_kernel`; radix is k > 64 or the legacy define only.
3. The scan comment at `select_smallk_identical_candidate.mojo:277-284`
   ("the block is latency-bound, not bandwidth-bound, measured on the
   L40S: 270 us a launch") predates the K specialization; the H100 phase
   logs show the launch cost rising 42% from k10 to k15 with identical
   loads, which a load-latency-bound kernel cannot do. The profile phase
   settles which comment is true now.
4. The "aligned-load default" is a benchmark-shape gate:
   `knn_brute_force.mojo:560` admits vector loads only for
   `n_index == 400000 and n_queries == 4000 and n_features == 32 and k in {10, 15}`.
   A request one row larger takes scalar loads. The Apple metadata default
   (`:569-571`) is keyed the same way. `bench/OPPONENT_REFERENCE.md` says
   "this measured NVIDIA scope", so the record is honest, but the standing
   rule against building to datasets applies: the C1 row must be keyed on
   the kernel-matrix column, not on the shape, and the gate's `ties` and
   `divergent_tail` fixtures deliberately fall outside the tuple.
5. `partial_topk_merge_kernel` reserves 24 KiB of static shared memory per
   block for `MERGE_MAX_K = 1024` even at k = 10. Harmless at 0.45 ms per
   request; recorded so nobody profiles the merge and blames occupancy.
6. `neighbors/estimator.mojo:493-498` allocates the radix scratch
   (`2 x query_tile x buf_len` floats and uint32, about 42 MB each at 400k)
   on every request although the small-k selector never reads it; that is
   allocation time inside the public boundary and a separate, non-kernel
   candidate for the request price (not in this lane's scope).
7. `HANDOFF_ai_classical_identical_next_2026-09-10.md`'s "selection was
   nearly as costly as distance at k15" is confirmed by the phase logs
   (14.61 vs 16.23 ms).

## Files created by this pass (uncommitted, working tree)

- `tools/knn_selection_gate.py` (new)
- `tools/knn_selection_gate.sh` (new)
- `docs/lanes/BRIEF_knn_selection_2026-09-10.md` (this file)

No kernel, binding, check or Python package file was modified.

## Run 1 results (H100, 2026-09-11 03:06Z to 03:14Z, `bench/results/e1g/2026-09-10_230357-nvidia/remote/knn-selection`)

Profile phase (the deliverable on unchanged source). Request-level phase
timers, 400k index rows, 4k queries, d32, query tile 512, index tile 65,536,
56 distance and 56 selection launches, 48 merges:

| build | k | distance ms | select ms | merge ms |
|---|---:|---:|---:|---:|
| default (K-specialized) | 10 | 15.23 | 10.28 | 0.45 |
| default (K-specialized) | 15 | 15.23 | 14.60 | 0.45 |
| generic bucket | 1 | 16.25 | 16.11 | 0.44 |
| generic bucket | 2 | 16.25 | 19.25 | 0.44 |
| generic bucket | 5 | 16.24 | 24.71 | 0.46 |
| generic bucket | 10 | 15.26 | 28.59 | 0.45 |
| generic bucket | 15 | 15.24 | 30.28 | 0.45 |

The default selector's k-slope is 0.86 ms per unit of k per request, 15.4 us
per launch per unit of k, with an intercept near 1.7 ms: 84 percent of the
k10 selection and 88 percent of the k15 selection is k-proportional. That is
C1's premise, confirmed. Full request medians (baseline arm, unwired trial
build, so both arms ARE the baseline): large k10 33.65 ms, k15 38.41 ms;
dyadic k10 32.34 ms, k15 36.22 ms. Kernels sum to about 26 ms at k10, so
about 7.7 ms of every request is outside the three kernels (upload, download,
Python). The cached-reference ratios the harness prints (3.16x, 3.35x) are
not a paired opponent measurement and are not the qualified 2.61 to 2.88x.

Gate phase: arms equal, order ok, planted ok, no hard oracle mismatch on any
fixture; REACH NOT PROVEN for every arm as designed (the trial define is not
wired). The only "distance mismatch" rows were the 16 planted exact-match
queries, where the float32 Gram form leaves a residual near a true zero; the
harness floor now includes that residual (71b2f975). Implementation of C4
and C1 behind the trial define is the next lane (DEVIATION 2497, 2498).
