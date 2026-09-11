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

## Implementation pass (DEVIATION 2497 C4, DEVIATION 2498 C1; source only, 2026-09-11)

Nothing here was built or run on the Mac. Every command is in RUN OWED
below. Files touched by this pass (uncommitted, working tree):

- `neighbors/checks/select_smallk_identical_candidate.mojo` (the kernel
  lives here, not under `neighbors/impl/`; section 1's table says so):
  `smallk_bucket_kernel[CAP, K, UNIFORM, BOUND, SABOTAGE]`, the trial hook
  constants, `smallk_select_arm_from_env`, `_smallk_launch_bucket`,
  `smallk_select_launch(..., arm)`.
- `neighbors/impl/detail/knn_brute_force.mojo`: one `select_arm =
  smallk_select_arm_from_env()` before the query loop (once per request)
  and the extra argument at the selector call; nothing else.
- `bindings/build.sh`: `MOJOLEARN_BUILD_EXTRA_DEFINES` (empty default)
  appended to the `mojo build` line; header comment documents it.
- `tools/knn_selection_gate.sh`: the build phase now calls
  `bindings/build.sh` through that hook instead of mirroring the Linux
  command; `MOJOLEARN_KNN_SELECTION_SKIP_PROFILE=1` skips the profile
  (already measured, "Run 1 results"); `MOJOLEARN_KNN_SELECTION_EXTRA_DEFINES`
  passes further defines. Outputs unchanged.
- `tools/knn_selection_gate.py`: docstring for the three arm names and the
  per-arm sabotage; the numpy selftest accepts `uniform`.
- `neighbors/checks/knn_selector_arms_check.mojo` (new).
- this section.

### What the kernel does now

Three arms, one kernel, chosen by comptime parameters:

| arm | instantiation | scan loop | reject test |
|---|---|---|---|
| `baseline` | `[CAP, K, False, False, S]` | 2026-09-09 per-thread condition `col + 7*256 < length` | `pending < threshold` |
| `uniform` (C4) | `[CAP, K, True, False, S]` | block-uniform `batch_base + 2048 <= length`, `col = batch_base + tid` | `pending < threshold` |
| `headbound` (C4+C1) | `[CAP, K, True, True, S]` | as `uniform`, plus a refresh after completed batch 1, 4, 12, 28 | `pending < min(threshold, bound)` |

WITHOUT `-D MOJOLEARN_KNN_SELECT_TRIAL=1` the binary holds exactly one
instantiation per bucket, `[CAP, K, SMALLK_UNIFORM_TRIP_DEFAULT,
SMALLK_HEAD_BOUND_DEFAULT, False]`, and both defaults are `False`, so it is
the `baseline` arm: the `comptime if UNIFORM` / `comptime if BOUND` /
`comptime if SABOTAGE` branches fold away and the remaining code is the
2026-09-09 kernel (the one textual difference in that path is the rank
phase's page `((rank + rounds) & 1)` where `rounds` is a constant 0; the
compiler folds it). `smallk_select_arm_from_env` returns the default
without reading the environment, and the launch refuses any other arm, so
the new check cannot pass on a one-arm binary.

WITH the define, `_tiled_brute_force_knn_impl` reads `MOJOLEARN_KNN_SELECT`
(baseline / uniform / headbound / unset = default / anything else raises)
and `MOJOLEARN_KNN_SELECT_SABOTAGE` (exactly "1") once per request and
passes the arm to every selector launch of that request, exactly the
`MOJOLEARN_KNN_VECTOR_TRIAL` pattern the harness expects (the harness sets
`os.environ` before each `kneighbors` call and unsets it after).

The defaults are two comptime constants in the selector file rather than
the kernel-matrix row section 4 asked for, because this pass may not edit
`checks/kernel_matrix.mojo`; the flip that promotes an arm moves them into
`knn_selector_head_bound_for[column, identical]` in the same session.

The refresh cadence "after batches 1, 4, 12, 28" is implemented as a count
of COMPLETED batches (`done`, block-uniform under C4), tested after the
`done += 1` at the bottom of each batch, so refresh 1 happens with 2,048
columns seen and every lane holding min(k, 8) real keys (every published
head is real), and refresh 4 (28) after 8,192 (57,344) columns. A
65,536-column partition has 32 batches and takes all four; the 6,784-column
last partition of 400k has 3 batches and takes the first; the carved k-wide
tail partition has no batch and is the baseline kernel. The refresh reuses
the rank phase's machinery verbatim (butterfly `shuffle_min_u64` plus one
shared slot per lane group plus one barrier per round on fixed-lane-width
columns; the eight-level shared tree elsewhere), popping the block's
smallest head k times with the popped lane's COPY replaced by the sentinel;
the k-th pop is `bound`. A running `rounds` counter (refresh rounds plus
rank rounds) drives the butterfly's page parity, so two consecutive rounds
never write the same page across the refresh / rank boundary; that is the
"known parity" section 3 asked for.

### Why the bits are unchanged (the identity argument)

1. C4 changes only which loop visits a column, never whether or in what
   order: a batch the block takes has all eight columns of every lane inside
   the row (tid + 1792 < 2048 <= length - b*2048), so the per-thread form
   took it too; the one batch the per-thread form took and the uniform form
   does not (the last, for (length - 1792) mod 2048 in 1..255, lanes tid <
   that remainder) is visited by those lanes in the tail loop instead, the
   same eight columns ascending through the same `_smallk_insert`, and the
   per-thread form's tail for those lanes was empty. No lane runs an extra
   iteration; nothing else is predicated.
2. Every head published at a refresh is a key of the union of the lanes'
   lists, so `bound` (the k-th smallest of those 256) is the k-th smallest of
   a SUBSET of the union: at least k union keys are <= bound.
3. That count never decreases afterwards: a key leaves a list only when a
   smaller key from the same lane pushes it off the end, and the smaller key
   is <= bound as well. So any later pending key p >= bound has at least k
   union keys below it and is not among the row's k smallest; p == bound is
   impossible because keys carry their column and p's column is in no list.
   Dropping it is the same act as the baseline's dropping of a key at or
   above the lane's own k-th (`threshold`), which is the existing invariant.
4. Hence after the scan the union still contains the row's true top-k under
   every arm; the rank phase is unchanged (k exact UInt64 minima of the
   union, ties decided by the index half of the same composite key, the
   winner's value read back from the original tile cell), so the k output
   (value bits, index) pairs and their order are the baseline's.
5. `bound` is the sentinel before the first refresh, when fewer than k real
   heads exist, and in every partition too short for one batch, and
   `min(threshold, sentinel) = threshold`: those cases are the baseline
   kernel by construction, not by argument.

### Sabotage (reach per arm)

This differs from section 4's spec (XOR on both arms) on purpose: the flip
must prove the ARM'S OWN code path ran.

- `baseline` and `uniform`: bit 0 of the index half of the composite key is
  flipped for `u == 0` of every batch, inside that arm's own loop form
  (the two loop forms are separate `comptime if` blocks). One candidate
  column in eight carries its neighbor's index and the rank phase gathers
  that neighbor's value, so thousands of cells move on the large shape and
  the planted zero at column 2048 moves in the new check. The flipped index
  stays inside the batch, so no out-of-range gather.
- `headbound`: inside the refresh, the bound is the FIRST pop (the block
  minimum head) instead of the k-th; from then on a lane admits only new
  record minima, so the union loses true neighbors on nearly every row of
  every partition with at least one batch. Rounds and barriers are
  unchanged. A bound-minus-one sabotage was rejected: with unique keys it
  can only drop a key exactly one below the bound, which is not a reliable
  flip.
- The DEFAULT (env unset) is the `baseline` instantiation with sabotage, so
  the harness's default-reach row is provable too.

### The new check

`neighbors/checks/knn_selector_arms_check.mojo`: hashed tile quantized to 61
values (dense value ties, index tie-break decides), planted +0.0 exact
matches at columns 3, 2047, 2048, length/2, length-1 (row 1 adds 5 and
2049), a -0.0, a subnormal, and an all-equal row; k = 10 and 15; lengths
1793, 2047, 3940, 4095, 65281, 65535 ((length - 1792) mod 2048 in 1..255),
65536, 65537, 65536 + 3940. Asserts baseline == exhaustive host rank, then
uniform == baseline and headbound == baseline cell for cell (value bits and
index), and on the 65,536 rows that each arm's sabotage flips at least one
cell. Raises at compile time without the trial define.

### RUN OWED (orchestrator; nothing ran)

The leg ships `git archive` of the COMMITTED tree and copies the extra body
verbatim; the local environment does NOT reach the pod, so the arm pair for
the C4-only run rides in a wrapper file. Commit this pass first.

Step 0, the native check on the box (or as the first line of the wrapper):

```
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_SELECT_TRIAL=1 \
    -I . neighbors/checks/knn_selector_arms_check.mojo
```

Step 1, C4 alone (arms baseline,uniform; profile skipped, it is measured):

```
cat > /tmp/knn_c4_extra.sh <<'SH'
#!/bin/sh
export MOJOLEARN_KNN_SELECTION_ARMS=baseline,uniform
export MOJOLEARN_KNN_SELECTION_SKIP_PROFILE=1
cd /root/mojolearn && PATH="$HOME/.pixi/bin:$PATH" pixi run mojo run \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_SELECT_TRIAL=1 \
    -I . neighbors/checks/knn_selector_arms_check.mojo \
    > /root/gemm_leg_out/knn-selector-arms-check.log 2>&1
echo "arms_check_exit=$?" >> /root/gemm_leg_out/leg.txt
exec sh /root/mojolearn/tools/knn_selection_gate.sh
SH
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
MOJOLEARN_GEMM_LEG_EXTRA=/tmp/knn_c4_extra.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-knn-selection-c4 \
sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" \
    --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
cat <leg>/remote/knn-selector-arms-check.log | tail -3
cat <leg>/remote/knn-selection/summary.txt
cat <leg>/remote/knn-selection/status.tsv
```

Verdict for step 1: `KNN SELECTOR ARMS PASS`; in the gate JSON every
fixture and k shows `uniform` and `default` equal to `baseline` (including
`divergent_tail`), row order, planted and oracle green, reach flipped > 0
on `baseline`, `uniform` and `default` with clean bits restored. No price
claim is made for C4 (the timing block runs and is recorded, nothing is
read from it). On green, flip `SMALLK_UNIFORM_TRIP_DEFAULT = True` (or the
kernel-matrix row) in the same session; it is the prerequisite, not a win.

Step 2, C1 (the default pair baseline,headbound; same leg, no wrapper
needed beyond the check line):

```
cat > /tmp/knn_c1_extra.sh <<'SH'
#!/bin/sh
export MOJOLEARN_KNN_SELECTION_SKIP_PROFILE=1
exec sh /root/mojolearn/tools/knn_selection_gate.sh
SH
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
MOJOLEARN_GEMM_LEG_EXTRA=/tmp/knn_c1_extra.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-knn-selection-c1 \
sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" \
    --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
cat <leg>/remote/knn-selection/summary.txt
cat <leg>/remote/knn-selection/status.tsv
```

Step 3, Apple column on the Mac (orchestrator only, one light thing, after
the H100 verdict): `MOJOLEARN_NUMERIC_MODE=identical
MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_KNN_SELECT_TRIAL=1" sh
bindings/build.sh`, then `PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical
python3 tools/knn_selection_gate.py --out /tmp/knn-sel-apple --arms
baseline,uniform,headbound --pairs 2 --deadline 300`. AMD: the same script
on a DigitalOcean MI325X droplet. Both are RUN OWED before any column other
than NVIDIA takes the row.

### What flips the default and what does not

- The C4 gate (step 1) flips `SMALLK_UNIFORM_TRIP_DEFAULT` on equality plus
  reach alone; it claims no price.
- The C1 gate (step 2) flips the head-bound default ONLY IF: every
  correctness check is green on every fixture and k; reach flipped on
  `baseline`, `headbound` and `default` with clean bits restored; AND both
  timed orders' medians favor `headbound` on `dyadic` AND `large` at k10
  AND k15 in the gate JSON (the request-level `NearestNeighbors.kneighbors`
  boundary, `timing` block). The phase-timer split, the section 2 model,
  the new check's tile and any per-launch number are NOT promotion
  evidence: a tile or kernel win alone cannot promote a request default.
- A pass with a request-level loss or a split verdict (one fixture or one k
  favoring baseline) leaves the default off; the arm stays in the binary
  behind the define and the result is recorded here with the JSON path.
- After a flip: rebuild without the trial define, rerun the gate with
  `--arms baseline` plus default to show the default equals the explicit
  arm, and record the `dyadic` medians against the cached rows in
  `bench/OPPONENT_REFERENCE.md` as cached-reference ratios (never as a
  paired opponent measurement; cuML is not rerun).

## Step 1 result: C4 gated alone on the H100 (2026-09-11 03:38Z, `bench/results/e1g/2026-09-10_233258-nvidia/remote/knn-selection`)

The arms check passed on the box (18 cases). The gate with arms
baseline,uniform: every fixture (large, dyadic, ties, divergent_tail) at
k10 and k15 bit-equal across baseline, uniform and default, order ok,
planted ok, oracle clean (the cancellation floor holds); reach proven for
each arm (sabotage flipped 38,306 / 70,635 cells, restored). Timing pairs
were within noise, as expected for a trip-count change (large k10 30.47 vs
30.41 ms, k15 35.64 vs 35.63 ms). C4 is the prerequisite, not a win;
SMALLK_UNIFORM_TRIP_DEFAULT flipped to True on this evidence and the M4
check. Step 2 (headbound) runs next.

## Step 2 result: C1 headbound is NEGATIVE (H100, 2026-09-11 03:48Z, `bench/results/e1g/2026-09-10_234251-nvidia/remote/knn-selection`)

Correctness: every fixture at k10 and k15 bit-equal across baseline,
headbound and default, order ok, planted ok, oracle clean; reach proven
for headbound on every fixture (36,556 / 75,129 / 36,208 / 74,708 / 9,408 /
16,396 / 5,256 cells flipped, restored). The arms check passed on the box.

Timing, full requests, three alternating pairs per order:

| fixture | k | baseline median ms | headbound median ms | delta |
|---|---:|---:|---:|---:|
| large | 10 | 31.15 | 33.53 | +2.38 |
| large | 15 | 35.93 | 39.09 | +3.16 |
| dyadic | 10 | 30.81 | 33.28 | +2.47 |
| dyadic | 15 | 35.98 | 39.36 | +3.39 |

Headbound loses on both fixtures at both k, in both orders. The four
block-wide refreshes (each a k-round butterfly with a barrier per round,
plus the publish and the 256-way k-th smallest) cost more than the
insertion chain they remove; the model in this brief priced the saved
insertions and not the refresh. SMALLK_HEAD_BOUND_DEFAULT stays False.
The code stays behind the trial define as a measured negative, and the
next candidate is C2 warpbound: the same bound at warp scope with shuffles
and no barriers, refreshed more often because it is cheap, which the
model in this brief already ranked second. C1's arm also gives C2 its
gate for free (same harness, arms baseline,warpbound).

## Implementation pass, C2 warpbound (DEVIATION 2515; source only, 2026-09-11)

Nothing here was built or run on the Mac. The C1 arm is untouched and stays
the measured negative. Files touched by this pass (uncommitted, working
tree):

- `neighbors/checks/select_smallk_identical_candidate.mojo`: sixth kernel
  parameter `WARPBOUND` on `smallk_bucket_kernel` (after `SABOTAGE`, so the
  five-parameter instantiations are unchanged), the helpers
  `_smallk_shuffle_xor_u64`, `_smallk_warp_group_bound`,
  `_smallk_warpbound_refresh_due`, the comptime cadence
  `SMALLK_WARPBOUND_EVERY = 2`, the default `SMALLK_WARPBOUND_DEFAULT =
  False`, the arm `SMALLK_ARM_WARPBOUND = 3` in `smallk_select_arm_from_env`
  and `_smallk_launch_bucket`, and the module docstring.
- `neighbors/checks/knn_selector_arms_check.mojo`: the fourth arm in the
  equality and reach properties, same 18 cases.
- `tools/knn_selection_gate.py`: docstring for the arm name and its
  sabotage; the numpy selftest accepts `warpbound`. The harness already
  selects arms by name from `--arms`, which the shell script fills from
  `MOJOLEARN_KNN_SELECTION_ARMS`, so `baseline,warpbound` needs no code.
- `tools/knn_selection_gate.sh`: the arm comment only (names pass through
  unchecked; the native side raises on an unknown one).
- this section.

### Mechanism

At a refresh point every lane publishes its `depth`-th smallest key,
`depth = ceil(k / lanes)` (its head for k <= 32 on NVIDIA). The warp is cut
into `lanes / group` aligned groups of `group` lanes, `group` the largest
power of two with `(lanes / group) * depth >= k`; each group takes the
MINIMUM of its published keys and the warp bound is the MAXIMUM over the
groups. Both folds are the rank phase's xor butterfly on the 64-bit key as
two 32-bit halves: five `shuffle_xor` steps on 32 lanes (one min step, four
max steps for k in 9..16), about 45 instructions, no barrier, no shared
memory, `rounds` untouched. For k = 10 and 15 on 32 lanes: depth 1, group 2,
sixteen pairs. Rejection is `pending < min(threshold, bound)` exactly as in
C1; the tail loop uses the same gate.

Refresh cadence: after every `SMALLK_WARPBOUND_EVERY = 2` completed batches,
starting at the first batch count after which every lane holds k keys
(`ceil(k / 8)`, so after batch 2 for k in 9..16), so every published key is
real. A 65,536-column partition refreshes 16 times (batches 2, 4, ..., 32);
the 6,784-column last partition of 400k once (batch 2); the 3,940-column
`divergent_tail` remainder (one batch) and the carved k-wide tail (none)
never, and there `bound` stays the sentinel, `min(threshold, sentinel) =
threshold`, and the arm IS the uniform arm by construction. A partition
between 2,048 and 4,095 columns at k in 9..16 is the same case (one batch).

Why this bound and not the one the lane brief suggested (the warp minimum
of the lanes' own k-th keys). That one is valid too and costs the same five
steps, but a lane's k-th key after i elements is Gamma(k)/i, spread
sqrt(k)/i, and the minimum over 32 of them sits near 4.7/i at k10 and 8.2/i
at k15 (Monte Carlo, 200k draws), not near k/(32 i): with 32 lanes each
passing at that rate the warp-level "some lane inserts" probability is
`1 - (1 - 4.7/i)^32`, above 0.5 until i is past 200 of the 256 elements a
lane sees, so almost no insertion chain is removed (event model: 0.88x of
the scan's instructions at k10, 0.94x at k15). The figure k/(32 i) belongs
to the k-th smallest of the 32 lane HEADS (0.37/i at k10, 0.62/i at k15),
which needs k removal rounds (450 to 675 instructions per refresh). The
group bound is the one-round compromise: 1.69/i (the maximum over 16 pairs
of the pair's minimum, independent of k). Against C1's block k-th of 256
heads (0.040/i at k10, 0.060/i at k15) it is 30 to 40 times looser, and it
is refreshed 16 times instead of 4 at a tenth of the price each.

### Why the bits are unchanged

1. A group's minimum m_q is the published key of some lane L_q of that
   group, and L_q holds `depth` keys at or below m_q (its `depth` smallest).
   The bound B is at or above every m_q, so every L_q holds `depth` keys at
   or below B; the L_q are distinct lanes (groups are disjoint) and no key
   lives in two lanes (each column is scanned by exactly one lane), so at
   least `(lanes / group) * depth >= k` distinct keys of the warp's union
   are at or below B. The warp's union is a subset of the block's, so the
   block union holds at least k keys at or below B.
2. That count never decreases afterwards: a key leaves a list only when a
   smaller key from the same lane pushes it off the end, and the smaller
   key is at or below B as well.
3. A pending key p >= B therefore has at least k union keys below it and is
   not among the row's k smallest; p == B is impossible because keys carry
   their column and p's column is in no list. Dropping it is the same act
   as the baseline's dropping of a key at or above the lane's own k-th.
4. Warps hold different bounds; each is a bound on its own union, a subset
   of the block's, so 1 to 3 hold per warp, and the union of all 256 lists
   still contains the row's true top-k after the scan under every arm.
5. The rank phase is untouched (no shared memory, no barrier, `rounds`
   stays 0, so the butterfly's page parity is the uniform arm's): k exact
   UInt64 minima of the union, ties decided by the index half of the same
   composite key, the winner's value read back from the original tile
   cell. Before the first refresh, and in every partition with fewer than
   `ceil(k / 8)` full batches, `bound` is the sentinel and the arm is the
   uniform arm by construction, not by argument.

### Sabotage (reach)

Inside the refresh, bit 63 of the reduced bound is cleared, and only when
the reduction returned a real key (a refresh that never produced one cannot
prove reach). `twiddle_in` sets bit 31 of every non-negative float's bits,
so every composite key with a non-negative distance has bit 63 set and the
sabotaged bound sits below all of them: from the first refresh on, every
lane rejects every non-negative-distance key, and the output is the top-k
of the first 4,096 columns (k in 9..16). That differs from the true top-k
whenever one true neighbor lies beyond those columns: certain on the arms
check (a planted +0.0 at length / 2 and at length - 1 on rows 0 and 1) and
with probability `1 - (4096 / 65536)^k` on every hashed row of every
partition with at least two batches (`large`, `dyadic`, `ties`, and the six
full partitions of `divergent_tail`). The two suggestions in the lane brief
were rejected on the arms-check fixture: its rows 0 and 1 are 61 consecutive
float bit patterns from 1.0 plus the planted specials, so the true top-k is
decided inside batch 1 except for the far zeros, which are the extreme
minimum and pass any bound that is merely too tight by a lane offset or by
one head; only a bound below the +0.0 key flips them.

### Expected cost (model; the gate's numbers replace it)

Event model (per warp per 65,536-column launch, iid keys; the brief's
section 2 model with the refresh priced): baseline 231 events at k10 and
246 at k15; warpbound 116 at k10 and 116 at k15 (the group bound does not
depend on k) plus 16 refreshes of about 45 instructions. Instructions per
lane: 16.4k -> 10.2k at k10 (0.62x), 24.7k -> 13.6k at k15 (0.55x). With the
profile's 1.7 ms tile-read intercept held: selection 10.28 -> about 7.0 ms
at k10 (about 3.3 ms per request) and 14.60 -> about 8.8 ms at k15 (about
5.8 ms per request). C1's refresh under the same model: 4 x k rounds x
about 90 instructions plus 10 to 15 barriers each, about 3.6k (k10) to 5.4k
(k15) instructions per lane, and the model priced it at 0.34x; it measured
+2.4 / +3.2 ms. So the model has been wrong once by about 8 ms, and the
honest expectation is bracketed: BEST CASE the model's 3.3 / 5.8 ms saving;
WORST CASE the insertion saving does not exist (the chain is not what the
k-slope measures, or the extra live state changes the code the compiler
emits, as C1 may have shown) and the arm costs its refresh alone, 16 x 45 =
720 instructions per lane, about 4 percent of the scan, about +0.4 ms at
k10 and +0.3 ms at k15 on a full request, inside the pair spread seen in
step 1 (30.47 vs 30.41 ms). Warpbound is therefore the discriminating
experiment for the whole bound family: with no barrier and a refresh an
order of magnitude cheaper than C1's, a neutral or negative result says the
insertion chain is not a divergent-branch cost the scan can shed, and C3
(compaction) is dead for the same reason; the next candidate is then C6
(fused distance and select) or the request-level non-kernel time (7.7 ms
of every request is outside the three kernels, "Run 1 results").

Refresh cost per batch, side by side: C1 (per refresh, four per partition)
k rounds of 10 shuffles + 1 shared store + 1 barrier + 8 shared loads + 7
compares; C2 (per refresh, sixteen per partition) 10 shuffles + 5
compares + 5 selects, no barrier, no shared memory, no loop over k.

### RUN OWED (orchestrator; nothing ran)

Commit this pass first (the leg ships `git archive` of the COMMITTED tree).
Same wrapper shape as step 1; the arm pair rides in the wrapper because the
local environment does not reach the pod.

```
cat > /tmp/knn_c2_extra.sh <<'SH'
#!/bin/sh
export MOJOLEARN_KNN_SELECTION_ARMS=baseline,warpbound
export MOJOLEARN_KNN_SELECTION_SKIP_PROFILE=1
cd /root/mojolearn && PATH="$HOME/.pixi/bin:$PATH" pixi run mojo run \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_SELECT_TRIAL=1 \
    -I . neighbors/checks/knn_selector_arms_check.mojo \
    > /root/gemm_leg_out/knn-selector-arms-check.log 2>&1
echo "arms_check_exit=$?" >> /root/gemm_leg_out/leg.txt
exec sh /root/mojolearn/tools/knn_selection_gate.sh
SH
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
MOJOLEARN_GEMM_LEG_EXTRA=/tmp/knn_c2_extra.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-knn-selection-c2 \
sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" \
    --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
cat <leg>/remote/knn-selector-arms-check.log | tail -3
cat <leg>/remote/knn-selection/summary.txt
cat <leg>/remote/knn-selection/status.tsv
```

Verdict lines: `KNN SELECTOR ARMS PASS` with `SELECTOR_ARMS_REACH_PASS
65536 <k> <base> <uni> <hb> <wb>` all four counts nonzero; in the gate JSON
every fixture and k shows `warpbound` and `default` equal to `baseline`,
row order, planted and oracle green, reach flipped > 0 on `baseline`,
`warpbound` and `default` with clean bits restored; then the `timing`
block. Apple column after the H100 verdict, orchestrator only, one light
thing: `MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_EXTRA_DEFINES="-D
MOJOLEARN_KNN_SELECT_TRIAL=1" sh bindings/build.sh`, then
`PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical python3
tools/knn_selection_gate.py --out /tmp/knn-sel-apple --arms
baseline,warpbound --pairs 2 --deadline 300`. AMD: the same script on a
DigitalOcean MI325X droplet (64-lane wavefront: depth 1, group 4, sixteen
groups of four, the same argument). Both RUN OWED before any column other
than NVIDIA takes the row.

### Promotion rule

`SMALLK_WARPBOUND_DEFAULT` flips to True (and moves into the kernel-matrix
SCHEDULING row in the same session) ONLY IF: every correctness check is
green on every fixture and k; reach flipped on `baseline`, `warpbound` and
`default` with clean bits restored; AND both timed orders' medians favor
`warpbound` on `dyadic` AND `large` at k10 AND k15 in the gate JSON's
`timing` block, the request-level `NearestNeighbors.kneighbors` boundary
(eight cells, all eight for warpbound). The phase-timer split, the event
model above, the arms check's tile and any per-launch number are NOT
promotion evidence: a tile or kernel win alone cannot promote a request
default. A split verdict leaves the default off, the arm stays behind the
define as a measured result, and the JSON path is recorded here. After a
flip: rebuild without the trial define, rerun the gate with `--arms
baseline` plus default to show the default equals the explicit arm, and
record the `dyadic` medians against the cached rows in
`bench/OPPONENT_REFERENCE.md` as cached-reference ratios (never as a paired
opponent measurement; cuML is not rerun). If warpbound is neutral or
negative, record it beside C1 and close the bound family (C2, C3) in
section 3 with the JSON path.

## Step 3 result: C2 warpbound is NEGATIVE; the bound family is closed (H100, 2026-09-11 04:11Z, `bench/results/e1g/2026-09-11_000701-nvidia/remote/knn-selection`)

Correctness: all four arms bit-equal on every fixture at k10 and k15, order
ok, planted ok, oracle clean; reach proven for warpbound everywhere
(79,362 / 119,362 / 79,444 / 119,444 / 13,892 / 20,892 / 11,900 / 17,900
cells flipped, restored); the arms check passed on the box.

| fixture | k | baseline median ms | warpbound median ms | delta |
|---|---:|---:|---:|---:|
| large | 10 | 30.96 | 32.33 | +1.37 |
| large | 15 | 36.02 | 36.85 | +0.83 |
| dyadic | 10 | 30.72 | 32.21 | +1.49 |
| dyadic | 15 | 36.49 | 37.69 | +1.20 |

A bound that costs five shuffles every second batch and no barrier still
loses, so the saved insertions are worth less than the model said: the
k-proportional cost of the selection is NOT the insertion chain. Both bound
arms stay behind the trial define as measured negatives; C3 (cross-batch
compaction) rests on the same premise and is closed with them.
SMALLK_WARPBOUND_DEFAULT stays False.

What the numbers say instead: the generic bucket's cost at k1 (16.1 ms) is
already above the K-specialized k10 (10.3 ms), and the specialized slope of
15.4 us per launch per k survives a bound that removes half the insertion
events. The remaining k-proportional work is the rank phase (k exact
minima, each a butterfly with a barrier, per row) and the k-dependent
register list itself. Next: measure, do not model. A trial arm that skips
the rank phase (timing only, output invalid) and one that skips the scan,
under the same define, give the split directly; then the rank phase's k
barrier rounds are the candidate (a single-pass bitonic or shuffle-based
extraction of k minima with the same comparison order), which the brief's
C5 already named and ranked too low on the wrong premise.

## Implementation pass, phase split measured on the box (DEVIATION 2516; source only, 2026-09-11)

Nothing here was built or run on the Mac (py_compile and `sh -n` only).
Measure, do not model: the k-proportional cost survived two bound arms that
removed half the insertion events (steps 2 and 3), so the split of the
selection launch between its scan phase and its rank phase is now measured
DIRECTLY with timing-only arms that run one phase and skip the other. Files
touched by this pass (uncommitted, working tree):

- `neighbors/checks/select_smallk_identical_candidate.mojo`: seventh kernel
  parameter `PHASE` on `smallk_bucket_kernel` (after `WARPBOUND`, default
  `SMALLK_PHASE_FULL`, so every six-parameter instantiation is unchanged),
  the constants `SMALLK_PHASE_FULL / SKIPRANK / SKIPSCAN` and the arms
  `SMALLK_ARM_SKIPRANK = 4`, `SMALLK_ARM_SKIPSCAN = 5`,
  `SMALLK_ARM_SCANONLY1 = 6` in `smallk_select_arm_from_env` and
  `_smallk_launch_bucket`, the `scan_length` alias, the skipscan fill and
  the skiprank epilogue, `PHASE` on `_smallk_enqueue`, the hook comment and
  the module docstring. `bitcast` imported from `std.memory`.
- `tools/knn_selection_gate.py`: `--timing-only-arms` (default empty),
  `--phase-timers auto|require|off`, descriptor-level capture of the
  binding's `KNN_PHASE_TIMERS` line per request (`PhaseCapture`), a shared
  `time_pair` helper for the arm pair and the timing-only pairs, the
  `timing_only` JSON table and summary lines ("output invalid; phase cost
  only"), `phase_ms_median` on every timing row, the numpy selftest's
  timing-only stand-in and a fake phase line written to fd 1 so the capture
  is exercised without a GPU.
- `tools/knn_selection_gate.sh`: `MOJOLEARN_KNN_SELECTION_TIMING_ONLY_ARMS`
  (passed through as `--timing-only-arms`) and
  `MOJOLEARN_KNN_SELECTION_PHASE_TIMERS=1` (adds
  `-D MOJOLEARN_KNN_PHASE_TIMERS=1` to the binding build and passes
  `--phase-timers require`); both recorded in `gate.txt`.
- this section.

The shipped default path is untouched: without the trial define the only
instantiations are the six-parameter ones with `PHASE = FULL`, in which
`scan_length` is a plain copy of `length` (the three scan loop conditions
read the copy; a value copy, no codegen), the skipscan fill and the
skiprank epilogue are `comptime if` branches on a constant and fold away,
and the rank phase is the `elif SMALLK_SHUFFLE` / `else` pair it was, with
its bodies' text unchanged. A `comptime assert` refuses any `PHASE` other
than `FULL` on a non-trial build, and another refuses it on any scan form
but the uniform default (no bound, no sabotage).

### The arms (OUTPUT INVALID BY CONSTRUCTION)

| arm | instantiation | what runs | what is written |
|---|---|---|---|
| `skiprank` | `[CAP, K, True, False, False, False, SKIPRANK]` | the scan exactly as `uniform` (the shipped default since 541ca9e2) runs it: same batches, same `_smallk_insert`, same tail loop; then an epilogue, NO rank phase | thread 0 writes one digest-addressed cell to slot 0 of the row and the sentinel key's two halves (index 0xFFFFFFFF, value bits 0xFFFFFFFF) to slots 1 .. k-1 |
| `skipscan` | `[CAP, K, True, False, False, False, SKIPSCAN]` | NO scan (the three scan loops see `scan_length = 0`); every lane's k slots are filled with a synthetic pattern; then the rank phase exactly as the default runs it (k rounds, one butterfly and one barrier per round, index-half tie rule, the winner's value gathered from the tile cell) | the k "winners" of the synthetic pattern with their gathered values: real columns, wrong answer |
| `scanonly1` | `[1, 1, True, False, False, False, SKIPRANK]` | `skiprank` with the register list one key deep (CAP = 1 and K = 1 folded, so the scan keeps a running minimum: one compare, one conditional swap, threshold = the minimum); the runtime k is ignored by the kernel | thread 0 writes one slot per row; the rest of the row is whatever the output buffer held |

`scanonly1` cost nothing beyond a third instantiation of the same body, so
it is in (the task allowed skipping it if it needed a second kernel body;
it did not).

The exact skipscan filling. For lane `tid` and slot `s < k`:

    ordinal = s * 256 + tid + 1                         (1 .. 256 k, block-distinct)
    column  = (ordinal * 2654435761) mod length         (a real column of the row)
    key     = ordinal << 32 | column

Slots k .. CAP-1 stay the sentinel, as after a real scan. Why this makes
the rank phase's cost representative: (i) the rank loop is `for rank in
range(k)` unconditionally, and every round costs the same instructions
(one `shuffle_min_u64`, one shared store by lane 0 of each warp, ONE
barrier, eight shared loads and seven compares, thread 0's two stores and
one gather, the winning lane's predicated CAP-1 shift), so any filling with
at least k real keys makes it do its full k rounds; (ii) the keys are
distinct and ascending within a lane (the ordinal grows with the slot), so
slot 0 is the lane's minimum exactly as after a real scan and the shift
keeps the list sorted; (iii) the k block minima are ordinals 1 .. k, one
lane each (lanes 0 .. k-1 of warp 0), so every round has exactly one
winning lane whose warp executes the shift, as in the real kernel, where
the winners are spread over the warps but there is still one shifting warp
per round; (iv) the index half is a hashed column inside the row, so the
winner's gather stays in range (an all-sentinel list would gather at
column 0xFFFFFFFF, out of the tile) and lands on a spread-out column the
way a real winner's does, not on a leading column that the distance
kernel's last writes may have left in L2.

Why skiprank cannot let the compiler drop the scan. The scan's loads have
no side effect of their own, so an arm that discards the lists would
measure an empty kernel. In the epilogue every lane XORs its whole list
and its threshold into one digest, the block folds the 256 digests (the
rank phase's own butterfly shape plus one barrier on fixed-lane-width
columns, the shared tree elsewhere), and thread 0 uses the block digest as
a GATHER ADDRESS (`values[base + digest mod length]`) and stores the
gathered cell and the column. A load address that depends on every lane's
list keeps every insert live, and the fold is a collective every lane
takes, so no lane's scan can be sunk under thread 0's branch. Cost of the
epilogue: about one rank round (ten shuffles, one barrier, eight shared
loads, one gather), independent of k, so `select_ms(skiprank)` overstates
the scan by about one round and its k-slope is the scan's k-slope alone.

The timing-only arms refuse the sabotage bit (a RAISE, not a fallback):
there is no reach to prove on an arm whose output is wrong by design. The
harness's only assertion on such an arm is that its output DIFFERS from
the clean reference (equality would mean the arm's body did not run); the
baseline samples in the same pairs must still equal the reference.

### The harness

`--timing-only-arms skiprank,skipscan,scanonly1` keeps those arms out of
the correctness, oracle and reach sections entirely (they never appear in
`correctness` or `reach`), and in the timing block pairs each one with the
FIRST `--arms` arm under the existing protocol (one warmup per arm,
`--pairs` pairs in order (A, B), then `--pairs` in order (B, A), every
sample kept, medians and minima per arm and per order). The rows go to a
separate `timing_only` table in the JSON, each carrying `"note": "output
invalid; phase cost only"` and `output_valid: {A: true, B: false}`, and
the summary prints them under a `timing_only: OUTPUT INVALID; PHASE COST
ONLY` header. Nothing in that table can feed a promotion.

Phase timers. The profile phase's `select_ms` comes from
`-D MOJOLEARN_KNN_PHASE_TIMERS=1`, a BUILD define (not an environment
switch): `_tiled_brute_force_knn_impl` then synchronizes after every launch
class and prints one `KNN_PHASE_TIMERS distance_ms ... select_ms ...
merge_ms ...` line per request to file descriptor 1 from inside the
binding. So the harness cannot flip it per request; the shell script adds
the define to the BINDING build under
`MOJOLEARN_KNN_SELECTION_PHASE_TIMERS=1`, and the harness redirects fd 1
into a scratch file around every request (outside the timed region),
flushes C stdio, restores the descriptor and parses the line. Every timed
sample then carries `distance_ms / select_ms / merge_ms`, every timing row
(the arm pair and the timing-only rows) carries `phase_ms_median` per arm
and per order, and the per-arm `select_ms` is READ, not inferred from
request deltas. The price: the synchronizations serialize the queue, so on
that build the request medians are slower than an untimed request, are
labeled so in the JSON (`phase_timers.serialized`) and the summary, and
are not comparable to the qualified numbers or admissible against the
cached cuML rows (the cached-reference note says so on that build).
`--phase-timers auto` (default) records the line when the build prints
one; `require` fails otherwise; `off` never redirects.

### RUN OWED (orchestrator; nothing ran)

Step 0, on the Mac, numpy only, no native import, seconds (the checker's
own smoke; the lane may not run tests): expect `PASSED`, a `timing_only`
table with three rows per timed fixture and k, `phase_timers.available:
true`, and `select_ms` medians in every row.

```
python3 tools/knn_selection_gate.py --selftest --quick --out /tmp/knn-sel-selftest \
    --arms baseline,uniform --timing-only-arms skiprank,skipscan,scanonly1 --pairs 1 --deadline 120
grep -A12 '^timing_only' /tmp/knn-sel-selftest/summary.txt
```

Step 4, the H100 leg. Commit this pass first (the leg ships `git archive`
of the COMMITTED tree); the arm lists ride in the wrapper file because the
local environment does not reach the pod. The first `--arms` arm is
`uniform`, not `baseline`, on purpose: the timing-only arms are the uniform
scan form, and `uniform` IS the shipped default since 541ca9e2 (`baseline`
is the pre-C4 per-thread trip count, kept as the second arm so the pair
timing re-measures C4 under the phase timers for free; step 1 found the
two within noise). ARMS=baseline alone, as the task specified, is also
admissible on that evidence.

```
cat > /tmp/knn_phase_extra.sh <<'SH'
#!/bin/sh
export MOJOLEARN_KNN_SELECTION_ARMS=uniform,baseline
export MOJOLEARN_KNN_SELECTION_TIMING_ONLY_ARMS=skiprank,skipscan,scanonly1
export MOJOLEARN_KNN_SELECTION_PHASE_TIMERS=1
export MOJOLEARN_KNN_SELECTION_SKIP_PROFILE=1
cd /root/mojolearn && PATH="$HOME/.pixi/bin:$PATH" pixi run mojo run \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_SELECT_TRIAL=1 \
    -I . neighbors/checks/knn_selector_arms_check.mojo \
    > /root/gemm_leg_out/knn-selector-arms-check.log 2>&1
echo "arms_check_exit=$?" >> /root/gemm_leg_out/leg.txt
exec sh /root/mojolearn/tools/knn_selection_gate.sh
SH
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
MOJOLEARN_GEMM_LEG_EXTRA=/tmp/knn_phase_extra.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-knn-selection-phases \
sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" \
    --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
cat <leg>/remote/knn-selector-arms-check.log | tail -3
cat <leg>/remote/knn-selection/summary.txt
cat <leg>/remote/knn-selection/status.tsv
```

The arms check line is a regression fence only (the timing-only arms are
not in it; it proves the FULL instantiations still pass with the seventh
parameter in place). Budget: about 300 requests at 50 ms serialized, well
inside the 300 s deadline.

### How to read the result

Everything below is read from the `timing_only` rows' `phase_ms_median`
(`select_ms`, per arm, pooled and per order; the two orders must agree
within the pair spread or the row is noise) at k10 and k15 on `large` and
`dyadic`. `select_ms(uniform)` is the reference (about 10.3 ms at k10 and
14.6 ms at k15 per the profile; 56 launches):

1. Rank phase share: `select_ms(uniform) - select_ms(skiprank)`, plus about
   one rank round's worth (the epilogue) that skiprank still pays.
2. Scan share: `select_ms(uniform) - select_ms(skipscan)`. Note skipscan's
   warps reach the first barrier together, whereas after a real scan they
   arrive staggered by their insert counts, so skipscan may UNDERSTATE the
   rank phase by that one-time skew; the per-round cost (the slope) is
   unaffected.
3. Additivity check: `select_ms(skiprank) + select_ms(skipscan)` against
   `select_ms(uniform)` plus 56 launch overheads (a few tenths of a
   millisecond). If the sum overshoots by more than that, the two phases
   overlap across blocks when both are present (a block in its rank phase
   hides behind another block's scan) or the missing phase changed the
   codegen of the present one (registers, occupancy); then the ABSOLUTE
   shares are not additive and only the k-slopes below are the verdict.
4. THE VERDICT, the k-slope of each phase per launch per unit of k:
   `(select_ms(arm, k15) - select_ms(arm, k10)) / 5 / 56` for `skiprank`
   (the scan's slope) and for `skipscan` (the rank phase's slope). Their
   sum should land near the profile's 15.4 us. Whichever phase owns the
   slope owns the k-proportional cost: if `skipscan` carries it, the k
   barrier rounds are the candidate (C5, a single-pass extraction of k
   minima with the same comparison order); if `skiprank` carries it, the
   cost is in the scan but not in the insertion events the bound arms
   removed, and the register list's depth is the next suspect.
5. List depth inside the scan: `select_ms(skiprank) - select_ms(scanonly1)`
   at k10 and at k15 is the price of maintaining a K-deep list versus a
   running minimum over the same loads and keys; its own k-slope
   (`scanonly1` is k-independent, so this is skiprank's slope again) says
   how much of the scan's slope is the list rather than the loads.

What this pass does NOT claim: no output of a timing-only arm is a result;
no request median from a phase-timer build is comparable to the qualified
26.66 / 31.13 ms or to the cached cuML rows; no default moves on this
evidence. The next candidate is chosen from the slopes, then gated as a
normal arm (bit-equal, reach, request-level price) before anything flips.

## Step 4 result: the phase split, measured (H100, 2026-09-11 04:40Z, `bench/results/e1g/2026-09-11_003148-nvidia/remote/knn-selection`)

Phase timers read from a phase-timer build (request medians in that run are
serialized and not comparable to anything else); `select_ms` per arm,
400k/4k/d32, 56 launches, paired medians:

| arm (what runs) | k10 select ms | k15 select ms |
|---|---:|---:|
| uniform (shipped: scan + rank) | 10.27 | 14.69 |
| skiprank (scan + digest epilogue) | 9.87 | 13.95 |
| skipscan (synthetic fill + rank) | 1.45 | 1.93 |
| scanonly1 (scan with CAP 1, K 1: a running minimum) | 3.22 | 3.22 |

So: rank phase 0.4 ms (k10) to 0.75 ms (k15); scan 8.8 to 12.8 ms; of the
scan, 3.2 ms is k-independent (tile reads and the per-element compare) and
5.6 to 9.6 ms is the per-lane K-deep list: 0.80 ms per unit of k of the
0.88 total slope. The rank phase and the merge are not the target. The
K-deep list is, and steps 2 and 3 already showed that REJECTING MORE does
not shrink it (both bounds halved the insertion events and lost), which
means the K-chain's cost is paid whether or not a lane inserts: the
predicated chain executes for the whole warp on every element step. The
lever is therefore execution frequency of the chain, not admission:
defer insertion (per-lane append of admitted keys into a short queue,
K-independent per element; drain the queue through the K-chain only when a
warp-uniform test says some queue is non-empty or full, so the chain runs
once per several elements instead of once per element). The final list is
the k smallest keys admitted, which is the k smallest overall regardless of
insertion order, and keys are unique, so the rank phase sees the same
list: bit-identical by construction. Per lane about 32 insertions happen in
256 elements at k10, so the chain should run an order of magnitude less
often. Expected: scan 8.8 to about 4 ms at k10 and 12.8 to about 5.5 ms at
k15; a request from 31 to about 26 ms and 36 to about 29 ms. Measured next
as arm `deferred` under the same define and the same gate.

## Implementation pass, deferred insertion (DEVIATION 2517; source only, 2026-09-11)

Nothing here was built or run on the Mac (py_compile of the gate and a
pure-Python event model only; no test, no native, no mojo). The step 4
verdict is the premise: of the scan's 8.8 (k10) to 12.8 ms (k15), 3.2 ms is
k-independent and 5.6 to 9.6 ms is the per-lane K-deep list, 0.80 ms per
unit of k, and both bound arms halved the lanes' insertion EVENTS and lost,
so the K-chain is paid on every element step whether or not a lane inserts.
The lever is how often the chain EXECUTES, not how often a lane admits.
Files touched by this pass (uncommitted, working tree):

- `neighbors/checks/select_smallk_identical_candidate.mojo`: eighth kernel
  parameter `DEFERRED` on `smallk_bucket_kernel` (after `PHASE`, default
  False, so every seven-parameter instantiation is unchanged), the helpers
  `_smallk_append` and `_smallk_drain`, the comptime knob `SMALLK_DEFER_Q =
  4`, the ballot width `SMALLK_MASK_DT`, the default `SMALLK_DEFERRED_DEFAULT
  = False`, the arm `SMALLK_ARM_DEFERRED = 7` in `smallk_select_arm_from_env`
  and `_smallk_launch_bucket`, `DEFERRED` on `_smallk_enqueue`, the hook
  comment and the module docstring. `vote` imported beside `shuffle_xor`.
- `neighbors/checks/knn_selector_arms_check.mojo`: the fifth arm in the
  equality and reach properties, same 18 cases; the REACH_PASS line gains a
  fifth count.
- `tools/knn_selection_gate.py`: docstring for the arm name and its
  sabotage; the numpy selftest accepts `deferred`. The harness selects arms
  by name from `--arms`, which the shell script fills from
  `MOJOLEARN_KNN_SELECTION_ARMS`, so `uniform,deferred` needs no code.
- this section.

The shipped default path is untouched: without the trial define the only
instantiations are `[CAP, K, True, False, False, False, FULL, False]`, in
which the queue and its count are constants, every `comptime if DEFERRED`
folds away, the per-element test is `pending < threshold` followed by
`_smallk_insert` as before, and the tail loop, the epilogue and the rank
phase are textually the uniform arm's. `SMALLK_DEFERRED_DEFAULT` excludes
both bound defaults by `comptime assert`.

### Mechanism

Per lane, an element step under `deferred` does: the composite key, one
compare against the lane's threshold, and, when admitted, an append into a
`SMALLK_DEFER_Q = 4` slot queue held in registers (newest at slot 0: three
comptime-indexed 64-bit moves and a store, plus a counter increment). No
K-chain per element. The queue drains at WARP-UNIFORM points only:

- at the end of every unrolled batch (block-uniform under C4), and
- immediately after any element step at which `vote[SMALLK_MASK_DT](count
  == Q)` over the warp is nonzero (some lane's queue is full). The ballot
  width follows the column's lane count, the ball-cover kernel's rule (a
  64-lane wavefront needs a 64-bit ballot). Every lane reaches the vote:
  the append is closed before it and the batch trip count has no `tid` in
  it, so it is convergent.

A drain runs `_smallk_insert` (unchanged) once per queued key, predicated
per slot on the lane's count, then clears the count; the warp therefore
executes the chain max(count over its 32 lanes) times per drain instead of
once per element step. The drain has no collective of its own (the chain is
per lane); warp uniformity is a scheduling choice that makes the lanes'
chains coincide. After the batch loop one more drain runs (a guard: the
queue is already empty, every batch drained at its end), then the tail loop
(the remainder under 2,048 columns, a per-lane trip count where a vote
would not be convergent) inserts eagerly through the same chain from that
empty queue; it holds at most eight elements per lane, nothing to defer.
The rank phase starts with every lane's list exactly as the eager arm
leaves it.

Why Q = 4: half an unrolled batch. Past the fourth batch at k in 9..16 a
lane admits under 2.4 elements per batch in expectation, so four slots
rarely fill and the cadence is the batch end; four UInt64 are eight
32-bit registers on top of the list's 32 at CAP = 16, where eight slots
would be sixteen for a six percent smaller chain count (model below).
Register pressure is the risk this arm carries that the bound arms did
not: nine extra live registers per thread (the queue and its count) at 256
threads per block; if the gate shows the arm register-bound (a resident
block per SM lost at a 64-register boundary shows as a scan that does not
speed up although the chain count fell), `SMALLK_DEFER_Q = 2` is the first
knob (0.66x / 0.74x in the model) and 8 the second.

### Why the bits are unchanged

1. Let S be the set of keys a lane scans; keys are unique (each carries its
   column). The eager path keeps L_e, the k smallest of the prefix seen so
   far, and admits p iff p < threshold_e, the k-th smallest of that prefix
   (sentinel while fewer than k). The deferred path keeps L_d, the k
   smallest of the set I of keys DRAINED so far, and admits p iff p <
   threshold_d, the k-th smallest of I.
2. I is a subset of the prefix, so threshold_d >= threshold_e at every
   step: the stale threshold admits a SUPERSET of what the eager path
   admits, never a subset.
3. Take any x among the k smallest of S. When x is scanned, at most k - 1
   keys of S are below x, hence at most k - 1 keys of I, and x is not in I,
   so the k-th smallest of I is above x (or the sentinel): x is admitted,
   queued, and inserted at the next drain. It never leaves L_d afterwards,
   because a key leaves only when k smaller keys of the same lane have
   been inserted and only k - 1 exist.
4. Every extra key the stale threshold admitted goes through the same
   `_smallk_insert`, which keeps "the list is the k smallest of everything
   inserted so far" under ANY insertion order and leaves the list unchanged
   for a key at or above its current k-th (the carry runs off the end). So
   after the last drain L_d holds every one of the k smallest of S, exactly
   min(k, |S|) keys, and only keys of S: L_d is the k smallest of S sorted,
   which is L_e. Equality of sets of unique keys is equality of the sorted
   lists slot for slot.
5. The rank phase reads only those lists (never the threshold), pops the
   union's exact minima with the same UInt64 compare, decides ties by the
   same index half, and gathers the winner's value from the same tile
   cell. Corners: the queue is empty before the tail loop and the rank
   phase (batch-end drains plus the guard drain); the tail inserts eagerly;
   a partition too short for one batch (the carved k-wide tail) never
   enters the batch loop and is the uniform arm by construction.

### Sabotage (reach)

At every drain the newest queued key (slot 0) is skipped on every lane
whose count is nonzero, inside `_smallk_drain` only. Late in the scan a
lane's queue at the batch-end drain usually holds one key, so a true
neighbor admitted there is the newest and is dropped with high
probability; a row has k of them, so on the hashed fixtures thousands of
cells move per request. On the arms check it is certain: the planted +0.0
at column length - 1 is the last element of the last batch, hence the
newest in lane 255's queue at that batch's drain, and it is a top-k key of
rows 0 and 1 (the keys below it are the -0.0, the other zeros and the
subnormal). The uniform arm's index flip is excluded from the deferred
instantiation so a flip proves the drain path, not the loop.

### Expected cost (model; the gate's numbers replace it)

Event model, pure Python (iid keys, 256 elements per lane, 32 lanes, 400
trials, the brief's section 2 model with the queue simulated): chain
executions per warp per 65,536-column launch, eager 231 (k10) and 246 (k15)
against deferred with Q = 4 drained every batch and on any full queue 116
(0.50x) and 138 (0.56x). Q = 2: 0.66x / 0.74x. Q = 8: 0.47x / 0.52x. Q = 8
drained every second batch: 0.42x / 0.47x. The floor is the first batches
(every lane admits every element, the queues fill every Q elements, the
cost equals today's) plus the warp MAXIMUM of a small binomial per batch
afterwards (two to three chains per batch of eight), not Q. Step 4's "an
order of magnitude less often" counted one lane's 32 admissions in 256;
the warp pays the max over its lanes, so the honest model is 2x, not 8x.

In the phase split's terms: the K-deep list is 5.6 ms at k10 and 9.6 ms at
k15 (0.80 ms per unit of k) and scales with the chain frequency, so 0.50 x
5.6 = 2.8 ms and 0.56 x 9.6 = 5.4 ms, a saving of 2.8 / 4.2 ms. Against it
the new k-independent work per lane: about 232 / 246 warp-level append
events at about 9 instructions, 256 votes at about 3, and 41 / 45 drains at
about 6 for the slot predicates, about 3.3k instructions, which at the
chain's measured rate (5.6 ms for 231 x 72 instructions per lane at k10,
about 0.34 ms per thousand) is about +1.2 ms unless it hides under the load
latency the k-independent 3.2 ms already pays.

Expected `select_ms` (56 launches, phase-timer build): k10 10.27 -> about
8.7 ms, k15 14.69 -> about 11.7 ms. Brackets: BEST CASE the append and vote
hide under the loads, 7.5 / 10.5 ms; IF-CONVERTED DRAIN (the compiler runs
all Q chains per drain regardless of the counts: 164 / 180 executions,
0.71x / 0.73x), 9.9 / 13.3 ms; WORST CASE as C1 and C2 showed, the extra
live state changes the code the compiler emits and nothing is saved, +1.2
ms. On an unserialized request that is about 31 -> 29.4 ms (k10) and 36 ->
33 ms (k15) at the model's center, which is what the promotion run reads.

### RUN OWED (orchestrator; nothing ran)

Commit this pass first (the leg ships `git archive` of the COMMITTED tree).
The arms line is the regression fence for all five arms; the gate pairs
`uniform` (the shipped default) with `deferred`, no timing-only arms, phase
timers ON so `select_ms` per arm is READ from `phase_ms_median` rather than
inferred from request deltas, profile skipped (measured).

```
cat > /tmp/knn_deferred_extra.sh <<'SH'
#!/bin/sh
export MOJOLEARN_KNN_SELECTION_ARMS=uniform,deferred
export MOJOLEARN_KNN_SELECTION_PHASE_TIMERS=1
export MOJOLEARN_KNN_SELECTION_SKIP_PROFILE=1
cd /root/mojolearn && PATH="$HOME/.pixi/bin:$PATH" pixi run mojo run \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_SELECT_TRIAL=1 \
    -I . neighbors/checks/knn_selector_arms_check.mojo \
    > /root/gemm_leg_out/knn-selector-arms-check.log 2>&1
echo "arms_check_exit=$?" >> /root/gemm_leg_out/leg.txt
exec sh /root/mojolearn/tools/knn_selection_gate.sh
SH
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
MOJOLEARN_GEMM_LEG_EXTRA=/tmp/knn_deferred_extra.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-knn-selection-deferred \
sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" \
    --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
cat <leg>/remote/knn-selector-arms-check.log | tail -3
cat <leg>/remote/knn-selection/summary.txt
cat <leg>/remote/knn-selection/status.tsv
```

Verdict lines: `KNN SELECTOR ARMS PASS` with `SELECTOR_ARMS_REACH_PASS 65536
<k> <base> <uni> <hb> <wb> <df>` all five counts nonzero; in the gate JSON
every fixture (large, dyadic, ties, divergent_tail) at k10 and k15 shows
`deferred` and `default` equal to `uniform`, row order, planted and oracle
green, reach flipped > 0 on `uniform`, `deferred` and `default` with clean
bits restored; then the `timing` block's `phase_ms_median.select_ms` per
arm, pooled and per order (the two orders must agree within the pair
spread or the row is noise), read against the expected 8.7 / 11.7 ms and
the brackets above. Request medians on that build are serialized and are
NOT a price.

### Promotion rule

Two runs, in this order. The phase-timer run above is the mechanism
verdict only: it says whether the chain count fell (`select_ms` down at
both k) and by how much. It promotes nothing. If it is green and
`select_ms(deferred) < select_ms(uniform)` at both k, the PROMOTION RUN is
the same wrapper WITHOUT `MOJOLEARN_KNN_SELECTION_PHASE_TIMERS` (an
unserialized build, request-level timing at the `NearestNeighbors.kneighbors`
boundary). `SMALLK_DEFERRED_DEFAULT` flips to True (and moves into the
kernel-matrix SCHEDULING row in the same session) ONLY IF, on that second
run: every correctness check is green on every fixture and k; reach flipped
on `uniform`, `deferred` and `default` with clean bits restored; AND all
EIGHT request-level timing cells (both orders' medians, `dyadic` and `large`,
k10 and k15) favor `deferred`. The phase-timer split, the event model, the
arms check's tile and any per-launch number are not promotion evidence. A
split verdict leaves the default off, the arm stays behind the define as a
measured result, and the JSON path is recorded here beside C1 and C2. After
a flip: rebuild without the trial define, rerun the gate with `--arms
uniform` plus default to show the default equals the explicit arm, and
record the `dyadic` medians against the cached rows in
`bench/OPPONENT_REFERENCE.md` as cached-reference ratios (never as a paired
opponent measurement; cuML is not rerun). Apple (the Mac, orchestrator
only, one light thing, after the H100 verdict: `MOJOLEARN_NUMERIC_MODE=identical
MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_KNN_SELECT_TRIAL=1" sh
bindings/build.sh`, then `PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical
python3 tools/knn_selection_gate.py --out /tmp/knn-sel-apple --arms
uniform,deferred --pairs 2 --deadline 300`) and AMD (a DigitalOcean MI325X
droplet; 64-lane wavefront, 64-bit ballot by `SMALLK_MASK_DT`) are RUN OWED
before any column other than NVIDIA takes the row.

## Step 5 result: deferred insertion is NEGATIVE (H100, 2026-09-11 04:57Z, `bench/results/e1g/2026-09-11_005250-nvidia/remote/knn-selection`)

Correctness green on every fixture (five arms bit-equal, reach for
deferred 75,060 / 114,164 / 75,012 cells). select_ms per arm from the
phase-timer build: k10 uniform 10.31, deferred 10.00 (0.97x); k15 uniform
14.73, deferred 16.26 (1.10x). Not promotable; default stays off.

Three mechanisms that reduce how often or how much the K-chain runs
(headbound, warpbound, deferred) all failed to move the 0.80 ms per unit
of k, and one of them ran the chain about half as often. The K-dependent
cost is therefore not the chain's execution count. What scales with K in
the kernel regardless of admissions is the register list itself: CAP=16
UInt64 keys per lane indexed by a runtime slot in the insert and drain
paths, which the compiler can only keep in registers if every index is a
compile-time constant after unrolling; otherwise the list lives in local
memory and every touch is a memory access, and register count itself
lowers occupancy. That is a property of the compiled kernel, so the next
step is to read it off the PTX/SASS: registers per thread, local memory
bytes, spill stores and loads, and whether the list accesses compile to
ld.local/st.local, for K=1, 10 and 15. If the list is in local memory the
fix is a fully unrolled compare-and-shift with constant indices (or a
sorting network) so the list stays in registers; if occupancy is the
limiter the fix is a smaller CAP for small k.

## Implementation pass, kernel resource stats (DEVIATION 2519; source only, 2026-09-11)

Step 5 left one question the gate cannot answer: is the CAP=16 UInt64
list in registers, and if so does it cost occupancy? Both are properties
of the compiled kernel, read off the compiled kernel. Nothing in this pass
touches the kernel or the gate; it adds a leg extra body and its parser:

- `tools/knn_selector_kernel_stats.sh` (the leg extra body; runs on the
  pod under `MOJOLEARN_GEMM_LEG_EXTRA`, writes
  `/root/gemm_leg_out/knn-kernel-stats/`)
- `tools/knn_selector_kernel_stats.py` (the parser; `--selftest` on
  embedded ptxas / cuobjdump / PTX / SASS / driver-log samples passes on
  the Mac; `--out <dir>` writes `stats.tsv`; `--split-driver-log` and
  `--extract-ptx` are the body's helpers)

A note on the premise. Step 5 suspected "a runtime slot index forces the
list into local memory". The source does not have one: every index into
`local_keys` (a `SIMD[DType.uint64, CAP]`) is a `comptime for` constant,
guarded by a runtime `slot < k` predicate on the generic bucket and folded
on the K-specialized ones; `threshold` is its own register. So the answer
is not in the source, which is exactly why it is measured: LLVM may still
demote a 16-wide 64-bit vector that is rewritten under predication in a
deep loop, and the register count of an in-register list is itself the
occupancy question.

### Method (documented API first, repo-proven route as cross-check)

1. Runtime attributes, the authoritative numbers. A Mojo driver, generated
   by the body on the pod (it must track the kernel's parameter list and
   the lane may not edit the kernel; the generated text is kept beside
   the results as `driver_files.mojo` / `driver_stdout.mojo`), calls for
   each instantiation
   `DeviceContext.compile_function[kernel, dump_asm=..., _dump_sass=..., _ptxas_info_verbose=True]()`
   and reads `DeviceFunction.get_attribute(Attribute.NUM_REGS)`,
   `Attribute.LOCAL_SIZE_BYTES`, `Attribute.SHARED_SIZE_BYTES`,
   `Attribute.CONST_SIZE_BYTES`, `Attribute.MAX_THREADS_PER_BLOCK` and
   `DeviceFunction.occupancy_max_active_blocks_per_multiprocessor(256, 0)`.
   These are the driver's own answers about the cubin it will launch
   (`Attribute` mirrors `CUfunction_attribute`: NUM_REGS = "the number of
   registers used by each thread of this function", LOCAL_SIZE_BYTES =
   "the size in bytes of local memory used by each thread"). Citation:
   max.modular.com/api/mojo/max/gpu/host/device_context/DeviceContext
   (`compile_function`, parameters `dump_asm`, `dump_llvm`, `_dump_sass`,
   `_ptxas_info_verbose`), .../device_context/DeviceFunction
   (`get_attribute`, `occupancy_max_active_blocks_per_multiprocessor`,
   `dump_rep`), .../func_attribute/Attribute. The docs say `_dump_sass`
   and `_ptxas_info_verbose` are NVIDIA-only and need the CUDA toolkit on
   the box, so the body requests them only when it finds `ptxas` and
   `cuobjdump`. `dump_asm` takes `True` (stdout), a `Path`, a static
   string, or a function returning a `Path`; the body builds the
   function-returning-Path variant first (dumps land in `dumps/<label>.ptx`
   and `.sass`), and if that variant does not compile it builds the `True`
   variant and the parser cuts each instantiation's PTX and SASS out of
   `driver.log`. Nothing is launched; the kernel body is the shipped one
   (the trial define only gates host dispatch and the timing-only PHASE
   assert, so it is needed for the CAP=1 scanonly1 instantiation and
   changes nothing else).
2. Spills and executed local traffic. `ptxas --verbose --gpu-name <the
   PTX's .target>` on each dumped PTX reports "N bytes stack frame, N
   bytes spill stores, N bytes spill loads" and "Used N registers"; then
   `cuobjdump --dump-resource-usage` (REG / STACK / SHARED / LOCAL) and
   `cuobjdump --dump-sass` (LDL / STL counts) on the cubin. This is the
   GEMM lane's H100 procedure (`tools/gemm_cuda_resources.py`,
   docs/lanes/HANDOFF_speed_gemm_2026-09-10.md "H100 resource
   inspection": 255 registers, 4144-byte stack, 44-byte spills, one block
   per SM), NVIDIA's binary-utilities tools. An offline `ptxas` is that
   toolkit's answer, not necessarily the runtime JIT's; where they
   disagree the runtime attribute wins and stats.tsv carries both. On a
   pod without the toolkit the spill columns and SASS counts are OWED
   and the runtime attributes plus the PTX-level `ld.local` / `st.local`
   counts stand.
3. The repo-proven sidecar route as a cross-check and as the fallback if
   the driver does not build: `mojo build --emit asm` of
   `bench/knn_reference_price_main.mojo` retains one
   `<out>_<module>_<hash>.ptx` per GPU kernel (the form behind
   bench/results/gemm_swizzle_2026-09-10/h100-current-128.ptx.gz). Entry
   names carry module and hash, not parameters, so instantiations are
   identified by elimination: the default build has [16,10], [16,15],
   [16,0], [32,0], [64,0]; the `-D MOJOLEARN_KNN_IDENTICAL_GENERIC_K=1`
   build has only the three K=0 buckets; the two sidecars present only in
   the default build are the k10 / k15 specializations (`emit/manifest.tsv`).
4. The trial binding is built the way the gate builds it
   (`bindings/build.sh` with `-D MOJOLEARN_KNN_SELECT_TRIAL=1` through
   `MOJOLEARN_BUILD_EXTRA_DEFINES`), hashed, and scanned for NVPTX text
   blobs (an attempt; whether a Mojo shared library embeds PTX as text is
   not documented, and `binding_ptx_blobs=` in `stats.txt` records the
   answer either way).

Instantiations measured (rows of `stats.tsv`): `cap16_k0_generic` (the
bucket k=1 and every other k <= 16 hits without the specialization),
`cap16_k10`, `cap16_k15` (the shipped specializations), `cap1_k1_scanonly1`
(the CAP=1 / K=1 SKIPRANK form from step 4), `cap32_k0_generic`,
`cap64_k0_generic` (the k <= 32 and k <= 64 buckets, for the CAP curve),
plus one row per emit-asm sidecar (`emit_default_*`, `emit_generic_*`).
Columns: label, cap, k, entry, registers, local_bytes, spill_stores,
spill_loads, ld_local, st_local, sass_ldl, sass_stl, shared_bytes,
blocks_per_sm_by_regs, blocks_per_sm_by_shared, blocks_per_sm_max,
occupancy_pct, runtime_blocks_per_sm, runtime_occupancy_pct, sources.
Occupancy is computed as the brief asked (65,536 registers per SM, 2,048
threads per SM, 256 threads per block; registers rounded up to the 8 per
thread allocation unit, 32 blocks per SM and 228 KB shared per SM as the
other limits) and the runtime's own `occupancy_max_active_blocks_per_multiprocessor`
answer sits beside it; when they disagree the runtime's is the number.
Raw resource text is kept (`dumps/<label>.ptxas.log`, `.resources.log`),
dumps over 256 KB are gzipped, binaries and cubins are deleted, and the
directory is fenced at 2 MB. Smoke: the body ran on the Mac against stub
`pixi` / `nvidia-smi` / `ptxas` / `cuobjdump` / `bindings/build.sh` (no
Mojo, no build, no GPU) end to end, both driver variants, the stdout
split, the emit manifest and the size fence; the selftest covers the
parsers and the occupancy arithmetic at the thresholds named below.

### RUN OWED (orchestrator; nothing ran on a GPU)

Commit this pass first (the leg ships `git archive` of the COMMITTED
tree). One H100 leg, 60 minutes, the body as the leg extra; the leg's own
device check and card run first, then the binding build (about 5 min), the
driver build and run (two compiles at most), two `--emit asm` builds and
the assembly step. Nothing in it is a timing number.

```
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
MOJOLEARN_GEMM_LEG_EXTRA=tools/knn_selector_kernel_stats.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-knn-kernel-stats \
sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" \
    --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
cat <leg>/remote/knn-kernel-stats/stats.txt
cat <leg>/remote/knn-kernel-stats/stats.tsv
cat <leg>/remote/knn-kernel-stats/status.tsv
cat <leg>/remote/knn-kernel-stats/driver_lines.txt
cat <leg>/remote/knn-kernel-stats/emit/manifest.tsv
```

The GPU is named because the numbers are per architecture (sm_90a; the
PTX `.target` is read back and recorded in `tools.txt` / the ptxas logs)
and the step 4 and 5 costs they explain were measured on the H100. Green
is: `status.tsv` shows `driver 0` (or, failing that, both `emit-asm-*`
rows 0 with a nonempty `emit/manifest.tsv`), `stats.tsv` has the three
rows `cap16_k0_generic`, `cap16_k10`, `cap16_k15` with `sources`
containing `runtime` (or `ptxas`), and `cuda_toolkit=1` in `stats.txt`
(without it the spill columns are blank and OWED, not zero).

### How to read the result

For each of `cap16_k10`, `cap16_k15` and `cap16_k0_generic`:

- `local_bytes > 0`, or `spill_stores`/`spill_loads > 0`, or `ld_local` /
  `st_local` (PTX) or `sass_ldl` / `sass_stl` (SASS) nonzero: the list
  (or part of the scan state) is in local memory and every K-chain step
  is a memory access, which is a K-proportional cost paid whether or not
  the lane inserts and explains why halving the insertion events did not
  move it. The fix is a list whose every touch the compiler keeps in
  registers: the constant-index compare-and-shift written so the
  predicated stores are selects on scalars rather than element writes
  into one wide vector (CAP scalars, or a sorting network for the K=10 /
  K=15 specializations), and no wide `SIMD` value live across the batch
  loop. `cap1_k1_scanonly1` is the control: a one-key list must show
  zero local traffic; if it does not, the local memory belongs to the
  scan state rather than the list, and that is a different fix (the
  `batch[8]` unroll or the composite-key temporaries).
- `local_bytes == 0` and no local traffic, but `registers > 128`: the
  list is in registers and its footprint holds the SM at one 256-thread
  block (`blocks_per_sm_by_regs = 1`, 12.5% occupancy; with the 8 per
  thread allocation unit, 81 to 128 registers gives 2 blocks at 25%, 65
  to 80 gives 3, 33 to 64 gives 4 to 6, 32 or fewer gives the full 8 at
  100%). The K-slope is then the
  latency the missing warps would have hidden, and the fix is a smaller
  CAP for small k (CAP 16 costs 32 registers for the list alone on
  every k <= 16; a CAP=8 bucket for k <= 8 and the K=10 / K=15
  specializations at CAP=K would cut that), read against the CAP curve
  in the `cap32` / `cap64` rows. `runtime_blocks_per_sm` is the driver's
  answer to the same question and wins.
- neither (no local traffic, registers at or below 64, two or more
  blocks per SM): the cost is instruction count, sixteen predicated
  compare-and-shift steps per element on every lane, and the fix is a
  cheaper chain: a shorter one (CAP=K on the specializations, K-1
  compares instead of CAP), a compare-and-swap whose per-step cost is
  one `setp` plus two `selp` rather than a compare, a copy and a store,
  or the bitonic per-warp merge that C5 deferred.

The generic bucket at K=0 is expected to be the worst of the three (its
`slot < k` guards are runtime predicates on a 16-deep chain); if the K=10
and K=15 rows differ from it only in instruction count and not in local
traffic or registers, that difference is the whole specialization win of
2026-09-09 (20.3 to 9.1 ms) and the same mechanism bounds what CAP=K can
still buy.

### What the docs did not settle

- Whether `_ptxas_info_verbose=True` prints the `ptxas -v` summary or
  changes what `dump_asm` writes: the doc says it "changes dump_asm to
  output verbose PTX assembly". The body captures stdout either way and
  runs its own `ptxas --verbose` on the dumped PTX, so the spill counts
  do not depend on it.
- Which of the four `dump_asm` Variant members is accepted at parameter
  position for a file path (a `Path` value, a static string, or a
  function returning a `Path`): the body builds the function form first
  and the `True` (stdout) form second, and records which one compiled in
  `stats.txt` (`driver_variant=`).
- Whether `mojo build --emit asm` on the pod (native target) writes the
  same `<out>_<module>_<hash>.ptx` sidecars the Mac cross-compile wrote
  on 2026-09-10; the manifest records what appeared.
- The MAX environment-variable reference (max.modular.com/environment-variables)
  lists no variable that dumps PTX or SASS for Mojo-compiled kernels
  (`MODULAR_DEBUG=ir-output-dir=` dumps MAX graph-compiler IR, not
  `mojo build` kernels), and the `mojo build` CLI reference is not in the
  MAX docs index reachable here, so `--emit asm` is cited from the repo's
  own use rather than from a doc page.

## Step 6 result: kernel resource stats (H100, 2026-09-11 05:18Z, `bench/results/e1g/2026-09-11_011544-nvidia/remote/knn-kernel-stats`)

From DeviceFunction attributes plus the dumped PTX (ptxas spill counts were
not emitted by the toolchain on the pod; ld.local/st.local counted in the
PTX):

| instantiation | registers | local bytes | ld.local / st.local | shared | blocks per SM (regs) | occupancy |
|---|---:|---:|---:|---:|---:|---:|
| cap16 generic k (k up to 16) | 99 | 0 | 0 / 0 | 2048 | 2 | 25% |
| cap16 k10 (shipped) | 54 | 0 | 0 / 0 | 2048 | 4 | 50% |
| cap16 k15 (shipped) | 56 | 0 | 0 / 0 | 2048 | 4 | 50% |
| cap1 k1 (scanonly1 control) | 31 | 0 | 0 / 0 | 2048 | 8 | 100% |
| cap32 generic | 107 | 0 | 0 / 0 | 2048 | 2 | 25% |
| cap64 generic | 178 | 0 | 0 / 0 | 2048 | 1 | 12.5% |

The list is register-resident with no spills, so the "local memory" reading
is closed. The K cost is instruction count plus occupancy: the shipped
kernels run four 256-thread blocks per SM where the k-independent control
runs eight, on a scan whose 3.2 ms floor is tile reads. Two levers follow
from the table, both bit-identical: (1) CAP = K for the specialized
instantiations (a key outside a lane's k smallest has k same-lane keys
below it and can never reach the block's top-k, so the sixteenth slot at
k10 is dead weight: 32 registers of list become 20 or 30), which may lift
the shipped kernels to 5 or 6 blocks per SM; (2) a cheaper compare-and-shift
(setp plus selp swaps instead of the branchy insert) to cut the
per-element instruction count. Lever 1 is one comptime argument and is
measured first, as arm `capk`.

## Implementation pass, CAP = K and the branch-free chain (DEVIATION 2521; source only, 2026-09-11)

Nothing here was built or run on the Mac (py_compile of the gate and `sh -n`
of the stats body only; no test, no native, no mojo). Step 6 is the premise:
the CAP = 16 list is register-resident with no spills, the shipped k10 / k15
kernels use 54 / 56 registers and run four 256-thread blocks per SM (50
percent occupancy) where the CAP = 1 control runs eight at 31 registers, on
a scan whose 3.2 ms floor is tile reads. Lever 1 is CAP = K; lever 2 is a
cheaper carry chain. Both are behind the trial define, OFF by default. Files
touched by this pass (uncommitted, working tree):

- `neighbors/checks/select_smallk_identical_candidate.mojo`: ninth kernel
  parameter `SELP` on `smallk_bucket_kernel` (after `DEFERRED`, default
  False, so every eight-parameter instantiation is unchanged), the comptime
  storage width `STORE` inside the kernel, `_smallk_insert` and
  `_smallk_drain` taking the storage width as an inferred parameter `W` and
  the depth as `CAP` (call sites now `[CAP=CAP]`, the two uniform-form sites
  `[CAP=CAP, SELP=SELP]`), the SELP min/max chain inside `_smallk_insert`,
  the arms `SMALLK_ARM_CAPK = 8` and `SMALLK_ARM_CAPK_SELP = 9` in
  `smallk_select_arm_from_env` and `_smallk_launch_bucket`, the defaults
  `SMALLK_CAPK_DEFAULT = False` and `SMALLK_SELP_DEFAULT = False` with
  `DEFAULT_CAP` on the non-trial path, the comment block above
  `_smallk_insert`, the hook comment and the module docstring.
- `neighbors/checks/knn_selector_arms_check.mojo`: the sixth and seventh
  arms in the equality and reach properties, same 18 cases; the REACH_PASS
  line gains two counts.
- `tools/knn_selection_gate.py`: docstring for the two arm names and their
  sabotage; the numpy selftest accepts `capk` and `capk_selp`. The harness
  selects arms by name from `--arms`, which the shell script fills from
  `MOJOLEARN_KNN_SELECTION_ARMS`, so `uniform,capk,capk_selp` needs no code.
- `tools/knn_selector_kernel_stats.sh`: four rows `capk_k10`, `capk_k15`,
  `capk_selp_k10`, `capk_selp_k15` (instantiations `[10, 10, True, False,
  False, False, SMALLK_PHASE_FULL, False, <selp>]` and the k15 twins), the
  ninth parameter on the existing rows, `SMALLK_SELP_DEFAULT` imported.
- this section.

The shipped default path is untouched: without the trial define the only
instantiations are `[CAP, K, True, False, False, False, FULL, False, False]`
with CAP the bucket capacity (`DEFAULT_CAP` folds to CAP while
`SMALLK_CAPK_DEFAULT` is False), `STORE` folds to CAP for 16 / 32 / 64, the
helpers infer `W == CAP` and their bodies are textually the ones that
shipped, and `comptime if SELP` folds to the branchy chain. The control for
that claim is in the runs below: the `uniform` arm's `select_ms` on the
phase-timer build must land on the step 4 / step 5 numbers (10.27 to 10.31
ms at k10, 14.69 to 14.73 ms at k15); if it moves, this pass moved the
shipped kernel and the diff is the suspect before any capk number is read.

### Mechanism

`capk` instantiates the K-specialized buckets with CAP = K:
`smallk_bucket_kernel[10, 10, ...]` and `[15, 15, ...]` instead of `[16, 10]`
and `[16, 15]`. The generic bucket keeps CAP = 16 (k is a runtime value
there; the arm RAISES on it rather than run the CAP = 16 uniform kernel
under its name, so a column without the specialization row, that is
anything but NVIDIA unless `-D MOJOLEARN_KNN_IDENTICAL_SPECIALIZE_COMMON=1`,
cannot pass the arm silently).

One fact of the language decides the shape: Mojo's SIMD width must be a
power of two (the repo pads every such queue, `warp_topk.mojo`'s TQP, and
asserts it in `gemm_unpinned.mojo`), so `SIMD[DType.uint64, 10]` cannot be
spelled. The kernel therefore separates the list DEPTH (CAP, what every
loop over the list is bounded by) from its STORAGE (a comptime conditional
chain `STORE` = the next power of two at or above CAP: 16 for 10 and 15,
CAP itself for 1, 16, 32, 64). A CAP = K instantiation reads and writes
exactly K lanes after the sentinel fill; lanes K .. 15 have no use at all,
so they carry no phi, no copy and no register in the compiled kernel. That
is what CAP = K means at the register level, and it is exactly the state
the shipped CAP = 16 kernel does NOT have, for one reason found by reading
the kernel for every place CAP exceeds K:

| where CAP is used | CAP = 16, K = 10 today | CAP = K |
|---|---|---|
| `SIMD[DType.uint64, CAP]` (the list) | 16 lanes, 32 registers if all live | STORE lanes, K touched |
| `_smallk_insert`: chain `range(CAP)` with `slot < k`, threshold at `slot == k - 1` | folds to slots 0 .. 9; slots 10 .. 15 never written | folds to all CAP slots; threshold at CAP - 1 |
| the rank phase's pop: shift `range(CAP - 1)` reading `slot + 1`, sentinel at `CAP - 1` | READS slots 10 .. 15 (the sentinel) into 9 .. 14: the one place that touches them, which keeps them live across the whole scan as twelve registers of the constant | shifts K - 1 slots, sentinel at K - 1; slots 0 .. K - 1 end up with the same content (the shifted-in slot 10 was the sentinel anyway) |
| skipscan fill `range(CAP)` with `slot < k`; warpbound published slot `range(CAP)`; C1 head `local_keys[0]`; skiprank digest `range(CAP)` | fine | fine; the capk arms exclude these arms anyway (asserts) |
| `_smallk_drain` (deferred) | calls `_smallk_insert`; no spare-slot assumption | same; excluded by assert |
| the sentinel | slots k .. 15 are the sentinel forever; the pops never read one while rank < k because the union has at least k keys (`length >= k` is the caller's guarantee) | no spare slot; the same argument, unchanged |
| `_smallk_launch_bucket[16, 10]` / `[16, 15]` | the bucket capacity is the instantiation's CAP | the capk branch enqueues `[K, K, ...]`; `DEFAULT_CAP` does the same on the non-trial path once `SMALLK_CAPK_DEFAULT` flips |

So the CAP > K assumptions were: the storage type (a language constraint,
handled by STORE), the insert's slot guard and threshold pick (fold
correctly at CAP == K), and the rank phase's shift past K (correct at CAP ==
K, and the reason the dead lanes are live today). No sentinel slot, drain
or digest depends on a spare slot.

`capk_selp` is `capk` plus SELP: every step of the carry chain becomes one
unsigned minimum into the slot and one unsigned maximum into the carry
(`min(pending, current)`, `max(pending, current)` on UInt64), no per-step
branch; the admission branch `pending < threshold` around the chain is kept,
so a warp with no admitting lane still skips the chain exactly as the
uniform arm does. It was cheap: a `comptime if SELP` inside `_smallk_insert`
and the parameter threaded through the two uniform-form call sites; the
CAP = 16 path is the `else` branch, textually the one that shipped. SELP
requires CAP == K (no slot guard in the chain) and the uniform scan form,
excludes both bounds, deferred and the timing-only phases, all by
`comptime assert`.

### Why the bits are unchanged

1. A lane's list holds its CAP smallest keys of everything it scanned
   (`_smallk_insert` keeps "the k smallest inserted so far, sorted" under
   any insertion order; the carry runs off the end of a full list).
2. Only a lane's k smallest can ever be in the block's top-k: any other key
   of that lane has k same-lane keys below it. With CAP = K the list holds
   exactly those k, so the union of the 256 lists still contains the row's
   true top-k after the scan.
3. Admission is the same test on the same threshold (the lane's k-th
   smallest, `local_keys[k - 1]` at CAP = 16, `local_keys[CAP - 1]` at CAP
   = K, the same slot), so every lane admits the same keys in the same
   order; the SELP chain preserves the multiset {slot, carry} at every step
   and leaves the minimum in the slot, which is the same sorted list
   whichever way it is computed (keys are unique; two sentinels tie to the
   sentinel either way).
4. The rank phase is textually the same loop: k exact UInt64 minima of the
   union, ties decided by the index half of the same composite key, the
   winner's value gathered from the same tile cell; its shift at CAP = K
   moves slots 1 .. K - 1 down and writes the sentinel at K - 1, which is
   the content slots 0 .. K - 1 had under CAP = 16 after the same shift.
5. Partitions shorter than one batch, the tail loop and the carved k-wide
   tail are the uniform arm's code with a shorter list: the same inserts in
   the same order, so no corner is argued separately.

### Sabotage (reach)

Nothing different: both arms are the uniform scan form and carry the
uniform arm's flip (bit 0 of the index half on `u == 0` of every batch,
inside the uniform loop), which proves the arm's own launch branch in
`_smallk_launch_bucket` and its loop ran, on the `[K, K, ...]` instantiation
that branch enqueues. A depth-specific perturbation (dropping every lane's
slot K - 1 after the scan) was considered and rejected: it flips only when
a row's whole top-k sits in one lane, which the hashed fixtures never
produce reliably. That the list is k deep is read from the stats leg's
register count, not from a flip.

### Expected effect (model; the stats leg and the gate replace it)

Registers. If the shipped k10 kernel carries lanes 10 .. 15 as twelve live
registers of the sentinel (the reading of the 54 / 56 pair: the list is 32
registers at both k, so the other state is about 22 to 24), `capk_k10` drops
to about 42, and at the 8-per-thread allocation unit that is 48 x 256 =
12,288 registers per block, 5 blocks per SM (62.5 percent); at 40 or fewer
it is 6 (75 percent). At k15 only one lane is dead (2 registers, 56 to 54,
still the 56 allocation), so `capk_k15` stays at 4 blocks unless ptxas
finds more; capk is a k10 experiment first. If `capk_k10` reports 54, ptxas
had already rematerialized the dead lanes and the register count is not the
list; then the follow-on is a storage form the compiler cannot widen (a
struct of K scalar slots, or `InlineArray[UInt64, K]` with constant indices,
neither of which can be added without changing the shipped path's list
type, so it would be a separate gated arm with its own control).

Time. The step 4 split says the scan is 8.8 ms at k10 (3.2 ms k-independent
floor, 5.6 ms the K-deep list) and 12.8 ms at k15 (9.6 ms list); the rank
phase is 0.4 to 0.75 ms. Occupancy helps only the latency-bound share, and
the floor (scanonly1 at eight blocks per SM) is the k-independent scan at
full occupancy, so the BEST CASE for capk is the list share shrinking in
proportion to the occupancy gain: 4 to 5 blocks, 5.6 x 0.8 = 4.5 ms, select
10.3 to about 9.1 ms at k10; 4 to 6 blocks, 5.6 x 0.67 = 3.7 ms, about 8.4
ms; a request-level saving of about 1 to 2 ms at k10 and none at k15. That
is the ceiling: the 3.2 ms floor and the k15 kernel's unchanged occupancy
mean capk alone cannot close the 2.6x / 2.9x gap to the cached cuML rows;
it is the cheapest bit-identical change that touches the register count
at all, which is why it is measured first. WORST CASE: no register change
(above), bit-identical, same time, and the arm is neutral; that result
retires the occupancy reading and leaves instruction count as the K cost.
For `capk_selp`: if ptxas already if-converted the per-step branch, the
SASS is the same and the arm is neutral; if it did not, each step loses a
branch and a reconvergence and the chain's 16 (now K) steps get cheaper by
a few instructions each, at most a modest fraction of the 5.6 / 9.6 ms
list share. Bracketed: neutral to about 1 ms at k10, neutral to about 2 ms
at k15. No default moves on any of this; the gate's numbers replace it.

### RUN OWED (orchestrator; nothing ran)

Commit this pass first (the leg ships `git archive` of the COMMITTED tree).
The arms check line is the regression fence for all seven arms and runs on
the K-specialized buckets (NVIDIA: the specialization row is on).

Run A, the mechanism verdict: the gate with phase timers ON, arms
`uniform,capk,capk_selp` (the first arm is the shipped default; each later
arm is paired with it; `select_ms` per arm is READ from
`phase_ms_median`, never inferred from request deltas; request medians on
that build are serialized and are NOT a price), profile skipped.

```
cat > /tmp/knn_capk_extra.sh <<'SH'
#!/bin/sh
export MOJOLEARN_KNN_SELECTION_ARMS=uniform,capk,capk_selp
export MOJOLEARN_KNN_SELECTION_PHASE_TIMERS=1
export MOJOLEARN_KNN_SELECTION_SKIP_PROFILE=1
cd /root/mojolearn && PATH="$HOME/.pixi/bin:$PATH" pixi run mojo run \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_SELECT_TRIAL=1 \
    -I . neighbors/checks/knn_selector_arms_check.mojo \
    > /root/gemm_leg_out/knn-selector-arms-check.log 2>&1
echo "arms_check_exit=$?" >> /root/gemm_leg_out/leg.txt
exec sh /root/mojolearn/tools/knn_selection_gate.sh
SH
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
MOJOLEARN_GEMM_LEG_EXTRA=/tmp/knn_capk_extra.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-knn-selection-capk \
sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" \
    --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
cat <leg>/remote/knn-selector-arms-check.log | tail -3
cat <leg>/remote/knn-selection/summary.txt
cat <leg>/remote/knn-selection/status.tsv
```

Verdict lines for run A: `KNN SELECTOR ARMS PASS` with
`SELECTOR_ARMS_REACH_PASS 65536 <k> <base> <uni> <hb> <wb> <df> <ck> <cs>`
all seven counts nonzero; in the gate JSON every fixture (large, dyadic,
ties, divergent_tail) at k10 and k15 shows `capk`, `capk_selp` and `default`
equal to `uniform`, row order, planted and oracle green, reach flipped > 0
on `uniform`, `capk`, `capk_selp` and `default` with clean bits restored;
then `phase_ms_median.select_ms` per arm, pooled and per order (the two
orders must agree within the pair spread or the row is noise), read
against the brackets above, and the `uniform` arm's own `select_ms`
against step 4 / step 5 (the control that the shipped kernel did not move).

Run B, the register numbers (independent of run A; can go first): the
stats leg, now ten rows.

```
MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
MOJOLEARN_GEMM_LEG_EXTRA=tools/knn_selector_kernel_stats.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date +%Y-%m-%d_%H%M%S)-nvidia-h100-knn-kernel-stats-capk \
sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
    --gpu "NVIDIA H100 80GB HBM3" \
    --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
cat <leg>/remote/knn-kernel-stats/stats.tsv
cat <leg>/remote/knn-kernel-stats/status.tsv
```

Read `capk_k10` / `capk_selp_k10` against `cap16_k10` (54 registers, 4
blocks per SM) and `capk_k15` / `capk_selp_k15` against `cap16_k15` (56, 4):
registers, `runtime_blocks_per_sm`, and that `local_bytes` and the local
counts stay zero (a CAP = K list that spilled would be a worse kernel, not
a smaller one).

Run C, the promotion run, ONLY if run A is green and `select_ms(arm) <
select_ms(uniform)` at both k for the arm in question: the same wrapper
WITHOUT `MOJOLEARN_KNN_SELECTION_PHASE_TIMERS` (an unserialized build,
request-level timing at the `NearestNeighbors.kneighbors` boundary), arms
`uniform,<arm>` for that one arm. `SMALLK_CAPK_DEFAULT` (and
`SMALLK_SELP_DEFAULT` for `capk_selp`) flips to True, and moves into the
kernel-matrix SCHEDULING row in the same session, ONLY IF on that run every
correctness check is green on every fixture and k, reach flipped on
`uniform`, the arm and `default` with clean bits restored, AND all eight
request-level timing cells (both orders' medians, `dyadic` and `large`, k10
and k15) favor the arm. A k10-only win (the model's likely shape for capk)
is a split verdict and leaves the default off, recorded here with the JSON
path; the register numbers from run B are recorded beside it either way.
After a flip: rebuild without the trial define, rerun the gate with `--arms
uniform` plus default to show the default equals the explicit arm, and
record the `dyadic` medians against the cached rows in
`bench/OPPONENT_REFERENCE.md` as cached-reference ratios (never as a paired
opponent measurement; cuML is not rerun). Apple and AMD columns take the
arm only with the specialization row or
`-D MOJOLEARN_KNN_IDENTICAL_SPECIALIZE_COMMON=1` in the build, and only
after their own gate runs; both RUN OWED before any column other than
NVIDIA takes the row.

## Step 7 result: CAP = K is NEUTRAL; occupancy is not the limiter (H100, 2026-09-11 05:45Z, `bench/results/e1g/2026-09-11_014146-nvidia` and `_014151-nvidia`)

Stats leg: capk_k10 40 registers, 6 blocks per SM, 75 percent occupancy
(from 54 / 4 / 50 percent); capk_selp_k10 48 / 5 / 62.5 percent; the k15
pair stays at 56 / 4 / 50 percent (the sixteenth slot was the only dead
one). No local memory anywhere. Mechanism run: seven arms bit-equal with
reach on every fixture; select_ms uniform 10.26 / 14.71 ms, capk 10.38 /
14.65 ms. A 50 percent occupancy gain at k10 bought nothing, so the scan
is not latency-bound on occupancy at this shape either. capk_selp was
not timed (the gate times only the first pair of --arms; every arm needs
a timed pair, fixed next).

What is left: the per-element instruction count of the K-deep chain
itself. Every arm that changed HOW OFTEN the chain runs (bounds, deferred)
or HOW MANY registers it takes (capk) measured neutral, which is only
consistent with the chain cost being paid on every element regardless
(the compiler if-converts the admission branch and the whole predicated
chain issues every step for every lane). The discriminating measurement is
a timing-only arm `noshift`: the K-deep list is present and the admission
compare runs, but an admitted key overwrites the last slot without the
shift (output invalid). select_ms(uniform) minus select_ms(noshift) is the
chain's own cost at k10 and k15; if it is the 0.80 ms per k, the fix is
a branch the compiler cannot if-convert around the chain (a warp-uniform
`vote.any` guard, which pays the chain only on steps where some lane
admits) combined with the branch-free selp chain inside it, and the
model says the admission rate per warp-step must be measured too
(a `vote` counter arm gives it).
