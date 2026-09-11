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
