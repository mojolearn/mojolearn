# LANE STATUS: lane/knn-selector-speed (2026-09-17)

The k-NN selection class at large k, and the per-call rebuild of what a
resident index derives from its own bytes, on NVIDIA in IDENTICAL mode with
no output bit moved. Builds on lane/knn-tiled-distance (DEVIATIONs 3000 to
3003, landed on main as bde7a7b24; `docs/lanes/LANE_STATUS_lane-knn-tiled-distance.md`),
whose closing table left "the small-k selector at k 64 is 68 ms on both
datasets, the largest remaining class" as the next target. DEVIATIONs 3060
to 3063 (range 3060 to 3079). Written for a session with no memory of this
lane, from the branch's commits and the pulled pod evidence; the lane was
hard-stopped at the 2026-09-17 ~21:00Z pause while building its final arms,
with no resume note, and resumed from this reconstruction.

Evidence outside the repo: `~/mojolearn-evidence/knn-selector-speed/`
(`final_pull_pause/kss_out/` is the complete output of the first pod;
`pod/` its create response, arm log and reaped id; `pull1`, `pull2`,
`pull_asm`, `orchestrator_snapshot_2053Z` are earlier pulls of the same
tree; the finish's pod writes under `finish/`).

Box for the first pod: RunPod 0btpza2l3g1plm, one NVIDIA GeForce RTX 4090
(driver 580.159.04, 24,564 MiB, sm_89), 96 vCPU AMD EPYC 7K62, 251 GB, image
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04, Mojo 1.0.0
(ed45d567), $0.74 per hour, created 19:42Z, reaped 20:54Z (72 minutes, about
$0.89). Data staged from R2 (taxi_speed.npz, istella_speed.npz), cut by
`tools/classical_two_datasets.py prep`: the kNN blocks are the 400,000-row
index and 4,000 queries of each dataset (220 features Istella-S, 11 taxi),
seven column tiles of 65,536 (the last 6,784 wide), query tile 4,000.

## What was measured first: why the k 64 selector is 68 ms

`-D MOJOLEARN_KNN_PHASE_TIMERS=1` build of the branch point (the `phase`
arm; `MOJOLEARN_KNN_SELECT_TRIAL=1` builds take the selector arm from the
environment). Medians of 5 timed calls after a warmup, 4,000 queries, ms;
`distance` is the tile class (with the block top-k's rank loop at k <= 16),
`select` the selector class. `final_pull_pause/kss_out/attrib/`.

| arm | dataset | k | call | distance | select |
|---|---|---|---|---|---|
| phase (shipped) | istella | 10 | 54.2 | 39.7 | 1.44 |
| phase | istella | 32 | 77.7 | 31.9 | 18.76 |
| phase | istella | 64 | 130.5 | 31.7 | 68.03 |
| phase | taxi | 10 | 27.4 | 19.2 | 1.54 |
| phase | taxi | 32 | 63.8 | 13.7 | 28.61 |
| phase | taxi | 64 | 105.0 | 13.7 | 69.50 |
| trial uniform (the shipped selector form) | istella | 64 | 147.8 | 31.8 | 85.54 |
| trial skipscan (no per-thread scan, output invalid) | istella | 64 | 76.6 | 33.4 | 9.35 |
| trial skiprank (no rank rounds, output invalid) | istella | 64 | 153.7 | 31.8 | 80.31 |
| trial scanonly1 (a 1-deep per-thread list, output invalid) | istella | 64 | 67.8 | 31.7 | 6.90 |
| trial noshift (no list shift on pop, output invalid) | istella | 64 | 97.7 | 31.8 | 35.75 |

Taxi reads the same split (skipscan 9.33, scanonly1 7.00, noshift 38.17,
skiprank 76.75 at k 64). So the 68 ms is the SCAN of the small-k selector
(`select_smallk_identical_candidate.mojo::smallk_bucket_kernel`): every
thread walks its 256 columns of a 65,536-column tile keeping its OWN k
smallest composite keys (a 64-slot UInt64 list per thread, past what a
thread holds in registers, and a 64-step carry chain on nearly every
element); a 1-deep list scans the same cells in 7 ms. The k rank rounds
with a barrier each are the 35 ms difference between noshift and the
shipped form, but they sit inside the scan's time, not beside it. The
matrix write and read are not the cost (the distance class is flat in k).

## What changed

### DEVIATION 3060: the bound-and-compact selector (FLIPPED ON NVIDIA, k >= 17)

`neighbors/checks/knn_selector_bound_compact.mojo` (new), kernel-matrix
row `knn_selector_bound_compact_for` and `KNN_SELECTOR_BOUND_MIN_K = 17`
(`checks/kernel_matrix.mojo`), wired in
`neighbors/impl/detail/knn_brute_force.mojo::_tiled_brute_force_knn_impl`
where `smallk_select_launch` served a column tile;
`select_smallk_identical_candidate.mojo` gains the `FLAGGED` instantiation
of `smallk_bucket_kernel` and `smallk_flagged_launch`.

One block of 256 threads per row of the distance tile, the small-k
selector's block shape and key (`composite_key(distance, tile-local
column)`), four phases and four barriers, none per rank: SCAN, every
thread keeps its C = 8 smallest keys (a branch-free min/max carry chain in
registers); BOUND, the 256 thread minima go to shared memory, every thread
ranks its own by counting, the thread of rank k - 1 publishes its minimum
as the bound B; COMPACT, every thread counts its keys at or below B and
says whether its full list ends below B (a hiding thread), and with no
hiding thread and at most 256 candidates the candidates go to shared
memory at prefix-sum offsets; RANK, thread j ranks candidate j by counting
and writes the index half and the tile cell it names at that rank. A row
with a hiding thread or more than 256 candidates raises `flags[row]` and a
second launch of the UNCHANGED small-k kernel (`FLAGGED=True`, a block
whose flag is 0 returns at its first statement) serves it.

WHY NO BIT MOVES. Keys are unique (they carry the column). B is a key of
the row and at least k distinct keys (the k thread minima of rank 0 to
k - 1, from k different threads) are at or below it, so any key above B has
k keys below it and is not among the row's k smallest. A thread's keys at
or below B are all in its list unless the list is full below B, which is
the hiding test; so on the fast path the candidate buffer holds EVERY key
at or below B, and a candidate's rank among the candidates (a count of
strictly smaller unique keys) is its rank in the row. The output at rank r
is the key's index half and `values[row, index]`, the cell the small-k
selector gathers; distance bits are copied, never computed. Every
reduction is an integer count, an integer sum or a UInt64 compare. Flagged
rows run the small-k selector's own kernel statement for statement.

Depth (`SBC_DEPTH`), measured on the RTX 4090 at 400,000 x 4,000, selection
class of a timer build (`final_pull_pause/kss_out/tune/`): depth 8 reads
8.0 ms with 0 of 28,000 row tiles flagged on both datasets at k 17, 32 and
64; depth 4 reads 13.2 (Istella-S) and 13.6 ms (taxi) with 1,147 and 1,276
flagged at k 64 (7.8 ms of it the scan and 5.4 the flagged launch, by the
`_TIMING_NOFLAG` arm); depth 2 flags nearly every row and reads 88 to 93
ms. Below k 17 the selector reads 7.7 to 8.1 ms against the small-k
selector's 7.1 to 13.6 (`tune/probe_bcphase_allk`, `tune/probe_phase_ms`)
and the L2 metrics take the block top-k there anyway, so the bound stays at
17. Kernel footprint (`checks/stats_neither.txt`): 40 registers, 5,128
bytes shared, 6 blocks of 256 per SM, against the tile kernel's 128
registers.

### DEVIATION 3061: a resident index keeps its derived buffers (FLIPPED ON NVIDIA)

`neighbors/impl/detail/knn_brute_force.mojo` (`KnnIndexCache`,
`KnnIndexCachePointer`, `cached_index_norm_ready`, the cached branch of
`tiled_brute_force_knn`), `neighbors/resident_index.mojo` (the entry owns
a `KnnIndexCache`, dropped before its context), `neighbors/estimator.mojo`
(`_knn_search_on_device_index` and the three resident entries take the
cache), kernel-matrix row `knn_resident_derived_cache_for`.

A resident k-NN index (DEVIATIONs 2921 and 3002) keeps, beside its
uploaded bytes, the transposed layout (352 MB at 400,000 x 220), the index
row norms of the metric (`norm_takes_sqrt` names which metric's; the other
metric rebuilds), and DEVIATION 2629's per-row admission metadata. Each
is built on the first search that needs it by the SAME kernel over the
SAME device bytes (`transpose_kernel`, `compute_norms_for_metric`,
`vector_exponent_admission_kernel`) and read by every later search. WHY
NO BIT MOVES: the device copy of a resident index is never written after
its upload and a refit releases the handle, so a later search reads the
values it would have computed. Attribution (`cachephase/`, one Istella-S
query, timer build): the per-call allocation of the transposed layout, its
transposition, the index norms and the admission scan were 6.5 of an 8.2
ms call; with the cache the call is 1.65 ms.

### DEVIATION 3062: the block top-k's rank loop bounded by the running top-k (candidate, OFF, LOST)

`smem_distance_tile_kernel[BOUNDED=True]` (`neighbors/checks/smem_distance_tile.mojo`),
`bound_compact_lists_kernel` and `bound_compact_lists_launch`
(`knn_selector_bound_compact.mojo`), `partial_keys_select_kernel[TERMINATED=True]`,
the ABSENT-slot rule in `partial_topk_merge_kernel`, kernel-matrix row
`knn_block_topk_bounded_for` and `KNN_BOUNDED_FIRST_TILE`. On every column
tile after a query tile's first, the tile kernel reads each row's k-th
RUNNING distance and never builds a key at or above it, leaves the rank
loop when no row of the thread row has a key left, and terminates the
list with a sentinel; the lists are selected by the bound-and-compact
phases and the merge skips absent slots. The argument (column tiles ascend,
so a key at or above the k-th running distance has k smaller composite
keys among the running entries and the merge would drop it) held: every
digest equal to the matrix arm's at k 1, 10, 17, 32 and 64 on both
datasets (`bounded/`, `bounded2/`, `bounded3/`). It LOST on time: the
distance class of the Istella-S k 10 call went 42.4 ms (DEVIATION 3061
alone) to 49 to 50 ms, and at k 64 to 57 to 60 against the matrix arm's
33.4 + 8.1 select; on taxi the k 64 call read 64 to 69 ms against 43.9.
The diagnostic arms (`bounded4/`) name the cost: with neither the bound
nor the early leave the kernel is back at 32.8 ms (k 1 Istella-S); the
bound load and compare alone 46.6, the early leave alone 48.8, both 46.2 to
50.2. The kernel footprint says why (`checks/stats_nobreak.txt`,
`asm_key64.txt`): the bounded instantiation holds 137 to 164 registers
against the unbounded 128 and drops from 2 blocks of 256 per SM to 1. A
narrower first tile (`_FIRST_TILE_8192`) did not recover it. Stays off on
every column; the row's docstring records the loss.

### DEVIATION 3063: the rank loop on 32-bit distance halves (candidate, OFF, LOST)

`smem_distance_tile_kernel` under `SMT_KEY32`, kernel-matrix row
`knn_block_topk_key32_for`. The thread keeps 32 UInt32 distance halves and
a 32-bit mask of empty slots instead of 32 UInt64 keys, rebuilding the key
it writes from the half and the slot's column; the pop order is the
composite order (lowest lane among equal halves, first live slot among
equal halves in a lane). Meant to free registers for DEVIATION 3062. It
did (118 registers unbounded, 124 bounded, 2 blocks per SM in both;
`checks/asm_key32.txt`) and still LOST on time (`key32/`): alone, the
Istella-S distance class at k 1 read 42.3 ms against 32.8 and at k 10 49.2
against 42.4; with the bound, 42.1 to 55.6 ms on Istella-S. On taxi with
the bound it won at k 10 (19.2 against 24.9 per call) and lost at k 1
(14.3 against 12.0). The slot-mask tests in the pop cost more than the
registers bought. Stays off on every column.

## Identity, sabotage and the k race (first pod, arms built from 0a582fe8d)

The `bcc` arm is the branch at 0a582fe8d with `-D MOJOLEARN_EXPERIMENTAL_KNN_SELECTOR_BOUND=1
-D MOJOLEARN_EXPERIMENTAL_KNN_RESIDENT_CACHE=1`, which is what the tip
ships on NVIDIA by default; `bc` the selector alone; `after0` the branch
with both off; `base` the branch point (main 86d33fcdf, shipped as
c7a117d88 whose kernels are main's). `tools/identity_break.py`, 15 lanes
(knn, knn-chebyshev, knn-clf, knn-clf-distance, knn-cosine, knn-manhattan,
knn-minkowski-p3, knn-rbc, knn-reg, knn-reg-distance, knn-sqeuclidean,
radius, radius-chebyshev, radius-manhattan, radius-minkowski-p3), fixtures
base, ties, odd, dupes, wide, two repeats (`identity1/`):

| diff | infer/model | batch | exit |
|---|---|---|---|
| cuda-base vs cpu-base (before) | IDENTICAL 150 | IDENTICAL 75 | 0 |
| cuda-bcc vs cpu-after (after) | IDENTICAL 150 | IDENTICAL 75 | 0 |
| cpu-base vs cpu-after | IDENTICAL 150 | IDENTICAL 75 | 0 |
| cuda-base, cuda-bcc, cuda-bcallms | IDENTICAL 150 | IDENTICAL 75 | 0 |
| cuda-bcallms vs cuda-bcallms_sabo (selector control) | DIVERGENT 50, IDENTICAL 100 | DIVERGENT 50, IDENTICAL 25 | 1 |
| cuda-bcc vs cuda-bccsabo (cache control) | DIVERGENT 30, RELOAD-MOVED 30, IDENTICAL 90 | DIVERGENT 30, IDENTICAL 45 | 1 |

The identity fixtures search 4,096 rows at k 8, below the selector's k 17
and on the block top-k for the L2 metrics, so the default build does NOT
reach DEVIATION 3060 there: `bcallms` adds `-D MOJOLEARN_KNN_SELECTOR_BOUND_ALL_K=1
-D MOJOLEARN_KNN_IDENTICAL_MATRIX_SELECT=1` (every k, every metric through
the matrix and the new selector) and its sabotage (`-D MOJOLEARN_KNN_SELECTOR_BOUND_SABOTAGE=1`,
bit 0 of the rank-0 index) diverges on the 10 knn lanes on every fixture
and leaves the 5 radius lanes IDENTICAL (no top-k). The cache sabotage
(`-D MOJOLEARN_KNN_RESIDENT_CACHE_SABOTAGE=1`, feature 0 of index row 0 of
the CACHED transposed layout becomes 1e30 on the second search) diverges
on the 6 lanes whose search runs the transposed IDENTICAL arm (knn,
knn-sqeuclidean, knn-clf, knn-clf-distance, knn-reg, knn-reg-distance) and
leaves the metric lanes IDENTICAL, the dispatch witness. Both controls
were seen to fail before the passes were read. `neighbors/checks/knn_selector_bound_compact_check.mojo`
(168 cases, six row kinds including a hiding thread and 65,537 columns,
host oracle and the small-k selector) PASSES at depth 8 with 42 flagged
rows and FAILS under each sabotage define (`checks/`).

At the 400,000 x 4,000 blocks the selector IS reached by default: the race
below compares the sha256 of the full distance and index arrays across
arms at k 32 and 64, equal on both datasets, and `sabotage_large/` shows
the selector sabotage moving those digests at k 32 and 64 on both.

Race (`bench/speed/knn_selector_race.py`, `race1/race_summary.tsv`): one
process per arm per dataset per round, order rotated, 5 rounds x (1 warmup
+ 3 timed) calls, host arrays in and out, digests equal across arms,
rounds and calls on every cell. Paired ratio is the median over rounds of
the round-median ratio to `base`; `u` marks a spread over 1.10 and no
ratio is quoted from it. 4,000 queries:

| dataset | k | base ms | bc ms | bc ratio | bcc ms | bcc ratio |
|---|---|---|---|---|---|---|
| istella | 1 | 42.68 | 42.70 | 1.001 | 35.12 | 0.823 |
| istella | 10 | 53.73 | 54.06 | 1.006 | 47.65 | 0.887 |
| istella | 17 | 70.58 | 60.83 | 0.863 | 55.56 | 0.781 |
| istella | 32 | 76.52 | 65.83 | 0.860 | 58.48 | 0.769 |
| istella | 64 | 130.31 | 69.57 | 0.534 | 62.18 | 0.478 |
| taxi | 1 | 12.18 (u) | 12.22 | 1.007 | 11.70 | 0.960 |
| taxi | 10 | 26.86 | 26.74 | 0.996 | 24.25 | 0.902 |
| taxi | 17 | 53.71 | 35.88 | 0.668 | 35.87 | 0.663 |
| taxi | 32 | 58.15 | 39.33 | 0.677 | 39.09 | 0.672 |
| taxi | 64 | 98.79 | 42.15 | 0.427 | 41.08 | 0.419 |

One query (the floor): Istella-S base 8.07 to 9.46 ms across k, bcc 1.47
to 1.74; taxi base 1.59 to 3.55, bcc 0.94 to 1.36 (several one-query
cells read `u` on the shared pod; the floor probe in `cachephase/` gives
1.65 and 1.13 ms at k 10 with spreads under 1.05). Geometric mean of the
bcc ratio over the two datasets: k 10 0.894, k 64 0.448, and no shape
regresses at 4,000 queries (the smallest gain is taxi k 1 at 0.960).

## What the first pod did NOT do (owed at the resume)

The tip 1dac78afb flipped the two rows and added the 3063 code; its
no-define arms (`final`, `final_selsabo`, `final_cachesabo`, `final_allk`,
`final_allk_sabo`) were built between 20:49Z and 20:54Z from a working
tree pushed 30 s before the commit, and NOTHING ran on them before the
reap at 20:54Z. So at the resume the tip had no identity column, no
sabotage seen, no race and no large-k digest of its own; every number
above is from arms built at 0a582fe8d with the rows forced by define. The
finish (next section) runs them all on the committed tip against the
branch point. The kernel-matrix docstrings of 3062 and 3063 said "OFF on
every column until measured"; they were measured and lost, and this
commit's docstrings say so.

## Commands

```
# Mac, from the lane worktree (commit first)
export MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key
export TREES_LEG_STATE=$HOME/mojolearn-evidence/knn-selector-speed/pod
TREES_LEG_NAME=mojolearn-knn-selector-speed TREES_LEG_CUDA_VERSIONS=13.0 \
  MOJOLEARN_STAGE_KEYS="gbm-bench/taxi/taxi_speed.npz gbm-bench/istella/istella_speed.npz" \
  sh tools/trees_leg.sh rent --gpu "NVIDIA GeForce RTX 4090" --minutes 150
git archive --format=tar 86d33fcdf -- . ':!bench/results' ':!mamba/corpus' ':!bench/oracle*' ':!archive' ':!upstream' ':!docs' ':!paper' \
  | gzip | sh tools/trees_leg.sh ssh 'rm -rf /root/mojolearn-base && mkdir -p /root/mojolearn-base && cd /root/mojolearn-base && tar xzf -'
sh tools/trees_leg.sh ssh 'apt-get update && apt-get install -y rsync'
# pod, from /root/mojolearn (tools/knn_selector_body.sh; each stage writes /root/kss_out/<stage>)
sh tools/knn_selector_body.sh setup
sh tools/knn_selector_body.sh arm final ""; sh tools/knn_selector_body.sh hostafter
sh tools/knn_selector_body.sh arm final_allk "-D MOJOLEARN_KNN_SELECTOR_BOUND_ALL_K=1 -D MOJOLEARN_KNN_IDENTICAL_MATRIX_SELECT=1"
sh tools/knn_selector_body.sh arm final_allk_sabo "... -D MOJOLEARN_KNN_SELECTOR_BOUND_SABOTAGE=1"
sh tools/knn_selector_body.sh arm final_selsabo "-D MOJOLEARN_KNN_SELECTOR_BOUND_SABOTAGE=1"
sh tools/knn_selector_body.sh arm final_cachesabo "-D MOJOLEARN_KNN_RESIDENT_CACHE_SABOTAGE=1"
sh tools/knn_selector_body.sh identity identity2 base; ... final; final_allk; final_allk_sabo; final_cachesabo
sh tools/knn_selector_body.sh cpuidentity identity2 base; sh tools/knn_selector_body.sh cpuidentity identity2 after
sh tools/knn_selector_body.sh diff identity2 <name> <json...>
sh tools/knn_selector_body.sh race race2 base,final 1,10,32,64 4000,1
sh tools/knn_selector_body.sh probe sabotage2 final_selsabo KSS_KS=32,64 KSS_ROWS=4000
sh tools/trees_leg.sh pull /root/kss_out ~/mojolearn-evidence/knn-selector-speed/finish/
```

## The finish ran: the committed tip has its own columns (2026-09-18)

Pod `uywhryt7b9vtz4`, one RTX 4090 (driver 580.159.04, sm_89, 24564 MiB),
96 vCPU AMD EPYC 7642, 251 GB, $0.74/h, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, Mojo 1.0.0
(ed45d567). Shipped tree `cb6520f7a4f23db966108752bbab993b8b2aa36d`, the
lane tip, base arm from `origin/main`. 07:17:07Z start, 07:30:25Z done,
13 minutes for the whole sequence. Raw JSON, consoles and logs are outside
the repo under
`~/mojolearn-evidence/knn-selector-speed/finish-2026-09-18/kss_out/`.

Six arms built on the tip with NO defines except the arms' own
(`base`, `final`, `final_allk`, `final_allk_sabo`, `final_selsabo`,
`final_cachesabo`), plus the two host families before and after. This is
what the first pod never did: every number in the sections above came from
arms built at `0a582fe8d` with the rows forced by define, and the tip had
no column of its own.

### Identity (`kss_out/identity2`)

| comparison | train | infer/model | batch |
|---|---|---|---|
| base vs final (the two flipped rows) | IDENTICAL=75 | IDENTICAL=150 | IDENTICAL=75 |
| base vs final_allk (bound at every k + matrix select) | IDENTICAL=75 | IDENTICAL=150 | IDENTICAL=75 |
| final vs cpu-after | IDENTICAL=75 | IDENTICAL=150 | IDENTICAL=75 |
| cpu-base vs cpu-after | IDENTICAL=75 | IDENTICAL=150 | IDENTICAL=75 |
| final_allk vs final_allk_sabo | **DIVERGENT=50**, IDENTICAL=25 | **DIVERGENT=50**, IDENTICAL=100 | **DIVERGENT=50**, IDENTICAL=25 |
| final vs final_cachesabo | **DIVERGENT=10**, IDENTICAL=65 | **DIVERGENT=30**, IDENTICAL=90, RELOAD-MOVED=30 | **DIVERGENT=30**, IDENTICAL=45 |

DEVIATIONs 3060 and 3061 move no bit on their own tip, on the GPU column
and against the CPU column, and BOTH sabotage arms bite, so the two
IDENTICAL rows above are not vacuous.

### The k race (`kss_out/race2/race_summary.tsv`, paired, digests equal)

Every cell's `digests` column reads `equal`; the ratio is final/base.

| dataset | k | rows | base ms | final ms | ratio |
|---|---|---|---|---|---|
| istella | 1 | 4000 | 42.680 | 34.896 | 0.819 |
| istella | 10 | 4000 | 53.476 | 45.614 | 0.856 |
| istella | 32 | 4000 | 72.263 | 55.544 | 0.765 |
| istella | 64 | 4000 | 124.441 | 55.949 | 0.453 |
| taxi | 1 | 4000 | 12.068 | 11.497 | 0.953 |
| taxi | 10 | 4000 | 24.957 | 23.639 | 0.943 |
| taxi | 32 | 4000 | 54.531 | 35.496 | 0.651 |
| taxi | 64 | 4000 | 94.057 | 36.215 | 0.384 |

The one-query rows (`rows=1`) all improve too (0.17x to 0.63x) but most
carry the `u` gate: at 1 to 9 ms a round the spread is the launch
sequence, not the kernel, and no claim rests on them.

### A defect in the finish runner, and its repair

`/root/finish.sh` wrote each identity column as `cuda-<arm>.json` and then
asked `diff` for `<arm>.json`. Five of the six diffs died with
`FileNotFoundError` and the runner printed an EMPTY summary for each,
which reads exactly like a clean comparison in the console log: the
per-diff line was `diff.base-vs-final.txt:` with nothing after it. The
failed tracebacks are preserved in the pulled tree. The five diffs were
re-run on the same pod against the same JSONs with the real filenames
before it was reaped, and those are the numbers in the table above. A
summary line that is EMPTY is not a summary line that says IDENTICAL.

Pod terminated after the pull, DELETE 204 / GET 404 at 07:34Z. No kNN
rental remains.

## Owed and not done

- The tip's own identity, sabotage and race columns are NO LONGER OWED; see
  the finish section above.
- Apple and AMD columns of DEVIATIONs 3060 and 3061 at the next release
  record; the rows are NVIDIA only and the kernels compile on every column
  (integer counts and UInt64 compares; the flagged launch is the small-k
  kernel), untimed and unverified there.
- The one-query Istella-S floor is now the launch sequence alone (about 1.6
  ms); the 4,000-query call at k 10 on taxi (24 ms against cuVS's 6.9,
  `bench/OPPONENT_REFERENCE.md`, RTX 4090 section) is the tile's per-cell
  epilogue and the seven column tiles at d = 11, not selection.
- The block top-k above k 16 stays on the matrix and this selector; the
  bounded rank loop (3062) and the 32-bit loop (3063) lost on the RTX 4090
  and would need a different register budget (a smaller thread tile) to be
  tried again.
