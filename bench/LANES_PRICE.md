# What IDENTICAL costs, lane by lane: the price harness

Status: **FIRST CLEAN-WINDOW MEASUREMENT TAKEN 2026-08-31, and it is on AMD,
not Apple.** MI325X, 5 alternated rounds, sha `035493e1`. Three of the six
lanes separate from 1.0; the other three do not, and are reported as no
separation rather than as a price. **APPLE AND NVIDIA ARE BOTH STILL OWED on
this harness**, so nothing here is a cross-vendor statement and no row ranks
a vendor.

Second status, 2026-09-02: **the six frozen lanes ran on an Apple M4 and an
NVIDIA H100 at one commit and MOST OF THE TABLE COULD NOT SEPARATE THE TWO
ARMS.** The Apple fixtures are far too small for a datacenter GPU and two of
the Apple rows compared a build against itself. That run is written up under
"2026-09-02: the run that could not separate the arms" below, and it is the
reason six more lanes and a size knob exist as of the same day.

Third status, 2026-09-02: **THE HARNESS PRICED NO TREE.** Twelve lanes and
not one tree fit, in the project whose largest certified family is decision
trees (109 of 180 bitwise-identical training configurations) and whose
performance section leads with them. Three tree lanes -- `gbdt`, `rf`, `et`
-- were added the same day at the 1,000,000-row floor and FIRST RAN on
2026-09-03 on a rented MI325X (`bench/results/lanes_price/SWEEP_2026-09-03/`).
Apple and NVIDIA columns are still OWED for them.

Files:

- `bench/lanes_price_main.mojo`, the driver: one lane per process, the lane's
  own public entry on the lane's own shipped fixture, ROUNDS timed rounds
  after one untimed warm-up, one `LPRICE` line per round.
- `tools/lanes_price.sh`, the window: builds the driver twice (FAST bare,
  IDENTICAL through `tools/with_identical_mode.sh`'s `mojo build` form),
  then per lane alternates the two binaries F I F I F I ... inside one
  window, reads the mode back from every leg, and writes the table.
- `bench/results/lanes_price/<timestamp>_<sha>[_SMOKE]/`, the evidence.

## What the harness measures

Each round is ONE call of the lane's public entry, the same call the lane's
`*_main.mojo` makes, on a fixture built once before the loop from the lane's
own fixture builder (imported, not re-spelled; the two transcriptions are
named in the driver's docstring). The clock is the host `perf_counter_ns`
around the call with a `ctx.synchronize()` on both sides, so a number is
wall seconds for the whole entry including its host work and its launches.
It is not device time and it is not a per-kernel number.

| lane | entry (what `*_main.mojo` calls) | shipped size (default) | SMOKE size | output hashed |
|---|---|---|---|---|
| cd | `solver.impl.solver.cd.cd_fit_traced` (Lasso, alpha 0.01, intercept, 1000 epochs, tol 1e-3) on `fixture_planted_sparse(n, d, 610)` | 2048 x 16 | 256 x 4 | coef, intercept, n_iter |
| kde | `kde.impl.kde.kde.score_samples` (gaussian, euclidean, weighted, h 2.75) on `train_fixture / query_fixture / weight_fixture` salt 1 | 1024 train x 256 query x 8 | 128 x 32 x 8 | the n_query log-densities |
| linkage | `hierarchy.impl.hierarchy.linkage.single_linkage` (pairwise, L2SqrtExpanded) on `FIX_BLOBS_DUPS` | 102 x 5, 3 clusters | `FIX_DUPS` 48 x 2 | children, labels, Boruvka rounds |
| svm | `svm.checks.svc_check._run_device` (svc_fit + svc_predict decision + classes) on `F2.xor`, the card fixture | 240 x 2 rbf | same (no smaller fixture exists) | decision function, classes, n_support, b |
| metrics | every ported metric in `metrics_main.mojo`'s order on its hashed fixtures, timed as one pass | labels 2053, floats 2053, silhouette 521 x 4, trust 301 x 6 | 257 / 257 / 67 x 4 / 61 x 6 | every returned value by its bits plus the silhouette samples |
| gemm | `gemm.checks.gemm_identical.identical_gemm` on one `bench/gemm_shapes.mojo` row | row 6 `kmeans.dist.4096x64x64` NT | row 4 `pca.transform.8192x4x4` | C |

Those six take NO size knob and must not gain one. The MI325X row below was
taken at their sizes, and a size that moves is a row that can no longer be
compared with it.

**Six more lanes, added 2026-09-02: the arms Section 7 of the paper prices.**
They were `bench/identity_price_main.mojo` and `bench/linalg_price_main.mojo`,
driven by `tools/price_unsupervised_identity.sh` and
`tools/price_linalg_identity.sh`, which have no remote-leg wiring and have
only ever run on an Apple M4. Ported into this driver so the same question
can be asked on Apple, NVIDIA and AMD. As of 2026-09-03 that is not a plan:
`tools/e2_remote_leg.sh`'s `lanes-price` check runs these lanes on a rented
MI325X and H100, and those columns are the ones the price is taken from,
because a single-tenant GPU has no laptop governor, no swap and no browser.
Three more lanes -- gbdt, rf and et -- joined on 2026-09-02 and first
compiled on 2026-09-03; the tree families had never been priced at all.

| lane | entry | what it prices | default size (Apple) | datacenter step | size knob | output hashed |
|---|---|---|---|---|---|---|
| kmeans | `cluster.impl.cluster.detail.kmeans.kmeans_fit_main` (INIT_ARRAY, n_init 1, max_iter 10, tol 1e-12) | the Lloyd loop end to end; pins only, no extra kernel | 100000 x 32, k 16 | 2000000 rows | `MOJOLEARN_LANES_PRICE_KMEANS_ROWS` | centroids, labels, inertia, n_iter |
| knn | `neighbors.impl.neighbors.detail.knn_brute_force.brute_force_knn_impl` (d 32, k 10, tile 256, is_sqrt) | the AUTO pin to TILED under IDENTICAL (DEVIATION 509), a kernel swap at k <= 64 | 20000 index x 1000 queries | 400000 index | `MOJOLEARN_LANES_PRICE_KNN_INDEX`, `_KNN_METHOD` | out_dist, out_idx |
| dbscan | `dbscan.impl.dbscan.dbscan.dbscan_fit_impl` (eps 1.2, min_pts 8, brute-force eps NN) | the brute eps neighbourhood plus propagation | 20000 x 8 | 100000 rows | `MOJOLEARN_LANES_PRICE_DBSCAN_ROWS` | labels, n_clusters |
| gram | `core.gemm.gemm_tn`, the Gram shape `A^T A` at the PCA/OLS aspect | on Apple the split-K PIN; off Apple the REPLACEMENT of the vendor matmul | 32 x 32 x 1000000 | 8000000 rows | `MOJOLEARN_LANES_PRICE_GRAM_ROWS` | Z |
| nt | `core.gemm.gemm_nt`, the N-T product | the vendor matmul against `pinned_gemm_nt_kernel`, one thread per cell (DEVIATION 526) | 4096 x 64 x 64 | 262144 rows | `MOJOLEARN_LANES_PRICE_NT_ROWS` | Z |
| gemv | `core.gemm.gemv_n`, OLS's step 6 | the vendor GEMV against `pinned_gemv_n_kernel`, one thread per output row | 128 x 128 | 8192 x 8192 | `MOJOLEARN_LANES_PRICE_GEMV_DIM` | Z |

`MOJOLEARN_LANES_PRICE_KNN_METHOD` is `auto` (the default and the shipped
dispatch) or `tiled` (the same arm asked for explicitly, and what `k > 64`
must take). Both are priced upstream and both are reachable here; the method
is written into the row's size field so the two never merge.

**Three tree lanes, added 2026-09-02, because there were none.** The twelve
lanes above cover clustering, neighbours, density, linkage, svm, metrics and
linear algebra and NOT ONE OF THEM FITS A TREE. Decision trees are the
largest certified family in this project (109 of 180 bitwise-identical
training configurations) and they are the workloads the performance section
leads with, so until today what identical mode costs a tree fit had never
been measured anywhere. Each lane fits on a fixture built IN the lane, out of
the driver's own `_price_u01`, because the tree benches that exist read HIGGS
or epsilon from `~/.cache/mojolearn` and a rented box has not staged them.

| lane | entry | what it prices | default size (Apple) | datacenter step | size knobs | output hashed |
|---|---|---|---|---|---|---|
| gbdt | `gbdt.train.train` (`grow_policy` SymmetricTree, border_count 254, lr 0.1, l2 1.0, Logloss with `leaf_estimation_iterations` 10, depth 6) | the symmetric (oblivious) boosting loop, plus the border build and quantization the raw-floats door does in front of it | 1000000 x 100, 20 trees | 2000000 rows, 60 trees | `MOJOLEARN_LANES_PRICE_GBDT_ROWS`, `_GBDT_TREES` | model bias, every tree's split list (feature, bin, type) and leaf values, the learn curve |
| rf | `ensemble.randomforest.fit_forest[ClsObj]` (bootstrap, `max_samples` 1.0, seed 20260821, `n_streams` 4, depth 12, `max_n_bins` 128, gini) | the quantile pass plus the pipelined forest loop | 1000000 x 50, 4 trees, 4 classes | 2000000 rows, 16 trees | `MOJOLEARN_LANES_PRICE_RF_ROWS`, `_RF_TREES` | per tree: depth/leaf counters, then DEVIATION 401's node fold (column, threshold bits, metric bits, left child, instance count) and `vector_leaf`. NOT `train_time` |
| et | `extratrees.impl.decisiontree.batched_levelalgo.builder.train_forest_classification_device` (`max_features` sqrt, seed 0x0F17, depth 12), on a `DeviceDataset` uploaded once before the loop | the merged device fit only; the upload is outside the clock | 1000000 x 28, 20 trees, 2 classes | 2000000 rows, 60 trees | `MOJOLEARN_LANES_PRICE_ET_ROWS`, `_ET_TREES` | the same node fold and `vector_leaf`, per tree |

**THE TREE ROW DEFAULTS ARE A FLOOR, NOT AN APPLE SIZE, AND THEY DO NOT MOVE
DOWNWARD.** `[[mojolearn-large-data-timing-floor]]` forbids measuring or
deciding tree performance below 1,000,000 rows, so the 2048-, 4096- and
20000-row defaults the six lanes above carry would be inadmissible here. The
reason is stronger than "too small to separate": under the floor a tree fit
is a DIFFERENT PROBLEM rather than a cheaper one, because the per-tree fixed
costs dominate. `bench/results/PERF_2026-08-20_fixed-cost.md` puts the
symmetric arm at 9.43 ms fixed plus 18.5 us per 1000 rows, which is 96% fixed
cost at 20,000 rows and 34% at 1,000,000. A 50,000-row symmetric window was
caught and killed on 2026-09-01 for exactly this and nothing from it votes
anywhere.

Every default above comes from a window this tree has already run at the
floor: et's 28 columns / 20 trees / depth 12 at 1M and 2M is verbatim the ET
arm of `bench/results/ab_large_2026-09-01/RESULTS.md` (5.9-6.3 s per fit at
1M), gbdt's 100 columns at depth 6 is `checks/estimation_bench.mojo`'s shape
with that window's patched `N_ROWS` (273.2 / 257.5 ms per tree in the d6 ll10
cell, so about 5.3 s at 20 trees), and rf's 50 columns / depth 12 / 128 bins
is `ensemble/bench/rf_bench.mojo`'s at that window's patched row count. **Only
the TREE counts were dialed, and only downward.** rf's window ran 30 trees at
22.3-26.4 s per fit, which is eight minutes of fitting for one lane at the
default ROUNDS; a forest is `n_trees` independent trees behind one shared
quantile pass, so four trees is the same work repeated fewer times and lands
near 3 s. et needed no dialing at all. The row count is the axis that changes
what the kernels are doing, and it is the axis that stays.

`MOJOLEARN_LANES_PRICE_SMOKE=1` gives the six new lanes tiny fixtures the
same way it does the six old ones: kmeans 2048 rows, knn 2048 index rows,
dbscan 2048 rows, gram 4096 rows, nt 256 rows, gemv 32 x 32; and the three
tree lanes 4096 rows with 2 trees each. As everywhere else in this harness a
SMOKE number is launch overhead and proves only the build, the mode witness
and the hash. **On the tree lanes a SMOKE number is not a price twice over**:
4096 rows is below the 1,000,000-row floor those lanes exist under, so it is
not a small price but an inadmissible one, and it may not be quoted, compared
or put in a table. An explicit size variable overrides the SMOKE size, which
is the same precedence `MOJOLEARN_LANES_PRICE_GEMM_SHAPE` already has.

Every default in that table is the size the RETRACTED Apple runs were taken
at; they are kept as the defaults only so a size is never implicit. The
datacenter column
is a STARTING POINT and not a measurement: it is the step to try first, and
if the band still straddles 1.0 the answer is a bigger fixture and not a
rerun. The k-NN and DBSCAN steps are not invented -- `bench/scaling_main.mojo`
already sweeps k-NN at 400,000 index rows through this same entry with this
same `buf_len` formula (`:180`, `:50`) and brute DBSCAN at 200,000 (`:185`).
DBSCAN's step stops below its own sweep because the brute arm is QUADRATIC in
the rows; that file puts brute at 800,000 near four and a half minutes per
repeat on the laptop, and this harness runs a warm-up plus ROUNDS rounds in
each of two modes.

The six new lanes took ONE protocol change from their sources, and it is
deliberate. `bench/identity_price_main.mojo` runs three repeats with NO
untimed warm-up; `bench/linalg_price_main.mojo` takes one warm-up and then
AVERAGES three timed reps into a single sample. Neither is carried over.
Every lane here does what cd and kde do: one untimed warm-up round, then
ROUNDS individually timed, individually hashed rounds. A row that kept its
source's protocol could not be read beside the rows above it.

Under FAST the same entry runs with its pins (`identical_mul_add`, `ftz`,
`identical_*`) compiled away; under IDENTICAL they are live. The two modes
are two binaries because `GLOBAL_NUMERIC_MODE` is comptime.

The hash is FNV-1a64 over the output bytes, the same function
`core/identity_trace.mojo` uses for the stage cards (byte at a time, little
endian), so a single-buffer lane's hash here equals its card's final-stage
hash where the card records that buffer whole. A price run is therefore also
a hash run: the driver prints `HASH-STABLE` or `HASH-MOVED` across its
in-process rounds and raises under IDENTICAL if the bytes moved; the script
repeats the comparison across legs and across the two modes.

## The mode witness

The driver prints the mode it COMPILED in, read from the comptime constant
(`_mode_name()`), in its header and on every `LPRICE` line. The script
asserts both against the mode the leg asked for and ABORTS the whole run
(exit 3) on a mismatch. Verified by sabotage: `MOJOLEARN_LANES_PRICE_SABOTAGE_SWAP=1`
hands the FAST binary to the IDENTICAL legs, and the run aborts with

    !! MODE MISMATCH (lane=kde round=1): asked for IDENTICAL,
    !! the binary's header says 'FAST'. ABORT. ...

(exit 3, observed 2026-08-23). The refusals were exercised too: with the
build lock held by another process the script exits 2 before building.

## How to run a CLEAN-WINDOW measurement

Nothing else on the GPU. The M4's GPU governor pins at MINIMUM clock under
heat (96% of an 11-second trace while 93.8% busy; up to 1.7x drift across
twenty minutes), so a FAST block followed by an IDENTICAL block measures
the drift, not the modes. The script alternates the two binaries per round
inside one window; that is the closest available thing to interleaving two
comptime modes, and the result is a BAND.

1. Confirm no other lane is running a leg (the script refuses if a `mojo`
   or `pixi` executable is running, or if another process holds
   `/tmp/cbsym-build.lock`; do not override with `ALLOW_BUSY` for a number).
2. Let the box idle a few minutes after any build.
3. Run, with the default five rounds and the shipped sizes:

       tools/lanes_price.sh

   (`MOJOLEARN_LANES_PRICE_ROUNDS=5` is the default; `MOJOLEARN_LANES_PRICE_LANES`
   narrows the lane list; the script holds the build lock for the whole
   run and writes `bench/results/lanes_price/<timestamp>_<sha>/`.)

   The six Section 7 lanes are opt-in and are NOT in the default lane list.
   At the Apple-sized defaults (small, and on a datacenter GPU too small to
   separate the arms):

       MOJOLEARN_LANES_PRICE_LANES="kmeans knn dbscan gram nt gemv" \
           tools/lanes_price.sh

   At the datacenter step, which is the run to take on a rented box:

       MOJOLEARN_LANES_PRICE_LANES="kmeans knn dbscan gram nt gemv" \
       MOJOLEARN_LANES_PRICE_KMEANS_ROWS=2000000 \
       MOJOLEARN_LANES_PRICE_KNN_INDEX=400000 \
       MOJOLEARN_LANES_PRICE_DBSCAN_ROWS=100000 \
       MOJOLEARN_LANES_PRICE_GRAM_ROWS=8000000 \
       MOJOLEARN_LANES_PRICE_NT_ROWS=262144 \
       MOJOLEARN_LANES_PRICE_GEMV_DIM=8192 \
           tools/lanes_price.sh

   The three tree lanes are opt-in the same way and are also NOT in the
   default lane list. At their defaults, which are already at the
   1,000,000-row floor and are the Apple run:

       MOJOLEARN_LANES_PRICE_LANES="gbdt rf et" tools/lanes_price.sh

   At the datacenter step, which is the run to take on a rented box:

       MOJOLEARN_LANES_PRICE_LANES="gbdt rf et" \
       MOJOLEARN_LANES_PRICE_GBDT_ROWS=2000000 \
       MOJOLEARN_LANES_PRICE_GBDT_TREES=60 \
       MOJOLEARN_LANES_PRICE_RF_ROWS=2000000 \
       MOJOLEARN_LANES_PRICE_RF_TREES=16 \
       MOJOLEARN_LANES_PRICE_ET_ROWS=2000000 \
       MOJOLEARN_LANES_PRICE_ET_TREES=60 \
           tools/lanes_price.sh

   The same variables set in `tools/diag/identity_cost_leg.sh`'s environment
   are forwarded to this script, so a rented-box leg asks for the lanes and
   the sizes without editing either file. With none of them set the leg does
   exactly what it did before: the six frozen lanes at their Apple sizes.
4. Read `ratio.tsv`: per lane, the median IDENTICAL seconds over the median
   FAST seconds, with `ratio_min`/`ratio_max` the per-round paired ratios
   (IDENTICAL_r / FAST_r). Report the median WITH the min/max band, the
   date, the sha, the machine (`env.txt` records host, CPU, load average at
   both ends, and whether a concurrent process appeared), and this
   sentence: measured on one M4 laptop whose governor drifts under heat,
   alternated per round inside one window, a band and not a figure.
5. Check the hash columns: `fast!=ident bits` is expected wherever a pin
   changes a rounding; `IDENTICAL HASH MOVED across legs` is a FINDING to
   report, not a row to drop.

A run by hand of one lane, one mode, for a look at the lines:

    MOJOLEARN_LANES_PRICE_LANE=kde tools/with_build_lock.sh \
        pixi run mojo run -I . bench/lanes_price_main.mojo
    MOJOLEARN_LANES_PRICE_LANE=kde tools/with_identical_mode.sh \
        pixi run mojo run -I . bench/lanes_price_main.mojo

No pixi task is registered for any of this.

## The price table

Median IDENTICAL / median FAST per lane at the shipped size, 5 alternated
rounds, one clean window.

**AMD MI325X, 2026-08-31, sha `035493e1`, 284 s of one lease.**

| lane | size | FAST median s | IDENTICAL median s | ratio (median) | ratio band (min .. max) | separates from 1.0 | bits |
|---|---|---|---|---|---|---|---|
| cd | 2048 x 16 | 0.001268 | 0.002593 | 2.045 | 1.940 .. 2.200 | YES | FAST != IDENTICAL |
| gemm | kmeans.dist.4096x64x64 | 0.000078 | 0.000109 | 1.401 | 1.282 .. 1.548 | YES | **FAST == IDENTICAL** |
| kde | 1024 x 256 x 8 | 0.000557 | 0.000711 | 1.277 | 1.245 .. 1.309 | YES | FAST != IDENTICAL |
| svm | F2.xor 240 x 2 | 0.002840 | 0.003102 | 1.092 | 0.869 .. 1.217 | no | FAST != IDENTICAL |
| linkage | blobs_dups 102 x 5 | 0.001587 | 0.001491 | 0.939 | 0.760 .. 1.220 | no | FAST != IDENTICAL |
| metrics | 2053 / 2053 / 521x4 / 301x6 | 0.001781 | 0.001658 | 0.931 | 0.886 .. 1.109 | no | FAST != IDENTICAL |

**THREE OF SIX LANES DO NOT SEPARATE, AND THEY ARE NOT A FINDING.** svm,
linkage and metrics have bands straddling 1.0, two of them with a median
BELOW 1.0. That does not mean identity is free on those lanes, and it must
not be quoted that way: it means this fixture at this size cannot tell the
two arms apart through the noise, which is what a band crossing 1.0 says.
The sizes are small -- linkage is 102 x 5 -- so launch overhead is a large
share of every number in those rows. A separated price for them needs a
bigger fixture, not a rerun.

**gemm IS NOT AN IDENTITY COST AND MAY NOT BE QUOTED AS ONE.** FAST and
IDENTICAL produce the SAME BITS on this box (`79adfe2dd5bd57e6` both). When
both modes return the same output bits the two arms are one answer, so the
ratio prices two BUILDS of that answer and not the price of identity. The
1.40x is a build-to-build difference on a lane where identity cost nothing in
bits; read it as a harness observation, not as `[[identity-is-not-free]]`.

The thermal caveat that belongs beside an APPLE row does not belong beside
these: this is a datacenter Linux box, not the laptop. `tools/lanes_price.sh`
printed the M4 drift note under these AMD numbers on the run that produced
them, and was fixed in the same session.

Nothing in this table ranks vendors: it holds ONE box. Apple and NVIDIA rows
are owed.

## 2026-09-02: the run that could not separate the arms

The six frozen lanes ran on an Apple M4 and on an NVIDIA H100 at one commit,
5 alternated rounds each. **Most of the table measured nothing, and the
reason is the fixtures, not the arms.**

- On the H100 the fixtures are Apple-sized and far too small. The cd lane
  took **1.3 ms** there against **89 ms** on the M4. The gemm lane took
  **100 microseconds** against 9.2 ms, and at 100 us the number is kernel
  launch overhead: its per-round ratio band came out **0.205 .. 0.773**,
  which is a band that never touches 1.0 from above and spends its width
  below it.
- **Four of six H100 lanes and five of six Apple lanes produced bands
  straddling 1.0.**
- Worse, on Apple the linkage and gemm lanes reported `fast==ident bits`.
  When the FAST arm produces the same bits as the IDENTICAL arm, the two
  arms are the same computation, the true ratio is 1.0 by construction, and
  every deviation from 1.0 in those rows is noise. Those rows measure
  nothing at all.
- Only `cd` separated on both boxes (1.248 Apple, 1.249 H100), and `kde` on
  the H100 (1.264).

THE RULE THIS RUN FIXES IN PLACE. **A band that straddles 1.0 says the
fixture cannot tell the two arms apart. It does not say identity is free.**
A fixture that cannot separate the arms is a fixture that is too small, and
that is a property of the harness and not a result about the library. The
100-microsecond gemm is the recorded example, and it is quoted in
`bench/lanes_price_main.mojo`'s header and in
`tools/diag/identity_cost_leg.sh` so the next person to read a straddling
band reads the rule beside it.

Note the difference between a straddling band and a `fast==ident bits` row,
because they are two different failures. A straddling band is a fixture that
cannot separate the arms. A `fast==ident bits` row is not an identity cost at
all, whether it separates or not: the two arms returned one answer, so the
ratio prices two builds of it. The AMD gemm row above separates cleanly at
1.40x and is still not a price of identity.

WHAT WAS DONE ABOUT IT, the same day: the six Section 7 arms were ported into
this driver (they had no remote-leg wiring of their own), and every one of
them takes its fixture size from the environment with the Apple size as the
default and a larger step named for a datacenter GPU. The six frozen lanes
were left exactly as they were, sizes and spellings both, so the MI325X table
above stays comparable.

**A SEPARATED PRICE FOR THE STRADDLING LANES NEEDS A BIGGER FIXTURE, NOT A
RERUN.** That is owed for the six frozen lanes too, and it is owed as a
harness change rather than as more rounds.

## Hashes from the smoke run (tiny sizes; hashes are not timing)

`bench/results/lanes_price/2026-08-23_125704_e69db89_SMOKE/`, sha e69db89,
ROUNDS=1 (one warm-up plus one timed round per leg), `MOJOLEARN_LANES_PRICE_SMOKE=1`.
Every lane built in both modes, every leg's header and every `LPRICE` line
read back the mode it was asked for, and every lane's hash was stable across
its two in-process rounds in both modes.

| lane | SMOKE size | FAST hash | IDENTICAL hash | same bits? |
|---|---|---|---|---|
| cd | 256 x 4 | 3be34440e79b3131 | 99d86c334a5c05cc | no |
| kde | 128 x 32 x 8 | 61bb0e73dc92f3b3 | 61bb0e73dc92f3b3 | yes |
| linkage | dups_lattice 48 x 2 | af1408d790e1a446 | af1408d790e1a446 | yes |
| svm | F2.xor 240 x 2 | 2def386e7266a86a | 9bec403e03e1a2cc | no |
| metrics | lab257 flt257 sil67x4 tru61x6 | 826b40eee60c57ee | 3bc0cdcca4a035cd | no |
| gemm | pca.transform.8192x4x4 | 5e18d6d88ce1cafd | 5e18d6d88ce1cafd | yes |

"Same bits" at the smoke size says only that on this Apple device, at this
size, the pins were bit-inert for that lane (Apple's FAST codegen contracts
and flushes too; IDENTITY_PATHS row 9). It is not a statement about another
vendor and not a statement about the shipped size.

The smoke output's head (from the run's stdout):

    == lanes_price: rounds 1, lanes [cd kde linkage svm metrics gemm], smoke 1 -> bench/results/lanes_price/2026-08-23_125704_e69db89_SMOKE ==
    == build FAST -> .../bin/lanes_price.FAST ==
    == build IDENTICAL (tools/with_identical_mode.sh, mojo build form) -> .../bin/lanes_price.IDENTICAL ==
    == lane cd: 1 rounds, FAST then IDENTICAL per round ==
       round 1: HASH-STABLE cd FAST 2 rounds all 3be34440e79b3131
       round 1: HASH-STABLE cd IDENTICAL 2 rounds all 99d86c334a5c05cc
    ...
    SMOKE RUN: tiny sizes, 1 round(s). The seconds above are launch
    overhead and prove only that both binaries run, the mode reads back,
    and the hash is stable. NO NUMBER HERE IS A PRICE.

One thing the smoke found and the driver now handles: `cdFit` MUTATES `x`
and `labels` in place under `fit_intercept` and `postProcessData` does not
restore the bits exactly (documented in `solver/impl/solver/cd.mojo`), so
the first smoke printed a warm-up hash for cd that differed from every later
round. The driver re-uploads `x` and `y` before every round; that is a
property of the entry and of the harness, not a deviation.

## Deviations

None spent. The block 700-704 is reserved for this harness and stays unused
unless a harness change moves bits. The 2026-09-02 work spent none either:
it adds nine callers of entries that already ship -- six ported from the two
Apple price mains and three tree lanes with no upstream price main to port
from -- and changes no library code, so no arm's bits move because of it.

## Hand-off

None required. The harness reads and imports from the lanes it prices and
edits none of them. Since 2026-09-02 that list also includes
`cluster.impl.cluster.detail.kmeans`, `dbscan.impl.dbscan.dbscan`,
`neighbors.impl.neighbors.detail.knn_brute_force` and `core.gemm`, all
imported at their public entries, and the three tree families: `gbdt.train`,
`ensemble.randomforest` (with `ensemble.decisiontree.*` for the parameter
block and the objective) and
`extratrees.impl.decisiontree.batched_levelalgo.builder` (with
`extratrees.estimator` for `max_features` resolution and
`extratrees.impl.randomforest.randomforest` for `class_ids_for`).

`DecisionTreeParams` is spelled once under `ensemble/` and once under
`extratrees/` and the two are DIFFERENT structs, so the driver aliases them
apart on import (`RfTreeParams` / `EtTreeParams`). The node folds are
likewise written out twice rather than shared, because
`ensemble/decisiontree/decisiontree.mojo`'s `TreeMetaDataNode` and
`extratrees/impl/decisiontree/flatnode.mojo`'s are two ports that agree on
their accessors and are not the same type. The ensemble one carries a
`train_time` member; it is a wall clock and is deliberately NOT hashed, since
folding it in would make every round disagree with every other round and the
harness would raise a false contract violation under IDENTICAL.

`bench/identity_price_main.mojo` and `bench/linalg_price_main.mojo` are kept
as the RECORD of the earlier protocol, and they are SUPERSEDED. The sentence
here used to say they "are not superseded on Apple: they are the files the
published Apple numbers came from." Those numbers were retracted on
2026-09-03 and deleted from `E3_RESULTS.md`, so there is nothing left for
them to be authoritative about.

The paragraph below already named the disagreement to watch for, and it
happened: those two mains time a cold first call, and this harness warms up
and times each round separately. That is a real protocol difference and it
is not the one that mattered. What mattered was the BOX. August's own source,
rebuilt in a detached worktree and run beside HEAD, measured the Gram arm at
26.33 ms against the 5.27 ms it had published, and HEAD measured the same --
on a 16 GB laptop carrying 7.5 GB of swap.

SO THE RULE IS NOT "PREFER THIS HARNESS." It is that a price run without a
recorded memory state is not evidence, whichever driver produced it. A loaded
box adds a fixed per-launch cost to BOTH arms, which pulls every ratio toward
1.0 and does it hardest on the smallest arm -- so it UNDERSTATES the tax and
is not a conservative bound on it. `tools/lanes_price.sh` now logs memory
pressure at both ends of every run for exactly this reason; the concurrency
gate and the load average both read CLEAN on the swapping box, because load
average does not see paging.
