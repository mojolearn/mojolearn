# Flagship gaps: t-SNE, DBSCAN, HDBSCAN (read-only analysis, main @ 42d1e42c6, 2026-10-09)

Board cells (nvidia-l40s.md:1728-1773): tsne istella 4,220.7 ms vs cuML 454 (9.3x), taxi 2,688.8 vs 397.5 (6.8x);
dbscan istella 542,218 vs cuML 66,379 (8.2x); dbscan taxi REFUSED (cuML 440,431); hdbscan REFUSED on every box
(cuML 862 / 369). Both REFUSED cells are STALE refusals of an older wheel, not current defects (details per lane).

## Shapes the board uses
- tsne (tools/bench_board_algos.py:1149, block "manifold" :1597): 20,000 stride rows, standardized; istella d=220, taxi d=11.
  perplexity 30 -> k = 3*30+1 = 91 neighbours (x_ann/tsne_core.mojo:520); max_iter 1000; exploration 250 iterations at
  exaggeration 12, momentum 0.5 then 0.8; learning_rate auto = max(n/12/4, 50) = 416.7; init = the seeded (n,2) array
  (no PCA). cuML arm: method="fft", 1000 iterations, learning_rate_method adaptive, its own random init.
- dbscan (tools/classical_two_datasets.py:215, :1832): 1,000,000 standardized rows (taxi d=11, istella d=220), eps=3,
  min_samples=2, ours algorithm='rbc' (our default, DEVIATION 35); cuML 'brute' (cuml-gpu-rbc "overflows at 1M rows",
  bench_board.py:288). Istella result: 40,131 clusters, 21.9% noise (sparse graph). Taxi: ~1e12 edges, nearly complete
  (docs/apple-fast/notes/dbscan-taxi.md). Round ceiling 3600 s istella / 1800 s taxi (bench_board.py:342).
- hdbscan (classical_two_datasets.py:216, :1918, :1977): the dbscan block's first 100,000 rows, min_samples=10 (k=11 incl.
  self), min_cluster_size=100, EOM, alpha 1, max_cluster_size=0.

# t-SNE

## Fit path
python/mojolearn/_expansion_ann.py:362 TSNE.fit -> binding x_ann_tsne_fit (bindings/_mojolearn_x_ann.mojo:100-122:
`in_f32` copies X into a host List, n*d host loop) -> x_ann/tsne_device.mojo:621 `tsne_fit_device`.

Stages (tsne_device.mojo:621-713):
1. upload X (17.6 MB istella; trivial), 3 n*91 buffers.
2. kNN, exact, x_ann/knn_device.mojo:301 `knn_enqueue`: istella d=220 -> `knn_wide_kernel` (NV/AMD, 64x64 tile, 16-feature
   chunks, 4x4 register cells, grid n/64 = 313 blocks x 256 threads); taxi d=11 -> `knn_tiled_kernel[12]` (one thread per
   row, 64-candidate shared tiles). BOTH end each tile with a serial owner phase: 64 owner threads (192 idle) each offer the
   tile's 64 candidates to the row's sorted top-91 list IN GLOBAL MEMORY (`ts_knn_offer`, tsne_core.mojo:121: insertion
   shift of up to 91 (dist, idx) pairs per accepted candidate). Expected accepted offers per row ~ k ln(n/k) = 490, ~45
   shifts each -> ~9e8 scattered 8-byte stores for the fit, 64-way parallel per block only.
   Distance work 4e8 pairs x 220 = 8.8e10 MAC (~5 ms at tile rates): the kNN stage cost is the insertion, not the FMAs.
3. `perplexity_kernel` (one thread per row, sklearn's 100-step bisection over 91 entries, ~2e8 exp): < 5 ms. sync.
4. `tsne_symmetrize_device` (x_ann/tsne_sym_device.mojo:141): ~7 launches, 1 sync; P stays on the device.
5. 1000 iterations x (`repulse_split_kernel` + `step_rows_kernel`) = 2000 launches, ZERO host syncs (TS_SPLIT, NV/AMD).
   Repulsion is EXACT N^2 (no Barnes-Hut, no FFT): tsne_device.mojo:312: 32 rows x 64 candidates per tile, 8 pairs per thread
   per tile (RS_Q), 2 barriers per tile, 313 tiles per block, grid 625 blocks x 256 threads (~0.5 wave on an L40S); the
   64-lane fold (TS_LANE_FOLD, tsne_core.mojo:44-66) keeps lane s = j mod 64 ascending; Z is folded by the last block
   (atomic ticket, fixed tree) in the same launch. Per pair: 2 sub, 2 fma, 1 add, 1 IEEE division (`ts_recip_den`),
   3 mul, 3 add, ~6 ftz flushes ~ 25-30 lane instructions.
6. `kl_kernel` + `device_sum_f32_fixed` + 2 downloads: 2 syncs. KL is read ONCE at the end (not per iteration).

## Cost model
- Iterations: 4e8 pairs x 1000 x ~28 instr = 1.1e13 lane-instr; L40S lane-instruction peak ~2.7e13/s -> ~0.45 s at peak.
  Measured: taxi fit 2,689 ms with a ~0.3 s kNN -> ~2.3 ms per iteration, ~5x off peak. Causes visible in the code: a
  barrier pair every 8 pairs per thread (313 tiles x 2 per block per iteration), every y read from shared memory per pair
  (no register reuse of y_i across candidates: the lane layout e = tid + q*256 gives each thread 8 DIFFERENT rows), the
  IEEE division, and 6 ftz calls per pair.
- kNN: the istella-minus-taxi difference (1,532 ms) is the only d-dependent stage, so the wide kNN (distance tile + serial
  global-memory insertion) costs ~1.5 s on istella and ~0.3 s on taxi. No stage log exists; MOJOLEARN_ANN_STAGES=1
  prints "ANN-STAGE tsne_iter repulse/sum/step" and the kNN stage (x_ann/stage_timer.mojo:20) and should be run first.
- cuML (FIT-SNE, method="fft"): O(n) per iteration (charges to an interpolation grid, FFT convolution, interpolation back)
  ~0.4 ms per iteration, launch-bound; kNN by cuVS brute force (GEMM + select). 454 / 397 ms total.
- Gap: ~2.3 s of exact N^2 repulsion (both datasets) + ~1.5 s kNN insertion (istella).

## Ideas (t-SNE)
T1. kNN top-k in shared memory, `-D MOJOLEARN_IDN_TSNE_KNN_TILE_SELECT`. Keep the 64x64 distance tile; per tile sort the
    64 candidates of each row in shared memory ((dist, j) keys, bitonic, all 256 threads) and merge with the row's current
    top-91 held in shared memory (91+64 -> 91), write the list to global once per row at the end. The top-k SET and its
    (dist, j) order are unique, so P and every later word are bit-identical. Effect: ~1.5 s -> ~50 ms on istella
    (4,220 -> ~2,750, 9.3x -> ~6x), taxi -0.25 s. Bits none. Identity risk none. Effort M. Opus: yes.
T2. Repulsion tile and register blocking, `-D MOJOLEARN_IDN_TSNE_REP_TILE`. 256 candidates per staged tile (4 lane groups),
    each thread owns ONE row (its y in registers) and 4 lanes, processing lane s over j ascending (j mod 64 = s) across the
    4 sub-tiles in order, so the lane partials are today's words; barriers per block per iteration 626 -> ~160; hoist the
    row's y loads; keep the division and flushes. Effect: 2.3 -> ~1.0 ms/iteration (taxi 2,689 -> ~1,400, 6.8x -> 3.5x;
    istella with T1 -> ~1,450). Bits none (same lane fold, same order). Effort M. Opus: yes (the lane-order invariant must
    be stated in the brief: lane s folds j = s, s+64, s+128, ... ascending from +0, then lanes added ascending, each add ftz).
T3. FFT-interpolation repulsion (cuML's FIT-SNE), `-D MOJOLEARN_IDN_TSNE_FFT`. 3 interpolation points per box, ~100x100 box
    grid over the embedding's bounding square, 4 charge fields (1, y0, y1, |y|^2) scattered by Lagrange weights (fixed
    per-point order), our own fixed-radix-2 2-D fp32 FFT (identical_mul_add, fixed butterfly order; no cuFFT/rocFFT), the
    1/(1+r^2) kernel, gather back, Z from the fields. Host column (x_ann/host/tsne_host.mojo) twins it. O(n) per iteration,
    ~10 launches: iterations 2.3 s -> ~0.4 s. Effect: taxi -> ~0.7 s (1.8x), istella (with T1) -> ~0.75 s. Bits: fold
    change on all columns together (allowed; a version change). Quality: approximate repulsion, as cuML's arm; the board's
    trustworthiness must stay >= cuML's (ours .9923 today). Identity risk: low by construction (no atomics, no library FFT)
    but a new numerics path with many fixed-order folds. Effort L. Opus: yes with a precise spec (cuML fft_tsne.cu is the
    reference; the FFT size must be a power of two padded 2x).
T4. Not a candidate: Barnes-Hut (tree build + atomics -> nondeterministic fold order) and GEMM-form kNN distances
    (||x||^2+||y||^2-2xy: bits change for ~5 ms of gain).
Existing switches/rows (do not repeat): `TSNE_FAST_SPLIT` DROPPED-quality (EXPERIMENTS.md:341); `MOJOLEARN_TSNE_SPLIT_OFF`
(main's 3-kernel iteration); `MOJOLEARN_IDN_TSNE_LANE_FOLD_OFF`; `MOJOLEARN_ANN3_TSNE_RB32/RB64/STEP_ROWS` switches exist
(x_ann/switches.mojo:56-89; no EXPERIMENTS row found); `MOJOLEARN_KNN_WIDE_OFF`; FAST_KNN_BIGD (Apple FAST).

# DBSCAN

## Fit path
python/mojolearn/density.py:242 DBSCAN.fit -> :371 `dbscan_fit_core` (bindings/_mojolearn_estimators.mojo:178, pointers
passed, no host copy) -> dbscan/impl/dbscan.mojo:239 (budget) -> dbscan/impl/runner.mojo:391 `dbscan_fit`.

Stages:
0. Upload X (880 MB istella, ~40 ms on PCIe 4; 44 MB taxi).
1. Batch plan (dbscan.mojo:286-293, compute_batch_size:91, cuML's formula): budget = 80% of device memory - X;
   est_mem_per_row = n_rows*1 + (n_rows+2)*4 ~ 5 MB per row (cuML's dense bool adjacency + int CSR worst case, which the RBC
   arm never allocates) -> L40S batch ~7,500 rows = 134 batches; MI325X ~40,000 rows = 25 batches. Edge cap 2^31 per batch.
2. RBC index (neighbors/impl/ball_cover/ball_cover.mojo:426): 1,000 landmarks (floor sqrt n), landmark 1-NN (n x 1000 x d =
   2.2e11 MAC istella, one thread per row), counting sort, rank kernel (one block per landmark, O(group^2)), radii. ~1-2 s.
3. Loop 1, count, per batch (runner.mojo:689-800): `block_rbc_kernel_eps_csr_pass` write_pass=0 (registers.mojo:209):
   ONE WARP PER QUERY (RBC_TPB = RBC_LANES -> RBC_QPB = 1), grid = batch queries. Per query: distance to all 1,000 landmarks
   (one per lane, d terms each), then for every landmark with d(q,L) <= eps + radius(L) every point of its group, ONE
   CANDIDATE PER LANE, the full d-term distance from GLOBAL memory (eps_dist_sq, common.mojo:91: serial c-ascending ftz FMA
   chain); no reuse across lanes or queries; the 32 lanes read rows 880 B apart (uncoalesced). Standardized istella
   pairwise distances are ~sqrt(2*220) = 21, landmark radii ~15-20, eps 3: the bound passes for most landmarks, so RBC prunes
   little and the pass is ~n^2 x d = 2.2e14 MAC. Then scan + 64-bit exact total (scan.mojo) + sync per batch; a batch over
   2^31 edges splits in halves and RECOUNTS (taxi: 7,500 -> 3,750 -> 1,875 rows: 3x the count work, 533 batches).
4. Loop 2, fill, per batch (runner.mojo:949): the SAME kernel write_pass=1 recomputing every distance (another 2.2e14 MAC),
   or the one-pass max_k arm when scratch allows (:268 rbc_take_one_pass); core mask; `weak_cc_batched`
   (dbscan/impl/sparse/detail/csr.mojo:358): one thread per batch vertex, pulls the min label over its CSR row AND
   Atomic.min-pushes to EVERY neighbour (2 atomics per edge per pass), passes until a flag reads unchanged (flag download +
   sync per gate; IDN_DBSCAN_CC_GATED batches passes, _SHORTCUT adds a jump); `merge_labels_run` per batch over all n rows
   (2-5 passes, sync each). ~4-6 syncs per batch: ~700 on the L40S for istella, ~3,000 for taxi.
5. Border pass (runner.mojo:1140): `border_needs_kernel` per batch, 1 download, then per needing batch a CSR refill (another
   fill pass for that batch) + `border_pull_kernel`.
6. `_dbscan_finish`: final_relabel + relabel_for_skl, 2 syncs.

## Cost model
- Istella (sparse graph, 40k clusters): the work is the two distance passes: 2 x 2.2e14 MAC = 8.8e14 flop. The warp-per-
  query kernel streams 880 B per candidate per lane with no reuse: at ~1-2 TFLOP/s effective that is 400-800 s, which is
  the 542 s. CC and merge are small (few edges). cuML brute: raft epsilon-neighbourhood tile kernel (register-tiled fp32,
  ~20-30 TFLOP/s) computing each distance ONCE into a dense bool batch adjacency + CSR, then weak_cc: ~30-40 s of
  distances + ~25 s of adjacency/CC = 66 s. The gap is one kernel run twice.
- Taxi (~1e12 edges): the board REFUSED with "the ball-cover neighbourhood has -1799116104 edges in one batch" = the int32
  exclusive-scan tail WRAPPED; fixed by lane dbscan-int64 (2026-09-29: 64-bit exact total + split, dbscan/checks/
  dbscan_edge_split_check.mojo:3-24); the board wheel (0.8.25) predates it. On main the fit runs but walks ~1e12 edges:
  count x3 (splits) + fill in the slow kernel (~80 s), 4 TB of column ids written, weak_cc 2 x 1e12 atomics x 2-3 passes
  (~100-250 s), 533 batches x ~6 syncs, merge_labels 533 x n: the M3 A/Bs timed out on both arms. cuML is also O(n^2)
  here (int32 labels cap its batch at 2,147 rows: ~466 batches of dense bool adjacency + CSR + weak_cc) at 440 s.

## Ideas (DBSCAN)
D1. Tiled epsilon count/fill kernel, `-D MOJOLEARN_IDN_DBSCAN_EPS_TILE` (THE idea). Replace the lane-per-candidate loop of
    `block_rbc_kernel_eps_csr_pass` with the 64x64 register-tiled pattern already in the tree (knn_wide_kernel,
    sparse_mr_search_tiled_kernel): 64 queries x 64 candidates, 16 features staged per chunk, 4x4 cells per thread, the
    SAME per-pair chain `acc = ftz(identical_mul_add(diff, diff, acc))` with c ascending -> the squared-distance word equals
    today's -> the same eps predicate, the same CSR (candidates visited in x_reordered order = today's ja order). Keep the
    landmark prune per (query tile, landmark group): run the group when any query of the tile passes the bound, mask per
    pair. Count pass: per-row warp reduce; fill pass: owner emits hits in order from shared memory. Effect: istella
    2 x 2.2e14 MAC at ~20 TFLOP/s = ~45 s + CC ~10 s: 542 -> ~60 s (8.2x -> ~1x). Taxi count/fill ~80 -> ~10 s. Bits none.
    Identity risk none. Effort M-L. Opus: yes.
D2. Bit-matrix adjacency instead of the second distance pass, `-D MOJOLEARN_IDN_DBSCAN_ADJ_BITMAP`. The count pass also
    writes a batch x n bit matrix (7,500 x 1M bits = 940 MB on the L40S, inside the budget); the fill becomes a bit scan
    (set bits ascending = today's ja order). Halves the distance work: istella ~60 -> ~37 s with D1. Bits none. Effort M.
    Opus: yes (after D1; the batch planner must count the bitmap, 125 MB per 1,000 rows).
D3. Edge-free labelling for the dense regime, `-D MOJOLEARN_IDN_DBSCAN_DENSEBALL` (port of dbscan/impl/denseball.mojo,
    today Apple FAST only). Dense balls (rows of a landmark group with d1 <= eps/2) are cliques -> one union each; counts
    early-exit at min_samples; components by CAS union-find per unpruned landmark pair stopping at the first hit; labels
    canonical = smallest core id + 1 (the weak_cc fixed point) so the labels are identical whatever the union order; the
    border rule (DEVIATION 5130) is already in denseball.mojo. Gate by MEASURED density, not shape: count 1,024 sampled
    rows first (one launch) and take this route when sampled mean degree x n exceeds ~4 edge-cap batches. Effect: taxi from
    a timeout (and cuML's 440 s) to ~20-60 s; istella untouched (sparse). Bits none. Risk: the host column must emit the
    same canonical labels (it does: weak_cc's fixed point). Effort L. Opus: yes. Note `DBSCAN_FAST_DENSEBALL` lost on
    istella (rab3-denseball, B timed out) BECAUSE it ran ungated on the sparse dataset; the density gate is the new part.
D4. Batch plan from the sampled degree, `-D MOJOLEARN_IDN_DBSCAN_BATCH_SAMPLE`. Size the batch to the edge cap from D3's
    sample instead of cuML's 5 MB/row dense estimate (which the RBC arm never allocates) and the count->split->recount loop.
    Istella: 134 batches -> ~4 (cap 2^31 / ~5 edges per row), i.e. ~700 syncs -> ~25 and 130 fewer weak_cc/merge rounds;
    taxi: no 3x recount. Batching moves no label (dbscan_edge_split_check). Bits none. Effort S-M. Opus: yes.
D5. weak_cc as hook-and-jump, `-D MOJOLEARN_IDN_DBSCAN_CC_HOOK`. One edge pass hooking the larger root under the smaller
    (one Atomic.min per edge, not two per edge per pass), then ceil(log2 n) pointer jumps with no flag readback; one sync per
    batch. Fixed point = component minimum -> bits none. Atomics ~5x fewer on taxi; syncs 4-6 -> 1 per batch. Effort M.
    Flag: IDN_DBSCAN_CC_GATED / _SHORTCUT (csr.mojo:166-182) already batch passes and jump; D5 replaces the loop.
Existing switches/rows: `MOJOLEARN_DBSCAN_DIRECT_DISTANCE` (grid control dbscan_direct, bits_change, classical-kmeans.json:168),
`MOJOLEARN_IDN_DBSCAN_CC_GATED/_SHORTCUT_OFF`, `DBSCAN_RBC_KEEP_COUNTS`, `MOJOLEARN_RBC_FAST_EPS_OFF` (Apple), `C32_COUNT_FUSION`
(experiments/classical_identical_ideas/graph_controls.mojo:69), Apple FAST rows `DBSCAN_FAST_CC_BATCH`, `DBSCAN_FAST_SCAN`
(superseded), `DBSCAN_FAST_DENSEBALL` (dropped, see D3).

# HDBSCAN

## Why the board says FAILED
Every box: REFUSED(error "hdbscan.build_mr_linkage: n_rows=100000 > 46340; the dense mutual reachability graph is m * m
cells and hierarchy's PAIRWISE connectivity refuses past that bound") (nvidia-l40s.BOARD.md:2622, amd-mi325x.BOARD.md:2841).
On main that refusal is unreachable at the board shape: graph=auto takes the sparse Boruvka arm when m > IDN_HDB_SPARSE_MIN_ROWS
(hdbscan/impl/cluster/detail/single_linkage.mojo:193-205, DEVIATION 1620), confirmed by classical-fixes.json:145. The wheel
the board raced predates it. NO IDENTICAL NV/AMD time exists (classical-misc.json:781). FIRST ACTION: race hdbscan IDENTICAL
on nv and amd from main; the ideas below are ordered by what that race will most likely show.

## Fit path and stages on NV/AMD (python/mojolearn/hdbscan.py:157 -> bindings/_mojolearn_hdbscan.mojo -> hdbscan/impl/runner.mojo:212)
1. Core distances: knn_self_search_resident -> neighbors/impl/detail/knn_brute_force.mojo (materialized tile_rows x m tile +
   select, k=11): 1e5 x 1e5 x 220 = 2.2e12 MAC (~0.1-0.3 s), ~6 syncs, two device copies of X (notes/hdbscan.md section 1).
2. Sparse MR Boruvka MST (hdbscan/impl/cluster/detail/sparse_mr_mst.mojo:1193): off Apple span = m, ONE tiled search launch per
   phase (`sparse_mr_search_tiled_kernel`, 64x64 tiles, <= 1024 j-slices, fold + merge); per round: compact (incl. a ONE-BLOCK
   scan `smr_scan_blocks_kernel`:699 over m/256 counts; flagged in classical-misc.json:760), status readback + sync, phase A,
   compact, readback + sync, phase B, cmin x3, winner, hook, 17 jumps, relabel, readback + sync. ~3 syncs and ~30 launches per
   round, ~10-17 rounds. Round 1 searches every point against all m: 2.2e12 MAC per phase (~0.1 s at tile rates); later
   rounds shrink with n_todo. Output sorted by (key, lo, hi) via radix sort.
3. Orient + dendrogram_device: 17 levels x (claim, own, {memset, hook, jump, 1-word readback, sync} x 2-3, top, size,
   relabel): ~100 launches, ~40 one-word syncs.
4. Condense (tree_device.mojo:643): ~100 launches, ~10 syncs, ~35 allocations. 5. Extract: ~30 launches, ~15 syncs.
6. Outputs: 7 downloads each with its own sync and host copy loop; the binding's host loops over n for 3 outputs.
Totals ~350 launches, ~115 syncs, ~90 allocations (docs/apple-fast/notes/hdbscan.md). M3 Ultra FAST: istella 3,800 ms
(after HDB_SMR_TILED), taxi ~430 ms. On an L40S the compute is ~0.3-0.5 s (search rounds 1-3) and the 115 syncs cost
~5-10 ms; predicted ~0.6-1.5 s istella, ~0.2-0.4 s taxi, i.e. near cuML (862 / 369; cuML = cuVS brute kNN + raft MST +
device condense/extract). On the MI325X syncs are ~0.1-0.3 ms each: ~30 ms of plumbing.

## Ideas (HDBSCAN)
H1. Measure (no define): `lq add nv|amd RACE main hdbscan istella,taxi`. Effort S. Everything below waits for its split.
H2. Round-1 edges from the kNN list, `-D MOJOLEARN_IDN_HDB_MST_SEED_KNN`. In round 1 (all singletons) point i's cheapest edge
    under max(core_i, core_j, d_ij) is bounded below by core_i; if some j in kNN(i) has core_j <= core_i (then d_ij <= core_i
    too) the bound is attained and the exact minimum is the smallest such j under (key, lo, hi): no full search for i. Only
    points whose 10 neighbours all have larger core distance are listed for the m-wide search. Expected: round-1 search
    2.2e12 -> ~0.3-0.5e12 MAC (the listed share), ~100 ms on istella. Bits none if the tie order is honoured exactly.
    Effort M. Opus: yes, with the tie rule in the brief.
H3. One readback per round, `-D MOJOLEARN_IDN_HDB_ROUND_ONE_SYNC`: launch phase B with phase A's grid bound and exit blocks on
    the device count; the 3 status syncs per round become 1. Bits none. ~2 x 15 syncs: ~1-3 ms NV, ~10 ms AMD. Effort S-M.
    (Apple twins HDB_ONE_SYNC / HDB_LINKAGE_DEVICE were DROPPED-noise; expect the same on NV unless H1 shows sync-bound.)
H4. Dendrogram fixed jumps, `-D MOJOLEARN_IDN_HDB_DENDRO_FIXED_JUMPS`: replace the per-level {hook, jump, readback, sync} until
    quiet with ceil(log2 m) = 17 jumps and no readback: ~40 syncs -> 0. Same roots. Effort S. Only if H1 shows it.
H5. Output plumbing: one staged device buffer + one download for the 7 outputs, one arena for condense's 35 allocations,
    device-side label/probability copy into the caller's arrays. Effort S; ~5-10 ms on AMD.
Existing switches/rows: `HDB_SMR_TILED` (KEEP, Apple), `HDB_CORE_TILE`, `HDB_DEV_BORUVKA`, `HDB_ONE_SYNC`, `HDB_LINKAGE_DEVICE`,
`HDB_SELECT_DEVICE`, `HDBSCAN2_ALL` (EXPERIMENTS.md:487-496), `MOJOLEARN_C62_SAME_COMPONENT_SKIP` (OPEN, :1543),
`MOJOLEARN_IDN_HDB_SPARSE_MIN_ROWS` (no board effect), `MOJOLEARN_SMR_TILED_OFF`, grid controls graph_direct / knn_direct.

# Priority
1. D1 (+D4): dbscan istella 542 -> ~60 s, the largest absolute gap; one kernel, bits none.
2. T2 + T1: tsne 4,220/2,689 -> ~1,450/1,400 ms with no bit change; T3 (FFT) is the way to parity (L, bits change).
3. D3 (+D4): dbscan taxi from timeout to below cuML's 440 s.
4. H1 now; H2 after the split is known.
