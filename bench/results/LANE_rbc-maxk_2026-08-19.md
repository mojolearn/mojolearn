# LANE rbc-maxk — the two-loop max_k dispatch is WIRED

**Verdict, one line.** The DBSCAN runner's RBC arm now walks the dataset
TWICE per fit instead of three times, taking cuML's own two-loop `max_k`
dispatch (`runner.cuh:257/:289/:327/:335` through `algo.cuh:119-135`); every
existing check is green edge-for-edge, both new branches carry named checks
and sabotage evidence, and labels moved nowhere. The `K_LIB_BALL_COVER_EPS`
row stays UNWIRED, decided from upstream rather than argued (§4).

This lane closes the task `LANE_rbc-build_2026-08-19.md` §7 specified and
priced at ~20% of a 200,000-row fit. NO TIMING RUN was made here; the number
remains that lane's estimate until the orchestrator times it (§6).

---

## 1. What upstream does, file:line

All from the pinned checkouts (cuML `00094f7`, cuVS `94c2819`).

**Loop 1** (`cuml/cpp/src/dbscan/runner.cuh:249-296`, reversed `i = n-1..0`):

- `:257` `need_ja_compute = sparse_rbc_mode && ((i == 0) || sample_weight)`
  — every batch is COUNTED (`:262` passes the literal `0` as `max_k`), and
  batch 0 alone also gets its `ja` FILLED, via the two-pass arm of
  `vertexdeg/algo.cuh:137-163` (`adj_graph` resized to batch 0's own edge
  count at `algo.cuh:150`).
- `:289-293` `maxklen.at(i) = thrust::reduce(vd, vd + n_points, 0,
  maximum{})` — the batch's longest row, measured on the device while the
  degrees are resident.

**Between the loops**: `:317` `adj_graph.resize(maxadjlen)` — a GROWING
resize that preserves batch 0's columns (`rmm::device_uvector` semantics).

**Loop 2** (`runner.cuh:319-350`):

- `:327` `if (i > 0)` — batch 0 runs NO neighborhood pass; the comment is
  "i==0 -> adj and vd for batch 0 already in memory".
- `:335` passes `maxklen.at(i)`; `:337` passes `nullptr` for `vd`.
- `vertexdeg/algo.cuh:119-122` then routes:

      int64_t spare_elemets_per_row =
        data.max_k > 0 ? (batch_size * data.N - data.ja->capacity()) / n : 0;
      if (data.max_k > 0 && data.max_k < spare_elemets_per_row) { ... }

  the ONE-PASS `rbc_eps_pass` max_k overload (`cuvs registers.cuh:1427-1482`)
  when the bound fits the spare room — a MEMORY test on the `n × max_k`
  scratch (`registers.cuh:1431`), not a correctness test — and the two-pass
  count + fill (`algo.cuh:137-163`) otherwise.
- `algo.cuh:135` `ASSERT(max_k == data.max_k, "given maximum rowsize was not
  sufficient")` — an EQUALITY, because the bound was measured on the same
  rows one loop earlier.

RAFT's own copy of the ball-cover kernels
(`raft/cpp/include/raft/spatial/knn/detail/ball_cover/registers.cuh`) was
opened as instructed; it is the pre-cuVS-migration ancestor of the same file
and has no eps/max_k overloads at the pin (`661a3b8`) — the eps machinery
lives in cuVS, which is why every citation above is cuVS/cuML.

## 2. What was wired, file by file

**`dbscan/gbdt/dbscan/runner.mojo`** (the dispatch itself):

- Loop 1 now measures `maxklen[i]` per batch: `rbc_max_reduce_kernel` (the
  existing port of `thrust::reduce(..., maximum{})`, `scan.mojo`) on the
  device, one scalar read back — inside the same phase window their nvtx
  VertexDeg range covers (`:255-296`).
- After `col_ind` is sized at `maxadjlen`, batch 0's fill runs ONCE against
  the `ex_scan`/`vd` that the reversed loop 1 left resident. This placement
  (theirs fills inside loop 1 and then grows the buffer) is **DEVIATION 39
  (PORTING.md)** — `DeviceBuffer` has no growing resize; same single fill,
  byte-identical state at loop 2's entry.
- Loop 2's RBC arm: batch 0 skips the neighborhood pass entirely (`:327`);
  batches > 0 call `rbc_eps_nn_query_max_k` with `maxklen[b2]` when
  `rbc_take_one_pass` (the verbatim port of `algo.cuh:119-122`, a named host
  function so the checks can call it) admits it, and fall back to
  count + fill otherwise. The `algo.cuh:135` equality assert is a raise.
  Their loop-2 `vd` is `nullptr` (`:337`, cuVS reads degrees off `adj_ia`,
  `registers.cuh:1429`); ours passes the same `vd` buffer, and nothing
  downstream reads it — the core mask came from loop 1.
- One `tmp` scratch for all one-pass batches, sized once from `maxklen`
  (theirs allocates `n × max_k` inside each call, `registers.cuh:1431`).
- The false header sentence listing "the ball-cover arm" as NOT PORTED is
  deleted in the same change.

**Net work-shape change**: `build + count(all) + count(all) + fill(all)`
became `build + count(all) + fill(batch 0) + max_k(batches 1..)` — two walks
over the dataset per RBC fit, not three. **No labels moved anywhere** (§3).

**`neighbors/mojo_only/ball_cover_check.mojo`** —
`check_ball_cover_max_k_wiring` EXTENDED into the runner's exact buffer
shape: loop one now runs REVERSED as theirs does (`:249`), all batches share
ONE `ja` sized `maxadjlen` (`:317`), batch 0 is filled once when that buffer
is sized, and loop two reads batch 0's RESIDENT `ia`/`ja` back before any
other batch runs — its docstring's survival claim (point 3) is now actually
asserted instead of trivially true on per-batch buffers. Every batch is
byte-compared (offsets AND column order) against a fresh two-pass CSR
computed after the arm under test. The bound-cut-by-one sabotage is kept.

**`dbscan/mojo_only/dbscan_check.mojo` + `dbscan/dbscan_main.mojo`** — new
`check_dbscan_rbc_two_loop_arms`, one named fixture per side of the new
switch (PORTING_RULES 8):

- one-pass: blobs of 250/200/150 rows (hashed jitter), n = 600, batch = 200
  — batches straddle blob boundaries, longest row 250 against spare room 350;
- fallback: blobs of 500/100 — `maxadjlen` eats 100,000 of the 120,000
  budget, spare room collapses to 100, the 500-long rows cannot take the
  one-pass arm.

The guard is data-derived, not a parameter, so each fixture PROVES its
routing by feeding host-recomputed degrees (same float32 arithmetic) to the
runner's own `rbc_take_one_pass` before trusting labels; both fits must then
label every point exactly as unbatched BRUTE_FORCE does.

**Docs corrected in the same change** (rule: a falsified sentence is deleted,
not annotated): `dbscan/UNPORTED.tsv`'s two-loop-dispatch row (now ported),
`dbscan/PORTED_MAP.tsv`'s runner row, `runner.mojo`'s header, PORTING.md 39
added, and `LANE_rbc-build_2026-08-19.md` §9's `K_LIB_BALL_COVER_EPS` bullet
replaced per §4 below.

## 3. Checks, all green (pasted)

`neighbors/ball_cover_main.mojo` — first five lines unchanged edge count for
edge count, which is what proves the check restructure changed the CHECK's
shape and not the answers:

    ball_cover: exact set match at eps 0.9 / 2.5 / 8.0, edges 5132 73050 1014290
    ball_cover: dense and max_k both match brute force at eps 1.6, longest row 39
    ball_cover: sabotage reached; radii x0.3 took edges from 73050 to 62612 and every survivor is still a true neighbor
    ball_cover: the in-group order is load bearing; reversing it took edges from 73050 to 69189
    ball_cover: n=8000 d=5, 200 sampled rows match brute force exactly; 130546 edges over 89 landmarks
    ball_cover: cuML's two-loop max_k dispatch is byte-identical to the two-pass CSR over 3 batches sharing one ja; batch 0's loop-one CSR is resident; bounds 69 63 65 ; a bound one short clamps 2 rows and still reports 63

`dbscan/dbscan_main.mojo` — the required `check_dbscan_rbc_matches_brute` and
`check_dbscan_batching_agrees` are green ON the new dispatch (the batching
check's five 128-row batches all route one-pass), as are the batching lane's
budget checks, whose 19-batch RBC fit exercises the new arm 18 times:

    check_fused_eps_agrees_with_materialized OK: 31950 cells identical to the materialized path AND to a float64 host oracle (15298 of them neighbours, so the pattern is irregular), 150 degrees identical, vd[m] = 15298
    check_dbscan OK: 3/3 blobs each one whole cluster with ids exactly {0, 1, 2}, 0 merges, 12/12 isolated points labelled -1 as scikit-learn does, converged in 2 propagation passes
    check_dbscan_eps_sensitivity OK: eps=2 gives 3 clusters and eps=12 gives 1, which is impossible without the neighbourhood kernel running on real distances
    check_exclusive_scan_beyond_the_old_cap OK: 2000000 entries exact across 977 blocks, total 12000001
    check_dbscan_batching_agrees OK: 5 batches with 4 merge_labels folds give labels identical to one batch, in an adj buffer 4x smaller than the unbatched run needs
    check_dbscan_rbc_matches_brute OK: 612 points labelled identically by the ball-cover index and by brute force, 3 clusters
    check_dbscan_rbc_two_loop_arms OK: the one-pass arm (3 clusters) and the two-pass fallback (2 clusters) both label identically to unbatched brute force, and algo.cuh:119's guard provably routes each fixture to its arm
    check_dbscan_max_mbytes_moves_the_batch OK: 1 MB -> 19 batches and 1000 MB -> 1 batch at n = 1932; at n = 50000 the dbscan.cuh:71 gate keeps rbc at 50000 rows where brute clamps to 42949, and both agree at 309 rows when the clamp cannot bind
    check_dbscan_tiny_budget_agrees OK (rbc): max_mbytes_per_batch = 1 forced 19 batches (56 passes vs 2 for one batch) and all 1932 labels match one batch: 6 blobs whole with ids 0..5, 12 noise points at -1
    check_dbscan_tiny_budget_agrees OK (brute): max_mbytes_per_batch = 1 forced 19 batches (38 passes vs 2 for one batch) and all 1932 labels match one batch: 6 blobs whole with ids 0..5, 12 noise points at -1

Build: `tools/with_build_lock.sh pixi run --manifest-path
~/CascadeProjects/mojotrees/pixi.toml mojo build -I . <main> -o <bin>`, exit
0; the only warnings are pre-existing deprecation warnings in
`epsilon_neighborhood.mojo` and `adjgraph/algo.mojo` (other lanes' files).

### Reach by sabotage, one per new branch, each reverted after failing

1. **One-pass arm reached, and its equality assert is live.** The bound cut
   by one inside the runner's one-pass call:
   `Unhandled exception: dbscan rbc: batch 1 was bounded at 200 columns by
   loop 1 and came back with 200; given maximum rowsize was not sufficient`
   (fails in `check_dbscan_batching_agrees`, the first batched RBC fit).
2. **Batch 0's post-loop-1 fill is load-bearing.** Fill skipped:
   `Unhandled exception: blobs 0 and 1 were merged, which no edge in the
   graph permits` (fails in `check_dbscan` itself).
3. **The fallback branch is reached by exactly its fixture.** Fallback fill
   skipped: every check INCLUDING the one-pass fixture stays green until
   `Unhandled exception: two-loop fallback: 100 of 600 labels differ between
   batched RBC and unbatched brute force` — which simultaneously proves the
   one-pass fixture never enters the fallback branch.

## 4. `K_LIB_BALL_COVER_EPS`: DECIDED — stays unwired, from upstream

The brief asked for a re-derivation of the right block-size SOURCE with the
two-loop structure in. Re-derived by opening their file, not by rule:

- **Upstream's block-size source is a launcher-local constant.** cuVS types
  `int tpb = 64;` at `registers.cuh:1330`, inside `rbc_eps_pass`, in the SAME
  file as the kernels. There is no table, no config, no header constant. Our
  `comptime RBC_TPB` in `registers.mojo` — the file holding our
  `rbc_eps_pass_*` launchers — mirrors that file for file. Wiring the kernel
  to `mojo_only/kernel_matrix.mojo`'s row would therefore NOT mirror
  upstream; it would replace their structure with a table they do not have.
- **The two-loop change moves nothing here.** It adds no launch site and
  changes no launch shape; the same three kernels run at the same
  `((n_queries + RBC_QPB - 1) // RBC_QPB) × RBC_TPB` geometry.
- **Wiring it as-is is a priced regression.** The row resolves to the
  fall-through 128, the measured WORST of the four shapes at 200,000
  (142.10 ms against 129.08, DEVIATION 30's sweep).

So: unwired, deliberately, with the reasoning now IN `registers.mojo`'s
DEVIATION 2 banner ("WHY `RBC_TPB` IS TYPED HERE AND NOT READ FROM THE
KERNEL MATRIX") and the stale "live violation of the do-not-type-a-constant
rule" framing in `LANE_rbc-build_2026-08-19.md` §9 replaced with this
derivation. A vendor measurement that wants a different value lands in the
matrix row first, with DEVIATION 30's sweep as the bar to clear.

## 5. What did NOT change

- No API signature moved: `dbscan_fit` / `dbscan_fit_impl` /
  `rbc_eps_nn_query_*` are call-compatible; `rbc_take_one_pass` is additive.
- No kernel changed: the three eps kernels, the scan, the max-reduce and the
  clamp are byte-for-byte the files the previous lane left.
- `dbscan/gbdt/dbscan/dbscan.mojo`, `compute_batch_size`, the batching
  policy, `fused_l2_knn.mojo` and `faiss_select/` — untouched (other lanes').
- No timing was run. Every run above is a correctness check.

## 6. What the orchestrator should time

The `LANE_rbc-build` §7 price to confirm: ~130 ms off a 632.7 ms fit at
n = 200,000. Suggested arms, one process, interleaved:

- `dbscan_fit_impl`, RBC default, the DBSCAN scaling fixture
  (`bench/scaling_main.mojo`: d = 8, eps = 0.30, min_pts as there) at
  n = 16,000 / 50,000 / 100,000 / 200,000 — HEAD against `e4eb7cc` (the
  commit before this lane). Every batch at these sizes routes one-pass
  (degrees ≈ 1 against spare ≈ n − nnz/np), so the win is
  `count(all) + fill(all)` becoming `fill(batch 0) + max_k(rest)`.
- Worth printing beside the number: `PHASE label.vertexdeg` lines
  (`phase_timing=True`) name the loop-2 cost directly; the plan line names
  the batch count.
- Note n = 16,000 is ONE batch at default budget: there loop 2 does no
  neighborhood work at all now (batch 0 resident), so the win at one-batch
  sizes is one full count+fill of the dataset, the larger relative saving.

## 7. Shared-checkout note for the orchestrator

The CODE of this lane -- the `runner.mojo` dispatch, the extended
`check_ball_cover_max_k_wiring`, `check_dbscan_rbc_two_loop_arms` and its
`dbscan_main.mojo` wiring -- was swept into `83d4bd9` from the shared working
tree by a concurrent session's commit (the score-kernel commit, which also
carries the batching lane's dev 37/38 work; its `LANE_dbscan-batching`
report §4 row `:327-349` records the entanglement from their side). Every
check output in §3 was produced on exactly that content; the only difference
between what was run and `83d4bd9` is an 8-line docstring reconciliation in
`runner.mojo`'s header, which the batching lane's own commit (`f60f575`)
then took. THIS commit therefore carries the lane's remaining truth:
PORTING.md 39, the `registers.mojo` K_LIB banner, the `LANE_rbc-build` §9
correction, the PORTED_MAP/UNPORTED rows, and this report. `neighbors/mutex_probe_main.mojo` and
`neighbors/gbdt/distance/` were left unstaged -- another lane's
working-tree files.

One caution for the §6 timing: `e4eb7cc` is the last commit before this
dispatch, but `83d4bd9` mixes it with the batching lane's
`compute_batch_size` gate fix (dev 37, which CHANGES the default batch count
above n = 46,341 on the RBC arm) and the CatBoost score-kernel work. A
HEAD-vs-`e4eb7cc` DBSCAN sweep prices the two lanes together; the PHASE
lines (`dbscan/phase_main.mojo`, the batching lane's harness) are what
attribute the difference between the dispatch and the batch-count change.
