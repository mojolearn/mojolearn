# LANE: fused L2 k-NN `gridDim.x > 1` — mutex merge ported, soundness probed first

2026-08-19. NO TIMINGS WERE RUN in this lane; correctness runs only. The
orchestrator times afterward (what to time is at the bottom).

## HEADLINE: the mutex protocol IS expressible and sound on Metal — but not in their spelling

The repo header said the cross-block merge was blocked on "`atomicCAS` plus
`__threadfence` ever established as sound on Metal." That question is now
CLOSED, in three compiler-verified parts and one probe. All three error
messages below are the Mojo 1.0 Metal backend's own, reproduced from real
compiles in this lane:

1. **No standalone device fence exists.** `std.gpu.intrinsics.threadfence`
   is comptime-asserted `"threadfence is only implemented on NVIDIA GPUs"`
   (`stdlib/std/gpu/intrinsics.mojo:790-792`).
2. **No strong CAS, no acquiring RMW.** `pop.atomic.cmpxchg` legalizes for
   Apple only as `weak`, and only RELAXED:
   - `"Apple GPU only supports 'weak' compare-exchange; AIR exposes no
     strong compare-exchange primitive"`
   - `"Apple GPU does not support 'acquire' atomic ordering"` (acq_rel
     rejected identically)
3. **Acquire loads and release stores DO legalize and run.**
   `Atomic.load[Ordering.ACQUIRE]` and `Atomic.store[Ordering.RELEASE]`
   (SEQUENTIAL too) compile and execute on the M4, verified by enqueue.

So their protocol is spelled as a **test-and-test-and-set**: spin on an
ACQUIRE load until the mutex shows the awaited state, claim it with a weak
RELAXED compare-exchange (failure re-enters the spin), hand off with a
RELEASE store. Identical protocol in the C++11 model — CUDA defines
`__threadfence()` as `atomic_thread_fence(seq_cst, thread_scope_device)` —
and no ABA hides in the relaxed claim because every mutex state value has
exactly one writer role.

### The probe (`neighbors/mutex_probe_main.mojo`), run BEFORE porting

A minimal dedicated kernel with the EXACT state machine of their
`rowEpilog_lambda` (`fused_l2_knn.cuh:241-338`): one consumer block and
`grid.x - 1` producer blocks per row contending on a per-row mutex
(states 0 / 1 / -2 / -1), producers writing HASHED payloads (splitmix32 per
(row, block, word) — never uniform) through a single exchange buffer that is
POISONED before every launch, host-verified for exact equality. Result:

    config grid=( 2 , 2 )  W=64 iters= 50  bad_iters=0
    config grid=( 8 , 8 )  W=64 iters=200  bad_iters=0
    config grid=(16 , 4 )  W=37 iters=200  bad_iters=0
    config grid=(12 ,10 )  W=64 iters=200  bad_iters=0     <- 120 blocks = capacity
    sabotage SKIP:          bad_iters= 25 of 25  (want 25)
    sabotage EARLY_RELEASE: bad_iters=200 of 200
    oversubscribed grid=(16,16), 256 blocks, 100 iters: bad_iters=0
    MUTEX PROBE PASS

650 in-envelope contended launches, zero mismatches. The two sabotage arms
are what make the passes meaningful: a producer that skips half its words is
caught 25/25, and a producer that RELEASES BEFORE WRITING (then dawdles) is
caught **200/200** — the probe demonstrably observes handoff violations every
single time one exists, and never when the protocol is intact. A 2.1x
oversubscribed grid also completed exactly (informative only; the launcher
never produces one).

**The protocol's real precondition is CO-RESIDENCY**, not the fence: a
spinning producer terminates only if its consumer runs. That is exactly what
their `launchConfigGenerator` cap (`numSMs * blocksPerSM`) buys, which is why
the launch computation is part of the merge's correctness, not a tuning knob.
(The same distinction fixes a half-true sentence in
`dbscan/ported/dbscan/adjgraph/algo.mojo` DEVIATION 32: device-scope
acquire/release EXISTS; what decoupled lookback lacks is the capacity-capped
grid, because its grid scales with data. Corrected in the same commit.)

## What was ported vs what upstream does

`neighbors/ported/neighbors/detail/fused_l2_knn.mojo`, mirroring
`cuvs/cpp/src/neighbors/detail/fused_l2_knn.cuh` at `94c2819`:

| upstream | ours |
|---|---|
| outer grid-stride over row tiles, `pairwise_distance_base.cuh:129` (`grid_offset_m = Mblk*blockIdx.y`, stride `Mblk*gridDim.y`) | ported; the kernel previously had NO y-stride because it always launched `grid_y = yChunks` |
| inner grid-stride over column tiles, `:131` (`grid_offset_n = Nblk*blockIdx.x`, stride `Nblk*gridDim.x`) | ported; previously started at 0 with stride `Nblk` (the `gridDim.x==1` special case) |
| final store guard `((gridStrideX + Nblk*gridDim.x) >= n) && gridDim.x == 1` (`fused_l2_knn.cuh:479`) | ported as the after-x-loop `if gdx == 1` store (equivalent: the guard is true only on the last x-iteration) |
| `rowEpilog_lambda` `:224-338`: consumer loop (`atomicCAS(-2,-1)` spin, copy `numOfNN` pairs, `atomicExch(0)`, merge via `heapArr[i]->add`, needSort vote + `reduce`, `storeWarpQGmem`); producer (`atomicCAS(0,1)` spin, write pairs, `atomicExch(-2)`) | ported step for step; fences spelled as ACQUIRE load / weak RELAXED CAS / RELEASE store (see headline) |
| consumer stages pairs `out -> regs -> shDumpKV -> regs` (`:258-276`, `:283-299`); producer reads its already-reduced queue from `shDumpKV` (`:325-330`) | DEVIATION BLOCK 1 continuation: our queues never spilled to shmem, so the consumer stops at the first registers and the producer `reduce()`s its registers before `write_out` — same pairs, same protocol steps, same order |
| mutex workspace: `ceildiv(m, Mblk)` int32, memset 0 only when `grid.x > 1` (`fusedL2ExpKnnImpl:777-790`) | ported in `fused_l2_knn_launch`; buffer always allocated (their kernel signature also takes the pointer unconditionally), zeroed only when `grid_x > 1` |
| `launchConfigGenerator` (`pairwise_distance_base.cuh:295-322`) | ported branch-for-branch in NEW `neighbors/ported/distance/detail/pairwise_distance_base.mojo`, M4 inputs (below) |

`gridDim.x == 1` behavior is unchanged: their `rowEpilog` returns immediately
there (`:226`), the mutex array is never touched, and the column sweep is the
byte-for-byte prior kernel. A `sabotage` parameter (check infrastructure
only, hard-coded 0 in production) lets `knn_check` poison the LAST producer's
handoff.

## The gridDim.x computation, M4-fed, in one place

`pairwise_distance_base.mojo`. Their two hardware queries have no Metal
counterpart through Mojo, so the M4 inputs are pinned there and ONLY there:

- `numSMs` -> `APPLE_M4_GPU_CORES = 10`
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` ->
  `max_active_blocks_per_core`: thread-slot term
  `APPLE_M4_MAX_THREADS_PER_CORE (3072) // Nthreads (256) = 12`. The static
  shared-memory partition term is deliberately NOT a divisor: family-9
  (M3/M4) threadgroup memory is dynamically cached, and the measured query
  sweep (SCALING_2026-08-19.md: deficit 0.66x at ~32 blocks rising
  monotonically to 0.93x at ~2,000 blocks, 18.5 KB threadgroup memory per
  block throughout) is only possible with many such blocks resident per
  core — a static 32 KB/core partition would cap them at one. `smem_bytes`
  is still checked against Metal's 32 KB per-threadgroup wall (raises).
- so `minGridSize = 10 * 12 = 120`, then their `:309-319` verbatim.

**Values at the bench shapes (2,000 queries, d=32, k=10):**

| shape | yChunks | grid |
|---|---|---|
| 2,000 q x 20k..400k index | 125 | **(1, 120)** at EVERY index size |
| 1,920 q | 120 | (1, 120) |
| 1,904 q | 119 | (2, 119) — the split boundary |
| 640 q x 4,093 | 40 | (3, 40) |
| 500 q x 200k | 32 | (4, 32) |
| 53 q x 4,093 | 4 | (16, 4) |

Read that first row carefully: **their own computation keeps `gridDim.x == 1`
at the 2,000-query bench shape**, because 125 row tiles already exceed the
M4's 120-block capacity. What DID change there is the geometry: 120 blocks
with a row grid-stride instead of 125 one-shot blocks. The x-split is the
small-query fix (their design's intent: `grid.x > 1` when `m` is small), and
it engages below ~1,905 queries at 256-thread occupancy 12.

## Checks (all green; output pasted)

`pixi run mojo run -I . neighbors/knn_main.mojo`:

    check_knn OK: 64 queries x k=8 over 4096 index points, every returned neighbor is in the exact true set
    check_knn_reach_by_sabotage OK: index_norm moved 512/512 neighbors; query_norm offset moved 0 sets, which is the predicted shape
    check_vendor_topk_matches_ported OK: nn.topk.top_k and the ported RAFT radix select agree on all 512 neighbours
    check_fused_l2_knn OK: k = 1, 8, 32, 64 over 53 queries x 4093 index x 20 features, every slot matches the host Float64 oracle in order, and is_sqrt is exactly one square root away
    check_fused_edge_shapes OK: k = 1, 32, 33, 64 over 17 queries x 1013 index x 13 features -- a partial tile on Mblk, Nblk and Kblk at once, and both WarpSelect instantiations
    check_fused_reach_by_sabotage OK: index_norm ramp moved 424/424 neighbours and 424 distances; query_norm offset moved 0, which is the predicted shape
    check_fused_queue_reach_by_sabotage OK: a co-located point planted at index column 3971 for query 0 came back at rank 0 at exactly 0.0, shifted that row's tail by exactly one, and moved no other row
    check_fused_k_ceiling OK: k = 64 accepted, k = 65 refused with their ASSERT message
    check_dispatch_takes_fused OK: with KNN_METHOD_FUSED, k=8 wrote out_idx and left out_idx32 at its sentinel and k=65 did the opposite, so the `k <= 64` branch at knn_brute_force.cuh:443 is wired both ways; with the argument omitted, k=8 took the TILED arm, which is DEVIATION 36's default
    check_launch_config_values OK: bench shape (2,000 q) computes grid (1, 120); 1,904 q is the smallest-split boundary (2, 119); 53 q gives (16, 4) with the x-chunk cap binding; 640 q gives (3, 40) with the occupancy term binding; 32 KB wall raises
    check_fused_griddimx_merge OK: computed grid (16, 4) matches the oracle per slot at k=10; poisoning the last producer moved 150 slots; forced grid_x=5 with a partial last x-tile is exact; grid_x=1 ignores the sabotage entirely
    check_fused_griddimx_one_capped_y OK: 2,000 queries x 1,013 index ran at computed grid (1, 120) -- 125 row tiles over 120 blocks, so the new outer grid-stride loop carried five blocks to a second tile -- and every slot matches the host oracle in order

Note the pre-existing fused checks now traverse the merge on their own:
`check_fused_l2_knn` (53 q) computes grid (16, 4) and
`check_fused_edge_shapes` (17 q) computes (4, 2), both WarpSelect
instantiations, with 4,093 and 1,013 giving partial last x-tiles. Reach is
still not inferred from green digests: `check_fused_griddimx_merge` poisons
the last producer's handoff and 150 slots MOVE, and the same poison at a
forced `grid_x = 1` moves NOTHING (that side never reads the mutex), so both
sides of the switch are exercised by name (PORTING_RULES 8).

`pixi run mojo run -I . neighbors/ball_cover_main.mojo` (no collateral
damage in the neighbors section):

    ball_cover: exact set match at eps 0.9 / 2.5 / 8.0, edges 5132 73050 1014290
    ball_cover: dense and max_k both match brute force at eps 1.6, longest row 39
    ball_cover: sabotage reached; radii x0.3 took edges from 73050 to 62612 and every survivor is still a true neighbor
    ball_cover: the in-group order is load bearing; reversing it took edges from 73050 to 69189
    ball_cover: n=8000 d=5, 200 sampled rows match brute force exactly; 130546 edges over 89 landmarks
    ball_cover: cuML's two-loop max_k dispatch is byte-identical to the two-pass CSR over 3 batches sharing one ja; batch 0's loop-one CSR is resident; bounds 69 63 65 ; a bound one short clamps 2 rows and still reports 63

And the probe itself: `pixi run mojo run -I . neighbors/mutex_probe_main.mojo`
(output quoted in the headline section above).

## DEVIATION 36 (default = tiled) NOT TOUCHED

`knn_method` still defaults to `KNN_METHOD_TILED`. The measured table that
decided it was taken on the old fixed `1 x ceil(m/16)` geometry and is now
stale in that respect; its note in `knn_brute_force.mojo` and PORTING.md
entry 36 say so explicitly. The orchestrator decides after timing.

## What the orchestrator should time

All fused-arm runs via `knn_method = KNN_METHOD_FUSED`, interleaved arms as
always:

1. **The standard sweep** (2,000 q, d=32, k=10, 20k-400k index), fused vs
   tiled. Expect little movement: same `grid_x == 1`, geometry 125 -> 120
   blocks + y-stride. This re-anchors the stale DEVIATION 36 table.
2. **The small-query regime, where the split engages** — the actual payoff
   surface: 500 q x 200k index now fields (4, 32) = 128 blocks where it
   fielded 32. The old point there was 0.66x vs tiled. Also 250 q and
   1,000 q x 200k-400k.
3. **The query sweep re-run** (500 / 2,000 / 8,000 / 32,000 q at 200k) to
   replace the one in DEVIATION 36 taken on the old geometry.
4. If the small-m fused arm now beats tiled somewhere, the DEVIATION 36
   decision becomes shape-dependent and is the orchestrator's call; nothing
   here pre-empts it.

Merge overhead to keep in mind when reading small-m numbers: per row tile,
`grid_x - 1` serialized handoffs of `numOfNN` pairs each. At (4, 32) and
k=10 that is 3 handoffs x 32 tiles — small, but it is a serial chain per
row tile.

## Files touched (this lane only)

- NEW `neighbors/mutex_probe_main.mojo` — the soundness probe (kept: it is
  the evidence and the regression test for the spelling)
- NEW `neighbors/ported/distance/__init__.mojo`, `.../detail/__init__.mojo`,
  `.../detail/pairwise_distance_base.mojo` — `launchConfigGenerator`, M4
  inputs in one place
- `neighbors/ported/neighbors/detail/fused_l2_knn.mojo` — both grid-stride
  axes, the mutex merge, `fused_l2_knn_launch` (grid + sabotage pinnable by
  checks), grid computed by `launch_config_generator`; DEVIATION BLOCK 2
  rewritten (the "not ported / OPEN item" text it falsified is deleted)
- `neighbors/ported/neighbors/detail/knn_brute_force.mojo` — DEVIATION 36
  note updated (merge landed, table stale, default unchanged)
- `neighbors/mojo_only/knn_check.mojo`, `neighbors/knn_main.mojo` — three
  new checks
- `neighbors/PORTED_MAP.tsv`, `PORTING.md` (entry 36 corrected, entry 40
  added), `dbscan/ported/dbscan/adjgraph/algo.mojo` (DEVIATION 32 sentence
  corrected)

## Commit provenance (shared-checkout incident, recorded per the standing rule)

This lane's entire content was committed as `94fa7c7` — but under ANOTHER
session's message ("README: the float-atomic flush IS reached on Apple"),
because that session committed in the shared checkout during the window
between this lane's `git add` (explicit paths only) and its `git commit`,
sweeping the staged index. Verified afterward: the working tree is
byte-identical to `94fa7c7` for every file of this lane, including the
final acquire-load protocol in both the probe and the kernel, so nothing
was lost or half-landed. The full intended commit message for the lane's
content is reproduced in this file's sections above; this addendum commit
exists so the lane has its own hash. The staged-index race is a THIRD
shared-checkout hazard beside the two already on record (silent merge
parents; anchored-edit clobbers): an explicit-path `git add` is not safe
until the `git commit` that consumes it, so the two must be one command.
