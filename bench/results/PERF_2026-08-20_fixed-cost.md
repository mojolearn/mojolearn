# Where our per-tree cost goes, and 25% of it removed

2026-08-20 evening, M4 base, 10 GPU cores, AC power. Every run held
`tools/with_build_lock.sh`. Harness unchanged.

## The diagnosis

Interleaved against CatBoost CPU at five row counts in one window, depth 6,
100 features, `ms/tree = a + b * rows` fitted across 50k..800k:

    arm            fixed a       slope b            max fit error
    ---------      ----------    ---------------    -------------
    ours  @254     12.56 ms      18.5 us / 1k rows  0.44 ms
    theirs @254     2.20 ms      35.6 us / 1k rows  0.20 ms
    ours  @128     10.11 ms      18.1 us / 1k rows  0.16 ms
    theirs @128     1.21 ms      37.3 us / 1k rows  0.25 ms

**Our per-row work is about twice as fast as their CPU. Our fixed per-tree
cost is 5.7x theirs.** That single fact explains every row of the
scoreboard: we lose at 50k, draw near 500k, win at 800k, and the crossover
is arithmetic, not mystery.

## Where the fixed cost was

`run_tree_layout`, 50k x 100 x 254, depth 6, host timestamps with no added
syncs (so no instrumentation distortion):

    setup before the first level   4.69 ms   38%
    the six levels                 6.08 ms   49%
      of which host enqueue          1.54 ms
      of which waiting on the GPU    4.47 ms
      of which host gate logic       0.08 ms
    after the last level           1.07 ms    9%

A tree spent nearly as long getting ready as growing. Isolated, allocating
the three large planes -- `hist` and `acc_i32` at 13 MB each plus the block
scratch -- is 1.9 ms of that setup; the memsets that follow are 0.5 ms and
are real work.

## What was wrong, and what CatBoost does

Their `TCudaManager` hands every device buffer out of a per-device memory
pool, so `CreateInitialSubsets` gets tree 2's histograms from the memory
tree 1 released. This port dropped the pool layer and called
`enqueue_create_buffer` directly, so every tree allocated its own planes.
`TTreeWorkspace` is a pool of one, held by `fit` across the boosting loop.
archive/reference/PORTING.md 59.

Second, smaller: their histogram copy and subtract move one float per
thread, which is free on NVIDIA (`__ldg` + `WriteThrough` fill a sector per
warp) and is not free here. Isolated at a depth-6 level's shape:
**11.0 GB/s at 4 bytes per thread, 65.2 GB/s at 16.** archive/reference/PORTING.md 60.

## Result

    border  rows    theirs   ours before   ours after   before -> after
    ------  ------  -------  -----------   ----------   ----------------
    254      50k     3.94      13.71         10.21      0.31x -> 0.39x
    254     100k     5.71      14.51         11.47      0.39x -> 0.50x
    254     200k     9.09      15.82         13.24      0.58x -> 0.69x
    254     400k    16.31      20.02         16.78      0.82x -> 0.97x
    254     800k    30.50      27.42         24.40      1.12x -> 1.25x
    128      50k     3.12      11.14          9.66      0.29x -> 0.32x
    128     200k     8.32      13.59         12.03      0.62x -> 0.69x
    128     800k    30.88      24.65         23.35      1.26x -> 1.32x

    fixed per-tree  12.56 -> 9.43 ms at 254 (25% off), 10.11 -> 8.59 at 128
    row slope       unchanged at ~18.5 us/1k rows, as a fixed-cost fix should
    we overtake at  616k -> 436k rows (254), 474k -> 396k (128)

Train mse is BIT-IDENTICAL to the previous build at every row count and both
border counts. That is the reach proof for the pool: a stale plane would
change tree 2 onward. Sabotage puts 12.06 ms back when the reuse branch is
forced off. All twenty checks pass.

## Two negative results, recorded so nobody re-runs them

**CatBoost's warp-scan shape is not a win here.** `archive/reference/PORTING.md 8` replaced
their `cub::WarpScan<double>` with one thread per feature because the port
believed Mojo had no warp primitives -- a belief that is stale. Measured
against their shape (32 lanes per feature, shared memory in place of the
shuffle): theirs wins at shallow levels, loses at deep ones, and over a
whole depth-6 tree it is 0.72 ms against our 0.68 ms. The serial scan stays.
Their accumulator is `double` and ours is float32; that gap is real and is
not closable on a target with no float64.

**This box drifts enough to invent findings.** The same binary on the same
fixture measured 21.1 ms/tree at 20:36 and 12.1 ms/tree at 20:55. An A/B
taken across that gap reported the pool as a 1.79x win; alternated inside
one window it is 1.31x, and the vectorisation alone is 2.7%, not the 43%
the drifted pair implied. Nothing here is quoted from a non-alternating
comparison.

## What is still open

The remaining 9.43 ms of fixed cost is spread thin: no single phase is more
than ~1.3 ms. The largest identified are the per-level GPU wait (4.47 ms
across six levels, which includes real work), host enqueue at 1.54 ms for
~84 launches, and the 1.07 ms tail. Beating their CPU below 400k rows needs
most of that, not one more fix.
