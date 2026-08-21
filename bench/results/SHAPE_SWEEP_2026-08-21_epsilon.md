# Where an epsilon tree's time lives: 83% accumulate, and the density cliff

2026-08-21, M4 base, `mojo_only/shape_sweep.mojo` on prefix-slices of the
epsilon fixtures (`tools/shape_slices.py`), 128 borders, RMSE, 20 trees,
one process, configs alternated round-robin, 3 reps, quiet box. OUR ARM
ONLY: attribution, not comparison -- no absolute number here is quotable
against CatBoost; the interleaved row
(`EPSILON_2026-08-21_interleaved.md`) is the comparison.

## 1. The decomposition: t = a + b*R + c*F + d*R*F

Five (rows, feats) points, all reps within 1-2% of each other:

    (100k,  500)  17.9 ms/tree      (400k,  500)  38.4
    (100k, 2000)  47.7              (200k, 1000)  36.5
    (400k, 2000) 127.6  (depth 6 throughout)

Solved (R in 100k rows, F in 500 features), full-shape shares:

    a  (launch/drain floor)                  7.7 ms    6%
    b*R (reorder, partition, gradients)      1.0 ms    1%
    c*F (per-feature cell passes)           13.4 ms   10%
    d*R*F (histogram accumulate traffic)   105.5 ms   83%

Lack of fit at the midpoint is ~13% (the d-term is not perfectly
bilinear; see section 3 for why). The headline stands: the epsilon tree
IS the accumulate.

## 2. Sibling subtraction: alive, exactly 2x, bit-identical

Same full-shape fixture, `use_subtraction` flipped:

    ON   123.7-128.0 ms/tree
    OFF  240.0-251.5 ms/tree      -- 1.96x
    mse 0.5779691796875 BOTH ARMS, bit-identical

The reach check the digest could never do: the switch demonstrably moves
the biggest term, and on the Int32 accumulate the subtraction arithmetic
is exact, so the mse equality is the correctness proof, not a
coincidence. (CatBoost's policy, for the record:
`BuildNecessaryHistograms` computes the SMALLER sibling and subtracts,
`split_properties_helper.cpp:1283-1355`.)

## 3. The density cliff, measured by depth differencing

Full shape at depths 2 / 4 / 6 (subtraction ON): 30.7 / 67.9 / 123.4
ms/tree. Per-level increments:

    levels 1-2   ~11.5 ms/level     (root builds 400k rows, then ~200k)
    levels 3-4   18.6 ms/level      (~200k built rows -- SAME as above)
    levels 5-6   27.8 ms/level      (~200k built rows -- SAME again)

Built rows per level are constant from level 2 on (subtraction builds
only smaller halves), yet per-level cost climbs 2.4x. That is CACHE-LINE
DENSITY: `LoadByIndexBins` reads each leaf's rows through the full-width
column, and at 64 leaves a 128-byte line yields ever fewer of its 32
elements. Effective accumulate bandwidth is ~26.5 GB/s against this
box's 90 GB/s streaming ceiling.

## 4. Their design has no arm for this at 2 stats -- checked, not assumed

`ELoadFromCompressedIndexPolicy` has exactly two arms
(`split_properties_helper.cpp:1338-1341`): `LoadByIndexBins` for
statCount <= 2, `GatherBins` above. GatherBins materializes each block's
bins into leaf order and hists from the dense copy
(`:1131-1143`, `:1177-1190`) -- but its gather step performs THE SAME
amplified indexed read, plus a dense write and re-read. It pays only
when >2 stat passes amortize the materialization, which is exactly the
threshold their code sets. At 2 stats with the fused hist_2 single pass,
their own cost model says indexed loads win, and there is nothing to
port. On V100-class bandwidth the cliff is tolerable; on Apple it is the
2.4x above.

## What this prices, and what it defers

A perfect density fix would flatten levels 3+ to the level-2 cost:
depth-6 tree ~123 -> ~76 ms, roughly 1.6x, growing with depth. The
candidate (level-wise bin compaction: partition the cindex rows alongside
the docs, so deep-level reads stay dense) is a DEVIATION -- CatBoost has
no such code at 2 stats -- and its own traffic (read+write 800MB/level
dense) eats most of the win at depth 6 unless triggered only below a
density threshold; the sketch nets ~15-30%. DEFERRED at that price while
we stand 3.3-3.4x ahead at this shape; it becomes interesting again at
depth 8+ or on a bandwidth-tighter GPU. If someone picks it up: trigger
compaction when leaf density crosses ~1/8, and re-measure the depth
differencing FIRST on that machine.
