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
the biggest term, and the mse equality held at this shape. (It is a
MEASURED equality, not implied: the "Int32 makes the subtraction exact"
argument recorded here originally is false -- archive/reference/PORTING.md 136a, cells
round through float32 before the subtraction.) (CatBoost's policy, for
the record:
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
candidate (level-wise bin compaction) was priced here at 15-30% when
this file was first written; THE ADDENDUM BELOW REFUTES IT BY
MEASUREMENT the same day -- the dense copy costs more than the
amplification it recovers at depth 6 -- and the estimate is deleted
rather than kept beside its refutation.

## Addendum, same day: the cliff has no profitable on-box fix -- probed

`mojo_only/density_probe.mojo` measured the mechanism directly: indexed
sums over hashed ascending subsets of a 400k x 500-uint32 buffer, against
dense reads and against the gather arm's true cost (the same amplified
read + dense write + dense re-read), alternated in one process.

    density   indexed useful-BW    indexed / gather-arm cost
      1/1        104.6 GB/s              0.29x
      1/2         51.7                   0.45x
      1/8         14.8                   0.68x
      1/32         7.3                   0.75x
      1/64         5.6                   0.75x

Two conclusions, both negative and both final for this box:

1. **Their 2-stat threshold holds on Apple.** Even at 1/64 density,
   where the indexed read is ~18x amplified, GatherBins would still cost
   ~1.33x more, because its gather step pays the identical amplification
   before adding a write and a re-read. No kernel-matrix flip exists.
2. **Level compaction is dead by the probe's own numbers**: moving the
   800MB bin matrix densely costs ~16-18 ms/level, more than the ~5-9
   ms/level that depth-6 amplification actually loses (27.8 measured vs
   ~19-23 dense-ideal). It could only pay at depths where amplification
   losses exceed a full dense copy -- far beyond depth 6.

So the density cliff is STRUCTURAL on this device at CatBoost's GPU
depth, the accumulate's indexed load is already the best arm at every
density, and the on-box speed hunt at the epsilon shape closes with a
measured answer at every door: floor priced, dispatch priced, cell
passes 10%, subtraction proven, gather refuted, compaction refuted. The
cliff re-opens only on other memory systems (measure the depth
differencing there first) or at much greater depths.
