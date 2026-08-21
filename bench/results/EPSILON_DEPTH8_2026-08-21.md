# EPSILON at depth 8: the margin WIDENS to 3.2-3.7x -- the cliff loses to their depth bill

2026-08-21, M4 base, `bench/interleaved/rmse_depth_interleaved.mojo`
(the committed pinned-RMSE harness with DEPTH moved from comptime to
argv; a separate file only because the standard harness is another
lane's working file -- delete it into that harness when the lanes
rejoin). 400k x 2000, 20 trees DEPTH 8 -- gbm-bench's pinned depth --
3 reps per border, arms interleaved, pinned settings unchanged.

WHY THIS ROW EXISTED TO BE RUN: the density-cliff finding
(`SHAPE_SWEEP_2026-08-21_epsilon.md`) measured our per-level accumulate
cost RISING with depth at constant built rows (11.5 -> 27.8 ms across
levels 1-6), trend up, so depth 8 -- two more levels at 128-256 leaves,
the worst indexed-read density yet -- was the shape most likely to hurt
us. Their CPU also pays more per level with depth. WHO PAYS MORE was
the open question.

## The table (speedup > 1 means ours is faster)

    254 borders:
      theirs 1083.1 / 936.9 / 1736.7 ms/tree
      ours    339.5 /  337.4 /  472.0     -> 3.19 / 2.78 / 3.68x
      mse ours 0.5435336328125 (bit-identical), theirs 0.5435336158
      -- 8 significant figures

    128 borders (their GPU default):
      theirs 1007.9 / 1123.0 / 1115.0
      ours    314.5 /  308.6 /  312.6     -> 3.20 / 3.64 / 3.57x
      mse ours 0.5428834375 (bit-identical), theirs 0.5428834048
      -- 8 significant figures

Ranges fully disjoint at both borders. Both arms swing thermally at
trees this long (theirs 937-1737 at 254); the interleaving carries it.

## What it means

* **Their CPU pays more for depth than our cliff costs us.** Depth 6 -> 8
  multiplies ours by ~2.2x (140 -> 310 ms/tree at 128) and theirs by
  ~3x (370 -> 1080): two more levels double their leaf count twice and
  their per-level CPU histogram work with it, while our extra levels pay
  the (already-measured, already-bounded) density amplification. The
  quiet-window depth-6 medians were 2.40x / 2.77x; depth 8 reads
  2.78-3.68x. THE MARGIN WIDENS WITH DEPTH.
* **The compaction deferral gets stronger, not weaker**: the reopen
  condition written into the shape-sweep record was "depth 8+" -- but
  comparatively, depth 8 is a BETTER shape for us than depth 6, so the
  on-box priority of a density fix goes down again.
* Third depth point for the scale story: parity-to-win at depth 6 800k,
  2.4-2.8x at depth 6 epsilon, 3.2-3.7x at depth 8 epsilon (128-border
  medians). Depth, like rows and features, amortizes our floor and
  multiplies their CPU's bill.
