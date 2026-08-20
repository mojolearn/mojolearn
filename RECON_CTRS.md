# CTR / categorical recon (2026-08-21) — the last feature block

Read from CatBoost `54a8143a`: `catboost/cuda/ctrs/*`,
`ctrs/kernel/ctr_calcers.cu` (429 lines — the kernels are SMALL; the
system around them is the work), `gpu_data/*` builders, and the options
defaults. This file is the warm-start for the port session; nothing
below is built yet.

## What their GPU learner actually does with a cat feature

1. HASH -> PERFECT HASH -> BINS (host, `libs/data`): raw category ->
   ui32 hash -> dense bin ids per feature. We receive datasets pre-binned
   through prep, so v1 can take dense cat bins from the prep script the
   same way numeric bins arrive today.
2. ONE-HOT for cardinality <= `one_hot_max_size` (GPU default 2): the
   histogram/split kernels ALREADY carry `one_hot_feature` (predicate
   `bin == value`; threaded through layout, scan skip, split kernels,
   and the evaluator's XorMask slot). This is mostly dataset plumbing.
3. SIMPLE CTRs for the rest. GPU defaults: `Borders` (binarized-target,
   permutation-DEPENDENT) + `FeatureFreq` (permutation-INDEPENDENT).
   The CTR VALUES are then float features: binarized by the SAME
   GreedyLogSum binarizer we already ported, and fed to the numeric
   histogram pipeline unchanged.
4. TREE CTRs (feature combinations) — DEFERRED; their own machinery
   (`tree_ctr_datasets_visitor`), not v1.
5. MODEL/INFERENCE: applied models need the final CTR tables. DEFERRED
   with tree CTRs; v1 gates on train-side quality only.

## The two calcers, by difficulty

* `FeatureFreq` (`TWeightedBinFreqCalcer`,
  `WeightedBinFreqCtrsImpl`): ctr = binWeight/total from bins SORTED by
  value — a segmented count, no history. EASY once the sort exists.
* `Borders`/history CTRs (`THistoryBasedCtrCalcer`): rows ordered by
  (bin, permutation position); running sums of preceding same-bin
  binarized-target stats via `SegmentedScanAndScatterNonNegativeVector`
  (segment flag packed in the float mask bit), then
  `(sum + prior) / (count + priorDenom)` (`MakeMeansImpl`), scattered
  back. Needs the segmented scan and the sorted order.

## The two vendor-lib decision points (VENDOR RULE)

* RADIX SORT: their CTR bins are sorted with cub radix sort
  (`cuda_util/sort.h`). Metal has no cub. CHECK MAX's algorithm/sort
  surface FIRST; if absent, a portable LSD radix sort over (bin ||
  permPosition) keys is the fallback — no warp intrinsics required
  (RAFT's select_radix precedent: zero warp intrinsics). This is the
  long pole of the whole block.
* SEGMENTED SCAN: cub segmented scan on their side. We already build
  two-phase scans (stable partition chunk scans, scan_histograms); a
  flagged Blelloch over 800k elements is the same pattern. Check
  MAX first anyway.

## Gates that exist to be reused

* Oracle: `tools/catboost_oracle.py` can emit a fixture WITH cat
  features and simple CTRs pinned (CPU CatBoost). CAUTION — the
  mojotrees trap from memory: CPU and GPU CatBoost may binarize
  targets/priors differently; verify their GPU defaults against the CPU
  oracle on a TINY fixture before trusting split-level gating. If they
  diverge (like bootstrap), fall back to the stochastic-style gate set:
  exact per-step device checks against host tallies + holdout quality.
* Holdout harness (`bench/interleaved/holdout_bayesian.mojo`) extends
  directly: Amazon (all-categorical, 32k rows, their own quality
  benchmark set) is the natural dataset; Adult as the mixed one.

## Suggested port order (each step gated before the next)

1. Prep + layout: cat bins into the compressed index; one-hot for
   card <= 2 end to end; oracle fixture with one-hot only. (Their
   dispatch: one-hot features never get CTRs.)
2. FeatureFreq CTR host-precomputed (permutation-independent — it can
   be computed ONCE, even on host in prep, bit-matching their formula)
   -> binarize -> train. This lands "categoricals work" with almost no
   device work and gates the plumbing.
3. The radix sort + segmented scan (vendor-check first).
4. History/Borders CTRs on-device, learn-order permutation.
5. Tree CTRs + model export + inference CTR tables.
