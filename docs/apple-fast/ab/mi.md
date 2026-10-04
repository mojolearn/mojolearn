# af-mi: mutual information candidates (FAST + Apple only)

Lane af-mi, branch lane/apple-fast-mi. Profile and the reason the regression variant is 60x the classif time:
docs/apple-fast/notes/mi.md (the `mi_cc` stage of mutual_info_regression is a brute-force O(n^2 d) unit; the
classif stage already runs a sorted search). Request lines: docs/apple-fast/ab/mi.txt (regression istella first,
then taxi, then classif istella and taxi). Every define is read with `is_defined` in x_prep/device.mojo under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`; default off; IDENTICAL and the other
vendors compile main's code. Kernels: x_prep/dmi_fast.mojo. Same k, same noise words (the units' own `_sec`
/ `_dsec` / `_within` functions): the counts are order statistics of the unit's distance pairs, so the sorted
arms write the same per-point terms as the unit. Expect the FAST digest unchanged for every arm except FAST_FOLDS.

**MOJOLEARN_MI_REG_SORTCOUNT** (the deciding arm; A/B "" vs it). `mi_cc` (op 68) as the sorted Kraskov search the
host binding already uses (x_prep/host/mutual_info.mojo `_cc_column`): every column bitonic-sorted by (key(x), j)
on the device, y sorted once; one thread per (point, column) walks outward in x and stops a side once |dx| exceeds
the current k-th primary distance, then nx and ny by binary search from the point's own sorted position. Columns
with a non-finite value keep the brute-force unit. Expected: istella regression from O(n^2 d) = 4.4e12 pair
evaluations to O(n log n d); the 46.5 s should fall toward the classif time (~0.8 s). Risk: a column with long
runs of equal x (taxi codes, istella zeros) makes the plain walk O(run) per point; TIES addresses that. Adds one
synchronize and ~1 GB of transient scratch on istella.

**MOJOLEARN_MI_REG_TIES** (A/B SORTCOUNT vs TIES; implies SORTCOUNT). Sorts each column by (key(x), key(y),
key(sx)) with a 128-bit compare, so inside an x-run the members are in y order and inside an (x, y) stretch in
noise-word order: the k nearest are outward walks with exact stop rules, and the counts use dmi.mojo's run counts
(a second column copy in (key(x), key(sx)) order, y in (key(y), key(sy)) order). Expected: large win on tied
columns (taxi low-cardinality features, istella zeros, the 5-valued istella target); small extra sort cost on
untied ones. Risk: twice the sort work and scratch.

**MOJOLEARN_MI_REG_RANKMAJOR** (A/B SORTCOUNT vs RANKMAJOR; implies SORTCOUNT). The point kernel maps thread
t = c * n + r (sorted rank r of column c) instead of t = i * d + c, so adjacent threads of a simdgroup walk
adjacent positions of one column's sorted arrays (shared cache lines) and have similar walk lengths (less
divergence). Expected: memory-bound point kernel faster; no change to the search. Risk: none on bits; can lose if
the per-column walk lengths vary sharply by rank.

**MOJOLEARN_MI_FAST_FOLDS** (both lanes; A/B "" vs it). `mi_colscale` (op 66) and `mi_reduce` (op 70) are one
thread per column over n rows (d threads busy); here one threadgroup of 256 per column with a tree fold. Changes
the float sum order of the column scale and the MI sum (FAST, pairwise is never less accurate): the digest may
move by ulps; selected features should not. Expected: small (two launches of n dependent loads per column).

**MOJOLEARN_MI_CLF_RANKMAJOR** (classif; A/B "" vs it). dmi.mojo's `mi_cd` point kernel (ties form) in the
rank-major mapping above, same sort, same split, same words. Expected: modest gain on the 783 ms istella classif.

**MOJOLEARN_MI_ALL**: every define above (TIES + RANKMAJOR for the regression search, FOLDS, CLF_RANKMAJOR).

**MOJOLEARN_MI_WORK=1** (env, read once at import, FAST mode only; afc_ab.sh with the default FAST build). The
noised columns, their noise words and the per-point terms (3 n d words, 265 MB on istella) become device-only
scratch (`_Prog.work`) instead of arena words the host zeroes and the device copies back (~20 ms per 64 MB on
Apple). Where a word lives moves no bit. Python: python/mojolearn/_expansion_prep.py `_mutual_info`.

Brief candidates not built, and why (notes/mi.md): REG_FEATBATCH already holds on main (one launch covers n*d
units, one readback); REG_KNN_TILE keeps the O(n^2 d) brute force (seconds at best) where the sorted search is
O(n log n d); CLF_CLASSBATCH is what dmi.mojo's class split already does; DIGAMMA_FUSED is covered by FAST_FOLDS
(the reduce becomes one threadgroup per column) and by MI_WORK (only d values come back).

## Builds (lane af-mi, 2026-10-03, compile only, head de9b194a0 code)
All passed (rc=0), x_prep binding, MOJOLEARN_SKIP_BUILD_GATE=1: FAST -D MOJOLEARN_MI_ALL; FAST -D MOJOLEARN_MI_REG_SORTCOUNT;
FAST -D MOJOLEARN_MI_REG_TIES; FAST -D MOJOLEARN_MI_REG_RANKMAJOR; FAST -D MOJOLEARN_MI_FAST_FOLDS;
FAST -D MOJOLEARN_MI_CLF_RANKMAJOR; FAST with no define; IDENTICAL. Compile owed: none.
