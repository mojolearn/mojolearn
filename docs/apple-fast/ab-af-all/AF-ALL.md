# AF-ALL: every Apple FAST experiment lane on one branch (2026-10-03)

Branch `lane/apple-fast-af-all`, cut from origin/main, with these lanes merged in this order: sym-hist, sym-iter,
sym-ctr, ptimpute, kde2, vsearch, optics2, hdbscan2, fa, bpe, sym-ordered, sym-multi, sym-est, sym-feat, mi, linsvr.
Every experiment is its own default-OFF `-D MOJOLEARN_*` define inside a FAST + Apple comptime guard. With every
define off, the branch compiles main's code. Nothing on this branch was compiled or run by the merge: **every build is
owed to the peer**, starting with FAST + each lane's ALL define, FAST with every define off, and IDENTICAL once per
binding.

All 204 request lines in the 16 lane files below now name `lane/apple-fast-af-all` as their branch, so the peer builds
one head. Tags did not change and are unique across docs/apple-fast/ab/.

| lane | binding | defines (ALL first) | request file | lines | compiled on its lane (lane .md / handoff) | owed compiles |
|---|---|---|---|---|---|---|
| sym-hist | gbdt | SYM_HIST_ALL, SYM_SORT_SWAP, SYM_RESOLVE_BLOCK, SYM_GATHER_FUSED, SYM_SCAN_SUB_FUSED, SYM_HIST_MULT, SYM_PART_STATS_PAR | sym-hist.txt | 15 | all rc=0 | none on the lane; af-all head |
| sym-iter | gbdt | SYM_ITER_ALL, SYM_BUF_ARENA, SYM_REUSE_PARTITION, SYM_DERIV_FUSED, SYM_LEAF_FROM_STATS | sym-iter.txt | 8 | FAST ALL rc=0 | each single, FAST off, IDENTICAL |
| sym-ordered | gbdt | ORD_ALL, ORD_FOLD_BINS_ONE, ORD_FOLD_INDEX, ORD_STD_PARALLEL, ORD_TREE_LEAN | sym-ordered.txt | 10 | all rc=0 | none on the lane; af-all head |
| sym-multi | gbdt | SYM_MULTI_ALL, MC_CLASS_BATCH_DERIV, MC_CLASS_BATCH_EST, PL_GROUP_NARROW, PL_PAIRS_ONCE, YR_TASK_FUSED | sym-multi.txt | 12 | all rc=0 (laptop .so; Metal side first checked on M3) | af-all head |
| sym-est | gbdt | SYM_EST_ALL, EST_STATS_FUSED, EST_SHRINK_FUSED, EST_REUSE_PART | sym-est.txt | 8 | NOT compiled (new 1071-line apple_fast_est.mojo) | all: ALL first, singles, off, IDENTICAL |
| sym-ctr | gbdt | SYM_CTR_ALL, SYM_CTR_PERM_BATCH, CTR_ONEHOT_DEVICE, CTR_PREP_SHARED, CTR_SORT_ONCE, CTR_INDEX_FUSED | sym-ctr.txt | 6 | ALL rc=1, two errors fixed, no rc=0 yet | all: ALL first, singles, off, IDENTICAL |
| sym-feat | gbdt | SYM_FEAT_ALL, GBDT_QUANT_DEVICE, GBDT_INDEX_PACK_DEVICE, GBDT_BOOT_DEVICE, GBDT_EVAL_FUSED, GBDT_EVAL_SKIP_EMPTY, GBDT_PREDICT_PACKED | sym-feat.txt | 13 | NOT compiled | all: ALL first, singles, off, IDENTICAL |
| ptimpute | x_prep | PTIMPUTE_ALL, PT_FOLD_NOX, PT_COLBATCH, PT_SPEC, PT_FUSED_TRANSFORM, SI_ONEPASS | ptimpute.txt | 14 | every single, FAST off, IDENTICAL rc=0 | ALL at head |
| mi | x_prep | MI_ALL, MI_REG_SORTCOUNT, MI_REG_TIES, MI_REG_RANKMAJOR, MI_FAST_FOLDS, MI_CLF_RANKMAJOR (+ MI_WORK env switch) | mi.txt | 20 | all rc=0 | none on the lane; af-all head |
| kde2 | estimators | KDE2_ALL, KDE_DIMTILE, KDE_LSE_FUSED, KDE_NORM_FUSED, KDE_KERNEL_VARIANTS, KDE_SAMPLE_FUSED | kde2.txt | 14 | FAST ALL rc=0 | each single, off, IDENTICAL (MOJOLEARN_SKIP_BUILD_GATE=1) |
| linsvr | estimators | LSVR_ALL, LSVR_FASTPATH_FIX, LSVR_FUSED_GRAD, LSVR_LINESEARCH_BATCH, LSVR_EVAL_SLIM, LSVR_DEVICE_CONVERGE, LSVR_DUAL_CD | linsvr.txt | 12 | first four + ALL rc=0; DEVICE_CONVERGE, DUAL_CD and the new ALL NOT compiled | DEVICE_CONVERGE, DUAL_CD, ALL, off, IDENTICAL |
| vsearch | ivf + x_ann | VSEARCH_ALL, IVF_KMEANS_LAZY_SHIFT, IVF_COARSE_RANDOM_INIT, IVF_DEVICE_VALIDATE, IVF_REFINE_TEAM, PQ_LUT_TILED, PQ_SCAN_FUSED | vsearch.txt | 27 | all rc=0, both bindings | none on the lane; af-all head |
| optics2 | x_cluster | OPTICS2_ALL, OPTICS_STEP_BATCH, OPTICS_FRONTIER_DEVICE, OPTICS_CORE_SQ, OPTICS_LIVEBUF | optics2.txt | 10 | FAST ALL rc=0 (FRONTIER_DEVICE never compiled) | each single, off, IDENTICAL |
| hdbscan2 | hdbscan | HDBSCAN2_ALL, HDB_CORE_TILE, HDB_DEV_BORUVKA, HDB_ONE_SYNC, HDB_SMR_TILED, HDB_LINKAGE_DEVICE, HDB_SELECT_DEVICE | hdbscan2.txt | 13 | first four, ALL, off, IDENTICAL rc=0 | LINKAGE_DEVICE, SELECT_DEVICE, ALL, off, IDENTICAL |
| fa | x_decomp | FA_ALL, FA_GRAM_ONCE, FA_EIG_SMALL, FA_ITER_DEVICE, FA_LIVEBUF, FA_LL_DEVICE, FA_TRANSFORM_FUSED | fa.txt | 14 | FAST ALL rc=0 | each single, off, IDENTICAL |
| bpe | tokenizer_fast (new) | BPE_ALL, BPE_TRAIN_DEVICE, BPE_MERGE_BATCH, BPE_GROUP_FILTER, BPE_ENCODE_DEVICE, BPE_LIVEBUF | bpe.txt | 8 | NOT compiled (py_compile only) | all |

Every lane's own status was recorded on its branch; the merged head adds the cross-lane resolutions below, so the
af-all build of each binding (FAST + every lane ALL that shares it, FAST off, IDENTICAL) is owed in any case.

## Merge resolutions (where two lanes edited the same code)

- `gbdt/methods/oblivious_tree_doc_parallel_structure_searcher.mojo` (sym-iter x sym-ordered): both new parameters
  kept (`sym_parts_out`, then `fold_part_off`, `obs_scratch`; every caller passes them by keyword). `d_observations`:
  SYM_BUF_ARENA's pooled handle when it applies (plain arm, `fold_count == 1`), else ORD_TREE_LEAN / ORD_FOLD_INDEX's
  choice, else main's `doc_count` allocation.
- `gbdt/methods/doc_parallel_boosting.mojo` loop head (sym-iter x sym-multi x sym-est): main's gradient dispatch sits
  inside sym-iter's `if not sym_skip_grad:`; inside it, sym-multi's MC_CLASS_BATCH_DERIV register launch for MultiClass,
  sym-est's `elif skip_derivs: pass` before the pointwise arms, and sym-multi's YR_TASK_FUSED magnitude block count
  before the folds.
- `_estimate_and_apply` (sym-iter x sym-est): parameters `yeti_seed, derivs_hook, tail_drain`; the perm-loop call passes
  `d_hook^` positionally and `tail_drain=not sym_fuse` by keyword. The sym-est Apple task returns before sym-iter's
  `tail_drain` logic, so it drains as on its own lane.
- `gbdt/train.mojo` (sym-ctr x sym-feat): both setup blocks kept (sym-ctr's pre-built grids, then sym-feat's device
  float matrix); `_quantize_training_columns` takes `pre_has/pre_borders/pre_folds` and `dev_cols`. The column sets are
  disjoint (sym-ctr: one-hot and CTR columns; sym-feat QUANT_DEVICE: raw float columns).
- `x_prep/device.mojo` (ptimpute x mi): both imports kept; their dispatch blocks auto-merged and handle disjoint ops
  (MI: 66, 68, 69, 70; PT: the pt_* ops).

No `comptime assert` was added: no two defines produce wrong code when both are on. The pairs below are still not
worth combining in one A/B arm.

## Define pairs that overlap (A/B each alone; pick one winner per pair)

| pair | why |
|---|---|
| SYM_LEAF_FROM_STATS (sym-iter) ~ EST_STATS_FUSED (sym-est) | both compute device Newton leaves; with LEAF_FROM_STATS on, the plain symmetric arm never calls `_estimate_and_apply`, so EST_* is inert there |
| SYM_DERIV_FUSED (sym-iter) ~ EST_SHRINK_FUSED (sym-est) | both move the next tree's derivative pass into this tree's tail; both on: the sym tail recomputes what the est task left and the head skips everything, so EST_SHRINK_FUSED gains nothing (correct, redundant) |
| sym-iter drains ~ GBDT_EVAL_FUSED (sym-feat) | EVAL_FUSED needs one drain per tree before the next packed `h_vals` write; in the merged code every tree still drains before its append (sym tail `synchronize`, the sym-est task's own drain), so they compose, but confirm the held-out losses match FAST off |
| SYM_BUF_ARENA (sym-iter) ~ ORD_TREE_LEAN / ORD_FOLD_INDEX (sym-ordered) | same `d_observations` slot; arena wins on the plain arm, ORD on the fold arm |
| SYM_REUSE_PARTITION (sym-iter) ~ EST_REUSE_PART (sym-est) | both reuse the searcher's partition for leaf estimation |
| kde2 x linsvr | share the estimators binding (independent code) |
| ptimpute x mi | share the x_prep binding and the `x_prep/device.mojo` dispatch (disjoint ops) |
| IVF_KMEANS_LAZY_SHIFT (vsearch) | lives in shared cluster/impl/detail/kmeans.mojo: also reaches KMeans in x_cluster |
| lane ALL defines | each ALL covers its own lane only; there is no cross-lane ALL |

## Flags from the lanes (HANDOFF_2026-10-03_af-lanes.md)

- sym-hist: SYM_PART_STATS_PAR uses float atomic adds (FAST run-to-run nondeterministic sums).
- vsearch: COARSE_RANDOM_INIT changes FAST bits (different start; needs a recall check); LAZY_SHIFT is in shared
  cluster/impl/detail/kmeans.mojo (check KMeans FAST lanes too).
- optics2: STEP_BATCH spins across threadgroups (Metal gives no co-residency guarantee), with a timeout fallback to
  main's path. FRONTIER_DEVICE was never compiled.
- bpe: pre-tokenization and corpus dedup still run on the host before upload; the A/B box also needs the host
  tokenizer binding. Nothing compiled.
- sym-feat: 2 unguarded edits (make_test_arm sizes via constants equal to main's values; new gbdt_fit_rowmajor binding
  entry + probe): verify IDENTICAL. EVAL_FUSED relies on the per-tree leaf readback draining the stream.
- sym-iter: SYM_LEAF_FROM_STATS ~ EST_STATS_FUSED; SYM_DERIV_FUSED ~ EST_SHRINK_FUSED; its removed drains interact with
  EVAL_FUSED. Do not combine overlapping pairs in one A/B arm.
- sym-ctr: never passed a build; its unguarded lines are empty defaults or constant-True conditions when off: verify
  IDENTICAL.
- sym-est: never compiled; FAST f32 leaf math vs the host's f64.
- fa: GRAM_ONCE uses an f32 covariance (squares the condition number vs main's SVD route): check quality on istella.
- mi: MI_WORK is an env switch read once at import, FAST only.
