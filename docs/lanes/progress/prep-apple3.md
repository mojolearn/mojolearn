# prep-apple3: progress

Apple FAST speed round 3 for the prep family (~/mojolearn-evidence/apple3_speed_brief.md): preprocessing,
feature selection, encoders, imputers, naive Bayes, LDA/QDA. Worktree `~/mojolearn-wt/prep-apple3`, branch
`lane/prep-apple3`, forked from lane/apple3-merged 6856b5f8f. Evidence: ~/mojolearn-evidence/prep-apple3/
(every job's command and full stdout). Round 2: docs/lanes/progress/prep-apple2.md.

A/B tool: `bench/x_prep_ab.sh` (round 2's), 1M rows (100k iterative-imputer and mutual-info-classif, 200k
multilabel-binarizer), reps 2, minimum reported, taxi + HIGGS from R2.

## The base (lane/apple3-merged 6856b5f8f)

### `_Prog.run` after the merge: every round-2 path is reached on a GPU install

Read of python/mojolearn/_expansion_prep.py `_Prog.run` and bindings/_mojolearn_x_prep.mojo: the GPU binding
exports x_prep_run, x_prep_run_scratch, x_prep_run_out, x_prep_run_ranges, x_prep_strat_folds and
x_prep_kfold_folds. With x_prep_run_out present, `dev_scratch` is on for any scratch, `dev_out` for an output
region, and the ranges runner (probed the same way) carries both sizes, so the device output, the device
scratch and the ranges transfer are all taken. `_native_folds` probes both fold entries. Nothing was lost in
the merge. Job 1 confirms it by the numbers (the `out0`, `ranges0` and TargetEncoder arms below).

### BROKEN on the base, fixed here: a read of an input slot

lane py-shared made x_prep refuse a read of an arena INPUT (inputs never come back from the device). Four
programs of this family read one, so on the base (every backend, ranges on or off) they raised
`AssertionError: x_prep: arena [..) was read but is an input`:

| estimator | what it read | fix |
|---|---|---|
| QuantileTransformer.fit | `references_` (an input no stage writes) | the attribute is the array that went up (same float32 words) |
| IterativeImputer fit_transform / transform | the filled block, imputed in place | `put(..., inout=True)`: the words come back |
| LinearDiscriminantAnalysis.fit (svd) | `meta` (lda_stage2 writes the ranks into it) | `put_list(..., inout=True)` |
| SimpleImputer.fit(strategy=callable), missing_values NaN | the input block itself | the input array |

`_Prog.put(arr, inout=True)` / `put_list(values, inout=True)`: an in/out input is uploaded like any input,
listed among the ranges that come back, never served from the resident store, and readable. Found by job 1
(XPERROR on quantile-transformer, iterative-imputer, lda, both datasets, both modes) and by a data-flow pass
over every `get` / `get_i32` / `values` call of the module (the SimpleImputer one).

## Job 1: the base, no code change (m4-a, Apple M4, request 1790626651574, commit 6856b5f8f)

Every digest, IDENTICAL and FAST, equals round 2's m4-a record (request 1790618180667), so the merge moved no
bit of this family. Seconds, minimum of 2 reps.

| what | mode | result |
|---|---|---|
| ranges runner (default) vs MOJOLEARN_ARENA_RANGES=0 | FAST | every case faster or equal with ranges: robust-scaler 0.573 -> 0.536, maxabs 0.105 -> 0.066, gaussian-nb 0.141 -> 0.075, multinomial-nb 0.134 -> 0.069, spline 0.369 -> 0.309, onehot taxi 1.597 -> 1.515; digests equal |
| ranges runner vs MOJOLEARN_ARENA_RANGES=0 | IDENTICAL | the same: gaussian-nb 0.360 -> 0.295, categorical-nb 0.695 -> 0.575, onehot taxi 1.588 -> 1.510; digests equal |
| output region (default) vs MOJOLEARN_XPREP_OUT=0 | FAST | taxi onehot 1.718 -> 1.515 (HIGGS below the size rule: 0.588 vs 0.584) |
| TargetEncoder parallel buckets MOJOLEARN_XPREP_TE_PBUCKET=1 (round 2: opt-in, never run) | FAST | taxi 1.204 -> 0.961, HIGGS 1.179 -> 0.930; digests equal |
| TargetEncoder parallel buckets | IDENTICAL | taxi 1.292 -> 1.017, HIGGS 1.243 -> 0.984; digests equal |
| PowerTransformer MOJOLEARN_XPREP_PT_INTERLEAVE=1 (round 2: opt-in) | IDENTICAL | taxi 3.518 -> 3.499, HIGGS 3.750 -> 3.757: no gain, stays opt-in |

### FAST phases on the M4 (XPPROF, taxi; the prefix of stage 0 carries the program's arena traffic)

| phase | seconds | cases |
|---|---|---|
| sort_cols, 16 columns x 1M | 0.44 to 0.48 | robust-scaler, kbins, simple-imputer (median AND mean), quantile-transformer |
| sort_cols, 9 columns x 1M | 0.28 | onehot, ordinal, target-encoder (the categories) |
| output traffic, 2.26 GB (566M words) | 1.13 | taxi onehot transform |
| output traffic, 0.97 GB | 0.61 | taxi label-binarizer transform |
| output traffic, 0.61 GB / 0.45 GB | 0.38 / 0.23 | poly-features / spline transform |
| te_bucket (serial) | 0.25 | target-encoder (the parallel buckets replace it) |
| mi_cd | 0.24 | mutual-info-classif (100k rows) |
| cat_counts | 0.17 | categorical-nb |
| Python outside the binding | 0.52 / 0.22 / 0.17 / 0.17 | multilabel-binarizer / target-encoder / label-encoder / label-binarizer |

SimpleImputer(strategy="mean") sorts every column and never reads the sort.

## Changes under test (job 2, m3ultra-b, request 1790627886703, tree b8f051632)

One job times three commits, each with its own build (bench/x_prep_ab.sh arms `base`, `at-<commit>-<label>`,
and the tree): the merged base 6856b5f8f, the tree without the radix sort 6ff627c6c, and the tree. The
command is tools/prep_apple3/job2.sh.

| change | mode | switch (state before its A/B) | where |
|---|---|---|---|
| in/out inputs (the fix above) | both | none (a fix) | python/mojolearn/_expansion_prep.py |
| TargetEncoder parallel buckets | both | default ON since job 1; MOJOLEARN_XPREP_TE_PBUCKET=0 | _expansion_prep.py |
| `imputer_nosort`: SimpleImputer sorts only for median / most_frequent | both | opt-in MOJOLEARN_XPREP_R3_ON | _expansion_prep.py |
| `mapped`: a host arena or output of 2^18 words or more is an anonymous mapping, never touched to allocate | both | opt-in | _expansion_prep.py |
| `view`: a read of 2^20 words or more is a view of the mapped block, not a copy | both | opt-in | _expansion_prep.py |
| `work`: the sorted columns (RobustScaler, SimpleImputer, KBins, QuantileTransformer, the encoders' categories, spline knots, weighted groups) and LDA's centered rows are device scratch | both | opt-in | _expansion_prep.py |
| `te_arrays`: TargetEncoder's binary target and folds cross as int32 arrays (i2f on the device), one label pass | both | opt-in | _expansion_prep.py |
| radix sort_cols (x_prep/dradix.mojo): 4 passes of a stable counting sort by chunks on a monotone 32-bit key, in place of the 41 bitonic passes; the same words (Python model of the key map and of the sort against `word_order`: 204017 words with every NaN and zero class, 6 shapes) | FAST, Apple only (comptime) | opt-in MOJOLEARN_XPREP_SORT_RADIX=1; MOJOLEARN_XPREP_SORT_CHUNK positions per chunk | x_prep/dradix.mojo, x_prep/device.mojo |
| XPPHASE / XPPROG phase lines | timing only | MOJOLEARN_XPREP_PROFILE=1 | x_prep/device.mojo, _expansion_prep.py |
