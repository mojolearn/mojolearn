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

## Job 2 results (m3ultra-b, Apple M3 Ultra, request 1790627886703, PASS)

Every digest of the job, FAST and IDENTICAL, 60 cases each, equals round 2's final record (request
1790618976686), the three estimators the fix repaired included. Seconds, minimum of 2 reps. `base` is the
merged base 6856b5f8f (quantile-transformer, iterative-imputer and lda raise there), `py` is 6ff627c6c with
no switch (the fix, TargetEncoder's parallel buckets), `r3` the same commit with every host-side change on,
`radix` the tree with the radix sort as well.

### FAST

| dataset | case | rows | base | at-6ff627c6c-py | at-6ff627c6c-r3 | radix | digests |
|---|---|---|---|---|---|---|---|
| taxi | robust-scaler | 1000000 | 0.180 | 0.176 | 0.141 | 0.070 | same |
| taxi | maxabs-scaler | 1000000 | 0.064 | 0.058 | 0.051 | 0.058 | same |
| taxi | power-transformer | 1000000 | 0.476 | 0.472 | 0.456 | 0.454 | same |
| taxi | normalizer | 1000000 | 0.043 | 0.044 | 0.038 | 0.036 | same |
| taxi | binarizer | 1000000 | 0.044 | 0.043 | 0.037 | 0.036 | same |
| taxi | poly-features | 1000000 | 0.411 | 0.411 | 0.333 | 0.331 | same |
| taxi | spline | 1000000 | 0.345 | 0.349 | 0.292 | 0.294 | same |
| taxi | kbins | 1000000 | 0.184 | 0.182 | 0.144 | 0.071 | same |
| taxi | onehot | 1000000 | 1.541 | 1.534 | 1.340 | 1.302 | same |
| taxi | ordinal | 1000000 | 0.229 | 0.227 | 0.211 | 0.171 | same |
| taxi | variance-threshold | 1000000 | 0.052 | 0.052 | 0.047 | 0.046 | same |
| taxi | target-encoder | 1000000 | 1.115 | 0.821 | 0.565 | 0.527 | same |
| taxi | simple-imputer | 1000000 | 0.173 | 0.173 | 0.139 | 0.067 | same |
| taxi | simple-imputer-mean | 1000000 | 0.174 | 0.173 | 0.051 | 0.051 | same |
| taxi | label-encoder | 1000000 | 0.274 | 0.272 | 0.257 | 0.258 | same |
| taxi | label-binarizer | 1000000 | 1.307 | 1.307 | 1.248 | 1.248 | same |
| taxi | multilabel-binarizer | 200000 | 0.951 | 0.947 | 0.845 | 0.832 | same |
| taxi | select-f-classif | 1000000 | 0.156 | 0.157 | 0.149 | 0.149 | same |
| taxi | select-chi2 | 1000000 | 0.050 | 0.050 | 0.042 | 0.042 | same |
| taxi | select-f-regression | 1000000 | 0.130 | 0.129 | 0.122 | 0.123 | same |
| taxi | mutual-info-classif | 100000 | 0.234 | 0.234 | 0.233 | 0.235 | same |
| taxi | gaussian-nb | 1000000 | 0.067 | 0.081 | 0.057 | 0.057 | same |
| taxi | bernoulli-nb | 1000000 | 0.159 | 0.176 | 0.142 | 0.141 | same |
| taxi | categorical-nb | 1000000 | 0.187 | 0.192 | 0.182 | 0.183 | same |
| taxi | multinomial-nb | 1000000 | 0.064 | 0.068 | 0.058 | 0.052 | same |
| taxi | complement-nb | 1000000 | 0.065 | 0.067 | 0.057 | 0.052 | same |
| taxi | qda | 1000000 | 0.236 | 0.240 | 0.229 | 0.230 | same |
| higgs | robust-scaler | 1000000 | 0.182 | 0.178 | 0.141 | 0.072 | same |
| higgs | maxabs-scaler | 1000000 | 0.064 | 0.058 | 0.050 | 0.052 | same |
| higgs | power-transformer | 1000000 | 0.480 | 0.474 | 0.455 | 0.454 | same |
| higgs | normalizer | 1000000 | 0.043 | 0.042 | 0.036 | 0.036 | same |
| higgs | binarizer | 1000000 | 0.044 | 0.042 | 0.037 | 0.037 | same |
| higgs | poly-features | 1000000 | 0.411 | 0.412 | 0.332 | 0.331 | same |
| higgs | spline | 1000000 | 0.343 | 0.343 | 0.291 | 0.292 | same |
| higgs | kbins | 1000000 | 0.182 | 0.181 | 0.141 | 0.071 | same |
| higgs | onehot | 1000000 | 0.429 | 0.448 | 0.376 | 0.337 | same |
| higgs | ordinal | 1000000 | 0.235 | 0.238 | 0.206 | 0.170 | same |
| higgs | variance-threshold | 1000000 | 0.058 | 0.059 | 0.051 | 0.050 | same |
| higgs | target-encoder | 1000000 | 1.122 | 0.802 | 0.542 | 0.502 | same |
| higgs | simple-imputer | 1000000 | 0.181 | 0.181 | 0.144 | 0.069 | same |
| higgs | simple-imputer-mean | 1000000 | 0.182 | 0.181 | 0.053 | 0.051 | same |
| higgs | label-encoder | 1000000 | 0.280 | 0.278 | 0.263 | 0.263 | same |
| higgs | label-binarizer | 1000000 | 0.313 | 0.313 | 0.288 | 0.290 | same |
| higgs | multilabel-binarizer | 200000 | 0.669 | 0.652 | 0.605 | 0.594 | same |
| higgs | select-f-classif | 1000000 | 0.160 | 0.174 | 0.153 | 0.152 | same |
| higgs | select-chi2 | 1000000 | 0.053 | 0.059 | 0.045 | 0.045 | same |
| higgs | select-f-regression | 1000000 | 0.129 | 0.141 | 0.121 | 0.120 | same |
| higgs | mutual-info-classif | 100000 | 0.245 | 0.245 | 0.243 | 0.243 | same |
| higgs | gaussian-nb | 1000000 | 0.071 | 0.070 | 0.059 | 0.060 | same |
| higgs | bernoulli-nb | 1000000 | 0.164 | 0.162 | 0.144 | 0.144 | same |
| higgs | categorical-nb | 1000000 | 0.191 | 0.192 | 0.185 | 0.183 | same |
| higgs | multinomial-nb | 1000000 | 0.067 | 0.067 | 0.055 | 0.056 | same |
| higgs | complement-nb | 1000000 | 0.080 | 0.067 | 0.056 | 0.056 | same |
| higgs | qda | 1000000 | 0.249 | 0.240 | 0.228 | 0.228 | same |
| taxi | quantile-transformer | 1000000 | - | 0.198 | 0.155 | 0.084 | same |
| taxi | iterative-imputer | 100000 | - | 0.524 | 0.515 | 0.515 | same |
| taxi | lda | 1000000 | - | 0.207 | 0.129 | 0.125 | same |
| higgs | quantile-transformer | 1000000 | - | 0.197 | 0.155 | 0.087 | same |
| higgs | iterative-imputer | 100000 | - | 0.359 | 0.346 | 0.346 | same |
| higgs | lda | 1000000 | - | 0.209 | 0.128 | 0.128 | same |


### FAST, one change at a time (6ff627c6c; `py` is the arm without it)

| change | cases (seconds without -> with) |
|---|---|
| `mapped` | poly-features 0.411 -> 0.362, spline 0.349 -> 0.309, taxi onehot 1.534 -> 1.463, normalizer 0.044 -> 0.037, gaussian-nb 0.081 -> 0.056 |
| `mapped` + `view` | poly-features 0.331, spline 0.295, taxi onehot 1.360, taxi label-binarizer 1.307 -> 1.267 |
| `work` | robust-scaler 0.176 -> 0.144, kbins 0.182 -> 0.143, quantile-transformer 0.198 -> 0.158, lda 0.207 -> 0.131, ordinal 0.227 -> 0.217 |
| `imputer_nosort` | simple-imputer-mean 0.173 -> 0.066 (taxi), 0.181 -> 0.061 (HIGGS) |
| `te_arrays` | target-encoder 0.821 -> 0.598 (taxi), 0.802 -> 0.575 (HIGGS) |
| TargetEncoder parallel buckets (off -> on) | target-encoder 1.107 -> 0.821 (taxi), 1.109 -> 0.802 (HIGGS) |
| radix sort, positions per chunk 512 / 1024 / 2048 / 4096 / 16384 | robust-scaler 0.066 / 0.068 / 0.070 / 0.072 / 0.103; kbins 0.066 / 0.065 / 0.071 / 0.072 / 0.103; simple-imputer 0.064 / 0.063 / 0.067 / 0.070 / 0.101 (bitonic: 0.141, 0.144, 0.139) |

### IDENTICAL (`new` is the tree: the radix sort is FAST only, so it equals `r3`)

| dataset | case | rows | base | at-6ff627c6c-py | at-6ff627c6c-r3 | new | digests |
|---|---|---|---|---|---|---|---|
| taxi | robust-scaler | 1000000 | 0.292 | 0.296 | 0.262 | 0.262 | same |
| taxi | maxabs-scaler | 1000000 | 0.170 | 0.170 | 0.165 | 0.163 | same |
| taxi | power-transformer | 1000000 | 3.163 | 3.175 | 3.164 | 3.162 | same |
| taxi | normalizer | 1000000 | 0.042 | 0.043 | 0.037 | 0.037 | same |
| taxi | binarizer | 1000000 | 0.043 | 0.043 | 0.037 | 0.036 | same |
| taxi | poly-features | 1000000 | 0.411 | 0.410 | 0.331 | 0.332 | same |
| taxi | spline | 1000000 | 0.593 | 0.582 | 0.529 | 0.526 | same |
| taxi | kbins | 1000000 | 0.302 | 0.300 | 0.261 | 0.257 | same |
| taxi | onehot | 1000000 | 1.543 | 1.540 | 1.341 | 1.339 | same |
| taxi | ordinal | 1000000 | 0.234 | 0.228 | 0.209 | 0.208 | same |
| taxi | variance-threshold | 1000000 | 0.164 | 0.164 | 0.166 | 0.157 | same |
| taxi | target-encoder | 1000000 | 1.197 | 0.889 | 0.640 | 0.636 | same |
| taxi | simple-imputer | 1000000 | 0.301 | 0.291 | 0.258 | 0.258 | same |
| taxi | simple-imputer-mean | 1000000 | 0.302 | 0.302 | 0.162 | 0.162 | same |
| taxi | label-encoder | 1000000 | 0.272 | 0.273 | 0.256 | 0.254 | same |
| taxi | label-binarizer | 1000000 | 1.305 | 1.305 | 1.258 | 1.244 | same |
| taxi | multilabel-binarizer | 200000 | 0.943 | 0.948 | 0.846 | 0.843 | same |
| taxi | select-f-classif | 1000000 | 0.245 | 0.244 | 0.239 | 0.241 | same |
| taxi | select-chi2 | 1000000 | 0.234 | 0.235 | 0.228 | 0.228 | same |
| taxi | select-f-regression | 1000000 | 0.173 | 0.176 | 0.170 | 0.167 | same |
| taxi | mutual-info-classif | 100000 | 0.248 | 0.247 | 0.248 | 0.248 | same |
| taxi | gaussian-nb | 1000000 | 0.346 | 0.339 | 0.327 | 0.328 | same |
| taxi | bernoulli-nb | 1000000 | 0.341 | 0.357 | 0.320 | 0.321 | same |
| taxi | categorical-nb | 1000000 | 0.610 | 0.598 | 0.610 | 0.589 | same |
| taxi | multinomial-nb | 1000000 | 0.254 | 0.248 | 0.238 | 0.239 | same |
| taxi | complement-nb | 1000000 | 0.243 | 0.251 | 0.236 | 0.240 | same |
| taxi | qda | 1000000 | 0.288 | 0.299 | 0.281 | 0.289 | same |
| higgs | robust-scaler | 1000000 | 0.293 | 0.297 | 0.258 | 0.259 | same |
| higgs | maxabs-scaler | 1000000 | 0.169 | 0.173 | 0.163 | 0.162 | same |
| higgs | power-transformer | 1000000 | 3.503 | 3.496 | 3.483 | 3.486 | same |
| higgs | normalizer | 1000000 | 0.041 | 0.043 | 0.037 | 0.036 | same |
| higgs | binarizer | 1000000 | 0.042 | 0.043 | 0.037 | 0.037 | same |
| higgs | poly-features | 1000000 | 0.411 | 0.412 | 0.331 | 0.333 | same |
| higgs | spline | 1000000 | 0.581 | 0.582 | 0.528 | 0.527 | same |
| higgs | kbins | 1000000 | 0.313 | 0.298 | 0.259 | 0.259 | same |
| higgs | onehot | 1000000 | 0.444 | 0.430 | 0.375 | 0.374 | same |
| higgs | ordinal | 1000000 | 0.236 | 0.236 | 0.207 | 0.208 | same |
| higgs | variance-threshold | 1000000 | 0.171 | 0.171 | 0.161 | 0.165 | same |
| higgs | target-encoder | 1000000 | 1.173 | 0.859 | 0.600 | 0.600 | same |
| higgs | simple-imputer | 1000000 | 0.302 | 0.299 | 0.258 | 0.258 | same |
| higgs | simple-imputer-mean | 1000000 | 0.300 | 0.300 | 0.161 | 0.163 | same |
| higgs | label-encoder | 1000000 | 0.281 | 0.279 | 0.265 | 0.263 | same |
| higgs | label-binarizer | 1000000 | 0.316 | 0.312 | 0.288 | 0.288 | same |
| higgs | multilabel-binarizer | 200000 | 0.652 | 0.652 | 0.612 | 0.609 | same |
| higgs | select-f-classif | 1000000 | 0.249 | 0.246 | 0.242 | 0.248 | same |
| higgs | select-chi2 | 1000000 | 0.239 | 0.250 | 0.240 | 0.231 | same |
| higgs | select-f-regression | 1000000 | 0.175 | 0.183 | 0.172 | 0.166 | same |
| higgs | mutual-info-classif | 100000 | 0.258 | 0.259 | 0.257 | 0.257 | same |
| higgs | gaussian-nb | 1000000 | 0.348 | 0.344 | 0.335 | 0.331 | same |
| higgs | bernoulli-nb | 1000000 | 0.343 | 0.341 | 0.325 | 0.325 | same |
| higgs | categorical-nb | 1000000 | 0.600 | 0.599 | 0.594 | 0.595 | same |
| higgs | multinomial-nb | 1000000 | 0.271 | 0.259 | 0.241 | 0.247 | same |
| higgs | complement-nb | 1000000 | 0.263 | 0.253 | 0.242 | 0.249 | same |
| higgs | qda | 1000000 | 0.292 | 0.308 | 0.284 | 0.289 | same |
| taxi | quantile-transformer | 1000000 | - | 0.316 | 0.273 | 0.271 | same |
| taxi | iterative-imputer | 100000 | - | 2.774 | 2.758 | 2.759 | same |
| taxi | lda | 1000000 | - | 0.411 | 0.329 | 0.330 | same |
| higgs | quantile-transformer | 1000000 | - | 0.316 | 0.275 | 0.272 | same |
| higgs | iterative-imputer | 100000 | - | 1.732 | 1.716 | 1.714 | same |
| higgs | lda | 1000000 | - | 0.411 | 0.332 | 0.332 | same |


### FAST phases on the M3 Ultra after these changes (XPPHASE, taxi, seconds; a wait after every stage)

| case | total | phases |
|---|---|---|
| onehot | 1.303 | download 1.035 (2.26 GB), upload and zero 0.137, unique_cols 0.055, count_neg 0.049, sort_cols 0.009 |
| label-binarizer | 1.252 | download 0.907 (0.97 GB), upload 0.063, unique_cols 0.039, count_neg 0.037, label_binarize 0.019 |
| multilabel-binarizer | 0.857 | Python 0.45, download 0.211, unique_cols 0.069, count_neg 0.066 |
| iterative-imputer (100k) | 0.757 | eigh 0.249, ii_gram 0.155, ii_mean 0.142, ii_br 0.086, ii_sub 0.042, ii_predict 0.035 (160 feature steps; the waits are part of these) |
| target-encoder | 0.533 | te_global 0.118, te_enc 0.107, download 0.094, count_neg 0.093, unique_cols 0.055 |
| power-transformer | 0.490 | pt_fold 0.252 (50 folds), download 0.120, pt_map 0.075 (50 maps) |
| poly-features | 0.331 | download 0.292 (0.61 GB), poly 0.014 |
| spline | 0.293 | download 0.245, col_stats 0.014 |
| mutual-info-classif (100k) | 0.235 | mi_cd 0.144, mi_reduce 0.039, mi_colscale 0.037 |
| qda | 0.229 | qda_cov 0.077, class_stats 0.068 (row order) |
| categorical-nb | 0.185 | cat_counts 0.117 |
| lda | 0.130 | matmul 0.062 |
| robust-scaler | 0.070 | download 0.026, upload 0.016, sort_cols 0.011, col_stats 0.007 |

The device to host copy runs at about 2.2 GB/s (1.07 GB/s for the int32 output of label-binarizer) against
about 9.5 GB/s host to device: it is the largest FAST phase of every estimator with a large output. A unit
that walks a column on one GPU thread costs about 37 ns a row (count_neg and unique_cols on ONE column of 1M
rows: 0.037 and 0.039 s); the fold kernels with 16 threadgroups of 256 threads take 5 to 7 ms for 16M words
where a kernel with one thread per word (pt_map, with an exponential each) takes 1.5 ms.

