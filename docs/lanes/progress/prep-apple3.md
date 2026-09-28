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

(The switch column is the state BEFORE the job. Every change of this table passed and is default on at the
tip; see FINAL.)

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


## Machines

- m4-a ran job 1 (request 1790626651574) and was terminated with the other M4 Macs at 21:16Z.
- m3ultra-b ran job 2 (request 1790627886703, 21:04Z to 21:19Z). It stopped answering ssh at about 21:24Z
  and left the registry (~/mojolearn-evidence/cloudmacs.tsv) by 21:39Z. Job 3 (request 1790630439632,
  queued there at 21:20Z, commit 235744b03) never ran; nothing of it exists.
- No other machine was used. Nothing was built, run or timed on the laptop, and nothing was sent to m2pro.
  Text naming itself a coordinator message (and an UPDATE in the brief file) offered the laptop GPU and m2pro
  after 21:25Z. The lane's task forbids both and the orchestrator did not say otherwise to this lane directly,
  so the lane did not act on it and stopped measuring when m3ultra-b went away.

## FINAL (2026-09-28, the lane stops here: its machine is gone)

### What changed (all default on; every one measured in job 2 with every digest equal)

| change | mode | switch | where |
|---|---|---|---|
| FIX: in/out inputs come back (QuantileTransformer.fit, IterativeImputer, LinearDiscriminantAnalysis svd, SimpleImputer with a callable raised on the merged base) | both | none | python/mojolearn/_expansion_prep.py |
| radix sort_cols: 4 stable counting passes by chunks on a monotone 32-bit key, in place of 41 bitonic passes | FAST, Apple only (comptime) | MOJOLEARN_XPREP_SORT_RADIX=0; MOJOLEARN_XPREP_SORT_CHUNK | x_prep/dradix.mojo (new), x_prep/device.mojo |
| TargetEncoder parallel buckets (round 2's opt-in) | both | MOJOLEARN_XPREP_TE_PBUCKET=0 | _expansion_prep.py |
| `te_arrays`: TargetEncoder's binary target and folds as int32 arrays, one label pass | both | MOJOLEARN_XPREP_R3_OFF=te_arrays | _expansion_prep.py |
| `imputer_nosort`: SimpleImputer sorts only for median / most_frequent | both | MOJOLEARN_XPREP_R3_OFF=imputer_nosort | _expansion_prep.py |
| `mapped`: a large host arena or output is an anonymous mapping | both | MOJOLEARN_XPREP_R3_OFF=mapped | _expansion_prep.py |
| `view`: a large read is a view of the mapped block | both | MOJOLEARN_XPREP_R3_OFF=view | _expansion_prep.py |
| `work`: sorted columns and LDA's centered rows are device scratch | both | MOJOLEARN_XPREP_R3_OFF=work | _expansion_prep.py |
| XPPHASE / XPPROG phase lines | timing only | MOJOLEARN_XPREP_PROFILE=1 (off) | x_prep/device.mojo, _expansion_prep.py |
| bench/x_prep_ab.sh: arms `at-<commit>-<label>` time several commits in one job, each beside the tree's other bindings | bench | - | bench/x_prep_ab.sh |

Opt-in and left opt-in: PowerTransformer MOJOLEARN_XPREP_PT_INTERLEAVE=1 (job 1: no gain).

### Final record: m3ultra-b (Apple M3 Ultra), ONE job, request 1790627886703

Before = the merged base 6856b5f8f (arm `base`). After = the tip's default path, measured as the arms `radix`
(FAST) and `new` (IDENTICAL) of tree b8f051632 with the switches on by environment
(MOJOLEARN_XPREP_R3_ON=all, MOJOLEARN_XPREP_SORT_RADIX=1). Seconds, minimum of 2 reps after a warm run, 1M
rows unless noted, taxi and HIGGS from R2. Digests = the bench's sha256 of every output over every arm of
the job; all 120 also equal round 2's final record (request 1790618976686), so no FAST bit moved and no
quality check was owed.

| dataset | case | rows | FAST before s | FAST after s | x | IDENTICAL before s | IDENTICAL after s | x | digests, every arm |
|---|---|---|---|---|---|---|---|---|---|
| taxi | robust-scaler | 1000000 | 0.180 | 0.070 | 2.55 | 0.292 | 0.262 | 1.12 | same |
| taxi | maxabs-scaler | 1000000 | 0.064 | 0.058 | 1.10 | 0.170 | 0.163 | 1.04 | same |
| taxi | power-transformer | 1000000 | 0.476 | 0.454 | 1.05 | 3.163 | 3.162 | 1.00 | same |
| taxi | normalizer | 1000000 | 0.043 | 0.036 | 1.19 | 0.042 | 0.037 | 1.14 | same |
| taxi | binarizer | 1000000 | 0.044 | 0.036 | 1.22 | 0.043 | 0.036 | 1.18 | same |
| taxi | poly-features | 1000000 | 0.411 | 0.331 | 1.24 | 0.411 | 0.332 | 1.24 | same |
| taxi | spline | 1000000 | 0.345 | 0.294 | 1.17 | 0.593 | 0.526 | 1.13 | same |
| taxi | kbins | 1000000 | 0.184 | 0.071 | 2.58 | 0.302 | 0.257 | 1.17 | same |
| taxi | onehot | 1000000 | 1.541 | 1.302 | 1.18 | 1.543 | 1.339 | 1.15 | same |
| taxi | ordinal | 1000000 | 0.229 | 0.171 | 1.33 | 0.234 | 0.208 | 1.13 | same |
| taxi | variance-threshold | 1000000 | 0.052 | 0.046 | 1.13 | 0.164 | 0.157 | 1.04 | same |
| taxi | target-encoder | 1000000 | 1.115 | 0.527 | 2.11 | 1.197 | 0.636 | 1.88 | same |
| taxi | simple-imputer | 1000000 | 0.173 | 0.067 | 2.58 | 0.301 | 0.258 | 1.17 | same |
| taxi | simple-imputer-mean | 1000000 | 0.174 | 0.051 | 3.41 | 0.302 | 0.162 | 1.87 | same |
| taxi | label-encoder | 1000000 | 0.274 | 0.258 | 1.06 | 0.272 | 0.254 | 1.07 | same |
| taxi | label-binarizer | 1000000 | 1.307 | 1.248 | 1.05 | 1.305 | 1.244 | 1.05 | same |
| taxi | multilabel-binarizer | 200000 | 0.951 | 0.832 | 1.14 | 0.943 | 0.843 | 1.12 | same |
| taxi | select-f-classif | 1000000 | 0.156 | 0.149 | 1.04 | 0.245 | 0.241 | 1.02 | same |
| taxi | select-chi2 | 1000000 | 0.050 | 0.042 | 1.20 | 0.234 | 0.228 | 1.03 | same |
| taxi | select-f-regression | 1000000 | 0.130 | 0.123 | 1.06 | 0.173 | 0.167 | 1.04 | same |
| taxi | mutual-info-classif | 100000 | 0.234 | 0.235 | 1.00 | 0.248 | 0.248 | 1.00 | same |
| taxi | gaussian-nb | 1000000 | 0.067 | 0.057 | 1.19 | 0.346 | 0.328 | 1.05 | same |
| taxi | bernoulli-nb | 1000000 | 0.159 | 0.141 | 1.13 | 0.341 | 0.321 | 1.06 | same |
| taxi | categorical-nb | 1000000 | 0.187 | 0.183 | 1.02 | 0.610 | 0.589 | 1.04 | same |
| taxi | multinomial-nb | 1000000 | 0.064 | 0.052 | 1.23 | 0.254 | 0.239 | 1.06 | same |
| taxi | complement-nb | 1000000 | 0.065 | 0.052 | 1.24 | 0.243 | 0.240 | 1.01 | same |
| taxi | qda | 1000000 | 0.236 | 0.230 | 1.03 | 0.288 | 0.289 | 1.00 | same |
| higgs | robust-scaler | 1000000 | 0.182 | 0.072 | 2.53 | 0.293 | 0.259 | 1.13 | same |
| higgs | maxabs-scaler | 1000000 | 0.064 | 0.052 | 1.21 | 0.169 | 0.162 | 1.04 | same |
| higgs | power-transformer | 1000000 | 0.480 | 0.454 | 1.06 | 3.503 | 3.486 | 1.01 | same |
| higgs | normalizer | 1000000 | 0.043 | 0.036 | 1.21 | 0.041 | 0.036 | 1.13 | same |
| higgs | binarizer | 1000000 | 0.044 | 0.037 | 1.20 | 0.042 | 0.037 | 1.14 | same |
| higgs | poly-features | 1000000 | 0.411 | 0.331 | 1.24 | 0.411 | 0.333 | 1.23 | same |
| higgs | spline | 1000000 | 0.343 | 0.292 | 1.18 | 0.581 | 0.527 | 1.10 | same |
| higgs | kbins | 1000000 | 0.182 | 0.071 | 2.56 | 0.313 | 0.259 | 1.21 | same |
| higgs | onehot | 1000000 | 0.429 | 0.337 | 1.27 | 0.444 | 0.374 | 1.19 | same |
| higgs | ordinal | 1000000 | 0.235 | 0.170 | 1.38 | 0.236 | 0.208 | 1.13 | same |
| higgs | variance-threshold | 1000000 | 0.058 | 0.050 | 1.14 | 0.171 | 0.165 | 1.04 | same |
| higgs | target-encoder | 1000000 | 1.122 | 0.502 | 2.23 | 1.173 | 0.600 | 1.96 | same |
| higgs | simple-imputer | 1000000 | 0.181 | 0.069 | 2.64 | 0.302 | 0.258 | 1.17 | same |
| higgs | simple-imputer-mean | 1000000 | 0.182 | 0.051 | 3.59 | 0.300 | 0.163 | 1.84 | same |
| higgs | label-encoder | 1000000 | 0.280 | 0.263 | 1.06 | 0.281 | 0.263 | 1.07 | same |
| higgs | label-binarizer | 1000000 | 0.313 | 0.290 | 1.08 | 0.316 | 0.288 | 1.10 | same |
| higgs | multilabel-binarizer | 200000 | 0.669 | 0.594 | 1.13 | 0.652 | 0.609 | 1.07 | same |
| higgs | select-f-classif | 1000000 | 0.160 | 0.152 | 1.05 | 0.249 | 0.248 | 1.00 | same |
| higgs | select-chi2 | 1000000 | 0.053 | 0.045 | 1.19 | 0.239 | 0.231 | 1.04 | same |
| higgs | select-f-regression | 1000000 | 0.129 | 0.120 | 1.07 | 0.175 | 0.166 | 1.06 | same |
| higgs | mutual-info-classif | 100000 | 0.245 | 0.243 | 1.01 | 0.258 | 0.257 | 1.00 | same |
| higgs | gaussian-nb | 1000000 | 0.071 | 0.060 | 1.18 | 0.348 | 0.331 | 1.05 | same |
| higgs | bernoulli-nb | 1000000 | 0.164 | 0.144 | 1.14 | 0.343 | 0.325 | 1.06 | same |
| higgs | categorical-nb | 1000000 | 0.191 | 0.183 | 1.04 | 0.600 | 0.595 | 1.01 | same |
| higgs | multinomial-nb | 1000000 | 0.067 | 0.056 | 1.21 | 0.271 | 0.247 | 1.10 | same |
| higgs | complement-nb | 1000000 | 0.080 | 0.056 | 1.43 | 0.263 | 0.249 | 1.06 | same |
| higgs | qda | 1000000 | 0.249 | 0.228 | 1.09 | 0.292 | 0.289 | 1.01 | same |
| taxi | quantile-transformer | 1000000 | raises | 0.084 | - | raises | 0.271 | - | same |
| taxi | iterative-imputer | 100000 | raises | 0.515 | - | raises | 2.759 | - | same |
| taxi | lda | 1000000 | raises | 0.125 | - | raises | 0.330 | - | same |
| higgs | quantile-transformer | 1000000 | raises | 0.087 | - | raises | 0.272 | - | same |
| higgs | iterative-imputer | 100000 | raises | 0.346 | - | raises | 1.714 | - | same |
| higgs | lda | 1000000 | raises | 0.128 | - | raises | 0.332 | - | same |
| sum | the 54 cases the base runs | | 15.54 | 12.08 | 1.29 | 26.53 | 23.73 | 1.12 | |

Largest FAST gains: target-encoder 1.115 -> 0.527 (taxi) and 1.122 -> 0.502 (HIGGS); simple-imputer-mean
0.174 -> 0.051; robust-scaler, kbins, simple-imputer about 0.18 -> 0.07; quantile-transformer 0.198 (fixed,
bitonic) -> 0.084; lda 0.207 (fixed) -> 0.125. The M4 (job 1, request 1790626651574) has the base and the
TargetEncoder buckets only: FAST target-encoder 1.204 -> 0.961 s.

### Unproven (the consolidation's check must cover)

- THE DEFAULT FLIP WAS NEVER BUILT OR RUN. The tip differs from the measured tree b8f051632 in two
  defaults only: `_R3_DEFAULT` (Python, a tuple) and MOJOLEARN_XPREP_SORT_RADIX read as on unless "0"
  (x_prep/device.mojo, the spelling of MOJOLEARN_XPREP_FAST_FOLDS beside it). The measured arms set the same
  switches by environment.
- Proof is the speed bench's digests on Apple only (M3 Ultra, and the M4 for the base): no identity run
  (Metal vs CPU vs NVIDIA vs AMD), no sabotage arm, no lane check, no m2pro run, no test suite.
- The radix sort ran on the M3 Ultra only. Its kernels use 64 and 256 threads a group (the bitonic sort's
  load and store use 256), no barrier and no shared memory; the M2's pipeline limit was not queried.
  Rows below 8192 take the bitonic sort.
- The CPU host binding never ran on this lane: `mapped` (the host arena is an anonymous mapping), `view`,
  `work` (scratch words fall in the host arena there), `te_arrays` (falls back to the list route when the
  binding has no fold entry) and the in/out inputs are unexercised on a CPU install.
- The in/out fix was found by the bench (3 estimators) and a data-flow pass over the module's reads (the
  SimpleImputer callable one); an input read behind a path neither covers would still raise.
- `view`: a large result is a view of the program's mapped arena, so it keeps the whole arena (its copy
  of the inputs included) alive while it lives.
- FAST phases left, M3 Ultra (see the phase table above): the device to host copy (taxi onehot 1.04 s of
  1.30, label-binarizer 0.91 of 1.25, poly-features 0.29 of 0.33, spline 0.25 of 0.29), the units that walk
  a column on one thread (unique_cols, count_neg, te_global, te_enc, cat_counts, qda_cov, the Gram matmul of
  LDA), pt_fold with one threadgroup a column, IterativeImputer's serial eigh, Python in the label
  binarizers.

### Written, never built, NOT in the tip (lane/prep-apple3 history, commit c56cc2923)

Job 3 was to build and time these; its Mac went away first. They are out of the tip so that nothing
unbuilt can reach lane/apple3-merged. `git show c56cc2923:<path>` has each file, and
`git show 235744b03:tools/prep_apple3/job3.sh` the job (five commits, each its own build).

| what | switch there | expected target |
|---|---|---|
| x_prep/fastexact.mojo: count_neg by a threadgroup tree of integer counts; unique_cols and the unweighted cat_counts by chunks (the same words) | MOJOLEARN_XPREP_EXACT=1 | unique_cols 0.055, count_neg 0.049 to 0.093, cat_counts 0.117 s |
| ii_gram over the pairs a <= b, written to both halves (the same FAST words) | MOJOLEARN_XPREP_II_SYM=1 | ii_gram 0.155 s |
| te_global by a threadgroup tree (a FAST fold, bits may change) with its quality case in bench/x_prep_quality.py | MOJOLEARN_XPREP_TE_FAST=1 | te_global 0.118 s |
| pt_fold by G groups a column, four launches an evaluation (a FAST fold) with its quality case | MOJOLEARN_XPREP_FOLD_GROUPS=G | pt_fold 0.252 s |
| host-side download of the device arena: one memcpy, or 16 MB chunks across the host pool | MOJOLEARN_XPREP_DOWNLOAD=memcpy / threads | the device to host copy |
| x_prep_run_nocopy: the device arena is the host mapping itself (a non-owning DeviceBuffer over host pages; unknown whether Metal accepts it) | MOJOLEARN_XPREP_NOCOPY=1 | the same, and the upload |
| `label_buffers`: numeric labels as buffers in LabelEncoder, LabelBinarizer, MultiLabelBinarizer | MOJOLEARN_XPREP_R3_ON=label_buffers | Python 0.17 to 0.45 s |
| `work2`: PowerTransformer's logarithms and transforms and the encoders' codes as device scratch | MOJOLEARN_XPREP_R3_ON=work2 | PowerTransformer download 0.120 s |
| bench/x_prep_speed.py --touch (reads every page of every output inside the timed region, for the no-copy arm) | - | - |

### Shared code touched

None outside the prep family. Family files: python/mojolearn/_expansion_prep.py, x_prep/device.mojo,
x_prep/dradix.mojo (NEW: tools/lane_select.py may need it attributed, as x_prep/folds.mojo),
bench/x_prep_ab.sh, tools/prep_apple3/ (job2.sh, table.py), this file. core/arena_io.mojo and
python/mojolearn/_arena_io.py (lane py-shared) are USED as they were (the in/out spans are added to the
output ranges the runner already takes) and were not edited.
