# prep-apple2: progress

Apple speed round 2 for the prep family (~/mojolearn-evidence/apple2_speed_brief.md): preprocessing,
feature selection, encoders, imputers, naive Bayes, LDA/QDA. Worktree `~/mojolearn-wt/prep-apple2`, branch
`lane/prep-apple2` (forked from lane/apple-merged 037daa353, lane/apple-merged merged again at 759f2019a;
NOT merged anywhere). Evidence: ~/mojolearn-evidence/prep-apple2/ (every job's full stdout, jobN_*.txt).
Round 1: docs/lanes/progress/prep-apple.md.

A/B tool: `sh bench/x_prep_ab.sh <base commit> <mode> <reps> <cases|all> <arm> ...` runs a base commit
(checked out beside the tree, its own build) and the tree's arms (build defines, env switches) in ONE job
on one Mac; every XPSPEED line carries the output digest. Base for every job: 8bacfde74 (037daa353 plus
the bench fix below). 1M rows (100k for iterative-imputer and mutual-info-classif, 200k for
multilabel-binarizer), reps 2, minimum reported, taxi + HIGGS from R2.

## categorical-nb on HIGGS ("Negative values in data")

A bench input issue, not a kernel one: HIGGS's `cat` block has negative codes (np.round(x * 4) and
floor(clip(x) * 3) of standardized columns), and CategoricalNB refuses negative input exactly as
scikit-learn does (`check_non_negative(X, "CategoricalNB (input X)")`). bench/x_prep_speed.py now gives
CategoricalNB a `catnn` block (each column shifted by min(col min, 0); taxi's codes are already
nonnegative and keep their bytes, so taxi's digest is unchanged). HIGGS categorical-nb now runs
(IDENTICAL digest 3a2b356296226499 on every Mac and commit). Commit 8bacfde74.

## Base profile (IDENTICAL, m4pro-b, 8bacfde74, request 1790609251171)

| case (taxi) | total s | where |
|---|---|---|
| mutual-info-classif (100k) | 8.25 | mi_cd: the point walk visits every point of a tied run (taxi columns tie in runs of 10^4 to 10^5) |
| power-transformer | 4.53 | 50 dependent serial folds of the golden search |
| iterative-imputer (100k) | 2.48 | 991 stages; ii_mean / ii_gram serial folds |
| target-encoder | 2.31 | te_enc 0.81, te_bucket 0.28, te_global 0.15, Python 0.60 (fold assignment 0.39) |
| multilabel-binarizer (200k) | 2.29 | Python 1.83 (label lists) |
| onehot | 1.68 | the arena round trip of the 1M x 566 dense output (1.27) |
| label-binarizer / label-encoder | 1.68 / 0.93 | Python (the per-label float32 test, nested lists) |
| robust / quantile / kbins / simple-imputer | ~0.52 | sort_cols 0.35 (16 columns) |

## Changes kept (default on unless noted)

| change | mode | switch | measured (same job, same Mac, digests equal) | commit |
|---|---|---|---|---|
| mutual_info (continuous feature): class arrays in (key(x), key(noise)) order (the sort carries the row index as a payload); each tied run is searched (first kl candidates in secondary-distance order, the walk's run order and stop rule) and `_count_within`'s boundary runs are counted by binary search. Same kl-th pair and count (Python model of both: 4631 queries, -0/+0, rounding-merged distances, tied and zero noise) | both | MOJOLEARN_XPREP_MI_TIES=0 | m4pro-b taxi 7.99 -> 0.258 s, HIGGS 2.98 -> 0.276 s (request 1790616582261) | 1bd38c6e6 |
| mi_cd class split: MRUN rows loaded at once, label counters in registers, one listing pass | both | - | part of base 8.20 -> ties0 7.99 s | 74ec32727 |
| PowerTransformer golden search speculated 3 steps deep (pt_spts / pt_smap / pt_sfold / pt_sres, ops 110-113); the last evaluation (never used) is not made. Same points, values, decisions, lambdas (Python model: 15000 searches) | IDENTICAL | MOJOLEARN_XPREP_PT_SPEC (default 3; 0 = staged) | m4pro-a taxi: staged 4.54, depth 2 3.37, depth 3 3.18, depth 4 3.14 s (request 1790618976686); m4pro-b depth 3 2.95 vs 4 3.09 | c3fd92296, 7b5269213, cb59cb961, 2c713f92f, e1d644de0 |
| TargetEncoder te_gather (op 114): each bucket's folds and targets in bucket order, te_enc streams them | both | MOJOLEARN_XPREP_TE_GATHER=0 | m4-a taxi 2.19 -> 1.53 s (request 1790614752790) | c3fd92296 |
| TargetEncoder fold assignment in Mojo on the host (x_prep/folds.mojo, integer arithmetic; transliteration matched the Python on 300 cases) | both | MOJOLEARN_XPREP_NATIVE_FOLDS=0 | m4-a taxi 1.89 -> 1.53 s | 091d3c9e2 |
| TargetEncoder buckets by chunks in parallel (te_hist / te_hsum / te_hstart / te_hscatter, ops 115-118; model matched te_bucket on 300 cases) | both | MOJOLEARN_XPREP_TE_PBUCKET=0 | see FINAL (request 1790619856364) | f7977ae25 |
| te_bucket counters in a thread-private array (the fallback path) | both | - | no measurable change (te_bucket 0.26 s before and after) | d24901c00 |
| LabelEncoder / LabelBinarizer / MultiLabelBinarizer: the numeric-label test and the label column at C speed (same answer; the per-label loop remains for what the fast test cannot settle); put_list without the nested-list walk | both | - | taxi label-encoder 0.95 -> 0.25, multilabel 2.32 -> 0.83 s (request 1790616582261) | c3fd92296, 7fb04e8ef |
| dsort: two bitonic strides per global pass (55 -> 30 passes at 1M rows) | both | MOJOLEARN_XPREP_SORT_QUAD=0 | m4pro-b robust-scaler 0.525 -> 0.455 s, all sorting cases 1.1-1.2x | 422896220 |
| cat_counts: unweighted counts load RUN rows at once | both | - | m4pro-b categorical-nb 0.96 -> 0.73 s (with the other changes) | f9c63df20 |
| one OUTPUT region per program (zeroed on the device, read back into its own host array, x_prep_run_out) for outputs of 2^27 words or more; device-only scratch words (x_prep_run_scratch) | both | MOJOLEARN_XPREP_OUT=0 | m4-a taxi onehot 1.84 -> 1.68 s; smaller outputs lost time (HIGGS onehot 0.52 -> 0.58 s, poly-features 0.45 -> 0.52 s) so they stay arena words | 7fb04e8ef, c3da4e161, 667a39aa0, 0daee8cb5 |
| acc_add in the row folds of ii_mean, ii_gram, f_regression, qda_cov | IDENTICAL (same word) | - | select-f-regression 0.231 -> 0.209 s | 59c118637 |

## Measured and reverted (or left opt-in)

- ftz_chain (acc_add's flush as compare + select): no case moved (m4-a, -D MOJOLEARN_XPREP_FTZ_INT=1 arm,
  request 1790614752790). Reverted (106471253).
- SIMD-group-fed serial folds (x_prep/simdfold.mojo: lanes load, every lane folds all rows from
  shuffle_idx broadcasts) for pt_sfold, ii_mean, ii_gram, col_stats, class_stats, te_global: digests equal
  but SLOWER on m4pro-b (request 1790617589677: iterative-imputer 2.48 -> 3.75 s, maxabs 0.197 -> 0.231 s,
  PowerTransformer 3.09 -> 4.43 s). Reverted (5d833d0c9, fb35ed489).
- PowerTransformer candidates interleaved (T[(c*n + i)*M + j]): m4pro-a depth 3 3.18 -> 3.01 s (same job),
  but m4-a 3.01 s contiguous (request 1790614752790) vs 3.61 s interleaved (request 1790618180667, another
  job). Left opt-in: MOJOLEARN_XPREP_PT_INTERLEAVE=1 (e1d644de0).
- The output region for PolynomialFeatures, Binarizer and the selectors' gather (slower): arena words.

Build-time trap found and fixed: pt_spts written as a per-node path walk (a while loop inside the node loop
around the golden step's `mut` helper) made the Mojo build of the binding take more than 40 minutes (15 s
before, bisected locally at one core); the parent-bracket spelling (7b5269213) builds in 15 s. Request
1790609251171 was cancelled in that build (its base arm is kept, job2_base_1790609251171.txt).

## Results

FINAL record: see the FINAL section. Intermediate same-job records: request 1790616582261 (m4pro-b,
106471253), 1790618180667 (m4-a, 2c713f92f), 1790618976686 (m4pro-a, e1d644de0). Every IDENTICAL digest
equal to the base's in every job (and FAST digests equal too: no FAST change moves a bit, so no quality
check was owed).

## Shared code touched (the integration run must cover)

x_prep/device.mojo (scratch and output regions, every x_prep program; quad sort; the mi_cd plumbing),
x_prep/dsort.mojo (every sort_cols), x_prep/dmi.mojo (mi_cd), bindings/_mojolearn_x_prep.mojo (new entries
x_prep_run_scratch, x_prep_run_out, x_prep_strat_folds, x_prep_kfold_folds), x_prep/stats.mojo /
iterative.mojo / naive_bayes/da.mojo / nb.mojo (acc_add, cat_counts), x_prep/target.mojo, transform.mojo,
units.mojo (ops 110-118, N_OPS 119). Nothing outside the prep family's files.
