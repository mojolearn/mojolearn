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
| TargetEncoder buckets by chunks in parallel (te_hist / te_hsum / te_hstart / te_hscatter, ops 115-118; model matched te_bucket on 300 cases) | both | OPT-IN MOJOLEARN_XPREP_TE_PBUCKET=1 | never measured (request 1790619856364 withdrawn unstarted at the freeze) | f7977ae25, e19ccd8ca |
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

## FINAL (freeze, 2026-09-28; tip pushed, NOT merged)

The lane stops here (orchestrator FREEZE for lane/apple2-merged at 19:30Z). Working tree clean; every
default-on change was measured in the final record below with IDENTICAL and FAST digests equal to the
base's; the one unmeasured change (TargetEncoder parallel buckets) is opt-in and off.

### Final record: m4pro-a (Apple M4 Pro), one job, base 8bacfde74 vs e1d644de0 (request 1790618976686)

The default path at the tip (e19ccd8ca) is e1d644de0's: the commits after it add only the opt-in parallel
buckets and this file. Seconds (minimum of 2 reps after a warm run); digests = the bench's sha256 of every
output, before vs after, both modes.

| dataset | case | rows | IDENTICAL before s | IDENTICAL after s | x | FAST before s | FAST after s | x | digests before == after |
|---|---|---|---|---|---|---|---|---|---|
| taxi | robust-scaler | 1000000 | 0.532 | 0.450 | 1.18 | 0.457 | 0.364 | 1.25 | same |
| taxi | maxabs-scaler | 1000000 | 0.194 | 0.196 | 0.99 | 0.102 | 0.116 | 0.89 | same |
| taxi | quantile-transformer | 1000000 | 0.574 | 0.485 | 1.18 | 0.481 | 0.408 | 1.18 | same |
| taxi | power-transformer | 1000000 | 4.564 | 3.184 | 1.43 | 0.580 | 0.578 | 1.00 | same |
| taxi | normalizer | 1000000 | 0.065 | 0.064 | 1.00 | 0.065 | 0.064 | 1.01 | same |
| taxi | binarizer | 1000000 | 0.065 | 0.065 | 1.01 | 0.065 | 0.065 | 1.00 | same |
| taxi | poly-features | 1000000 | 0.395 | 0.391 | 1.01 | 0.390 | 0.391 | 1.00 | same |
| taxi | spline | 1000000 | 0.524 | 0.533 | 0.98 | 0.344 | 0.344 | 1.00 | same |
| taxi | kbins | 1000000 | 0.533 | 0.455 | 1.17 | 0.446 | 0.365 | 1.22 | same |
| taxi | onehot | 1000000 | 1.660 | 1.471 | 1.13 | 1.659 | 1.534 | 1.08 | same |
| taxi | ordinal | 1000000 | 0.372 | 0.330 | 1.13 | 0.371 | 0.328 | 1.13 | same |
| taxi | variance-threshold | 1000000 | 0.190 | 0.193 | 0.98 | 0.100 | 0.099 | 1.01 | same |
| taxi | target-encoder | 1000000 | 2.357 | 1.300 | 1.81 | 2.241 | 1.212 | 1.85 | same |
| taxi | simple-imputer | 1000000 | 0.532 | 0.450 | 1.18 | 0.444 | 0.364 | 1.22 | same |
| taxi | simple-imputer-mean | 1000000 | 0.531 | 0.452 | 1.17 | 0.444 | 0.370 | 1.20 | same |
| taxi | iterative-imputer | 100000 | 2.566 | 2.541 | 1.01 | 0.829 | 0.826 | 1.00 | same |
| taxi | label-encoder | 1000000 | 0.919 | 0.255 | 3.61 | 0.930 | 0.256 | 3.63 | same |
| taxi | label-binarizer | 1000000 | 1.703 | 1.080 | 1.58 | 1.660 | 1.076 | 1.54 | same |
| taxi | multilabel-binarizer | 200000 | 2.290 | 0.884 | 2.59 | 2.293 | 0.885 | 2.59 | same |
| taxi | select-f-classif | 1000000 | 0.333 | 0.350 | 0.95 | 0.227 | 0.227 | 1.00 | same |
| taxi | select-chi2 | 1000000 | 0.310 | 0.308 | 1.01 | 0.147 | 0.150 | 0.98 | same |
| taxi | select-f-regression | 1000000 | 0.229 | 0.209 | 1.09 | 0.155 | 0.155 | 1.00 | same |
| taxi | mutual-info-classif | 100000 | 8.853 | 0.267 | 33.22 | 7.902 | 0.253 | 31.28 | same |
| taxi | gaussian-nb | 1000000 | 0.451 | 0.449 | 1.00 | 0.223 | 0.220 | 1.01 | same |
| taxi | bernoulli-nb | 1000000 | 0.467 | 0.470 | 0.99 | 0.324 | 0.312 | 1.04 | same |
| taxi | categorical-nb | 1000000 | 0.968 | 0.774 | 1.25 | 0.507 | 0.389 | 1.30 | same |
| taxi | multinomial-nb | 1000000 | 0.379 | 0.375 | 1.01 | 0.239 | 0.214 | 1.12 | same |
| taxi | complement-nb | 1000000 | 0.377 | 0.384 | 0.98 | 0.217 | 0.214 | 1.02 | same |
| taxi | lda | 1000000 | 0.500 | 0.501 | 1.00 | 0.332 | 0.332 | 1.00 | same |
| taxi | qda | 1000000 | 0.442 | 0.444 | 1.00 | 0.377 | 0.376 | 1.00 | same |
| higgs | robust-scaler | 1000000 | 0.532 | 0.450 | 1.18 | 0.444 | 0.365 | 1.22 | same |
| higgs | maxabs-scaler | 1000000 | 0.206 | 0.196 | 1.05 | 0.103 | 0.104 | 0.99 | same |
| higgs | quantile-transformer | 1000000 | 0.571 | 0.487 | 1.17 | 0.480 | 0.400 | 1.20 | same |
| higgs | power-transformer | 1000000 | 4.821 | 3.450 | 1.40 | 0.582 | 0.588 | 0.99 | same |
| higgs | normalizer | 1000000 | 0.065 | 0.065 | 0.99 | 0.064 | 0.064 | 1.00 | same |
| higgs | binarizer | 1000000 | 0.065 | 0.066 | 1.00 | 0.065 | 0.065 | 1.00 | same |
| higgs | poly-features | 1000000 | 0.420 | 0.420 | 1.00 | 0.420 | 0.416 | 1.01 | same |
| higgs | spline | 1000000 | 0.534 | 0.545 | 0.98 | 0.365 | 0.356 | 1.02 | same |
| higgs | kbins | 1000000 | 0.550 | 0.453 | 1.21 | 0.446 | 0.369 | 1.21 | same |
| higgs | onehot | 1000000 | 0.541 | 0.491 | 1.10 | 0.538 | 0.489 | 1.10 | same |
| higgs | ordinal | 1000000 | 0.368 | 0.324 | 1.14 | 0.366 | 0.323 | 1.13 | same |
| higgs | variance-threshold | 1000000 | 0.194 | 0.193 | 1.00 | 0.103 | 0.104 | 1.00 | same |
| higgs | target-encoder | 1000000 | 2.246 | 1.278 | 1.76 | 2.191 | 1.190 | 1.84 | same |
| higgs | simple-imputer | 1000000 | 0.533 | 0.450 | 1.19 | 0.443 | 0.365 | 1.21 | same |
| higgs | simple-imputer-mean | 1000000 | 0.536 | 0.452 | 1.19 | 0.443 | 0.364 | 1.22 | same |
| higgs | iterative-imputer | 100000 | 1.616 | 1.595 | 1.01 | 0.549 | 0.540 | 1.02 | same |
| higgs | label-encoder | 1000000 | 0.972 | 0.262 | 3.71 | 0.945 | 0.263 | 3.60 | same |
| higgs | label-binarizer | 1000000 | 0.975 | 0.302 | 3.22 | 0.997 | 0.304 | 3.28 | same |
| higgs | multilabel-binarizer | 200000 | 2.027 | 0.611 | 3.32 | 2.033 | 0.620 | 3.28 | same |
| higgs | select-f-classif | 1000000 | 0.341 | 0.343 | 0.99 | 0.223 | 0.226 | 0.99 | same |
| higgs | select-chi2 | 1000000 | 0.303 | 0.309 | 0.98 | 0.148 | 0.166 | 0.89 | same |
| higgs | select-f-regression | 1000000 | 0.227 | 0.208 | 1.09 | 0.153 | 0.161 | 0.95 | same |
| higgs | mutual-info-classif | 100000 | 3.385 | 0.284 | 11.90 | 2.920 | 0.269 | 10.84 | same |
| higgs | gaussian-nb | 1000000 | 0.447 | 0.451 | 0.99 | 0.215 | 0.215 | 1.00 | same |
| higgs | bernoulli-nb | 1000000 | 0.463 | 0.461 | 1.00 | 0.320 | 0.305 | 1.05 | same |
| higgs | categorical-nb | 1000000 | 0.884 | 0.732 | 1.21 | 0.512 | 0.334 | 1.53 | same |
| higgs | multinomial-nb | 1000000 | 0.369 | 0.370 | 1.00 | 0.213 | 0.209 | 1.02 | same |
| higgs | complement-nb | 1000000 | 0.372 | 0.373 | 1.00 | 0.210 | 0.211 | 0.99 | same |
| higgs | lda | 1000000 | 0.508 | 0.500 | 1.02 | 0.326 | 0.325 | 1.00 | same |
| higgs | qda | 1000000 | 0.434 | 0.434 | 1.00 | 0.376 | 0.374 | 1.01 | same |

Same-job records on the other Macs agree (all digests equal): m4-a request 1790618180667 (2c713f92f; taxi
mutual-info 16.57 -> 0.31 s, target-encoder 2.50 -> 1.44 s, power-transformer 4.44 -> 3.61 s) and m4pro-b
request 1790616582261 (106471253).

### Unproven commits (the integration run must cover them)

Proof on this lane is the speed bench's output digests on Apple only (M4 Pro and M4), before == after,
IDENTICAL and FAST. No identity steward request (Metal vs CPU vs NVIDIA vs AMD), no sabotage arm, no
lane check, no m2pro run. Every code commit below is therefore unproven cross-vendor:
- c3fd92296 PT speculation (ops 110-113), device scratch, te_gather (op 114), label fast paths, bench/x_prep_ab.sh
- 7fb04e8ef, c3da4e161, 667a39aa0, 0daee8cb5 output region (x_prep_run_out) and its size rule
- 091d3c9e2 native TargetEncoder folds (x_prep/folds.mojo; new binding entries)
- cb59cb961 PT speculation IDENTICAL only
- 59c118637 acc_add in ii_mean / ii_gram / f_regression / qda_cov
- 7b5269213 pt_spts spelling
- 422896220 dsort quad passes
- 74ec32727 mi_cd class split
- f9c63df20 cat_counts RUN loads
- d24901c00 te_bucket private counters
- 1bd38c6e6 mutual_info tied-run search (the device path now orders by (x, noise); the host path
  x_prep/host/mutual_info.mojo is unchanged, so Metal vs CPU equality rests on the argument and the
  Python model, not on a run)
- c8fbb75e2, e1d644de0 PT candidate layout (interleave opt-in)
- 106471253 ftz_chain reverted; 5d833d0c9, fb35ed489 SIMD folds reverted (the tree equals the earlier
  spelling there)
- 2c713f92f PT depth 3 default
- f7977ae25, e19ccd8ca parallel TE buckets, opt-in, never executed

### Known risks for the integration run
- m2pro (never used here): dmi `_tile_kernel` now moves a row payload beside each sort word (two 8 KB
  shared arrays, 512 threads per group); the M2 drops dispatches above its pipeline's max threads
  silently (db5d6fb01), so check that kernel's limit there. dsort's quad pass uses 256-thread blocks.
- The CPU host binding never ran on this lane: its fallbacks (scratch and output words in the host
  arena, Python fold assignment, te_bucket-free te_enc, pt_fit) are unexercised since round 1.
- New ops 110-118 (N_OPS 119) have no seam or sabotage arm of their own; the existing patches still apply.
- The selector (tools/lane_select.py) may need bench/x_prep_ab.sh and x_prep/folds.mojo attributed.
