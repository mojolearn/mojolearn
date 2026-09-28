# prep-apple2: progress

Apple speed round 2 for the prep family (~/mojolearn-evidence/apple2_speed_brief.md): preprocessing,
feature selection, encoders, imputers, naive Bayes, LDA/QDA. Worktree `~/mojolearn-wt/prep-apple2`, branch
`lane/prep-apple2` (forked from lane/apple-merged 037daa353; NOT merged). Evidence:
~/mojolearn-evidence/prep-apple2/. Round 1: docs/lanes/progress/prep-apple.md.

A/B tool: `sh bench/x_prep_ab.sh <base commit> <mode> <reps> <cases|all> <arm> ...` runs a base commit
(checked out beside the tree, its own build) and the tree's arms (build defines, env switches) in ONE job
on one Mac; every XPSPEED line carries the output digest.

## categorical-nb on HIGGS ("Negative values in data")

A bench input issue, not a kernel one: HIGGS's `cat` block has negative codes (np.round(x * 4) and
floor(clip(x) * 3) of standardized columns), and CategoricalNB refuses negative input exactly as
scikit-learn does (`check_non_negative(X, "CategoricalNB (input X)")`). bench/x_prep_speed.py now gives
CategoricalNB a `catnn` block (each column shifted by min(col min, 0); taxi's codes are already
nonnegative and keep their bytes). HIGGS categorical-nb now runs: 0.873 s on m4pro-b (8bacfde74), digest
3a2b356296226499. Commit 8bacfde74.

## Base profile (IDENTICAL, m4pro-b, 8bacfde74, request 1790609251171; stage 0 of a program includes its
arena round trip)

| case | total s (taxi) | hot stage |
|---|---|---|
| power-transformer | 4.53 | the 50 serial folds of the golden search (skipped by the profiler: 106 stages) |
| iterative-imputer (100k) | 2.48 | 991 stages; ii_mean / ii_gram serial folds (FAST's tree folds take it to 0.53) |
| target-encoder | 2.31 | te_enc 0.81, te_bucket 0.28, te_global 0.15, Python 0.60 (fold assignment 0.39) |
| multilabel-binarizer (200k) | 2.29 | Python 1.83 (label lists) |
| onehot | 1.68 | the arena round trip of the 1M x 566 dense output (program 1 stage 0: 1.27) |
| label-binarizer | 1.68 | Python 1.00, the 1M x 243 output 0.53 |
| label-encoder | 0.93 | Python 0.81 (the per-label float32 test) |
| mutual-info-classif (100k) | 8.25 | mi_cd (HIGGS 3.11 of 3.21) |
| robust / quantile / kbins / simple-imputer | ~0.52 | sort_cols 0.35 (16 columns), col_stats 0.10 |

## Changes (lane prep-apple2)

| change | mode | default | commit |
|---|---|---|---|
| PowerTransformer golden search speculated S steps deep (pt_spts / pt_smap / pt_sfold / pt_sres, ops 110-113): the 2^S - 1 candidate points of the next S steps are evaluated side by side, then pt_finish's steps are taken with their values; the last evaluation (never used) is not made. Same points, values, decisions, lambdas by construction; a Python model matched the serial search on 15000 cases | IDENTICAL | MOJOLEARN_XPREP_PT_SPEC=4 (0 = staged search) | c3fd92296, 7b5269213, cb59cb961 |
| device-only scratch words (_Prog.scratch, x_prep_run_scratch) | both | on | c3fd92296 |
| te_gather (op 114): each bucket's folds and targets in bucket order, te_enc streams them RUN at a time; te_global / te_enc accumulators acc_add | both (same order) | MOJOLEARN_XPREP_TE_GATHER=1 | c3fd92296 |
| ftz_chain: acc_add's flush on the device as compare + select (the same word as ftz for every input) | IDENTICAL | on; -D MOJOLEARN_XPREP_FTZ_INT=1 reverts | c3fd92296 |
| LabelEncoder / LabelBinarizer / MultiLabelBinarizer: numeric-label test and label column at C speed (same answer; the loop remains for what the fast test cannot settle) | both | on | c3fd92296 |
| one OUTPUT region per program: zeroed on the device, never uploaded, read back into its own host array (x_prep_run_out); encoders, transforms, label binarizers, selectors | both | MOJOLEARN_XPREP_OUT=1 | 7fb04e8ef |
| TargetEncoder fold assignment in Mojo on the host (x_prep/folds.mojo, integer arithmetic; a transliteration matched the Python on 300 cases) | both | MOJOLEARN_XPREP_NATIVE_FOLDS=1 | 091d3c9e2 |
| acc_add in ii_mean, ii_gram, f_regression, qda_cov row folds | IDENTICAL (same word) | on | 59c118637 |
| dsort: two bitonic strides per global pass (quad), 55 -> 30 passes at 1M rows | both (sort has one answer) | MOJOLEARN_XPREP_SORT_QUAD=1 | 422896220 |

Build-time trap found and fixed: pt_spts written as a per-node path walk (a while loop inside the node loop
around the golden step's `mut` helper) made the Mojo build of the binding take more than 40 minutes (15 s
before); the parent-bracket spelling (7b5269213) builds in 15 s. Request 1790609251171 (c3fd92296) was
cancelled in that build; its base arm is kept (~/mojolearn-evidence/prep-apple2/job2_base_1790609251171.txt).

## Results

(pending: request 1790614752790 on m4-a, base 8bacfde74 vs 7b5269213, IDENTICAL and FAST)

## Shared code touched (the integration run must cover)

x_prep/prims.mojo `acc_add` (every x_prep device fold that uses it), x_prep/device.mojo (scratch and output
regions, every x_prep program), bindings/_mojolearn_x_prep.mojo (new entries x_prep_run_scratch,
x_prep_run_out, x_prep_strat_folds, x_prep_kfold_folds), naive_bayes/da.mojo qda_cov. Nothing outside the
prep family's files.
