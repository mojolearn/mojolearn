# prep-apple: progress

Apple (Metal) speed lane for the prep family (~/mojolearn-evidence/apple_speed_brief.md):
preprocessing, feature selection, imputers, encoders, naive Bayes, LDA/QDA.
Worktree `~/mojolearn-wt/prep-apple`, branch `lane/prep-apple` (NOT merged; the gate runners merge).
Home Mac for speed jobs: m4pro-b (Apple M4 Pro). Evidence: ~/mojolearn-evidence/prep-apple/.

Timing board: `bench/x_prep_speed.py` (the board's prep shapes on taxi + HIGGS from R2;
`XPSPEED <ds> <case> <rows> <total_s> <binding_s> <programs> <digest>`; `--profile` names the
hot stage of each case by prefix timing).

## Profile (IDENTICAL, m4pro-b, 100k rows, taxi; request 1790580105499)
Where Apple's time went, before this lane:
| case | total s | hot stage (s) |
|---|---|---|
| RobustScaler / QuantileTransformer / KBins / SimpleImputer | ~0.49 | sort_cols 0.41 (one heapsort thread per column) |
| OneHot / Ordinal | 0.52 / 0.39 | sort_cols 0.36, lookup 0.12 |
| PowerTransformer | 13.98 | pt_fit 13.83 (50 llf evaluations, exp+log1p per row, one thread per column) |
| TargetEncoder | 2.97 | te_enc 2.02 (every (fold, column, category) thread scans every row) |
| IterativeImputer | 21.81 | ii_mean + ii_gram ~0.115 per feature step, ii_conv 0.22 (one thread) |
| MaxAbs / VarianceThreshold | ~0.075 | col_stats 0.07 (~350 ns per row: one memory latency per row) |

The pattern: a unit that folds a column in row order on ONE GPU thread pays a full memory
latency per row on Metal. Cure without moving a bit: load RUN (16) rows before folding them
one by one (`run_block`), move row-independent work (transforms, row sums) to one thread per
element, and replace the single-thread heapsort by a bitonic sort under a total word order.

## IDENTICAL changes (same bits by construction; proof by steward identity request)
| change | commit |
|---|---|
| device bitonic sort for sort_cols (x_prep/dsort.mojo) under `word_order` (key, then bits), host heapsort on the same order | 95bb12971 |
| col_stats batched loads (`run_block`) | 95bb12971 |
| PowerTransformer golden search as stages: pt_init, pt_map (per element), pt_fold (column fold + step) | 0f8408c9b |
| IterativeImputer: ii_mean / ii_gram batched, ii_rowabs (row sums per thread) + ii_conv max fold | 0f8408c9b |
| TargetEncoder: te_bucket (each category's rows, ascending), te_enc walks its bucket; te_global batched | eab4cfc00 |
| class_stats, count_neg, matmul (mm_step), qda_cov batched | be03c403b, 211572cfd |
| seam arms 5400/5401/5402 regenerated for the new spellings (~/mojolearn-evidence/prep-apple/mkpatches.py) | 95bb12971, 211572cfd |

## Results (before -> after, per algorithm, per mode, per Mac)

### IDENTICAL, 1M rows (100k for IterativeImputer and mutual_info), seconds, fit + transform/predict
Before: m4pro-b at 0cbfb3151 (main + the bench; request 1790582828756). After: m3ultra at bd8057aa7
(request 1790584412836). Digest = the bench's sha256 of every output: the SAME on all 60 cases, M4 Pro
before vs M3 Ultra after (IDENTICAL bits across Macs and across the change). The same-Mac (m4pro-b)
record at the final head is below when it lands.

| dataset | algorithm | rows | before s (m4pro-b, 0cbfb3151) | after s (m3ultra, bd8057aa7) | digest |
|---|---|---|---|---|---|
| taxi | robust-scaler | 1000000 | 6.234 | 0.393 | same |
| taxi | maxabs-scaler | 1000000 | 0.780 | 0.244 | same |
| taxi | quantile-transformer | 1000000 | 6.249 | 0.422 | same |
| taxi | power-transformer | 1000000 | 139.716 | 10.158 | same |
| taxi | normalizer | 1000000 | 0.067 | 0.072 | same |
| taxi | binarizer | 1000000 | 0.067 | 0.079 | same |
| taxi | poly-features | 1000000 | 0.431 | 0.449 | same |
| taxi | spline | 1000000 | 1.732 | 0.715 | same |
| taxi | kbins | 1000000 | 6.221 | 0.400 | same |
| taxi | onehot | 1000000 | 6.239 | 1.863 | same |
| taxi | ordinal | 1000000 | 4.883 | 0.401 | same |
| taxi | variance-threshold | 1000000 | 0.773 | 0.231 | same |
| taxi | target-encoder | 1000000 | 25.786 | 2.959 | same |
| taxi | simple-imputer | 1000000 | 6.351 | 0.404 | same |
| taxi | simple-imputer-mean | 1000000 | 6.359 | 0.400 | same |
| taxi | iterative-imputer | 100000 | 21.823 | 3.102 | same |
| taxi | label-encoder | 1000000 | 4.733 | 1.143 | same |
| taxi | label-binarizer | 1000000 | 5.503 | 1.918 | same |
| taxi | multilabel-binarizer | 200000 | 9.239 | 2.717 | same |
| taxi | select-f-classif | 1000000 | 1.044 | 0.734 | same |
| taxi | select-chi2 | 1000000 | 1.289 | 0.395 | same |
| taxi | select-f-regression | 1000000 | 0.745 | 0.896 | same |
| taxi | mutual-info-classif | 100000 | 14.061 | 4.937 | same |
| taxi | gaussian-nb | 1000000 | 1.847 | 0.574 | same |
| taxi | bernoulli-nb | 1000000 | 1.460 | 0.572 | same |
| taxi | categorical-nb | 1000000 | 2.635 | 0.999 | same |
| taxi | multinomial-nb | 1000000 | 1.361 | 0.487 | same |
| taxi | complement-nb | 1000000 | 1.359 | 0.450 | same |
| taxi | lda | 1000000 | 1.793 | 0.615 | same |
| taxi | qda | 1000000 | 1.145 | 0.480 | same |
| higgs | robust-scaler | 1000000 | 7.614 | 0.393 | same |
| higgs | maxabs-scaler | 1000000 | 0.778 | 0.238 | same |
| higgs | quantile-transformer | 1000000 | 7.663 | 0.409 | same |
| higgs | power-transformer | 1000000 | 190.181 | 10.487 | same |
| higgs | normalizer | 1000000 | 0.065 | 0.072 | same |
| higgs | binarizer | 1000000 | 0.067 | 0.071 | same |
| higgs | poly-features | 1000000 | 0.429 | 0.454 | same |
| higgs | spline | 1000000 | 1.714 | 0.671 | same |
| higgs | kbins | 1000000 | 7.617 | 0.398 | same |
| higgs | onehot | 1000000 | 4.891 | 0.610 | same |
| higgs | ordinal | 1000000 | 4.719 | 0.398 | same |
| higgs | variance-threshold | 1000000 | 0.784 | 0.247 | same |
| higgs | target-encoder | 1000000 | 12.022 | 2.970 | same |
| higgs | simple-imputer | 1000000 | 7.579 | 0.400 | same |
| higgs | simple-imputer-mean | 1000000 | 7.610 | 0.414 | same |
| higgs | iterative-imputer | 100000 | 13.478 | 1.935 | same |
| higgs | label-encoder | 1000000 | 4.498 | 1.124 | same |
| higgs | label-binarizer | 1000000 | 4.542 | 1.172 | same |
| higgs | multilabel-binarizer | 200000 | 9.027 | 2.392 | same |
| higgs | select-f-classif | 1000000 | 0.910 | 0.747 | same |
| higgs | select-chi2 | 1000000 | 1.162 | 0.388 | same |
| higgs | select-f-regression | 1000000 | 0.740 | 0.972 | same |
| higgs | mutual-info-classif | 100000 | 13.736 | 4.768 | same |
| higgs | gaussian-nb | 1000000 | 1.585 | 0.546 | same |
| higgs | bernoulli-nb | 1000000 | 1.333 | 0.574 | same |
| higgs | multinomial-nb | 1000000 | 1.233 | 0.465 | same |
| higgs | complement-nb | 1000000 | 1.232 | 0.454 | same |
| higgs | lda | 1000000 | 1.655 | 0.621 | same |
| higgs | qda | 1000000 | 0.917 | 0.491 | same |

### FAST folds (x_prep/fastred.mojo), m3ultra, 1M rows, FAST build at 54eaead19, one job (A/B by MOJOLEARN_XPREP_FAST_FOLDS)
| case | taxi folds0 -> folds1 s | higgs folds0 -> folds1 s |
|---|---|---|
| power-transformer | 4.242 -> 0.490 | 4.647 -> 0.502 |
| maxabs-scaler | 0.201 -> 0.105 | 0.208 -> 0.107 |
| variance-threshold | 0.197 -> 0.098 | 0.210 -> 0.108 |
| robust-scaler | 0.358 -> 0.257 | 0.371 -> 0.268 |
| quantile-transformer | 0.377 -> 0.277 | 0.378 -> 0.278 |
| kbins | 0.360 -> 0.263 | 0.368 -> 0.270 |
| spline | 0.572 -> 0.390 | 0.573 -> 0.394 |
| simple-imputer-mean | 0.359 -> 0.255 | 0.362 -> 0.262 |
| gaussian-nb | 0.463 -> 0.364 | 0.457 -> 0.374 |
| multinomial-nb | 0.393 -> 0.301 | 0.404 -> 0.296 |
| lda | 0.552 -> 0.460 | 0.573 -> 0.452 |
| select-chi2 | 0.322 -> 0.219 | 0.318 -> 0.224 |

Paired quality against scikit-learn 1.7.2 float64 (bench/x_prep_quality.py; 5 seeds = 5 draws of 200k rows,
taxi + HIGGS; mean over seeds; lower error is better). Quality never worse; mostly better (tree sums):
| dataset | case | metric | folds0 (row order) | folds1 (tree) |
|---|---|---|---|---|
| higgs | gaussian-nb | accuracy | 0.561 | 0.561 |
| higgs | gaussian-nb | predict_agree | 1 | 1 |
| higgs | gaussian-nb | proba_err | 0.0002 | 0.0002 |
| higgs | lda | accuracy | 0.561 | 0.561 |
| higgs | lda | predict_agree | 1 | 1 |
| higgs | lda | proba_err | 0.000115 | 0.000104 |
| higgs | maxabs-scaler | transform_err | 2.98e-08 | 2.98e-08 |
| higgs | multinomial-nb | accuracy | 0.555 | 0.555 |
| higgs | multinomial-nb | predict_agree | 1 | 1 |
| higgs | multinomial-nb | proba_err | 1.84e-05 | 1.84e-05 |
| higgs | power-transformer | lambda_err | 0.037 | 0.00375 |
| higgs | power-transformer | transform_err | 0.00463 | 0.000725 |
| higgs | simple-imputer-mean | transform_err | 0.000123 | 3.68e-07 |
| higgs | variance-threshold | variances_err | 0.00118 | 4.99e-06 |
| taxi | gaussian-nb | accuracy | 0.22 | 0.22 |
| taxi | gaussian-nb | predict_agree | 1 | 1 |
| taxi | gaussian-nb | proba_err | 0.0567 | 0.0567 |
| taxi | lda | accuracy | 0.817 | 0.817 |
| taxi | lda | predict_agree | 1 | 1 |
| taxi | lda | proba_err | 0.00468 | 0.00423 |
| taxi | maxabs-scaler | transform_err | 2.97e-08 | 2.97e-08 |
| taxi | multinomial-nb | accuracy | 0.775 | 0.775 |
| taxi | multinomial-nb | predict_agree | 1 | 1 |
| taxi | multinomial-nb | proba_err | 3.45e-05 | 3.45e-05 |
| taxi | power-transformer | lambda_err | 0.943 | 0.943 |
| taxi | power-transformer | transform_err | 0.0139 | 0.00708 |
| taxi | simple-imputer-mean | transform_err | 3.68e-05 | 3.1e-08 |
| taxi | variance-threshold | variances_err | 4.88e-05 | 1.78e-07 |

### FAST round 2: tree folds for class_stats, ii_mean, ii_gram (m3ultra, FAST build at 7073ac4cf, A/B in one job)
| case | taxi folds0 -> folds1 s | higgs folds0 -> folds1 s |
|---|---|---|
| power-transformer | 3.915 -> 0.480 | 4.220 -> 0.494 |
| maxabs-scaler | 0.199 -> 0.105 | 0.210 -> 0.112 |
| iterative-imputer | 2.382 -> 0.534 | 1.485 -> 0.365 |
| gaussian-nb | 0.450 -> 0.229 | 0.467 -> 0.236 |
| bernoulli-nb | 0.511 -> 0.357 | 0.513 -> 0.359 |
| lda | 0.560 -> 0.379 | 0.556 -> 0.384 |
| qda | 0.410 -> 0.336 | 0.414 -> 0.343 |
| select-f-classif | 0.337 -> 0.276 | 0.357 -> 0.280 |
| select-chi2 | 0.321 -> 0.159 | 0.317 -> 0.161 |
| categorical-nb | 0.897 -> 0.540 | - |

Paired quality, round 2 (request 1790592147011 at a90c94403; 5 seeds x 200k rows, mean over seeds; accuracy /
predict_agree higher is better, errors lower). The fitted means / variances are 40-600x closer to the reference
with the tree sums; GaussianNB now agrees with scikit-learn on every row. QuadraticDiscriminantAnalysis did NOT
pass (proba max error 0.0011 -> 0.0014 on HIGGS, 3/5 seeds worse; agreement 0.999768 -> 0.999753): its stage now
keeps the row-order class sums in FAST (d1e2a411a), so QDA's FAST numbers equal folds0. taxi IterativeImputer
folds0 = NaN: the row-order FAST fold produced a NaN imputed value on seed 2 (and 0.27 on seed 4); the tree fold
does not.
| dataset | case | metric | folds0 (row order) | folds1 (tree) |
|---|---|---|---|---|
| higgs | f-classif | scores_err | 0.00024044 | 0.00023096 |
| higgs | gaussian-nb | accuracy | 0.560529 | 0.560524 |
| higgs | gaussian-nb | predict_agree | 0.999929 | 1 |
| higgs | gaussian-nb | proba_err | 0.000200302 | 7.04032e-06 |
| higgs | gaussian-nb | theta_err | 0.000639654 | 1.06866e-06 |
| higgs | gaussian-nb | var_err | 0.000640417 | 3.32544e-06 |
| higgs | iterative-imputer | imputed_err | 2.55895e-05 | 1.3819e-06 |
| higgs | lda | accuracy | 0.560882 | 0.560887 |
| higgs | lda | means_err | 0.000639654 | 1.06866e-06 |
| higgs | lda | predict_agree | 0.999889 | 0.999924 |
| higgs | lda | proba_err | 0.000115016 | 7.05873e-05 |
| higgs | maxabs-scaler | transform_err | 2.9802e-08 | 2.9802e-08 |
| higgs | multinomial-nb | accuracy | 0.554653 | 0.554653 |
| higgs | multinomial-nb | predict_agree | 0.999992 | 0.999992 |
| higgs | multinomial-nb | proba_err | 1.84004e-05 | 1.84004e-05 |
| higgs | power-transformer | lambda_err | 0.0370429 | 0.00374743 |
| higgs | power-transformer | transform_err | 0.00462897 | 0.000725062 |
| higgs | qda | accuracy | 0.595458 | 0.595446 |
| higgs | qda | means_err | 0.000639654 | 1.06866e-06 |
| higgs | qda | predict_agree | 0.99978 | 0.999752 |
| higgs | qda | proba_err | 0.00109649 | 0.00142123 |
| higgs | simple-imputer-mean | transform_err | 0.000123076 | 3.67862e-07 |
| higgs | variance-threshold | variances_err | 0.00118265 | 4.98626e-06 |
| taxi | f-classif | scores_err | 0.000747424 | 0.000724972 |
| taxi | gaussian-nb | accuracy | 0.219562 | 0.219575 |
| taxi | gaussian-nb | predict_agree | 0.999975 | 1 |
| taxi | gaussian-nb | proba_err | 0.0566684 | 0.000185795 |
| taxi | gaussian-nb | theta_err | 4.76206e-05 | 3.68618e-08 |
| taxi | gaussian-nb | var_err | 0.000217361 | 4.60615e-07 |
| taxi | iterative-imputer | imputed_err | nan | 0.000281396 |
| taxi | lda | accuracy | 0.81671 | 0.816707 |
| taxi | lda | means_err | 4.76206e-05 | 3.68618e-08 |
| taxi | lda | predict_agree | 0.999859 | 0.999872 |
| taxi | lda | proba_err | 0.00468055 | 0.00395763 |
| taxi | maxabs-scaler | transform_err | 2.97382e-08 | 2.97382e-08 |
| taxi | multinomial-nb | accuracy | 0.774563 | 0.774563 |
| taxi | multinomial-nb | predict_agree | 0.999999 | 0.999999 |
| taxi | multinomial-nb | proba_err | 3.45148e-05 | 3.45148e-05 |
| taxi | power-transformer | lambda_err | 0.943186 | 0.943055 |
| taxi | power-transformer | transform_err | 0.0138623 | 0.00707759 |
| taxi | qda | accuracy | 0.217215 | 0.217211 |
| taxi | qda | means_err | 4.76206e-05 | 3.68618e-08 |
| taxi | qda | predict_agree | 0.999877 | 0.999879 |
| taxi | qda | proba_err | 0.0629779 | 0.0632438 |
| taxi | simple-imputer-mean | transform_err | 3.68163e-05 | 3.0952e-08 |
| taxi | variance-threshold | variances_err | 4.88425e-05 | 1.78088e-07 |

### mutual_info_classif: sorted neighbour search (x_prep/dmi.mojo), m3ultra, IDENTICAL, 100k rows
A/B by MOJOLEARN_XPREP_MI_SORTED in one job (request 1790588108842), digests SAME: taxi 4.927 -> 3.556 s,
HIGGS 4.779 -> 1.595 s (taxi's columns tie heavily; the walk over tied points is the remaining cost).

### Measured and reverted
- RUN 16 -> 64 (more rows in flight per column thread): 2-4x SLOWER (register spills), reverted (87f517ee9).
- IDENTICAL folds staged through threadgroup memory, thread 0 folding (dstage): SLOWER (PowerTransformer 6.16 ->
  9.25 s); the serial fold is bound by its dependent add chain, not by loads once RUN rows are in flight. Reverted.
- Copying up only the arena's input prefix (dense one-hot output): 1.884 -> 1.844 s, noise. Reverted.

## FINAL (wind-down, 2026-09-28; tip c0a29cf5d, pushed, NOT merged)

The lane stops here. Working tree clean, no half-done change; every speed change below is either
IDENTICAL by construction (same words, default on) or FAST only (numeric mode FAST, a build opt-in).

### What changed
- IDENTICAL default path (Metal device program, x_prep): bitonic sort_cols (dsort.mojo), RUN = 16 batched
  loads in col_stats / f_classif / f_regression / class_stats / count_neg / mm_step / qda_cov / te_global /
  ii_mean / ii_gram, PowerTransformer golden search as stages (pt_init, pt_log once, pt_map per element,
  pt_fold per column), IterativeImputer ii_rowabs + ii_conv, TargetEncoder te_bucket + bucket walk, acc_add,
  unique_cols RUN loads, mutual_info sorted neighbour search (dmi.mojo; MOJOLEARN_XPREP_MI_SORTED=0 restores
  the old walk, default 1).
- Host (CPU) binding: PowerTransformer keeps op pt_fit and TargetEncoder keeps te_enc without te_bucket when the
  binding has x_prep_host_column (the prep-cpu lane's host spellings); the device-only stages never reach it.
- FAST only (fastred.mojo): tree folds for col_stats, pt_fold, class_stats, ii_mean, ii_gram
  (MOJOLEARN_XPREP_FAST_FOLDS=0 restores row order, default 1); QDA keeps the row-order class sums (quality).
- Reverted after measuring: RUN 64, staged IDENTICAL folds (dstage), input-prefix upload.

### Same-Mac record at the tip (m4pro-b, request 1790595789320 at c0a29cf5d, 1M rows, reps 1)
IDENTICAL digests equal the before record (0cbfb3151) and the bd8057aa7 record on all 20 cases run:
power-transformer, onehot, ordinal, target-encoder, label-encoder, label-binarizer, multilabel-binarizer,
qda, lda, gaussian-nb on taxi + HIGGS. Seconds, before (m4pro-b, 0cbfb3151) -> tip (m4pro-b, c0a29cf5d):
| dataset | case | IDENTICAL before | IDENTICAL tip | FAST tip |
|---|---|---|---|---|
| taxi | power-transformer | 139.716 | 4.521 | 0.561 |
| taxi | target-encoder | 25.786 | 2.315 | 2.214 |
| taxi | onehot | 6.239 | 1.670 | 1.656 |
| taxi | ordinal | 4.883 | 0.361 | 0.359 |
| taxi | label-encoder | 4.733 | 0.948 | 0.934 |
| taxi | label-binarizer | 5.503 | 1.681 | 1.693 |
| taxi | multilabel-binarizer | 9.239 | 2.265 | 2.270 |
| taxi | qda | 1.145 | 0.419 | 0.359 |
| taxi | lda | 1.793 | 0.509 | 0.335 |
| taxi | gaussian-nb | 1.847 | 0.443 | 0.236 |
| higgs | power-transformer | 190.181 | 4.792 | 0.559 |
| higgs | target-encoder | 12.022 | 2.169 | 2.090 |
| higgs | onehot | 4.891 | 0.537 | 0.520 |
| higgs | ordinal | 4.719 | 0.355 | 0.355 |
| higgs | label-encoder | 4.498 | 0.935 | 0.948 |
| higgs | label-binarizer | 4.542 | 0.967 | 0.982 |
| higgs | multilabel-binarizer | 9.027 | 1.965 | 2.015 |
| higgs | qda | 0.917 | 0.423 | 0.366 |
| higgs | lda | 1.655 | 0.520 | 0.354 |
| higgs | gaussian-nb | 1.585 | 0.458 | 0.224 |
At 8f4891237 (m4pro-b, ~/mojolearn-evidence/prep-apple/chk_8f48912.txt) the IDENTICAL digests of
iterative-imputer, mutual-info-classif, categorical-nb (taxi) and the cases above also equal bd8057aa7's.

### Unproven: the integration check must cover these
The digests above are the speed bench's output hashes on Apple only; no steward identity request
(cross-vendor, sabotage arms) ran on this lane after bd8057aa7. Commits after bd8057aa7 that change code:
- d46468860 f_classif / f_regression RUN loads (IDENTICAL; select-f-* digests not rerun since)
- 234944b66 pt_map column major, pt_fold reads X only at K = 0 (IDENTICAL; PT digest same at tip)
- 32d8150c8, 7073ac4cf, d1e2a411a FAST folds (FAST only; paired quality on record, QDA excluded)
- 09e7e984e acc_add (IDENTICAL, touches every device fold: scalers, kbins, imputers, select-chi2, the other
  naive Bayes cases were NOT rerun after it)
- de9bdd7a5 mi_cd sorted search (IDENTICAL; MI digest same at 8f4891237)
- 2a4e46f7e pt_fold NaN test skip, 8f85d9b99 te_enc category test skip (digests same at tip)
- 0baee7d94 unique_cols RUN loads (onehot / ordinal / label digests same at tip)
- e68e0c26a pt_log (new op 109, N_OPS 110; device digest same at tip; the host-binding branch to pt_fit not run)
- c0a29cf5d TargetEncoder host-binding branch without te_bucket (not run on the CPU host binding)

### Known issues
- The identity steward (Metal vs CPU vs AMD / NVIDIA) and the seam arms 5400/5401/5402 must be rerun on
  the merged head; a new op (pt_log, 109) may need its own arm or selector entry.
- The CPU host binding paths for PowerTransformer and TargetEncoder (the hasattr(x_prep_host_column) branches)
  were never executed on this lane.
- bench/x_prep_speed.py: higgs categorical-nb raises "Negative values in data passed to CategoricalNB"
  (a bench input issue, not a kernel one; it has no before record either).
- taxi mutual-info-classif remains the slowest IDENTICAL case per row (tied points walk).
