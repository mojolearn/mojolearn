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
(filled as the steward returns)

## Next
