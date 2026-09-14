# Wider Gram outputs: two H100s

RunPod `rbtojh7e0esekh`, two H100 80GB HBM3, IDENTICAL, sm_90a. All
compilation and model execution occurred on the pod, using the R2 enwik8 corpus.

The qualified job passes:

- 21 complete Gram bit comparisons: FP32-v1 column-Gram output widths
  129/257/513 with contraction lengths 3/127/259/1025; sequential row-Gram
  widths 3/17/33 with contraction lengths 7/129/257. Includes subnormals and
  signed zero. No tolerance or hash-only comparison.
- Two oversized-index refusals before staging: output and input-copy limits.
- 18 full fitted-state/output cases: tall OLS, weighted/intercept OLS, Ridge,
  whitened covariance PCA and TruncatedSVD at 129/257 features; wide minimum-norm
  OLS at 3x7, 17x65, 17x129 and 33x257, with/without intercept.
- All 20 previous 7/16/65/128-feature cases, with exactly unchanged JSON
  receipts and output hashes (`golden-comparison.log`).

Source archives are applied to the pod's initial `eaec62839` checkout, after
this day's pointwise Python updates (the preceding pointwise receipt records
those). `gram-output-qualified-source.tgz` is the complete final overlay at
`7c18b7a6f`. Native GPU enqueue rejected omitted default bounds at first; the
explicit-argument correction is retained. Passing intermediate runs before the
index guard and before preserving the one-device full-extent sentinel are also
retained. The final qualified build and all gates were rerun from the final
source, not inferred from the earlier runs. Binary and archive hashes, all job
scripts/exit codes and compiler logs are included.

No local builds or tests. No new cross-vendor, speedup, pooled root-state or
large-memory claim. Full root data/eigensolver state remain. The pod stays
leased for the following QR/full-PCA batch.
