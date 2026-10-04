# MCD repaired-quality review

Reviewed measured source ab4265c9a on private lane/apple-fast-mcd-review.
M3 tag gap26-mcdrepair-small-ready, taxi cap3000, failed meaningful quality.
Source remains opt-in, no promotion and no full-board timing.

Observed location/covariance/precision relative differences .032817/.085678/
.999817; query-distance difference .997423. Support Jaccard .941431, raw
support .798209; raw covariance ranks both10. Quality-only fit times5595.80ms
and828.96ms do not qualify as a board win. Flags both all true are degenerate.

## Concrete mismatch

Main DMcd.emp_cov computes centered X^T X through DKit.mm; pinvh multiplies
weighted sorted eigenvectors through that same route; mahal uses Xc*P there.
On Apple FAST, launch_gemm selects DECOMP_FAST_GEMM_MMA by default. Its output
is native matrix-unit arithmetic; for one output tile and k>=1024 it splits
k into at least two aligned chunks (DFG_MIN_SPLIT_STEPS512), combined using
FP32 atomics. At n3000/default h about1509, final covariance hits that path.

mc_moment_kernel instead computes scalar FMA chains over4096-row leaves;
mc_pinvh_kernel uses a scalar weighted Gram; mf_dist_kernel uses scalar dot
products. Matching main's Jacobi schedule and legacy scalar fold is therefore
insufficient. Singular determinant signs, rank cutoffs and h-smallest support
selection can amplify tiny initial differences into a different local fit.
That amplification is a hypothesis; final A/B artifacts do not prove its first
point of divergence. State-machine audit found the same permutation streams,
C-step limits and use_prev tie rules, with no definite new ordering bug.

## Next steps preserving fit semantics

1. Analyze existing A/B NPZs with tools/mcd_compat_diagnose.py(A B). No fit,
   no GPU and no timing occurs. It checks input identity and reports covariance
   spectra, conditioning, Moore-Penrose residual and support disagreements.
2. If necessary isolate the mismatch on a new tiny diagnostic fixture with
   DECOMP_FAST_GEMM_MMA_OFF in BOTH the baseline and COMPAT builds, keeping
   the same1%/.99 gates. This is causal diagnosis only: even a pass cannot
   approve COMPAT against actual main and must never update the board.
3. Implement candidate batching around the existing main matrix kernels:
   compact support in the same order; create centered candidate matrices;
   use exactly main's covariance MMA shape/K splitting and scale; preserve
   sorted eigenvectors, reciprocal mask, weighted-Gram MMA; use main's
   Mahalanobis MMA and row fold. Add candidate dimension to launch scheduling,
   retaining per-candidate fold shape rather than changing the estimator.
4. Capture first-candidate/step support, mean, covariance, determinant and
   reciprocal-mask digests in a separate diagnostic build. Compare before
   allowing subsequent selection, so a final-model failure identifies its
   source instead of triggering another blind full fit.
5. Repeat capped quality against unmodified current main. Only after all
   fitted-model metrics pass should full timing be eligible. Preserve
   non-positive logdet behavior, support ties, seed streams and stop rules;
   do not alter thresholds, drop precision/distance checks or accept flags
   alone. Main's split-K atomic ordering can itself vary: FAST permits bits
   to differ, but meaningful fitted quality must remain within existing gates.

This review changes comments/documentation and adds artifact diagnosis only.
No numerical repair, runtime default, compilation or remote job is claimed.
