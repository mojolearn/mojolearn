# MCD phase A/B G1 self-Gram (OPEN, uncompiled)

Base201fe7367461af236f4adf0331f367d8a02cd29d. New algorithm-specific flag
MOJOLEARN_MCD_FAST_G1_GRAM, FAST+Apple only, default off. Reuses existing
scoped_dispatch.mojo::scoped_kernel[64,64,False] through a gated/batched wrapper.
No duplicate GEMM body or change to shared selector flags.

## Credible actual caller

MinCovDet / EllipticEnvelope Istella remains roughly86s versus44s opponent.
Public _binding_fast_mcd -> x_decomp_mcd -> MCD_BMMA -> _mma_covariance ->
launch_gemm_mma_batched. For100000x220, default support h≈50111 gives phase-A
subset K≈151 and merged phase-B K≈752. Thousands of slots run phase A and up
to30 phase-B steps. TN self-Gram M=N=220 is non-split. Other robust-fit work
may dominate; this is a hypothesis, not a speed prediction.

Eligibility (2026-10-04, no shape window): any M=N, K>=1 with ta=True/tb=False,
identical input pointers and batch strides, sufficient batch strides, distinct
output base, existing splits==1, Int32-safe shapes and strides. These are the
kernel's correctness limits only. The old window 129<=M<=256, 128<=K<=1023
bracketed the board and was removed as benchmark-tuned; it survives only behind
default-off MOJOLEARN_LEGACY_NARROW_MCD_G1_GRAM. The window-free route is
UNMEASURED. Weighted precision (nonalias), Mahalanobis NN and all split/atomic
products remain incumbent.
Original active gate is checked before operand access. Candidate z selects
original batch strides; output grid64x64 and128 threads are unchanged.
Non-split output overwrites; inactive candidates retain no-write behavior.
No zero/scaling/publication/support/eigen/pinvh/sorting/convergence changes.

This is not a retry of held full-PCA atomic Gram or ordered-covariance MCD.
Prior G1 matrix gains and scoped probe PASS do not admit batched MCD quality.

## Minimal build and reach

Changed binding x_decomp only. M2 manager/semaphore, compile-only:
MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_NUMERIC_MODE=fast
MOJOLEARN_SKIP_BUILD_GATE=1 bash bindings/build_x_decomp.sh.
Flags go through MOJOLEARN_MOJO_BUILD_FLAGS:

A: -D MOJOLEARN_MCD_FAST_G1_GRAM_AUDIT
B: -D MOJOLEARN_MCD_FAST_G1_GRAM_AUDIT -D MOJOLEARN_MCD_FAST_G1_GRAM

No generic SCOPED_GEMM flags or held ordered-covariance flags. Manifest exact
source/binding/mode/ordered-flags/A+B hashes. No artifact built by this lane.
AUDIT-only binding exports:

- mcd_g1_gram_on():0/1.
- mcd_g1_gram_count(i):0 all batched launch requests,1 eligible requests,
  2 selected requests,3 eligible potential candidate slots (NOT active counts).
- mcd_g1_gram_last(i):M,K,N,nc,a_batch_stride,c_batch_stride,splits.

Fresh-process actual-fit deltas must show A eligible>0/selected0 and B
selected==eligible>0. NO_REACH is a control, never timing admission.

## Quality-first plan (still owed)

1. Small exact-source batched-entry matrix probe over a generic spread
(d 8..1500, K 2..1500, odd and tile-multiple sizes); all-inactive/mixed gates;
batch padding; sentinels. Controls: weighted nonalias, NN, output alias,
split plans. Independent FP64 vs A
with existing5e-6 error bound and zero relative/maxabs regression allowance.
Unchanged inactive outputs and split plans mandatory. Probe/helper still owed.
2. Actual MinCovDet then EE over the generic spread in tools/mcd_g1_quality.py. Smaller cases may reject early
but do not certify100k scope. Preserve mcd_compat_quality.py gates (1% fitted
state,.99 support/flag Jaccard,equal raw rank) and independent oracle-error
checks. Better objective never excuses changed anomaly decisions. No loosening.
3. Pinned fresh installation must explicitly stage IDENTICAL base finiteness/
conversion dependencies used by _M.from_input where required. Validate actual
loaded symbols/paths before fit. Existing quality script is not an artifact
installer or pinned policy. No M3 build fallback.
4. Only after reachable matrix and fitted quality pass: one scored fit+first
read per applicable arm/dataset, no opponents/replays. Keep opponent-quality
holds separate. Review native current-main drift and default/OFF before merge.

Candidate+diagnostics are source-ready only. No compilation, model execution,
numerical test or queue action performed. Manager owns build/admission.
