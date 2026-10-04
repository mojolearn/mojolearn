# MCD main-MMA arithmetic repair

Branch lane/apple-fast-mcd-mma, base main d4bb2b795. Follows process in
EXPERIMENT_PROCESS.md. Prior scalar COMPAT failed capped quality at
ab4265c9a, tag gap26-mcdrepair-small-ready; failures remain documented.

Candidate define MOJOLEARN_MCD_BATCH_MMA, opt-in Apple FAST only, binding
x_decomp. It requires main's DECOMP_FAST_GEMM_MMA to be enabled; adding the
existing MMA_OFF define disables this experiment instead of changing baseline
arithmetic. Wide d>64 and d<2 still use the existing main path.

The code retains batched support compaction, column means, round-robin
Jacobi eigensolves, candidate stopping/ranking and support selection. It
replaces the scalar candidate covariance, weighted-eigenvector Gram and
Mahalanobis products with calls to the SAME _launch_gemm_mma used by main
DKit.mm. Each candidate keeps its own dimensions and K-split dispatch; calls
are queued without intermediate host reads. Centering, reciprocal scaling,
publication and distance row sums remain parallel GPU kernels. Inactive
candidate state is preserved by guarded publication; inactive MMA inputs
are zeroed. No host computation of model state or CPU fallback is added.

This conservative version enqueues one GEMM per candidate rather than inventing
a new batched GEMM reduction order. It may sacrifice some of COMPAT's speed
while eliminating per-candidate synchronization. Speed is UNKNOWN. Matrix
scratch adds approximately8*nc*r*d+12*nc*d*d bytes per phase. For the full
n100000 taxi phase B(nc about3330,r1500,d11), this is about440MB; concurrent
live phases may increase peak memory. Main's split-K atomics may still vary
last bits; no bit equality or quality pass is claimed before M3 evidence.

Actual saved cap3000 artifacts: d11,h1506, raw ranks both10 but final pinvh
ranks7(A) vs6(B). A's retained eigenvalue.000444677 disappears in B; precision
norm2249.28 vs58.83. Both satisfy their own C*P*C residual below5e-7. Raw
support differs338 entries; B final support1961 is a subset of A2083.
This is selection/model drift, not an isolated final precision inconsistency.

## Manager validation plan

Compile two FAST x_decomp arms with empty defines and
-D MOJOLEARN_MCD_BATCH_MMA. Verify binary manifests and exact checkout.
No local compile or GPU test was run by the lane.

Use a NEW quality tag with tools/mcd_compat_ab.sh, MCD_DEFINE=MOJOLEARN_MCD_BATCH_MMA,
MCD_A_SO and MCD_B_SO pointing to verified prebuilt arms. Start taxi cap3000,
then EllipticEnvelope on the same cap only if MinCovDet passes. Saved baseline
A.npz may be reused via MCD_BASELINE_NPZ after manager review of relevant source
changes; comparator still enforces dataset/lane/shape/data_sha equality.

MCD_DEFINE=MOJOLEARN_MCD_BATCH_MMA MCD_A_SO=... MCD_B_SO=... \
  bash tools/mcd_compat_ab.sh NEW_TAG taxi min-cov-det 3000

Existing gates remain: fitted location/covariance/precision/raw quantities/
query and training distances relative difference<=.01, flags/support/raw
support Jaccard>=.99, equal raw rank. Both flag masks being all true is
insufficient. Keep full timing gated until capped fitted state passes; run
only one full-size scored fit per arm, and do not rerace opponents. If quality
still diverges, instrument the first candidate/step covariance/determinant/
reciprocal mask before revising kernels again. This candidate changes no
thresholds, seed streams, tie rules or selection/stopping semantics.
