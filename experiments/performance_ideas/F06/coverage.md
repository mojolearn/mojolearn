# F06 pending qualification

Implemented independent bounded batches, active compaction, covariance reuse,
and active_cov interaction variants. New flags require FAST Apple and remain
default off. Promoted BMMA/wide/skip-pinvh paths retain their current defaults.

Compaction uses stock parallel integer scan kernels, preserving original IDs,
output slots, gate rechecks, split-K seeds/windows and epilogues. Three Int32
candidate planes plus bounded chunk sums are phase-owned and limited to65536
candidates (approximately768KiB); larger phases retain main. Count read and
completion precede compact submission and must be charged in full-fit timing.

Covariance reuse requires the exact unchanged support mask inside an immutable
phase/row mapping. Step0 and inactive candidates refuse reuse. Previous mean
and covariance are copied to current parity; a separate moment-work gate skips
recomputation. Determinant, stopping, eigensolve/rank, precision, final distance,
support and reweighting follow the incumbent path. No cache crosses a phase or
fit. Stable covariance can remove noise-induced extra iterations; FAST quality
admission must therefore compare fitted state and downstream consumers.

Native prerequisite: empty/all/sparse membership, multichunk scan tails, stable
original-ID ordering and poison preservation; support invalidation, step0 and
inactive cache refusals; actual raw MCD search and raw support/distance gates.
Public fixture: contaminated, singular and near-singular inputs, positive actual
route/reuse counters, heldout Mahalanobis ranking, actual MinCovDet.score and
EllipticEnvelope predict/decision_function/score_samples/score. x_metrics is an
explicit prerequisite for actual EllipticEnvelope.score. Process RSS is reported
separately; GPU peak scratch/resources remain a device qualification gate.

At the final frozen commit, the Apple compile campaign enumerates default,
active_compact, cov_reuse and active_cov from manifest.json. Build x_decomp and
core/x_metrics prerequisites with existing binding scripts; compile native_check
for portable apple-m1 + metal:1 with each variant's defines. Compiler, device
quality and performance remain pending; no preliminary F06 compile or GPU
execution occurred. IDENTICAL matrix-bit matching is not a FAST gate.
