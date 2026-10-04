# Eigh panel review and next MMA experiment, 2026-10-04

Recommendation: HOLD original tridiagonal solver, do not promote. Numerical
quality substantially improves relative to main, but its preregistered absolute
quality gate failed and its panel T-factor launch uses a single block. The new
panel scheduling candidate repairs the latter only and remains opt-in.

## Original evidence and scope

Source lane/apple-fast-w4-eigh @ 97304d34cb4ed1cb9131123881ca247f4e0d4c32.
Manager supplied generic timing: A 43729.2 -> B 701.9 ms (62.3x).
Board4096 eigenvalue error 6.08669e-5 -> 4.26951e-7 (142.6x better),
residual 5.37499e-5 -> 6.13409e-7, orthogonality .0011713 -> 1.68868e-6.
The fixed eigenvalue target is 3.5e-7, so this remains FAIL, never a PASS.
Gram1024 fallback orthogonality .00026960937863630826 is unchanged A/B,
but exceeds the fixed 2e-4 absolute bound; retain that separate failure too.
The opponent-quality HOLD (3.5e-8 eigenvalue error) remains visible.

The original gate is FAST + Apple, n>=512, called only by DevExec.eigh.
Small matrices retain main; repeated/clustered or invalid spectra refuse via
a GPU-computed flag and rerun Jacobi on untouched input. No host numerical
solver: the sole intermediate host read decides refusal. Eigenpair recurrences
are serial within each eigenpair but parallel across all eigenpairs. Original
T-factor preparation was explicitly grid_dim=1 with 32 threads, not compliant
as a new default. Shared downstream decomposition does not call this route.

Source drift review: merge-base with origin/main cd8d095cf is 4b85be62c.
Main changes after that base only amend EXPERIMENTS.md; no measured kernel,
quality fixture, binding or dispatch changed. The new branch merges cd8d095cf.

## Opt-in repair

Define MOJOLEARN_EIGH_FAST_TRIDIAG_PANELS (no default or _OFF promotion yet).
After tridiagonalization, stored reflectors and tau never change; the reverse
back-transform loop changes only Z and temporary S buffers. Compute all panel
Gram matrices in one two-dimensional grid, then all T factors in one grid with
one block per panel (at least 16 panels in the routed domain). Each thread's
fold order is unchanged. Reverse back-transform indexes its precomputed T.
For board4096, extra Gram/T storage is about 1 MiB, replacing 256 launches
with two launches; no output residency change. No quality improvement claimed.

The existing pair helper now validates the new define in its binary manifest.
The eight-case quality fixture and all thresholds are unchanged. M2 A/B compile
is owed; manager alone stages/enqueues the new exact SHA. Run the quality helper
once per arm under a fresh tag; expect the same absolute failures unless the
compiler changes rounding. Timing remains blocked without its exact PASS
receipt. Do not replay the original timing or relax its failed thresholds.

## Concrete next MMA candidate

The shared AFN kernel uses 64x64 tiles, four simdgroups and KB32 by default.
Its padded shared page is 17,920 bytes, so only one page fits and every window
uses two barriers. KB16 needs 9,472 bytes per page and enables existing double
buffering. LU's lfm_gemm_sub_kernel duplicates the loaders/MMA but always uses
one page; changing only shared AFN will not accelerate LU. The reported .4
TFLOP/s is whole-factorization throughput, not isolated MMA throughput, and
cannot by itself establish an MMA hardware bottleneck.

Propose two separate candidates, not a bundled default: (1) decomposition-only
KB16 specialization via a new MOJOLEARN_DECOMP_FAST_MMA_K16 define at its tile
launcher; keep global AFN_GEMM_KB unchanged to avoid neural/MCD/LU spillover;
(2) LU-only two-page KB16 implementation after isolated LU update evidence.
First add an opt-in standalone device harness that calls the actual relevant
launchers, uploads inputs before timing, synchronizes before/after one launch,
and reports 2*m*n*k / elapsed plus first-read-inclusive end-to-end time. Use
one scored A and B launch for each preregistered shape; no repeats/medians,
no opponent rerace. Shapes: dense 4096^3, LU-like 4096x4096x256, tall Gram
220x220x65536, and thin 64x220x65536, all needed transposes; odd tails such as
257x259x271 belong in correctness, not another speed sweep. Include actual
board caller timings only after quality passes; standalone gains do not update
board cells.

Before measuring, fix quality checks: float64 reference relative Frobenius
residual no worse than main, finite outputs, cancellation and rank-deficient
fixtures; downstream PCA/RSVD reconstruction and LU factor/solve residual
checks unchanged. One sample cannot establish a noise bound; tiny speed gains
stay inconclusive. No standalone harness or GEMM implementation is claimed
complete by this review.
