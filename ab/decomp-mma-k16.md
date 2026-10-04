# Decomposition MMA KB16 candidate, 2026-10-04

Base origin/main cd8d095cf. Define MOJOLEARN_DECOMP_FAST_MMA_K16, default off,
binding x_decomp. Hypothesis: the existing AFN 64x64 kernel at KB32 uses a
17,920-byte shared page and two barriers/window; KB16 fits two 9,472-byte
pages and activates its existing double-buffer path. This candidate specializes
only non-split calls from x_decomp/device.mojo::_launch_gemm_mma. Same ascending
8-step MMA fragments and tile dimensions, no split-K reduction-order changes.
IDENTICAL/other vendors unchanged. LU's separate MMA kernel, Cholesky's kernel,
core/gemm, neural routes and MCD's separate batched MMA kernel are unchanged.
Other direct users of this shared decomposition launcher can change: downstream
PCA/RSVD/KPCA/LLE and unbatched MCD calls require their own quality/board evidence
before any promotion. No universal speed gain is claimed.

No prior KB16 classical measurement found in the ledger. Neural double-buffer
flags exist as merged-unmeasured opt-ins; this does not validate them.

Manager: M2 compile A empty, B -D MOJOLEARN_DECOMP_FAST_MMA_K16. Stage exact
SHA/hashes via intake, then queue quality with tools/decomp_mma_k16_pair.py
quality SOURCE w2-mma-k16-q-20261004. Queue timing with the same helper:
timing SOURCE w2-mma-k16-q-20261004 w2-mma-k16-t-20261004. Matching quality
PASS receipt is mandatory; no opponent calls or replay. Source SHA must equal
checkout and manifest. Both helpers retain raw JSON and logs in exclusive
new output directories. Subagent has not compiled or executed GPU code.

Quality: 12 fixtures, odd 257x271x259 dimensions, all four transposes, random,
rank-one and cancellation operands. NumPy float64 reference is verification
only; scaled residual must be <= A with no added tolerance and output finite.
Fixture outputs/hashes retained. Timing: five preregistered shapes, each has
one scored host-facing call and one distinct resident-route call per arm. Dense
4096^3, rank-256 4096-square update, transposed 1024^3, and tall/skinny split-K
controls. Cold launch effects are included; no warmups/repeats/medians.

Host-facing time includes allocations, upload, solve, download and full first
read. Resident inputs upload before timing; a one-element download fences
completion, then the full download and full CPU read complete the measurement.
Resident completion is an upper bound including Python/allocator/fence cost,
not a pure Metal-event GPU timer; do not compare it to advertised peak as if
it were isolated hardware throughput. Full call+read governs any speed claim.
No output-residency change. CPU sums/reference are harness verification/first
consumption, not part of the production GPU algorithm.

These low-level numbers do not update board cells. If promising, run affected
algorithm quality and one scored A/B caller pair each before proposing a scoped
default and _OFF rollback; default/_OFF M2 builds are then required.
