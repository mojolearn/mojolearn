# I21 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual complete GMM E/M oracle, planted/collapsed/empty components, likelihood, fit iteration and launch-invariance gates under fused-Cholesky and one-drain rollback controls.

Remaining original-card scope: A new default-off IDENTICAL component-batch route now bounds all extra precision/sample/mean/GEMM workspace to16 MiB, reuses buffers between batches, and retains every per-cell GEMM/mahal fold. Tail batches1/3/9/17 components compare all E-step stages to the host replay. New centered-tile reuse remains unimplemented; no result claims to overcome historical AMD stacking loss.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.
