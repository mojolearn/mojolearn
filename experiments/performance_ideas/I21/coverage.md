# I21 implementation coverage

Mode: IDENTICAL. New source mechanisms are default off.

Implemented: Actual complete GMM E/M oracle, planted/collapsed/empty components, likelihood, fit iteration and launch-invariance gates under fused-Cholesky and one-drain rollback controls.

Remaining original-card scope: A new default-off IDENTICAL component-batch route now bounds all extra precision/sample/mean/GEMM workspace to16 MiB, reuses buffers between batches, and retains every per-cell GEMM/mahal fold. Tail batches1/3/9/17 components compare all E-step stages to the host replay. A separate default-off pair staging kernel now loads each X word once for two components, reuses the original shared subtraction/multiply helper, and executes the unchanged per-component covariance GEMMs with bounded16 MiB scratch and an included lifetime wait; no result claims to overcome historical AMD stacking loss.

Qualification: compile checks only on the development machine. Same-version host/NVIDIA/AMD/Apple output identity and NVIDIA+AMD full-operation performance acceptance have not been demonstrated. Apple IDENTICAL is an identity witness only. No speed claim or production promotion is made.

Merge admission: new source candidates remain explicit default-off opt-ins.
Qualification remains pending. `native_arms.json` lists independently compiled
incumbent/candidate and available rollback arms; compilation never promotes a
switch or supplies performance evidence. Existing promoted defaults remain unchanged.
