# Apple performance experiments

These are source-only candidates for Apple GPU throughput and call overhead. None has been compiled, executed, benchmarked, or certified for numerical correctness. No speedup is claimed. The user's reported timings motivated the work; they were not reproduced here.

This worktree is `experiment/apple-kernel-lanes-20261004`, based on `c96137714`. The original checkout and its uncommitted changes were left alone. Some named benchmark rows exist in later Apple branches rather than this base; candidate availability here does not establish integration into those rows.

## Experiment lanes

| Lane | Source | Intended change | Integration |
| --- | --- | --- | --- |
| GEMM | [Apple GEMM](apple_fast/gemm/README.md) | FP32 MMA tile shapes, cooperative threadgroup staging, K depth and padding | Explicit opt-in FAST dispatch in core GEMM; existing identical arithmetic is excluded |
| AutoARIMA | [Kalman scan](apple_fast_path/kalman/README.md) | Gaussian associative time scan, three blocked sizes, scalar specialization | Isolated Metal source and integration contract; production search is unchanged |
| Shared call overhead | [Resident calls](apple_callpath/README.md) | Persistent transfer and scratch buffers, grouped readback, enqueue then collect | Explicit experimental APIs and an existing-kernel adapter |

The dedicated bitwise-identity call-path experiment lives in a separate sibling worktree, `../mojolearn-identical-callpath`, branch `experiment/identical-callpath-all-algorithms-20261004`. Its scope is all algorithm families and GPU vendors, with a shared typed session and explicit adapters. It is intentionally not merged into this one.

## Numerical boundaries

MMA tiling and time-parallel Kalman composition can change floating-point association. These belong to FAST experiments, not bitwise-identical dispatch. Reusing buffers and moving waits can preserve identity when every arithmetic operation, launch geometry, input bit pattern, and dependency stays the same. That property still requires later verification; structural intent is not a certificate.

The Kalman rewrite targets likelihood evaluation, not taxi forecast quality. It must retain initialization, differencing exclusions, likelihood convention, optimizer inputs, and failure handling before any production integration. Increasing search breadth is outside these experiments.

## Deferred validation

Compilation, correctness checks, device execution, timings, and peak comparisons are deliberately deferred at the user's request. Future work should first establish correctness for each individual candidate and confirm that the intended caller reaches it. Only then would standalone GEMM timing and end-to-end comparisons be meaningful. No runner or background measurement is started by this worktree.
