# Fused small-MLP step (PR #7, lane/apple-mlp-fused fd791fc95 + two compile fixes) on NVIDIA L4, 2026-09-30

Job `tools/mlp_fused_job.sh` (nvc1-0004). The PR as pushed did not compile; two fixes on
lane/mlp-fused-measure-20260930: `_launch_mlp` generic over its pointers' origins, and the two
row-sum launches pass a distinct unused buffer as `other` (operation 3 never reads it).

| check | result |
|---|---|
| `tools/mlp_step_check.py` 256 rows, 64 steps | PASS, every byte equal; per-operation 1.819 ms, fused 1.126 ms (1.61x) |
| `tools/mlp_step_check.py` 32 rows, 64 steps | PASS, every byte equal; 1.633 ms -> 1.006 ms (1.62x) |
| `matmul` digests, main's GEMM host entry vs the PR's (one wait fewer), 7 shapes + transposes | 21 of 21 equal; timings unchanged |
| small-MLP surface and numerical-edge tests | 17 passed, 9 skipped |
| mlp-train-step board cell, `MOJOLEARN_MLP_FUSED=0` -> 1 | 2.149 ms -> 1.644 ms (1.31x), same losses |

Apple (the target: M3 Ultra 18 ms vs M2 Pro 9 ms) and AMD: not run.

## The 9 skipped tests (GPU numerical gate, opt-in `MOJOLEARN_RUN_SMALL_MLP_GPU=1`)
Run on the L4 with the flag, `MOJOLEARN_MLP_FUSED=1` and `=0`: 26 passed each, after one test fix.
`test_public_adamw_complete_state_matches_independent_equations` failed on main as well (main's linalg,
per-operation path): it wrote into `state_dict()`'s optimizer `Array`s in place and did numpy math on them,
which mojolearn's `Array` does not support. The test now assigns fresh arrays and converts to numpy in its
FP64 reference; the assertions are unchanged.
