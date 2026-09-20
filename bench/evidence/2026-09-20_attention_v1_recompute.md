# V1-exact attention backward recompute arm (2026-09-20)

The opt-in build flag `MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD` prevents the
forward wrapper from retaining its `[B,H,L,S]` exponent buffer. Backward then
uses the pre-existing production v1 recompute kernels. Default routing is
unchanged.

Apple B1 H12 L2048 HD64: the default and forced recompute paths both retained
zero cells on this column. Default backward was 130.160, 132.702, 129.905 ms;
forced recompute was 126.382, 127.510, 133.349 ms. All dQ/dK/dV cells were
bit-identical (`exact_bad 0`). The full fused/eager gate also passed all 15
cases under the flag.

RunPod L40S (pod `9k6bq011nxaz8t`, $0.79/hour, driver 580.159.03, destroyed
and GET-confirmed 404), same B1 H12 L2048 HD64 inputs:

- tuned v1 estash backward: 3.342 warmup, 2.728, 2.732 ms;
- v1 exact recompute: 4.677 warmup, 4.465, 4.472 ms;
- recompute is 1.64x the warm estash time;
- recompute removes 50,331,648 float cells, 201,326,592 bytes;
- dQ/dK/dV comparison: `exact_bad 0`.

This is a memory/throughput tradeoff, not a new default. It is useful when the
201 MB stash prevents the requested batch/sequence from fitting; the tuned
estash path remains appropriate when memory is available.
