# Operation-level Cholesky — two H100s

RunPod pod `klnsccqg3fsri4`, two NVIDIA H100 80GB HBM3 (sm_90a), 2026-09-14
22:03-22:06Z. Source: the `git archive` of commit `00037ee37`
(`commit.txt`); box and Mac source SHA256 agree (`source_sha256.txt`). Body
`body.sh`; GPU work ran serially. Design: `docs/multi_gpu/cholesky.md`.

## Result (`out/gate.txt`)

- `training/checks/cholesky_parallel_check.mojo` (production build): 40 PASS
  lines, `PASS cholesky parallel gate`. For n in 1, 2, 31, 32, 33, 65, 100,
  257, 513 with 1, 2, 3 and 7 right-hand sides at the pinned ridge, and n=129
  with no ridge: the one-device and two-device factor traces (every panel's
  factored, solved and trailing matrix) are equal, and so are the factor bits,
  `info`, `nb` and `logdet`; the solve traces (forward and back) and solution
  bits are equal. Two non-positive-definite matrices fail at the same `info`
  (51 at n=100, 129 at n=257) with the same partial factor.
- The same gate built with `-D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1` FAILS:
  `solve trace differs n1-r2: 0 chol.solve.forward f32 2 c0d945299ff3c0f7
  VS 0 chol.solve.forward f32 2 c356822c8d994d99` (the first case with more
  than one right-hand side). That run reached the solve before any trailing
  update existed, so it demonstrates the column partition only; the
  trailing-row sabotage is shown failing on its own in the later
  kernel-methods legs with `MOJOLEARN_CHOLESKY_CHECK_FACTOR_ONLY=1`.
- `tools/parallel_cholesky_check.py`: `PASS 7 Cholesky configurations and 5
  refusals` (`out/public.json`), n up to 777 and 8 right-hand sides, one
  failed factor (info 49) whose solve is refused.
- `cholesky/checks/cholesky_check.mojo`, the unchanged single-device gate:
  `ALL PASSED`, including its 10 sabotage arms.

No speed or capacity claim is made.
