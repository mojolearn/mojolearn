# Public CPU Cholesky inference, Apple M4 CPU column

lane/inference-embedding-ivf-cholesky, 2026-09-15, stage 0. The Cholesky door
(`cholesky_profile_jitter`, `cholesky_factor`, `cholesky_solve`) joined the
linalg host binding, which ships in the inference wheel. On a CPU-only install
`Cholesky` binds `_mojolearn_linalg`; its `fit` factors a given matrix
publicly (`_CPU_FIT_IS_INFERENCE`), and `save`/`load` (format
`mojolearn-cholesky-1`) plus `mojolearn.host_model` (a `HostCholesky`) carry a
factor from a GPU box to a CPU solve.

## How it ran

Apple M4, one core, shared machine (`nice -n 19`, thread variables 1, one
process at a time). Host bindings built from the lane worktree at 0486e44ba
plus the uncommitted stage 0 edit (the commit that carries this directory):

- `_mojolearn_linalg_host.so` sha256 prefix 16e965d71088af52
- sabotage build (`-D MOJOLEARN_HOST_SABOTAGE=1`) sha256 prefix 32c4e5d79ba87619

    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_HOST_DIR=<set> \
      python tools/identity_break.py --lanes cholesky --repeats 2 --json cpu-apple-m4.json
    python tools/identity_break.py --diff bench/results/identity_break/2026-09-14_166-lanes/{apple-m4,nvidia-h100-sm_90a,amd-mi325x-gfx942}.json \
      cpu-apple-m4.json --require-columns 4 --lanes cholesky --owed-json owed_cells.json
    python tools/cpu_identity_gate_check.py owed owed_cells.json \
      --production cpu-apple-m4.json --sabotage cpu-apple-m4.sabotage.json

## Result (diff.record-vs-cpu.txt, exit 0)

| part | verdict |
|---|---|
| train (factor, logdet, info, nb, jitter, solve), 9 fixtures | IDENTICAL x4 = 9 |
| infer (solve of held-out right-hand sides), 9 fixtures | IDENTICAL x4 = 9 |
| batch (each right-hand side alone vs the whole B), 9 fixtures | IDENTICAL x4 = 9 |
| model (save, load, solve), 9 fixtures | OWED x1 = 9 |

The model cells are new: every GPU record reads `n/a:no-save` for them because
`Cholesky` had no `save` when the 166-lane record was taken. They are owed to
the next release record (one Apple, one NVIDIA, one AMD column).

## Sabotage (diff.record-vs-cpu-sabotage.txt, exit 1 as required)

- train DIVERGENT on 9 of 9 fixtures.
- owed check: 9 of 9 owed model cells MOVED (verdict OK).
- infer and batch DIVERGENT on 7 of 9. The two that do not move are
  `denormal` and `denormal_ftz`: their held-out right-hand side solves to the
  same bytes under the wrong factor too, and all three GPU columns carry that
  same hash for both fixtures. The train cell of both fixtures moves.

## Not measured here

No Metal column was taken for the model part (it is a Python save and load
around the device solve the infer cell already holds). No installed wheel was
built for this stage; the linalg binding was already in the wheel's host set.
