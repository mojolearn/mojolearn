# Lane status: lane/cpu-training-mamba (agent cpumamba, 2026-09-15)

## Done
- a59319644: the mamba host family (`bindings/_mojolearn_mamba_host.mojo`,
  `mamba/host/`, `tools/mamba_host_gen.py`), lanes mamba2, mamba2-dtlimit,
  mamba1, mamba3 declared. M4 one core: IDENTICAL=36 train, 36 infer, 36 batch
  against the 166-lane record; sabotage DIVERGENT=36
  (`bench/results/identity_break/2026-09-15_cpu-mamba/`).
- ddd875cc8: merge of origin/main (4eac5719a), no conflicts, pushed.
- The Mamba-1 host backward oracle fix (this commit): the oracle carried three
  readings the plan rejected (B7 fused silu', B18 two folds, T1 stored seed);
  fixed, new host check `mamba/checks/mamba_backward_host_oracle_check.mojo`
  (`pixi run check-mamba1-backward-oracle-host`) FAIL 151 before, PASS after
  (`bench/results/mamba_backward_oracle_host/2026-09-15/`); `mamba/README.md`
  claim corrected.

- Restart (Sep 15 ~11:15 ET): gate run 34973248141 at ddd875cc8 was cancelled
  (1h47m, the old job limit). origin/main (590c11c86) merged in with no
  conflicts as 082fbb569. On the M4, one core, core and mamba host bindings
  rebuilt from 082fbb569 into a fresh directory: `summary: IDENTICAL=36`,
  `summary (infer/model): IDENTICAL=36, N/A=36`, `summary (batch):
  IDENTICAL=36`, `require-columns 4 ... : OK` (204 s); a
  `-D MOJOLEARN_HOST_SABOTAGE=1` mamba build read DIVERGENT=36 train, 35 infer
  (mamba1/negative infer IDENTICAL, as in the committed sabotage diff) and 36
  batch. docs_facts --check and wheel_ci pins pass.

## Running
- Nothing. Pushed with [skip ci] until Codex's routine gate (08b50887a) is on
  main; then this branch gates after lane/cpu-training-transformer merges. No
  boxes rented.

## Next commands
1. `gh run view 34973248141 --json jobs --jq '.jobs[] | "\(.name): \(.conclusion)"'`
   must be success on all seven runners (covered lanes step reads
   `require-columns 4 ... OK`, sabotage step "sabotage caught").
2. When green: `git fetch origin && git merge origin/main`, resolve lists and
   spans (`python3 tools/docs_facts.py --write` then `--check`), rerun
   `python3 packaging/wheel_ci.py pins .`, `python3 packaging/wheel_ci.py inventory python/mojolearn`,
   `cd python && python3 -m mojolearn.tests.test_host_surface` (or pytest),
   `python3 -m mojolearn.tests.test_cpu_training_mamba`, then
   `git push origin HEAD:lane/cpu-training-mamba && git push origin HEAD:main`.
3. Samba lanes belong to agent cpumamba13 (builds on this branch). A patch
   adding `neural_rng` to the training host binding over a generated
   `core/philox_neural.mojo` is at $SP/cpumamba/training_neural_rng.patch
   (needs `core/philox.mojo` and `core/philox_neural.mojo` added back to
   `tools/mamba_host_gen.py` SOURCES; not built).
