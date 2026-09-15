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
  (1h47m, the old job limit). origin/main (450423c95, the routine CPU gate that
  moved full training certification off pushes) merged in with no conflicts as
  082fbb569; the [skip ci] commit 3cecb4238 said 590c11c86, which was wrong.
  The runs below were repeated at 3cecb4238 with the same counts. On the M4, one core, core and mamba host bindings
  rebuilt from 082fbb569 into a fresh directory: `summary: IDENTICAL=36`,
  `summary (infer/model): IDENTICAL=36, N/A=36`, `summary (batch):
  IDENTICAL=36`, `require-columns 4 ... : OK` (204 s); a
  `-D MOJOLEARN_HOST_SABOTAGE=1` mamba build read DIVERGENT=36 train, 35 infer
  (mamba1/negative infer IDENTICAL, as in the committed sabotage diff) and 36
  batch. docs_facts --check and wheel_ci pins pass.

- Merge rule (Andrew, Sep 15 afternoon): merge on one-core M4 evidence at the head, CI informational.
  origin/main ee13e0d4b (public CPU inference only) merged as a7efe374a, then 6ad50f394 (with
  lane/cpu-training-transformer). Conflicts: generated spans (taken from main, regenerated),
  host_surface.py TRAINING_LANE_NAMES (both sides kept) and NO_CPU_PATH (now "the Samba blocks"),
  the gate workflow (main's file, no push trigger), test_cpu_training_misc's sentence assertion
  ("Samba blocks"); test_cpu_training_transformer's sentence assertion now expects "Samba blocks".
  The mamba family is ships_in_wheel=False (training-only reference build).
- main's wording pass changed docstrings in six generator sources, so mamba/host/gen/ was regenerated
  (`python3 tools/mamba_host_gen.py --write`; the diff is docstring prose only; `--check` read STALE on
  five files before and passes after). origin/main 115eb51ec then merged in as e292e0f19 (generated
  spans only).
- M4, one core, shared machine, at e292e0f19 (`lane_merge_checks.sh wt-mamba
  mamba2,mamba2-dtlimit,mamba1,mamba3 "core mamba" mamba "test_host_surface test_cpu_inference_boundary
  test_cpu_training_mamba test_cpu_training_transformer test_cpu_training_misc"`: core and mamba host
  bindings built into a fresh directory, identity_break diffed against the 166-lane record with
  --require-columns 4): IDENTICAL=36 train, IDENTICAL=36 infer (N/A=36 model), IDENTICAL=36 batch,
  require-columns 4 OK (341 s); test_host_surface 106 passed, test_cpu_inference_boundary 8,
  test_cpu_training_mamba 7, test_cpu_training_transformer 8, test_cpu_training_misc 14;
  mamba_host_gen --check, docs_facts --check, wheel_ci pins and inventory pass.
  The -D MOJOLEARN_HOST_SABOTAGE=1 mamba arm at the same head read DIVERGENT=36 train, 35 infer (mamba1/negative
  infer IDENTICAL, as in the committed sabotage diff) and 36 batch.

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
