# Lane status: lane/cpu-training-samba (CPU training for samba and samba-untied-dropout-accum)

Updated 2026-09-15 by the cpusamba agent (launched as cpumamba13, re-pointed to Samba by the orchestrator).

## Base

- Built on origin/lane/cpu-training-mamba (the mamba host family) with
  origin/lane/cpu-training-transformer (the transformer host family, gate 34969598898) and origin/main
  merged in. Neither peer branch is on main yet; this branch cannot merge to main before both do.

## Done

- `neural_rng` in `bindings/_mojolearn_training_host.mojo`, over `mamba/host/gen/philox_neural.mojo`
  and `mamba/host/gen/philox.mojo`, which `tools/mamba_host_gen.py` generates from
  `core/philox_neural.mojo` and `core/philox.mojo`. That was the only missing piece: SambaStack is
  Python over the training, mamba and transformer bindings.
- Declared: training family training_lanes, exports and host_modules; TRAINING_LANE_NAMES; NO_CPU_PATH
  drops the Samba blocks; docs regenerated; gate workflow paths; tests
  (python/mojolearn/tests/test_cpu_training_samba.py new).
- Evidence (M4, one core, all four families built from 4cc3609e2):
  bench/results/identity_break/2026-09-15_cpu-samba/. IDENTICAL=18 train, 36 infer/model, 18 batch,
  require-columns 4 OK; sabotage set DIVERGENT on every cell; a dropout-mask arm DIVERGENT on the
  dropout lane only.

- Restart (Sep 15 ~11:30 ET): origin/main (450423c95, the routine CPU gate), the transformer branch
  (b5a651dfc) and the mamba branch (082fbb569, then 3cecb4238) merged in with no conflicts. On the M4,
  one core, core, training, mamba and transformer host bindings rebuilt into a fresh directory at
  260a160fc: identity_break --lanes samba,samba-untied-dropout-accum diffed against the 166-lane record
  with --require-columns 4 read IDENTICAL=18 train, IDENTICAL=36 infer/model, IDENTICAL=18 batch,
  require-columns 4 OK (98 s); all four families built with -D MOJOLEARN_HOST_SABOTAGE=1 at 483df954b
  read DIVERGENT=18 train, 36 infer/model, 18 batch. docs_facts --check and wheel_ci pins pass.
  The old gate run 34975751193 ran the pre-routine workflow.

## Running

- Nothing. Pushed with [skip ci]; the routine push gate on this branch runs after lane/cpu-training-transformer
  and lane/cpu-training-mamba merge, one lane gating at a time.

## Next commands

1. When the gate is green AND origin/main carries lane/cpu-training-mamba and
   lane/cpu-training-transformer: `git fetch origin && git merge origin/main`, resolve lists and spans
   (`python3 tools/docs_facts.py --write` then `--check`), rerun
   `python3 packaging/wheel_ci.py pins .`, `python3 packaging/wheel_ci.py inventory python/mojolearn python/mojolearn_diagnostics.py`
   and the host_surface, misc, mamba, transformer and samba test modules, then
   `git push origin HEAD:lane/cpu-training-samba && git push origin HEAD:main`.
2. After merging, move untracked evidence to ~/mojolearn-evidence/cpusamba/ and remove the worktree.

Local scripts (orchestrator scratchpad, cpusamba/): `build_set.sh <fresh dir> "" core training mamba transformer`,
`run_lanes.sh <dir> <tag> samba,samba-untied-dropout-accum`.

No boxes rented.
