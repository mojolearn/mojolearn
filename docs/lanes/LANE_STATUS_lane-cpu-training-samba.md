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

## Running

- The seven-runner CPU identity gate on this branch (push of the merge with main). Check with
  `gh run list --branch lane/cpu-training-samba -L 5`.

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
