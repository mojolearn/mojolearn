# Lane status: lane/cpu-training-samba (CPU training for samba and samba-untied-dropout-accum)

Updated 2026-09-15 by the cpumamba13 agent (re-pointed to Samba by the orchestrator).

## Base

- Branch from origin/lane/cpu-training-mamba (the mamba host family, gate owed) with
  origin/lane/cpu-training-transformer merged in (the transformer host family, gate 34969598898).
  Neither is on main yet. Merge origin/main into this branch as they land.

## Done

- `neural_rng` added to `bindings/_mojolearn_training_host.mojo`, over
  `mamba/host/gen/philox_neural.mojo` and `mamba/host/gen/philox.mojo`, which
  `tools/mamba_host_gen.py` now generates from `core/philox_neural.mojo` and `core/philox.mojo`.
  That was the only missing piece: SambaStack is Python over the training, mamba and transformer bindings.
- On the M4 (one core, shared machine), training host built from this tree and the mamba, core
  and transformer host bindings copied from the peers' builds: samba and samba-untied-dropout-accum,
  nine fixtures, two repeats, read `summary: IDENTICAL=18`, `summary (infer/model): IDENTICAL=36`,
  `summary (batch): IDENTICAL=18`, `require-columns 4 ... OK` against TRAINING_GPU_COLUMNS.

## Owed, in order

1. Sabotage build of the four families (`-D MOJOLEARN_HOST_SABOTAGE=1`) must read DIVERGENT; a
   throwaway RNG-only sabotage must also move the samba cells.
2. Rebuild core, training, mamba and transformer host bindings from this tree into a fresh dir and
   rerun both lanes for the committed evidence (bench/results/identity_break/2026-09-15_cpu-samba/).
3. Declare: host_surface training family (exports gain `neural_rng`, host_modules gain the two
   philox gens, training_lanes gain the two samba lanes), NO_CPU_PATH drops "the Samba blocks",
   `python3 tools/docs_facts.py --write`, test_cpu_training_misc follows, new test module, gate
   workflow paths for core/philox*.mojo.
4. Push, seven-runner CPU identity gate, merge to main after the mamba and transformer branches land.

## Commands (scripts in the orchestrator scratchpad, cpusamba/)

    SP=<orchestrator scratchpad>
    $SP/cpusamba/build_set.sh $SP/cpusamba/<fresh dir> "" core training mamba transformer
    $SP/cpusamba/run_lanes.sh $SP/cpusamba/<dir> <tag> samba,samba-untied-dropout-accum
    # diff: python3 tools/identity_break.py --diff $(python3 python/mojolearn/host_surface.py --training-gpu-columns) <cpu json> --require-columns 4 --lanes samba,samba-untied-dropout-accum

No boxes rented.
