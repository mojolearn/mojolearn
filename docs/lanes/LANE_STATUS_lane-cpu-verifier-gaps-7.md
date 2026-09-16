# Lane status: lane/cpu-verifier-gaps-7 (2026-09-15)

Seven one-device identity lanes had a CPU host function and no gate wiring:
gmm-sample, gmm-random-init-sample, gp-sample-y, gp-sample-y-normalize,
tokenizer, gbdt-categorical-ctr-tables and gbdt-tensor-ctr-tables. This lane
declares them in `python/mojolearn/host_surface.py`, so the full CPU identity
gate (`FULL_CPU_VERIFY`) runs every one-device lane.

## What changed

- Manifest: the mixture family covers the two sample lanes, gp the two
  sample_y lanes, tokenizer the tokenizer lane, and forest the two CTR table
  lanes (their CPU cells are the forest binding's predictions from Metal-saved
  models; CPU training of CTR tables still refuses by name). New:
  `GBDT_CTR_MODELS_DIR` (`--gbdt-ctr-models`), `GATE_SABOTAGE_OWN_DEFINES` and
  `sabotage_build_defines()` (`--sabotage-build-defines FAMILY`).
- Gate workflow: checks out the models directory and exports
  `MOJOLEARN_IDENTITY_GBDT_CTR_MODELS`; builds each family of the sabotage
  host set with the manifest's defines (tokenizer with
  `MOJOLEARN_TOKENIZER_HOST_SABOTAGE`, forest with the CTR arm
  `MOJOLEARN_GBDT_CTR_HOST_SABOTAGE`, both beside `MOJOLEARN_HOST_SABOTAGE`);
  the covered sabotage run points `MOJOLEARN_FOREST_HOST_BINARY` at host-sab's
  forest binding.
- identity_break: a CPU column that loads a GPU-saved CTR model reports the
  model part `n/a:gpu-saved-file` (the file is the GPU column's bytes; no
  sabotage could move its hash and the owed check would fail on it). A second
  load that predicts differently is refused.
- Models: the committed directory had base, ties and odd only. The six other
  fixtures were fitted on the Apple M4 Metal column with the CTR lane's own
  HEAD Metal build (1386833b4), first validated on base: the saved npz bytes
  and the train, infer, model and batch hashes equal the committed ones for
  both lanes. Evidence: `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/metal-ctr/`.
  About 25 s per categorical fixture.

## Done

- [x] manifest, workflow, identity_break, tests (branch pushed at 11c5f2192;
  its message says test_host_surface 153 passed, the run read 149 passed, and
  test_cpu_identity_gate 26 OK)
- [x] Metal models for the six missing fixtures
- [x] RunPod CPU pod 03krlhu4sqjw8e (deleted and verified gone, $0.0153): the
  gate's covered-lanes path for the seven lanes. Production 63 cells STABLE,
  column check 0 failures, `--require-columns 4 --owed-json` OK with 189 OWED
  parts and nothing DIVERGENT; sabotage column still STABLE and the owed check
  189 of 189 moved. Evidence
  `bench/results/identity_break/2026-09-15_cpu-verifier-gaps-7/`.
- [ ] merge to main

## Resume

Worktree `scratchpad/wt-cpu-gaps7`, origin/main merged (the one conflict was
the end of `tools/test_cpu_identity_gate.py`, where main and this lane each
appended a test class; both kept). Pod command: the evidence's `cmd.sh`.
