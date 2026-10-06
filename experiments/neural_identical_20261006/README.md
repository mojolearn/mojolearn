# NEURAL IDENTICAL source campaign

**NOT TESTED — NOT COMPILED — NOT MEASURED.** The owner requested programming
only. Neither the source nor this harness has been executed or verified.

The [64-card inventory](../../docs/plans/NEURAL_IDENTICAL_EXPERIMENTS_2026-10-06.md)
defines the A/B hypotheses, workload reach and numerical/quality obligations.
`catalog.json` is the complete idea list. `lanes/*.json` records implementation
scope card by card. A reused arm is not a new kernel. A component API without
caller wiring is not a complete experiment; its blocker remains explicit.

`source_implemented_unverified` means only that source was written. Read each
card's limitations: that status may cover one sub-arm, a component adapter or
an explicit combination of controls. It never means every proposal in the
card has complete caller integration. `existing_arm_unverified` means source
was reused, with no new campaign qualification. Blocked cards are retained in
the inventory; no dummy defines or claimed implementations replace their asks.

New controls have an INLINE `NOT TESTED — NOT COMPILED — NOT MEASURED` comment
and are OFF by default. Existing controls keep their historical defaults and
evidence; recording them here makes no new claim. Numerical profiles may
change old-version bits, but the candidate must agree across NVIDIA, AMD,
Apple and the host within its own version. Quality is a separate obligation.

`tools/neural_identical_ab.py` provides metadata listing, isolated A/B plans,
explicit variants/combinations and a future runner for already built artifacts.
It does not build anything. No command in this document was executed here.

Future metadata-only examples (not run):

```sh
python3 tools/neural_identical_ab.py list
python3 tools/neural_identical_ab.py plan G01:amd_mfma16 --source-sha FROZEN_SHA --output /tmp/ni-g01-plan.json
python3 tools/neural_identical_ab.py plan A03 T05 --source-sha FROZEN_SHA --output /tmp/ni-combination-plan.json
```

Plan generation preserves unresolved source prerequisites. The future `run`
entry refuses those plans and requires a separate full-workload recipe plus
retained artifacts for the exact source/defines/compiler/target. It retains
complete logs and process exit codes; process duration is never a speed score.
The actual driver must own the excluded warmup, restored initial state,
scored whole operation, synchronization, consumed outputs and receipt.
There is no automatic promotion, board update or global identity claim.

The separate `tools/neural_identical_compare.py` source describes future
receipt comparison; [RECEIPT_FORMAT.md](RECEIPT_FORMAT.md) specifies its
evidence schema. That comparer was also not executed in this task.

## Future driver contract

The recipe must supply `source_sha`, `vendor`, `mode: identical`,
`coverage_status: full_workload_attested`, `sampling` with
`excluded_warmups: 1` and `scored_samples: 1`, `dataset` (name, version, hash,
split, intended extent and actual extent), actual dimensions, intrinsic caps,
estimator settings, operation, initial-state and input-manifest hashes,
complete timed boundary, quality gates and worker resource policy.

For synthetic-only lanes the intended extent is the complete declared fixture.
LM/Samba corpus claims need actual corpus coverage, not merely `--shape full`.
The existing host board's length cap is not an uncapped identity witness.
Missing mappings stay pending. Recipes must cover every affected estimator,
neighboring shapes and a non-board dataset when a route rule changes.

Each of the `A` and `B` entries under `arms` supplies a string-array `argv`,
optional cwd/environment and an `artifact` with source SHA, `build_status:
PASS`, vendor, mode, exact defines, compiler, target, retained accepted-build
receipt path and every artifact file's path/hash. Column/target build flags
belong in the retained build receipt; the `defines` list is the experiment
define set from the plan. Binding installation and driver setup must not
implicitly rebuild. Source/build provenance is attested, not inferred from a
filename. Reuse already accepted exact artifacts; no compile command is added.

The driver writes JSON at `$MOJOLEARN_NEURAL_AB_RECEIPT`. It must include
actual worker hardware/cgroup/thread pools, source and artifact identities,
input/settings/state hashes, selected profile and reached routes, executed
extent/caps, warmup and scored-sample counts, cold/fit/inference/repeated
operation times with synchronization/consumption, output/gradient/model-state
witness hashes, quality metric values and gate outcomes, refusals, skipped
coverage and failures. The parent only retains this receipt; it does not
interpret a process exit or self-reported PASS as global qualification.

Compare witness sets **A NVIDIA = A AMD = A Apple = A host**, and separately
the B set. A may differ from B for a named numerical revision. S-class arms
also owe the declared baseline-arithmetic equivalence. Witnesses must have
the same logical input, shape, model settings and initial state. Forward,
gradient, optimizer-state, checkpoint and refusal coverage are distinct.
Apply the declared quality gates independently of hashes. NVIDIA and AMD
jointly decide timing; Apple is an identity column only. Never manufacture
opponent ratios from own-only A/B results.

All execution, compilation, verification, quality qualification and measurement
remain future work. The harness itself is untested source.
