# Neural IDENTICAL source experiments

[Experiment and source file index](EXPERIMENT_FILE_INDEX.md) lists every NI card
and recorded variant with its A/B controls, implementation files, and the older
neural experiments found in existing runners and recipes.

[The complete 60-idea catalog](../../docs/plans/NEURAL_IDENTICAL_EXPERIMENT_IDEAS_2026-10-06.md)
defines the A/B hypotheses, numerical contracts, quality risks, full-workload
scope and interaction experiments. It was written before the three implementation
lanes were delegated; six graph-neural/dropout additions were then specified
before their root-owned implementation. This worktree starts from `main` at `fd6cf8045` on branch
`ideas/neural-identical-20261006-v2`.

Bits may change across versions. Within one version, host, NVIDIA, AMD and Apple
must match exactly. A is the candidate; B is each card's reference. All new source
candidates remain disabled by default. Existing runtime defaults are preserved.

[Implementation status](IMPLEMENTATION_STATUS.md) records 41 newly wired
candidate subarms, 15 reused existing A/B paths, two partial cards and two
proposals rejected from source findings. Broader variants remain pending where
recorded. Every count is source status, with qualification pending.

## Lane source records

| Ideas | Scope | Machine readable record | Source handoff |
| --- | --- | --- | --- |
| NI01–NI18 | Neural GEMM and CNN | [gemm_cnn.json](gemm_cnn.json) | [gemm_cnn.md](gemm_cnn.md) |
| NI19–NI36 | Transformer, training and embedding | [transformer_training.json](transformer_training.json) | [transformer_training.md](transformer_training.md) |
| NI37–NI54 | Mamba, Samba and neural sequence | [sequence.json](sequence.json) | [sequence.md](sequence.md) |
| NI55–NI60 | Graph neural message passing and channel dropout | [neural_aux.json](neural_aux.json) | [neural_aux.md](neural_aux.md) |

Each record distinguishes wired candidates, reused existing arms, partially
integrated source, rejected premises, pending variants and toolchain blockers.
None of these source statuses implies compilation, identity, quality or speed.
Known old failures remain failures; a source repair does not erase them.

## Selecting source arms

The metadata-only [selector](../../tools/neural_identical_ideas.py) can describe
individual or combined controls. It has no build, run, check or promotion command.
The commands below are usage examples; they were not executed in this task:

```sh
python3 tools/neural_identical_ideas.py list
python3 tools/neural_identical_ideas.py show NI19
python3 tools/neural_identical_ideas.py plan NI19 --arm A
python3 tools/neural_identical_ideas.py plan NI19 --arm B
python3 tools/neural_identical_ideas.py plan NI08 --arm A --variant NI08=existing_leaf64
python3 tools/neural_identical_ideas.py plan NI38 --arm A --variant NI38=state_window
python3 tools/neural_identical_ideas.py plan NI49 --arm A --variant NI49=row_serial_scan
python3 tools/neural_identical_ideas.py plan NI27 NI28 --arm A --output /tmp/neural-source-selection.json
```

Plans preserve unresolved implementation/dependency work and are never execution
approval or evidence. They reject conflicting controls instead of silently
choosing one. Combining controls is not proof the combination is semantically
valid. Use a clean per-arm worker environment later; inherited experiment toggles
could otherwise contaminate an isolated baseline. A presence-tested Mojo define
set to `0` is still present; use the recorded omission or explicit OFF switch.

`--variant IDEA=NAME` selects only an alternative already recorded by its source
lane. Explicit variant fields replace those primary fields; other fields retain
the primary card's requirements. It does not create an implementation, override
pending integration, or erase historical losing evidence.

NI08/I04 GEMM arms also expose their exact numerical profile through both native
linalg bindings. The public API shell retains the default v1 requirement; an
experimental linalg caller needs the exact `MOJOLEARN_EXPERIMENT_GEMM_PROFILE`
value recorded with its arm. Unknown/mismatched profiles refuse. Experimental
profile metadata does not inherit v1's certification or claim qualified identity.
This is metadata plumbing only, with no Python tensor work or execution here.

## Integration with existing A/B tools

The same live catalog is reachable through both existing entrypoints:

```sh
python3 tools/performance_ideas.py neural list
python3 tools/neural_experiments.py ideas show NI20
```

[integration.json](integration.json) maps every idea to affected binding families,
host companions, public operation groups and saved workload recipe sources.
The map is conservative for shared GEMM: its neural consumers must be qualified
together. Statistical algorithms sharing a binding are outside this neural
campaign. Exact dataset hashes, settings, actual dimensions and cap audits remain
pending; the map does not claim those facts have been measured or verified.

`build-plan` emits separate A/B commands for the existing
`tools/identical_wave_native_build.py`. Its repeatable `--neural-idea` and
`--neural-variant` controls resolve the lane records and required bindings at the
specified frozen commit. `--recipe-role` selects candidate or baseline; the builder
records definitions, runtime environment and source-selection provenance. It
refuses mixed arbitrary flags or a blanket ALL_OFF reference. The same defines
must reach every affected GPU and host binding. Apple M2 identity build scripts
are listed in the plan; the native GPU builder itself remains NVIDIA/AMD Linux.

Example for a later frozen round, with the worker paths and actual target supplied
by its controller (these commands were not executed):

```sh
python3 tools/performance_ideas.py neural build-plan NI20 \
  --source-sha "$NEURAL_FROZEN_SHA" --vendor nvidia --gpu-arch "$NEURAL_GPU_TARGET" \
  --repo /worker/mojolearn --artifacts /worker/neural-ab \
  --python /worker/venv/bin/python --output /tmp/neural-build-plan.json
python3 tools/performance_ideas.py neural queue-template NI20 \
  --source-sha "$NEURAL_FROZEN_SHA" --vendor nvidia --repo /worker/mojolearn \
  --output /tmp/neural-full-ab.json
```

`queue-template` emits one visibly blocked job per affected workload for
`tools/performance_full_ab_queue.py`. Fill the real full-workload recipes,
accepted artifacts, complete adapter commands and evidence before a later run.
The queue resolves `neural_selection` against its frozen source, checks declared
artifact controls and quality evidence, and supplies the selected runtime
environment to both arms in fresh processes. It removes inherited MOJOLEARN
settings before applying recorded worker and arm settings. Output and captured
model-state bits must match A/B for S-only combinations; V combinations compare
bits across columns within each version instead. The old stage diagnostic's
unchanged-bit contract remains intact.

The neural queue's `quality_evidence.path` references a retained JSON receipt:
`status: PASS`, matching `source_sha`, `dataset_sha256`, `workload_id`,
`dimensions`, `estimator_settings`, `neural_selection`, `artifact_provenance`,
the selected `contracts`, `task_quality: PASS`, and `same_version_identity`
mapping **each of A and B** to host/NVIDIA/AMD/Apple PASS outcomes. These fields
must come from actual accepted evidence. A completed build cannot create them.
Each neural artifact record also names its `binding_family` and exact defines;
the existing full-operation queue checks binary hashes and actual loaded artifacts.
Shared support libraries may be listed separately. No such receipt was created
or admitted during this source task.

Relevant combinations use multiple idea IDs in either command. Conflicting flags
or mutually exclusive variants refuse; a selectable combination is still unproven.
Source references and historical failure records remain attached to every plan.

Do not use the older `tools/neural_experiments.py` changed-digest rejection to
reject a declared new arithmetic version. That diagnostic runner retains its
original same-bits contract; this catalog does not relax its safeguards.

## Qualification remains pending

No compiler, test, linter, syntax checker, manifest validator, candidate workload,
benchmark or verification command was run for this source work. The selector
itself is also unexecuted. No binaries or qualification receipts were produced;
no boards, defaults, existing frozen runs or `main` were changed. The owner
subsequently requested integration plus commit/push of this experiment branch.
Normal Git policy hooks accompany that delivery; they do not qualify kernels.

[workload_requirements.json](workload_requirements.json) records the full-operation
mapping requirements. Missing recipes/hashes/actual dimensions stay pending.
Later performance evaluation must cover every affected full estimator and the
relevant combinations, retaining both vendor results and quality/identity evidence.
Reuse accepted evidence; do not repeat completed validation merely to time a new
candidate. No speedup or quality-preservation result is claimed here.
