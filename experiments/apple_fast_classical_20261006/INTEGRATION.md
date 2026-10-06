# Apple FAST classical source integration

The initial source delivery wired candidate controls into existing production
callers but used a standalone configuration selector. This follow-up registers
all 54 AFCL cards with the common performance experiment workflow, adds paired
package construction and full-workload adapters, and keeps controls OFF.

## One catalog and one set of arm controls

`tools/performance_ideas.py` recognizes 114 cards: the original 60 plus 54 AFCL
cards. Each AFCL has a registration under `experiments/performance_ideas/<ID>/`.
The registration reads its actual source paths, compiler defines and runtime
environment from the original lane JSON. Both the standalone selector and the
shared runner therefore use the same control definition.

The common `list`, `plan`, `execute` and explicit `check` subcommands recognize
the registrations. None of those commands was run as part of this integration.
`source_ready` does not mean built, verified, measured, quality-qualified or
promoted. Original F01–F20 relationships are retained as `related_ideas`, which
are references to prior mechanisms, not inherited acceptance or compile evidence.

## Production and package integration

[INTEGRATION_SCOPE.md](INTEGRATION_SCOPE.md) records the source call paths,
existing ABI exports, prerequisites and excluded routes. Existing kernels are
selected through their production launchers; no new Python numerical runtime
or public estimator API is needed to expose the experimental schedules.

[build_pair.py](build_pair.py) constructs isolated complete classical packages
for A and B (or explicitly selected interactions), applying the requested
compile-time flags to every affected binding. Unchanged dependencies are shared
only with matching source, compiler, mode, target and define provenance. The
IDENTICAL core transport helper remains an unchanged dependency in both arms.
[build_bindings.json](build_bindings.json) records module coverage; the build
handoff explains its conservative closure and artifact layout. Per-arm runtime
settings are retained with binary hashes. Existing compile-slot control and
build-only suppression of device smoke work apply to future builds.

A build plan can be emitted later through the common tool, for example:

```sh
python3 tools/performance_ideas.py plan AFCL-L01 --stage build --vendor apple --output /tmp/afcl-l01-pair
```

This example is documentation; it was not executed. The source work does not
schedule builds, provision a machine or mutate another active frozen run.

## Complete workload integration

[full_workloads.json](full_workloads.json) relates every card to saved classical
workloads, actual affected callers and known route/row-cap limitations.
[run_pair.py](run_pair.py) uses isolated arm packages and audited inputs to reuse
those full-operation workloads. [WORKLOAD_INTEGRATION.md](WORKLOAD_INTEGRATION.md)
describes the audit configuration, saved recipe coverage and operation boundaries.

The common device-stage interface takes `--artifacts DIR` and `--workloads FILE`.
The latter must record the real data identity/hash, actual input dimensions,
estimator settings, numeric mode, full-coverage audit and required quality-floor
evidence. Existing intrinsic subsampling stays visible; `--rows full` alone is
insufficient. Unmapped or unreachable caller recipes stay pending. A recipe that
never calls the changed path cannot qualify it, even if its entire process exits
successfully. Separate mean-only GP prediction does not establish variance-path
coverage, for example.

Future A/B execution preserves preparation, fit, synchronization and output
consumption inside the declared whole-operation boundary and reports distinct
fit/inference/cold/repeated scopes. Work runs serially per GPU with one excluded
warmup and one scored sample per arm. Task-quality and API/route evidence are
explicit gates; missing evidence remains pending, never inferred from an exit
code or copied metadata. Timing admission also requires matching hashes of the
audited workload configuration and paired-build manifest.

The runner writes per-card quality receipts, with a top-level receipt for a
single-card run. The common runner passes an admitted receipt to the timing
stage. `--attach-evidence-to` can attach accepted evidence for the same frozen
source, packages and workloads to retained results in a fresh output directory,
without rerunning device work or overwriting the original attempt. Missing
route/API evidence and unresolved workload scope remain pending.

## Interactions and board inputs

The selector emits factorial descriptions for up to six interacting cards and
retains define-absence constraints. G01 and G02 cannot form one combined route:
G02 requires disabling the MMA path whose geometry G01 changes. Other recorded
prerequisites are equal in both arms. Successful isolated cases still do not
qualify the complete proposed default configuration.

`measurement_inventory.json` and `measurement_index.json` are source inputs for
`tools/performance_measurement_board.py`. The index contains no measurements or
decisions. The board tool now accepts explicit campaign evidence/identity policy
so a future AFCL render cannot label this uncompiled campaign as previously
validated. No board was rendered or measurement value updated in this task.

`export_measurements.py` adapts a retained `run_pair.py` result into a fresh index
for that existing board tool. It retains failed/unqualified attempts, source,
input/build hashes and evidence paths. It does not fit an opponent, invent an
opponent ratio, render a board or change a default. Only a single-card scored
run with matched source/machine/artifacts, complete warmup/sample structure and
all genuine quality gates can be exported as MEASURED. Multi-card runs remain
explicit interaction evidence. Existing indices and result files are never
overwritten; repeated imports keep original attempts distinct by result hash.
The exporter itself was not run.

## Execution status

All integration is source-only. No compiler, binding build, selector, manifest
checker, test, validation driver, benchmark or GPU job was executed. Existing
catalog-count assertions were updated as source; tests were not run. New controls
remain OFF. Commit/push uses hooks disabled and a CI-skip marker.
