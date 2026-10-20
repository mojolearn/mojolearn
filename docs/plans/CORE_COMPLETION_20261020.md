# Core completion integration — 2026-10-20

Internal implementation and evidence record. This branch is not release
qualification. The integration branch is `finish/core-completion-20261020`;
merge to `main` remains pending the relevant correctness and runtime checks.
Public documentation must not advertise the unqualified additions.

## Compilation policy

Freeze committed source and rebuild only changed native dependency closures.
Reuse unchanged numerical artifacts with their source and binary provenance.
Changes to tests, annotations or documentation alone do not require a native
rebuild. A new import into a shared binding does change that binding's closure.
CPU, Metal, CUDA and HIP are distinct compilation targets; a pass on one does
not qualify the others. Source overlays used for incremental testing are not
installed-wheel evidence.

## Retained evidence and unresolved findings

All paths below are relative to
`~/mojolearn-evidence/core-completion-20261020/`, unless stated otherwise.
Groups overlap and use explicit source freezes; their counts must not be added
into a single suite result.

| Work | Evidence | Interpretation |
|---|---|---|
| Generation and model loading | `validation-fb5c59d7420d8042e27b6a7c6d2d5da554f7af5d/metal-generation.xml`; matching current CPU/Metal generation reports in the same directory | 90 Metal tests pass. Four generation cases with 44 audited calls match CPU/Metal; source qualification only. |
| Causal-model parameter inventory | `stage21-causal-model.xml` | 19 tests pass, including the repaired optional and fused-parameter inventories. |
| Native transport, process workers and block backward | `stage28-native-tests.xml`, `stage28-tests-tmp/` | 43 of 44 tests pass; the remaining test used default options while expecting nondefault refusal. The corrected fixture passes separately against the same binary in `backward-refusal-supplement/`. Child audit reports retain the complete child-loop boundary. |
| Cross-vendor worker protocol on Apple | `stage28-worker-metal.xml` | Both chained and unchained protocol tests pass with independent numerical oracles and native audits. One physical GPU; no general multi-device qualification implied. |
| Native block construction | `stage28-neural-block.xml` | 11 tests pass, including Mamba3 construction. |
| Graph optimizer and diagnostics | `validation-graph-stage27/{cpu,metal}.json` | CPU: 19/20 cases, 85 digests. Metal: 18/20 cases, 77 digests. Lifetime fails on both. Metal also exposed a missing raw-word flag in the tiled forward transpose route; its separate repair requires fresh affected-case evidence. |
| Native MLP and Samba state | `stage30-model-state-summary.json` | First run: 18 pass, 14 fail, 2 GPU-only skips. Twelve MLP cases encounter Python attribute interception; two Samba cases encounter NumPy preparation. These are product defects to repair, not audit exemptions. |
| Installed AMD graph execution | `retrieved/amd-installed/graph-stage15-tierfix/` | 13 cases and 36 digests match the corresponding CPU reference on this earlier freeze. This does not qualify subsequent graph source. |
| PTX references and four weighted-tree batch checks | `apple-0855-seven-lanes.json`, `cpu/seven-lanes.json`, `retrieved/{amd,ptx}/` | Current CPU, Apple, AMD and PTX results agree for the selected checks. Independent native NVIDIA evidence is still missing; the four original PTX reference mismatches remain unadmitted. Compatible rental attempts have returned out of stock. |

## Supported-toolchain blocker

The standalone [PythonObject ownership investigation](../../experiments/python_object_ownership/README.md)
reproduces retained references using only the supported Mojo 1.0 standard
library, without importing mojolearn. The same compiled extension reproduces
the failures under CPython 3.12 and 3.14. Model-state and Session lifetime tests
remain failures. No compiler patches, unsupported build mode or compensating
reference-count operations are used. This requires a supported upstream fix.
Passing numerical tests or audits that exclude lifetime does not remove it.

The full reproducer, binary and results are retained locally in
`stdlib-ownership/` and in R2 at
`measurements/core-completion/2026-10-20/mojo-1.0-ownership/stdlib-ownership.tgz`.

## Work still required

- Finish native training owners and repair preparation/attribute paths found
  by whole-operation audits. Arbitrary Python callbacks and custom schedule
  execution remain explicit debt; an unchanged source-checker baseline is
  not permission to execute them.
- Qualify conditional training and checkpoint continuation, broader operation
  and derivative coverage, and each supported distributed partition plan.
  Physical multi-GPU execution and installed-wheel checks remain distinct
  from single-device source tests.
- Continue the [graph optimizer](GRAPH_OPTIMIZATION.md) beyond its present
  structural passes where supported. It remains opt-in. No fusion, general
  storage-recycling policy, performance improvement or default promotion is
  claimed by the current correctness evidence.
- Complete native NVIDIA reference evidence, then use the reference and board
  tools to admit valid results. Preserve the original failures. Do not replace
  a native witness with agreement between PTX and another backend.
- Rebuild and qualify other affected shared native families before any whole
  release claim. Retain unchanged accepted compilation/identity evidence.
  Full-workload NVIDIA/AMD A/B is required for performance/default decisions.

## Rental retention

The owned AMD machine `610973312` was archived, deleted and verified absent
(provider HTTP 404). Its final archive is
`~/mojolearn-evidence/box-archive/do-amd-610973312-20261020T2143Z/archive-lq.tgz`
and R2
`measurements/box-archive/do-amd-610973312/do-amd-610973312-20261020T2143Z.tgz`.
Accepted binaries excluded from the general archive were retained separately
with their build provenance. The shared NVIDIA machine belongs to multiple
sessions and must not be deleted by this lane.
