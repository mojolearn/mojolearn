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
| Graph optimizer and diagnostics | `validation-graph-stage27/{cpu,metal}.json`; `stage30-targeted-graph-comparison.json` | The initial CPU run passed 19/20 cases and Metal 18/20. Lifetime fails on both. The tiled forward transpose repair subsequently passes all seven affected cases on CPU and Metal, with 56 matching digests. These overlapping incremental runs are not a full qualification pass. |
| Conditional graph training and continuation | `stage31-targeted-control-comparison.json`, `stage31-ordinary-graph.xml` | Five conditional/control/checkpoint cases pass on CPU and Metal with 34 matching digests; 11 ordinary Program tests pass without skips. Installed-wheel and physical NVIDIA/AMD checks remain pending. |
| Native MLP and Samba state | `stage30-model-state-summary.json` | First run: 18 pass, 14 fail, 2 GPU-only skips. Twelve MLP cases encounter Python attribute interception; two Samba cases encounter NumPy preparation. These are product defects to repair, not audit exemptions. |
| Native model and public parallel-training integration | `stage32-native-models-summary.json`, `stage32-native-models-tmp/` | After those repairs, 46 of 48 CPU tests pass with no skips. Both remaining failures concern restored Samba attention blocks inheriting the wrong normalization default. The separate default repair retains the original failures and needs an affected-case rerun. Actual MLP child loops, rollback and checkpoint replay pass their strict native audits. |
| Samba gradient, generator state and process admission repairs | `stage34-repairs-{cpu,metal}-summary.json`, corresponding logs and child audit directories | CPU: 41 pass without skips. Metal: 40 pass and one CPU-only refusal check skips. Repairs cover restored attention normalization, ignored-target buffer lifetime, exhausted RNG stream checkpoint continuation, and worker vendor admission. The earlier stage32/33 failures remain retained. |
| Complete current graph verifier on CPU and Metal | `stage34-full-graph-{cpu,metal}.json`, `stage34-full-graph-comparison.json` | Each run passes 22 of 23 cases; all 106 output digests from passing cases match. The sole failure on each is the upstream Session ownership issue. Both full runs remain failed qualification reports. Unchanged sequence binding binaries from stage31 are reused. |
| Native Samba constructor and standard training step | `stage35-constructor-cpu-summary.json`, `freeze-694269a1b2b96ac5561caf24b96fcd634961ffb2/tokenizer-host.log` | Constructor integration initially passes 23 of 34 CPU tests; mixed-model failures expose selection of inference-only Mamba blocks. A separate step build finds an invalid consume of a borrowed argument. Both have source repairs awaiting validation. The private standard-step owner does not remove the public custom-callback contract debt. |
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
with their build provenance. The shared NVIDIA machine was later retired by its
idle lease; the provider confirms it is gone. This lane did not delete it.
Its accepted binaries and our required results were retained before retirement.

A second owned AMD lease, `611011332`, started around 22:44 UTC with a
60-minute deadline at $3.80/hour. Its task is changed-binding compilation and
installed correctness checks, not performance tuning. State is tracked in
`amd-state/`; result and binary retention must finish before its teardown.

A snapshot of 653 complete retained logs, reports and controller artifacts is
also in R2 at
`measurements/core-completion/2026-10-20/session-record/results-20261020T2222Z.tgz`.
That snapshot is an incremental record, not an all-green release result.
