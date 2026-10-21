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
| Public Samba training, inference and checkpoints | `stage49-native-{cpu,metal}-summary.json` and full logs; `stage49-test-import-repair.json` | The actual public methods now pass the exercised native audits: CPU has 10 training, 12 inference and 12 checkpoint passes plus two GPU-only skips; Metal has 12 passes in each group. Original public-wrapper Python-frame audit and test collection failure remain retained. These tests do not certify custom protocol admission across the whole library; that audit continues. |
| First-order and higher-rank derivative repairs | `stage47-graph-comparison.json`, `stage49-graph-comparison.json`, `stage49-native-{cpu,metal}.xml` | Seeded VJP output ranks and recorded higher derivatives are repaired. Each current full verifier passes 29 of 30 cases, with 163 matching CPU/Metal digests and no mismatch. The sole Session ownership failure remains. Thirteen new rank/direction cases and eight existing higher-derivative cases pass on both devices. |
| Native Sequential input compatibility | `stage49-native-{cpu,metal}-summary.json`, `stage50-build-controller.log` | Both stage49 runs retain seven failures: half/uint64 labels need native label-engine support, and one broadcast test oracle needs its original canonical input boundary. The tests remain; source repairs and additional exhaustive half/range tests are in the next frozen build. |
| Standalone native-only release gate | `native-only-cpu-round2/summary.json`, `native-only-controller-tests-v3.log`, [design](NATIVE_ONLY_RELEASE.md) | Four standalone CPU cases compile and pass locally, exercising tape derivatives, Sequential fit/predict, optimizers and tokenizer training/round-trip. The controller has 11 passing negative-control tests. Twenty-eight fixtures are inventoried; missing public coverage and CPython-bound owners remain explicit. Local execution is dependency screening, not interpreter-free Linux qualification or a complete standalone library. |
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
| Native Samba constructor and standard training step | `stage37-samba-{cpu,metal}-summary.json`; stage35/36 original failure logs | The repaired constructor, gradient/state handling, private standard step and parallel neural checks pass 46 cases on CPU and 46 on Metal. This does not migrate the remaining public Samba forward/state/training/checkpoint wrappers or remove custom-callback debt. |
| Parallel causal generation and optimizer finalization | `stage39-repaired-{cpu,metal}-summary.json`, `validation-stage39-repaired/closure-reuse.json` | Repaired CPU run: 29 pass, two skip; Metal: 30 pass, one CPU-only skip. Real child processes, continuation bytes, position metadata and Apple mode admission are checked. One physical GPU, with sequential layer handoff; no overlapping-pipeline or complete lifetime claim. |
| Extended graph passes and control-state validation | `stage41-graph-comparison.json`, `retrieved/amd-round2-stage41/three-vendor-comparison.json` | CPU, Metal and the installed AMD overlay each pass 26 of 27 cases; all 128 passing output digests agree. Every full verifier run remains failed because of Session ownership. Includes opt-in ReLU motion, integer loop-state refusals and native distributed boundaries; no performance claim. |
| Latest installed AMD boundary suite | `retrieved/amd-round2-stage41/native-repairs-installed-summary.json` | 89 of 109 pass, no skips and no product modules imported from source. The same 20 failures occur on CPU and Metal: wrapper dictionaries, RNG/fixture compatibility and context-exit return values. Separate fixes preserve the original reports. These qualification wheels combine declared native overlays and reused artifacts; they are not release wheels. |
| Partial native setup failures | `setup-fault-35913a657866b76ac2154f4ac250f804254bfb81/`; `distributed-setup-fault-f8a0031409cc11228669f8b4f5c745440ae41107/` | All 32 graph setup/cast fault cases and all 21 distributed setup fault cases pass independently. Checks include allocation counts, original-error preservation and successful recovery. This does not remove the separate upstream PythonObject ownership defect. |
| Distributed detached gradients and lifecycle boundaries | `stage42-distributed-{cpu,metal}-summary.json` | Ten independent cases pass on each of CPU and Metal after preserving the original `None` context-exit result. Physical multi-GPU and current-source Linux qualification remain separate. |
| Public Sequential and Generator boundaries | `stage45-native-{cpu,metal}-summary.json`, `validation-stage45/closure-reuse.json`; original stage41/43/44 failures retained | CPU and Metal each pass 111 cases with no skips: 42 Sequential configuration/training/prediction cases and 69 Generator draw/initializer cases. Checks include resume, validation/best-state, layouts, seed/stream consumption and whole-operation audits. Sequential checkpoint/import migration and current-source Linux qualification remain pending; no complete native-surface claim. |
| Installed AMD graph execution | `retrieved/amd-installed/graph-stage15-tierfix/` | 13 cases and 36 digests match the corresponding CPU reference on this earlier freeze. This does not qualify subsequent graph source. |
| PTX references and four weighted-tree batch checks | `apple-0855-seven-lanes.json`, `cpu/seven-lanes.json`, `retrieved/{amd,ptx}/` | Current CPU, Apple, AMD and PTX results agree for the selected checks. Independent native NVIDIA evidence is still missing; the four original PTX reference mismatches remain unadmitted. Compatible rental attempts returned out of stock; the later attempt also hit the shared pod cap. No other session's pod was stopped or claimed. |

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

- Finish public input-protocol and transitive callback audits, including native
  callers that can invoke custom Python descriptors, conversion methods or path
  hooks. Remove all callback execution, preserve standard native buffer behavior
  and provide native scheduling/checkpoint configuration. Public Samba operations
  now have the bounded stage49 evidence above; broader admission and separate
  CPU SambaInference checks are still pending. Checker baselines are debt, never
  exemptions. The no-callback rule is explicit in AGENTS.md and project memory.
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

The second owned AMD lease, `611011332`, was deleted before the requested
23:44 UTC deadline: DELETE returned HTTP 204 and a subsequent GET returned
404. It performed changed-binding compilation and installed correctness
checks, with no performance tuning or lease extension.
`amd-stage41-teardown-retention-proof.json` and `amd-stage41-down.log` record
the ownership check, no-active-work confirmation and deletion. The verified
23:26 full archive, final 33-file incremental archive, and separately retained
source/binary/wheel/result bundles cover 128,181 files with actual bytes on
the Mac and in R2. A redundant full rearchive was cancelled in favor of that verified
incremental composition; deletion waited for complete coverage and R2 byte
verification. The partial rearchive log is retained.

A snapshot of 653 complete retained logs, reports and controller artifacts is
also in R2 at
`measurements/core-completion/2026-10-20/session-record/results-20261020T2222Z.tgz`.
That snapshot is an incremental record, not an all-green release result.
