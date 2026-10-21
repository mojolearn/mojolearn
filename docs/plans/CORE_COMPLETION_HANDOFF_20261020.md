# Core completion handoff — 2026-10-20

Andrew asked to finish the changes already in progress, stop opening new work,
and leave the broader work in this handoff. This is an internal engineering
record, not release qualification or public feature documentation.

## Where to resume

- Integration branch: `finish/core-completion-20261020`, pushed to
  [GitHub](https://github.com/mojolearn/mojolearn/tree/finish/core-completion-20261020).
  Local worktree: `/Users/andrewhendel/mojolearn-wt/core-completion`.
- Main worktree: `/Users/andrewhendel/CascadeProjects/mojolearn`. Main has the
  standalone gate and qualified incremental fixtures, including SVC/SVR.
  The broad runtime migration remains on the integration branch.
- Evidence root, abbreviated **E** below:
  `/Users/andrewhendel/mojolearn-evidence/core-completion-20261020`.
- Lane evidence: sibling directories `finish-generation-public-classical-20261020`,
  `finish-graph-20261020`, and `finish-distributed-20261020`.
- Detailed earlier implementation history is in the integration branch's
  `docs/plans/CORE_COMPLETION_20261020.md`. Do not use its older counts as the
  latest qualification result; use the frozen evidence below.

## Contract that must remain in force

No Python execution, including callbacks from Mojo/native code, on product
paths. Audit actual public constructors, first calls, fit/training loops,
attributes and consumed results, including transitive input/output hooks.
Private owner tests do not qualify unchanged public wrappers. A CPython
binding that calls native C APIs is still distinct from an interpreter-free
artifact. The rule is in `AGENTS.md` and project memory.

Preserve supported interfaces and ordinary input behavior. Refuse unsupported
custom Python hooks before invoking them; do not delete supported features to
claim compliance. Do not patch the toolchain or compensate reference counts
for the reproduced upstream Mojo ownership problem.

Bitwise results may change between versions. Within one version, identical
mode must match Apple, NVIDIA and AMD. The optimizer stays opt-in until the
required full-workload NVIDIA/AMD A/B, quality and cross-vendor correctness
evidence supports a default change. No performance improvement is claimed by
the tests in this handoff.

## What the simple GitHub check does

`.github/workflows/native-only-smoke.yml` already builds standalone native
callers, rejects Python dependencies, and runs them in a Linux filesystem with
no interpreter, Python library, packages or source checkout. It uses two
parallel builds and a ten-second execution budget per case; compilation has a
separate budget. Python may set up the compiler and control the test.

This is the requested simple build/run check. Passing callers prove only
their reached native paths. Missing algorithm coverage and Python callbacks
in the separate public package require their own checks. Main has 37 fixtures;
the integration inventory has 41. These are smoke cases, not 41 independently
qualified public algorithms. Partial selections never qualify a release.

## Latest established evidence

| Scope | Frozen evidence | Result and limits |
|---|---|---|
| Graph and captured Transformer | Source `3f35c9721418613bd81f81792f9c43b63f4974d9`; `E/stage63-graph-full-{cpu,metal}-summary.json`, `stage63-graph-cpu-metal-comparison.json` | Each full graph verifier passes 30/31 cases. All 176 digests from passing cases match CPU/Metal. Both full runs remain failed because of the upstream Session lifetime problem. |
| Full saved Transformer block | Lane graph evidence `full-transformer-stage63-{cpu,metal}/` and corresponding receipt files | CPU and Metal pass specialized/captured equivalence, optimizer on/off equivalence, checkpoint replay and the independent float64 oracle. Complete saved workload: batch 1, length 2048, width 384, six heads, feedforward 1024. This is not full language-model training or a performance A/B. Actual arrays are retained. |
| Linear owners, pandas and graph regressions | `E/stage63-native-{cpu,metal}-r1-summary.json` | Each passes 201/202 cases; one integer-promotion test fails. Its repair is in the next row. Original failures remain retained. |
| Exact PCA, host linear owners and integer-conversion repair | Native source `b1e55cfb056f370334f4f00a308b93da4cb04ccf`; Python package source `bb82270ea82ae853d61c10c2c559c5132c9df9df`; `E/stage64-native-{cpu,metal}-summary.json` | CPU 226/230 and Metal 228/230 pass. All integer/pandas cases and all 38 Metal PCA cases pass. Two Host tests incorrectly require identical independently generated exception classes; two CPU PCA cases assume an explicitly unsupported wide full-SVD route. Corrections preserve public exception inheritance/messages and the existing CPU refusal, with original failures retained. These remain private owners. |
| Real Torch input admission | `E/stage63-torch21-cpu-summary.json`, `stage63-torch-current-{cpu,metal}-summary.json` | Each has 13 passes, no skips. Torch 2.1.2 CPU and Torch 2.14.1 CPU/Metal are covered. Custom Python modes are refused before execution. |
| Standalone native smoke changes | `E/native-only-stage63-{cpu,metal}/summary.json` | Five changed cases pass on CPU and five on Metal. Other 31 fixtures have unchanged native closures with earlier retained proof. Local tests are not interpreter-isolated; new changed AMD/NVIDIA paths remain unqualified. Earlier Linux/AMD interpreter-free results remain scoped to their frozen source. |
| Installed package | `E/installed-public-stage58-archive-repair-{cpu,metal}/receipt.json`; `installed-public-stage58-generation-device-metal/receipt.json` | Earlier installed wheel: Metal-labelled suite 179 passes, including CPU generation cases; separate explicit Metal generation 23/23 closes that device gap and overlaps those 179. CPU 177 passes plus two explicit GPU-only skips, so the strict completeness controller fails. Not a current-source or whole-release wheel. |

### Final current-batch closeout

Final code/test commit: `f89ef53bff28a95e64e96c81a938e1cebd56c851`.
Native binding source: `ddabfd9321dec3418f7e465a8818269de38686f4`.
Only the changed tokenizer/owner binding was rebuilt; the sequence dependency
closure exactly matches the accepted stage63 binary and was reused.

The final full targeted suite records **289/290 passes, no skips, on each of
CPU and Metal**. Its one failure occurs in callback-poison test setup, before
the operation being audited. The corrected exact case separately passes 1/1
on each device, with strict native audits and no callback execution. Thus all
290 distinct selected cases have passing evidence; this is composed evidence,
not a claim that the original full run passed. The selected cases include
48 private PCA cases, 30 warning cases and 20 temporary activations of real
public linear-model classes. The public installer remains inactive.

See [the machine-readable closeout](core_completion_closeout_20261020.json),
`E/stage67-native-{cpu,metal}-summary.json`, and
`E/final-key-fixture/{cpu,metal}-summary.json`. The final test-only repairs
changed no product source or native binary. Original stage65 linker failure,
stage66/67 test failures and their precise repairs are retained. The linker
repair uses supported `OwnedDLHandle.get_function` on the active CPython
library, without toolchain changes or a Python callback.

No whole-library, installed-current-wheel, cross-vendor or performance claim
follows from this slice. Never add overlapping groups above into a suite count.

## Current implementation boundary

- Public Samba, Sequential and generation paths have bounded native runtime
  audit evidence. Broader constructor, input, attribute and transitive callback
  coverage across the whole API is incomplete.
- LinearRegression/Ridge (and HostLinearRegression/HostRidge) are public
  one-call shells into the native owner since 2026-10-21 (lane
  activate-linear-installer): constructors, fit/predict/score/save/load,
  parameters, tags, attribute reads and `host_model(path)` for linear files.
  The owner uses balanced CPython ownership (`bindings/py_balanced.mojo`).
  These classes now need NumPy; a NumPy-free native reader/writer is owed.
- PCA exact and randomized native owners are staged. Public activation remains
  held for ordinary sparse inputs, inherited validation, feature names/tags,
  output behavior and HostPCA admission. Public CPU training is already allowed;
  the existing `require_training` helper is a no-op, not a gate to add. The CPU wide full-SVD
  refusal must remain explicit; no new CPU algorithm was added to hide it.
- Graph capture, native replay/training, serialization and selected structural
  optimization/control-flow paths exist. This is not arbitrary rank/control
  flow, a comprehensive optimizing compiler or proven general parallelism.
- Distributed workers, checkpoint and selected model paths have numerical and
  runtime evidence. Single-device process tests do not prove physical
  multi-GPU sharding, overlapping pipelines or arbitrary user-model plans.

## Deferred work, in priority order

1. Finish existing public native API migrations. For the linear installer,
   verify actual constructors/getattr/first calls, ordinary feature names and
   inherited validation, warning behavior, optional NumPy behavior and the
   separate `host_model(path)` router. Then complete PCA compatibility. Audit
   the rest of the public surface without treating checker debt as exemptions.
2. Resolve or obtain supported upstream support for the Mojo Session/
   PythonObject lifetime failure. The independent reproducer uses Mojo 1.0.0
   `ed45d567` without mojolearn and reproduces under CPython 3.12 and 3.14.7.
   See `E/stdlib-ownership/` and `experiments/python_object_ownership/README.md`
   on the integration branch. Passing arithmetic does not close this blocker.
3. Qualify a coherent installed package and expand the standalone callers.
   Source overlays currently reuse 105 retained extension artifacts. The
   reuse audit found no strict unchanged closure among those old extensions:
   85 are entry-wrapper refactor changes and 20 have additional source changes.
   Historical compiler/ABI provenance is incomplete; none is silently approved
   for whole-package reuse. See lane distributed evidence
   `build-reuse-stage62/`. Rebuild only the changed dependencies required by
   the chosen qualification batch; do not rerun accepted unrelated evidence.
4. Obtain the independent native NVIDIA witness for four retained PTX
   discrepancies and four weighted-tree batch checks. CPU, Apple, AMD and PTX
   agree on the selected new results, but PTX cannot witness itself. Do not
   hand-edit references or admit the results without the missing witness.
   The retained driver is `E/run-reference-repair.py`.
5. Qualify affected NVIDIA/AMD closures and physical distributed configurations.
   General tensor/pipeline parallelism, broader partition plans, complete
   distributed checkpoint behavior and richer graph/profile tooling remain
   separate development and hardware qualification work.
6. Measure the optimizer/default candidates with the saved full workloads on
   NVIDIA and AMD, preserving Apple identity. Only then decide defaults. No
   additional optimization or broader feature project was opened at winddown.

## Machines and retained artifacts

No new NVIDIA machine was obtained in nine cheap-GPU attempts; the final log
is `E/nvidia-native-round9-community-up.log`. No new retry is scheduled by this
handoff. Other account pods belong to other sessions and must not be stopped.

Owned AMD lease `611037189` was archived, deleted and verified absent
(DELETE 204, GET 404); see `E/amd-round3-teardown.json` and
`amd-native-only-final-teardown-ready.json`. Earlier leases were also retired.
Actual required binaries/data/results were retained separately from the
general archiver, which excludes some large/native artifacts.

Recent actual source, binaries, original failures, standalone executables and
full Transformer arrays are retained in `E/stages61-64-handoff-artifacts.tgz`:
176,072,164 bytes, SHA-256
`422bd8db5a402566357b400db3db57696b144dee1749281dc754a4122d7724b7`.
Local manifest and verification are `stages61-64-handoff-manifest.json` and
`stages61-64-handoff-retention.json`; R2 receipt is
`stages61-64-handoff-r2.json`, which confirms a full R2 readback at
`measurements/core-completion/2026-10-20/native-runtime/stages61-64-handoff-artifacts.tgz`.
Earlier sources, installed wheel and evidence are
in the stages47–50, stages51–58, stages59–60-installed and native-only bundles
recorded in E. Keep both successful and original failed runs.

The final batch, including the failed and repaired builds, original test
failures, corrected fixture, compiled binary and source, is also verified
locally and by full R2 readback: `E/final-core-completion-artifacts.tgz`,
136,166,812 bytes, SHA-256
`cc78c7ae304e0c5fbae28cc6dfb822c1d5fd1354870c6eb50f3cc65860dd1979`.
Its R2 key is
`measurements/core-completion/2026-10-20/native-runtime/final-core-completion-artifacts.tgz`;
receipts are `final-core-completion-{retention,r2}.json` in E.

All three delegated lanes have finished. No build, test, development or owned
GPU job is left running by this handoff. Resume the deferred work only when
Andrew asks; the present session stops after committing this record.

Do not publish the site/docs/PyPI as part of resumption without Andrew asking.
Do not merge the broad migration as fully qualified while the failures and
unverified public/vendor paths above remain.
