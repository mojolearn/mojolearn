# Apple consolidation, 2026-09-28

The Apple work was consolidated first and then merged into `main`, following
Andrew's instruction to merge everything before one coordinated validation.
The consolidation includes all eleven original Apple branches, all eleven
`apple2` branches, and the device-context lifetime repair. The integration
worktree remains useful for preparing fixes; promotion to `main` is no longer
conditional on finishing the native sweep. Release qualification remains a
separate requirement.

Later main updates include `apple-merged-owed` repairs and M2 evidence, the
resample manifest correction `79559ad0f`, and the neighbors final evidence
`75ddcae2e`. The broad native run is deliberately frozen at `308878e80`;
subsequent fixes receive targeted checks rather than restarting that sweep.

## Repairs

- Preserve both CNN grouped/FAST matrix plans and neural double buffering,
  small windows, and opt-in automatic splitting when resolving GEMM changes.
- Retain caller-supplied Mamba contexts; direct callers also reuse a
  process-lifetime context.
- Remove Apple radix selection's unfenced cross-thread device-memory
  communication. Apple rescans immutable input and stages winners in shared
  memory; other backends keep their existing path.
- Run installed verification in bounded fresh workers, check results as they
  arrive, checkpoint progress, and stop after a timeout or unhealthy device.
- Keep inapplicable parallel-device configurations in the coverage inventory
  without executing them in the default single-device verifier.
- Declare missing resample routes and tree exports; bound routine Python
  parity tests instead of silently sweeping the entire CPU surface.
- Add a consolidated checker that propagates failures, refuses incomplete
  evidence, and fingerprints the committed source, native libraries, and
  execution settings before resuming.

## Validation scope and outstanding work

The consolidated inventory has 504 configurations. The default source-tree
comparison selects 445 single-device configurations and explicitly records
59 parallel-device exclusions. It uses the existing base fixture, one CPU
column, and a 120-second limit per arm. GPU execution is sequential on each
device; compilation may use two workers. This checks a specific fixture and
configuration set, not all input sizes or all possible executions.

The old 101-lane M2 Pro result passed in 962 seconds at `1c0677d7b8`.
It predates this consolidation and does not qualify this source snapshot.
The separate devctx lifetime script is deliberately a much larger stress
run; it is not the installed verifier's normal workload.

Python/controller tests use published 0.8.24 CPU bindings for imports and
bounded CPU parity where needed. They do not establish correctness of the
newly merged native source. Native evidence must identify the exact committed source and built binaries.
The broad `308878e80` run on Apple M4 and AMD covers the 445 base-fixture
comparisons plus the separate radix regression. Its results do not by
themselves qualify later source changes or other fixtures. Completion and
cross-vendor agreement have not yet been established for the final main tree.

Two cross-platform problems found in historical records now have source fixes:
GLM target generation used platform-dependent NumPy `exp`; TreeSHAP traversed
unreachable zero-cover paths and produced differently signed NaNs. CPU-only
probes established both causes. Portable GLM targets match across Arm/x86, and
an isolated native TreeSHAP replay produces 512 identical finite values with
additivity. End-to-end validation is targeted at the four GLM lanes and two
TreeSHAP lanes, including the negative TreeSHAP fixture. Direct comparison of
Apple and AMD records is required: local GPU/CPU agreement alone is insufficient.

## Reference evidence still owed

The initial consolidated audit found 230 configurations with no shipped reference cells;
`gmm`, `gmm-sample`, and `par-gmm` have stale references. These are distinct
from structural hardware inapplicability and remain visibly unverified.
A same-commit CPU/GPU comparison can validate fresh results independently of
the reference table. Reference admission is a separate step requiring the
specified independent, complete evidence; never replace references merely
to make a mismatch pass. The initial current-revision GMM artifacts were partial and insufficient to
admit all nine fixtures. A separate recovery of historical records produced a
172-lane reference candidate; it has not been promoted into the shipped table.
The four GLM and two TreeSHAP lanes now carry new revisions, so pre-fix records
cannot establish their current references. See
[CONSOLIDATED_REFERENCE_AUDIT.md](CONSOLIDATED_REFERENCE_AUDIT.md).

The published byte-LM host mismatch was a stale 0.8.24 reference following a
weight-decay default change. Existing outputs for all nine fixtures (18
training/inference hashes) match the already-admitted current source table.
