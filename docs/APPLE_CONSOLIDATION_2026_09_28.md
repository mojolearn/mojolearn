# Apple consolidation, 2026-09-28

The integration branch `fix/apple-consolidated-verifier` starts at
`lane/apple-merged` (`7483efa40`) and includes all eleven original Apple
branches, all eleven `apple2` branches, and `devctx-lifetime` (`e395c3fa7`).
The last moving tips included in this snapshot are cluster `809eccd629`,
linear `c128c4f3e8`, neighbors `9aa0952285`, and prep `2c713f92f1`.
Consolidation preceded execution of checks. Later branch changes require a
new integration snapshot and evidence.

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

## Validation scope

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
newly merged native source. Native builds and CPU/GPU comparisons must use
the exact committed integration snapshot before promotion to main/release.

## Reference evidence still owed

At this snapshot 230 configurations have no shipped reference cells;
`gmm`, `gmm-sample`, and `par-gmm` have stale references. These are distinct
from structural hardware inapplicability and remain visibly unverified.
A same-commit CPU/GPU comparison can validate fresh results independently of
the reference table. Reference admission is a separate step requiring the
specified independent, complete evidence; never replace references merely
to make a mismatch pass. The existing current-revision GMM artifacts are
partial and insufficient to admit all nine fixtures.

The published byte-LM host mismatch was a stale 0.8.24 reference following a
weight-decay default change. Existing outputs for all nine fixtures (18
training/inference hashes) match the already-admitted current source table.
