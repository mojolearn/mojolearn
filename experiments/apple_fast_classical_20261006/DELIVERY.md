# Source-only delivery

Branch: `lane/apple-fast-classical-ideas-20261006`.
Worktree: `/Users/andrewhendel/CascadeProjects/mojolearn-apple-fast-classical-20261006`.
Forked from local `main` at `fd6cf8045`; this task does not merge into `main`.

The inventory contains 54 Apple FAST classical ML hypotheses. Four implementation
lanes added independent opt-in controls and source changes, with exact mechanisms,
caller limits, paired prerequisite flags and future quality gates in `lanes/`.
New source switches are OFF. Floating-point bits may change when candidates are
enabled; no quality or performance outcome has been established.

The owner explicitly prohibited compilation, verification and measurement.
Accordingly no compiler, test runner, linter, syntax checker, manifest checker,
benchmark, device job or selector invocation was run. No validation is claimed.
Commit hooks are bypassed and the commit carries a CI-skip marker. No boards,
measured results, release defaults or remote measurement jobs are updated.

Remaining qualification is deliberately outside this task: compile, establish
actual caller reach, preserve task quality and API contracts, resolve saved full
workload recipes including intrinsic caps, and measure complete A/B operations
and interacting configurations. Source descriptions are not evidence for those
steps. Unsupported future compiler/runtime needs must be raised with Modular,
not worked around by patching the toolchain.

Evidence paths: `README.md`, `ideas.json`, `lanes/{linear,geometry,trees,preprocessing}.{json,md}`,
and source paths named in each lane manifest. `select.py` is unexecuted offline
configuration glue and cannot launch a compiler or workload.
