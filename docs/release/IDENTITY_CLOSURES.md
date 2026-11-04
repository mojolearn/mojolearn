# Identity closures: what re-records a lane, and how the ledger reuses it

Andrew (2026-10-31, 2026-11-04): "how can we avoid unnecessary verification checks RELIABLY" and
"connect the ledger to reference admission". Publishing never needs a waiver for verified work.

## One comparison, used by the recorder, the admission and the ledger

A lane's identity evidence is reused when its **source closure** has not changed since the record its
admitted rows rest on. There is exactly one implementation of that comparison:
`tools/lane_select.py` (`changed_paths(base)` + `select(paths, ref=base, backend=<column backend>)`).

| Consumer | Question | Base |
| --- | --- | --- |
| `tools/record_identity_column.sh` (`identity_columns.py changed-lanes --class`) | REUSED or RECORD this lane? | newest good evidence of the column's class, or the release floor |
| `tools/admit_identity_columns.sh` (`identity_columns.py reuse-check`) | may an older incoming record stand for the admitted code? | the record's commit |
| release ledger (`tools/release_identity_reference.py`) | is `identity-<vendor>/<lane>` passed at the freeze? | every record commit the lane's admitted rows of that class rest on, or the release floor |

The ledger runs the comparison at the freeze, in a byte-exact frozen view of the freeze, with the
freeze's own `lane_select.py` (`release_requirement_dependencies._reference_closures`). A row is
`passed` when the admitted table committed at the freeze carries the lane under the freeze's harness,
every compared column agrees (re-read from the cited record bytes at the freeze), there is no
conflict, and the closure is unchanged. The evidence is the table, the record files and the
admission READMEs that name them, all at the freeze. Otherwise the row is `untested`, for example
`closure changed since 0814836ffb72 (nvidia cuda): 3 file(s): core/a.mojo, ...`, and the recorder
reaches the same answer and records that lane only.

The release floor: a record older than the newest NVIDIA and AMD release tags that are strict
ancestors of the freeze is measured from the floor, as the recorder measures it, because that
release's identity check (`python -m mojolearn verify` on the boxes) verified every table row for its
code. A tag made at the freeze itself never vouches for the freeze.

The old cross-commit selector over ledger attempts (`dependency equivalence blocked: selector
provenance unresolved`) no longer gates identity rows. `release_requirements.py plan --explain` still
runs it as a diagnostic (`selector_diagnostic`). Precision limitations (`unresolved-native-dispatch`)
are not a separate verdict: the closure keeps every reachable dispatcher branch, and a limitation is
only reported for a lane the selection already holds.

## What a closure contains

A lane's closure is derived (never hand-listed) by `lane_select.lane_sources()`:

- the lane's own harness code in `tools/identity_break.py` and `tools/identity_lanes/`, by call graph;
- the Python door it calls in `python/mojolearn/`, closed over the package's imports, and the CPU
  route tables (`host_surface.py` rows) of that lane's families only;
- the bindings those files name, and for each binding only the Mojo modules its called EXPORTS
  import (definition by definition, the way the build resolves imports);
- the data files the lane reads, and named every-lane rules (pinned toolchain, the Linux set builder
  for the Linux columns, harness code every column runs).

A changed path is judged by what changed, not only by which file: a Mojo or Python edit selects the
lanes that reach the changed definitions; a `host_surface.py` edit selects the lanes of the changed
rows; a docstring- or comment-only Python edit selects nothing.

Inert (selects no lane): the `__version__` literal in `python/mojolearn/_version.py` when its
reviewed consumers are unchanged (`release_label_only`, shared with the ledger), prose and docs,
release tooling no lane runs, and evidence directories. See the measured table below.

## Measurements on main (6519a54c65, 0.8.65)

MEASUREMENTS

## Changing the closure definition

No closure hash is stored anywhere: every consumer recomputes the comparison from the record commit
to the code at hand with the selector of that code. A change to `lane_select.py`'s rules therefore
invalidates no stored evidence and costs no GPU run by itself. The new rules decide the next
comparison; a lane is re-recorded only when a path changed since its record reaches it under them.
`CLOSURE_DEFINITION` in `tools/release_identity_reference.py` names the rule the ledger applied, for
provenance only.

## Tooling after a freeze

The frozen view of a source checkout no longer refuses tooling-only differences
(`release_requirement_dependencies.tooling_only`: `tools/`, `docs/`, `docs-site/`, `.github/`,
`experiments/`, `CHANGELOG.md`, `AGENTS.md`, `CLAUDE.md`, `README.md`; the identity harness
`tools/identity_break.py` and `tools/identity_lanes/` stay checked). Those paths are taken from the
frozen commit's Git objects, so the analysis is still exactly the freeze, and the view records them
(`source_view.tooling_from_revision`) with the tooling commit (`source_view.tooling`). Every other
tracked path must still be byte-identical to the freeze, and HEAD must still be the frozen commit.
