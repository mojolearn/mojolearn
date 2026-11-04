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
lanes that reach the changed definitions; a `host_surface.py` edit to a row the selector names
selects that row's lanes, and any other code edit there is unattributed; a docstring- or
comment-only Python edit selects nothing.

Inert (selects no lane): the `__version__` literal in `python/mojolearn/_version.py` when its
reviewed consumers are unchanged (`release_label_only`, shared with the ledger), prose and docs,
release tooling no lane runs, and evidence directories. See the measured table below.

## Measurements on main (6519a54c65, 0.8.65)

Measured with the real `tools/lane_select.py` on this tree (513 registered lanes): each edit applied
alone to the working tree, `select([path], ref=HEAD)` with every per-file memo dropped, then restored.
Scripts and raw results: `~/mojolearn-evidence/ledger-trusts-reference-notes/` (`measure_closures.py`,
`measure.json`, `measure-shared.json`).

| File | Lanes whose derived file set holds it | One-line edit | Lanes re-recorded (any backend; CUDA identical) |
| --- | --- | --- | --- |
| `python/mojolearn/_version.py` | 431 | bump the `__version__` literal | 0 (release-label rule) |
| `CITATION.cff` | 0 | bump `version:` | 0 (prose) |
| `python/pyproject.toml` | 0 | bump `version =` | 0 (nothing reaches it) |
| `CHANGELOG.md`, `AGENTS.md`, `docs/**` | 0 | add a line | 0 (prose) |
| `tools/release.py` | 0 | append a comment | 0 |
| `python/mojolearn/host_surface.py` | 1 | append a comment | 0 (docstring/comment-only) |
| `python/mojolearn/host_surface.py` | 1 | change one family row (`_mojolearn_byte_lm` module) | UNATTRIBUTED: every lane resting on that commit is re-recorded |
| `python/mojolearn/host_surface.py` | 1 | one statement in `expansion_lane_of()` | UNATTRIBUTED: every lane |
| `bindings/_mojolearn.mojo` | 513 | append a comment | 425 |
| `bindings/_mojolearn.mojo` | 513 | one statement in `knn_search_binding()` | 482 |
| `bindings/_mojolearn.mojo` | 513 | one statement in the shared helper `_f32_ptr()` | 513 |

The version bump, citation, packaging metadata, changelog, docs and tooling already move no lane,
so the release commit never re-records anything. The two cascade points are:

- `bindings/_mojolearn.mojo` is in every lane's file set because it is the core extension every
  `import mojolearn` loads (the merged entry point is `bindings/entry/entry__mojolearn.mojo`, which
  `bindings/merged_groups.json` places per group). Even a comment edit re-records the 425 lanes that
  call one of its exports, because the selector has no Mojo comment parser by design; a body edit of
  one export re-records the lanes its definition graph reaches (482 for `knn_search_binding`, through
  the module's shared registration and helpers).
- `host_surface.py`: a code edit that is not a named row the selector knows is UNATTRIBUTED, which
  the recorder and the ledger both treat as "every lane resting on that commit changed".

Proposed (not implemented here; both are `tools/lane_select.py` rules with their own refusal tests,
and the selector is the frozen tool every consumer shares, so they belong in a selector lane):
(1) a Mojo comment/docstring-only rule mirroring the Python `docstring_only` (a `#` comment or a
docstring line inside a Mojo file moves no lane); (2) export-level attribution for the core
extension: a body edit of export X selects the lanes whose Python door calls X, and the module's
registration body is judged per `def_function` line; (3) host_surface function bodies attributed
to the lanes that call the changed function, as rows already are. Each is a narrowing, so each needs
a test that it still refuses where it must.

## The 0.8.65 freeze replayed (6519a54c65)

Every `identity-<vendor>/<lane>` row of the 513 lanes judged by this rule (raw:
`~/mojolearn-evidence/ledger-trusts-reference-notes/dryrun-0865b.json`; the GPU classes of every lane
treated as required, PTX for NVIDIA wherever the table carries a PTX column):

| Column | Passed | Untested |
| --- | --- | --- |
| apple | 428 | 85 |
| amd | 428 | 85 |
| cpu | 425 | 85 changed, 3 refused (no CPU `batch` column: device-only lanes) |
| nvidia | 0 (428 with the NVIDIA column unchanged) | 513: the PTX column rests on 4f7cdf9c2f |

- The 428 passed rows per column are exactly the recorder's routine scope, recorded or reused at
  0.8.65. No waiver is needed for them.
- The 85 untested lanes are the neural-training scope, outside the recorder's routine scope. Their
  rows rest on records before the 0.8.64 floor, and their closures changed after it
  (`autograd/distributed_*.mojo`, `core/device_plan.mojo`: the multi-GPU encapsulation). They were
  never re-recorded; they need a record (`SCOPE=neural-training`), or their release rows need to be
  scoped out.
- PTX: every PTX column row rests on 4f7cdf9c2f, before the floor, and the CUDA closure of every
  lane changed after it (the same change that re-recorded the NVIDIA column at 0814836ffb). When the
  NVIDIA environment binds the sm_80 PTX set, the PTX column needs a record (`nvidia-ptx-*`).

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
