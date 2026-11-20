# The release status report

One authoritative report for a release: one row per release-ledger requirement (identity lane x
column, `adapters-*`, `runtime-*`, `compile-*`, `candidate-*`, `published-*`, `coverage-source`,
`runtime-source`, `performance-*`) and one row per release step (gates, builds, packs, publishes),
each in exactly one of seven states, with its evidence, the step that produces it and, for every row
that is not passed or reused, the single command that moves it. A waiver stays visible as a waiver.

    pixi run release 0.8.66 --report [--report-json FILE] [--owner-waiver W.json]
    python3 tools/release_report.py --version 0.8.66 --commit C --rows rows.json [--state state.json] \
        [--waiver W.json ...] [--plan plan.json] [--stage pre|post] [--json report.json]
    python3 tools/release_report.py --version 0.8.65 --commit 6519a54c6579 \
        --blocked-tsv tools/testdata/blocked-rows-0865.tsv --waiver tools/testdata/owner-waiver-0865.json

`--report` runs, rents and publishes nothing. It writes `report.json` and `report.md` beside the
release state (`<release root>/<version>/<commit12>/`) and prints the totals line and the unresolved
rows. It exits 0 when ready, 3 otherwise. `--plan` embeds the same totals line in `plan.json`
(`report_totals`) and prints it.

## The totals line

    REPORT 0.8.66 passed=.. reused=.. failed=.. missing=.. blocked=.. n/a=.. waived=..  ready=YES|NO

`ready=YES` means zero `failed`, zero `missing-execution` and zero `blocked-provenance` rows. Waived
rows do not stop `ready`, but they are counted on the line, listed in `report.md` with their reason and
approver, and broken down by their underlying state (`waived_underlying`).

## The seven states

| state | meaning | derived from |
|---|---|---|
| `passed` | executed at this freeze with these inputs; evidence verified | ledger row `status=passed` tested at the freeze's own revision with no reuse decision; a step recorded done for this commit |
| `reused` | passed on older evidence that the ledger proved still applies | ledger row `status=passed` with `dependency_decision` `reference-closure-unchanged` (the admitted reference, docs/release/IDENTITY_CLOSURES.md) or `retained`, or a `tested_revision` other than the freeze; a step done whose result says it reused (`REUSE`, `reused`, `adopted`, `taken from`) |
| `failed` | an execution failed, or a failure contradicts the admitted reference | ledger `status=failed` or `action=repair`; a step recorded failed for this commit |
| `missing-execution` | no evidence for these inputs: something must run | ledger `status=untested` with `action` `run` or `resume` (no evidence, closure changed, `dependency equivalence affected`); a step owed, or held for `--publish` |
| `blocked-provenance` | evidence may exist but cannot be linked to these inputs | ledger `action` `complete-provenance` or `restore-evidence`, or `dependency_decision=blocked` (the old cross-commit selector's `selector provenance unresolved`); an unreadable ledger |
| `not-applicable` | not required for publication | ledger `required=false` (the lane's applicability excludes this column, `performance-*`); a step done as `SKIPPED` |
| `waived` | an owner waiver names a row that is not passed or reused | a prefix or name in `--owner-waiver`, a committed `docs/release/<version>-*waiver*.json`, or the release state's `owner_waivers` |

Rules:

- The state is derived from the evaluated ledger rows the gate and `--status` read
  (`Release.readiness_status`, read-only). The report does not judge evidence itself.
- A waiver never turns a row into `passed`. A waived row keeps `underlying_state`, the waiver rule that
  matched, its file and sha256, the approver and the reason.
- A row that passed on evidence and is also named by a waiver stays `passed` or `reused`, and records
  `waiver_unused`, so a waiver that is no longer needed shows up.
- `not-applicable` rows are never waived: they need no waiver.

## Evidence and producing step

`evidence` lists the row's evidence paths, each at the revision it was read at (`path@commit12`).
A retained item that names an R2 tag is listed as `r2:<tag>`. `producing_step` names the step
whose execution writes the row's evidence:

| row | producing step |
|---|---|
| `identity-<vendor>/<lane>` | the reference refresh (`tools/reference_refresh.sh`), admitted at the freeze |
| `compile-apple` / `compile-<nvidia,amd,cpu>` | `macos-build` / `linux-builds` |
| `runtime-*`, `adapters-*`, `candidate-*` (apple, nvidia, amd) | `gate-<vendor>` |
| `published-<vendor>` | `published-<vendor>` |
| `coverage-source` | `tools/verify_coverage_audit.py`, admitted by `tools/release_ledger_admission.py` |
| `runtime-cpu`, `runtime-source` | none in the release step graph (the report says so) |
| a release step | itself |

## The next-action catalogue

| row, state | next action |
|---|---|
| identity, `missing-execution` | `tools/reference_refresh.sh <commit12> <column>`. The column is `ptx` when the reason names the PTX set, otherwise the row's vendor. The refresh records only the lanes whose closure changed, this one included. |
| identity, `failed` | `fix the code (a digest mismatch is a bug, never a rerun), then tools/reference_refresh.sh <commit12> <column>` |
| identity, `blocked-provenance` from the old selector | `re-judge with the reference admission (...): tools/release.py <v> --report` |
| identity, `blocked-provenance`, reference analysis unavailable | `python3 tools/release_requirements.py plan --ledger <ledger> --explain --json` |
| any row, `action=restore-evidence` | `restore retained evidence: tools/box_persist.py fetch --tag <tag>` when the evidence or the plan's `restore` row names a tag; otherwise `restore retained evidence from <plan restore source>`; with no retained copy, the paths and the producing step to rerun |
| any row, `action=resume` | `wait for or resume the unfinished run <id>: tools/release.py <v> --status` |
| `runtime-*`, `adapters-*`, `candidate-*` | `tools/release.py <v> --only gate-<vendor>` (failed: `fix the failure it recorded, then ...`) |
| `compile-*` | `tools/release.py <v> --only macos-build` or `--only linux-builds` |
| `published-<vendor>` | `tools/release.py <v> --only published-<vendor>` |
| `coverage-source` | `python3 tools/release_ledger_admission.py admit --ledger <ledger> --root . --adapter coverage-source --producer audit=<audit json>` |
| a row with no producer | `add a producer for <row> (...), then tools/release.py <v> --report` |
| a step, owed or failed | `tools/release.py <v> --only <step>` (failed: after fixing the failure it recorded) |
| a step held for publication | `tools/release.py <v> --only <step> --publish pypi` |

## The 0.8.65 fixture

`tools/testdata/blocked-rows-0865.tsv` (the 455 rows the 0.8.65 gate refused) and
`tools/testdata/owner-waiver-0865.json` (451 prefixes) are copied from the 0.8.65 release evidence.
`tools/test_release_report.py` replays them:

- With the waiver: `REPORT 0.8.65 passed=0 reused=0 failed=0 missing=4 blocked=0 n/a=0 waived=451
  ready=NO`. Exactly the 451 waived rows are `waived` (395 blocked-provenance and 56 missing-execution
  underneath), none is `passed`. The 4 missing rows are `candidate-nvidia` and the three
  `published-*` rows, which the waiver does not name. The fixture holds only the blocked rows, so
  passed rows do not appear.
- The 442 identity rows replayed through the reference-admission judgement
  (`tools/release_identity_reference.py`) in a throwaway Git repository. The repository is shaped like
  0.8.65: four record commits, tooling commits after them, four column admissions, then the version
  bump that the `hf-*` lanes read. Result: 0 `blocked-provenance`, 442 `reused`, 0 `waived`,
  `ready=YES`. Each row records the waiver as `waiver_unused`. A closure edited after its record
  gives `missing-execution` with a `tools/reference_refresh.sh` next action, never `blocked`. The
  closures are fake. The real 0.8.65 closures and the full 1.1 GB ledger are not replayed here;
  docs/release/IDENTITY_CLOSURES.md gives that replay (85 neural-training lanes untested).
