# Automatic compact result retention

MojoLearn is the repository for compact measurement and verification results.
The existing run directories and R2 archives continue to retain bulk evidence.
Collection copies records; it never deletes originals, changes a verdict,
admits a comparison, or regenerates the wheel's verification reference table.

## What is collected automatically

| Maintainer runner | Repository destination | When |
|---|---|---|
| `tools/bench_board.py` | `docs/measurements/latest/auto/bench-board-<vendor>.json` | After each recorded race and at completion; smokes use a separate `-smoke` stream |
| `tools/run_full_do_measurements.py` | `docs/measurements/latest/auto/campaign-<vendor>.json` | At vendor completion and controller exit, including partial/failed runs |
| `tools/light_verifier.py` | `docs/verification/latest/auto/light-<vendor>[-fault-<name>].json` | After capture or clean/injected/restored checks |
| `tools/verify_api_endpoints.py` | `docs/verification/latest/auto/api-<family>[-faults].json` | After endpoint capture and requested negative controls |
| `tools/verify_neural_provider.py` | `docs/verification/latest/auto/neural-provider-<mode>.json` | On controller exit if a result was produced; source-only and runtime stay distinct |
| `tools/record_identity_column.sh` | `bench/results/identity_break/retained/<record-sha256>/` and `docs/verification/latest/auto/identity-<vendor>.json` | After recording the column, including records that fail admission |

Latest summary files replace the prior snapshot of the same stream. A snapshot
with an unfinished state or a null exit code is not a completed pass. Historical
snapshots already committed remain in Git history. This does not promise that
every intermediate snapshot was committed. Existing dated summary directories
are historical copies and are not rewritten by these hooks.

Identity columns retain exact JSON bytes at immutable paths, with provenance
sidecars. Recollecting the same record reuses its path. A new receipt for the
same bytes is retained separately. Generate from those repository copies, then
commit the records and updated table together. Existing admission and source checks decide which cells can
enter `python/mojolearn/verify_reference/table.json`. Failed or partial evidence
is never automatically promoted. Wheel builds still use the reference inputs
committed in their selected MojoLearn source revision.
The next step after collecting columns is
`tools/admit_identity_columns.sh --lanes <changed-lanes> <returned-record-dirs...>`.
That command retains missing external columns in MojoLearn, reuses existing
copies, updates only the admitted lanes, and preserves unrelated table rows.
Commit the records, updated table and receipts together before the next wheel
build; no preliminary evidence commit is required. Collection and ordinary
verification do not trigger this promotion themselves; see
[the incremental release sequence](VERIFY.md#admitting-one-verified-group-without-rewriting-other-references).

## Size and provenance

Each generated summary and each identity input is capped at 4 MiB. Oversized
output reports `RETENTION FAILED` and leaves the previous summary and original
run files intact. Summary replacement is atomic. Boards retain recorded timings,
quality, comparison decisions, clock boundaries, source/artifact/data identities,
and references to full evidence. Repeated context is stored once by hash inside
each summary. Per-cell worker details remain in the full evidence receipts.
Nonfinite values in older reports are represented explicitly as
`{"$float":"NaN"}`, `{"$float":"Infinity"}` or `{"$float":"-Infinity"}`
so the summaries remain standard JSON; they are never replaced with zero.

No datasets, compiled libraries, source bundles, wheels, raw logs or array/model
archives are copied by this collector. It reads existing JSON, one board at a
time, and performs no computation on datasets or GPU work. Historical identity
records accumulate on disk; normal verification does not load that archive.

## Collection versus GitHub publication

Automatic collection writes to the checkout containing these tools. Set
`MOJOLEARN_SUMMARY_REPO=/absolute/path/to/mojolearn` to collect into another
checkout. An exported source tree without Git metadata cannot serve as the
destination; its controller must collect returned results into a real checkout.

Collection does not run Git commits or push. The measurement controller's
existing explicit `commit_boards: true` setting now commits its final summaries
alongside the boards and pushes from `main`; it excludes unrelated staged files.
Other runners leave the records for the normal result commit/push. There is no
watcher or automatic publication from an installed PyPI user's verifier.

For another runner's saved JSON output, collect it without rerunning verification:

```sh
python3 tools/retain_run_summary.py verification --producer verify-all-cpu /path/to/report.json
python3 tools/retain_run_summary.py identity /path/to/cpu.identical.json
```

The first command also supports multiple JSON reports. Its producer names a
stable latest stream; distinguish devices, normal checks and fault injections.
Raw files remain in their original location. A retention failure is reported
separately from the runner's numerical verdict and must be resolved before
claiming that the results reached the repository. Existing bulk archive and
machine-teardown requirements still apply.
