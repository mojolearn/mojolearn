# Repository retention

Age alone is not a deletion criterion. A three-day-old result may still be the
only evidence supporting a shipped claim or explaining a failed attempt.

Keep summaries, cards, verdicts, hashes, provenance, logs, raw numerical evidence,
and unresolved lane status. Archive superseded coordination documents, preserve
relative links, and route current work through the root ROADMAP.md.

For new work, update one status file per active lane instead of adding a new
per-session handoff. Keep its current state and next action short, linking to
records for detail. When a lane closes or a plan is superseded, carry unresolved
work into the maintained status file and archive the old coordination notes.
Update the root roadmap when the active release or priorities change.

Before committing result output, stage explicit paths and review the staged
file list. Keep build products outside the repository; retain the summary,
provenance and evidence needed to support claims. Install the repository hooks
with `sh tools/hooks/install.sh`; their checks supplement this review and do
not classify every possible artifact automatically.

Wheels, compiled executables, shared libraries and packed build/source archives
do not belong in the working source tree. Before removing existing copies,
verify a recovery copy by SHA-256 and retain a manifest. Do not remove unique
untracked artifacts or use broad age-based deletion. Raw evidence migration
requires a verified durable destination and updated citations first.

The September 17 cleanup manifest is
[cleanup-artifacts-2026-09-17.json](cleanup-artifacts-2026-09-17.json). It records
the original paths, sizes, hashes, source commit and local backup directory.
Restore an artifact by copying its relative path from that directory back to
the same repository path. The backup is local, not a published evidence store.
No Git history was rewritten; deleting tracked files does not shrink old clones
or historical Git objects.

`.rgignore` excludes `archive/` and `bench/results/` from ordinary ripgrep
searches. Use `rg --no-ignore PATTERN archive/ bench/results/` for evidence
research. This affects searches, not Git tracking or test/runtime access.
