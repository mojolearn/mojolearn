# AMD captured-result normalizer

This is the exact deployed campaign normalizer snapshot. Restore it beside the
AMD evidence directory's `normalized-measurements.json` and `live/repairs/results.json`;
it resolves inputs relative to its own file. The live copy is
`/Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/normalize_stream.py`.
The local collector imports `run()`; restart only that collector after deployment.
Do not start a second normalization writer alongside the collector.

It rebuilds streamed metadata from retained receipts without measuring or modifying
raw results. The campaign machine identifier is intentionally frozen. I17 excludes
only the resident-frontier and LG-exact controls on Depthwise, retaining the distinct
inheritance controls. I14 is a gated/shortcut/chunk bundle, not isolated chunk-width
evidence. Existing same-route, confounded-schedule and baseline-domain exclusions
are retained. No classification here promotes a default.

The 2026-10-06 metadata refresh exited zero: two I14 pairs carry bundle metadata and
six I17 Depthwise control rows remain excluded. No tests, builds, or measurements
were run for this archive. Original receipts remain in the evidence directory.
