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

## Neural resource retry

`neural-repair-tail.py` is the deployed replacement for the existing serial
worker's opponent tail. `neural-repair-selection.json` freezes exactly eight
previously failed neural races and a cutoff timestamp; completed races and
unsupported/quality-failed opponents are preserved. Restore the selection at
`/root/overnight-ab/neural-repair-selection.json` and the controller as
`/root/overnight-ab/opponent-tail.py` only on this owned campaign machine.

The immutable harness freeze `3e55ba47e581fa75dffba286327d023f96801831` is
base `b0f5242d244ef75cd9ba7d78ca92e10008d75510` with the existing opponent
binding-gate, byte-corpus fallback, and explicit failed-race retry patches listed
in the selection. Stage tools, bench/speed, python/mojolearn Python corpus,
mamba/corpus/gen_corpus.py and transformer/corpus/gen_corpus.py from that freeze.
The archive hash is recorded in the selection. No binding build is required.

Before deployment, the original board, selected failure logs, and raw neural
receipts were copied to
`results/opponents/attempts/neural-pre-resource-repair`. Only the existing
worker's `results/repairs/OPPONENTS_DONE` marker was cleared to queue this
retry; no cloud owner or idle policy changed. Active tail work removes DONE
and protects capture. No identity or compilation validation was performed.
