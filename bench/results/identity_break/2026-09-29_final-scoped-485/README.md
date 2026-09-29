# Final scoped reference qualification, 2026-09-29

These are unmodified saved columns from harness `485c28c69`, numerical native
source `3dded3bd4`. Apple ran the installed 0.8.25 candidate built from
`b7154e9cd039` (wheel SHA256
`4aee0c707086aa578129d064facd0e243d284a0622536389b330628229808283`).
The hardware columns are Apple M4 Metal, AMD HIP, and NVIDIA H100 CUDA;
AMD and NVIDIA also captured their native CPU oracle arms. Every original
record retains its actual package paths, source commit, input hashes, loaded
binding hashes, repeats and property protocols. `raw-inventory.json` records
source archive paths and exact byte hashes. Apple has two repetitions; each
other arm has one, with independent corroboration across device classes.

The scope is **41 batch lanes × eight remaining fixtures**, plus **TSVD and
UMAP × all nine fixtures**. It does not repeat the global445 base sweep.
Saved-column comparison reports no numerical differences or incomplete cells:
880 numeric part matches for batch41 and90 for the two contract revisions.
Structural N/A and unrecorded sampler/trainer parts remain separately counted.

The strict `_verify_reference.build_table` and `merge_reference_lanes` paths
admit43 complete lanes. For the41 batch lanes, the original table's123
referenced base-record files are retained as inputs; they are not relabeled
as new source485 captures. The eight new fixtures have Apple, AMD, NVIDIA,
and CPU evidence. Historicalbase retains Apple, AMD, CPU where those were the
available classes. No existing class was dropped. Exactly18 values change:
TSVD's expanded train digest and UMAP's option-extras saved-model digest, each
on nine fixtures. All41 batch lane reference values remain unchanged.
Unrelated lanes/cells and global legacy admission metadata remain unchanged.

`par-graph-umap` is **not promoted** by these single-device columns. Its old
reference revision remains stale until separately qualified physical-parallel
records are admitted.

Reproduce saved-column comparisons from this directory (adjust REPO):

```sh
python "$REPO/tools/consolidated_check/compare.py" --plan batch41-plan.json \
  --column metal=apple/all --column hip=amd/batch41 --column cuda=nvidia/batch41 \
  --fixtures ties,hashed,wide,denormal,denormal_ftz,dupes,odd,negative --json-out /tmp/batch41-comparison.json
python "$REPO/tools/consolidated_check/compare.py" --plan contract2-plan.json \
  --column metal=apple/all --column hip=amd/contract2 --column cuda=nvidia/contract2 \
  --fixtures base,ties,hashed,wide,denormal,denormal_ftz,dupes,odd,negative --json-out /tmp/contract2-comparison.json
```

Full local derivation logs/scripts and row-by-row comparisons are preserved
in `mojolearn-evidence/consolidated-2026-09-28/packaging-preparation/scoped485-crossvendor`.
This reference update is not a claim that unrelated release gates passed.
