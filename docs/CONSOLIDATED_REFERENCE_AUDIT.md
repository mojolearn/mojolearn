# Consolidated verifier evidence audit

Read-only audit of consolidated source `a3bcdeab5` on 2026-09-28: 504
registered identity lanes, 230 lanes with no shipped reference-table cells,
and three stale lanes (`gmm`, `gmm-sample`, `par-gmm`). All 230 missing lanes
are declared `no reference`; they are not hardware-inapplicable. They comprise
174 `x-*`, 22 `sequence-*`, 25 `trees-*`, four `resample-*`, two `pca-*`, and
one each under `tsvd`, `optim`, and `ivf`. The shipped table was not changed.

The only committed current-revision GMM numerical columns found were
`bench/results/gmm_classic_pp_2026-09-26/cpu-column.json` and
`metal-column.json`. Both cover base/denormal/odd only and explicitly omit
batch and sampler/replay. `_verify_reference.admit` rejects both as partial
columns. Their matching numbers are useful diagnosis, not sufficient evidence
to clear the stale-reference holds. Other records carry current revision
metadata but contain no GMM cells. `par-gmm` also requires physical multi-GPU
qualification; a Mac's local column cannot discharge that claim.

## Producing a reviewable reference candidate

The consolidated check's clean GPU/CPU lane JSONs use the identity harness
with all declared parts and one repetition. Current admission requires two
matching witnesses: different device classes fitted once each suffice. It
also requires the expected fixture bytes, revisions, complete declared parts
and property protocols. A passing text log is not a reference record.

First use base-fixture checks for diagnosis. Then collect clean records for
all nine fixtures at the committed consolidated source and built binaries;
the algorithm lane runner defaults to all fixtures when `--fixtures` is
omitted. Preserve build provenance and clean JSONs. Do not feed sabotaged,
partial, or failed records into a replacement reference. Preserve independent
NVIDIA/AMD evidence separately: GPU/CPU agreement on Apple alone is not a
new measurement of those vendors.

With `CLEAN_RECORD_DIR` naming those reviewed records and `LANES` naming
exactly the ordinary lanes to admit, emit a separate candidate (no native
fits are performed by this command):

```sh
PYTHONPATH=python python -m mojolearn verify --all --batch-checks \
  --emit-reference /tmp/consolidated-reference-candidate.json \
  --reference-table python/mojolearn/verify_reference/table.json \
  --records "$CLEAN_RECORD_DIR" --lanes "$LANES"
```

Review admission logs, conflicts, per-part device witnesses, lane revisions
and all-nine-fixture coverage before promoting the candidate in a separate
change. A base-only scoped candidate must not replace a lane's all-nine
reference. Reference promotion must also remove only the corresponding
resolved `no reference` or `stale reference` manifest holds. Do not remove
holds merely because an unrelated suite passes.

## Targeted native radix regression

`neighbors/checks/knn_identity_check.mojo` covers planted ties and query
tiling, but does not cover the 1024-capacity boundary or wider rounds.
`knn_selector_long_rows_check.mojo` checks the separate small-k selector
(k <= 64); passing it cannot qualify the radix correction.

After the consolidated job has built the core binding, run once, without
another build:

```sh
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \
  pixi run -e default python -u tools/check_radix_public.py \
  --json-out "$OUT/radix-public.json"
```

This uses 2053 index rows, two integer features, three queries, k=1024/1025/2000,
mixed and all-equal distance ties, query tiles 1 and 3, and two repeated
resident queries. An independent integer-distance oracle compares exact
Float32 distance bits and index tie order. CPU bindings are refused by this
native driver because they do not exercise the GPU kernel. Its separate
unit tests validate the oracle and show that swapped tied indices and a
single changed distance bit are detected; they do not qualify native code.
