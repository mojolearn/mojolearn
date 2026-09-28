# Consolidated verifier evidence audit

The initial read-only audit of consolidated source `a3bcdeab5` on 2026-09-28: 504
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

## Current evidence and reference candidate

The Apple branches and subsequent owed/manifest fixes are on `main`, including
resample declarations `79559ad0f` and neighbors evidence `75ddcae2e`. The user
chose merge-then-check. The broad native run is frozen at `308878e80` on Apple
M4 and AMD: 504 inventory lanes, 445 base-fixture CPU/GPU comparisons, and 59
explicitly excluded physical parallel-device claims. It is not an exhaustive
nine-fixture qualification of the final main tree.

Historical CPU/GPU JSON records were recovered without rerunning models. The
recovery produced a candidate for 172 lanes (`bfe508d30` records the witnesses),
but the candidate has **not** been promoted to the shipped reference table.
Candidate eligibility must be evaluated against current lane revisions and
property declarations; the historical count is not a count of newly verified
current lanes.

Cross-platform comparison exposed failures that local GPU/CPU agreement had
missed. The four GLM lanes (`x-glm-poisson`, `x-glm-gamma`, `x-glm-tweedie`,
`x-glm-poisson-sw`) now use `portable-positive-target-exp-1`: identical NumPy
exponential arguments previously produced different Arm/x86 target words.
The two TreeSHAP lanes (`trees-dart-options`, `trees-shap-tree`) now use
`tree-shap-zero-cover-path-1`: unreachable paths previously produced `0/0`,
with different NaN signs on Arm/x86. Old records for all six lanes are excluded
from current admission until targeted new evidence is available. Do not
normalize NaNs or replace references to conceal these failures.

Follow-up native checks target the changed lanes and their properties; they do
not restart the full sweep. The GLM targets need all nine fixtures, while the
TreeSHAP regression specifically needs `base,negative`, including held-out and
batch SHAP where those parts are declared. A partial targeted record may
establish a repair without qualifying an all-nine-fixture reference.

## Producing a reviewable reference candidate

Clean lane JSONs must retain their original source, platform, input and property
metadata. Admission requires complete declared parts/protocols and two matching
witnesses; different device classes fitted once each can satisfy that rule.
Compare the Apple and AMD records directly as well as each local GPU/CPU pair.
A passing text log or local agreement alone is not cross-platform evidence.
The consolidated saved-record comparator reports omitted or incomplete scope;
retain that report alongside the per-arm JSONs.

Use existing complete records where applicable. Collect only the additional
records needed for changed revisions or missing coverage. Never admit
sabotaged, partial, failed, or superseded-revision records merely to clear a
hold. Preserve independent vendor evidence separately.

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
