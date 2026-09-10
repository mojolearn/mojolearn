# kNN selector lane — September 9, 2026 continuation

This supersedes the earlier unfinished selector handoff. Final safe source
is `28273628` with strict actual-arm gate `fa3a1cf7`; NVIDIA prices were
measured at `729ffa55` and the subsequent repair changes only Apple dispatch. Keep the root GEMM/attention matrix hunks.

See `bench/results/knn/2026-09-09-selector-resume/README.md` for provenance,
full evidence map, exact prices, and the measured cost of the Apple repair.

Accepted: butterfly selection, eight-load scan with ordered insertion,
NVIDIA constant-k=10/15 specialization. Rejected and removed: partitioned
selectors, deeper16/32 load scans, four-feature hoist, hardware FTZ FMA.
All final prices use the original software round-then-FTZ distance seam.

H100 400k/4000/k10 request 66.536 →40.665 ms, reducing the pinned same-model,
same-driver cuML ratio from6.51x to3.98x. All16 request shapes improve.
L40S UMAP1M total57.034 →48.377 seconds, embedding hash unchanged at
160286130194205729; one timed round each, no opponent ratio claim.

## Gates

The exhaustive selector gate scans long rows (8192/8193/65537) independently
for every next composite-key winner; 36 min/max cases cover k1/10/15/17/33/64,
duplicates, signed zeros, subnormals, infinity, and NaN payloads. Components
also compare the actual register distance tile across24 fixtures.

Root Apple ordinary-fixture gates passed: default and forced-specialized
36-case selector,24 distance fixtures, four143628-cell captures with SHA256
49c0f02513c2722db1855acd01d6d02941ccd29f4fb1b27d59a533db6b7350ce,
identity4/4, knn_main26, six kNN card stages, UMAP186/690 stage cells,
20k embedding12938647291752780014, and default/notile400k/1000/k15.
Root owns compressed Apple evidence from `/tmp/mojolearn-knn-final-apple`.

The new Apple boundary gate initially FAILED (preexisting FMA issue):
row0 gave0 versus expected8388608. The original failure is preserved.
The accepted Apple-only cold integer repair now passes the hardcoded8 and
strict396584-triple oracle. The strict production oracle also passed on H100
before that root-owned rental was terminated. NVIDIA arithmetic is unchanged.

Apple cost is explicit:400k/1000/k15, three timed rounds, request median
183.256 ms repair-disabled versus254.752 ms default (+39.0%); device168.122
versus238.294 (+41.7%). The original inline repair was rejected at13.3x cost.
The cold helper uses @no_inline and skips repair when ae+be>=151 proves the
exact product/accumulator lattice cannot round a subnormal up to minnormal.
The default retains this correctness repair. The opt-out flag is
MOJOLEARN_KNN_IDENTICAL_NO_ZERO_FMA_REPAIR, for reproducing the old price arm.

Final safe NVIDIA four-arm layout,5440 adversarial cells in each arm,
identity/main/card, and UMAP stage/20k gates pass. Layout ran separately after
a restart correctly refused to overwrite an interrupted evidence directory;
that orchestration failure is preserved alongside the successful fresh
`layout-safe` results. Root's final cold-helper Apple four-arm raw
hashes,24 distance fixtures, identity4/main26/card6, and UMAP186/690+20k
bundle all pass. Root owns the compressed archive under
`bench/results/knn/2026-09-09-selector-final/apple-cold-final` and the
three-round price under `apple-cold-price`.

## Reproduce

Use guarded own pod `tools/knn_selector_leg.sh`, credentials only through
`MOJOLEARN_RUNPOD_KEY_FILE` file path. Never touch samba/tree pods. For old
NVIDIA drivers, the on-pod harness selects system ptxas at build and run.
Use isolated `MOJOLEARN_KNN_ROOT` and fresh `MOJOLEARN_KNN_OUT` directories
when changing arithmetic; never reuse old compiled binaries.

```
MOJOLEARN_KNN_CHECKS_ONLY=1 bash tools/knn_selector_pod_run.sh gates
bash tools/knn_selector_pod_run.sh ref final
bash tools/knn_selector_pod_run.sh umap1m
```

The card check runs IDENTICAL kNN alone, avoiding the legacy both-mode wrapper:
```
MOJOLEARN_IDENTITY_TRACE=/tmp/knn.card MOJOLEARN_UNSUP_ARM=knn pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/unsupervised_trace_main.mojo
python tools/identity_trace_diff.py bench/results/e1/2026-08-28_122543-runpod-nvidia/e1u/knn.card /tmp/knn.card
```

## RUN OWED

No kNN lane gate remains owed. Final Apple ordinary fixtures, strict
boundary/oracle and price passed. Other Apple FMA consumers outside this
kNN register seam remain open and were not changed. This lane ran no Mac
builds/tests; root ran them. No FAST, DETERMINISTIC, or tree work.

All lane evidence was fetched. Own L40S almtzr6iqth0k9 was terminated and
verified DELETE204 / GET404 on September9 at22:21 EDT. No kNN rental remains.
