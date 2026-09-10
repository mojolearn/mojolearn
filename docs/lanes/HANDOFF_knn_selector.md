# kNN selector lane — September 9, 2026 continuation

This supersedes the earlier unfinished selector handoff. Final safe source
is `729ffa55`; the root integrated earlier selector commits and receives the
hardware-FTZ removal separately. Keep the root GEMM/attention matrix hunks.

See `bench/results/knn/2026-09-09-selector-resume/README.md` for provenance,
full evidence map, exact prices, and the important Apple boundary limitation.

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

NEW Apple boundary gate FAILS (preexisting FMA issue):
`neighbors/checks/knn_distance_fma_boundary_check.mojo` row0, got0 versus
expected8388608. Do not describe all correctness as green. NVIDIA safe
software arithmetic passes this boundary. Final safe NVIDIA four-arm layout,
5440 adversarial cells in each arm, identity/main/card, and UMAP stage/20k
gates also pass. The layout ran separately after a restart correctly refused
to overwrite an interrupted evidence directory; this orchestration failure
is preserved alongside the successful fresh `layout-safe` evidence. Any exact Apple repair needs a
fresh boundary/adversarial oracle gate before adoption; scope to kNN only.

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

Apple exact-boundary fix and gate remain owed. Ordinary Apple gates above
were run by root, not by this lane. No FAST, DETERMINISTIC, or tree work.
