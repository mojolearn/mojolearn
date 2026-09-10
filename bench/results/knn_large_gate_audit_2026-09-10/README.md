# Large kNN gate audit, September 10

Apple IDENTICAL, Mojo 1.0.0 (ed45d567), dyadic-v1. Both arms retain exact
zero-FMA repair: default uses complete-chain preflight; control forces
MOJOLEARN_KNN_IDENTICAL_NO_PREFLIGHT. No algorithm or default changed.

Root ran phase and ordinary-price windows sequentially under the shared build
lock at nice 19, with two build jobs and OMP/OPENBLAS workers. Normal desktop
activity remained; there is no continuous GPU clock/thermal telemetry and no
claim of a certified uncontended machine. Both windows are fixed four-shape,
two-order experiments, not searches for a favorable sample. Do not compare
absolute medians between phase and price windows to estimate instrumentation
cost: they ran at different times and phase timing drifted materially.

Commands from the isolated publication worktree:

```sh
pixi run --manifest-path /Users/andrewhendel/CascadeProjects/mojolearn/pixi.toml bash tools/knn_residual_phase_probe.sh apple /tmp/mojolearn-knn-large-phases-20260910
MOJOLEARN_KNN_PROBE_MODE=price pixi run --manifest-path /Users/andrewhendel/CascadeProjects/mojolearn/pixi.toml bash tools/knn_residual_phase_probe.sh apple /tmp/mojolearn-knn-large-prices-20260910
python3 tools/knn_probe_summary.py /tmp/mojolearn-knn-large-phases-20260910
python3 tools/knn_probe_summary.py /tmp/mojolearn-knn-large-prices-20260910
```

Phase ran the pre-price-mode script from aaab88be; its working patch is empty.
Price ran the archived working patch. Production Mojo was built from aaab88be;
subsequent source edits only correct stale comments/docstrings. Build-base and
final direct-file hashes distinguish these versions (not a full source closure).
Binaries are retained locally with hashes; build logs, individual samples,
full selected index/distance outputs and summaries are retained here.

Both modes pass eight paired full-output comparisons; summaries additionally
check equality across orders. Each process checks every request against its
warmup output and device output against the request. Phase has three timed
rounds; ordinary price has five; each region has two warmups.

## Ordinary large target results

400000 index rows / 4000 queries / 32 features. All values are medians in ms.
Pass 0 runs default then control; pass 1 runs control then default.

| k | Order | Default request | Control request | Default device | Control device |
|---|---|---:|---:|---:|---:|
|10|0|1380.890|1432.033|1300.699|1359.430|
|10|1|1305.022|1409.590|1282.292|1356.930|
|15|0|1332.921|1389.374|1293.975|1376.044|
|15|1|1351.913|1442.922|1329.189|1381.618|

Current preflight has 3.6–7.4% lower k10 request time and 4.1–6.3% lower
k15 request time in these two orders. Large-target last/first timed-sample
ratios span 0.9715–1.0276; retain all five samples, including intermediate
variation. These observations support retaining the existing gate on this
fixture; they are not confidence intervals or a new optimization claim.
The ragged small control has pronounced request noise in pass 1; it does not
drive the decision. Low-feature and ragged controls remain in the summary.

Phase diagnostics put distance ahead of selection and merge on the large
fixtures. This justifies testing per-vector exponent metadata reuse next;
it does not establish the size of a potential saving. Metadata reuse is not
implemented by this pass. No cuML/Torch timing was rerun and no opponent ratio
is assigned to these Apple measurements. Trees are untouched.
