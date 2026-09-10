# September 10 performance continuation

Scope: IDENTICAL Mamba3, kNN, transformer admission and GEMM experiments.
Tree sources and behavior were not changed. FAST/DETERMINISTIC paths were
preserved; this lane ran no tests of those modes.

Mamba3 now initializes discarded fresh-prefill state on device and copies
live caller arrays directly, preserving the stateful path, public reports,
input lifetimes and per-call mutable-weight refusals. Same H100 before/after:

| Public shape | Baseline ms | Final default ms | Less time |
|---|---:|---:|---:|
| B8/L4096/D512 |87.192103|70.828952|18.8%|
| B8/L1024/D2048 |166.113550|124.409189|25.1%|

Final default NVIDIA checks cover39,087,232 exact output/report cells,
full-output SHA, mutable refusals and102 Python surface checks. Rebuilt
Apple defaults pass146,560 cells and102 surface checks. The admitted
archived torch scan gives4.32x/6.42x; parity remains unmet. That torch call
uses resident device inputs and returns device y, whereas ours includes
host transfers, per-call weight checks and four public reports. The baseline
and final use the same current physical pod/runtime, avoiding attribution
of differences between today's host conditions and older runs.
See `bench/results/mamba3/2026-09-10-fresh/`.

kNN now uses correctly rounded NVIDIA FMA followed by hardware FTZ
multiplication by one. Apple proves the complete dot-product chain cannot
need its expensive zero-result repair before selecting the simpler loop;
unproved chains retain the exact repair. The selector remains the
September9 version: redux/key-recovery changes were too small to retain.

| H100400k index,32 features | Pristine5011239a ms | Final default ms | Less time |
|---|---:|---:|---:|
|4000 queries,k10|49.452488|39.310253|20.5%|
|4000 queries,k15|54.973115|44.817340|18.5%|

The clean control confirms helper extraction did not create an artificial
baseline regression. Final medians use seven timed rounds; all16 grid
shapes pass. Strict396,584-case three-arm oracle, eight literal boundary
cases, long-row selector and24 distance-layout fixtures passed. Index and
distance fingerprints remain unchanged. The reused cuML k10 request
reference10.225ms gives3.84x; it was not rerun on this physical pod.

Apple alternating5-round runs at400k/1000/k15 show approximately6–10%
less device time. Three additional d/query shapes improved device time in
both orders. Host load was variable and one small-shape request timing
regressed despite a device win, so these do not establish a universal
end-to-end speedup or recover the full previous39% correctness cost.
Rejected four-feature/redux/key-recovery candidates are archived but
removed from production. Final Apple defaults pass396,584 oracle cases,
eight boundaries,24 layouts and a rebuilt public Python kNN smoke with
exact dyadic distances/tie ordering/repeat bits at k1/10/15/33.
See `bench/results/knn/2026-09-10-residual-final/`.

Transformer admission is now localized more narrowly. The pinned inverse
frequency table differs from the original NumPy-powered table in35of64
HD128 words. Sharing the complete production RoPE tables reduces the
original errors but does not admit either shape. Both FP32 implementations
also fail the same tolerance against a matched FP64 diagnostic with
comparable errors. This is consistent with remaining FP32 rounding and
fixture conditioning, not a stage-by-stage proof of every residual.
No arithmetic, comparator default or tolerance was changed, no new opponent
time was measured, and no qualified transformer ratio is claimed.
See `bench/results/transformer_admission_2026-09-10/`.

No GEMM experiment was accepted. Smaller local fold stacks produced no
useful dense win. The existing64x64 tile lost to128x128 under corrected
arithmetic. Per-fragment and whole-tile exponent proofs passed device
checks but their checking/code-generation costs made kernels slower.
All rejected patches, scripts and raw seven-call mean probe timings are
retained under `bench/results/performance_residual_2026-09-10/` so these
ideas need not be repeated. Corrected RN-then-flush GEMM remains intact.

The opponent table now includes the archived three Mamba3 rows, eight
k=15 cuML rows, failed L40S transformer measurements and their admission
metadata. Every backfilled numerical price was checked against its raw
log/JSON. Future measurements must record the exact tuple, fixture,
timing scope, samples and admission status; valid existing rows are reused.
No opponent was retimed during this pass.

All owned GPU work used one guarded H100, gdkumy79olopt6, driver580.126.09.
The pod was deleted on September10 at04:55EDT; GET returned404.
No clocks were changed. Earlier differences between Mojo1.0.0 and MAX26.5.0
labels do not demonstrate compiler drift: they name separate packages,
and the earlier run already recorded the same Mojo hash. Apple/NVIDIA
validation is recorded; no AMD run was performed in this pass.
