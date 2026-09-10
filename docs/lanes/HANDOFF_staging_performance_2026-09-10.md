# September 10 staging performance continuation

Continues main `d557b851`; trees and the concurrent `checks/fixed_point.mojo`
changes are excluded. IDENTICAL arithmetic remains fixed. Root owns serial
Apple tests and one H100; independent lanes supply kNN and Mamba candidates.
No opponent is retimed. Raw evidence and reproduction scripts are in
`bench/results/staging_performance_2026-09-10/`.

## GEMM operand staging

The tuned loader now applies the existing bitwise input FTZ seam before
shared staging. Consumers then read the same flushed operand word. This
removes repeated consumer-side bit tests; the ascending FMA chain, corrected
round-then-flush accumulator seam, leaf boundaries, and fold tree do not move.
Both loader paths (including outer-contiguous early return), both shared
page configurations, and full/ragged consumers use the same policy.

NVIDIA IDENTICAL is enabled by default. Apple/AMD retain their prior default;
`MOJOLEARN_GEMM_STAGE_FTZ` opts in and
`MOJOLEARN_GEMM_LEGACY_STAGE_FTZ` supplies the baseline. The capability choice
uses the existing hardware row, not an independent vendor constant.

Existing seven GEMM device gates pass Apple and H100. A new independent
`gemm_stage_ftz_check.mojo` uses actual signed subnormal operands multiplied
by large normal values so a missed input flush produces visible normal
nonzero output. All19 plans, three orientations, seven K lengths, both operand
roles and output tails match flat-device and host oracles:798 cases and447678
output words on both GPUs. The literal-zero witness does not depend on the
oracle. This closes a test gap; existing tiny-product fixtures mostly used
normal inputs. A first Apple compile used the wrong DeviceContext import;
corrected to the repository's max.gpu.host before qualification.

H100 seven-round runs in both orders retain complete output fingerprints:

| Shape | Baseline first ms | Candidate second ms | Candidate first ms | Baseline second ms |
|---|---:|---:|---:|---:|
| qkv.t512 |1.942260|1.522974|1.523785|1.940172|
| mlp_up.t512 |7.584773|6.171459|6.177937|7.581844|
| mlp_down.t512 |6.779385|5.355679|5.356788|6.791881|
| pca.transform.wide.8192x64x128 |0.033404|0.025224|0.025159|0.032999|
| kmeans.dist.4096x64x64 |0.022915|0.019414|0.018727|0.022839|

The dense improvements are18.5–21.6% of same-pod baseline time. Existing
cuBLAS rows remain cached references with their original shape/runtime scope;
these are not newly paired opponent runs. Apple performance was not measured.

## kNN experiments

The first candidate coalesces register-column ownership. It passes exact
Apple/H100 adversarial and public-layout gates and all20 complete price-output
comparisons, but target k10/k15 requests do not improve. It is removed from
production and archived with patch, gate and driver. The small32-query control
costs4.2% more device time; the ragged control improves1.9%. No default change
is justified by those results. The complete-output benchmark dump is retained
for future comparisons. See `KNN_COALESCED_COLUMNS_2026-09-10.md`.

A second candidate tests512-query batches and accounts for the actual
65536-column distance buffer instead of the complete index when enforcing the
distance-buffer limit. It retains every row's distance, selection and merge
arithmetic, while increasing peak scratch use. H100 both-order pooled
request savings are5.8% k10 and5.2% k15; final default seven-round prices
are27.525704/31.860726ms. The NVIDIA-only default is bounded to400k rows
and starting tiles<=512; larger indices or explicit tiles retain historical
budgeting. Known batch-dependent scratch is <=522.6MiB in the admitted
scope. Apple stays256. Native planner/output/public-layout checks pass.

Python's previous default explicitly passed256, so it now passes0 for
automatic planning. Positive explicit tiles remain honored. Rebuilt Apple
and H100 bindings pass nearest-neighbor, classifier/probability and regressor
exact-output gates against explicit256; the observed automatic tile is256
on Apple and512 on NVIDIA.

## Mamba scratch and transformer

Mamba's candidate skips zero initialization only for27 stages with complete
producers before every admitted consumer. Six split-producer working arrays
and all recurrent/pending state retain zeros. The public stage constructor
keeps its zero default; the binding can request scratch explicitly. The
per-buffer producer/tail/lifetime audit is in
`HANDOFF_mamba3_scratch_2026-09-10.md`.

Apple and H100 baseline/poison-filled scratch produce byte-equal native
default and long traces; decode-cross, continuation and refusal gates pass.
H100 also passes all public gates and six complete-output SHA comparisons.
The first uninitialized narrow case costs236.131446ms versus56.767391ms;
restoring poison fills returns56.415820ms. Later the identical original
baseline library also enters a225–250ms narrow regime, so the first contrast
cannot attribute a4x regression to scratch. Wide is104.340496→103.713224ms;
tiny is1.188908→1.068041ms. This does not justify a default or a tiny-shape
carveout. The implementation is removed and archived with its driver/audit.
No cause for the narrow regression is established. GPU jobs were strictly
serialized; allocation/first-touch/synchronization remain hypotheses.
The final end-to-end tests use the GEMM change alone. The first H100
attempt stopped because git archive excluded the historical grid helper;
explicit fixture staging fixes the harness without changing production.
Transformer cache scratch was reviewed and deliberately retains zeros because
whole-buffer cache copies expose unused capacity. Transformer's original
Torch numerical admission remains unqualified.


## Final full-block qualification

GEMM-only final builds preserve all three original Mamba complete-output
SHA256 values, both transformer large-output hashes, all82 transformer
array hashes, and public refusal/surface gates. Transformer narrow paired
medians206.074415→202.123621ms and206.111947→200.625729ms; wide
191.636596→172.340987ms and189.356467→168.650821ms. Thus observed
savings are1.9–2.7% narrow and10.1–10.9% wide. The original Torch numerical
admission remains failed; no qualified opponent ratio is claimed.

Initial wide Mamba pairs show102.724629→95.521010ms and
103.468360→93.319228ms. However the final isolated same-binary repeat
(baseline/default/baseline) gives narrow248.337356/251.352344/250.333956ms
and wide102.489213/97.158484/283.757282ms. Every output hash is unchanged;
the baseline libraries are SHA256-equal. Both large shapes therefore lack
a stable new qualified speed/ratio claim. No cause is established. Complete
samples, GPU before/after reports and process snapshots are retained.
This is a real unresolved own-side measurement issue, not a reason to rerun
opponents or loosen numerical admission.

## Runtime and artifacts

One H10080GB HBM3, UUID GPU-504d7226-23e4-42fe-6ed9-64586e4da2e2,
driver580.126.09, Mojo1.0.0(ed45d567), public Python3.11/NumPy1.26.3.
GPU jobs strictly wait for predecessor exit records. All owned work used
pod kpq865uami33s3, created06:34EDT with a60-minute self-kill. No clocks
were changed. Local tests use the shared build lock. The raw initial harness
failure, every rejected candidate and both-order samples are retained.

The raw logs/cells above128KiB are gzip-compressed without changing bytes;
SHA256SUMS covers the shipped artifacts. Decompress those files in a copy
before running parsers that expect raw .log/.cells paths. Large Mamba and
transformer tensors are represented by hashes over every output byte; small
kNN complete output dumps are retained. Source and artifact hashes are
verified against the final checkout.

The pod was deleted07:03:54EDT (HTTP204), then verified absent07:03:59
(GET404). All evidence was fetched before deletion. Nine completed device
jobs have exit0; the earlier missing-helper attempt remains explicitly
recorded as failed. Final GEMM/Mamba/transformer/kNN source hashes match the
checkout, and all six public kNN result arrays match across Apple/H100.

Main changes: GEMM `c89a73d8`, bounded kNN default `de042700`, Python automatic
planning `78248bb7`, public Python gate `a8f0ee88`. Rejected candidate source
and drivers are preserved under the evidence's rejected-knn/rejected-mamba
subdirectories. No GPU or owned job remains active.
