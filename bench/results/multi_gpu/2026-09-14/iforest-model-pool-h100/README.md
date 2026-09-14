# Resident IsolationForest models — two H100s

Cloud checks passed on RunPod `smqlvlvt7exixd`, two H100 80GB HBM3 GPUs,
driver 580.126.09, Mojo/MAX 26.5, 2026-09-14. No local builds or tests ran.

Tree owners retain their node buffers through scoring. The root holds only
one-cell placeholders. The original per-row FP32 accumulator visits all trees
in canonical order across owners, with the original global division once at
the end. Separate owner sums are never combined. Full training/query data are
still replicated; tree scratch and model allocations must fit their owners.
The original global int32 node-count bound remains. This receipt qualifies
ownership and recorded arithmetic, not a beyond-one-GPU capacity run or scaling.

The native gate checks four bootstrap/feature-subset configurations. Every
float threshold byte and all seven integer model arrays match the original
forest, with tree offsets rebased only for comparison. It verifies complete
owner coverage and release of full root buffers on a single-to-multi-device
refit. Path lengths and scores match exactly. A planted leaf-payload witness
catches independently summing owner totals instead of carrying the accumulator.

Eight public IsolationForest configurations cover 1/5 trees, bootstrap,
feature subsampling and automatic/fixed contamination. Fitted fields,
score_samples, decision_function and predict match the original path. Invalid
fit input preserves prior caller state. Eight SVC/SVR configurations also pass
after rebuilding their shared binding. Complete JSON receipts equal the prior
IsolationForest and SVM references; `out/comparison.log` records that check.
The historical scope text in those unchanged JSON reports still describes the
older assembled-root implementation; the updated native gate records the new
resident ownership and root placeholders.

`out/` preserves build/native/public logs, complete JSON reports, references,
hardware/corpus identities, binary hashes and exact job scripts/return codes.
The source overlay is `out/iforest-pool-overlay.tgz`, SHA256
`cea95542cd38fbd309ffd296988cd48b2c2faac919ee33981e0cf298935b6897`.
It applies to the frozen source plus lifetime fix documented in the neighboring
`neural-clip-pool-h100/README.md`. The SVM binding and native gate both built
successfully on the first attempt. No new cross-vendor execution is claimed.

The pod remains leased for subsequent work; termination is recorded separately.
