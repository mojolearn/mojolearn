# Tall full PCA and TSQR: two H100s

RunPod `rbtojh7e0esekh`, two H100 80GB HBM3, IDENTICAL, sm_90a.
All compilation and execution occurred on RunPod. Both jobs exit zero.

The native gate compares every bit of destroyed input, stacked-R scratch and
final R for 17x7, 259x7, 259x17, 1025x65 and 4099x7 matrices, exercising
1, 8, 2, 2 and 64 original panels. Includes subnormals and signed zero.
The public gate passes eight full-PCA fits with/without whitening, comparing
complete fitted state, transform and inverse transform, plus failed-fit
atomicity. Original panel arithmetic and root stacked-R factorization remain.

`qr-source.tgz` captures the overlay at `7c5b0c567`, applied after the qualified
Gram overlay `7c18b7a6f` and pointwise updates on initial source `eaec62839`.
Those preceding receipts record the complete source chain. Build logs, exact
source archive, binary hashes, gate output and job scripts/exit codes are kept.

No local tests/builds. This qualifies same-hardware compute partitioning, not
new cross-vendor identity, speedup or pooled root/model capacity. The one-panel
path remains unchanged; wide full PCA's transpose-QR route remains single-device.
The pod remains leased for the next neural optimizer-pooling implementation.

Final lifecycle: all owed jobs completed and evidence was downloaded before
RunPod `rbtojh7e0esekh` was deleted on 2026-09-14 at 18:11 UTC; GET returned
404. See `../pooled-source-freeze/h100-termination.log`. The exact final source
tree was frozen for subsequent RTX 5090 qualification.
