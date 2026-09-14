# Byte-LM optimizer-state pooling: two H100s

RunPod `rbtojh7e0esekh`, two H100 80GB HBM3, IDENTICAL, sm_90a. All
builds and tests ran on the pod with the R2 enwik8 corpus. No local execution.

Final production and fault builds use `byte-pool-qualified-source.tgz` at
`e3f0dec3f`, over the previous QR/Gram/pointwise overlays on `eaec62839`.
Binary/archive hashes and job scripts, logs and zero exit codes are retained.
The first gate failed because the harness assigned into an immutable exported
Array; replacing the whole seed moment arrays corrected that harness error.
The original build/source and failing log remain, alongside the final reruns.

Both one-layer and three-layer models pass pooled two-GPU vs replicated
two-GPU vs single-GPU ordered replay for three steps from nonzero moments,
nonzero weight decay and a resumed step count. Every parameter, moment, flag,
gradient and loss bit is compared. Native ownership reports verify moments
plus rollback allocations total 20 bytes per parameter across the group,
versus 40 with two complete replicas. Full parameter replicas and activations
are excluded from that memory number and remain allocated.

The default driver passes the existing K1/odd-K replay and checkpoint gate;
its complete report is unchanged from the initial replicated-default run.
`golden-comparison.json` also compares gradient/state hashes with the preceding
continued-H100 receipt. No changed arithmetic is inferred from timing.

Recovery checks cover malformed results after a complete native update,
invalid final-shard tokens, and one-GPU pooled checkpoint migration. A separate
`MOJOLEARN_BYTE_POOL_FAULT_INJECT` build plants bad gradients, refused moments,
nonfinite updated moments and negative updated moments on the SECOND owner.
The first owner has already updated for the moment/update faults. Every state
bit on both ranks is restored, and failed gradients remain uncommitted.

This qualifies optimizer moments/rollback pooling for these byte-LM shapes,
not full model capacity, cross-vendor identity, speedup, or lost-device recovery.
The pod remains leased for the shared neural optimizer pooling batch.

Final lifecycle: all owed jobs completed and evidence was downloaded before
RunPod `rbtojh7e0esekh` was deleted on 2026-09-14 at 18:11 UTC; GET returned
404. See `../pooled-source-freeze/h100-termination.log`. The exact final source
tree was frozen for subsequent RTX 5090 qualification.
