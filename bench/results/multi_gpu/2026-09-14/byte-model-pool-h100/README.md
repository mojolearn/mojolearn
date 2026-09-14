# Same-source H100 / RTX 5090 model-pool replay

2026-09-14, RunPod `lgie1o62m251xp`, two H100 80GB HBM3 GPUs,
driver 580.126.09, IDENTICAL sm_90. Source archive SHA256:
`cd53edd9ba9d8973d490ce47a777c68d30d91a2f16b0448caf920fb4d31e9094`.
It is byte-for-byte the archive retained under
`../byte-model-pool-rtx5090/model-pool-v2-out/source-780b7b2b4.tgz`.
Both use the same R2 enwik8 corpus, SHA256
`2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`.
No local builds or tests.

The complete structured receipts match for all nine configurations:
1/2/3 decoder layers crossed with 1/3/5 logical microbatches. They include
canonical ownership, final full-state SHA256 (parameters, moments and flags),
gradient SHA256 and committed loss-history SHA256. `comparison.log`,
`cross-architecture.json`, the comparison script and RTX reference JSON files
are retained. Both architectures independently compare the model-pooling path
with one-GPU ordered replay, checkpoint migration and transaction recovery.
The native five-fixture gate also passes, including four layers/length 33.

The named hardware is two NVIDIA architectures, not two different GPU vendors.
The RTX receipt separately qualifies injected native faults and actual
beyond-one-GPU model capacity; those larger/fault runs were not repeated here.
No new AMD/Apple, eight-GPU or throughput claim follows from these checks.

The H100 build/gates and comparison completed successfully before termination;
see `job-status.txt` and `termination.log`.
