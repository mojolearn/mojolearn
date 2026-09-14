# Streamed Samba checkpoints — H100 cloud gate

The final gate passed on RunPod `smqlvlvt7exixd`, H100 80GB HBM3,
driver 580.126.09, Mojo/MAX 26.5, 2026-09-14. All execution was in the cloud.

The new format writes canonical metadata and checksummed little-endian arrays
in 1 MiB chunks. Metadata is bounded to 1 MiB; total archive size is not capped
at 256 MiB. Saving borrows arrays and atomically replaces the destination.
Loading validates registry, file size and every array checksum before model
construction. It avoids per-tensor parameter copies and defaults to the saved
numeric mode. Caller ownership during save and full host arrays on restore
remain requirements. Legacy JSON remains readable through its bounded reader.

The public fixture trains an attention Samba stack with dropout, clipping,
a learning-rate schedule and populated Adam moments. Save/load reproduces
complete state, and the following training step agrees exactly. Seven corrupt
header/data/registry/length fixtures refuse before model construction. An
injected publication failure leaves the old checkpoint intact and removes its
temporary file. Legacy roundtrip also passes.

A separate admitted Samba registry has 34,607,616 parameters and produces a
415,293,679-byte archive. All four arrays roundtrip bitwise, including signed
zero and subnormal payloads. Archive SHA256:
`c3d724f8be45d288b1797bc276340ba5b4b7c80655b93a7fdf318c110768bd33`.
This is host checkpoint capacity, not a large-model GPU training result.
The existing complete clipped attention/dropout parallel-replay JSON remains
unchanged (`out/comparison.log`).

`out/` contains final receipts, logs, hardware/corpus identities, binding hashes
and exact job scripts/return codes. Source overlay `out/samba-checkpoint-final.tgz`
has SHA256 `b0af02e05a8cf584f3bc93850edc5d307e3f8e14cf8680fe84472cb93f13f313`.
It applies to the qualified neural clipping source documented in
`../neural-clip-pool-h100/README.md`. The final checked-in module additionally
updates only its descriptive docstring. `initial/` preserves the successful
initial run before the final loading/header refinements; both versions pass.
No new cross-vendor or throughput claim follows from these storage checks.

The pod remains leased for subsequent work; termination is recorded separately.
