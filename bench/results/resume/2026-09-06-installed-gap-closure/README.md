# Installed API gap closure — September 6, 2026

Checkpoint: AMD complete; NVIDIA in progress. Main operator alone runs every
build, test and measurement. Agent contributions were static code edits only.

Native/package source: `eb835021dcd79a59a7e8f78c754a75db3c1fea83`
(implementation `29f21791`). Controller-only fixes are in `fc0ed746`;
future automatic wheel normalization and the ordered comparator are in
`540c5ae9`. Both GPU legs use the frozen native/package source above.

## AMD installed artifact

Canonical evidence:
[qualification-normalized](../../e1/2026-09-06_091452-mojolearn-e2-amd/diag/candidate/qualification-normalized/qualification.json).
All 45 extensions build, report HIP/gfx942, and pass the CPU ISA audit.
All 24 installed jobs pass: smoke, UMAP fit, transform, expanded held-out
quality, ordered RMSE, Mamba, Transformer and fitted ARIMA, each in FAST,
DETERMINISTIC and IDENTICAL. Each job records all 15 installed binding hashes.
The six UMAP quality fixtures retain inputs and embeddings as raw uint32 bits.
Ordered RMSE retains full serialized model text and prediction bits.

Qualified normalized wheel SHA256:
`7c5f9af825cbcbd74a293adc75ad15670a30a179d3a9a8a8cfd993cd9476f7be`.
Native build source inventory SHA256:
`ade965b90496132596d8dda79860a87f472193c14129093c5c3a72f273b8159f`.
See [local downloaded-artifact validation](amd-installed-validation.json).

The first candidate stopped before model execution because auditwheel added
eight empty ZIP directory entries absent from RECORD. Its original FAILED
status remains retained. Normalization removed only those directory entries;
all 91 payload files, including binaries, RECORD and platform metadata, are
byte-for-byte unchanged. Both original and normalized wheel hashes and the
normalizer source are retained in the candidate directory. The fresh
`qualification-normalized` directory is the passing qualification; the
original `qualification` directory is not a pass.

AMD MI325X droplet `598204069` ran serially from 09:13 UTC and was deleted at
09:40 UTC; post-delete GET returned HTTP 404 at 09:40:46. All artifacts were fetched
and locally validated before releasing teardown.

## NVIDIA continuation

RTX 4090 pod `xbwb0ksmgel1w8` was created only after AMD deletion. Its lease
watchdog is armed. The initial source upload was slow, so the main operator
paused the local controller and stopped only its upload SSH child, retaining
the independent local and on-pod watchdogs. A direct GitHub fetch verified the
same frozen commit in `/root/mojolearn-frozen`; the incomplete uploaded tree
is not used. Builds are serial with CPU affinity 0–3, two compiler workers, and one BLAS/OpenMP thread.
No NVIDIA installed qualification or cross-vendor claim is recorded yet.

## Scope and follow-up

`OrderedRMSE` is numeric, single-permutation RMSE, not full CatBoost ordered
boosting. Python Mamba backward, corrected Apple backward qualification,
full categorical/ordered parity and release publication remain separate.
These are exact candidate-wheel checks on named architectures, not universal
GPU support or identity. PyPI has not been updated by this work.
