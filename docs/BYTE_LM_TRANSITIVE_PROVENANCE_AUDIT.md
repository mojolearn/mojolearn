# Byte-LM transitive Mamba provenance audit

Source/evidence inspection only, September7,2026. No tests, model execution,
comparators, measurements, API calls or digest calculations were run by this
author. Stored hash strings below were read from retained records; they are
not newly computed proofs. Old capture JSON/certificates remain unchanged.

## Root follow-up verification

Root subsequently passed a bounded physical-file check: all 45 Mamba Mojo
file hashes agree across the six full remote inventories. Both original
continuous archives reproduce those hashes, as does the retained resume
snapshot at `/tmp/mojolearn-byte-lm-resume-20260906`. Separate physical resume
archives remain unavailable; this distinction is retained in the record.
[Supplemental audit](../bench/results/resume/2026-09-07-root-byte-lm-do-amd/old-transitive-provenance.json)
and [resource receipt](../bench/results/resume/2026-09-07-root-byte-lm-do-amd/old-transitive-provenance-check.guard.json).

Future capture source at `eac39c36` inventories 258 files, including all 45
Mamba files, and refuses missing transitive representatives. Root verified
the actual prepared upload against that expanded inventory. Historical
213-file capture records are preserved with their narrower original scope.

## Finding

The original byte-LM `source_inventory()` walks `checks`, `core`, `gemm`,
`embedding`, `transformer`, `training` and selected binding/Python/tool/Pixi
files. It **does not include `mamba`**. This was confirmed by reading the
capture source directly from NVIDIA run2's physical `source.tar.gz`, without
executing that source. Therefore the old capture's reported213-file subset
must not be described as a complete transitive numerical-source inventory.

The same archive's Llama module imports `batchinv_norm_chunk`, `pinned_mul`
and `residual_add_kernel` from
`mamba/impl/transformers/models/mamba/modeling_mamba.mojo`. Its adjacent comment
explicitly says the Transformer build compiles that module and its scan file.
That module imports `mamba/checks/mamba_fixture.mojo` and
`mamba/impl/mamba_ssm/ops/selective_scan_interface.mojo`. Package initializer
files and other declared imports are additional provenance dependencies; the
three highlighted files are not a claim of exhaustive compiler dependency
closure.

The omission is from the **capture subset**, not from the retained full
transport inventories. The inspected records contain Mamba paths and their
stored hashes. Both continuous-run source tarballs physically list the three
highlighted members. This supplies independent retained evidence to audit;
it does not retrospectively enlarge the original capture comparator's checks.

## Records inspected

All links are within the source repository:

| Role | Full retained remote inventory | Source evidence |
|---|---|---|
| NVIDIA continuous128 | [run2 inventory](../bench/results/resume/2026-09-06-root-byte-lm-nvidia/run2/remote/source_inventory.json) | [physical source archive](../bench/results/resume/2026-09-06-root-byte-lm-nvidia/run2/source.tar.gz), original source d921eade |
| AMD continuous128 | [run3 inventory](../bench/results/resume/2026-09-06-root-byte-lm-amd/run3/remote/source_inventory.json) | [physical source archive](../bench/results/resume/2026-09-06-root-byte-lm-amd/run3/source.tar.gz), original source d921eade |
| NVIDIA head64 | [head4 inventory](../bench/results/resume/2026-09-06-root-byte-lm-resume-nvidia/head4/remote/source_inventory.json) | [retained record](../bench/results/resume/2026-09-06-root-byte-lm-resume-nvidia/README.md), transport3d255241 |
| AMD foreign resume128 | [resume2 inventory](../bench/results/resume/2026-09-06-root-byte-lm-resume-amd/resume2/remote/source_inventory.json) | [retained record](../bench/results/resume/2026-09-06-root-byte-lm-resume-amd/README.md), transport d5893e2a |
| AMD head64 | [head1 inventory](../bench/results/resume/2026-09-06-root-byte-lm-resume-amd/head1/remote/source_inventory.json) | [retained record](../bench/results/resume/2026-09-06-root-byte-lm-resume-amd/README.md), transport d5893e2a |
| NVIDIA foreign resume128 | [resume1 inventory](../bench/results/resume/2026-09-06-root-byte-lm-resume-nvidia/resume1/remote/source_inventory.json) | [retained record](../bench/results/resume/2026-09-06-root-byte-lm-resume-nvidia/README.md), transport d5893e2a |

Each head/resume directory also retains `source_inventory_local.json`,
`source_inventory_preship_remote.json` and `source_sha256_local.txt`. No source
`.tar.gz`/`.tgz` was found inside those four head/resume artifact directories
by filename inspection. Do not assume their physical archives remain available
merely because their original controllers shipped one. Root may locate an
external retained snapshot/archive separately.

## Stored values, not fresh hash calculations

The following exact strings appear for the corresponding paths in **all six
remote inventories listed above**:

| Path | Recorded SHA256 |
|---|---|
| `mamba/impl/transformers/models/mamba/modeling_mamba.mojo` | `b7c683d65a151ab3e43d51e0e32e48f4de51e135a2e0e65e762f393b989ad08a` |
| `mamba/checks/mamba_fixture.mojo` | `d4c9f21d2ea882d3f14fc1959af5d6a936bf359fec4e0126ec77375ee2121d4c` |
| `mamba/impl/mamba_ssm/ops/selective_scan_interface.mojo` | `ab24c17f729df1c17899b6ff98fb7e6eb1ad458798b9d2b5a55d8dc4d560b20d` |

The inspected inventories also contain matching stored entries for Mamba2,
Mamba3, Mamba-simple and SSD-minimal modules. Those examples show the transport
inventory was broader than the capture subset; they do not prove every Mamba
file or every physical archive byte matches. No automated map comparison was
performed for this audit.

Existing controller logs record source agreement, for example
[AMD resume2](../bench/results/resume/2026-09-06-root-byte-lm-resume-amd/resume2-controller.log)
and [NVIDIA resume1](../bench/results/resume/2026-09-06-root-byte-lm-resume-nvidia/resume1-controller.log).
These are historical controller assertions, not a new archive-byte check by
this audit. The resume records also say they reused the exact retained vendor
binding instead of rebuilding; preserve that binary/source distinction.

## What is established, and what root must still check

There is no disagreement in the inspected **recorded transitive-file hashes**
across the old NVIDIA/AMD continuous and head/resume records. The continuous
archives physically contain the dependencies omitted by capture `source.json`.
This narrows the concern to an incomplete capture inventory; it does not by
itself demonstrate a historical arithmetic mismatch or negate the retained raw
trajectory comparisons. Equally, recorded equality alone is not a freshly
verified statement that physical archived Mamba bytes match every run.

Root-only follow-up, under bounded file checks:

1. Recompute the selected archive-member hashes from both physical continuous
   tarballs, compare their bytes and bind them to the full recorded inventories.
2. Compare complete relevant Mamba inventory maps across all six runs and their
   local/preship/remote copies; inspect any mismatch rather than replacing it.
3. Locate original head/resume archives or exact transport snapshots where
   available. Confirm their physical transitive files and the retained binding
   chain, explicitly noting any missing physical archive.
4. Add Mamba to future capture inventories and freeze a new common source.
   New capture/comparator admissions must use that expanded contract; do not
   edit old `source.json`, receipts or certificates to make them appear to have
   checked dependencies they did not enumerate.

The existing [bidirectional result](../bench/results/resume/2026-09-06-root-byte-lm-cross-vendor/README.md)
remains its historical result with this provenance limitation disclosed. No new
Metal, broader model, training-speed or universal identity claim follows.
