# Installed API gap closure — September 6, 2026

Checkpoint: both GPU legs finished and deleted; final release remains blocked. Main operator alone runs every
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
All 45 CUDA/sm_89 extensions build and pass ISA/readback checks. The same
empty-directory repair issue is retained, with all 91 payload files unchanged
by normalization. Normalized wheel SHA256:
`805970b2bc44a002194cee66cbe378996f7e4a483aa3b6ca18a4f6da06060663`.
The frozen NVIDIA candidate has **19 passing jobs and five failures**:
Mamba FAST/DETERMINISTIC miss four reference tolerances each; Transformer
stalls in all three modes. The main operator stopped those stalled jobs
(SIGTERM, exit143) to continue the serial matrix within the lease. Original
failure logs and interruption records remain retained. No tolerance changed.
See [downloaded-artifact validation](nvidia-installed-validation.json).

The redundant inner 35-minute timer was removed without changing the candidate
process, source, compiler caps or one-hour deletion watchdog. A replacement
continuation reserved time for fetching; its exact script, original137 wrapper
exit, timestamps and final candidate failures are retained. The full build
completed at10:22 UTC and the full installed matrix at10:31 UTC.

## Installed UMAP and ordered identity

The [separate lane comparison](installed-lane-comparison.json) validates local
wheel RECORD/payload hashes, all45 build outputs, identical native and
qualification source inventories, all24 original job statuses, and installed
mode/binary provenance. All12 UMAP/ordered target jobs per vendor pass.
Six IDENTICAL UMAP fixtures match in 24 input/embedding arrays. The complete
OrderedRMSE model and all72 prediction cells match. The model SHA256 is
`934825a771807002ec0a9c6743c032fb1d7453060b847b85ddfcf8a8b4937c46`.
This diagnostic always keeps `overall_release_eligible:false`; the full
NVIDIA failure remains visible. Existing release admission was not weakened.

## Corrective source overlays

After the frozen matrix finished, the main operator alone ran serial bounded
follow-ups in a separate source tree. Full results and retained binary/log
hashes are under `nvidia-installed/`; [local verification](followup-validation.json)
checks every retained follow-up hash.

- `sequence-followup` records the initial compiler invocation failure because
  Pixi's environment was absent. It is not a native success.
- `sequence-followup-env-fixed` builds all six Mamba/Transformer mode bindings.
  The context lifetime pin removes Transformer stalls. IDENTICAL passes all
  102 Mamba and44 Transformer checks. Full-FP32 Mamba projections remove all
  three block-output accuracy misses in FAST/DETERMINISTIC; one Mamba3 k_last
  mismatch remains in each. Transformer non-IDENTICAL still has two accuracy
  misses before its matrix products are corrected.
- `transformer-fp32-followup` applies full-FP32 to all nine Transformer matrix
  products. FAST/DETERMINISTIC then pass all44 checks each. IDENTICAL was
  tested with the preceding lifetime patch, not rebuilt after these final
  compile-time dispatch arguments. A final-source full matrix remains pending.
  `execution-command.sh` is the actual command; `command.sh` was copied from
  the preceding six-binding script by its inherited recorder.
- `mode-persistence-followup` installs the frozen wheel into a separate
  environment and overlays only the Python serialization wrapper at29a8c848.
  With the process default changed before load, all three real ordered gates
  pass and retain the original model/prediction bits. This modified installed
  Python payload is explicitly not a release-wheel qualification.

The final native corrections are committed at3963a0fc. IDENTICAL retains its
existing kernel plans; new cross-vendor backward certification is not inferred.
Non-IDENTICAL AMD/Apple paths changed and require refreshed qualification.
The remaining Mamba3 failure is flat151 of k_last: -0.0214189123 versus
-0.0214173001, exceeding the existing tolerance by3.980e-7. Next inspect public
theta_last across modes to distinguish angle arithmetic from BC normalization.

## Teardown and retained artifacts

All candidate and follow-up artifacts were fetched before release; local
payload/hash checks completed before resuming the paused controller. NVIDIA
DELETE returned204 and verification returned404. The final
[cloud inventory](cloud-cleanup.json) confirms both named resources absent and
both provider inventories empty. The pre-existing dirty AMD log was restored
byte-for-byte from its original backup.

Git retains the normalized NVIDIA wheel, all build/qualification proofs and
logs, and follow-up binaries. Redundant raw/repaired NVIDIA wheels and the
staged native set are retained locally; their hashes and repair provenance
are recorded. AMD's earlier checkpoint also retains its original wheels.

## Scope and follow-up

`OrderedRMSE` is numeric, single-permutation RMSE, not full CatBoost ordered
boosting. Python Mamba backward, corrected Apple backward qualification,
full categorical/ordered parity and release publication remain separate.
These are exact candidate-wheel checks on named architectures, not universal
GPU support or identity. PyPI has not been updated by this work.

## Fingerprint follow-up

The frozen qualification snapshot included the top-level Mamba1 corpus but
used nonexistent top-level paths for the nested Mamba2/3 corpora. The gates
ran from the frozen commit; their nested fixture files were not individually
fingerprinted. This omission is retained as a limitation of the old records.
Future snapshots include all38/63/67 files and refuse missing corpora. Final
release admission now also requires these current corpus hashes. Historical
manifests were not rewritten. The separate UMAP/ordered comparison does not
make a Mamba corpus or release claim. See [local follow-up checks](provenance-followup.json).
