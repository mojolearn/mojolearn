# Expanded NVIDIA sequence and UMAP qualification: failures retained

RTX 4090, driver 580.159.04, frozen source `5658d28e`.
The five baseline Mamba backward cases passed (54 captured gradient tensors).
The supplementary Mamba1 L64 case passed all 11 public-prefill gradients.
The supplementary certificate as a whole is RED: Mamba3 L65 still wrote its
old partial S17 manifest, omitting the public leaf inventory even though its
driver computed and wrote those gradients. The strict gate refused it.
The failed capture did not retain the full temporary native/oracle directories;
later tooling must preserve those diagnostic files before deleting a rental.

UMAP's six expanded held-out cases passed in IDENTICAL. In both FAST and
DETERMINISTIC, the two larger cubic fixtures (128 training rows, 128 queries,
seeds 3 and 41, 15 neighbors) failed during fitting with
`UMAP expects self in k-NN slot zero`. The other four cases passed.
These are fitting failures, not misses of a loosened quality threshold.
Public API, native sparse graph/fit, native transform and the two public kNN
dispatch checks passed. The fresh Mamba forward/state Python API also passed.

Classification: `ALGORITHM_FAILURE` for UMAP's same-data neighbor convention;
`INFRA_FAILURE` (qualification harness) for the obsolete Mamba3 L65 manifest.
Neither is a passing expanded certificate. Raw failing JSON,
logs, manifests and the successful native captures are retained under `remote/`.

The archive has no `.git` directory. Before the supplementary certificate,
the main controller wrote root `commit.txt` from the already validated pinned
baseline certificate revision; no numerical source was changed. See
[the provenance log](../../resume/2026-09-05-next-certification/nvidia-followup-provenance.log).
The future controller now passes the revision directly to follow-up jobs.

All artifacts were collected and pod 8u2ij3sq5boddm was deleted with verified
HTTP 404. The [controller log](../../resume/2026-09-05-next-certification/nvidia-expanded-controller.log)
retains the failing overall exit code and successful teardown.
