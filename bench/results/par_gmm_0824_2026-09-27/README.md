# par-gmm on two physical GPUs, published wheel 0.8.24 (2026-09-27)

0.8.24 changed GaussianMixture's kmeans init (DEVIATION 3133), so the IDENTICAL bits of gmm,
gmm-sample and par-gmm moved on purpose. `verify --par quick` does not include par-gmm, so this
run takes it on its own.

## Box

RunPod, 2x NVIDIA L40S (sm_89, driver 580.178.04), $2.18/h. No 2x RTX 4090 were in stock (create
refused, no pod made). The box was rented through `tools/gemm_remote_leg.sh nvidia --rent` with
`MOJOLEARN_GEMM_LEG_GPU_COUNT=2`, and `body.sh` was the `MOJOLEARN_GEMM_LEG_EXTRA` body. The body
builds a fresh venv outside the checkout and runs `pip install mojolearn==0.8.24 numpy` from PyPI
(this pulls mojolearn, mojolearn-nvidia and mojolearn-amd 0.8.24; see `wheel_record.txt`).

- GPU 0 `GPU-7bfd23a8-de13-8ac8-59d8-c5fd16611b13` (PCI 45:00.0)
- GPU 1 `GPU-3ed32994-f022-2653-d113-a4b5e43ded9b` (PCI 83:00.0)

Lease 1 (pod 521y9g988tfu80) measured nothing: the body's import check caught a tcmalloc warning
on stderr and refused a correct install. Lease 2 (pod rr0bimme29mpkn) ran after the fix. Both
pods show `VERIFIED: <pod> is gone (HTTP 404)` in `*/runner_teardown.txt`. Each pod was up for
about 4 minutes, roughly $0.30 for both.

## Results (lease2-l40s/)

| step | exit | result |
|---|---|---|
| `verify --par --par-self-test` | 0 | SELF-TEST PASSED, 4 perturbed parts DIVERGENT |
| `verify --par --lanes par-gmm --json` | 1 | DIVERGENT against the shipped table, as expected: the table predates DEVIATION 3133. The witness saw 6 pools and physical placement on both UUIDs. |
| `direct.py`: par-gmm through the verifier's own `par_check` at devices 0, 1 and 0,1 | 0 | all three give the same result, see below |
| `verify --lanes gmm,gmm-sample --fixtures base` | 4 | STALE REFERENCE, not compared, the same known reason |

Cell hashes for par-gmm/base, the same at devices `0`, `1` and `0,1`:
train `7d625382ed9cf9b5`, infer `1acafa8144b5d947`, model `8663c7a3c9da8585`,
batch `ea3e4a064ad2b07e`. The two-device run is BIT-IDENTICAL to the one-device run.

The par-gmm sub-parts (weights, means, covariances, precisions, n_iter, lower_bound, labels) equal
the plain `gmm` lane's sub-parts from the same box. They also equal the 0.8.24 release columns'
`gmm/base` parts on CUDA (smoke-linux/column-cuda.json), HIP (column-amd/column-hip.json) and
Metal (release-check/c8654671b848/metal/column.json), all at commit c8654671b. The model hash
`8663c7a3c9da8585` also equals `gmm/base`'s model hash in all three columns. No release column
carries a par-gmm cell, because `par-*` lanes are excluded from records. The par-gmm train, infer
and batch cell hashes therefore have no release counterpart. They hash a different set of keys
and probes than gmm does.
