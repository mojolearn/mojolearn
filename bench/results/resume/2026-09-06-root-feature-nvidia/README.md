# Root-only feature continuation — 2026-09-06

Status: first NVIDIA campaign finished; artifacts recovered and RunPod
deletion confirmed (DELETE 204, subsequent GET 404). No accepted timing
comparison resulted from this rental.

The retained `run/remote/feature-finish/` and
`run/remote/feature-supplement/` directories distinguish the frozen campaign
from subsequent root-run harness corrections. NVIDIA IDENTICAL Mamba passes
102 API checks. Mamba2/3 backward passes all five surface tests, including
the 20 exposed gradient tensors against native dumps and an independent
CUDA float64 reference. UMAP passes 15 API tests in each mode, direct FAST
and IDENTICAL optimizer-control checks, finite-parameter and stage-identity
checks. The corrected FAST GBDT build and boundary checks pass.

Remaining failures: one Mamba3 FAST state accuracy check; Forbidden-NaN
check compilation; cuML shared-library loading. CatBoost comparison was
stopped by the GPU-memory guard, so its partial timings are not admitted.
Source fixes are prepared but are not validated by these earlier results.
The next validation must run remotely on NVIDIA/AMD; no new Apple tests.

Second NVIDIA run: `run2/`, frozen validation snapshot
`4a271ae6d719c79e2e871f4776b74c9a12f380ee`. SSH timed out during source
upload before tests started. Cleanup confirmed DELETE 204 and GET 404 with
57 minutes remaining on the lease. No validation result came from this
attempt. The frozen payload includes the Mamba3 FAST
dt-softplus candidate, corrected native NaN check, native backward dumps,
CatBoost's GPU allocation limited to 25%, and the isolated cuML loader path.
These changes still require remote validation.

Retry: `run3/` used the same frozen snapshot on a different RTX 4090 host
(driver 580.159.04). **26 of 27 jobs passed**; only external UMAP library
loading failed. All correctness jobs passed, including 102 Mamba API checks
in each mode, both native backward captures, the five Mamba2/3 backward
surface tests, both UMAP API suites, Forbidden-NaN and all native UMAP checks.
The Mamba3 FAST dt-softplus repair is validated on these fixtures. This does
not qualify a new installed wheel or DETERMINISTIC mode.

The GBDT comparison passed all seven timed rounds plus warmup and all
24 held-out quality admissions. IDENTICAL output hashes remain stable.
This is a small matched numeric symmetric-RMSE fit-plus-GPU-prediction case:
1024 float32 rows/8 features, 768 train and 256 held out, 16 trees, depth 4,
learning rate 0.1, L2=3, 32 borders, and exactly one external, CatBoost CUDA.
Independent binning is retained; no cross-library bitwise equality is claimed.

| Arm | Median milliseconds | IQR milliseconds |
| --- | ---: | ---: |
| MojoLearn FAST | 25.570568 | 4.6132175 |
| MojoLearn IDENTICAL | 24.722147 | 6.044117 |
| CatBoost CUDA | 155.329897 | 3.143613 |

All held-out RMSE values are approximately 0.389033, below the declared
quality limits. These timings do not establish throughput for larger trees,
datasets, other objectives or categorical features. See the exact
[comparison record](run3/remote/feature-finish/gbdt-three-arm/results.json)
and [job statuses](run3/remote/feature-finish/results.tsv).

UMAP produced **no accepted comparison**: cuML's CUDA 12.9 solver loaded an
incompatible cuBLAS, reporting missing `cublasSetEnvironmentMode`. The next
targeted runner isolates the external CuPy/cuML process from Torch and puts
its matching venv CUDA libraries before Pixi fallback. That fix is not
validated by run3.

Artifacts and [frozen source inventory](run3/source-snapshot.json) were
recovered, with the complete frozen source in `run3/source.tar.gz`.
Deletion was verified (DELETE 204, GET 404), with 28 minutes left on the lease;
see the [controller log](run3/controller.log). The overall campaign remains
RED because UMAP comparison failed; GBDT's independently admitted result is
retained separately.

Three implementation agents were explicitly forbidden from running tests,
measurements, benchmarks, builds or model code. They performed source work
only. All execution belongs to the root/main thread.

Implemented source slices:

- Mamba2/3: IDENTICAL zero-state prefill backward API returning input and nine
  parameter gradients; retained device buffers, explicit unsupported-mode and
  stale-binary refusals. Continuation-cache VJPs and FAST backward remain open.
- UMAP: learning_rate, repulsion_strength and negative_sample_rate through fit
  and transform, validated at both boundaries. Existing defaults retain their
  ABI and arithmetic. General cuML feature parity remains open.
- Trees: authoritative tree leaf counts/values, plus label/weight/empty-input
  safety and native Forbidden-NaN rejection. Ranking, multi-target losses,
  generalized Ordered boosting and categorical combinations remain open.

Main-thread host-only checks passed:

- Tree input safety, metadata, OVA surface, serialization: 60 tests, 13 subtests.
- Mamba2/3 ABI, ownership, input/mode refusals and stale binding: 3 tests.
- NVIDIA guard refusal, RSS/memory-pressure stops and signal cleanup: 4 tests.
- Python/shell syntax and git diff whitespace checks passed.

These checks used no local GPU workload or local Mojo compilation. They are
not numerical certificates. The two existing Pixi Python environments lacked
pytest; a temporary isolated binary-wheel environment supplied the tree checks.

Remote execution uses one RunPod NVIDIA RTX 4090, a 60-minute termination
lease, 3000-second shared work deadline, one active process group, two CPU
cores, two compiler/BLAS threads, a 12 GiB group-RSS cap, and host/GPU memory
pressure stops. Both independent watchdogs are armed before model work.
Polling cannot guarantee protection from an instantaneous allocation spike;
small fixtures and serial execution further bound the workload.

Frozen source commit: `202c44caacd2ee56164da148662393d77a1840d3` in the isolated
local validation snapshot. `source-snapshot.json` records every source file
hash and the original repository base commit. User changes were preserved;
no commit was made to the user's active branch.

Comparisons requested by this campaign: exactly one external per workload,
CatBoost CUDA for numeric symmetric RMSE fit plus GPU prediction, and cuML
CUDA for UMAP fit_transform. FAST, IDENTICAL and external arms rotate through
seven rounds on the same GPU with identical input bytes, matched workload
settings, host transfers included, and independent held-out error/quality
admission. UMAP uses 1024 rows to reach its public FAST GPU optimizer.

IDENTICAL repetition is checked by raw bytes. External libraries use a stated
accuracy/quality gate; neither external bitwise parity nor universal
cross-vendor identity follows from this campaign. Full feature completion,
whole-wheel qualification and a complete all-feature NVIDIA matrix remain open.
