# Next AMD small-MLP run — source preparation only

**Execution update, September 7:** root completed the RunPod MI300X campaign.
All 18 jobs and admission passed; all 16 raw steps match NVIDIA.
[Saved results and verified deletion](../bench/results/resume/2026-09-07-root-mlp-amd/README.md).
The recipe below is retained preparation history, not a request to rerun it.

The other neural network is the public fixed FP32 **8→16→3 MLP**, distinct
from the two-block byte language model. NVIDIA completed all 18 integration
jobs, independent reference/edge checks, 16 learning steps and real-file
same-device 8+8 continuation. Loss fell 1.1047444 → 0.0450696. At preparation
time AMD equality was unmeasured; the completed execution is linked above.

## Exact retained reference

- Numerical/source commit: `8f6ed4112a3c140326011f908e8caf14c0dee4af`.
- Existing isolated repository: `/private/tmp/mojolearn-nvidia-training-20260906`;
  its HEAD still names that commit. Do not edit this retained checkout.
- [Frozen source archive](../bench/results/resume/2026-09-06-root-training-nvidia/run4/source.tar.gz).
- [NVIDIA record](../bench/results/resume/2026-09-06-root-training-nvidia/README.md),
  [18 job statuses](../bench/results/resume/2026-09-06-root-training-nvidia/run4/remote/training-validation/results.tsv),
  [continuous capture metadata](../bench/results/resume/2026-09-06-root-training-nvidia/run4/remote/training-validation/mlp-continuous/metadata.json).

Use a fresh isolated copy of that repository and exact source commit; do not
substitute current main. `small_mlp_training_capture.py` includes all `.mojo`
files under `training`, `gemm`, `core`, specified binding/Python sources and
itself in its equality inventory. Even unrelated new files under those
directories can invalidate equality. Preserve the original NumPy 1.26.4 and
package version witnesses, pinned Pixi files, input bytes and schedule.

## Transport blockers to close before any rental

The **current** `gemm_remote_leg.sh` profile 4 accepts AMD, invokes the HIP
serial guard, validates HIP Torch oracle metadata, and applies the successful
1 GiB runtime-pool override. The frozen source contains the training runner,
AMD guard and vendor-generic admission. Reuse these numerical sources with a
separately retained transport overlay; there is no need to implement MLP arithmetic.

However, current profile 4 does **not forward an explicit interpreter or
`MOJOLEARN_GPU_ARCHS`** into its remote runner. Setting them on the local shell
alone is insufficient. Before launching, root must add these two environment
assignments to the profile-4 remote invocation in an isolated transport copy:

```sh
MOJOLEARN_PYTHON="@BYTELMPYTHON@" MOJOLEARN_GPU_ARCHS="@GPUARCHS@"
```

The placeholders already exist in the current launcher and are substituted
from `MOJOLEARN_BYTE_LM_PYTHON` and `MOJOLEARN_GPU_ARCHS`. Keep them on the same
continued command as `MOJOLEARN_TRAIN_EXPECT_VENDOR` and the runner invocation.
Record the transport hash independently; root must inspect the generated body
before rental. Profile 4 also lacks profile 5's enforced ROCm-PyTorch image
admission, so select the explicit known image below. Plain ROCm development
images do not supply the required independent Torch reference environment.

The frozen training runner has a simple EXIT-status trap, unlike the newer
compact helper's active-child cleanup. Preserve the external payload/lease
watchdogs, require serial guard cleanup and fetch all statuses; an interrupted
runner must never trigger overlapping work. Tightening this orchestration is
permitted only as a disclosed nonnumerical overlay with root review.

## Root-only launch after transport review

In a fresh copy of the isolated repository, use the reviewed current transport
overlay while keeping `--source-ref` at the exact original commit. Root must
perform source/file-only checks and dry-run admission first. This author has
executed none of these commands:

```sh
MOJOLEARN_NVIDIA_CAMPAIGN=4 \
MOJOLEARN_GPU_ARCHS=gfx942 \
MOJOLEARN_BYTE_LM_PYTHON=/opt/conda/envs/py_3.10/bin/python \
MOJOLEARN_GEMM_LEG_OUT=/absolute/NEW_MLP_AMD_OUTPUT \
tools/gemm_remote_leg.sh amd --payload mamba \
  --source-ref 8f6ed4112a3c140326011f908e8caf14c0dee4af \
  --image rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0 \
  --gpu 'AMD Instinct MI300X OAM' --rent --minutes 60 --work-timeout 3000
```

That image tag and interpreter started successfully for the retained AMD
byte-LM runs; the digest-form first attempt failed at SSH startup. Preserve
both historical outcomes in the [AMD record](../bench/results/resume/2026-09-06-root-byte-lm-amd/README.md).
Do not claim image startup is guaranteed. A current remote metadata check must
confirm MI300X/gfx942, actual HIP execution and HIP Torch, not CUDA Torch.

The generated remote body must contain all three allocator settings:

```sh
MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE=1073741824
MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY=true
MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=100
```

Retain two-core affinity/two-thread compilation and runtime, one serial root
GPU job, guard RSS/VRAM/reserve thresholds, remaining-deadline clamping,
60-minute lease watchdog and verified pod deletion. This pool cap worked for
the byte LM; it has not yet been exercised for this AMD MLP workload. Profile 4
also runs the bounded one-block Transformer gradient prerequisite; it is not
a minimal MLP-only campaign. Do not silently skip its required jobs/admission.

## Root-only comparison after fetch

First require all 18 AMD jobs and `training_validation_admit.py` to pass. Then
use the frozen MLP comparator with a two-thread NumPy 1.26.4 interpreter:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /absolute/NUMPY1264_PYTHON tools/small_mlp_training_capture.py compare \
  --left /absolute/NVIDIA_RUN4/remote/training-validation/mlp-continuous \
  --right /absolute/AMD_RUN/remote/training-validation/mlp-continuous \
  --output /absolute/NEW_MLP_CUDA_HIP_COMPARISON.json
```

This comparator reads retained archives, compares all 16 steps' full raw arrays
(including optimizer state and loss), checks source/input/configuration
witnesses and effective moment-reset controls, and reports vendor witnesses.
Require `identity=PASS`, `learning.status=PASS`, and left CUDA/right HIP. It
supports continuous versus head/resume chains too, with real checkpoint SHA
linkage; this launch's head/resume remains **same-device AMD**. Matching CUDA
and HIP continuous runs does not itself demonstrate cross-vendor checkpoint
resume, Metal equality, arbitrary network coverage or training performance.

All tests, builds, models, measurements and rentals remain root-only. No local
Apple model execution is part of this plan.
