# NVIDIA RTX 4090 campaign, 2026-09-05

Base source: 1d1f8fdd37022dad1c9e5752311296580e7762dd. Supplemental
source files are retained in their run directories and must not be relabeled
as the base commit. The overall campaign was incomplete: retain every failure.
The machine was deleted and its absence verified with HTTP 404.

Native Mamba backward certificates passed five cases / 54 tensors. Source
IDENTICAL Mamba Python API passed 102 checks. Both native UMAP fit profiles,
source UMAP API and two synthetic quality cases passed. Transformer native
forward/backward passed, but Python API timed out twice. None of these are
complete current Linux installed-wheel qualification.

Host-array public comparisons use current APIs, seven rotating rounds, FP32
CUDA references with TF32 disabled, and retained numerical admission checks.
They are not the September 3 kernel-only identity-cost fixtures.

| Fixture | FAST median ms | IDENTICAL median ms | CUDA reference median ms |
| --- | ---: | ---: | ---: |
| kNN 100k index, 32 queries, d32, k10 | 3.981918 | 8.165275 | 1.107609 |
| kNN 100k index, 128 queries, d32, k10 | 4.516717 | 23.475313 | 1.386469 |
| kNN 100k index, 1000 queries, d32, k10 | 6.850315 | 215.468543 | 5.638816 |
| GEMV 2048 squared | 1.680231 | 1.718041 | 1.067451 |
| NT 16384 x 64 x 64 | 2.040811 | 1.523541 | 0.884171 |
| Gram 65536 x 32 | 3.785673 | 3.776442 | 1.096630 |

See per-fixture JSON for raw samples, dispersion, precision, scope and gates.
kNN uses PyTorch CUDA cdist plus sorted topk, not cuML. Two cuML dependency
installation attempts timed out, so cuML kNN/UMAP comparison remains open.
The current production IDENTICAL kNN slowdown is not confined to tiny query
batches. The much faster specialized selector is experimental and is measured
separately in final-supplement/smallk-specialized.log.

Mamba-1 upstream whole-block forward/backward comparison passed after setting
CUBLAS_WORKSPACE_CONFIG before Torch initialization (last-checks/mamba1).
Mamba-2 failed the upstream fused-convolution stride constraint. Mamba-3 timed
out after the workspace fix. The separate larger Mamba-1 forward timing fixture
passed numerical admission; transformer timing failed numerical admission,
so its timings do not establish a competitive, correct implementation.
Upstream agreement is tolerance-based, never a cross-library bitwise claim.

All remote artifacts are retained here, including failing runs and raw arrays.
Repeated NPZ files have only 16 unique contents (about 8.8 MB); Git deduplicates
them. The raw directory's logical size is about 107 MB.
