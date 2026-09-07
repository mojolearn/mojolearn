# NVIDIA UMAP completion — 2026-09-06

**PASS for this bounded workload.** All 15 root-run jobs passed on a RunPod
RTX 4090 at frozen source `6146b121608d4cf73706d6540bb884e134df409c`.
The fetched-evidence admission passed. Pod `ludm26ufgrjikf` was deleted with
HTTP 204 and its absence verified with HTTP 404, with 42 minutes left on its
60-minute lease. See [controller and admission](run/controller.log),
[job statuses](run/remote/umap-finish/results.tsv), and
[complete frozen source](run/source.tar.gz).

Root ran jobs serially with two CPU cores/threads, memory monitoring and
deadlines. No Apple GPU/model execution occurred. Subagents authored source
and checks only. The earlier cuML import failure remains in the previous
campaign record; this fresh run isolates CuPy/cuML CUDA libraries from the
older Torch runtime and passes its explicit library-readiness gate.

## Matched NVIDIA comparison

The bounded 1,024-row, 50-epoch UMAP fit workload uses exactly one external
NVIDIA implementation, cuML. FAST, IDENTICAL and external arms passed the
independent quality policy. Seven rotated timed rounds follow warmup;
host transfers are included in the recorded boundary.

| Arm | Median milliseconds | IQR milliseconds |
|---|---:|---:|
| MojoLearn FAST | 48.012 | 0.462 |
| MojoLearn IDENTICAL | 196.126 | 0.858 |
| cuML CUDA | 27.048 | 0.151 |

cuML is faster in this case. These numbers do not establish throughput on
other datasets or sizes. See [raw results and settings](run/remote/umap-finish/umap-three-arm/results.json).

## Pinned real-data quality

The sklearn digits dataset uses a fixed stratified split of 1,024 fit rows
and 256 held-out rows, each with 64 features. The dataset hash was captured
before quality execution. Neighborhood trustworthiness and retention use
k=10. umap-learn is a CPU **quality-only** reference; no CPU timing ratio is
claimed. Both native modes ran on NVIDIA.

| Arm | Fit trustworthiness | Held-out transform trustworthiness |
|---|---:|---:|
| umap-learn | 0.984219 | 0.980918 |
| MojoLearn FAST | 0.984224 | 0.982841 |
| MojoLearn IDENTICAL | 0.984339 | 0.980353 |

All declared trustworthiness, retention, reference-gap and scrambled-row
negative-control gates passed. Independently repeated IDENTICAL fit and
transform output bytes matched; fitted model state and inputs stayed
unchanged through transform. See [quality results and provenance](run/remote/umap-finish/digits-quality/results.json)
and its retained input/embedding archives.

This closes the planned real-dataset UMAP experiment for these pinned bytes
and settings. It does not establish arbitrary dataset coverage, equality to
external embeddings, AMD/Metal equality, or full CatBoost/UMAP feature parity.
