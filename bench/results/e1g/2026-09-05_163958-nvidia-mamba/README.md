# NVIDIA four-arm public k-NN layout qualification

RTX 4090, driver 580.159.04, pinned source `9fe07a33`. All eight builds,
four correctness runs and 108 rotating timing invocations succeeded.
Strict post-collection admission passes: complete outputs match across all
four arms and all rounds. Each correctness arm also matches the 143,628
selected pairs retained from Apple; see apple-correctness-comparison.json.

100,000 indexed points, 32 features, K=10, dyadic-v1 fixture. Median native
request milliseconds (upload, search, download and synchronization):

| Queries | Baseline | Selector | Transpose | Both |
|---:|---:|---:|---:|---:|
| 32 | 25.754 | 4.781 | 26.182 | 5.009 |
| 128 | 82.135 | 6.575 | 84.076 | 6.237 |
| 1000 | 759.333 | 17.241 | 752.376 | 9.273 |

Nine interleaved rounds per arm and shape; two excluded warmups in each
process. knn-layout-summary.json retains IQRs, paired ratios, individual
samples, output hashes and source hashes. The transpose-only path does not
show a clear benefit; the combined path improves the largest tested request
over selector-only. This is neither an external-library comparison nor a
claim about other datasets, Python API overhead or all GPUs.

The original controller returned 1 because it incorrectly required the
fetched console to be nonempty. Every child writes its own log, so an empty
console is normal on success. The initial parser also rejected a CUDA
allocator warning before the activation header. Both reporting defects were
fixed and covered by negative controls. Raw logs were not edited; admission
still requires complete ordered protocol records and exact output equality.
See classification.json and the retained controller/dry-run logs.

Source hashes matched at both ends. Pod g0iy6aygcufcz7 was deleted after
collection: DELETE 204 followed by verified GET 404. No rental was repeated
to repair reporting. AMD qualification remains pending.
