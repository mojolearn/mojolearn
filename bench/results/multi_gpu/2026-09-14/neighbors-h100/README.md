# Whole-query neighbors and density partitions on two H100s

Builds and model execution ran only on RunPod pod `4ra98lfm0pqum0`.
The training fixture was R2 enwik8, verified against the dataset manifest:
`2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`.
No local compilation, model execution or tests ran.

The gate compares array shape, dtype and every output byte against the original
single-GPU query. It uses 41 reference rows, 17 query rows (41 for radius self
queries), seven-row logical shards across two devices, duplicate references,
zero distances, far-away queries and 3/129 features. Coverage includes brute
KNN with Euclidean/Manhattan/Chebyshev/cosine metrics, Euclidean/Manhattan/
Chebyshev RBC KNN and radius queries, sorted and index-only ragged outputs,
self edges, uniform/distance weights, single/multiple targets and probability
matrices. KDE covers all six kernels with/without strictly positive weights.
Every case repeats on persistent workers. KNN also checks rejected worker
requests and restart; all cases check atomic diagnostics after invalid input.

All 64 final cases passed bitwise; the final job exited 0. See
`neighbors-out/report.json` and the retained job receipts.

Each worker receives the complete fitted reference data. Every query preserves
its original reference-row distance/selection/vote/density fold; host assembly
copies output bytes in input order. No reference reduction or neighbor merge
is introduced. The driver records per-shard diagnostics separately and leaves
the fitted owner's diagnostics untouched. Fit still stores reference data;
these are distributed GPU queries, not iterative training.

This is neither pooled reference-index memory nor cross-vendor qualification.
It establishes no speedup or beyond-single-GPU capacity, and broader shapes,
metrics and configurations still need qualification.

Base source was shipped at `f8b5f68a23073fe5cc047cfe06052f7c51b5b570`.
The cloud gate received the failure/restart checks from `2baf9e439`, followed
by the positive-weight fixture correction in `8c5c93a94`. Runtime source did
not change after the initial ship. The exact final source overlay, source and
binary hashes, build logs, job scripts, exit codes and reports are retained.
The initial KDE fixture had a zero weight, admitted by host fit validation but
refused by the existing native scorer. Its failure log is preserved. The
fixture now uses weights 1..41; no numerical equality assertion was weakened.
