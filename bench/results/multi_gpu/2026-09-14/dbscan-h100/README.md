# DBSCAN neighborhood rows on two H100s

All compilation and execution ran on RunPod `4ra98lfm0pqum0`; no local build
or test ran. This reused the neighbors pod and its verified R2 enwik8 corpus.

- 18 public fits passed exact labels, iteration counts, feature counts, and
  all three identity-stage hashes (core mask, merged labels, final labels).
  Fixtures: 41/1025 rows, 3 features, duplicate rows, separated blobs/noise,
  1 MiB batch budget, RBC L2 and brute L2/L1, unweighted/uniform/signed weights.
  Invalid weight lengths left the owner's fitted labels unchanged.
- Nine existing native gate groups passed with one and two devices. They cover
  batching/tiny budgets, both RBC loops, L1 label changes, weighted-degree
  oracle and pinned-fold checks, uniform-weight agreement and duplicate versus
  weight-two semantics with a weight-1.5 sabotage.
- Twelve direct cases passed every adjacency byte, degree, CSR offset and
  neighbor index, plus returned edge counts and maximum degrees. Fixtures:
  1/3/17 query rows at a nonzero global start, 37 references, 3/129 features,
  duplicate rows, sparse/dense radii, both dense metrics and RBC count/fill/
  bounded one-pass operations. One-row calls exercise the one-active-device arm.

Root weighted-degree folds, core-point decisions, propagation and label merges
are unchanged. Workers own complete reference/index replicas and independent
neighborhood rows; integer offsets alone join CSR output. This is not pooled
reference/graph memory, a speed claim, or new cross-vendor qualification.

The pod's base source was `f8b5f68a23073fe5cc047cfe06052f7c51b5b570`.
Runtime overlay: `ba8ec24d6`; native/public gates: `906797884` and `033e8c9a8`.
Exact overlays, source/binary hashes, build/job logs, exit codes, stage traces
and the final JSON report are retained in `dbscan-out/`. All final jobs exited
zero; no runtime correction was needed after the initial build.
