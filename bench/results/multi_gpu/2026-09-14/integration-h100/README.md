# Main integration and IsolationForest: two H100s

All builds and checks ran on RunPod pod `nlfngsvejhgiq5` using the R2 enwik8
corpus verified against the dataset-store manifest. No local tests or builds.

The earlier multi-GPU branch was merged with main at
`5ad4f3493b5c51690e54341739182be5ab7f22d2`. Main's changed Gram default selects
the scalar accumulator arm. The merged extension passed 40 native fixtures
comparing every Gram partial and output, 20 OLS/Ridge/covariance PCA/SVD fit
fixtures, and 16 binary/multiclass logistic fit fixtures. See
`integration-evidence/merge-out/`.

IsolationForest was then added in `07de1c4d4`, with its ownership-transfer
compile correction in `4d8f0bd7d`. The final source overlay is retained under
`integration-evidence/source/`. Later upstream merges changed documentation
only. Both cloud jobs returned zero after the documented setup corrections.

The new tree driver passed four native cases comparing all eight model buffers
(threshold bits, node features/children, feature subsets, tree offsets, node
counts and depths), and eight public cases covering one/five trees, bootstrap,
feature subsets, automatic/numeric contamination, all three scoring methods,
and rejected-fit publication. See `integration-evidence/iforest-out/`.
Whole-tree scratch is partitioned, training data is replicated, and the complete
forest is assembled on the root. Score-time tree rebuilds can use the explicit
parallel entry; the original root scoring fold and threshold remain intact.
Diagnostic trace export is refused by the distributed path.

These are small two-H100 identity fixtures, not new cross-vendor, throughput,
or beyond-single-GPU memory qualification. The unchanged earlier paths retain
their separately scoped receipts. The full estimator/pooling request remains
incomplete; see [coverage](../../../../../docs/multi_gpu/COVERAGE.md).

Development failures are retained: the initial integration Python gate lacked
the fresh pod's base extension, and the first IsolationForest compile required
an explicit move of its owned shard configuration. Building the base extension
and correcting that ownership transfer resolved those failures. No arithmetic
gate was weakened. Source archive hashes, binary hashes, hardware, job commands
and return codes are included with the receipts.
