# SVC/SVR and Gaussian-process multi-GPU kernels: two H100s

All builds and gates ran on RunPod pod `ok7m278wjgyo04`, using R2 enwik8
verified against the dataset-store manifest. No local build or test ran.

| Gate | Final result |
| --- | --- |
| SVM native kernel cells | 18 cases: linear/RBF, 1/3/17 output rows, 7/129/257 features, subnormals and signed zero |
| SVC/SVR public fits | 8 cases: linear/RBF, 7/137 features, duplicate rows, five iterations; complete fitted arrays and predictions |
| GP native covariance cells | 16 cases: RBF/three Matern forms, 5/17 rows, ARD, WhiteKernel training/cross-covariance semantics |
| GP public fits | 12 cases: 5/33 rows, RBF, WhiteKernel, constant/product/sum expressions, all three Matern forms; complete factors/duals/scalars and mean/std/clamp diagnostics |

The public gates also verify that invalid targets do not replace the owner's
fitted state. SVC is binary, SVR uses the existing epsilon-regression surface,
and GP uses fixed hyperparameters and the existing IDENTICAL ridge 2^-20.
Every final native/public comparison passed bitwise. See `svm-gp-evidence/` for
logs, JSON model hashes, hardware, binaries, job commands and return codes.

SVM distributes output rows of the original kernel operation, retaining its
per-cell FP32-v1 contraction and RBF expansion. Original working sets, updates,
stopping rules and support-vector prediction folds remain on the root. One-row
operations have one active device. Builds with global GEMM phase counters
refuse concurrent rows because those counters are not thread-safe.

GP distributes covariance rows without rewriting the postfix expression or
feature fold. WhiteKernel compares its column index to the original global row
index for self covariance; cross covariance still gets zero noise even when
its input coordinates equal training coordinates. Root Cholesky, solves,
likelihood and prediction variance are unchanged. The prediction wrapper
publishes the original clamp diagnostics after a successful return.

Both paths retain complete root state and assembled matrices. Per-call staging
and allocation may cost more than the distributed computation saves. These
small fixtures establish neither throughput scaling nor beyond-single-GPU
memory capacity, and they are not new cross-vendor qualification.

Source was shipped at `effbe1145`, with the SVM phase-counter guard from
`c0570d3f0` applied before building. GP was added from `9406d237a`; corrected
fixture parameters are in `b18fa071b`. The exact changed source from the pod is
retained with hashes. Subsequent upstream integration changed unrelated tools
and documentation. Builds were incremental, with separate binary hashes.

Development refusals are retained: SVC's first fixture selected unsupported
`gamma='scale'`, then a zero prediction-buffer size; GP's first fixture selected
an unpinned ridge of 0.2. The fixtures were corrected to `gamma='auto'`, a 1 MiB
prediction buffer, and ridge 2^-20. No numerical equality assertion was changed.

[Remaining estimator and pooling work](../../../../../docs/multi_gpu/COVERAGE.md).
