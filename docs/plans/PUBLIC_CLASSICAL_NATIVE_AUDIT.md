# Internal public classical runtime audit

This is completion evidence and an implementation plan, not public API documentation or a native-runtime qualification.

The CPU matrix on 2026-10-20 exercised actual public constructors, fits, and consumed predictions/transforms for KMeans, LinearRegression, Ridge, LogisticRegression, PCA, ExtraTreesClassifier, and ExtraTreesRegressor. All five CPU-supported fits and seven consumed-output operations completed, with expected shapes and finite outputs. Every one still ran package Python. The seven constructors did too. The two ExtraTrees fits correctly refused GPU-only training; their three output operations were dependency-blocked, not silently skipped or claimed covered.

The run exited 1: 24 planned operations, 19 runtime violations, two expected route refusals, three dependent operations blocked, and zero native passes. Numerical success does not erase the runtime violations. Tiny deterministic inputs are smoke coverage only; this is neither timing nor an exhaustive option matrix.

| Estimator | Fit Python calls | Consumed public output Python calls |
|---|---:|---:|
| KMeans | 288 | predict 171; transform 172 |
| LinearRegression | 2,007 | predict 208 |
| Ridge | 504 | predict 208 |
| LogisticRegression | 548 | predict 276; predict_proba 366 |
| PCA | 296 | transform 215 |

Counts include transitive Python frames, including standard-library functions reached by the package. The audit also records forbidden NumPy C calls; in this matrix these were `ndarray.__dlpack_device__` metadata calls, not evidence of NumPy numerical kernels. The Python orchestration failures are independently established by the traced package frames and wrapper bytecode.

The common paths are `_mode.py` public guards/output handling, `_sk_validate.py` validation, `_buffer.py` buffer setup/release, `_array.py` shape/allocation work, and `_backend.py` runtime module selection/loading. LinearRegression.fit additionally executes `_expansion_decomp._M` orchestration (254 scalar metadata accesses and 102 address accesses in this call). A shared numerical kernel passing a standalone test therefore does not qualify these public operations.

The next bounded repair is a native public owner for LinearRegression/Ridge covering construction, admission, weighted/unweighted fit, prediction, and the other existing public operations. Preserve the existing TSQR, normal-equation, and weighted Ridge routes and numeric-mode semantics; connect verified compiled targets once at package setup; retain CPU/GPU execution. Reuse strict native buffer admission patterns, not Python helpers from a native callback. LinearRegression's CPU decomposition walk needs its own native orchestration rather than a generic wrapper rewrite. KMeans, LogisticRegression, PCA, and ExtraTrees remain separate public-runtime debt.

A targeted Metal ExtraTrees follow-up completed all seven operations with expected output shapes and finite values, and all seven failed the runtime contract. Classifier fit/predict/predict_proba reached 917/553/363 Python calls; regressor fit/predict reached 776/429. Both constructors also executed Python. There were no operation errors or skipped operations. The GPU slot was released when this probe returned.

Reproduction uses `tools/audit_public_classical_native.py` with the retained stage50 validation overlay. Complete evidence is under `~/mojolearn-evidence/finish-generation-public-classical-20261020/`: `cpu.json`, `cpu.log`, `import-cpu.log`, `metal-extratrees.json`, and `metal-extratrees.log`. The JSON retains all traced frames, wrapper violations, output digests, source hashes, and loaded native module paths/hashes (47 CPU; 135 Metal). It embeds the overlay's full reuse provenance. The exact Metal controller source was retained before execution as `audit-public-classical-metal-controller.py`.

The package source is `f9109326489a5d6681e744b32e1c7d01330f31e5`; reused classical numerical binaries originate at `a8255828597a3bd19ac7bb5f657a38d09b8c732f` (release 0.8.55). New tokenizer/sequence families are separately recorded in the overlay. This is source-validation evidence, not an installed-wheel result, and does not qualify current changed cast/label closures against the retained classical binaries. No native compilation occurred in this lane.
