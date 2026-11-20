# Spectral final source boundary for the combined native compile

This producer adds CPU 8 and GPU 6: assignment CPU 2/GPU 2 and RBF CPU 6/GPU 4.
All are authored and unrun. It also pins the existing pure Spectral lease
callback counters through assertion unwinding, without changing their six
cases or assertions. No further numerical scope is included in this batch.

Import these additional suites:

- `checks.native_tests.spectral_assignment_cpu_suite.run_checks`: 2.
- `checks.native_tests.spectral_assignment_gpu_suite.run_checks`: 2.
- `checks.native_tests.spectral_rbf_cpu_suite.run_checks`: 6.
- `checks.native_tests.spectral_rbf_gpu_suite.run_checks`: 4.

Both columns use `MOJOLEARN_NUMERIC_IDENTICAL=1`; CPU also uses
`MOJOLEARN_COLUMN_CPU=1`. Run from repository root for saved inputs. Dependencies
are pilot assignment `4084fbbe82`, RBF plan `eb7b5a81e2`, and the previously
handed-off Spectral API/backend/preparation and native helper producers.
Do not pick local API adoptions `8ca626a350` or `05bb879abe` separately.
The pilot's additive COO owner `5946fef3b7` does not change the request ABI or
the existing dense fixture expectations; it is collected by the integrator.

All authored Spectral coverage now totals CPU 99/GPU 31, including the existing
dense 14/8, contract 58/0, fit/embedding 7/9, prediction 6/6, and ownership 6/2.
GP theta `6f4ead351c` remains a separate queued 2/2 producer and is not included
in those Spectral totals. The root's prior 118/41 aggregate would become 126/47
after adding this producer, assuming its other suites are unchanged.

Assignment keeps the original 180 by 3 child-controller input and settings.
An independent input-graph certificate establishes the three-component label
partition without fixing label identities or eigenvector signs. The original
Python subprocess/second-invocation seam stays in place.

RBF checks are explicitly supplementary: audited Spectral Python files had
public acceptance/refusal seams but no numerical RBF formula gate to delete.
They use the original 150 by 2 input and existing default gamma values, with
an independent NumPy binary64 formula. The new absolute/relative bounds are
2e-6/2e-5 for Float32 rounding; no original expectation or tolerance changes.
Resource tests check allocation-order cleanup, partial failures and quiet
unwinding using live native records, without an actual mock device allocator.

CPU independent cases and cell assertions are parallel. GPU complete graphs
run in order on the physical device while existing kernels remain parallel.
The known shared-context concurrent-submission failure remains separate.
No product/kernel, algorithm, default, benchmark or timing changes. Nothing
removed: all original Python coverage remains pending native acceptance, and
necessary process/API/buffer/lifetime seams stay Python. Full source-only and
independent reference-preparation evidence is under `E/pca-native-tests/`.
