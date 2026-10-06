# Native sparse preprocessing source repair

Implementation ID: `REPAIR.PREP.NATIVE_SPARSE.20261006`.
Branch: `repair/prep-native-sparse-20261006`.
Base: `e4cd912e842bd6168320f634e4dc0d79a8392a29`.

**Source only. NOT COMPILED, NOT EXECUTED, NOT MEASURED; quality and identity
are unverified. No production deployment, promotion or board admission.**
The existing campaign coordinator owns compilation and any later execution.
This branch does not alter active freezes, retained failures or accepted artifacts.

The problem is runtime SciPy conversion in `_expansion_prep._csr_input` and
both dense fallback callers, plus count-block CSR preparation in the benchmark
runner's whole-operation boundary. Installing SciPy would not repair those paths.
The native implementation replaces conversion; it does not complete the owner's
larger removal of Python from product execution.

## Native source and callers

`x_prep/sparse_input.mojo` defines typed raw-buffer descriptors and shared
validation, expansion, dtype conversion and duplicate-fold units. The callable
native entry points are `sparse_to_csr_device` / `sparse_dense_nnz_device` in
`x_prep/sparse_input_device.mojo` and their CPU counterparts in
`x_prep/host/sparse_input.mojo`. These three modules do not import Python.
The GPU entry uploads byte spans, performs computation on the device and returns
CSR buffers; it never constructs a dense matrix on the host.

CSR and dense inputs have direct paths. Dense conversion counts first, allowing
allocation of the actual CSR size; it has no dense-sized coordinate/sort scratch.
The transitional caller uploads the dense source twice, once for capacity and
once for conversion. Both calls remain preparation work within the declared
whole-operation boundary. This is not a performance claim.

CSC/COO/BSR/DIA conversion uses source-order IDs, stable integer radix sorting
on the device, and the same ordering in the CPU implementation. Temporary storage
is proportional to stored entries plus rows. No benchmark-specific thresholds,
host sorting on the GPU route or toolchain modifications were introduced.

Both `x_prep` bindings register the converter unconditionally. The existing
CSR-to-dense program stage (164) is available in all modes/columns. The historical
`MOJOLEARN_IDN_NB_CSR_DENSE_OFF` and `MOJOLEARN_IDN_ALL_OFF` flags no longer restore
a Python densifier. Other experiment controls, including sparse NB optimization
and AFCL-P04, are unchanged. This mandatory runtime repair changes the meaning
of that old fallback flag and therefore requires a new source freeze.

Production caller changes cover discrete NB fit, partial_fit and scoring,
including weighted/binarized dense-program fallbacks. Scoring normalizes with the
fitted mode. The benchmark adapter changes only our sparse-input preparation;
the opponent's SciPy conversion, roster, parameters, datasets, preprocessing and
timing definitions are retained. Existing benchmark aliases and unresolved
full-workload coverage are recorded in the companion JSON, not silently replaced.

## Numerical and format contracts

- CSR and BSR retain stored ordering and duplicate entries. CSC visits columns
  in source order within each resulting row.
- COO/DIA use a stable `(row, column, source-slot)` order. Duplicate values fold
  in the input dtype before narrowing to float32; half input promotes to float32.
  Boolean duplicate addition is logical OR; integer addition retains dtype width.
- Dense input drops exact source zeros before conversion, including negative zero.
  DIA drops padding and explicit zeros. CSR keeps explicit zeros.
- Raw-buffer conversion handles supported real/integer/bool types, endian layout
  and strides without the shared Python dtype-conversion fallback. Compressed
  pointers and indices are validated before use. All-zero storage is supported.
- LIL/DOK are retained through compiled object marshalling. That bridge still
  depends on CPython objects and methods; it is not a compliant final product API.
- A custom object exposing only a Python `tocsr()` method and no recognized storage
  cannot be converted without executing that Python method. It fails explicitly.
  Native interoperability for such providers is an outstanding compatibility gap.

The ordering contract is deliberate, but equality with a particular SciPy
duplicate-sort order, NaN payload handling, subnormals or any old binary has NOT
been verified. Dense fallback uses the existing float32/FTZ stage. Do not relabel
old baseline/rollback evidence as evidence for this implementation. SciPy's
[COO conversion source](https://raw.githubusercontent.com/scipy/scipy/v1.15.3/scipy/sparse/_coo.py)
and [DOK representation](https://raw.githubusercontent.com/scipy/scipy/v1.15.3/scipy/sparse/_dok.py)
were read as reference material only; no SciPy runtime was executed.

## No-product-Python migration remains incomplete

`python/mojolearn/_sparse_input.py`, the existing estimator wrappers, `Array`,
buffer acquisition/allocation, binding selection and label/output paths remain
Python execution. `bindings/prep_sparse_common.mojo` also uses CPython object
protocols for LIL/DOK. Moving a loop into a Mojo file while calling Python object
methods does not eliminate that dependency. These paths must remain visible in
the coordinator's product-execution inventory. This patch is not permission to
resume a campaign prohibited by the stricter owner rule.

A supported compiled public interface is feasible in principle: the repository
already uses `PythonModuleBuilder.add_type` for native tokenizer and byte-LM
handles. Official [Mojo binding documentation](https://docs.modular.com/mojo/manual/python/mojo-from-python/)
documents prebuilt modules, native types and methods. Its exact keyword-only
binding syntax is unsupported; supported keyword dictionaries are documented.
Preserving constructor validation, methods, fitted attributes, buffer ownership,
serialization and estimator protocols requires an explicit native implementation
and contract review. The raw native sparse APIs here are reusable by that work.
No full estimator-class replacement or verified compatibility claim is included.

The installed environment ships compiled standard-library packages rather than
readable `std/python/bindings.mojo` sources at the inspected location. API claims
above come from retained repository examples and official docs, not an invented
CPython C API. Do not patch toolchain internals or add a Python fallback when a
specific required interface capability is unsupported; record the Modular ask.

## Validation and next freeze

Only source parsing, source reach/contract inspection and Git whitespace checks
were performed. See the retained `source-review.json` and its complete log in
`mojolearn-integration-evidence-20261006/peer-prep-native-sparse/`.
These checks do not validate Mojo syntax, compilation, numerical behavior, memory
safety under execution or the complete production call chain.

After coordinator approval of the source/API transition, compile the changed
`x_prep` device and host bindings for supported IDENTICAL/FAST configurations,
including the sparse optimization off paths, the old densifier-off/global-off
defines, AFCL-P04 and affected combined configurations. DETERMINISTIC x_prep is
not shipped by the repository wrapper. Reuse unrelated accepted bindings.
Inspect wrappers first: compile-only, no module import/smoke execution.

Later authorized execution must cover all formats, empty storage, duplicate and
noncanonical rows, dtype/stride/endian cases, invalid pointers/indices, weighted
and Bernoulli fallbacks, partial_fit, mode retention, output/model state and the
saved full workloads. No such tests or measurements ran in this assignment.
The original matrix still marks recipe/cap coverage unresolved; use the
coordinator's admitted full recipes without inventing substitute workloads.
