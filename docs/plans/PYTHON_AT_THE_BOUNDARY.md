# Python at the boundary: two functions per public call, Mojo values everywhere else

Design lane `b1-python-boundary-design`, 2026-11-09. Read-only on product code; this document is the only output.
Read against `origin/integ/pre-0867-w3` at e68654a757. Nothing here was compiled or run. Every Mojo type below is a
sketch to be settled by the pilot's compile, not a finished signature.

Andrew, 2026-11-09: "Why do we need to touch Python at all, why can't it be purely at the boundary, why can't we use
Mojo objects? We need to get rid of Python objects."

## 0. One screen

- **Today.** The compute kernels are clean: 0 of 1,959 kernel files touch Python. The binding layer is not a
  boundary, it is a layer: 825 of its 842 files handle Python objects (`~/mojolearn-evidence/py-audit-a-imports.md`
  section 1), with about 47,000 object-layer lines, about 813 calls from Mojo into Python code (141 in loops), 2,173
  loops over Python objects (607 data-sized) and 730 compiled entries that Mojo reaches through the CPython call
  protocol with Python lists as arguments.
- **Target.** A public call is three steps. UNPACK turns the Python call into one plain Mojo request value. RUN is
  pure Mojo and takes only Mojo values. PACK turns the Mojo result value into the Python object. Python object types
  appear in the unpack and pack files and nowhere else. Typed interfaces and executed-call tests establish the
  boundary; the retired source-count guard is not evidence that this target has been reached.
- **Not touched.** Hand-tuned kernels, fused paths and every kernel's arithmetic (AGENTS.md "A measured winner is
  never deleted on reasoning"). This design changes what calls the kernels, never the kernels. Bits do not move.
- **Order.** Pilot on ONE small family, compiled, tested and measured, before a second family starts. Then family by
  family, worst offenders first, each landing green.
- **Size.** About 840 binding files in 179 families. This is the largest program the repo has taken on. The pilot is
  small (section 6). The rollout is weeks (section 7).

## 1. What the code does today (evidence)

`bindings/classical_ridge.mojo` is the Ridge and LinearRegression door: 255 lines, 8 functions, 27 uses of
`PythonObject`, 18 `py_call`s. `classical_ridge_fit` (:84) shows the pattern every door has:

1. Parameters are read from a Python dict one key at a time, inside the logic: `fields["alpha"]`,
   `fields["fit_intercept"]`, `fields["solver"]` (:61-70, :94-95), and validated with Python comparisons
   (`alpha < PythonObject(0.0)`, :64).
2. X, y and the weights are admitted into Python arrays and stay Python objects: `x.shape[0]` is read back through
   the object layer (:90-91), and every use takes `classical_address(x)`, itself a Python object (:108-110).
3. Scratch and outputs are allocated as Python arrays (`classical_array(cols, 0, context)`, :97-98, :105, :113).
4. The compiled entry is reached through a Python dict of callables with Python lists built per call:
   `py_call_typed(entries["x_linear_fit"], PythonObject(8), classical_address(x), ..., Python.list(rows, cols, ...),
   Python.list(1, 1 if intercept else 0, ...), Python.list(PythonObject(alpha)), ...)` (:114-119). The receiving
   binding unpacks those lists again (`bindings/_mojolearn_x_linear.mojo:5`: `x_linear_fit(algo, x_addr, y_addr, dims,
   ip, fp, out_addr)`).
5. Route choice asks Python: `py_get(entries, "linear_gram_fit") is not None`, `entries["linear_gram_solve_default"]()`
   (:129-131).

So one fit crosses the Python object layer dozens of times for work that is two pointers, six integers and a float.
The same shape at family scale: `bindings/x_linear_native.mojo` is 2,718 lines with 376 `PythonObject` uses and 87
`py_call`s; `logistic_native.mojo` 830 lines, 106 and 29; `kmeans_native.mojo` 599 lines, 78 and 21.

Three facts that make the target reachable:

- The admission path already returns a Mojo struct. `AdmittedX` (`bindings/admitted_input.mojo:137`) holds address,
  rows, cols, ndim, borrowed and finite. Its remaining Python fields are `array` and `source`, which exist only to
  keep storage alive.
- `admit_typed` / `AdmittedTyped` (:1100, :1278) and `admit_view` (:1315) already cover non-float32 inputs.
- Process state is shared across extension modules by name (`~/mojolearn-evidence/design-shared-process-state.md`),
  and `SharedSlot[T, name, version]` (`core/shared_slot.mojo:122`) gives a versioned slot. A native function table
  can be published in one, so one extension module can call another's compiled entry with no Python in between.

## 2. The contract

### 2.1 UNPACK (one function per public operation, in a boundary file)

Input: the Python arguments exactly as the user passed them, plus the estimator's Python instance.
Output: one Mojo value. Sketch:

```mojo
struct Matrix(Copyable, Movable):      # what AdmittedX already is, minus its Python fields
    var address: Int                    # first word; 0 when the rows are on the device
    var rows: Int
    var cols: Int
    var dtype: DType                    # float32 for kernels; admit_typed covers the others
    var order_f: Bool
    var borrowed: Bool
    var finite: FiniteState             # TESTED, DEFERRED (the kernel entry returns the status word), ACCEPTS_NAN
    var device: DeviceRowsRef           # store id + row index buffer when a fold lives on the device, else empty
    var keep: OwnerToken                # opaque lifetime token; the ONLY thing that references Python storage

struct RidgeFitRequest(Movable):
    var x: Matrix
    var y: Matrix
    var sample_weight: Optional[Matrix]
    var alpha: Float64
    var fit_intercept: Bool
    var solver: RidgeSolver             # an enum, not a string
    var mode: NumericMode               # identical / fast
    var est: StaticString               # for messages
```

Rules:

- Every Python read happens here, once: buffer metadata through the one admission path, scalars, options. Strings
  become enums or a `StaticString` here. Labels become a native code table here (lane P2 builds it). A Python list
  becomes one typed buffer here, in one pass.
- Parameter validation that is a comparison on a number moves to RUN, on Mojo values. UNPACK only refuses what it
  cannot convert (wrong type, sparse where dense is required, a user callable).
- `OwnerToken` is how a borrowed caller array stays alive for the call. It is an opaque value: RUN can hold and drop
  it, and cannot look inside it. Its implementation (a strong reference to the Python object) lives in the boundary
  file.
- UNPACK holds the GIL. It returns, and the boundary releases the GIL for the whole of RUN.

### 2.2 RUN (pure Mojo)

`def ridge_fit(req: RidgeFitRequest, mut state: RidgeState) raises -> RidgeFitResult`.

- Takes and returns Mojo values only. It calls the kernels directly when they are in the same extension module, or
  through a native function table when they are in another (section 2.5). No dict of callables, no Python lists.
- Scratch and outputs are Mojo-owned buffers (the uninitialised allocator `admitted_storage` already exists for the
  host side; device buffers are already Mojo-owned).
- Errors are Mojo values: `raise MojolearnError(kind, message)` with `kind` an enum (VALUE, TYPE, NOT_FITTED,
  NONFINITE_INPUT, INDEX, MEMORY, NOT_IMPLEMENTED). Text comes from one place per kind; the non-finite wording stays
  `admitted_finite_wording`.

### 2.3 PACK (one function per public operation, in a boundary file)

Input: the Mojo result value, or the Mojo error.
Output: the Python return value, or the Python exception of the mapped type.

- An array result is handed over as one buffer with its shape: PACK wraps it in the package's array type without
  copying (section 4.3).
- A fit returns the estimator instance; PACK stores the new state handle on it (section 3) and nothing else.
- Warnings are values in the result (`warnings: List[WarningRecord]`); PACK issues them after RUN returned. Lane P1 is
  already moving `prep_encode_native.mojo:1967` to this.

### 2.4 What a boundary file is

`providers/python/**/*_pyapi.mojo` and the registration roots `bindings/_mojolearn*.mojo`. Nothing else. A boundary
file contains UNPACK, PACK, the `OwnerToken` implementation and the module's method table. Today's
`bindings/*_native.mojo` doors become RUN files and lose every Python import.

### 2.5 Calling compiled entries in another extension module

Today: `py_call_typed(entries["x_linear_fit"], ..., Python.list(...))`, 730 sites.
Target: each binding publishes a table of `abi("C")` function pointers in a `SharedSlot`
(`MojolearnEntries<Binding><Tier>V1`), filled at module init. A caller reads the slot once and calls the pointer with
plain arguments or a pointer to a plain struct. The existing `kit_call` capsule ABI and the 30 descriptor dispatch
tables S1 tier-keyed are this mechanism already; this makes it the only one. The tier (identical / fast) and the
vendor are part of the slot name, so the base-binding choice an install makes at import is a slot lookup, not a Python
attribute.

## 3. Fitted state as a native object

- `RidgeState` is a Mojo struct (coefficients, intercept, n_features, the resident device handles, the numeric mode).
  It lives in a native registry keyed by an integer handle, the pattern `ResidentKnnIndex`, the ops session table and
  the msel store already use.
- The Python estimator exposes constructor parameters for scikit-learn compatibility and retains one opaque
  native state owner. `get_params`, `set_params` and `clone` enter native operations; published Python attributes
  are boundary representations, not the internal execution plan. Preserve each method's existing state semantics.
- A fitted attribute (`coef_`, `intercept_`, `n_features_in_`, `labels_`, `components_`) is one native call:
  `state_attribute(handle, ATTR_COEF) -> buffer view`, wrapped by PACK with the public API's existing mutability.
  In particular, Ridge/OLS coefficient arrays remain writable and replaceable. UNPACK admits the current array,
  retains its owner and refreshes the typed descriptor; RUN never follows Python attributes. Existing protected
  weight APIs retain their protection. Lazy materialization is allowed only when it preserves these semantics.
- Pickling is one native reduce: `state_dump(handle) -> one bytes buffer` with a layout version, and
  `state_load(bytes) -> handle`. Audit B found Python running once per Array field today (`_array.py:278`, `:530`,
  `:892`).
- Across extension modules: the registry is one `SharedSlot` per state type with a layout version, so a meta-estimator
  in the trees binding can run a Ridge member through its handle and the function table with no Python in between.
  The three hazards the shared-state note lists (layout versioning, per-thread scopes, single creation) are already
  handled by S1 and S2; a new state type gets them by using `SharedSlot`.
- Lifetime: the handle is released by the Python object's finaliser, or by `close()`. A handle used after release is a
  NOT_FITTED error, never a dangling pointer: the registry checks a generation counter.

## 4. What cannot leave Python, and how it is confined

1. **The public Python API.** Python exposes the public classes and hands each operation to one native call.
   Parameter storage and validation, `get_params`, `set_params`, Mojolearn-owned cloning and representation work
   belong in Mojo, with Python objects confined to UNPACK/PACK. Only `__sklearn_tags__` and its tag-only helpers
   may execute Python metadata code (the 2026-10-24 exception). Native product operations must never call them.
2. **User-supplied Python objects.** A callable `init`, a custom scorer, a custom estimator inside a pipeline or a
   search. UNPACK refuses them before any hook runs and names the native alternative. One exception exists in code
   today, KMeans callable `init` (`kmeans_native.mojo:244-262`); lane P1 turns it into a refusal.
3. **The array type the user holds.** `mojolearn.Array` (`python/mojolearn/_array.py:304`) is a Python class with a
   `_store` attribute that Mojo code reads (`classical_wrap_array(array._store)` broke KNeighborsRegressor today when
   the store type changed). Target: `Array` is a thin Python class over a native storage handle. It implements the
   buffer protocol natively, so NumPy and UNPACK read it without Python code. Its methods are one native call each.
   No Mojo file names `_store`.
4. **Strings.** Estimator names, option strings and string labels arrive as Python `str`. UNPACK converts each once.
5. **Import-time wiring.** Connecting the modules at import stays Python, runs once, and is tagged as such (lane P5).

## 5. Typed boundaries and execution evidence

Owner direction, 2026-11-10: retire the Python source-count/graph compliance gates, including the kernel-path,
shell-loop, host-route, runtime-import and public-runtime counters. Do not
extend them, restore them under another name, or replace them with a source-count baseline gate. Excluding
binding/provider code or rejecting necessary UNPACK/PACK loops cannot establish that an operation avoids Python.

The architecture still confines Python objects and CPython interaction to UNPACK/PACK (section 2.4). RUN takes
typed Mojo requests and returns typed Mojo results, uses native state handles and native function tables, and
imports no Python support. Complete the actual call paths rather than declaring files compliant by name or tag.

Track the remaining migration work in an inventory with its scope stated. Source counts describe work remaining;
they do not prove runtime correctness, callback freedom or speed. Move kernel and native-state correctness tests
to Mojo, retain Python tests of the Python API boundary, and audit executed public calls for Python callbacks.
Measure the affected operations on NVIDIA and AMD before claiming that removing object-layer work improved speed.

## 6. The pilot

**Family: Ridge and LinearRegression** (`bindings/classical_ridge.mojo`, 255 lines, 8 functions; Python shell
`python/mojolearn/linear_model.py:435` and `:706`; kernel entries `x_linear_fit` algo 8 and the gram fit in
`bindings/_mojolearn_x_linear.mojo`).

Why this one:

- It is the smallest door that has the full shape: X, y, optional sample_weight, scalar parameters, a string option,
  two routes, fitted attributes, predict, score, pickling, and use as a member in CV and in meta-estimators.
- It has no labels and no strings in the data, so it does not wait on lane P2's label table.
- OLS was one of the worst regressions (45-70x), so its door path is already well understood and measured, and Ridge
  fit over a device fold already exists (`x_linear_fit_fold`, A8c), which exercises the `Matrix.device` case.
- Its kernels are not hand-tuned paths in dispute. No tuned kernel is touched.

A second proof, only after the first is green and measured: **LogisticRegression** (`logistic_native.mojo`, 830
lines), which adds the label table and a predict_proba result.

Files the pilot touches (to be confirmed by the lane, with line counts in its reply):

| file | change |
|---|---|
| `bindings/classical_ridge.mojo` | becomes RUN: `ridge_fit`, `linear_predict`, `linear_score` over Mojo values; loses every Python import |
| `providers/python/linear_ridge_pyapi.mojo` (new) | UNPACK and PACK for fit, predict, score, state attribute, dump, load |
| `core/boundary_types.mojo` (new) | `Matrix`, `FiniteState`, `OwnerToken`, `MojolearnError`, `WarningRecord` |
| `core/state_registry.mojo` (new) | handle registry on `SharedSlot`, generation check |
| `core/entry_table.mojo` (new) | the published function table (section 2.5), with the x_linear entries the pilot calls |
| `bindings/admitted_input.mojo` | a form of `admit_x` that returns `Matrix` (the fields exist; this drops the two Python fields behind `OwnerToken`) |
| `bindings/_mojolearn_x_linear.mojo` | publishes `x_linear_fit` / gram entries in the table beside today's Python-callable registration (both stay until the family's other doors move) |
| `python/mojolearn/linear_model.py` | Ridge / LinearRegression hold parameters and a state object; fitted attributes are properties making one native call |
| tests | the existing Ridge / LinearRegression tests, unchanged; one new test asserting the Python-operation count per call is constant in n |

Before / after, fit: today about 60 object-layer operations and 3 Python lists per fit on the path in section 1;
after, one UNPACK (3 buffer reads, 3 scalars), one RUN, one PACK.

What is deleted with the pilot: the Python-dict parameter reads, `classical_ridge_parameters`' Python comparisons,
the `entries[...]` calls and their `Python.list` packing for this door, the Python-array scratch allocations.

Gate before any second family starts:

1. Compile green on the Mac and the NVIDIA / AMD cross-compile.
2. Import, the Ridge and LinearRegression test files, and the light smoke pass on the Mac.
3. Focused correctness and lifetime tests of the compiled pilot. No identity runs or old-version digest matching
   during development: cross-vendor identity is recorded once after the final source freeze, before release.
4. Measured on NVIDIA and AMD against the base: ols and ridge fit and predict on the board rows, and the call
   overhead on a 16-row fit and a single-row predict (where the object-layer cost is the whole time). The pilot is
   kept only if nothing is slower.

## 7. Rollout after the pilot

Sizes are binding-file counts from the audit's per-family table (179 families, 842 files). S is under 5 files, M is
5 to 15, L is over 15.

| order | families | files | size | why this order |
|---|---|---:|---|---|
| 0 | Ridge / LinearRegression (pilot), then LogisticRegression | 3 | S | section 6 |
| 1 | classical (the rest), x_linear, pca, svc, kernel methods | ~50 | L | same shape as the pilot; the regressed rows |
| 2 | neighbors, cluster / kmeans, mixture, gp, metrics | ~60 | L | labels and state objects, after P2's label table |
| 3 | trees (71 calls into Python, 368 loops: the worst family), prep, resample | ~58 | L | most callbacks; meta-estimators need the entry table |
| 4 | msel, parallel, cross, arena | ~47 | L | orchestration; depends on members having state handles |
| 5 | autograd, sequence, sequential, optimizer, training | ~88 | L | sessions and programs are already handle-based |
| 6 | causal, samba, mamba, transformer, byte, lm, neural, xcnn, xseq, tokenizer | ~66 | L | after the executor work settles |
| 7 | host, archive, array, native_*, params, tsa, autoarima and the remaining ~120 small families | ~470 | L in total, S each | long tail; mechanical once the types exist |

Waves 1 to 6 touch disjoint files and can run as parallel lanes once `core/boundary_types.mojo`,
`core/state_registry.mojo` and `core/entry_table.mojo` are frozen by the pilot. Each wave is compiled, smoked and
tested before it merges; no wave merges on compile alone.

The design spans roughly 840 files; the pilot and rollout need successive implementation, compile, test and
measurement rounds. Owner starting order, 2026-11-10: compile the current integration branch and merge it into main
if compilation succeeds, without holding that merge for import or smoke tests; then begin this full migration.
That starting instruction does not turn compilation into runtime correctness or performance evidence.

## 8. Risks, and what would make me stop

1. **Compile memory.** RUN files importing kernels directly, not through Python-callable entries, changes
   which closures each binding compiles. Measure the pilot binding's compile time and peak memory against the base.
   If a merged group stops fitting the Mac's 16 GB, the entry table (section 2.5) keeps the kernel in its own binding.
2. **Mojo features this needs.** `abi("C")` function pointers stored in a struct and called from another extension
   module; a generic registry of Mojo structs keyed by handle; releasing the GIL around a call. All three exist in the
   repo in some form (descriptor tables, the ops session table, existing GIL release in the doors). If the pilot finds
   one missing, the answer is the ask for Modular, not a workaround.
3. **Lifetime bugs.** Borrowed caller arrays, cached attribute buffers and released handles are where this can go
   wrong. Check handle generations, retain borrowed owners through their last native use, and test mutation,
   replacement, use-after-release and pickle round trips. Temporary native scratch is owned per operation;
   a process-wide slot is not a substitute for concurrent callers' separate state.
4. **Public behaviour.** Error types and messages must not change; the pilot diffs them against the base's tests.
   Preserve fitted-attribute mutability and validation order; moving storage to Mojo does not authorize narrowing
   the supported Python interface.
5. **Bits.** Matching an older version's output is not a requirement. Preserve algorithm correctness and quality;
   the final frozen release must agree across the supported GPU vendors and the required CPU reference column.
6. **Scope creep.** This design does not move any algorithm, does not touch a tuned kernel and does not change the
   executor. A lane that finds itself editing a kernel has left the design.

Fix compiler failures and measured regressions without restoring Python execution or replacing a measured winning
kernel on reasoning alone. A missing Mojo feature gets a minimal reproducer and an upstream ask; it does not get a
Python workaround. Record unbuilt targets and unmeasured performance separately from accepted native correctness.

## 9. Implementation decisions

1. **Initial family.** Ridge and LinearRegression establish the typed boundary; the rollout continues across the
   remaining families and shared input, archive and optimizer paths.
2. **The full rollout gates 0.8.67.** Decided by Andrew on 2026-11-09 and reaffirmed on 2026-11-10: complete the
   Python-object to Mojo-value migration and the applicable test migration before release. Do not defer the full
   rollout to a later release or propose a release date.
3. **`mojolearn.Array`.** Keep it as a thin Python class over a native storage handle, with every method one native
   call and no Mojo code reading `_store`. Recommendation: yes.
4. **List and string inputs.** Preserve supported inputs through native admission and typed preparation.
5. **User callables.** Refuse every user-supplied callable (KMeans `init`, custom scorers, custom estimators inside
   search and pipelines) before its hook runs, with the native alternative named. Recommendation: yes; it is today's
   rule with its one exception removed.
6. **Entry tables.** Replace the 730 calls to compiled entries through the CPython call protocol with published
   native function tables. Recommendation: yes; it is part of the contract, and it is what removes the per-call
   Python lists.
7. **Fitted attributes.** Preserve each public attribute's mutability, replacement and ownership behavior.
   Internal typed state and optional lazy materialization must respect that boundary contract.
