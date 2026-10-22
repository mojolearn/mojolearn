# Sequence device array protocol: native candidate

The public `SequenceDeviceTensor.__array__` still downloads through `.numpy()`
and uses Python/NumPy for dtype conversion. It is **not migrated** by this
candidate. The new internal `seqdev_array` operation lets the orchestrator
compile and qualify native output conversion without replacing or narrowing
the existing public interface prematurely.

The retained pre-migration record has 111 accepted cases: 36 dtype requests
with three `copy` values, plus three `numpy.asarray` consumer calls. `copy`
is ignored by the old method. The source words include signed zeros, finite
large values, infinities, NaN and subnormals. Evidence is in
`~/mojolearn-evidence/takeover-0856-20261021/optimizer-lane/sequence/`:
`array-contract-4725-metal.json` and `array-contract-remaining.json`.

Implemented candidate work:

- Native fixed dtype metadata admission, field/subarray layout walks and
  native ndarray allocation; only allocation/metadata uses NumPy's C entries.
- Native normalization of compact record/subarray strings and recursively
  supplied list/dictionary field schemas. Field offsets are ordered in Mojo;
  NumPy's Python `_commastring` and `_usefields` helpers are not invoked.
  Twenty-six additional ordinary schema baselines are retained; candidate
  execution and malformed-schema compatibility remain pending.
- Typed output conversion in `core/sequence_array_cast.mojo`: integer, bool,
  half, float, complex, byte order, datetime/timedelta units, void storage and
  extended host floating layouts. NaN payload widening is separate from
  CastIn's deliberate canonicalization.
- Shortest float32 text conversion through supported Mojo libc FFI, with a
  thread-local C numeric locale, and byte/Unicode storage writes.
- Native boxing into owned object-array slots. No user object conversion or
  finalizer is invoked.
- Native warning recording/filtering and NumPy error-state metadata, including
  ignore/warn/raise/print and native log streams. Python dtype-discovery and
  error callbacks are refused before their hooks run.
  NumPy 1.26's native `geterrobj` mask is decoded in Mojo; NumPy 2's native
  `_get_extobj_dict` metadata is read directly. The six legacy action settings
  were checked in an isolated NumPy 1.26.4 controller environment; native
  candidate execution with that version remains pending.

`test_sequence_array_native_candidate.py` exercises the candidate directly.
Passing those tests would qualify that helper, **not** the unchanged public
method. The lane agent does not compile; native compilation and all candidate
test execution remain assigned to the orchestrator.

Remaining work before public connection:

1. Run the candidate tests and repair compiler, warning and byte differences.
   Test-controller prototypes matched 20,008 sampled text conversions and
   20,010 values for each of eleven integer/half/date forms on the ARM Mac;
   these are algorithm screening, not execution of the Mojo implementation.
2. Qualify the native schema normalizer for ordinary and malformed dtype
   inputs, nested titles/metadata and version-specific parsing behavior.
   The additional 26-case schema controller baseline uses NumPy 2.4.4; its
   retained array bytes and dtype objects are in `array-native/` under the
   evidence directory. It is not native candidate execution evidence.
3. Qualify NumPy's variable-width `StringDType` separately from fixed `S`/`U`
   storage. The separate native candidate uses NumPy's public NpyString
   acquire/pack/release API for storage and Mojo for formatting. It validates
   the native capsule/table/function owners before calls, retains descriptor
   ownership through allocator release, and admits native missing-value
   metadata before output allocation. Python missing-value hooks are refused.
   Ordinary metadata, copy/coerce settings, subarrays and signaling-NaN
   warning behavior have candidate tests; native execution remains pending.
   A concrete remaining support constraint is structured-void/ndarray
   missing-value metadata: NumPy's native descriptor finalizer reconstructs
   the descriptor using scalar comparison/text hooks, entering Python
   `_promote_fields`/`arrayprint`. Five retained allocation probes observed
   3–43 Python frames; raw void metadata has none and is admitted. The private
   candidate refuses those callback-dependent forms before allocation, while
   the public method remains unchanged. No supported callback-free ownership
   path has been verified yet; this is not proof that none exists. No private
   StringDType memory layout is accessed or patched.
4. Qualify non-default locales, nested/object schemas, empty/scalar shapes,
   NaN payloads, error filters and NumPy versions. Extended dtype layout and
   unsafe out-of-range integer casts require Linux host qualification. NumPy
   documents out-of-range floating-to-integer results as platform-dependent;
   the old ARM record must not be presented as an x86 reference.
5. Only after supported ordinary forms are preserved, replace public
   `__array__` with one native call, run cold/consumed-output audits, and repeat
   the actual three `numpy.asarray` consumer cases.

No missing Mojo/Modular facility has been demonstrated. These are remaining
implementation and qualification tasks, not a claimed toolchain blocker.
No Python fallback has been added to native code and the existing public
interface remains intact.

Qualification checkpoint: frozen `43b5b5e13` compiled successfully. The parent
reported 146/146 selected CPU cases passing and 296/304 selected Metal cases
passing; all eight Metal failures were in the private array candidate/guards.
The follow-up repairs normalize scalar subarray shapes before the native dtype
factory, preserve unsized void adaptation, refuse non-type warning categories
before descriptor access, and explicitly select the C numeric locale category.
Scalar shape-one compatibility retains NumPy 1.26's deprecation warning and
scalar result through native warning delivery. Fourteen added schema cases and
two added guard cases still require execution on the repaired frozen build.
These counts cover the selected combined suites, not an all-green release.

Managed-string allocation investigation (NumPy 2.5.3): the retained controller
traces are `array-native/stringdtype-void-array-metadata.json` under the evidence
directory above. The native `ndarray(..., dtype=existing_descriptor)` path
reaches [`stringdtype_finalize_descr`](https://github.com/numpy/numpy/blob/v2.5.3/numpy/_core/src/multiarray/stringdtype/dtype.c#L700),
which clones an already-owned descriptor using `new_stringdtype_instance`.
That constructor invokes comparison and text protocols at lines 73 and 87.
Calling the public StringDType constructor directly uses that same helper.
[`PyArray_DescrNew`](https://github.com/numpy/numpy/blob/v2.5.3/numpy/_core/src/multiarray/descriptor.c#L1963)
explicitly rejects new-style dtypes. A remaining supported alternative to
investigate is public `PyArray_NewFromDescr` with an externally owned data
buffer: [`ctors.c`](https://github.com/numpy/numpy/blob/v2.5.3/numpy/_core/src/multiarray/ctors.c#L619)
skips descriptor finalization when data is supplied. Safe managed-string
destruction, buffer/base lifetime, and ordinary missing-value semantics would
have to be established before using that alternative. If it cannot preserve
those contracts, the precise upstream need is a supported native descriptor
clone/allocation operation that retains supplied missing-value metadata
without invoking comparison or text protocols. This investigation does not
establish a Mojo/Modular compiler limitation or authorize a public refusal.

References: [Mojo's supported libc/variadic FFI](https://mojolang.static.modular.com/docs/manual/c-ffi/)
and [NumPy's cast-warning and undefined-result notes](https://numpy.org/doc/1.24/release/1.24.0-notes.html#numpy-now-gives-floating-point-errors-in-casts).
The managed-string candidate uses [NumPy's public string storage API](https://numpy.org/doc/2.1/reference/c-api/strings.html).
