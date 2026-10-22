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

`test_sequence_array_native_candidate.py` exercises the candidate directly.
Passing those tests would qualify that helper, **not** the unchanged public
method. The lane agent does not compile; native compilation and all candidate
test execution remain assigned to the orchestrator.

Remaining work before public connection:

1. Run the candidate tests and repair compiler, warning and byte differences.
   Test-controller prototypes matched 20,008 sampled text conversions and
   20,010 values for each of eleven integer/half/date forms on the ARM Mac;
   these are algorithm screening, not execution of the Mojo implementation.
2. Preserve additional ordinary dtype input forms, including compact record
   strings, shaped string descriptions and recursively supplied field-name
   schemas. NumPy can invoke its Python parser for such descriptions; the
   native parser must replace that before they can be passed through.
3. Investigate NumPy's variable-width `StringDType` separately from fixed
   `S`/`U` storage. Its managed native string storage is not implemented here.
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

References: [Mojo's supported libc/variadic FFI](https://mojolang.static.modular.com/docs/manual/c-ffi/)
and [NumPy's cast-warning and undefined-result notes](https://numpy.org/doc/1.24/release/1.24.0-notes.html#numpy-now-gives-floating-point-errors-in-casts).
