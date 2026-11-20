# PolynomialCountSketch typed execution

`core/pcs_native_api.mojo` now owns PCS options, feature/augmented dimensions,
output shapes, fit plans and fixed draw/query requests. `PCSNativeDriver` uses
the existing Xn descriptor with `pcs_draw`, `pcs` and `pcs_resident`; no new ABI,
provider export, numerical kernel, default or threshold is introduced.

`bindings/neighbors_native.mojo` retains public admission and publication.
Its new `pcs_native_boundary.mojo` helper resolves the same explicit provider,
host facade or model-selected binding without rebuilding the argument tuple.
A `PCSConnection` retains the admitted native provider/image and driver across
each GIL-free draw or query. Existing native mode admission remains at its
original stage. Both normal and already-admitted descriptor inputs enter the
same typed execution path.

## Preserved contracts

- Exact builtin integer degree/component admission, positive size refusal,
  then coef0 admission retain their order. Fit does not convert gamma; query
  still converts it only after fitting has published the tables.
- The augmented feature count uses unrounded Float64 coef0. Query gamma/coef0
  then make the same Float32 round trip as the old Python-list envelope.
- `pcs_seed` remains the actual boundary for exact integer, native NumPy
  RandomState and None. Its NumPy capsule, native lock and one-word RNG draw
  implementation is unchanged. Integer seeds retain CPython's arbitrary-size,
  signed `PyLong_AsUnsignedLongLongMask` conversion followed by the original
  low-32-bit mask in pure `pcs_seed_word`.
- Seed resolution precedes draw capability refusal; output allocations precede
  integer seed unpacking; the native draw precedes indexHash_, bitHash_ and
  n_features_in_ publication. Empty augmentation retains its no-draw path.
  Public field keys are readmitted after the draw releases the GIL.
- Hash arrays are still allocated once as zero-filled public int32 Arrays and
  filled directly. Query captures current indexHash_/bitHash_ storage each
  time, keeping rank, degree, dtype, contiguity and fitted-width refusal order.
  Rebound and mutable valid tables remain supported. No private shadow table,
  new data copy, extra upload or changed output allocation format is added.
- Query still reads `MOJOLEARN_XN_OLD_ITEMS` after output allocation and chooses
  the same original route. The actual source uses direct circular convolution
  (`items.mojo`, documented DEVIATION 5203), with the existing device split and
  tuned row/cell kernel choice. No FFT or convolution implementation is changed.

Removed from this public execution path: the argument tuple/list reconstruction,
Python shape tuples decoded by family_array, Python numerical address/integer/
float lists, the legacy draw-function lookup/call and `xn_execute` dispatch.
The old shared raw wrappers remain for other internal/raw callers; this bounded
change does not alter their exports or regenerate unrelated bindings.

## Boundary state still present

The public lifecycle still captures admitted options/owners in its boundary
settings dictionary and performs its existing validation/publication. Python
objects remain for public estimator state, input/hash/output pins, import and
mode metadata, and optional NumPy RNG capsule/lock admission. Native RUN sees
only typed values and addresses. The preexisting public-input admission and
finite-check implementation is unchanged; this is not a claim that all shared
neighbor boundary plumbing has been migrated. No Python fallback or source
guard is introduced.

## Fixtures and evidence

The new Python boundary file has 9 definitions / 16 expanded cases, covering
legacy-dispatch traps for both routes, arbitrary-size signed seeds, exactly one
RandomState seed draw, mutable/read-only rebound hashes, descriptor and seed
hook refusal, ignored gamma, augmentation before Float32 rounding, and refusal/
publication order. It was AST-parsed only; no lane tests or compilation ran.

Native fixture handoff (`core.pcs_native_api`):

- `pcs_options` positive-size refusal; PCSFitPlan feature/augmentation/shapes,
  including zero and a nonzero Float64 coef0 that rounds to Float32 zero.
- `pcs_seed_word` low-word behavior, draw request count and query address/shape/
  Float32 fields, width mismatch ordering, old-items/resident route selection.
- Actual native draw ranges/signs and small independent count-sketch/circular-
  convolution oracles using both existing routes, with output guards/lifetimes.
- Keep the existing independent sklearn seam: `test_x_neighbors_sanity.py` uses
  60x6 default_rng(0) normal input, scales its first five rows by 6, and sets
  degree=3, gamma=.5, coef0=1, components=32, seed=7. It places native hash tables
  into the sklearn reference and compares at rtol=atol=1e-3. The separate
  RandomState(4), 16-component seam and NumPy late-connect coverage stay intact.

A source-only concrete host/device closure traversal found 79 local native
Mojo files with zero Python imports. Seven numerical files and the existing
RNG boundary source match `6d91698ad3` byte for byte. Full closure, hashes and
counts are in `~/mojolearn-evidence/python-mojo-migration-20261110/pcs-native-owner/source-audit.json`.
Source comparisons do not establish runtime behavior, performance or identity.

Compile scope: `_mojolearn_runtime_family_host` owns the changed public entry.
Existing host/device x_neighbors descriptor images are unchanged and may be
reused when their build closures match. Root owns combined native fixture and
product compilation/execution. All public/native runtime evidence is pending;
no box job, timing, A/B define or cross-vendor identity run was started here.
