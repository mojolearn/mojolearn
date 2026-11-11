# Mojolearn native-owner replacement admission

Source review and implementation only; compilation and installed boundary
execution are owed. No numerical kernels, routes, allocations, or defaults
change. No performance or cross-device identity claim is made.

`params_setter_native._release_admit` and
`full_reset_params_native._release_owned` (through `declared_params_native._release`)
reach `native_owner_lifetime.native_release_admit` before replacing fields.
The previous seven-name native-owner whitelist omitted newly migrated field
owners and skipped Python pins inside the owners it did accept.

The replacement registry holds exact native type identities, their registrar
modules, and compiled C-ABI pin inspectors. It is private to Mojo: there is no
Python registration or inspector dispatch surface. All access holds the GIL.
Retaining the class prevents pointer reuse; retaining the module pins the
inspector's compiled image. Duplicate attaches keep the original registration.
The existing direct-object/no-instance-dictionary/no-weakref/no-GC layout and
no-Python-lifecycle-hook checks remain. Class mapping keys are admitted before
lookup, independently of the type's registration.

An inspector returns only borrowed pins while its exact owner is retained.
Each pin enters the existing recursive admission using the same context and
`depth + 1`. The callers' visited collections and the existing native depth
bound remain intact. A native owner therefore does not conceal a rebound
Array pin or reset the recursion bound. There are no tensor copies or new
numerical loops. Only fixed owner metadata is traversed.

| Registered native type | Retained Python pins followed |
| --- | --- |
| `NeuralDescriptorProvider` | module, factory; its internal descriptor capsule is checked to have no destructor |
| `NativeDescriptorResource` | provider |
| `TreesResident` | release entry |
| `TreesResidentDeath` | none |
| `TsvdResidentLease` | none |
| `GpResidentLease` | none |
| `SvmResidentLease` | none |
| `KMeansResidentOwner` | provider, centers |
| `KDEResidentOwner` | provider, train, weights |
| `DBSCANReferenceOwner` | provider, refs, keys, labels |
| `DBSCANHolder` | resident owner |

The first seven retain their previous admitted roles. The final four are the
actual newly migrated estimator-field owners. DBSCAN's inspectors and its
registrar calls are the separate dependency `12da123eea`; the additive shared
registry is `8cb9aaa98c`. Later Agglomerative owners have a separate registrar
change and are not claimed as part of this snapshot.

The provider capsule is created by `neural_descriptor_capsule` with a null
destructor and borrowed module-global native name/table storage. Its module
and compiled factory remain ordinary recursively admitted pins. Their module
namespace and `sys.modules` receive existing exact-string-key admission before
keyed reads; admitting the owner does not permit a foreign key equality hook.

Four prepared-only types are deliberately outside field-release admission:
`PCAQueryCache`, `KDEAdmissionOwner`, `TSVDResidentOwner`, and
`IPCAResidentOwner`. They are held in `RECORDS_PREPARED`, through
`native_prepared_owner.prepared_put` or `records_stamped_put`, rather than
estimator dictionaries. Their hit checks retain existing identity/generation
validation; explicit invalidation uses `prepared_drop`/`records_drop`.
`native_model_records.records_put` and `records_drop` move an old payload out
of the map before its references drop. Model death reaches the native weak
callback and `records_release`, which removes the entire record before
destroying it. TSVD/IPCA destructors retain their current direct native
release behavior; PCA/KDE admission records contain metadata pins. This work
does not add these caches to the field API or claim that their separate
teardown path has acquired the new field-pin admission. No additional
prepared-record repair was established by this bounded source inventory.

Focused installed-boundary fixtures cover full resets of KMeans, KDE and
DBSCAN, nested opaque owners in Ridge's declared fitted state, fake Python
types using native-looking names, a private Array pin with a Python finalizer,
and an owner/Array cycle. Resident-only fixtures explicitly skip if the
installed image cannot provide that owner; those skips are not coverage.
The existing numerical fixtures are unchanged.

Removed: the native type-name whitelist and the unconditional acceptance of
its owners without traversing pins. Remaining Python objects are boundary
image/buffer lifetime pins and mandatory public API values. No source guard,
Python fallback, new algorithm route, or experiment define is added.
