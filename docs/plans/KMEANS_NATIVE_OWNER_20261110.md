# KMeans native requests and resident owner

Production commit `3a5b05c487` replaces KMeans' internal Python parameter lists,
compiled callable dispatch and resident handle dictionary/tuple with typed Mojo
requests, results, capability metadata and a native resident owner.

The six-operation descriptor is exported as `kmeans_native_api_v1` by the existing
GPU base binding and CPU core binding. The family boundary admits the exact
factory, pins its module/capsule and releases the GIL around RUN. The native
request code and owner state import no Python support. Python references in the
owner boundary only pin the compiled provider and public centroid storage.

Numerical route preservation:

| Operation | Existing implementation retained |
| --- | --- |
| GPU fit | `cluster/estimator.mojo::kmeans_fit`, same options and synchronization |
| GPU fit-transform | The same fit with `transform_out`; one X upload remains |
| Predict/transform | Existing dense operations, same metric and finite status |
| Cached centers | `cluster/resident_centers.mojo`, same prepare/query/release registry |
| DeviceRows folds | `cluster/kmeans_fold.mojo`, same fold context, IDs and status words |
| CPU | `core/core_host_impl.mojo` over the existing host oracle; original finite scan and two-step fit-transform |

No tuned kernel, experiment default, data-sized loop, numerical allocation or
copy was changed. Legacy exported entries remain available for direct callers;
the public KMeans owner no longer calls them through Python. Optional-entry
probing and its process-global Python record list are removed; typed provider
capabilities decide the same available routes.

`_resident_holder` now contains a native owner or `None`. Its Mojo state stores
the model/centroid/binding identities, admitted native connection, shape, native
handle and driver. A replaced provider factory therefore cannot reuse a handle
minted by another admitted driver, even on the same binding module. Only
freezable Arrays cache their centers, as before. Writable caller ndarrays keep
the per-query upload so in-place changes remain visible. Each query retains its
owner through RUN; replacing/refitting the model retires the old owner, whose
native destructor releases after the last query drops it. No Python release
callable or KMeans finalizer registration remains.

Independent source review caught a missing registration for the allocated Mojo
owner before compilation. The single owning runtime-family initializer now
registers `_KMeansResidentOwner` with the existing guarded lookup/add/attach
pattern. Re-entering the initializer under another package namespace attaches
the same canonical type instead of registering it twice. Every allocating
KMeans operation is reached through that initialized family's entry.

Copy/pickle still omit `_resident_holder`, and restored models obtain their own
resident copy on the next query. Exact dictionary copy/update at that boundary
now uses the CPython API and refuses mapping callbacks before invoking them.

The suite manifest is
`docs/plans/native_test_migration/kmeans-native-suites.json`: 26 native wire/owner
fixtures, 6 actual CPU fixtures, 9 actual GPU fixtures and 51 Python boundary
cases. All native suites expose `run_checks` plus thin standalone mains. The CPU
kernel suite requires `MOJOLEARN_COLUMN_CPU=1`. CPU fixtures use host workers;
GPU cases run in order over the shared context.

No compilation or test execution occurred in the lane. The orchestrator owns
the combined compile and any installed runtime execution. NVIDIA/AMD timing and
cross-vendor identity remain unmeasured; there is no speed or bits claim and no
new A/B define. Existing public KMeans tests stay intact. This is the KMeans
execution/ownership migration; shared public input, lifecycle and option
admission continue to use their existing boundary helpers.

Static source evidence is retained under
`~/mojolearn-evidence/python-mojo-migration-20261110/kmeans-native-owner/`.
The normal source walker scanned 89 roots without errors; this family affects
three development groups on both Darwin and Linux: identical core, host core,
and host runtime-family. The native host/device RUN closure contains 134 source
files and no Python/binding imports. The five directly reached numerical
implementation files listed above are byte-for-byte unchanged from the parent.
These source findings are not compilation or execution evidence.

Independent audit follow-up: scalar admission now precedes KMeans' first
comparison, hash or conversion of rebound model parameters. The init/metric
name route uses exact builtin and NumPy text identities plus a native C type
hierarchy check, removing `isinstance` and its `__class__` protocol. Init codes
remain exact Python integers; NumPy integer initializers still reach the
deferred center-array admission. Oversampling retains its former builtin
integer/float and NumPy float64 forms, including refusal of bool and NumPy
float32. Its refusal reads the canonical type-name getset directly so a custom
metaclass property cannot execute while constructing the message.

Cluster-count admission stays after X preparation and before comparison. The
tol and seed checks stay after initial centers and weights, immediately before
their conversions; max-iteration and restart admission retains its later
position. Exact NumPy text/byte and extended-real scalar conversion forms remain
available. No kernel, tuned route, owner, output allocation or buffer copy was
changed. The obsolete KMeans `family_is_str/int/isinstance` calls and imports
were removed. New fixtures cover 44 scalar boundary cases, bringing the Python
file to 51 expanded cases; those are Python hook/compatibility seams, while the
41 numerical and native state fixtures remain in Mojo.

Static syntax/fixture counts and commit/push output for this follow-up are in
`~/mojolearn-evidence/python-mojo-migration-20261110/kmeans-scalar-admission/`.
Compilation and installed execution of this follow-up remain owed to the
orchestrator; earlier native execution evidence does not cover this new patch.
