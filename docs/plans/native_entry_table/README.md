# Native entry tables: Ridge and LinearRegression

This first conversion publishes native estimator fit/predict, weighted Ridge,
OLS decomposition and score-metric endpoints at extension initialization.
`core/entry_table.mojo` provides the shared mechanism; `core/linear_entry_table.mojo`
defines explicit tier/vendor/version names. Missing providers, execution-key
mismatches, incompatible schemas/counts and conflicting publications refuse.
Publication is release/acquire synchronized and immutable. Existing capsule
and boxed exports remain for unconverted clients.

The public boundary captures an opaque canonical connection owner before
releasing the GIL. Import-time creation admits compiled metadata/factory entries
from the same exact module and compares their endpoint with the published table.
The owner stores an immutable native key and retains that exact module; mutable
Python tuples cannot substitute its key or pins. Entry-admission caches include
the owner identity, so replacement rechecks the selected mode/vendor. No native request/table/registry contains
Python fields. Import-time wiring remains Python; it does not run inside an
estimator operation. No numerical kernels or route-selection code is changed.
Weighted/one-feature Ridge, Gram, resident Ridge/OLS, TSQR and metric ranges
keep the existing native requests, kernels, defaults and copies.

Score also checks the canonical metrics key against the selected metrics
mode/vendor before table lookup and RUN. It intentionally uses the metrics
selection: explicit CPU estimators may retain a GPU metrics connection. The
boundary controls replace a real metrics owner after successful scoring and
change only the selected metrics vendor. Every linear GIL-release scope now
borrows a named interpreter owner, following the existing boundary pattern;
the release scopes and native operations are unchanged.

`core/state_registry.mojo` provides typed generation-checked handles, immutable
Arc leases, replacement and release. Only registry metadata is locked. Native
state destruction occurs after unlock. A lease acquired before replacement or
release retains that native state; stale handles refuse. Slot layouts include
the stored state's size as well as the shared-slot layout check.

Ridge/OLS public buffers remain writable and replaceable. Their registry state
contains admitted descriptors, with coefficient/mean pins retained at the
actual Python boundary. It does not independently own a borrowed Python buffer.
The prediction boundary refreshes a changed descriptor and acquires its native
lease while holding the GIL, then retains that lease and the admitted buffers
through RUN. An unpublished/manual public state still follows the same native
admission and handle path. Archive restoration publishes the same handle owner.
Native clients of the generic registry may store fully native owning values.

Removed from the converted calls: capsule factory connections and provider
unwrapping for Ridge fit, OLS fit/TSQR, linear predict, linear score and build
metadata; import-time dictionaries no longer retain those unused factory
entries. The generic metrics capsule helper remains for other linear-family
consumers. No tuned kernel, public API, fallback or source-count gate was added
or removed.

Compile roots: `_mojolearn_estimators{,_host}`,
`_mojolearn_x_linear{,_host}`, `_mojolearn_x_decomp{,_host}`,
`_mojolearn_x_metrics{,_host}`, plus the host runtime roots importing the
linear boundary/state owner (integrator computes the current merged closures).

Source status: authored only. No lane compilation, runtime, profiling, timing
or identity run. Root owns all execution. Native contract and cross-image
publisher/consumer tests must cover immutable first publication, concurrent
lookup/registration, missing and mismatched provider keys/layouts, stale and
reused handles, replacement, active leases and native owner destruction.
Public tests must retain mutable/rebound buffers, archive/copy behavior,
explicit Host classes, original numerical routes and error ordering. Trap old
capsule factories after import to distinguish the new route. A separate native
publisher/consumer image control must show the actual published pointer is used.
These controls must not add production per-call instrumentation.

The independent Python crossing observer is root-owned. Its `sys.setprofile`
counts have known direct-C-call blind spots; zero observed calls is not runtime
qualification. No speed improvement or Python-free whole-family claim is made
before the corresponding runtime evidence and NVIDIA/AMD measurements.
