# SVR native fit and query transport

Source only: no compilation, runtime tests, timing or identity was run by this lane.

Producers: inventory `78dba1f376`, API/providers `653707b1f4`, fit switch
`01e5e9a1e8`, query/lease switch `e60da78486`, followed by this fixture commit.
Shared prerequisites are canonical `b25d2e9357` and `5053679aa0`; local adoptions
`f9e79e5550` and `3472918982` are not additional producers.

The four scoped roles are the fit option list, fit address list, worst-case
dual/support/vector/info scratch, and query option/status transport. They now
use `SvrFitRequest`, `SvrFitPlan`, `SvrQueryRequest` and native status words.
Public fitted prefixes retain their original allocations, copies and publication
order. The exact existing host SMO, device SMO, candidate short fit, device-X
fit/fold and resident/direct query implementations remain the numerical routes.

`SvrPreparedLease` reuses the existing pure `GPLeaseState` identity/take-once
state. Its native mirror list holds boundary storage pins; the exact binding,
provider image and mirrors remain alive through final native lease release.
The active query holds the owning prepared record through released-GIL work.
Generation, binding and mirror identity still invalidate the cache. Models that
cannot retain the weak prepared record keep the existing per-call upload route.
This is shared prepared-lease schema reuse, not another distinct aggregate role.
The type is registered once by the family runtime, never stored as an estimator
field, and therefore does not extend canonical field-owner admission.

Removed: `_kernel_tail`, `svr_execute`, SVR Python parameter/address lists,
resident status envelopes and its private prepared tuple record. The old exported
SVR bindings and candidate descriptor remain live compatibility interfaces.
Shared `_c_rows` storage, precomputed `_i32_array`/`_gather` storage and SVR's
prediction/R2 score envelopes still need subsequent SVM work. Public arrays and
mandatory provider/storage pins are retained, not counted as eliminated objects.

The new native suites contain 12 independent CPU cases and 21 ordered GPU
numerical cases: all five kernel choices, explicit per-row bounds, bias-only
empty support, invalid dimensions/support counts, candidate short fit, the
original 150x5 planted linear/RBF surface workload and nine resident query cases.
Two-point references come from the closed-form constrained dual; planted salts,
noise, parameters and quality bounds come from `test_svr_surface.arm_planted`.
CPU cases use `host_parallelize`; numerical input/check units are parallel.
GPU case ordering retains the documented upstream shared-DeviceContext
concurrent-submit park. No timing or identity is claimed.

Nine retained Python boundary cases cover public storage/pickle lifetime, mirror
replacement, refit generation, early target refusal, compiled-factory refusal
and storage-hook traps. No existing Python numerical tests are deleted.

Affected roots: `_mojolearn_svm`, `_mojolearn_svm_host`,
`_mojolearn_native_classical_family_host`, `_mojolearn_runtime_family_host`;
shared freeze dependencies retain their separately documented closure.
No numerical define/default or native shared-slot layout changed. The additive
descriptor is `mojolearn.svr.native.v1` (six operations).

Source evidence:
`~/mojolearn-evidence/python-mojo-migration-20261110/svm-native/svr/`.
`query-source-conservation.json` records six unchanged numerical source files
and nine unchanged unrelated methods, including the existing Python-object
query-refusal overload. `fixtures-source-inventory.json` records authored
coverage. Original snapshots and full adoption/commit/push logs are retained.
Compilation, all authored runtime cases, other-device execution and performance
remain owed to the orchestrator; source conservation proves none of those.
