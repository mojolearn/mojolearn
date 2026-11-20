# LocalOutlierFactor native plans

Producer lane: `lane/mojo-finite-tests-20261110`. Baseline adoption merge:
`205835e1ed` (root source `b0d28426b8`). No lane compilation, test execution,
timing, identity work or box job was performed.

`core/lof_plan.mojo` owns LOF's private scratch and execution state. Public
fit arrays remain direct kernel destinations; query score/decision intermediates,
neighbor indices, squared distances, unfused distance matrices, query densities,
sign labels and percentile buffers are native Lists. Fixed native envelopes
reach the existing xn descriptor. The provider pin stays at the boundary.

`core/neighbors_native_driver.mojo` contains the prior `NcDriver` body verbatim
apart from its `XnDriver` name. NC imports that name as `NcDriver`; there is no
new capability query for LOF. The entire NC/KNNImputer boundary block remains
byte-identical to the baseline. Existing tuned numerical sources are unchanged.

The three existing KNN choices remain `knn_sq_tiled`, `knn_sq`, and
`sqdist` plus `knn_select`, followed by the same square root. Density and score
retain their exact address/word order. Finite refusal still follows neighbor
selection/root and precedes density/score and fit publication. Query reads of
`_fit_dist` and `_lrd` remain after that refusal. Mutable fitted `n_neighbors_`
uses the existing native Array layout checks before private allocation; shape
overflow and negative-dimension errors are retained. Empty scratch keeps address
zero so existing descriptor refusal is retained.

Percentile uses the existing row sort, padding/step calculation, pinned Float64
multiplication and asymmetric linear interpolation, then the existing Float32
offset rounding. Sign mode 0 for fit labels, mode 1 for query labels, and the
existing parallel int32-to-int64 cast are unchanged. No kernels, fusion,
switch/default choices or ABI operations were replaced. No speed claim is made.

Removed in this change: LOF's private Python argument lists and scratch/result
Arrays, compiled Python numerical dispatch, `_pow2_steps`, `_astype_i64`,
`_sqdist`, `_l1dist` (already dead), `_knn_select`, `_unary`, `_knn` and
`_lof_percentile`, plus obsolete imports. Repository consumers of those helpers
were audited before removal. `_op`, `_copy_f32` and `_i32_from_ints` remain live
for KNNImputer and were retained.

The invalid metric/p diagnostic had an inherited callback path through NumPy
array repr. Exact native ndarray-valued invalid options now render as
`<native scalar array>` without entering Python printing/formatter hooks.
Ordinary primitive diagnostics and their validation order are unchanged.

Python remains at shared public admission/lifecycle, model-field reads and
publication, public buffer ownership/output wrapping and scalar conversion.
In particular, scalar offset unpack keeps native NumPy integer negation rules;
the admitted boundary converts the result to a Mojo Float64. Shared argument
lists and prepared-state identity stamps have not been migrated here. Staging
preserves existing mutation, refusal and partial-fit semantics; this is not a
claim that the whole public family has a single uninterrupted UNPACK/RUN/PACK
region or no Python imports.

Static evidence: 89 product roots inspected; two affected platform records,
both `_mojolearn_native_classical_family_host`, member
`bindings/_mojolearn_runtime_family_host.mojo` (Darwin and Linux host flags).
The native plan closure has 25 source files and zero Python imports. Nine
numerical sources match the baseline; NC/Imputer boundary block is byte-equal;
the extracted driver body is byte-equal after renaming. These are source facts,
not compile/runtime results.

`python/mojolearn/tests/test_lof_native_state.py` adds eight definitions / 18
independent cases: all three route choices bypass replaced legacy exports;
rebound/in-place public densities; signed-zero, float32, zero-D and unsigned
offsets; negative rebound k; scalar-hook refusal; failed-refit publication;
novelty refusal order; formatter-hook refusal. All are unrun. Existing numerical
LOF Python cases remain pending their replacement by direct native descriptor
oracles; the native-test lane has the plan API for the next CPU fixture batch.
The new host/device plan code remains uncompiled on every target. No dedicated
fast build or identity run is requested. Shared MAX/Metal context concurrency
remains parked under root's upstream reproduction; this change adds no
serialization or workaround.

Evidence directory:
`~/mojolearn-evidence/python-mojo-migration-20261110/lof-native-plan/`.
It retains before sources, source-audit script/log/JSON, full patch and final
source snapshots. `source-audit.json` records the final boundary inventory
refresh separately from its earlier closure scan. Root must bundle the affected
product compile and the native fixtures before claiming runtime correctness.
