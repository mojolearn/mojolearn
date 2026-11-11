# NearestCentroid native numerical plans

Product source is frozen at `7cac218d3d8ad2b74bb8ec036f5ce2e1511ee9d7`.
Subsequent fixture and documentation commits change no product code. Dependency: integration `52074495df`, adopted into this lane
by merge `eea9db9f40`. Integrate the ordered range
`eea9db9f40..lane/mojo-finite-tests-20261110` through the final handoff head,
without the dependency merge.

No compilation, tests, installed-wheel execution, timing, identity run or box
job was started by this lane. All new fixtures are authored and unrun. The
orchestrator owns the combined development compile. Runtime correctness and
performance remain separate from that compile.

## Migrated numerical execution

`core/nearest_centroid_plan.mojo` owns typed fit/query requests and configuration,
private counts/offsets/centroids/prior/decision scratch, fixed native descriptor
envelopes and the numerical operation sequence. Its source closure has 27 files
and no Python imports. Requests execute the existing x_neighbors native provider;
no numerical kernel or fallback was introduced.

The original count, mean/median, fused statistics, standard deviation, shrinkage,
distance, selection, decision, matrix product, softmax and fused prediction
operations remain in the same mathematical paths. `MOJOLEARN_NC_SPLIT_OPS` keeps
its existing default and kernel choices. The existing `NC_PROBA64` condition is
reported as capability bit 7 by both provider owners, preserving the Float64
probability setting without probing a legacy Python numerical export. Provider
images remain pinned at the boundary; immutable capability words are reused for
the same admitted provider. A replaced factory connects a distinct provider.

Empirical priors still use the existing typed label ratio descriptor with mode
1 and Float64 count/n. User priors retain negative-value precedence, the same
Shewchuk fsum function/status interpretation, the original isclose tolerance and
optional rescale. Independent prior scans, narrowing, normalization and integer
class-prefix ranges run through the existing host worker policy. The fsum's
inherent arithmetic dependency is unchanged. No floating reduction was replaced
by a new fold or a generic kernel.

Private arrays now use native Lists. Public fitted/output Arrays remain allocated
at the boundary and are written directly by the kernels, without an added native
result-to-Python data copy. The nonfused distance-label path writes the same
selected indices directly into the final code buffer, removing its private
index Array and copy. No intermediate distance/decision matrix is downloaded by
the existing fused sessions. The existing nonfused sessions retain their route.

Every query reads live fitted buffers/attributes anew; this introduces no fitted
state cache that could ignore in-place edits or rebinding. Original label handling
and public result packing remain shared with the existing boundary helpers.

## Review repairs and removals

Root's source review at `7cb89afb18` checked kernel argument order, prior fsum
delegation, shrinkage flags and zero-fill behavior. Subsequent product repairs:

- `6016a28536` emits address zero for empty private Float32 scratch, matching the
  old empty Array descriptor and its existing null-buffer refusal.
- `e646f823b5` defers centroid/dataset-centroid allocation until statistics, after
  prior validation and `class_prior_` publication. Invalid priors do not allocate
  C*d scratch.
- `d9a1e2912a` returns before constructing pointers for empty prior narrowing and
  class-offset spans; zero-length prior normalization keeps its division refusal.
- `7cac218d3d` retains split/metric configuration reads after prior publication,
  at their original stage.

Removed from NC execution: Python address/int/float request lists; compiled-Python
numerical dispatch; private Array counts, widened counts, centroids, median/status
scratch, priors, decision/distance and selected-index buffers; the Python query
operand tuple; the score method's temporary argument tuple; duplicate prior
copies; serial per-class normalization/narrowing/offset loops; legacy numerical
export capability probes; and the unused local isclose helper/fsum import.

Twelve numerical/provider-dispatch source files are byte-identical to the
dependency, including x_neighbors items, CPU/GPU operators, iterators and generated
descriptor dispatchers, the probability kernel and shared ratio/cast helpers.
The two auxiliary provider files change metadata only. All 40 non-NC neighboring
functions are unchanged by the source audit; LOF/KNNImputer behavior is outside
this change. No experiment/default was changed or removed. A/B define: none.
No intended numerical bit change; runtime bits and speed have not been measured.

## Boundary work still present

This migrates NC numerical requests, state and dispatch. It does not establish a
single completed UNPACK/RUN/PACK operation or make the whole public family
Python-free. Shared public lifecycle, state admission, label packing, feature
metadata, mode selection and output publication remain existing helpers.

Fit retains staged metadata handling to preserve current refusal and partial-fit
semantics: metric/label admission precedes native counts; prior input admission
precedes native normalization and `class_prior_` publication; split/metric reads
and public output allocation precede native statistics; finite/variance refusals
precede shrink-threshold truth/Float64 admission; native shrinkage precedes final
fitted-attribute publication. Those interleaved Python field reads, scalar input
objects, native builtin conversions and public attribute writes remain at the
boundary. Query preparation similarly reads public arrays and metadata before
constructing typed operands. `_NcOperands` retains Python objects solely as
boundary lifetime pins plus its native prior buffer/request.

Future work toward a complete single execution phase must preserve these
publication/refusal rules and mutable public attributes. The current change does
not silently prevalidate later options, remove public objects or claim every
transitive shared helper is already migrated.

## Compile scope and queued coverage

The import audit inspected all 89 release roots. Six platform/group records are
affected (four physical names):

- Darwin identical: `_mojolearn_native_cluster_m1`
- Linux identical: `_mojolearn_native_cluster`
- Darwin/Linux host: `_mojolearn_native_classical_family_host`
- Darwin/Linux host: `_mojolearn_native_x_host`

The changed x_neighbors provider metadata must be built with the changed native
owner; an older provider without capability bit 7 is not the measurement build.
No dedicated fast build/test or identity run is requested.

RUN OWED, orchestrator only:

- `checks/native_tests/nearest_centroid_plan.mojo`: 20 independent native prior,
  offset, narrowing and refusal fixtures, including large worker-boundary sizes.
- `checks/native_tests/nearest_centroid_native_cpu.mojo`: 11 fixtures through the
  actual CPU descriptor: eight metric/split/shrink combinations with independent
  symmetric-data oracles and three empty-query refusal cases.
- `checks/native_tests/nearest_centroid_native_gpu.mojo`: the same 11 actual-device
  fixtures through a thin GPU wrapper. GPU compilation/execution is pending.
- `python/mojolearn/tests/test_nearest_centroid_native_state.py`: 13 Python boundary
  cases for in-place/rebound fitted state, detached output lifetimes, empty-query
  refusals, partial prior publication and replaced legacy numerical exports.
- Retain existing NC cases in `test_neighbors_loc_native.py`, the independent
  legacy-controller oracle and `test_nearest_centroid_native_capability.py`.

Native fixture units run independently through `host_parallelize`; metadata
passes split their ranges by the existing host worker policy. Python cases are
independent pytest work units. No test controller is product execution.

Complete evidence lives in
`~/mojolearn-evidence/python-mojo-migration-20261110/nearest-centroid-native/`:
`source-audit.json`, `source-audit.log`, `fixture-followup.json`, the static audit
script, commit logs and push logs. The full audit records groups/members, native
closure files, unchanged numerical owners and unchanged non-NC functions.
`git diff --check` passed. Normal push required the previously authorized wide
rebuild override only for inherited `native_torch_admission` history; normal
connection checks passed (10 owners, 15 literal consumers, zero errors).
