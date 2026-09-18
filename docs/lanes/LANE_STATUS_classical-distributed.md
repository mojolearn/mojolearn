# Classical distributed inference and fit

Branch: lane/classical-distributed. Worktree: classical-distributed.

## Forecast checkpoint

Public import: `from mojolearn.parallel_forecasting import predict_arima,
forecast_arima, predict_exponential_smoothing, forecast_exponential_smoothing`.

Whole series are sent to ordered persistent GPU workers. Each worker receives
only its series' fitted state; results preserve ARIMA series-major and
Holt-Winters time-major/single-series/index shapes. Uneven shards preserve every
component byte. Failures close the pool and leave the source estimator untouched.
ARIMA exogenous models are explicitly deferred. Holt-Winters in-sample arithmetic
is host arithmetic in the existing binding, so this GPU API explicitly refuses
in-sample ranges. No public parallel CPU promise is added.

Validation: 20 binding-free partition/assembly tests passed. These exercise
transported state, ordering, component slicing, uneven shards, original state
immutability, shape contracts, worker failure cleanup and invalid input. They
are not numerical GPU evidence. NVIDIA/AMD one/two/reversed GPU, native fault,
installed-wheel replay and verifier admission remain owed. No hardware rental
or qualification claim in this checkpoint.

Next: class-level GPC partition; IVF requires preserving global coarse probe
selection with disjoint candidate storage, not independently trained indexes.

## Class-level GPC checkpoint

Public import: `from mojolearn.parallel_gaussian_process import
fit_gaussian_process_classifier, predict_gaussian_process_classifier`.
One-vs-rest class problems use the unchanged binary solver. Predictions ship only
one class covariance state to each worker and fold class probabilities in the
original float64 order, including zero-sum handling, ties and binary signs.
Fit publishes complete state atomically after all classes return. Full training
input is replicated; one binary covariance still must fit on a device. This
addresses multiclass throughput, not a larger single binary fit.

Validation: combined forecasting/GPC suite: 27 passed, binding-free. GPC tests
compare the original fit and composition against scheduled binary tasks with
numeric and string labels, zero probabilities, exact ties, binary signs and
failing tasks. Native numerical, physical multi-GPU and installed replay are
still owed.

## IVF implementation constraint

Existing native saved-index search rejects partitions with fewer stored rows
than global centers (`ivf_validate_index_arrays`) and queries with fewer than k
candidates (`ivf_flat_search`). Keeping global coarse centers and zeroing other
lists therefore cannot safely compose distributed search. A correct new native
entry must preserve global probe selection, admit disjoint candidate storage,
return an explicit valid count for empty/short candidate shards, and merge on
squared-distance/original-ID keys before Euclidean rooting. Independently built
indexes or replicated full indexes would not meet the requested capacity scope.

## Disjoint IVF storage/search implemented

The constraint above is now addressed by a new native `ivf_flat_partial_search`
entry. It preserves full global coarse centers/probe selection but admits a
smaller local candidate store. Short and empty local candidate results carry
explicit counts; padding is ignored. The default full-index entry retains its
existing short-result refusal. Global result merge compares squared distances
and original row IDs, then applies the original native identical square root.

`from mojolearn.parallel_ivf import DistributedIVFIndex`; use
`with DistributedIVFIndex.from_index(index, devices=(0, 1)) as distributed:`
and `distributed.search(queries)`. Disjoint index rows persist in workers' host
memory and only that shard is uploaded to its GPU for each search. Coarse centers
and query rows are replicated. This is actual partitioned candidate storage,
not a full-index replica per GPU. Persistent device residency, distributed
initial quantizer building, distributed extension, and beyond-host-memory
loading are not implemented. Each shard and global coarse centers must fit on
one GPU; no throughput improvement is asserted without measurements.

A new IVF binding was compiled on Apple under the shared slot in 21.8 seconds.
61 combined checks passed: 36 binding-free tests, 11 CPU-native logical
forecast/GPC partition tests, and 14 Apple-GPU native IVF partition checks.
The latter compare full vs disjoint storage bytes for both metrics, empty/short
local probe hits, ties, repeated calls, and uneven 33x3 and 65x17 fitted indexes.
These logical shard tests execute sequentially on one physical GPU and do not
constitute physical multi-GPU qualification.

Retained local evidence: `/Users/andrewhendel/mojolearn-evidence/classical-distributed/`
contains `native-partitions.xml` and `_mojolearn_ivf_partial.so`.
Native build source changes and tests are committed alongside this checkpoint.

## Two-GPU capture preparation

`python tools/distributed_classical_check.py --devices 0,1 --out NEW.json`
checks one-device, two-device and reversed-device schedules, twice each, for
ARIMA, Holt-Winters, GPC inference/fit state and IVF. Inputs are small and include
uneven series/storage partitions. It checkpoints each case, retains source and
native binding hashes/vendor readbacks, package/distribution provenance and
worker PID/operation receipts. `--require-installed` requires imports to match
an installed distribution; launch from outside the checkout without PYTHONPATH
for a wheel gate. It never equates process placement with actual GPU execution.
Success remains `NUMERICAL_MATCH_EXECUTION_TRACE_OWED`; native fault controls and
NVIDIA/AMD physical kernel traces remain owed. No GPU rental was provisioned by
this lane. The capture runner has been syntax/CLI checked but not run on two GPUs.

Post-checkpoint hardening: seven additional binding-free checks reject missing
shards, malformed output shapes, negative/excess candidate counts, invalid local
IDs, duplicate global IDs, and failed initial storage. The current lightweight
suite passes 43 checks with 14 native IVF checks skipped unless explicitly
requested. These deliberate transport-corruption tests do not substitute for
the still-owed native arithmetic sabotage controls.
